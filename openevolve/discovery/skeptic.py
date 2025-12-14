"""
Adversarial Skeptic for Falsification-Based Evaluation

This module replaces "LLM-as-a-Judge" with adversarial falsification.
Instead of asking "Is this code good?", we actively try to BREAK it.

Key insight: A hypothesis is only scientifically valid if it can survive
attempts to falsify it (Popperian epistemology).
"""

import ast
import json
import logging
import random
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from openevolve.database import Program
    from openevolve.llm.ensemble import LLMEnsemble

logger = logging.getLogger(__name__)


@dataclass
class FalsificationResult:
    """Result of an adversarial falsification attempt"""

    survived: bool  # Did the code survive the attack?
    attack_type: str  # Type of attack used
    attack_input: str | None = None  # The adversarial input that was tried
    error_message: str | None = None  # Error if code crashed
    execution_time: float = 0.0  # How long the attack took
    confidence: float = 0.0  # How confident we are in the result (0-1)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "survived": self.survived,
            "attack_type": self.attack_type,
            "attack_input": self.attack_input,
            "error_message": self.error_message,
            "execution_time": self.execution_time,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


# Import SkepticConfig from config module for consistency
from openevolve.config import SkepticConfig
from openevolve.discovery.skeptic_plugins import load_plugins

# Prompts for adversarial input generation
ADVERSARIAL_SYSTEM = """You are a security researcher and bug hunter.
Your job is to find inputs that will CRASH or produce INCORRECT results from code.

You should think like an attacker:
1. What edge cases might the code not handle?
2. What inputs violate assumptions the code makes?
3. What could cause crashes, hangs, or memory issues?
4. What could cause incorrect but plausible-looking output?

Generate inputs that are likely to expose bugs. Be creative and adversarial.

Output your response as STRICT JSON only (no prose, no trailing commas, double quotes).
The "input" field must be a valid Python literal (e.g., [], None, {"a": 1}).
{
    "attack_inputs": [
        {
            "input": "<the adversarial input as Python literal>",
            "attack_type": "<edge_case|type_confusion|overflow|malformed|logic_error>",
            "rationale": "<why this input might cause problems>"
        }
    ]
}"""

ADVERSARIAL_USER = """Analyze this code and generate adversarial inputs to test it:

```{language}
{code}
```

The code is supposed to: {description}

Generate {num_attacks} different adversarial inputs that might:
- Cause crashes (exceptions, segfaults)
- Cause incorrect output
- Cause hangs or infinite loops
- Expose security vulnerabilities

Focus on attack type: {attack_focus}"""


# Prompt for blind reproduction test
REPRODUCTION_SYSTEM = """You are a scientist trying to reproduce published results.
You will be given a description of what code should do, but NOT the original code.
Your job is to write code that achieves the same functionality based only on the description.

If the description is vague or unclear, note what assumptions you had to make.
If you cannot reproduce the functionality, explain why."""

REPRODUCTION_USER = """Based on the following description, write code that implements this functionality:

## Description
{description}

## Expected Behavior
{expected_behavior}

## Constraints
{constraints}

Write your implementation in {language}. Include comments explaining your approach."""


class AdversarialSkeptic:
    """
    Replaces LLM-as-a-Judge with adversarial falsification.

    This implements Karl Popper's falsificationism: a hypothesis (code)
    is only considered valid if it can survive active attempts to disprove it.

    Attack Modes:
    1. Edge Case Attacks: Empty inputs, null values, boundary conditions
    2. Type Confusion: Wrong types, mixed types, unexpected structures
    3. Overflow Attacks: Huge numbers, tiny numbers, precision limits
    4. Malformed Inputs: Corrupted data, invalid encodings
    5. Logic Attacks: Inputs that expose logical flaws

    Optional: Blind Reproduction Test
    - Pass the description (not code) to a separate LLM
    - If that LLM can't reproduce the results, the code is too vague
    """

    def __init__(
        self,
        config: SkepticConfig,
        llm_ensemble: Optional["LLMEnsemble"] = None,
        reproduction_llm: Optional["LLMEnsemble"] = None,
    ):
        self.config = config
        self.llm_ensemble = llm_ensemble
        self.reproduction_llm = reproduction_llm

        # Load optional task-specific plugins
        self.plugins = load_plugins(getattr(config, "plugins", None))

        # Track attack history for learning
        self.attack_history: list[FalsificationResult] = []
        self.successful_attacks: dict[str, list[str]] = {}  # attack_type -> inputs

        if self.plugins:
            plugin_names = []
            for plugin in self.plugins:
                plugin_names.append(getattr(plugin, "name", plugin.__class__.__name__))
            logger.info(f"Loaded skeptic plugins: {', '.join(plugin_names)}")

        logger.info("Initialized AdversarialSkeptic")

    def _select_attack_focus(self) -> str:
        """Select which type of attack to focus on"""
        rand = random.random()
        cumulative = 0.0

        attacks = [
            ("edge_case", self.config.edge_case_prob),
            ("type_confusion", self.config.type_confusion_prob),
            ("overflow", self.config.overflow_prob),
            ("malformed", self.config.malformed_prob),
        ]

        for attack_type, prob in attacks:
            cumulative += prob
            if rand < cumulative:
                return attack_type

        return "edge_case"

    async def falsify(
        self,
        program: "Program",
        description: str = "",
        language: str = "python",
        fitness: float | None = None,
    ) -> tuple[bool, list[FalsificationResult]]:
        """
        Attempt to falsify a hypothesis (program).

        Args:
            program: The Program to test
            description: What the code is supposed to do
            language: Programming language

        Returns:
            Tuple of (survived_all_attacks, list_of_results)
        """
        results: list[FalsificationResult] = []
        code = program.code

        logger.info(f"Starting adversarial falsification of program {program.id}")

        entrypoint_name = self.config.entrypoint or ""
        use_training_attacks = (
            language.lower() in ("python", "py")
            and entrypoint_name in ("run_training", "skeptic_entrypoint")
            and ("torch" in code or "StockPredictor" in code)
        )

        # Phase 1: Static Analysis Attacks
        static_result = self._static_analysis_attack(code, language)
        results.append(static_result)
        if not static_result.survived:
            logger.info(f"Program {program.id} failed static analysis")
            return False, results

        # Phase 2: Generated Adversarial Inputs
        num_rounds = int(self.config.num_attack_rounds)
        if getattr(self.config, "adaptive_attack_rounds", False) and fitness is not None:
            min_r = max(1, int(getattr(self.config, "min_attack_rounds", 1)))
            max_r = int(getattr(self.config, "max_attack_rounds", num_rounds) or num_rounds)
            max_r = max(min_r, max_r)
            f = min(max(float(fitness), 0.0), 1.0)
            scaled = min_r + (max_r - min_r) * f
            num_rounds = int(round(scaled))

        # Run any plugin-provided attacks once up front.
        for plugin in self.plugins:
            plugin_name = getattr(plugin, "name", plugin.__class__.__name__)
            try:
                plugin_attacks = await plugin.generate_attacks(program, description, language)
            except Exception as e:
                logger.warning(f"Skeptic plugin {plugin_name} failed: {e}")
                continue
            if plugin_attacks:
                logger.info(
                    f"Running {len(plugin_attacks)} skeptic plugin attacks from {plugin_name}"
                )
            for attack in plugin_attacks:
                result = await self._execute_attack(
                    code, attack["input"], attack.get("attack_type", "plugin"), language
                )
                results.append(result)
                self.attack_history.append(result)
                if not result.survived:
                    logger.info(
                        f"Program {program.id} FALSIFIED by plugin {plugin_name} ({attack.get('attack_type','plugin')})"
                    )
                    return False, results
            if plugin_attacks:
                logger.info(f"Plugin {plugin_name}: all attacks survived")

        for round_num in range(num_rounds):
            if use_training_attacks:
                attacks = self._generate_training_attacks()
            else:
                attack_focus = self._select_attack_focus()

                # Generate adversarial inputs
                if self.llm_ensemble:
                    attacks = await self._generate_adversarial_inputs(
                        code, description, language, attack_focus
                    )
                else:
                    attacks = self._generate_simple_attacks(attack_focus)

            # Execute attacks
            for attack in attacks:
                result = await self._execute_attack(
                    code, attack["input"], attack["attack_type"], language
                )
                results.append(result)
                self.attack_history.append(result)

                if not result.survived:
                    # Track successful attacks for future use
                    if attack["attack_type"] not in self.successful_attacks:
                        self.successful_attacks[attack["attack_type"]] = []
                    self.successful_attacks[attack["attack_type"]].append(attack["input"])

                    logger.info(f"Program {program.id} FALSIFIED by {attack['attack_type']} attack")
                    return False, results

        # Phase 3: Blind Reproduction Test (optional)
        if self.config.enable_blind_reproduction and self.reproduction_llm:
            repro_result = await self._blind_reproduction_test(code, description, language)
            results.append(repro_result)
            if not repro_result.survived:
                logger.info(f"Program {program.id} failed blind reproduction test")
                return False, results

        logger.info(f"Program {program.id} SURVIVED all {len(results)} falsification attempts")
        return True, results

    def _static_analysis_attack(self, code: str, language: str) -> FalsificationResult:
        """
        Static analysis attacks that don't require execution.
        """
        start_time = time.time()

        if language.lower() in ("python", "py"):
            try:
                tree = ast.parse(code)

                # Check for obvious issues
                issues = []

                for node in ast.walk(tree):
                    # Check for bare except
                    if isinstance(node, ast.ExceptHandler) and node.type is None:
                        issues.append(
                            "Bare except clause (catches everything including SystemExit)"
                        )

                    # Check for eval/exec
                    if isinstance(node, ast.Call):
                        if isinstance(node.func, ast.Name):
                            if node.func.id in ("eval", "exec"):
                                issues.append(f"Use of {node.func.id}() - security risk")

                    # Check for assert statements (disabled in production)
                    if isinstance(node, ast.Assert):
                        issues.append("Assert statement (disabled with -O flag)")

                if issues:
                    return FalsificationResult(
                        survived=False,
                        attack_type="static_analysis",
                        error_message="; ".join(issues),
                        execution_time=time.time() - start_time,
                        confidence=0.9,
                        metadata={"issues": issues},
                    )

                return FalsificationResult(
                    survived=True,
                    attack_type="static_analysis",
                    execution_time=time.time() - start_time,
                    confidence=0.8,
                )

            except SyntaxError as e:
                return FalsificationResult(
                    survived=False,
                    attack_type="static_analysis",
                    error_message=f"Syntax error: {e}",
                    execution_time=time.time() - start_time,
                    confidence=1.0,
                )

        # For other languages, just check basic syntax
        return FalsificationResult(
            survived=True,
            attack_type="static_analysis",
            execution_time=time.time() - start_time,
            confidence=0.5,
            metadata={"note": "Limited static analysis for non-Python"},
        )

    def _extract_json_candidate(self, response: str) -> str | None:
        """Extract a likely JSON object from an LLM response."""
        # Prefer fenced json blocks
        fenced = re.search(r"```json\s*(.*?)```", response, re.DOTALL | re.IGNORECASE)
        if fenced:
            return fenced.group(1).strip()

        # Fallback: slice from first { to last }
        start_idx = response.find("{")
        end_idx = response.rfind("}") + 1
        if start_idx >= 0 and end_idx > start_idx:
            return response[start_idx:end_idx].strip()

        return None

    def _strip_trailing_commas(self, json_str: str) -> str:
        """Remove trailing commas before closing braces/brackets."""
        return re.sub(r",\s*([}\]])", r"\1", json_str)

    def _parse_json_flex(self, json_str: str) -> Any | None:
        """Parse JSON with light repairs and Python-literal fallback."""
        # First try strict JSON
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

        # Remove trailing commas and retry
        cleaned = self._strip_trailing_commas(json_str)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Last resort: accept Python literal dict/list
        try:
            return ast.literal_eval(cleaned)
        except Exception:
            return None

    def _normalize_attacks(self, data: Any) -> list[dict[str, Any]]:
        """Normalize parsed data into a list of attack dicts."""
        attacks: list[Any] = []
        if isinstance(data, dict):
            attacks = data.get("attack_inputs") or data.get("attacks") or data.get("inputs") or []
        elif isinstance(data, list):
            attacks = data

        normalized: list[dict[str, Any]] = []
        for item in attacks:
            if not isinstance(item, dict):
                continue
            inp = item.get("input") or item.get("attack_input") or item.get("value")
            if inp is None:
                continue
            attack_type = item.get("attack_type") or item.get("type") or "edge_case"
            normalized.append(
                {
                    "input": str(inp),
                    "attack_type": str(attack_type),
                    "rationale": item.get("rationale", ""),
                }
            )
        return normalized

    async def _generate_adversarial_inputs(
        self,
        code: str,
        description: str,
        language: str,
        attack_focus: str,
    ) -> list[dict[str, Any]]:
        """Generate adversarial inputs using LLM"""
        prompt = ADVERSARIAL_USER.format(
            language=language,
            code=code,
            description=description or "Unknown functionality",
            num_attacks=3,
            attack_focus=attack_focus,
        )

        try:
            response = await self.llm_ensemble.generate_with_context(
                system_message=ADVERSARIAL_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )

            json_str = self._extract_json_candidate(response)
            if json_str:
                data = self._parse_json_flex(json_str)
                normalized = self._normalize_attacks(data)
                if normalized:
                    return normalized

        except Exception as e:
            logger.warning(f"Failed to generate adversarial inputs: {e}")

        # Fallback to simple attacks
        return self._generate_simple_attacks(attack_focus)

    def _generate_simple_attacks(self, attack_focus: str) -> list[dict[str, Any]]:
        """Generate simple adversarial inputs without LLM"""
        attacks = {
            "edge_case": [
                {"input": "[]", "attack_type": "edge_case", "rationale": "Empty input"},
                {"input": "None", "attack_type": "edge_case", "rationale": "Null input"},
                {"input": "[0]", "attack_type": "edge_case", "rationale": "Single element"},
                {"input": '""', "attack_type": "edge_case", "rationale": "Empty string"},
            ],
            "type_confusion": [
                {
                    "input": "[1, 'a', 2.5, None]",
                    "attack_type": "type_confusion",
                    "rationale": "Mixed types",
                },
                {
                    "input": "{'key': 'value'}",
                    "attack_type": "type_confusion",
                    "rationale": "Dict instead of list",
                },
                {
                    "input": "123",
                    "attack_type": "type_confusion",
                    "rationale": "Int instead of collection",
                },
            ],
            "overflow": [
                {"input": "[10**100]", "attack_type": "overflow", "rationale": "Huge number"},
                {"input": "[float('inf')]", "attack_type": "overflow", "rationale": "Infinity"},
                {"input": "[float('nan')]", "attack_type": "overflow", "rationale": "NaN"},
                {"input": "[1e-300]", "attack_type": "overflow", "rationale": "Tiny number"},
            ],
            "malformed": [
                {
                    "input": "b'\\xff\\xfe'",
                    "attack_type": "malformed",
                    "rationale": "Invalid UTF-8",
                },
                {
                    "input": "[object()]",
                    "attack_type": "malformed",
                    "rationale": "Non-serializable",
                },
            ],
        }

        return attacks.get(attack_focus, attacks["edge_case"])

    def _generate_training_attacks(self) -> list[dict[str, Any]]:
        """Generate adversarial inputs for ML training entrypoints.

        These attacks pass small (prices, volumes, targets) arrays as kwargs.
        They are designed to surface numerical instability and shape assumptions.
        """
        return [
            {
                "input": (
                    "(lambda np=__import__('numpy'): {"
                    "'prices': 100 + np.cumsum(np.random.randn(32,60)*0.01, axis=1), "
                    "'volumes': np.abs(np.random.randn(32,60))*1e6, "
                    "'targets': np.random.randn(32)*0.01"
                    "})()"
                ),
                "attack_type": "edge_case",
                "rationale": "Small realistic random-walk market slice",
            },
            {
                "input": (
                    "(lambda np=__import__('numpy'): {"
                    "'prices': np.where(np.random.rand(32,60)<0.05, np.nan, "
                    "100 + np.cumsum(np.random.randn(32,60)*0.01, axis=1)), "
                    "'volumes': np.abs(np.random.randn(32,60))*1e6, "
                    "'targets': np.random.randn(32)*0.01"
                    "})()"
                ),
                "attack_type": "malformed",
                "rationale": "Inject NaNs to test robustness to missing data",
            },
            {
                "input": (
                    "(lambda np=__import__('numpy'): {"
                    "'prices': np.full((32,60), 100.0), "
                    "'volumes': np.abs(np.random.randn(32,60))*1e6, "
                    "'targets': np.zeros(32)"
                    "})()"
                ),
                "attack_type": "logic_error",
                "rationale": "Zero-variance prices/targets can expose divide-by-zero or degenerate loss",
            },
            {
                "input": (
                    "(lambda np=__import__('numpy'): {"
                    "'prices': 100 + np.cumsum(np.random.randn(32,60)*5.0, axis=1), "
                    "'volumes': np.abs(np.random.randn(32,60))*1e8, "
                    "'targets': np.random.randn(32)"
                    "})()"
                ),
                "attack_type": "overflow",
                "rationale": "Extreme volatility and volume outliers",
            },
            {
                "input": (
                    "(lambda np=__import__('numpy'): {"
                    "'prices': 100 + np.cumsum(np.random.randn(16,60)*0.01, axis=1), "
                    "'volumes': np.abs(np.random.randn(16,59))*1e6, "
                    "'targets': np.random.randn(16)*0.01"
                    "})()"
                ),
                "attack_type": "type_confusion",
                "rationale": "Mismatched shapes should be handled gracefully",
            },
        ]

    def _needs_remote_execution(self, code: str) -> bool:
        """Check if code requires remote execution (e.g., imports torch)"""
        # Check if remote execution is configured
        if not self.config.remote_execution or not self.config.remote_host:
            return False

        # Check for imports that require remote execution
        required_imports = self.config.required_imports_for_remote or [
            "torch",
            "tensorflow",
            "jax",
            "cupy",
        ]

        for import_name in required_imports:
            # Check various import patterns
            patterns = [
                f"import {import_name}",
                f"from {import_name}",
            ]
            for pattern in patterns:
                if pattern in code:
                    return True

        return False

    def _check_local_deps(self, code: str) -> tuple[bool, str | None]:
        """Check if required imports are available locally"""
        # Extract imports from code
        import_pattern = r"^(?:import|from)\s+([a-zA-Z_][a-zA-Z0-9_]*)"

        missing = []
        for line in code.split("\n"):
            match = re.match(import_pattern, line.strip())
            if match:
                module_name = match.group(1)
                try:
                    __import__(module_name)
                except ImportError:
                    missing.append(module_name)

        if missing:
            return False, f"Missing imports: {', '.join(missing)}"
        return True, None

    async def _execute_attack(
        self,
        code: str,
        attack_input: str,
        attack_type: str,
        language: str,
    ) -> FalsificationResult:
        """Execute an adversarial attack and capture the result"""
        start_time = time.time()

        if language.lower() not in ("python", "py"):
            # For non-Python, we can't easily execute
            return FalsificationResult(
                survived=True,
                attack_type=attack_type,
                attack_input=attack_input,
                execution_time=time.time() - start_time,
                confidence=0.3,
                metadata={"note": "Execution not supported for this language"},
            )

        # Check if we need remote execution
        needs_remote = self._needs_remote_execution(code)

        # If not using remote, check local dependencies
        if not needs_remote and self.config.skip_missing_deps:
            deps_ok, missing_msg = self._check_local_deps(code)
            if not deps_ok:
                logger.debug(f"Skipping execution due to missing deps: {missing_msg}")
                return FalsificationResult(
                    survived=True,  # Don't fail, just skip
                    attack_type=attack_type,
                    attack_input=attack_input,
                    execution_time=time.time() - start_time,
                    confidence=0.2,  # Low confidence since we didn't actually test
                    metadata={"note": f"Skipped: {missing_msg}", "skipped": True},
                )

        entrypoint_name = self.config.entrypoint

        # Create test harness
        test_code = "\n".join(
            [
                "import sys",
                "import traceback",
                "import math",
                "import copy",
                "",
                "# The code under test (executed in an isolated namespace so __main__ blocks don't run)",
                f"code_under_test = {code!r}",
                "",
                "# Try to execute with adversarial input",
                "try:",
                "    code_globals = {'__name__': '__skeptic__', '__file__': '<candidate>'}",
                "    exec(compile(code_under_test, '<candidate>', 'exec'), code_globals)",
                f"    test_input = {attack_input}",
                "",
                "    import inspect",
                f"    entrypoint = {entrypoint_name!r}",
                "",
                "    def _has_nonfinite(obj, depth=0):",
                "        if depth > 6:",
                "            return False",
                "        if obj is None:",
                "            return False",
                "        if isinstance(obj, bool):",
                "            return False",
                "        if isinstance(obj, (int, float)):",
                "            try:",
                "                return not math.isfinite(float(obj))",
                "            except Exception:",
                "                return False",
                "        try:",
                "            import numpy as np",
                "            if isinstance(obj, (np.ndarray, np.generic)):",
                "                arr = np.asarray(obj)",
                "                flat = arr.ravel()",
                "                if flat.size:",
                "                    sample = flat[: min(1024, flat.size)]",
                "                    return not bool(np.all(np.isfinite(sample)))",
                "                return False",
                "        except Exception:",
                "            pass",
                "        if isinstance(obj, dict):",
                "            for k, v in obj.items():",
                "                if _has_nonfinite(k, depth + 1) or _has_nonfinite(v, depth + 1):",
                "                    return True",
                "            return False",
                "        if isinstance(obj, (list, tuple, set)):",
                "            for item in obj:",
                "                if _has_nonfinite(item, depth + 1):",
                "                    return True",
                "            return False",
                "        return False",
                "",
                "    def _call_with_input(fn, inp):",
                '        """Call fn with inp, supporting dict kwargs or tuple args."""',
                "        if isinstance(inp, dict):",
                "            try:",
                "                sig = inspect.signature(fn)",
                "                params = sig.parameters",
                "                kw_params = {",
                "                    name",
                "                    for name, p in params.items()",
                "                    if p.kind in (",
                "                        inspect.Parameter.POSITIONAL_OR_KEYWORD,",
                "                        inspect.Parameter.KEYWORD_ONLY,",
                "                    )",
                "                }",
                "                accepts_varkw = any(",
                "                    p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()",
                "                )",
                "                if accepts_varkw:",
                "                    return fn(**inp)",
                "                filtered = {k: v for k, v in inp.items() if k in kw_params}",
                "                if filtered:",
                "                    return fn(**filtered)",
                "                return fn(inp)",
                "            except Exception:",
                "                try:",
                "                    return fn(**inp)",
                "                except TypeError:",
                "                    return fn(inp)",
                "        if isinstance(inp, (list, tuple)):",
                "            try:",
                "                return fn(*inp)",
                "            except TypeError:",
                "                return fn(inp)",
                "        return fn(inp)",
                "",
                "    def _resolve_target():",
                "        if entrypoint:",
                "            cand = code_globals.get(entrypoint)",
                "            if callable(cand):",
                "                return cand, entrypoint",
                "        for name, obj in list(code_globals.items()):",
                "            if callable(obj) and not name.startswith('_') and inspect.isfunction(obj):",
                "                return obj, name",
                "        return None, None",
                "",
                "    def _set_seed(seed, device=None):",
                "        try:",
                "            import random as _random",
                "            _random.seed(int(seed))",
                "        except Exception:",
                "            pass",
                "        try:",
                "            import numpy as _np",
                "            _np.random.seed(int(seed))",
                "        except Exception:",
                "            pass",
                "        try:",
                "            import torch as _torch",
                "            _torch.manual_seed(int(seed))",
                "            dev = str(device).lower() if device is not None else ''",
                "            if dev.startswith('cuda') or dev.startswith('gpu'):",
                "                try:",
                "                    _torch.cuda.manual_seed_all(int(seed))",
                "                except Exception:",
                "                    pass",
                "                try:",
                "                    _torch.backends.cudnn.deterministic = True",
                "                    _torch.backends.cudnn.benchmark = False",
                "                except Exception:",
                "                    pass",
                "        except Exception:",
                "            pass",
                "",
                "    def _numeric_metrics(res):",
                "        if isinstance(res, dict):",
                "            out = {}",
                "            for k, v in res.items():",
                "                if isinstance(v, bool):",
                "                    out[k] = 1.0 if v else 0.0",
                "                elif isinstance(v, (int, float)) and not isinstance(v, bool):",
                "                    out[k] = float(v)",
                "            return out",
                "        if isinstance(res, bool):",
                "            return {'output': 1.0 if res else 0.0}",
                "        if isinstance(res, (int, float)) and not isinstance(res, bool):",
                "            return {'output': float(res)}",
                "        return {}",
                "",
                "    def _get_tol(tol, key, default):",
                "        if isinstance(tol, dict):",
                "            try:",
                "                return float(tol.get(key, default))",
                "            except Exception:",
                "                return float(default)",
                "        try:",
                "            return float(tol)",
                "        except Exception:",
                "            return float(default)",
                "",
                "    def _assert_close(a, b, keys=None, atol=1e-3, rtol=0.0):",
                "        ma = _numeric_metrics(a)",
                "        mb = _numeric_metrics(b)",
                "        if keys is None:",
                "            keys = sorted(set(ma.keys()) & set(mb.keys()))",
                "        if not keys:",
                "            raise ValueError('No comparable numeric metrics')",
                "        failures = []",
                "        for k in keys:",
                "            if k not in ma or k not in mb:",
                "                continue",
                "            av = float(ma[k])",
                "            bv = float(mb[k])",
                "            a_tol = _get_tol(atol, k, 1e-3)",
                "            r_tol = _get_tol(rtol, k, 0.0)",
                "            if abs(av - bv) > (a_tol + r_tol * abs(av)):",
                '                failures.append(f"{k}: {av} vs {bv} (atol={a_tol}, rtol={r_tol})")',
                "        if failures:",
                "            raise ValueError('Metric mismatch: ' + '; '.join(failures))",
                "",
                "    def _run_target(fn, kwargs, seed=None):",
                "        if seed is not None:",
                "            _set_seed(seed, kwargs.get('device') if isinstance(kwargs, dict) else None)",
                "        result = _call_with_input(fn, kwargs)",
                "        if _has_nonfinite(result):",
                "            raise ValueError('Non-finite numeric output detected')",
                "        return result",
                "",
                "    if isinstance(test_input, dict) and '__metamorphic__' in test_input:",
                "        spec = test_input.get('__metamorphic__') or {}",
                "        test_name = str(spec.get('test') or '')",
                "        base_kwargs = spec.get('base') or {}",
                "        seed = spec.get('seed')",
                "        atol = spec.get('atol', 1e-3)",
                "        rtol = spec.get('rtol', 0.0)",
                "        keys = spec.get('metrics')",
                "        fn, fn_name = _resolve_target()",
                "        if fn is None:",
                "            print('NO_FUNCTION: No callable function found')",
                "            sys.exit(1)",
                "        try:",
                "            base_res = _run_target(fn, base_kwargs, seed=seed)",
                "            if test_name == 'seed_repeatability':",
                "                res2 = _run_target(fn, base_kwargs, seed=seed)",
                "                _assert_close(base_res, res2, keys=keys, atol=atol, rtol=rtol)",
                "            elif test_name == 'scale_invariance':",
                "                import numpy as _np",
                "                factor = float(spec.get('scale_factor', 10.0))",
                "                scaled = copy.deepcopy(base_kwargs)",
                "                if 'prices' in scaled:",
                "                    scaled['prices'] = _np.asarray(scaled['prices']) * factor",
                "                if 'volumes' in scaled:",
                "                    scaled['volumes'] = _np.asarray(scaled['volumes']) * factor",
                "                scaled_res = _run_target(fn, scaled, seed=seed)",
                "                _assert_close(base_res, scaled_res, keys=keys, atol=atol, rtol=rtol)",
                "            elif test_name == 'shuffle_invariance':",
                "                import numpy as _np",
                "                tr = float(base_kwargs.get('train_ratio', 0.8) or 0.8)",
                "                targets = _np.asarray(base_kwargs.get('targets'))",
                "                n = int(targets.shape[0])",
                "                n_train = int(n * tr)",
                "                n_train = max(1, min(n_train, n - 1))",
                "                rng = _np.random.RandomState(int(seed) if seed is not None else 0)",
                "                train_perm = _np.arange(n_train)",
                "                rng.shuffle(train_perm)",
                "                val_perm = _np.arange(n_train, n)",
                "                rng.shuffle(val_perm)",
                "                perm = _np.concatenate([train_perm, val_perm])",
                "                shuffled = copy.deepcopy(base_kwargs)",
                "                if 'prices' in shuffled:",
                "                    shuffled['prices'] = _np.asarray(shuffled['prices'])[perm]",
                "                if 'volumes' in shuffled:",
                "                    shuffled['volumes'] = _np.asarray(shuffled['volumes'])[perm]",
                "                if 'targets' in shuffled:",
                "                    shuffled['targets'] = _np.asarray(shuffled['targets'])[perm]",
                "                shuf_res = _run_target(fn, shuffled, seed=seed)",
                "                _assert_close(base_res, shuf_res, keys=keys, atol=atol, rtol=rtol)",
                "            elif test_name == 'val_target_leakage':",
                "                import numpy as _np",
                "                metric = str(spec.get('metric', 'mse'))",
                "                min_delta = float(spec.get('min_delta', 0.05))",
                "                tr = float(base_kwargs.get('train_ratio', 0.8) or 0.8)",
                "                targets = _np.asarray(base_kwargs.get('targets'))",
                "                n = int(targets.shape[0])",
                "                n_train = int(n * tr)",
                "                n_train = max(1, min(n_train, n - 1))",
                "                rng = _np.random.RandomState(int(seed) if seed is not None else 0)",
                "                mutated = copy.deepcopy(base_kwargs)",
                "                t2 = _np.array(targets, copy=True)",
                "                t2[n_train:] = rng.uniform(-1.0, 1.0, size=(n - n_train,))",
                "                mutated['targets'] = t2",
                "                mut_res = _run_target(fn, mutated, seed=seed)",
                "                m_base = _numeric_metrics(base_res).get(metric)",
                "                m_mut = _numeric_metrics(mut_res).get(metric)",
                "                if m_base is None or m_mut is None:",
                "                    raise ValueError(f'Missing metric for leakage check: {metric}')",
                "                if float(m_mut) < float(m_base) + min_delta:",
                "                    raise ValueError(",
                "                        f'Leakage check failed: {metric} did not increase enough ({m_base} -> {m_mut})'",
                "                    )",
                "            else:",
                "                raise ValueError(f'Unknown metamorphic test: {test_name}')",
                '            print(f"SUCCESS: metamorphic {test_name} via {fn_name}")',
                "            sys.exit(0)",
                "        except Exception as e:",
                '            print(f"EXCEPTION: metamorphic {test_name} raised {type(e).__name__}: {e}")',
                "            sys.exit(1)",
                "",
                "    if entrypoint:",
                "        target = code_globals.get(entrypoint)",
                "        if callable(target):",
                "            try:",
                "                result = _call_with_input(target, test_input)",
                "                if _has_nonfinite(result):",
                '                    raise ValueError("Non-finite numeric output detected")',
                '                print(f"SUCCESS: {entrypoint}({test_input!r}) = {result!r}")',
                "            except Exception as e:",
                '                print(f"EXCEPTION: {entrypoint}({test_input!r}) raised {type(e).__name__}: {e}")',
                "                sys.exit(1)",
                "        else:",
                "            # Fallback to heuristic function selection",
                "            entrypoint = None",
                "",
                "    if entrypoint is None:",
                "        # Find callable functions",
                "        for name, obj in list(code_globals.items()):",
                "            if callable(obj) and not name.startswith('_'):",
                "                if inspect.isfunction(obj):",
                "                    try:",
                "                        result = _call_with_input(obj, test_input)",
                "                        if _has_nonfinite(result):",
                '                            raise ValueError("Non-finite numeric output detected")',
                '                        print(f"SUCCESS: {name}({test_input!r}) = {result!r}")',
                "                    except Exception as e:",
                '                        print(f"EXCEPTION: {name}({test_input!r}) raised {type(e).__name__}: {e}")',
                "                        sys.exit(1)",
                "                    break",
                "        else:",
                '            print("NO_FUNCTION: No callable function found")',
                "",
                "except Exception as e:",
                '    print(f"SETUP_ERROR: {type(e).__name__}: {e}")',
                "    traceback.print_exc()",
                "    sys.exit(1)",
                "",
            ]
        )

        try:
            if needs_remote:
                return await self._execute_remote_attack(
                    test_code, attack_input, attack_type, start_time
                )
            else:
                return await self._execute_local_attack(
                    test_code, attack_input, attack_type, start_time
                )

        except subprocess.TimeoutExpired:
            return FalsificationResult(
                survived=False,
                attack_type=attack_type,
                attack_input=attack_input,
                error_message="Execution timed out (possible infinite loop)",
                execution_time=self.config.attack_timeout,
                confidence=0.85,
            )

        except Exception as e:
            return FalsificationResult(
                survived=False,
                attack_type=attack_type,
                attack_input=attack_input,
                error_message=f"Execution error: {e}",
                execution_time=time.time() - start_time,
                confidence=0.7,
            )

    async def _execute_local_attack(
        self,
        test_code: str,
        attack_input: str,
        attack_type: str,
        start_time: float,
    ) -> FalsificationResult:
        """Execute attack locally"""
        import os

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(test_code)
            temp_path = f.name

        try:
            result = subprocess.run(
                [sys.executable, temp_path],
                check=False,
                capture_output=True,
                text=True,
                timeout=self.config.attack_timeout,
            )

            output = result.stdout + result.stderr
            output = output[: self.config.max_output_size]

            if result.returncode != 0 or "EXCEPTION:" in output or "SETUP_ERROR:" in output:
                return FalsificationResult(
                    survived=False,
                    attack_type=attack_type,
                    attack_input=attack_input,
                    error_message=output,
                    execution_time=time.time() - start_time,
                    confidence=0.95,
                )

            return FalsificationResult(
                survived=True,
                attack_type=attack_type,
                attack_input=attack_input,
                execution_time=time.time() - start_time,
                confidence=0.9,
                metadata={"output": output, "execution": "local"},
            )

        finally:
            os.unlink(temp_path)

    async def _execute_remote_attack(
        self,
        test_code: str,
        attack_input: str,
        attack_type: str,
        start_time: float,
    ) -> FalsificationResult:
        """Execute attack on remote host via SSH"""
        import os

        remote_host = self.config.remote_host
        remote_python = self.config.remote_python or "python3"
        remote_work_dir = self.config.remote_work_dir

        # Create local temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(test_code)
            local_path = f.name

        try:
            # Ensure remote directory exists
            mkdir_cmd = f'ssh {remote_host} "mkdir -p {remote_work_dir}"'
            subprocess.run(mkdir_cmd, shell=True, capture_output=True, timeout=10)

            # Generate unique remote filename
            import uuid

            remote_filename = f"skeptic_attack_{uuid.uuid4().hex[:8]}.py"
            remote_path = f"{remote_work_dir}/{remote_filename}"

            # Copy file to remote
            scp_cmd = f"scp {local_path} {remote_host}:{remote_path}"
            scp_result = subprocess.run(scp_cmd, shell=True, capture_output=True, timeout=30)

            if scp_result.returncode != 0:
                return FalsificationResult(
                    survived=True,  # Don't fail on infrastructure issues
                    attack_type=attack_type,
                    attack_input=attack_input,
                    execution_time=time.time() - start_time,
                    confidence=0.1,
                    metadata={
                        "note": f"SCP failed: {scp_result.stderr.decode()}",
                        "skipped": True,
                    },
                )

            # Execute on remote
            ssh_cmd = f'ssh {remote_host} "{remote_python} {remote_path}"'
            result = subprocess.run(
                ssh_cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.config.attack_timeout + 10,  # Extra time for SSH overhead
            )

            output = result.stdout + result.stderr
            output = output[: self.config.max_output_size]

            # Cleanup remote file
            cleanup_cmd = f'ssh {remote_host} "rm -f {remote_path}"'
            subprocess.run(cleanup_cmd, shell=True, capture_output=True, timeout=10)

            if result.returncode != 0 or "EXCEPTION:" in output or "SETUP_ERROR:" in output:
                return FalsificationResult(
                    survived=False,
                    attack_type=attack_type,
                    attack_input=attack_input,
                    error_message=output,
                    execution_time=time.time() - start_time,
                    confidence=0.95,
                    metadata={"execution": "remote", "host": remote_host},
                )

            return FalsificationResult(
                survived=True,
                attack_type=attack_type,
                attack_input=attack_input,
                execution_time=time.time() - start_time,
                confidence=0.9,
                metadata={"output": output, "execution": "remote", "host": remote_host},
            )

        finally:
            os.unlink(local_path)

    async def _blind_reproduction_test(
        self,
        original_code: str,
        description: str,
        language: str,
    ) -> FalsificationResult:
        """
        Blind reproduction test: Can another LLM reproduce the functionality
        from the description alone?

        If the description is so vague that reproduction fails, the code
        is considered insufficiently documented/specified.
        """
        if not self.reproduction_llm:
            return FalsificationResult(
                survived=True,
                attack_type="blind_reproduction",
                confidence=0.0,
                metadata={"note": "No reproduction LLM configured"},
            )

        start_time = time.time()

        prompt = REPRODUCTION_USER.format(
            description=description,
            expected_behavior="(See description above)",
            constraints="None specified",
            language=language,
        )

        try:
            response = await self.reproduction_llm.generate_with_context(
                system_message=REPRODUCTION_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )

            # Check if reproduction succeeded
            # Simple heuristic: if response contains code blocks, consider it a success
            has_code = "```" in response or "def " in response or "function " in response

            return FalsificationResult(
                survived=has_code,
                attack_type="blind_reproduction",
                execution_time=time.time() - start_time,
                confidence=0.6,
                metadata={
                    "reproduction_response": response[:1000],
                    "has_code": has_code,
                },
            )

        except Exception as e:
            return FalsificationResult(
                survived=True,  # Don't fail on LLM errors
                attack_type="blind_reproduction",
                error_message=str(e),
                execution_time=time.time() - start_time,
                confidence=0.3,
            )

    def get_attack_statistics(self) -> dict[str, Any]:
        """Get statistics about adversarial attacks"""
        total = len(self.attack_history)
        if total == 0:
            return {"total_attacks": 0}

        failed = sum(1 for r in self.attack_history if not r.survived)

        by_type = {}
        for result in self.attack_history:
            if result.attack_type not in by_type:
                by_type[result.attack_type] = {"total": 0, "failed": 0}
            by_type[result.attack_type]["total"] += 1
            if not result.survived:
                by_type[result.attack_type]["failed"] += 1

        return {
            "total_attacks": total,
            "successful_falsifications": failed,
            "falsification_rate": failed / total,
            "by_attack_type": by_type,
            "most_effective_attack": max(
                by_type.keys(), key=lambda k: by_type[k]["failed"] / max(by_type[k]["total"], 1)
            )
            if by_type
            else None,
        }


# Need sys for subprocess
import sys
