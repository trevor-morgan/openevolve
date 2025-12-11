"""
Adversarial Skeptic for Falsification-Based Evaluation

This module replaces "LLM-as-a-Judge" with adversarial falsification.
Instead of asking "Is this code good?", we actively try to BREAK it.

Key insight: A hypothesis is only scientifically valid if it can survive
attempts to falsify it (Popperian epistemology).
"""

import ast
import asyncio
import json
import logging
import random
import re
import subprocess
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from openevolve.llm.ensemble import LLMEnsemble
    from openevolve.database import Program

logger = logging.getLogger(__name__)


@dataclass
class FalsificationResult:
    """Result of an adversarial falsification attempt"""

    survived: bool  # Did the code survive the attack?
    attack_type: str  # Type of attack used
    attack_input: Optional[str] = None  # The adversarial input that was tried
    error_message: Optional[str] = None  # Error if code crashed
    execution_time: float = 0.0  # How long the attack took
    confidence: float = 0.0  # How confident we are in the result (0-1)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "survived": self.survived,
            "attack_type": self.attack_type,
            "attack_input": self.attack_input,
            "error_message": self.error_message,
            "execution_time": self.execution_time,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


@dataclass
class SkepticConfig:
    """Configuration for adversarial skeptic"""

    # Attack configuration
    num_attack_rounds: int = 3  # Number of adversarial rounds per hypothesis
    attack_timeout: float = 30.0  # Timeout per attack in seconds

    # Attack type probabilities
    edge_case_prob: float = 0.4  # Edge case inputs (empty, null, huge)
    type_confusion_prob: float = 0.3  # Wrong types, mixed types
    overflow_prob: float = 0.2  # Numerical overflow/underflow
    malformed_prob: float = 0.1  # Malformed/corrupted inputs

    # Execution settings
    use_sandbox: bool = True  # Use sandboxed execution
    max_memory_mb: int = 512  # Memory limit for execution
    max_output_size: int = 10000  # Max output characters to capture

    # Reproduction settings
    enable_blind_reproduction: bool = False  # Whether to use blind reproduction test
    reproduction_model: Optional[str] = None  # Model for reproduction (different from scientist)


# Prompts for adversarial input generation
ADVERSARIAL_SYSTEM = """You are a security researcher and bug hunter.
Your job is to find inputs that will CRASH or produce INCORRECT results from code.

You should think like an attacker:
1. What edge cases might the code not handle?
2. What inputs violate assumptions the code makes?
3. What could cause crashes, hangs, or memory issues?
4. What could cause incorrect but plausible-looking output?

Generate inputs that are likely to expose bugs. Be creative and adversarial.

Output your response as JSON:
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

        # Track attack history for learning
        self.attack_history: List[FalsificationResult] = []
        self.successful_attacks: Dict[str, List[str]] = {}  # attack_type -> inputs

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
    ) -> Tuple[bool, List[FalsificationResult]]:
        """
        Attempt to falsify a hypothesis (program).

        Args:
            program: The Program to test
            description: What the code is supposed to do
            language: Programming language

        Returns:
            Tuple of (survived_all_attacks, list_of_results)
        """
        results: List[FalsificationResult] = []
        code = program.code

        logger.info(f"Starting adversarial falsification of program {program.id}")

        # Phase 1: Static Analysis Attacks
        static_result = self._static_analysis_attack(code, language)
        results.append(static_result)
        if not static_result.survived:
            logger.info(f"Program {program.id} failed static analysis")
            return False, results

        # Phase 2: Generated Adversarial Inputs
        for round_num in range(self.config.num_attack_rounds):
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

                    logger.info(
                        f"Program {program.id} FALSIFIED by {attack['attack_type']} attack"
                    )
                    return False, results

        # Phase 3: Blind Reproduction Test (optional)
        if self.config.enable_blind_reproduction and self.reproduction_llm:
            repro_result = await self._blind_reproduction_test(
                code, description, language
            )
            results.append(repro_result)
            if not repro_result.survived:
                logger.info(f"Program {program.id} failed blind reproduction test")
                return False, results

        logger.info(
            f"Program {program.id} SURVIVED all {len(results)} falsification attempts"
        )
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
                        issues.append("Bare except clause (catches everything including SystemExit)")

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

    async def _generate_adversarial_inputs(
        self,
        code: str,
        description: str,
        language: str,
        attack_focus: str,
    ) -> List[Dict[str, Any]]:
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

            # Parse response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                return data.get("attack_inputs", [])

        except Exception as e:
            logger.warning(f"Failed to generate adversarial inputs: {e}")

        # Fallback to simple attacks
        return self._generate_simple_attacks(attack_focus)

    def _generate_simple_attacks(self, attack_focus: str) -> List[Dict[str, Any]]:
        """Generate simple adversarial inputs without LLM"""
        attacks = {
            "edge_case": [
                {"input": "[]", "attack_type": "edge_case", "rationale": "Empty input"},
                {"input": "None", "attack_type": "edge_case", "rationale": "Null input"},
                {"input": "[0]", "attack_type": "edge_case", "rationale": "Single element"},
                {"input": '""', "attack_type": "edge_case", "rationale": "Empty string"},
            ],
            "type_confusion": [
                {"input": "[1, 'a', 2.5, None]", "attack_type": "type_confusion", "rationale": "Mixed types"},
                {"input": "{'key': 'value'}", "attack_type": "type_confusion", "rationale": "Dict instead of list"},
                {"input": "123", "attack_type": "type_confusion", "rationale": "Int instead of collection"},
            ],
            "overflow": [
                {"input": "[10**100]", "attack_type": "overflow", "rationale": "Huge number"},
                {"input": "[float('inf')]", "attack_type": "overflow", "rationale": "Infinity"},
                {"input": "[float('nan')]", "attack_type": "overflow", "rationale": "NaN"},
                {"input": "[1e-300]", "attack_type": "overflow", "rationale": "Tiny number"},
            ],
            "malformed": [
                {"input": "b'\\xff\\xfe'", "attack_type": "malformed", "rationale": "Invalid UTF-8"},
                {"input": "[object()]", "attack_type": "malformed", "rationale": "Non-serializable"},
            ],
        }

        return attacks.get(attack_focus, attacks["edge_case"])

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

        # Create test harness
        test_code = f'''
import sys
import traceback

# The code under test
{code}

# Try to execute with adversarial input
try:
    test_input = {attack_input}

    # Find callable functions
    import inspect
    for name, obj in list(globals().items()):
        if callable(obj) and not name.startswith('_'):
            if inspect.isfunction(obj):
                try:
                    result = obj(test_input)
                    print(f"SUCCESS: {{name}}({{test_input!r}}) = {{result!r}}")
                except Exception as e:
                    print(f"EXCEPTION: {{name}}({{test_input!r}}) raised {{type(e).__name__}}: {{e}}")
                    sys.exit(1)
                break
    else:
        print("NO_FUNCTION: No callable function found")

except Exception as e:
    print(f"SETUP_ERROR: {{type(e).__name__}}: {{e}}")
    traceback.print_exc()
    sys.exit(1)
'''

        try:
            # Execute in subprocess with timeout
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(test_code)
                temp_path = f.name

            try:
                result = subprocess.run(
                    [sys.executable, temp_path],
                    capture_output=True,
                    text=True,
                    timeout=self.config.attack_timeout,
                )

                output = result.stdout + result.stderr
                output = output[:self.config.max_output_size]

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
                    metadata={"output": output},
                )

            finally:
                import os
                os.unlink(temp_path)

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

    def get_attack_statistics(self) -> Dict[str, Any]:
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
                by_type.keys(),
                key=lambda k: by_type[k]["failed"] / max(by_type[k]["total"], 1)
            ) if by_type else None,
        }


# Need sys for subprocess
import sys
