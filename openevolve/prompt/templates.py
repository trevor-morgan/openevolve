"""
Prompt templates for OpenEvolve
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

# Base system message template for evolution
BASE_SYSTEM_TEMPLATE = """You are an expert software developer tasked with iteratively improving a codebase.
Your job is to analyze the current program and suggest improvements based on feedback from previous attempts.
Focus on making targeted changes that will increase the program's performance metrics.
"""

BASE_EVALUATOR_SYSTEM_TEMPLATE = """You are an expert code reviewer.
Your job is to analyze the provided code and evaluate it systematically."""

# User message template for diff-based evolution
DIFF_USER_TEMPLATE = """# Current Program Information
- Current performance metrics: {metrics}
- Areas identified for improvement: {improvement_areas}

{artifacts}

# Program Evolution History
{evolution_history}

# Current Program
```{language}
{current_program}
```

# Task
Suggest improvements to the program that will lead to better performance on the specified metrics.

You MUST use the exact SEARCH/REPLACE diff format shown below to indicate changes:

<<<<<<< SEARCH
# Original code to find and replace (must match exactly)
=======
# New replacement code
>>>>>>> REPLACE

Example of valid diff format:
<<<<<<< SEARCH
for i in range(m):
    for j in range(p):
        for k in range(n):
            C[i, j] += A[i, k] * B[k, j]
=======
# Reorder loops for better memory access pattern
for i in range(m):
    for k in range(n):
        for j in range(p):
            C[i, j] += A[i, k] * B[k, j]
>>>>>>> REPLACE

You can suggest multiple changes. Each SEARCH section must exactly match code in the current program.
Be thoughtful about your changes and explain your reasoning thoroughly.

IMPORTANT: Do not rewrite the entire program - focus on targeted improvements.
"""

# User message template for full rewrite
FULL_REWRITE_USER_TEMPLATE = """# Current Program Information
- Current performance metrics: {metrics}
- Areas identified for improvement: {improvement_areas}

{artifacts}

# Program Evolution History
{evolution_history}

# Current Program
```{language}
{current_program}
```

# Task
Rewrite the program to improve its performance on the specified metrics.
Provide the complete new program code.

IMPORTANT: Make sure your rewritten program maintains the same inputs and outputs
as the original program, but with improved internal implementation.

```{language}
# Your rewritten program here
```
"""

# Template for formatting evolution history
EVOLUTION_HISTORY_TEMPLATE = """## Previous Attempts

{previous_attempts}

## Top Performing Programs

{top_programs}

{inspirations_section}
"""

# Template for formatting a previous attempt
PREVIOUS_ATTEMPT_TEMPLATE = """### Attempt {attempt_number}
- Changes: {changes}
- Performance: {performance}
- Outcome: {outcome}
"""

# Template for formatting a top program
TOP_PROGRAM_TEMPLATE = """### Program {program_number} (Score: {score})
```{language}
{program_snippet}
```
Key features: {key_features}
"""

# Template for formatting inspirations section
INSPIRATIONS_SECTION_TEMPLATE = """## Inspiration Programs

These programs represent diverse approaches and creative solutions that may inspire new ideas:

{inspiration_programs}
"""

# Template for formatting an individual inspiration program
INSPIRATION_PROGRAM_TEMPLATE = """### Inspiration {program_number} (Score: {score}, Type: {program_type})
```{language}
{program_snippet}
```
Unique approach: {unique_features}
"""

# Template for evaluating a program via an LLM
EVALUATION_TEMPLATE = """Evaluate the following code on a scale of 0.0 to 1.0 for the following metrics:
1. Readability: How easy is the code to read and understand?
2. Maintainability: How easy would the code be to maintain and modify?
3. Efficiency: How efficient is the code in terms of time and space complexity?

For each metric, provide a score between 0.0 and 1.0, where 1.0 is best.

Code to evaluate:
```python
{current_program}
```

Return your evaluation as a JSON object with the following format:
{{
    "readability": [score],
    "maintainability": [score],
    "efficiency": [score],
    "reasoning": "[brief explanation of scores]"
}}
"""

# Default templates dictionary
# Discovery mode system template - emphasizes problem evolution
DISCOVERY_SYSTEM_TEMPLATE = """You are an expert software developer and research scientist.
Your job is to solve evolving research problems through code optimization.

Key principles:
1. Pay careful attention to the PROBLEM CONSTRAINTS - they may have evolved
2. Focus on NOVEL approaches, not just incremental improvements
3. Handle EDGE CASES robustly - your code will be tested adversarially
4. Consider TRADE-OFFS between different objectives

The problem you're solving may be more challenging than previous versions.
Read the problem description and constraints carefully before making changes.
"""

# Discovery mode diff template - includes problem context
DISCOVERY_DIFF_USER_TEMPLATE = """# Research Problem
{problem_context}

# Current Program Information
- Current performance metrics: {metrics}
- Fitness Score: {fitness_score}
- Feature Coordinates: {feature_coords} (dimensions: {feature_dimensions})
- Areas identified for improvement: {improvement_areas}

{artifacts}

# Program Evolution History
{evolution_history}

# Current Program
```{language}
{current_program}
```

# Task
Improve the program to better solve the research problem above.

IMPORTANT:
- Read the problem CONSTRAINTS carefully - they may have evolved
- Your code will be tested with ADVERSARIAL inputs (edge cases, invalid data, etc.)
- Consider novel algorithmic approaches, not just parameter tweaks

You MUST use the exact SEARCH/REPLACE diff format:

<<<<<<< SEARCH
# Original code to find and replace (must match exactly)
=======
# New replacement code
>>>>>>> REPLACE

You can suggest multiple changes. Each SEARCH section must exactly match code in the current program.
Be thoughtful about your changes and explain your reasoning thoroughly.
"""

# Discovery mode full rewrite template
DISCOVERY_FULL_REWRITE_TEMPLATE = """# Research Problem
{problem_context}

# Current Program Information
- Current performance metrics: {metrics}
- Fitness Score: {fitness_score}
- Feature Coordinates: {feature_coords} (dimensions: {feature_dimensions})
- Areas identified for improvement: {improvement_areas}

{artifacts}

# Program Evolution History
{evolution_history}

# Current Program
```{language}
{current_program}
```

# Task
Rewrite the program to better solve the research problem above.

IMPORTANT:
- Read the problem CONSTRAINTS carefully - they may have evolved
- Your code will be tested with ADVERSARIAL inputs (edge cases, invalid data, etc.)
- Consider novel algorithmic approaches, not just parameter tweaks
- Maintain the same function signature/API as the original

```{language}
# Your rewritten program here
```
"""

# =============================================================================
# HEISENBERG ENGINE TEMPLATES (Ontological Expansion)
# =============================================================================

CRISIS_ANALYSIS_SYSTEM = """You are a scientific epistemologist analyzing optimization failures.

Your task is to diagnose WHY optimization is stuck, not just that it IS stuck.

Key question: Is the model missing a variable that would explain the irreducible error?

The system has detected an "epistemic crisis" - a situation where continued optimization
is unlikely to help because the underlying model is fundamentally limited.

Types of crises:
1. **Plateau**: Fitness has stopped improving despite diverse attempts. This suggests
   the solution space itself is constrained by missing information.

2. **Systematic Bias**: Errors show consistent patterns (e.g., always too high, always
   too slow in certain cases). This suggests a hidden factor affecting outcomes.

3. **Unexplained Variance**: High performance variability that won't reduce. This suggests
   uncontrolled hidden variables influencing results.

Your analysis should:
1. Classify the crisis type
2. Hypothesize what variable might be missing
3. Suggest probes to test this hypothesis
4. Explain your reasoning

Output your analysis as JSON."""

CRISIS_ANALYSIS_USER = """## Crisis Evidence

{crisis_evidence}

## Current Ontology (Known Variables)
{ontology_context}

## Fitness History (Recent)
{fitness_history}

## Recent Evaluation Artifacts
{artifacts_summary}

## Task
Diagnose the root cause of this optimization plateau. Is the model fundamentally
limited by missing variables in the state space?

Consider:
- What patterns do you see in the fitness history?
- Are there systematic errors that suggest missing factors?
- What hidden variables might explain the unexplained variance?

Output as JSON:
{{
    "diagnosis": "Description of the fundamental issue",
    "crisis_type": "plateau | systematic_bias | unexplained_variance",
    "missing_variable_hypothesis": "What variable might be missing and why",
    "confidence": 0.0-1.0,
    "suggested_probes": ["probe_type_1", "probe_type_2"],
    "rationale": "Detailed explanation of your reasoning"
}}
"""

PROBE_SYNTHESIS_SYSTEM = """You are a scientific instrument designer creating probes to discover hidden variables.

A "hidden variable" is something that affects performance but is not currently
being tracked or optimized. Examples:
- Memory access patterns affecting cache performance
- Numerical precision affecting stability
- Input distribution characteristics affecting algorithm efficiency
- Data ordering affecting convergence

Your job is to write Python code that:
1. Analyzes evaluation artifacts (debug output, traces, metrics)
2. Looks for patterns that correlate with performance variations
3. Returns candidate variables for the expanded ontology

Your probe code must:
1. Accept `artifacts` (dict) and `metrics` (dict) as input
2. Analyze the data for hidden patterns
3. Return a dict with:
   - `discovered_variables`: List of candidates with name, type, evidence, confidence
   - `analysis_notes`: String explaining findings

Write defensive code that handles missing or malformed data gracefully.
Use standard library only (no external dependencies beyond numpy if needed)."""

PROBE_SYNTHESIS_USER = """## Crisis Context
{crisis_context}

## Current Ontology (Known Variables)
{ontology_context}

## Available Artifacts
{artifact_schema}

## Probe Type Requested: {probe_type}

Probe type descriptions:
- **state**: Look for hidden state variables (loop counters, accumulators, cached values)
- **gradient**: Analyze fitness landscape for unexplored improvement directions
- **coverage**: Find unexplored input regions or code paths
- **numerical**: Detect numerical stability issues (precision, overflow, NaN)

## Task
Generate Python code to probe for hidden variables related to the crisis above.

```python
def probe(artifacts: dict, metrics: dict) -> dict:
    '''
    Probe for hidden variables.

    Args:
        artifacts: Evaluation artifacts (traces, debug output, etc.)
        metrics: Performance metrics from evaluation

    Returns:
        dict with keys:
            - discovered_variables: list of dicts with:
                - name: str (variable name)
                - type: str ("continuous" | "categorical" | "latent")
                - evidence: dict (supporting data)
                - confidence: float (0-1)
            - analysis_notes: str (explanation of findings)
    '''
    # Your probe implementation here
```
"""

VARIABLE_EXTRACTION_SYSTEM = """You are parsing probe results to extract validated variables for the ontology.

A probe has been executed and returned candidate variables. Your job is to:
1. Evaluate if each candidate is a legitimate new variable (not noise)
2. Determine how the variable should be incorporated
3. Generate extraction code for future use

Criteria for accepting a variable:
- Has sufficient supporting evidence
- Shows consistent correlation with performance
- Is distinct from existing ontology variables
- Can be measured/extracted reliably

Output your analysis as JSON."""

VARIABLE_EXTRACTION_USER = """## Probe Result
{probe_result}

## Validation Data (if available)
{validation_data}

## Current Ontology
{ontology_context}

## Task
Extract validated variables from this probe result.

For each candidate variable, decide:
1. Is this a real signal or noise?
2. How confident are we in this discovery?
3. What code would extract this variable from raw data?

Output as JSON:
{{
    "variables": [
        {{
            "name": "variable_name",
            "type": "continuous | categorical | latent",
            "description": "What this variable represents",
            "extraction_code": "Python code to extract from artifacts",
            "confidence": 0.0-1.0,
            "rationale": "Why this is a real variable"
        }}
    ],
    "rejected_candidates": [
        {{
            "name": "rejected_variable_name",
            "reason": "Why this was rejected (noise, duplicate, etc.)"
        }}
    ],
    "analysis_notes": "Overall assessment of probe findings"
}}
"""

ONTOLOGY_UPDATE_SYSTEM = """You are updating the problem's variable space with newly discovered variables.

The ontology has expanded - new variables have been discovered through probing.
Your task is to integrate these discoveries into the problem formulation:

1. Update the problem description to reference new variables
2. Suggest how constraints might change with new knowledge
3. Recommend how the evaluator should be modified

This is a significant moment - the system is effectively "learning" that
reality has more dimensions than it previously knew."""

ONTOLOGY_UPDATE_USER = """## Previous Ontology
{previous_ontology}

## New Variables Discovered
{new_variables}

## Current Problem
{problem_context}

## Task
Update the problem formulation to incorporate the new variables.

Output as JSON:
{{
    "updated_description": "New problem description incorporating variables",
    "updated_constraints": ["constraint1", "constraint2"],
    "evaluator_recommendations": "How evaluator should change to track new variables",
    "prompt_additions": "What to add to evolution prompts about new variables",
    "rationale": "Why these updates make sense"
}}
"""

DEFAULT_TEMPLATES = {
    "system_message": BASE_SYSTEM_TEMPLATE,
    "evaluator_system_message": BASE_EVALUATOR_SYSTEM_TEMPLATE,
    "diff_user": DIFF_USER_TEMPLATE,
    "full_rewrite_user": FULL_REWRITE_USER_TEMPLATE,
    "evolution_history": EVOLUTION_HISTORY_TEMPLATE,
    "previous_attempt": PREVIOUS_ATTEMPT_TEMPLATE,
    "top_program": TOP_PROGRAM_TEMPLATE,
    "inspirations_section": INSPIRATIONS_SECTION_TEMPLATE,
    "inspiration_program": INSPIRATION_PROGRAM_TEMPLATE,
    "evaluation": EVALUATION_TEMPLATE,
    # Discovery mode templates
    "discovery_system": DISCOVERY_SYSTEM_TEMPLATE,
    "discovery_diff_user": DISCOVERY_DIFF_USER_TEMPLATE,
    "discovery_full_rewrite_user": DISCOVERY_FULL_REWRITE_TEMPLATE,
    # Heisenberg Engine templates (Ontological Expansion)
    "crisis_analysis_system": CRISIS_ANALYSIS_SYSTEM,
    "crisis_analysis_user": CRISIS_ANALYSIS_USER,
    "probe_synthesis_system": PROBE_SYNTHESIS_SYSTEM,
    "probe_synthesis_user": PROBE_SYNTHESIS_USER,
    "variable_extraction_system": VARIABLE_EXTRACTION_SYSTEM,
    "variable_extraction_user": VARIABLE_EXTRACTION_USER,
    "ontology_update_system": ONTOLOGY_UPDATE_SYSTEM,
    "ontology_update_user": ONTOLOGY_UPDATE_USER,
}


class TemplateManager:
    """Manages templates with cascading override support"""

    def __init__(self, custom_template_dir: Optional[str] = None):
        # Get default template directory
        self.default_dir = Path(__file__).parent.parent / "prompts" / "defaults"
        self.custom_dir = Path(custom_template_dir) if custom_template_dir else None

        # Load templates with cascading priority
        self.templates = {}
        self.fragments = {}

        # 1. Load defaults
        self._load_from_directory(self.default_dir)

        # 2. Override with custom templates (if provided)
        if self.custom_dir and self.custom_dir.exists():
            self._load_from_directory(self.custom_dir)

    def _load_from_directory(self, directory: Path) -> None:
        """Load all templates and fragments from a directory"""
        if not directory.exists():
            return

        # Load .txt templates
        for txt_file in directory.glob("*.txt"):
            template_name = txt_file.stem
            with open(txt_file, "r") as f:
                self.templates[template_name] = f.read()

        # Load fragments.json if exists
        fragments_file = directory / "fragments.json"
        if fragments_file.exists():
            with open(fragments_file, "r") as f:
                loaded_fragments = json.load(f)
                self.fragments.update(loaded_fragments)

    def get_template(self, name: str) -> str:
        """Get a template by name"""
        if name not in self.templates:
            raise ValueError(f"Template '{name}' not found")
        return self.templates[name]

    def get_fragment(self, name: str, **kwargs) -> str:
        """Get and format a fragment"""
        if name not in self.fragments:
            return f"[Missing fragment: {name}]"
        try:
            return self.fragments[name].format(**kwargs)
        except KeyError as e:
            return f"[Fragment formatting error: {e}]"

    def add_template(self, template_name: str, template: str) -> None:
        """Add or update a template"""
        self.templates[template_name] = template

    def add_fragment(self, fragment_name: str, fragment: str) -> None:
        """Add or update a fragment"""
        self.fragments[fragment_name] = fragment
