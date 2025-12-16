"""
SpiceAgony - The transformation that integrates new variables into the ontology.

"The Spice Agony. It is the awareness spectrum narcotic, the mind-transformer.
It is dangerous. It is ecstatic. It is necessary."

After a hidden variable passes the Gom Jabbar, SpiceAgony integrates it:
1. Generates computation code for the evaluator
2. Modifies the evaluator to track the new variable
3. Updates the scoring function to use the new dimension
4. Records the ontology transformation

This is the painful but necessary transformation that expands what we can see.
"""

import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class OntologyTransformation:
    """Record of an ontology transformation."""

    timestamp: str
    variable_name: str
    description: str
    computation_code: str
    correlation_at_discovery: float
    incremental_r2_at_discovery: float

    # Integration details
    integration_method: str  # "evaluator_modification", "runtime_injection", etc.
    files_modified: list[str] = field(default_factory=list)

    # Impact tracking
    fitness_before: float | None = None
    fitness_after: float | None = None
    discovery_enabled: bool = True


@dataclass
class SpiceAgonyConfig:
    """Configuration for SpiceAgony."""

    # Integration settings
    auto_integrate: bool = True
    backup_before_modify: bool = True

    # Scoring weights for new variables
    default_variable_weight: float = 0.1  # Weight in final score
    max_variables_per_transformation: int = 3

    # Code generation
    generate_validation_code: bool = True


class SpiceAgony:
    """
    SpiceAgony - integrates validated hidden variables into the system.

    "The sleeper must awaken."
    """

    def __init__(
        self,
        config: SpiceAgonyConfig | None = None,
        evaluator_path: str | None = None,
    ):
        self.config = config or SpiceAgonyConfig()
        self.evaluator_path = evaluator_path

        self.transformations: list[OntologyTransformation] = []
        self.active_variables: dict[str, dict[str, Any]] = {}  # Runtime variable registry

    def integrate_variable(
        self,
        variable: Any,  # HiddenVariable
        validation_result: Any,  # ValidationResult
    ) -> OntologyTransformation:
        """
        Integrate a validated hidden variable into the system.

        Args:
            variable: The HiddenVariable to integrate
            validation_result: ValidationResult from GomJabbar

        Returns:
            OntologyTransformation record
        """
        logger.info(f"SpiceAgony integrating: {variable.name}")

        transformation = OntologyTransformation(
            timestamp=datetime.now().isoformat(),
            variable_name=variable.name,
            description=variable.description,
            computation_code=variable.computation_code,
            correlation_at_discovery=validation_result.correlation,
            incremental_r2_at_discovery=validation_result.incremental_r2,
            integration_method="runtime_injection",
        )

        # Method 1: Runtime injection (always works, no file modification)
        self._inject_runtime(variable, validation_result)

        # Method 2: Evaluator modification (if path provided and auto_integrate)
        if self.evaluator_path and self.config.auto_integrate:
            try:
                self._modify_evaluator(variable, validation_result)
                transformation.integration_method = "evaluator_modification"
                transformation.files_modified.append(self.evaluator_path)
            except Exception as e:
                logger.warning(f"Evaluator modification failed: {e}, using runtime injection")

        self.transformations.append(transformation)
        logger.info(f"✓ {variable.name} integrated via {transformation.integration_method}")

        return transformation

    def _inject_runtime(
        self,
        variable: Any,
        validation_result: Any,
    ) -> None:
        """
        Inject variable into runtime registry for on-the-fly computation.

        This doesn't modify any files - variables are computed at evaluation time
        by the Golden Path controller.
        """
        # Compile the computation function
        namespace = {
            "np": __import__("numpy"),
            "re": __import__("re"),
            "ast": __import__("ast"),
            "math": __import__("math"),
        }

        try:
            exec(variable.computation_code, namespace)

            # Find the compute function
            compute_func = None
            for name, obj in namespace.items():
                if callable(obj) and name.startswith("compute"):
                    compute_func = obj
                    break

            if compute_func:
                self.active_variables[variable.name] = {
                    "compute_func": compute_func,
                    "correlation": validation_result.correlation,
                    "weight": self.config.default_variable_weight,
                    "description": variable.description,
                }
                logger.info(f"Runtime injection successful: {variable.name}")

        except Exception as e:
            logger.error(f"Runtime injection failed for {variable.name}: {e}")

    def _modify_evaluator(
        self,
        variable: Any,
        validation_result: Any,
    ) -> None:
        """
        Modify the evaluator file to include the new variable.

        This is more permanent but requires file modification.
        """
        if not os.path.exists(self.evaluator_path):
            raise FileNotFoundError(f"Evaluator not found: {self.evaluator_path}")

        # Backup
        if self.config.backup_before_modify:
            backup_path = f"{self.evaluator_path}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            with open(self.evaluator_path) as f:
                content = f.read()
            with open(backup_path, "w") as f:
                f.write(content)
            logger.info(f"Backup created: {backup_path}")

        with open(self.evaluator_path) as f:
            content = f.read()

        # Generate the code to add
        new_code = self._generate_integration_code(variable, validation_result)

        # Find insertion point (after imports, before first function)
        # Look for the first function definition
        insertion_match = re.search(r"\n(def \w+\()", content)

        if insertion_match:
            insertion_point = insertion_match.start()
            modified_content = (
                content[:insertion_point]
                + f"\n\n# === GOLDEN PATH DISCOVERY: {variable.name} ===\n"
                + f"# Discovered: {datetime.now().isoformat()}\n"
                + f"# Correlation: {validation_result.correlation:.3f}\n"
                + new_code
                + "\n\n"
                + content[insertion_point:]
            )

            with open(self.evaluator_path, "w") as f:
                f.write(modified_content)

            logger.info(f"Evaluator modified: added {variable.name}")
        else:
            logger.warning("Could not find insertion point in evaluator")

    def _generate_integration_code(
        self,
        variable: Any,
        validation_result: Any,
    ) -> str:
        """Generate Python code to integrate the variable."""

        code = f'''
# Auto-discovered variable: {variable.name}
# {variable.description}
# Correlation with fitness: {validation_result.correlation:.3f}

{variable.computation_code}

def _integrate_{variable.name}_into_score(metrics: dict, code: str = "") -> float:
    """
    Integrate {variable.name} into the evaluation score.
    Auto-generated by Golden Path SpiceAgony.
    """
    try:
        value = compute_{variable.name}(code, metrics)
        # Normalize to 0-1 range (adjust bounds based on observed range)
        normalized = max(0, min(1, value))
        return normalized * {self.config.default_variable_weight}
    except Exception:
        return 0.0
'''
        return code

    def compute_runtime_variables(
        self,
        code: str,
        metrics: dict[str, float],
    ) -> dict[str, float]:
        """
        Compute all runtime-injected variables for a program.

        This is called by the evaluation pipeline to compute discovered variables.
        """
        results = {}

        for var_name, var_info in self.active_variables.items():
            try:
                compute_func = var_info["compute_func"]
                value = compute_func(code, metrics)

                if value is not None and not (
                    hasattr(value, "__iter__") or (isinstance(value, float) and (value != value))
                ):  # NaN check
                    results[var_name] = float(value)
            except Exception as e:
                logger.debug(f"Runtime computation failed for {var_name}: {e}")

        return results

    def get_score_adjustment(
        self,
        code: str,
        metrics: dict[str, float],
    ) -> float:
        """
        Compute score adjustment from all discovered variables.

        Returns a value to add to the base fitness score.
        """
        total_adjustment = 0.0

        runtime_values = self.compute_runtime_variables(code, metrics)

        for var_name, value in runtime_values.items():
            var_info = self.active_variables.get(var_name, {})
            weight = var_info.get("weight", self.config.default_variable_weight)

            # Normalize value (assume 0-1 range, clip if necessary)
            normalized = max(0, min(1, value))
            adjustment = normalized * weight

            total_adjustment += adjustment

        return total_adjustment

    def get_ontology_state(self) -> dict[str, Any]:
        """Get current state of the ontology."""
        return {
            "active_variables": list(self.active_variables.keys()),
            "total_transformations": len(self.transformations),
            "transformations": [
                {
                    "name": t.variable_name,
                    "timestamp": t.timestamp,
                    "correlation": t.correlation_at_discovery,
                    "method": t.integration_method,
                }
                for t in self.transformations
            ],
        }

    def export_discovered_variables(self, output_path: str) -> None:
        """Export all discovered variables to a Python file."""

        code = '''"""
Discovered Variables - Auto-generated by Golden Path

These variables were discovered through ontological exploration and validated
to have significant correlation with fitness.

To use: Import this module and call the compute_* functions on program code.
"""

import numpy as np
import re
import ast
import math

'''

        for t in self.transformations:
            code += f"""
# =============================================================================
# {t.variable_name}
# Discovered: {t.timestamp}
# Correlation: {t.correlation_at_discovery:.3f}
# Incremental R²: {t.incremental_r2_at_discovery:.4f}
# =============================================================================
# {t.description}

{t.computation_code}

"""

        with open(output_path, "w") as f:
            f.write(code)

        logger.info(f"Exported {len(self.transformations)} discovered variables to {output_path}")
