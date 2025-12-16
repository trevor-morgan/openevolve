"""
Physics ML Tool - PINN and Gradient Analysis using JAX/DeepXDE.

"Nature does not jump."

Leverages JAX for automatic differentiation and DeepXDE for physics-informed
learning to analyze artifacts and potentially discover differential equations.
"""

import logging
from typing import ClassVar

import numpy as np

from ..toolkit import Discovery, DiscoveryTool, DiscoveryType, ToolContext

logger = logging.getLogger(__name__)


class PhysicsMLTool(DiscoveryTool):
    """
    Analyzes data using Physics-Informed ML (DeepXDE) and Autodiff (JAX).
    """

    name = "physics_ml"
    description = "Analyze gradients and fit differential equations"
    discovery_types: ClassVar[list[DiscoveryType]] = [DiscoveryType.MATHEMATICAL_FORMULA]
    dependencies: ClassVar[list[str]] = ["jax", "deepxde"]

    async def discover(self, context: ToolContext) -> list[Discovery]:
        # This is a placeholder for a complex implementation.
        # In a real scenario, this would try to fit various PDEs (Burgers, Navier-Stokes, etc.)
        # to the artifact data using DeepXDE's library of equations.

        # For now, we'll implement a simple Gradient Analysis using JAX
        # to find "roughness" or "smoothness" metrics that correlate with fitness.

        import jax.numpy as jnp

        programs = context.programs
        discoveries = []

        # Look for 1D/2D scalar fields
        fields = []
        fitnesses = []

        for prog in programs:
            artifacts = prog.get("metadata", {}).get("artifacts", {})
            for key, value in artifacts.items():
                if isinstance(value, (list, np.ndarray)):
                    arr = np.array(value)
                    if arr.ndim in [1, 2] and arr.size > 20:
                        fields.append((key, arr))
                        fitnesses.append(prog.get("fitness", 0.0))
                        break

        if not fields:
            return []

        # Compute gradient norms using JAX
        grad_norms = []

        def compute_grad_norm(arr):
            arr_j = jnp.array(arr)
            grads = jnp.gradient(arr_j)
            if isinstance(grads, tuple):
                norm = jnp.sqrt(sum(g**2 for g in grads))
            else:
                norm = jnp.abs(grads)
            return float(jnp.mean(norm))

        # JIT compile? Maybe not for variable shapes in loop, but useful principle.
        # jit_grad = jax.jit(compute_grad_norm)

        for _, field in fields:
            try:
                norm = compute_grad_norm(field)
                grad_norms.append(norm)
            except Exception:
                grad_norms.append(0.0)

        # Correlate
        from scipy import stats

        if np.std(grad_norms) > 0:
            corr, _p = stats.pearsonr(grad_norms, fitnesses)
            if abs(corr) > 0.3:
                discoveries.append(
                    Discovery(
                        name=f"physics_grad_norm_{fields[0][0]}",
                        description=f"Gradient norm of {fields[0][0]} correlates with fitness (r={corr:.2f})",
                        discovery_type=DiscoveryType.LATENT_VARIABLE,
                        content={"feature": fields[0][0], "correlation": float(corr)},
                        confidence=abs(corr),
                        computation_code=self._generate_code(),
                    )
                )

        return discoveries

    def _generate_code(self) -> str:
        return """
def compute_grad_norm(artifacts):
    import jax.numpy as jnp
    import numpy as np

    field = None
    for v in artifacts.values():
        arr = np.array(v)
        if arr.ndim in [1, 2] and arr.size > 20:
            field = arr
            break

    if field is None: return 0.0

    try:
        grads = jnp.gradient(jnp.array(field))
        if isinstance(grads, tuple):
            norm = jnp.sqrt(sum(g**2 for g in grads))
        else:
            norm = jnp.abs(grads)
        return float(jnp.mean(norm))
    except:
        return 0.0
"""
