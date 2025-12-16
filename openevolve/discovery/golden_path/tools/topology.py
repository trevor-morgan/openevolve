"""
Topology Analysis Tool - Topological Data Analysis (TDA) using Gudhi.

"The shape of data contains its meaning."

Uses persistent homology to analyze the shape of artifact data (e.g., point clouds,
magnetic field lines, optimization trajectories). Can detect holes, voids, and clusters.
"""

import logging
from typing import ClassVar

import numpy as np

from ..toolkit import Discovery, DiscoveryTool, DiscoveryType, ToolContext

logger = logging.getLogger(__name__)


class TopologyAnalysisTool(DiscoveryTool):
    """
    Analyzes the topology of numerical artifacts using Gudhi.
    """

    name = "topology_analysis"
    description = "Analyze topological features (Betti numbers, persistence) of data artifacts"
    discovery_types: ClassVar[list[DiscoveryType]] = [
        DiscoveryType.MATHEMATICAL_FORMULA,
        DiscoveryType.LATENT_VARIABLE,
    ]
    dependencies: ClassVar[list[str]] = ["gudhi"]

    async def discover(self, context: ToolContext) -> list[Discovery]:
        import gudhi

        programs = context.programs
        discoveries = []

        # Collect point cloud data from artifacts
        point_clouds = []
        fitnesses = []

        for prog in programs:
            artifacts = prog.get("metadata", {}).get("artifacts", {})
            # Look for 2D/3D arrays that could be point clouds or fields
            for key, value in artifacts.items():
                if isinstance(value, (list, np.ndarray)):
                    arr = np.array(value)
                    # Heuristic: Arrays with shape (N, 2) or (N, 3) where N > 10
                    if arr.ndim == 2 and arr.shape[1] in [2, 3] and arr.shape[0] > 10:
                        point_clouds.append((key, arr))
                        fitnesses.append(prog.get("fitness", 0.0))
                        break  # Only take one per program for now

        if not point_clouds:
            return []

        # Analyze Betti numbers

        feature_name = point_clouds[0][0]

        betti_0_vals = []
        betti_1_vals = []

        for _, cloud in point_clouds:
            try:
                # Rips complex
                rips = gudhi.RipsComplex(points=cloud)
                simplex_tree = rips.create_simplex_tree(max_dimension=2)
                simplex_tree.persistence()

                b0 = simplex_tree.betti_numbers()[0] if len(simplex_tree.betti_numbers()) > 0 else 0
                b1 = simplex_tree.betti_numbers()[1] if len(simplex_tree.betti_numbers()) > 1 else 0

                betti_0_vals.append(b0)
                betti_1_vals.append(b1)
            except Exception:
                betti_0_vals.append(0)
                betti_1_vals.append(0)

        # Correlate
        from scipy import stats

        fitnesses = np.array(fitnesses[: len(betti_0_vals)])  # Trim if needed

        for dim, vals in [(0, betti_0_vals), (1, betti_1_vals)]:
            if np.std(vals) > 0:
                corr, _p = stats.pearsonr(vals, fitnesses)
                if abs(corr) > 0.3:
                    discoveries.append(
                        Discovery(
                            name=f"topology_betti_{dim}_{feature_name}",
                            description=f"Betti-{dim} number of {feature_name} correlates with fitness (r={corr:.2f})",
                            discovery_type=DiscoveryType.LATENT_VARIABLE,
                            content={
                                "dimension": dim,
                                "feature": feature_name,
                                "correlation": float(corr),
                            },
                            confidence=abs(corr),
                            computation_code=self._generate_code(dim),
                        )
                    )

        return discoveries

    def _generate_code(self, dim: int) -> str:
        return f"""
def compute_betti_{dim}(artifacts):
    import gudhi
    import numpy as np

    # Extract first valid point cloud
    cloud = None
    for v in artifacts.values():
        arr = np.array(v)
        if arr.ndim == 2 and arr.shape[1] in [2, 3] and arr.shape[0] > 10:
            cloud = arr
            break

    if cloud is None: return 0

    try:
        rips = gudhi.RipsComplex(points=cloud)
        st = rips.create_simplex_tree(max_dimension=2)
        st.persistence()
        bettis = st.betti_numbers()
        return bettis[{dim}] if len(bettis) > {dim} else 0
    except:
        return 0
"""
