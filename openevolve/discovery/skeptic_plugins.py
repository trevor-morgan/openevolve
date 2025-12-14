"""
Skeptic plugin registry.

Plugins allow task-specific adversarial attacks to be added to the Popperian
falsification loop without modifying core skeptic logic.

Each plugin exposes an async `generate_attacks(program, description, language)` that
returns a list of dicts with keys:
  - input: Python literal or expression string passed to the entrypoint
  - attack_type: label for logging
  - rationale: short reason
"""

from __future__ import annotations

import importlib
import inspect
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

if True:  # typing-only, avoid circular imports at runtime
    from openevolve.database import Program

logger = logging.getLogger(__name__)


class SkepticPlugin(Protocol):
    """Protocol for skeptic plugins."""

    name: str

    async def generate_attacks(
        self, program: Program, description: str, language: str
    ) -> list[dict[str, Any]]: ...


_PLUGIN_FACTORIES: dict[str, Callable[[], SkepticPlugin]] = {}


def register_plugin(name: str) -> Callable[[type], type]:
    """Class decorator to register a built-in plugin."""

    def _decorator(cls: type) -> type:
        _PLUGIN_FACTORIES[name] = cls  # type: ignore[assignment]
        return cls

    return _decorator


def _load_external_plugin(spec: str) -> SkepticPlugin | None:
    """Load plugin from 'module:attr' spec."""
    if ":" not in spec:
        return None
    mod_name, attr_name = spec.split(":", 1)
    try:
        mod = importlib.import_module(mod_name)
        attr = getattr(mod, attr_name)
        if inspect.isclass(attr):
            return attr()  # type: ignore[call-arg]
        if callable(attr):
            plugin = attr()
            return plugin  # type: ignore[return-value]
    except Exception as e:
        logger.warning(f"Failed to load external skeptic plugin {spec}: {e}")
    return None


def load_plugins(names: list[str] | None) -> list[SkepticPlugin]:
    """Instantiate plugins by name."""
    plugins: list[SkepticPlugin] = []
    for name in names or []:
        if name in _PLUGIN_FACTORIES:
            try:
                plugins.append(_PLUGIN_FACTORIES[name]())
            except Exception as e:
                logger.warning(f"Failed to init skeptic plugin {name}: {e}")
            continue
        ext = _load_external_plugin(name)
        if ext is not None:
            plugins.append(ext)
        else:
            logger.warning(f"Unknown skeptic plugin: {name}")
    return plugins


# -----------------------
# Built-in plugins
# -----------------------


@register_plugin("ml_data_perturbation")
@dataclass
class MLDataPerturbationPlugin:
    """Adds stronger, task-specific data perturbations for ML training entrypoints."""

    name: str = "ml_data_perturbation"

    async def generate_attacks(
        self, program: Program, description: str, language: str
    ) -> list[dict[str, Any]]:
        # Use the same (prices, volumes, targets) kwargs schema as training attacks.
        return [
            {
                "input": (
                    "(lambda np=__import__('numpy'): {"
                    "'prices': 100 + np.cumsum(np.random.randn(64,60)*0.02, axis=1), "
                    "'volumes': np.abs(np.random.randn(64,60))*1e6, "
                    "'targets': np.random.randn(64)*0.02"
                    "})()"
                ),
                "attack_type": "ml_robustness",
                "rationale": "Larger random-walk slice to surface instability",
            },
            {
                "input": (
                    "(lambda np=__import__('numpy'): {"
                    "'prices': 100 + np.cumsum(np.random.randn(32,60)*0.01, axis=1), "
                    "'volumes': np.abs(np.random.randn(32,60))*1e6, "
                    "'targets': np.random.randn(32)*0.01, "
                    "'train_ratio': 0.5"
                    "})()"
                ),
                "attack_type": "ml_robustness",
                "rationale": "Alter split ratio to check assumptions about train/val sizes",
            },
            {
                "input": (
                    "(lambda np=__import__('numpy'): {"
                    "'prices': 100 + np.cumsum(np.random.randn(32,60)*0.01, axis=1), "
                    "'volumes': np.abs(np.random.randn(32,60))*1e6, "
                    "'targets': -np.random.randn(32)*0.01"
                    "})()"
                ),
                "attack_type": "ml_robustness",
                "rationale": "Sign-flipped targets stress directional losses",
            },
        ]


@register_plugin("ml_metamorphic_invariants")
@dataclass
class MLMetamorphicInvariantsPlugin:
    """Metamorphic checks for ML training robustness/reproducibility.

    These checks are executed via the skeptic's "__metamorphic__" harness:
    - Seed repeatability (same seed, same results)
    - Scale invariance (scale prices/volumes, metrics should match)
    - Shuffle invariance (reorder samples within split, metrics should match)
    - Val target leakage check (scramble val targets, MSE should worsen)
    """

    name: str = "ml_metamorphic_invariants"

    async def generate_attacks(
        self, program: Program, description: str, language: str
    ) -> list[dict[str, Any]]:
        # Deterministic synthetic dataset with an easy signal (target = last return)
        # to make the leakage check reliable.
        data_seed = 1337
        train_seed = 4242
        num_samples = 64
        seq_len = 60
        train_ratio = 0.8

        base_expr = (
            "lambda np=__import__('numpy'): "
            f"(lambda rng=np.random.RandomState({data_seed}): "
            f"(lambda prices=100 + np.cumsum(rng.randn({num_samples},{seq_len})*0.01, axis=1), "
            f"volumes=np.abs(rng.randn({num_samples},{seq_len}))*1e6: "
            "{"
            "'prices': prices, "
            "'volumes': volumes, "
            "'targets': (prices[:, -1] - prices[:, -2]) / (prices[:, -2] + 1e-8), "
            f"'train_ratio': {train_ratio}, "
            "'device': 'cpu'"
            "})())()"
        )

        def _wrap(test_spec: str) -> str:
            return (
                f"(lambda base=({base_expr})(): "
                f"{{'__metamorphic__': {test_spec.replace('__BASE__', 'base')}}})()"
            )

        attacks = [
            {
                "input": _wrap(
                    "{"
                    "'test': 'seed_repeatability', "
                    f"'seed': {train_seed}, "
                    "'metrics': ['mse','mae','direction_accuracy'], "
                    "'atol': {'mse': 1e-3, 'mae': 1e-3, 'direction_accuracy': 0.05}, "
                    "'rtol': 0.0, "
                    "'base': __BASE__"
                    "}"
                ),
                "attack_type": "ml_metamorphic",
                "rationale": "Same seed should yield repeatable metrics",
            },
            {
                "input": _wrap(
                    "{"
                    "'test': 'scale_invariance', "
                    f"'seed': {train_seed}, "
                    "'metrics': ['mse','mae','direction_accuracy'], "
                    "'scale_factor': 10.0, "
                    "'atol': {'mse': 1e-2, 'mae': 1e-2, 'direction_accuracy': 0.05}, "
                    "'rtol': 0.0, "
                    "'base': __BASE__"
                    "}"
                ),
                "attack_type": "ml_metamorphic",
                "rationale": "Scaling inputs should not materially change returns-based features",
            },
            {
                "input": _wrap(
                    "{"
                    "'test': 'shuffle_invariance', "
                    f"'seed': {train_seed}, "
                    "'metrics': ['mse','mae','direction_accuracy'], "
                    "'atol': {'mse': 1e-2, 'mae': 1e-2, 'direction_accuracy': 0.05}, "
                    "'rtol': 0.0, "
                    "'base': __BASE__"
                    "}"
                ),
                "attack_type": "ml_metamorphic",
                "rationale": "Reordering samples within split should not materially change metrics",
            },
            {
                "input": _wrap(
                    "{"
                    "'test': 'val_target_leakage', "
                    f"'seed': {train_seed}, "
                    "'metric': 'mse', "
                    "'min_delta': 0.05, "
                    "'base': __BASE__"
                    "}"
                ),
                "attack_type": "ml_leakage_check",
                "rationale": "Scrambling validation targets should significantly worsen validation MSE",
            },
        ]

        return attacks
