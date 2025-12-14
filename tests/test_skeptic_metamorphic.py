import unittest

from openevolve.config import SkepticConfig
from openevolve.discovery.skeptic import AdversarialSkeptic


class TestSkepticMetamorphic(unittest.IsolatedAsyncioTestCase):
    async def test_seed_repeatability_passes(self):
        config = SkepticConfig(entrypoint="skeptic_entrypoint", attack_timeout=5.0)
        skeptic = AdversarialSkeptic(config=config, llm_ensemble=None)

        code = """
import random
def skeptic_entrypoint(prices, volumes, targets, train_ratio=0.8, device=None):
    return {"mse": random.random()}
"""

        attack_input = (
            "(lambda np=__import__('numpy'): {"
            "'__metamorphic__': {"
            "'test': 'seed_repeatability',"
            "'seed': 123,"
            "'metrics': ['mse'],"
            "'atol': {'mse': 0.0},"
            "'rtol': 0.0,"
            "'base': {"
            "'prices': np.ones((4,2)),"
            "'volumes': np.ones((4,2)),"
            "'targets': np.zeros(4),"
            "'train_ratio': 0.8,"
            "'device': 'cpu'"
            "}"
            "}"
            "})()"
        )

        result = await skeptic._execute_attack(
            code=code,
            attack_input=attack_input,
            attack_type="ml_metamorphic",
            language="python",
        )
        self.assertTrue(result.survived)

    async def test_scale_invariance_can_fail(self):
        config = SkepticConfig(entrypoint="skeptic_entrypoint", attack_timeout=5.0)
        skeptic = AdversarialSkeptic(config=config, llm_ensemble=None)

        code = """
import numpy as np
def skeptic_entrypoint(prices, volumes, targets, train_ratio=0.8, device=None):
    prices = np.asarray(prices)
    return {"mse": float(prices.mean())}
"""

        attack_input = (
            "(lambda np=__import__('numpy'): {"
            "'__metamorphic__': {"
            "'test': 'scale_invariance',"
            "'seed': 1,"
            "'metrics': ['mse'],"
            "'scale_factor': 10.0,"
            "'atol': {'mse': 0.0},"
            "'rtol': 0.0,"
            "'base': {"
            "'prices': np.ones((4,2)),"
            "'volumes': np.ones((4,2)),"
            "'targets': np.zeros(4),"
            "'train_ratio': 0.8,"
            "'device': 'cpu'"
            "}"
            "}"
            "})()"
        )

        result = await skeptic._execute_attack(
            code=code,
            attack_input=attack_input,
            attack_type="ml_metamorphic",
            language="python",
        )
        self.assertFalse(result.survived)

    async def test_val_target_leakage_check_can_fail(self):
        config = SkepticConfig(entrypoint="skeptic_entrypoint", attack_timeout=5.0)
        skeptic = AdversarialSkeptic(config=config, llm_ensemble=None)

        code = """
def skeptic_entrypoint(prices, volumes, targets, train_ratio=0.8, device=None):
    return {"mse": 0.0}
"""

        attack_input = (
            "(lambda np=__import__('numpy'): {"
            "'__metamorphic__': {"
            "'test': 'val_target_leakage',"
            "'seed': 7,"
            "'metric': 'mse',"
            "'min_delta': 0.05,"
            "'base': {"
            "'prices': np.ones((10,2)),"
            "'volumes': np.ones((10,2)),"
            "'targets': np.zeros(10),"
            "'train_ratio': 0.8,"
            "'device': 'cpu'"
            "}"
            "}"
            "})()"
        )

        result = await skeptic._execute_attack(
            code=code,
            attack_input=attack_input,
            attack_type="ml_leakage_check",
            language="python",
        )
        self.assertFalse(result.survived)

    async def test_shuffle_invariance_can_pass(self):
        config = SkepticConfig(entrypoint="skeptic_entrypoint", attack_timeout=5.0)
        skeptic = AdversarialSkeptic(config=config, llm_ensemble=None)

        code = """
import numpy as np
def skeptic_entrypoint(prices, volumes, targets, train_ratio=0.8, device=None):
    targets = np.asarray(targets)
    return {"mse": float(targets.mean())}
"""

        attack_input = (
            "(lambda np=__import__('numpy'): {"
            "'__metamorphic__': {"
            "'test': 'shuffle_invariance',"
            "'seed': 123,"
            "'metrics': ['mse'],"
            "'atol': {'mse': 0.0},"
            "'rtol': 0.0,"
            "'base': {"
            "'prices': np.arange(20).reshape(10,2),"
            "'volumes': np.ones((10,2)),"
            "'targets': np.arange(10),"
            "'train_ratio': 0.8,"
            "'device': 'cpu'"
            "}"
            "}"
            "})()"
        )

        result = await skeptic._execute_attack(
            code=code,
            attack_input=attack_input,
            attack_type="ml_metamorphic",
            language="python",
        )
        self.assertTrue(result.survived)


if __name__ == "__main__":
    unittest.main()
