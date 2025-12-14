import unittest

from openevolve.config import SkepticConfig
from openevolve.discovery.skeptic import AdversarialSkeptic


class TestSkepticNonFiniteOutput(unittest.IsolatedAsyncioTestCase):
    async def test_nan_output_fails_attack(self):
        config = SkepticConfig(entrypoint="skeptic_entrypoint", attack_timeout=5.0)
        skeptic = AdversarialSkeptic(config=config, llm_ensemble=None)

        code = """
def skeptic_entrypoint(x=None, **kwargs):
    return {"loss": float('nan')}
"""

        result = await skeptic._execute_attack(
            code=code,
            attack_input="None",
            attack_type="edge_case",
            language="python",
        )
        self.assertFalse(result.survived)


if __name__ == "__main__":
    unittest.main()
