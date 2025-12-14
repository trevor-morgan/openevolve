import unittest
from unittest.mock import AsyncMock, MagicMock

from openevolve.config import SkepticConfig
from openevolve.discovery.skeptic import AdversarialSkeptic, FalsificationResult


class TestSkepticPlugins(unittest.IsolatedAsyncioTestCase):
    async def test_plugin_attacks_are_executed(self):
        config = SkepticConfig(num_attack_rounds=0, plugins=["ml_data_perturbation"])
        skeptic = AdversarialSkeptic(config=config, llm_ensemble=None)

        skeptic._execute_attack = AsyncMock(
            return_value=FalsificationResult(
                survived=True,
                attack_type="plugin",
                confidence=0.5,
            )
        )

        program = MagicMock()
        program.id = "prog"
        program.code = "def skeptic_entrypoint(**kwargs): return {}"
        program.language = "python"
        program.metadata = {}
        program.metrics = {}

        survived, _results = await skeptic.falsify(program, description="", language="python")
        self.assertTrue(survived)
        # Built-in plugin emits 3 attacks.
        self.assertGreaterEqual(skeptic._execute_attack.await_count, 3)

    async def test_metamorphic_plugin_attacks_are_executed(self):
        config = SkepticConfig(num_attack_rounds=0, plugins=["ml_metamorphic_invariants"])
        skeptic = AdversarialSkeptic(config=config, llm_ensemble=None)

        skeptic._execute_attack = AsyncMock(
            return_value=FalsificationResult(
                survived=True,
                attack_type="plugin",
                confidence=0.5,
            )
        )

        program = MagicMock()
        program.id = "prog"
        program.code = "def skeptic_entrypoint(**kwargs): return {}"
        program.language = "python"
        program.metadata = {}
        program.metrics = {}

        survived, _results = await skeptic.falsify(program, description="", language="python")
        self.assertTrue(survived)
        # Metamorphic plugin emits 4 attacks.
        self.assertGreaterEqual(skeptic._execute_attack.await_count, 4)


if __name__ == "__main__":
    unittest.main()
