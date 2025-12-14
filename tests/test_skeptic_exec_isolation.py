import unittest

from openevolve.config import SkepticConfig
from openevolve.discovery.skeptic import AdversarialSkeptic


class TestSkepticExecIsolation(unittest.IsolatedAsyncioTestCase):
    async def test_main_guard_is_not_executed(self):
        config = SkepticConfig(entrypoint="skeptic_entrypoint", attack_timeout=5.0)
        skeptic = AdversarialSkeptic(config=config, llm_ensemble=None)

        code = """
def skeptic_entrypoint(x=None, **kwargs):
    return {"ok": True}

if __name__ == "__main__":
    raise RuntimeError("should not run")
"""

        result = await skeptic._execute_attack(
            code=code,
            attack_input="None",
            attack_type="edge_case",
            language="python",
        )
        self.assertTrue(result.survived)

    async def test_future_imports_work(self):
        config = SkepticConfig(entrypoint="skeptic_entrypoint", attack_timeout=5.0)
        skeptic = AdversarialSkeptic(config=config, llm_ensemble=None)

        code = """
from __future__ import annotations

def skeptic_entrypoint(x=None, **kwargs):
    return {"ok": True}
"""

        result = await skeptic._execute_attack(
            code=code,
            attack_input="None",
            attack_type="edge_case",
            language="python",
        )
        self.assertTrue(result.survived)


if __name__ == "__main__":
    unittest.main()
