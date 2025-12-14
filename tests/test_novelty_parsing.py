import unittest

from openevolve.config import Config
from openevolve.database import Program, ProgramDatabase


class MockLLM:
    def __init__(self, response: str):
        self._response = response

    async def generate_with_context(self, system_message: str, messages: list):
        return self._response


class TestNoveltyParsing(unittest.TestCase):
    def test_not_novel_token_is_respected(self):
        config = Config()
        config.database.in_memory = True
        db = ProgramDatabase(config.database, novelty_llm=MockLLM("NOT_NOVEL: trivial refactor"))

        existing = Program(id="p1", code="def f(): return 1", language="python", metrics={})
        proposed = Program(id="p2", code="def f():\n    return 1\n", language="python", metrics={})

        self.assertFalse(db._llm_judge_novelty(proposed, existing))

    def test_novel_token_is_respected(self):
        config = Config()
        config.database.in_memory = True
        db = ProgramDatabase(config.database, novelty_llm=MockLLM("NOVEL: changed algorithm"))

        existing = Program(id="p1", code="def f(): return 1", language="python", metrics={})
        proposed = Program(id="p2", code="def f(): return 2", language="python", metrics={})

        self.assertTrue(db._llm_judge_novelty(proposed, existing))


if __name__ == "__main__":
    unittest.main()
