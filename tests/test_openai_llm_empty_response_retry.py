import unittest
from unittest.mock import Mock, patch

from openevolve.llm.openai import OpenAILLM


class TestOpenAILLMEmptyResponseRetry(unittest.IsolatedAsyncioTestCase):
    def _model_cfg(self) -> Mock:
        model_cfg = Mock()
        model_cfg.name = "gpt-4"
        model_cfg.system_message = "system"
        model_cfg.temperature = 0.0
        model_cfg.top_p = None
        model_cfg.max_tokens = 128
        model_cfg.timeout = 30
        model_cfg.retries = 1
        model_cfg.retry_delay = 0
        model_cfg.api_base = "http://localhost:8317/v1"
        model_cfg.api_key = "test-key"
        model_cfg.random_seed = None
        model_cfg.reasoning_effort = None
        return model_cfg

    async def test_none_content_is_retried(self):
        model_cfg = self._model_cfg()
        with patch("openai.OpenAI"):
            llm = OpenAILLM(model_cfg)

        calls = 0

        async def fake_call_api(_params):
            nonlocal calls
            calls += 1
            if calls == 1:
                return None
            return "ok"

        llm._call_api = fake_call_api  # type: ignore[method-assign]

        result = await llm.generate_with_context(
            system_message="sys", messages=[{"role": "user", "content": "hi"}]
        )
        self.assertEqual(result, "ok")
        self.assertEqual(calls, 2)

    async def test_blank_content_is_retried(self):
        model_cfg = self._model_cfg()
        with patch("openai.OpenAI"):
            llm = OpenAILLM(model_cfg)

        calls = 0

        async def fake_call_api(_params):
            nonlocal calls
            calls += 1
            if calls == 1:
                return "   "
            return "ok"

        llm._call_api = fake_call_api  # type: ignore[method-assign]

        result = await llm.generate_with_context(
            system_message="sys", messages=[{"role": "user", "content": "hi"}]
        )
        self.assertEqual(result, "ok")
        self.assertEqual(calls, 2)

    async def test_empty_after_retries_raises(self):
        model_cfg = self._model_cfg()
        with patch("openai.OpenAI"):
            llm = OpenAILLM(model_cfg)

        async def fake_call_api(_params):
            return ""

        llm._call_api = fake_call_api  # type: ignore[method-assign]

        with self.assertRaises(ValueError):
            await llm.generate_with_context(
                system_message="sys", messages=[{"role": "user", "content": "hi"}]
            )


if __name__ == "__main__":
    unittest.main()
