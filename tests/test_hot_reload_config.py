import tempfile
import unittest
from pathlib import Path

from openevolve.config import Config
from openevolve.database import ProgramDatabase
from openevolve.process_parallel import ProcessParallelController


class TestHotReloadConfig(unittest.TestCase):
    def test_hot_reload_updates_timeouts(self):
        cfg = Config()
        db = ProgramDatabase(cfg.database)

        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "llm:",
                        "  primary_model: 'test-model'",
                        "evaluator:",
                        "  timeout: 7",
                        "discovery:",
                        "  skeptic:",
                        "    attack_timeout: 12.5",
                        "    num_attack_rounds: 0",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            controller = ProcessParallelController(
                cfg,
                evaluation_file=str(config_path),  # not used by hot reload
                database=db,
                config_path=str(config_path),
                hot_reload=True,
                hot_reload_interval=0.0,
            )

            # Sanity: defaults differ
            self.assertNotEqual(controller.config.evaluator.timeout, 7)
            self.assertNotEqual(controller.config.discovery.skeptic.attack_timeout, 12.5)

            controller._maybe_hot_reload_config()

            self.assertEqual(controller.config.evaluator.timeout, 7)
            self.assertEqual(controller.config.discovery.skeptic.attack_timeout, 12.5)
            self.assertEqual(controller.config.discovery.skeptic.num_attack_rounds, 0)


if __name__ == "__main__":
    unittest.main()
