import json
import tempfile
import unittest

from openevolve.config import load_config
from openevolve.init_project import init_project


class TestInitProject(unittest.TestCase):
    def test_init_project_writes_valid_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            prompt = "Implement a function.\nSecond line should not break YAML."
            result = init_project(
                prompt=prompt,
                init_dir=tmp,
                project_name="my project",
                test_cases=[{"input": 1, "output": 1}],
            )

            self.assertTrue(result.project_dir.exists())
            self.assertTrue(result.config_path.exists())
            self.assertTrue(result.initial_program_path.exists())
            self.assertTrue(result.evaluator_path.exists())
            self.assertTrue(result.test_cases_path.exists())

            cfg = load_config(str(result.config_path))
            self.assertGreater(len(cfg.llm.models), 0)
            self.assertTrue(cfg.discovery.enabled)

            cases = json.loads(result.test_cases_path.read_text(encoding="utf-8"))
            self.assertEqual(len(cases), 1)

    def test_init_project_refuses_overwrite_without_force(self):
        with tempfile.TemporaryDirectory() as tmp:
            prompt = "Do a thing"
            init_project(
                prompt=prompt,
                init_dir=tmp,
                project_name="demo",
                test_cases=[{"input": 1, "output": 1}],
            )
            with self.assertRaises(FileExistsError):
                init_project(
                    prompt=prompt,
                    init_dir=tmp,
                    project_name="demo",
                    test_cases=[{"input": 1, "output": 1}],
                    force=False,
                )


if __name__ == "__main__":
    unittest.main()
