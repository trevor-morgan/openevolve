import unittest

from openevolve.config import Config, DatabaseConfig
from openevolve.database import Program, ProgramDatabase
from openevolve.discovery.epistemic_archive import EpistemicArchive
from openevolve.process_parallel import ProcessParallelController


class TestPhenotypeHooks(unittest.TestCase):
    def test_custom_phenotype_extractor_and_mirroring(self):
        db = ProgramDatabase(DatabaseConfig(num_islands=1, feature_dimensions=["complexity"]))

        def custom_extractor(code, metrics, artifacts=None):
            return {"robustness": 3}

        archive = EpistemicArchive(
            database=db,
            phenotype_dimensions=["complexity"],
            custom_phenotype_extractor=custom_extractor,
            mirror_dimensions=["robustness"],
        )

        program = Program(
            id="p1",
            code="def solve(x): return x",
            metrics={"combined_score": 0.5},
            metadata={},
        )

        archive.add_with_phenotype(program)

        self.assertIn("robustness", program.metrics)
        self.assertEqual(program.metrics["robustness"], 3.0)
        self.assertIn("phenotype", program.metadata)
        self.assertIn("robustness", program.metadata["phenotype"])


class TestBudgetedCascade(unittest.TestCase):
    def test_budgeted_stage_selection_by_parent_fitness(self):
        cfg = Config()
        cfg.database = DatabaseConfig(num_islands=1, feature_dimensions=["complexity"])
        cfg.evaluator.budgeted_cascade_enabled = True
        cfg.evaluator.budget_stage3_parent_threshold = 0.6
        cfg.evaluator.budget_max_stage_low = 2
        cfg.evaluator.budget_max_stage_high = 3

        db = ProgramDatabase(cfg.database)
        parent = Program(
            id="parent",
            code="def solve(x): return x",
            metrics={"combined_score": 0.7},
            metadata={"island": 0},
        )
        db.add(parent, iteration=0, target_island=0)

        controller = ProcessParallelController(
            cfg,
            evaluation_file="dummy_eval.py",
            database=db,
            evolution_tracer=None,
            prompt_sampler=None,
            discovery_engine=None,
        )

        snap = controller._create_iteration_snapshot(parent, inspirations=[], island_idx=0)
        self.assertEqual(snap["max_evaluation_stage"], 3)


if __name__ == "__main__":
    unittest.main()
