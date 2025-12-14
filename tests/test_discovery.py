"""
Unit tests for the Discovery Mode modules

Tests the three core components:
1. ProblemEvolver - Problem space evolution
2. AdversarialSkeptic - Falsification-based testing
3. EpistemicArchive - Behavioral diversity tracking
"""

import asyncio
import time
import unittest
from unittest.mock import MagicMock

from openevolve.discovery.engine import (
    DiscoveryConfig,
    DiscoveryEngine,
    DiscoveryEvent,
)
from openevolve.discovery.epistemic_archive import (
    EpistemicArchive,
    Phenotype,
    PhenotypeExtractor,
    SurpriseMetric,
)

# Import discovery modules
from openevolve.discovery.problem_space import (
    ProblemEvolver,
    ProblemEvolverConfig,
    ProblemSpace,
)
from openevolve.discovery.skeptic import (
    AdversarialSkeptic,
    FalsificationResult,
    SkepticConfig,
)


class TestProblemSpace(unittest.TestCase):
    """Tests for ProblemSpace dataclass"""

    def test_problem_space_creation(self):
        """Test basic ProblemSpace creation"""
        problem = ProblemSpace(
            id="test_problem",
            description="Sort a list of numbers",
            constraints=["O(n log n) time complexity"],
            objectives=["correctness", "efficiency"],
        )

        self.assertEqual(problem.id, "test_problem")
        self.assertEqual(problem.generation, 0)
        self.assertEqual(problem.difficulty_level, 1.0)
        self.assertEqual(len(problem.constraints), 1)

    def test_problem_space_to_prompt_context(self):
        """Test formatting problem for prompts"""
        problem = ProblemSpace(
            id="test",
            description="Test problem",
            constraints=["Constraint 1", "Constraint 2"],
            objectives=["Objective A"],
            difficulty_level=2.5,
            generation=3,
        )

        context = problem.to_prompt_context()

        self.assertIn("Test problem", context)
        self.assertIn("Constraint 1", context)
        self.assertIn("Constraint 2", context)
        self.assertIn("Objective A", context)
        self.assertIn("2.5", context)
        self.assertIn("3", context)

    def test_problem_space_serialization(self):
        """Test serialization/deserialization"""
        problem = ProblemSpace(
            id="test",
            description="Test",
            constraints=["c1"],
        )

        data = problem.to_dict()
        restored = ProblemSpace.from_dict(data)

        self.assertEqual(problem.id, restored.id)
        self.assertEqual(problem.description, restored.description)
        self.assertEqual(problem.constraints, restored.constraints)

    def test_is_solved(self):
        """Test solution tracking"""
        problem = ProblemSpace(id="test", description="Test")

        self.assertFalse(problem.is_solved())

        problem.solved_by.append("program_1")
        self.assertTrue(problem.is_solved())


class TestProblemEvolver(unittest.TestCase):
    """Tests for ProblemEvolver"""

    def test_evolver_creation(self):
        """Test ProblemEvolver initialization"""
        config = ProblemEvolverConfig()
        evolver = ProblemEvolver(config)

        self.assertIsNotNone(evolver)
        self.assertEqual(len(evolver.problem_history), 0)

    def test_set_genesis_problem(self):
        """Test setting genesis problem"""
        config = ProblemEvolverConfig()
        evolver = ProblemEvolver(config)

        problem = ProblemSpace(id="genesis", description="Initial problem")
        evolver.set_genesis_problem(problem)

        self.assertEqual(evolver.current_problem, problem)
        self.assertIn("genesis", evolver.problem_history)

    def test_simple_evolve(self):
        """Test simple evolution without LLM"""
        config = ProblemEvolverConfig()
        evolver = ProblemEvolver(config)

        parent = ProblemSpace(
            id="parent",
            description="Sort a list",
            constraints=[],
            difficulty_level=1.0,
            generation=0,
        )
        evolver.set_genesis_problem(parent)

        # Simple evolution (no LLM)
        child = evolver._simple_evolve(parent)

        self.assertNotEqual(child.id, parent.id)
        self.assertEqual(child.parent_id, parent.id)
        self.assertEqual(child.generation, 1)
        self.assertGreater(child.difficulty_level, parent.difficulty_level)
        self.assertGreater(len(child.constraints), len(parent.constraints))

    def test_get_problem_lineage(self):
        """Test lineage tracking"""
        config = ProblemEvolverConfig()
        evolver = ProblemEvolver(config)

        # Create lineage
        p0 = ProblemSpace(id="p0", description="Gen 0")
        evolver.set_genesis_problem(p0)

        p1 = evolver._simple_evolve(p0)
        p2 = evolver._simple_evolve(p1)

        lineage = evolver.get_problem_lineage(p2.id)

        self.assertEqual(len(lineage), 3)
        self.assertEqual(lineage[0].id, "p0")
        self.assertEqual(lineage[2].id, p2.id)


class TestAdversarialSkeptic(unittest.TestCase):
    """Tests for AdversarialSkeptic"""

    def test_skeptic_creation(self):
        """Test AdversarialSkeptic initialization"""
        config = SkepticConfig()
        skeptic = AdversarialSkeptic(config)

        self.assertIsNotNone(skeptic)
        self.assertEqual(len(skeptic.attack_history), 0)

    def test_static_analysis_valid_code(self):
        """Test static analysis on valid code"""
        config = SkepticConfig()
        skeptic = AdversarialSkeptic(config)

        valid_code = """
def solve(data):
    return sorted(data)
"""

        result = skeptic._static_analysis_attack(valid_code, "python")

        self.assertTrue(result.survived)
        self.assertEqual(result.attack_type, "static_analysis")

    def test_static_analysis_syntax_error(self):
        """Test static analysis catches syntax errors"""
        config = SkepticConfig()
        skeptic = AdversarialSkeptic(config)

        invalid_code = """
def solve(data)  # Missing colon
    return data
"""

        result = skeptic._static_analysis_attack(invalid_code, "python")

        self.assertFalse(result.survived)
        self.assertIn("Syntax error", result.error_message)

    def test_static_analysis_security_issues(self):
        """Test static analysis catches security issues"""
        config = SkepticConfig()
        skeptic = AdversarialSkeptic(config)

        risky_code = """
def solve(data):
    return eval(data)  # Security risk!
"""

        result = skeptic._static_analysis_attack(risky_code, "python")

        self.assertFalse(result.survived)
        self.assertIn("eval", result.error_message)

    def test_generate_simple_attacks(self):
        """Test simple attack generation"""
        config = SkepticConfig()
        skeptic = AdversarialSkeptic(config)

        attacks = skeptic._generate_simple_attacks("edge_case")

        self.assertGreater(len(attacks), 0)
        self.assertTrue(all("input" in a for a in attacks))
        self.assertTrue(all("attack_type" in a for a in attacks))

    def test_falsification_result_serialization(self):
        """Test FalsificationResult can be serialized"""
        result = FalsificationResult(
            survived=True,
            attack_type="edge_case",
            attack_input="[]",
            execution_time=0.1,
            confidence=0.9,
        )

        data = result.to_dict()

        self.assertEqual(data["survived"], True)
        self.assertEqual(data["attack_type"], "edge_case")


class TestPhenotypeExtractor(unittest.TestCase):
    """Tests for PhenotypeExtractor"""

    def test_extractor_creation(self):
        """Test PhenotypeExtractor initialization"""
        extractor = PhenotypeExtractor(num_bins=10)
        self.assertIsNotNone(extractor)

    def test_extract_simple_code(self):
        """Test extracting phenotype from simple code"""
        extractor = PhenotypeExtractor(num_bins=10)

        code = """
def solve(data):
    result = []
    for item in data:
        result.append(item)
    return result
"""

        phenotype = extractor.extract(code)

        self.assertIsInstance(phenotype, Phenotype)
        self.assertTrue(phenotype.uses_iteration)
        self.assertFalse(phenotype.uses_recursion)

    def test_extract_recursive_code(self):
        """Test detecting recursion"""
        extractor = PhenotypeExtractor(num_bins=10)

        code = """
def solve(data):
    if len(data) <= 1:
        return data
    return solve(data[:-1]) + [data[-1]]
"""

        phenotype = extractor.extract(code)

        self.assertTrue(phenotype.uses_recursion)

    def test_extract_numpy_code(self):
        """Test detecting vectorization"""
        extractor = PhenotypeExtractor(num_bins=10)

        code = """
import numpy as np

def solve(data):
    arr = np.array(data)
    return np.sort(arr).tolist()
"""

        phenotype = extractor.extract(code)

        self.assertTrue(phenotype.uses_vectorization)

    def test_approach_signature(self):
        """Test approach signature generation"""
        extractor = PhenotypeExtractor(num_bins=10)

        code1 = """
def solve(data):
    for x in data:
        pass
"""

        code2 = """
class Solver:
    def solve(self, data):
        for x in data:
            pass
"""

        p1 = extractor.extract(code1)
        p2 = extractor.extract(code2)

        # Different structural approaches should have different signatures
        self.assertNotEqual(p1.approach_signature, p2.approach_signature)


class TestSurpriseMetric(unittest.TestCase):
    """Tests for SurpriseMetric"""

    def test_surprise_calculation(self):
        """Test surprise score calculation"""
        surprise = SurpriseMetric(
            predicted_fitness=0.5,
            actual_fitness=0.9,
        )

        score = surprise.calculate()

        self.assertAlmostEqual(score, 0.4)
        self.assertTrue(surprise.is_positive_surprise)

    def test_negative_surprise(self):
        """Test negative surprise (worse than expected)"""
        surprise = SurpriseMetric(
            predicted_fitness=0.8,
            actual_fitness=0.3,
        )

        score = surprise.calculate()

        self.assertAlmostEqual(score, 0.5)
        self.assertFalse(surprise.is_positive_surprise)


class TestEpistemicArchive(unittest.TestCase):
    """Tests for EpistemicArchive"""

    def setUp(self):
        """Set up mock database"""
        self.mock_db = MagicMock()
        self.mock_db.programs = {}
        self.mock_db.island_feature_maps = [{}]
        self.mock_db.config.feature_dimensions = ["complexity", "diversity"]
        self.mock_db._calculate_feature_coords = MagicMock(return_value=[0, 0])
        self.mock_db._feature_coords_to_key = MagicMock(return_value="0-0")

    def test_archive_creation(self):
        """Test EpistemicArchive initialization"""
        archive = EpistemicArchive(self.mock_db)
        self.assertIsNotNone(archive)

    def test_predict_fitness_unknown_approach(self):
        """Test fitness prediction for unknown approach"""
        archive = EpistemicArchive(self.mock_db)

        code = "def solve(data): return data"
        prediction = archive.predict_fitness(code)

        # Should return a reasonable default
        self.assertGreaterEqual(prediction, 0.0)
        self.assertLessEqual(prediction, 1.0)

    def test_approach_library_tracking(self):
        """Test that approach library tracks signatures"""
        archive = EpistemicArchive(self.mock_db)

        # Create mock program
        mock_program = MagicMock()
        mock_program.code = "def solve(data): return sorted(data)"
        mock_program.metrics = {"combined_score": 0.8}
        mock_program.metadata = {}
        mock_program.id = "test_prog"

        # Add to archive
        archive.add_with_phenotype(mock_program)

        # Check approach was tracked
        self.assertGreater(len(archive.approach_library), 0)

    def test_surprise_tracking(self):
        """Test surprise tracking"""
        archive = EpistemicArchive(self.mock_db)

        mock_program = MagicMock()
        mock_program.code = "def solve(data): return data"
        mock_program.metrics = {"combined_score": 0.9}
        mock_program.metadata = {}
        mock_program.id = "test_prog"

        # Add with prediction
        was_novel, surprise = archive.add_with_phenotype(
            mock_program,
            predicted_fitness=0.3,
        )

        self.assertIsNotNone(surprise)
        self.assertAlmostEqual(surprise.surprise_score, 0.6)
        self.assertEqual(len(archive.surprise_history), 1)


class TestDiscoveryEngine(unittest.TestCase):
    """Tests for DiscoveryEngine"""

    def setUp(self):
        """Set up mock OpenEvolve"""
        self.mock_openevolve = MagicMock()
        self.mock_openevolve.llm_ensemble = MagicMock()
        self.mock_openevolve.database = MagicMock()
        self.mock_openevolve.database.programs = {}
        self.mock_openevolve.database.island_feature_maps = [{}]
        self.mock_openevolve.database.config.feature_dimensions = ["complexity"]
        self.mock_openevolve.evaluation_file = "test_eval.py"

    def test_engine_creation(self):
        """Test DiscoveryEngine initialization"""
        config = DiscoveryConfig()
        engine = DiscoveryEngine(config, self.mock_openevolve)

        self.assertIsNotNone(engine)
        self.assertIsNotNone(engine.problem_evolver)
        self.assertIsNotNone(engine.archive)

    def test_set_genesis_problem(self):
        """Test setting genesis problem"""
        config = DiscoveryConfig()
        engine = DiscoveryEngine(config, self.mock_openevolve)

        problem = engine.set_genesis_problem("Sort a list of numbers")

        self.assertIsNotNone(engine.current_problem)
        self.assertEqual(engine.current_problem.description, "Sort a list of numbers")
        self.assertEqual(len(engine.discovery_events), 1)
        self.assertEqual(engine.discovery_events[0].event_type, "genesis")

    def test_get_current_problem_context(self):
        """Test getting problem context for prompts"""
        config = DiscoveryConfig()
        engine = DiscoveryEngine(config, self.mock_openevolve)

        engine.set_genesis_problem("Test problem")
        context = engine.get_current_problem_context()

        self.assertIn("Test problem", context)

    def test_statistics_tracking(self):
        """Test statistics are tracked"""
        config = DiscoveryConfig()
        engine = DiscoveryEngine(config, self.mock_openevolve)

        engine.set_genesis_problem("Test")

        stats = engine.get_statistics()

        self.assertIn("total_iterations", stats)
        self.assertIn("current_problem", stats)
        self.assertEqual(stats["current_problem"]["generation"], 0)


class TestMinimalCriterionScreening(unittest.TestCase):
    """Tests for minimal-criterion transfer screening in coevolution."""

    def _run(self, coro):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    def _make_engine(self, eval_returns, **cfg_kwargs):
        mock_openevolve = MagicMock()
        mock_openevolve.llm_ensemble = None

        mock_db = MagicMock()
        mock_db.config.feature_dimensions = []

        # Two dummy programs in the archive
        p1 = MagicMock()
        p1.id = "prog1"
        p1.code = "def solve(x): return x"
        p2 = MagicMock()
        p2.id = "prog2"
        p2.code = "def solve(x): return x"
        mock_db.get_top_programs.return_value = [p1, p2]
        mock_openevolve.database = mock_db

        mock_eval = MagicMock()

        async def _eval(*a, **k):
            return eval_returns.pop(0)

        mock_eval.evaluate_program = _eval
        mock_openevolve.evaluator = mock_eval

        config = DiscoveryConfig(
            coevolution_enabled=True,
            problem_evolution_enabled=True,
            **cfg_kwargs,
        )
        engine = DiscoveryEngine(config, mock_openevolve)
        engine.set_genesis_problem("Genesis")
        return engine

    def test_candidate_too_easy_rejected(self):
        engine = self._make_engine(
            eval_returns=[
                {"combined_score_raw": 0.85},
                {"combined_score_raw": 0.2},
            ],
            solution_threshold=0.8,
            min_transfer_fitness=0.3,
        )
        cand = ProblemSpace(id="cand", description="Harder", parent_id="genesis")
        passed = self._run(engine._passes_minimal_criterion(cand))
        self.assertFalse(passed)

    def test_candidate_too_hard_rejected(self):
        engine = self._make_engine(
            eval_returns=[
                {"combined_score_raw": 0.1},
                {"combined_score_raw": 0.2},
            ],
            solution_threshold=0.8,
            min_transfer_fitness=0.3,
        )
        cand = ProblemSpace(id="cand", description="Harder", parent_id="genesis")
        passed = self._run(engine._passes_minimal_criterion(cand))
        self.assertFalse(passed)

    def test_candidate_in_range_admitted(self):
        engine = self._make_engine(
            eval_returns=[
                {"combined_score_raw": 0.5},
                {"combined_score_raw": 0.4},
            ],
            solution_threshold=0.8,
            min_transfer_fitness=0.3,
        )
        cand = ProblemSpace(id="cand", description="Harder", parent_id="genesis")
        passed = self._run(engine._passes_minimal_criterion(cand))
        self.assertTrue(passed)


class TestDiscoveryEvent(unittest.TestCase):
    """Tests for DiscoveryEvent"""

    def test_event_creation(self):
        """Test DiscoveryEvent creation"""
        event = DiscoveryEvent(
            timestamp=time.time(),
            event_type="solution",
            problem_id="p1",
            program_id="prog1",
            details={"fitness": 0.9},
        )

        self.assertEqual(event.event_type, "solution")
        self.assertEqual(event.problem_id, "p1")


class TestIntegration(unittest.TestCase):
    """Integration tests for Discovery Mode"""

    def test_full_flow_without_llm(self):
        """Test complete flow without LLM calls"""
        # Create mock OpenEvolve
        mock_openevolve = MagicMock()
        mock_openevolve.llm_ensemble = None  # No LLM
        mock_openevolve.database = MagicMock()
        mock_openevolve.database.programs = {}
        mock_openevolve.database.island_feature_maps = [{}]
        mock_openevolve.database.config.feature_dimensions = ["complexity"]
        mock_openevolve.evaluation_file = "test.py"

        # Create engine without skeptic (to avoid execution)
        config = DiscoveryConfig(
            skeptic_enabled=False,
            problem_evolution_enabled=True,
            evolve_problem_after_solutions=2,
        )
        engine = DiscoveryEngine(config, mock_openevolve)
        engine.set_genesis_problem("Sort numbers")

        # Simulate solutions
        for i in range(3):
            mock_program = MagicMock()
            mock_program.id = f"prog_{i}"
            mock_program.code = f"def solve(data): return sorted(data)  # v{i}"
            mock_program.metrics = {"combined_score": 0.85}
            mock_program.metadata = {}
            mock_program.language = "python"

            # Process synchronously (mock the async)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                is_valid, metadata = loop.run_until_complete(engine.process_program(mock_program))
            finally:
                loop.close()

            # Should be valid (no skeptic)
            self.assertTrue(metadata.get("is_solution", False) or not is_valid)


if __name__ == "__main__":
    unittest.main()
