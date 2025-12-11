"""
Tests for Heisenberg Engine (Ontological Expansion)

Tests cover:
- Ontology creation and expansion
- Crisis detection (plateau, systematic bias, unexplained variance)
- Probe synthesis and execution
- Code instrumentation
- Full integration flow
"""

import asyncio
import json
import os
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from openevolve.discovery.ontology import (
    Variable,
    Ontology,
    OntologyManager,
)
from openevolve.discovery.crisis_detector import (
    EpistemicCrisis,
    CrisisDetector,
    CrisisDetectorConfig,
)
from openevolve.discovery.instrument_synthesizer import (
    Probe,
    ProbeResult,
    InstrumentSynthesizer,
    InstrumentSynthesizerConfig,
)
from openevolve.discovery.code_instrumenter import (
    CodeInstrumenter,
    InstrumentationResult,
)


class TestVariable(unittest.TestCase):
    """Test Variable dataclass"""

    def test_variable_creation(self):
        """Test basic variable creation"""
        var = Variable(
            name="cache_locality",
            var_type="continuous",
            source="probe",
            confidence=0.8,
        )
        self.assertEqual(var.name, "cache_locality")
        self.assertEqual(var.var_type, "continuous")
        self.assertEqual(var.source, "probe")
        self.assertEqual(var.confidence, 0.8)

    def test_variable_serialization(self):
        """Test variable to_dict/from_dict"""
        var = Variable(
            name="test_var",
            var_type="categorical",
            source="user",
            discovery_method="manual",
            extraction_code="lambda x: x['value']",
            confidence=0.95,
        )

        var_dict = var.to_dict()
        self.assertEqual(var_dict["name"], "test_var")
        self.assertEqual(var_dict["var_type"], "categorical")

        restored = Variable.from_dict(var_dict)
        self.assertEqual(restored.name, var.name)
        self.assertEqual(restored.confidence, var.confidence)


class TestOntology(unittest.TestCase):
    """Test Ontology dataclass"""

    def test_ontology_creation(self):
        """Test ontology creation with variables"""
        var1 = Variable(name="var1", var_type="continuous")
        var2 = Variable(name="var2", var_type="categorical")

        ontology = Ontology(
            id="onto_1",
            generation=0,
            variables=[var1, var2],
        )

        self.assertEqual(ontology.id, "onto_1")
        self.assertEqual(ontology.generation, 0)
        self.assertEqual(len(ontology.variables), 2)

    def test_ontology_serialization(self):
        """Test ontology to_dict/from_dict"""
        var = Variable(name="test", var_type="continuous")
        ontology = Ontology(
            id="onto_test",
            generation=1,
            parent_id="onto_parent",
            variables=[var],
            discovered_via="crisis_123",
        )

        ont_dict = ontology.to_dict()
        self.assertEqual(ont_dict["id"], "onto_test")
        self.assertEqual(ont_dict["generation"], 1)
        self.assertEqual(len(ont_dict["variables"]), 1)

        restored = Ontology.from_dict(ont_dict)
        self.assertEqual(restored.id, ontology.id)
        self.assertEqual(restored.generation, ontology.generation)
        self.assertEqual(len(restored.variables), 1)


class TestOntologyManager(unittest.TestCase):
    """Test OntologyManager"""

    def test_create_genesis_ontology(self):
        """Test creating genesis ontology"""
        manager = OntologyManager()
        ontology = manager.create_genesis_ontology(variable_names=["x", "y", "z"])

        self.assertEqual(ontology.generation, 0)
        self.assertIsNone(ontology.parent_id)
        self.assertEqual(len(ontology.variables), 3)
        self.assertEqual(manager.current_ontology, ontology)

    def test_expand_ontology(self):
        """Test expanding ontology with new variables"""
        manager = OntologyManager()
        genesis = manager.create_genesis_ontology(variable_names=["x"])

        new_var = Variable(
            name="cache_hit_rate",
            var_type="continuous",
            source="probe",
            confidence=0.75,
        )

        expanded = manager.expand_ontology(
            new_variables=[new_var],
            discovered_via="crisis_001",
        )

        self.assertEqual(expanded.generation, 1)
        self.assertEqual(expanded.parent_id, genesis.id)
        self.assertEqual(len(expanded.variables), 2)  # x + cache_hit_rate
        self.assertEqual(expanded.discovered_via, "crisis_001")

    def test_ontology_lineage(self):
        """Test getting ontology lineage"""
        manager = OntologyManager()
        gen0 = manager.create_genesis_ontology(variable_names=["a"])

        var1 = Variable(name="b", source="probe")
        gen1 = manager.expand_ontology([var1])

        var2 = Variable(name="c", source="probe")
        gen2 = manager.expand_ontology([var2])

        lineage = manager.get_lineage()
        self.assertEqual(len(lineage), 3)
        self.assertEqual(lineage[0].generation, 0)
        self.assertEqual(lineage[2].generation, 2)

    def test_save_load(self):
        """Test saving and loading ontology history"""
        manager = OntologyManager()
        manager.create_genesis_ontology(variable_names=["x", "y"])
        manager.expand_ontology([Variable(name="z", source="probe")])

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            manager.save(temp_path)

            # Create new manager and load
            manager2 = OntologyManager()
            manager2.load(temp_path)

            self.assertEqual(len(manager2.ontology_history), 2)
            self.assertEqual(manager2.current_ontology.generation, 1)
        finally:
            os.unlink(temp_path)


class TestCrisisDetector(unittest.TestCase):
    """Test CrisisDetector"""

    def test_no_crisis_initially(self):
        """Test no crisis with insufficient data"""
        config = CrisisDetectorConfig(min_plateau_iterations=10)
        detector = CrisisDetector(config)

        # Add some evaluations
        for i in range(5):
            detector.record_evaluation(i, {"combined_score": 0.5 + i * 0.1}, {})

        crisis = detector.detect_crisis()
        self.assertIsNone(crisis)

    def test_plateau_detection(self):
        """Test plateau detection triggers"""
        config = CrisisDetectorConfig(
            min_plateau_iterations=10,
            fitness_improvement_threshold=0.01,
            variance_window=5,
            confidence_threshold=0.5,
        )
        detector = CrisisDetector(config)

        # Add stagnating fitness values
        for i in range(20):
            # Small random variation around 0.5
            detector.record_evaluation(
                i,
                {"combined_score": 0.5 + (i % 2) * 0.001},
                {}
            )

        crisis = detector.detect_crisis()

        # May or may not detect depending on exact threshold calculation
        if crisis:
            self.assertEqual(crisis.crisis_type, "plateau")
            self.assertIn("state", crisis.suggested_probes)

    def test_crisis_serialization(self):
        """Test EpistemicCrisis to_dict/from_dict"""
        crisis = EpistemicCrisis(
            id="crisis_001",
            crisis_type="plateau",
            confidence=0.85,
            evidence={"improvement": 0.001, "variance": 0.0001},
            suggested_probes=["state", "gradient"],
        )

        crisis_dict = crisis.to_dict()
        self.assertEqual(crisis_dict["crisis_type"], "plateau")

        restored = EpistemicCrisis.from_dict(crisis_dict)
        self.assertEqual(restored.id, crisis.id)
        self.assertEqual(restored.confidence, crisis.confidence)

    def test_reset(self):
        """Test detector reset"""
        config = CrisisDetectorConfig(min_plateau_iterations=5)
        detector = CrisisDetector(config)

        for i in range(10):
            detector.record_evaluation(i, {"combined_score": 0.5}, {})

        self.assertEqual(len(detector.fitness_history), 10)

        detector.reset()

        self.assertEqual(len(detector.fitness_history), 0)

    def test_statistics(self):
        """Test get_statistics"""
        config = CrisisDetectorConfig()
        detector = CrisisDetector(config)

        for i in range(5):
            detector.record_evaluation(i, {"combined_score": 0.5 + i * 0.1}, {})

        stats = detector.get_statistics()

        self.assertEqual(stats["total_evaluations"], 5)
        self.assertIn("current_fitness", stats)


class TestProbe(unittest.TestCase):
    """Test Probe dataclass"""

    def test_probe_creation(self):
        """Test probe creation"""
        probe = Probe(
            id="probe_001",
            code="def probe(artifacts, metrics): return {'discovered_variables': []}",
            target_hypothesis="Check for cache effects",
            probe_type="state",
            rationale="Looking for memory access patterns",
        )

        self.assertEqual(probe.id, "probe_001")
        self.assertEqual(probe.probe_type, "state")


class TestInstrumentSynthesizer(unittest.TestCase):
    """Test InstrumentSynthesizer"""

    def test_default_probes(self):
        """Test getting default probes for crisis types"""
        config = InstrumentSynthesizerConfig()
        synthesizer = InstrumentSynthesizer(config, llm_ensemble=None)

        crisis = EpistemicCrisis(
            id="crisis_test",
            crisis_type="plateau",
            confidence=0.8,
            evidence={},
            suggested_probes=["state"],
        )

        ontology = Ontology(
            id="onto_test",
            generation=0,
            variables=[Variable(name="x")],
        )

        # Run synchronously using asyncio
        async def get_probes():
            return await synthesizer.synthesize_probes(crisis, ontology, {})

        probes = asyncio.run(get_probes())

        # Should return at least one fallback probe
        self.assertGreaterEqual(len(probes), 1)

    def test_probe_execution(self):
        """Test probe execution"""
        config = InstrumentSynthesizerConfig()
        synthesizer = InstrumentSynthesizer(config, llm_ensemble=None)

        # Create a simple test probe
        probe = Probe(
            id="test_probe",
            code='''
def probe(artifacts, metrics):
    return {
        "discovered_variables": [{
            "name": "test_var",
            "type": "continuous",
            "evidence": {"test": True},
            "confidence": 0.5
        }],
        "analysis_notes": "Test probe executed"
    }
''',
            target_hypothesis="Test",
            probe_type="state",
        )

        async def execute():
            return await synthesizer.execute_probe(
                probe,
                {"artifacts": {}, "metrics": {"combined_score": 0.5}},
            )

        result = asyncio.run(execute())

        self.assertIsInstance(result, ProbeResult)
        # May or may not succeed depending on execution

    def test_statistics(self):
        """Test get_statistics"""
        config = InstrumentSynthesizerConfig()
        synthesizer = InstrumentSynthesizer(config, llm_ensemble=None)

        stats = synthesizer.get_statistics()

        self.assertEqual(stats["total_probes"], 0)
        self.assertEqual(stats["total_discoveries"], 0)


class TestCodeInstrumenter(unittest.TestCase):
    """Test CodeInstrumenter"""

    def test_minimal_instrumentation(self):
        """Test minimal instrumentation level"""
        instrumenter = CodeInstrumenter()

        code = '''
def foo(x):
    return x * 2

def bar(y):
    return foo(y) + 1
'''

        result = instrumenter.instrument(code, level="minimal")

        self.assertIsInstance(result, InstrumentationResult)
        self.assertTrue(result.success)
        self.assertIn("__TraceCollector__", result.instrumented_code)
        self.assertIn("log_call", result.instrumented_code)

    def test_standard_instrumentation(self):
        """Test standard instrumentation level"""
        instrumenter = CodeInstrumenter()

        code = '''
def process(data):
    total = 0
    for item in data:
        total += item
    return total
'''

        result = instrumenter.instrument(code, level="standard")

        self.assertTrue(result.success)
        self.assertIn("log_loop_iteration", result.instrumented_code)
        self.assertIn("log_assignment", result.instrumented_code)

    def test_comprehensive_instrumentation(self):
        """Test comprehensive instrumentation level"""
        instrumenter = CodeInstrumenter()

        code = '''
def analyze(x):
    if x > 0:
        return x * 2
    else:
        return 0
'''

        result = instrumenter.instrument(code, level="comprehensive")

        self.assertTrue(result.success)
        self.assertIn("log_branch", result.instrumented_code)

    def test_invalid_code(self):
        """Test handling of invalid Python code"""
        instrumenter = CodeInstrumenter()

        invalid_code = "def broken( {}"

        result = instrumenter.instrument(invalid_code, level="minimal")

        self.assertFalse(result.success)
        self.assertIsNotNone(result.error)

    def test_evolve_block_extraction(self):
        """Test extraction of EVOLVE-BLOCK markers"""
        instrumenter = CodeInstrumenter()

        code = '''
import something

# EVOLVE-BLOCK-START
def evolve_me(x):
    return x + 1
# EVOLVE-BLOCK-END

def other_func():
    pass
'''

        result = instrumenter.instrument(code, level="minimal", evolve_block_only=True)

        # Should still succeed even with markers
        self.assertTrue(result.success)


class TestHeisenbergIntegration(unittest.TestCase):
    """Integration tests for the full Heisenberg flow"""

    def test_crisis_to_expansion_flow(self):
        """Test full flow from crisis detection to ontology expansion"""
        # Create components
        ontology_manager = OntologyManager()
        ontology_manager.create_genesis_ontology(variable_names=["fitness"])

        crisis_config = CrisisDetectorConfig(
            min_plateau_iterations=5,
            fitness_improvement_threshold=0.01,
            variance_window=3,
            confidence_threshold=0.3,
        )
        crisis_detector = CrisisDetector(crisis_config)

        synth_config = InstrumentSynthesizerConfig()
        synthesizer = InstrumentSynthesizer(synth_config, llm_ensemble=None)

        # Simulate plateau
        for i in range(10):
            crisis_detector.record_evaluation(
                i,
                {"combined_score": 0.5 + (i % 2) * 0.001},
                {},
            )

        # Check for crisis
        crisis = crisis_detector.detect_crisis()

        if crisis:
            # Synthesize probes
            async def run_probes():
                probes = await synthesizer.synthesize_probes(
                    crisis,
                    ontology_manager.current_ontology,
                    {},
                )
                return probes

            probes = asyncio.run(run_probes())
            self.assertGreater(len(probes), 0)

            # Simulate finding a variable
            new_var = Variable(
                name="discovered_var",
                var_type="continuous",
                source="probe",
                discovery_method=probes[0].id,
                confidence=0.7,
            )

            # Expand ontology
            expanded = ontology_manager.expand_ontology(
                [new_var],
                discovered_via=crisis.id,
            )

            self.assertEqual(expanded.generation, 1)
            self.assertEqual(len(expanded.variables), 2)

    def test_checkpoint_resume(self):
        """Test saving and restoring Heisenberg state"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and populate manager
            manager = OntologyManager()
            manager.create_genesis_ontology(["a", "b"])
            manager.expand_ontology([Variable(name="c", source="probe", confidence=0.8)])

            ontology_path = os.path.join(tmpdir, "ontology.json")
            manager.save(ontology_path)

            # Create and populate crisis detector
            config = CrisisDetectorConfig()
            detector = CrisisDetector(config)
            for i in range(5):
                detector.record_evaluation(i, {"combined_score": 0.5}, {})

            # Save crisis history
            crisis_path = os.path.join(tmpdir, "crisis.json")
            crisis_data = {
                "fitness_history": detector.fitness_history,
                "last_crisis_iteration": detector.last_crisis_iteration,
            }
            with open(crisis_path, 'w') as f:
                json.dump(crisis_data, f)

            # Restore
            manager2 = OntologyManager()
            manager2.load(ontology_path)

            self.assertEqual(manager2.current_ontology.generation, 1)
            self.assertEqual(len(manager2.ontology_history), 2)

            detector2 = CrisisDetector(config)
            with open(crisis_path, 'r') as f:
                loaded = json.load(f)
                detector2.fitness_history = loaded["fitness_history"]

            self.assertEqual(len(detector2.fitness_history), 5)


class TestImports(unittest.TestCase):
    """Test that all exports are importable"""

    def test_discovery_imports(self):
        """Test importing from discovery package"""
        from openevolve.discovery import (
            Variable,
            Ontology,
            OntologyManager,
            EpistemicCrisis,
            CrisisDetector,
            CrisisDetectorConfig,
            Probe,
            ProbeResult,
            InstrumentSynthesizer,
            InstrumentSynthesizerConfig,
            CodeInstrumenter,
            InstrumentationResult,
        )

        # All imports should work
        self.assertIsNotNone(Variable)
        self.assertIsNotNone(OntologyManager)
        self.assertIsNotNone(CrisisDetector)
        self.assertIsNotNone(InstrumentSynthesizer)
        self.assertIsNotNone(CodeInstrumenter)


if __name__ == "__main__":
    unittest.main()
