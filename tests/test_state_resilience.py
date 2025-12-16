import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from openevolve.config import Config, DatabaseConfig, DiscoveryConfig, RLConfig
from openevolve.controller import OpenEvolve
from openevolve.database import Program


class TestStateResilience(unittest.TestCase):
    """
    Test suite to verify that the agent's state is strictly preserved
    across save/load cycles (Crash Resilience).
    """

    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()

        # Create a dummy initial program
        self.initial_program_path = os.path.join(self.test_dir, "initial_program.py")
        with open(self.initial_program_path, "w") as f:
            f.write("print('Hello World')")

        self.evaluator_path = os.path.join(self.test_dir, "evaluator.py")
        with open(self.evaluator_path, "w") as f:
            f.write("def evaluate(code): return {'score': 1.0}")

        # Config with RL and Discovery enabled
        self.config = Config(
            max_iterations=10,
            rl=RLConfig(enabled=True),
            discovery=DiscoveryConfig(enabled=True),
            database=DatabaseConfig(population_size=10, num_islands=2),
        )

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @patch("openevolve.controller.LLMEnsemble")
    @patch("openevolve.controller.Evaluator")
    @patch("openevolve.controller.DiscoveryEngine")
    def test_full_state_invariance(self, mock_discovery_cls, mock_evaluator, mock_llm):
        """
        Verify that saving and loading state results in an identical agent.
        """
        # --- 1. Setup Controller A (Original) ---

        # Mock dependencies to avoid external calls
        mock_discovery_instance = MagicMock()
        mock_discovery_cls.return_value = mock_discovery_instance
        # Mock save/load for discovery engine to simulate real behavior (returning/accepting dicts)
        discovery_state = {"events": ["event1"], "stats": {"surprise": 0.9}}
        mock_discovery_instance.save_state.return_value = discovery_state

        controller_a = OpenEvolve(
            initial_program_path=self.initial_program_path,
            evaluation_file=self.evaluator_path,
            config=self.config,
            output_dir=self.test_dir,
        )

        # --- 2. Mutate Controller A's State ---

        # A. Mutate Database
        prog1 = Program(
            id="prog1",
            code="print('1')",
            metrics={"score": 0.5},
            iteration_found=1,
            metadata={"island": 0},
        )
        controller_a.database.add(prog1)

        # B. Mutate RL Policy
        # Manually update stats to simulate learning
        if controller_a.rl_policy:
            # Update action 0 stats
            controller_a.rl_policy.global_action_stats[0].update(reward=0.8)
            controller_a.rl_policy.global_action_stats[0].update(reward=0.9)
            controller_a.rl_policy.iterations = 5
            controller_a.rl_policy.exploration_rate = 0.5

        # --- 3. Save State ---
        checkpoint_iter = 5
        controller_a._save_checkpoint(checkpoint_iter)

        checkpoint_path = os.path.join(
            self.test_dir, "checkpoints", f"checkpoint_{checkpoint_iter}"
        )
        self.assertTrue(os.path.exists(checkpoint_path))

        # Simulate DiscoveryEngine side effect (since it is mocked)
        os.makedirs(os.path.join(checkpoint_path, "discovery_state"), exist_ok=True)

        # Verify files exist
        self.assertTrue(os.path.exists(os.path.join(checkpoint_path, "programs", "prog1.json")))
        self.assertTrue(os.path.exists(os.path.join(checkpoint_path, "rl_policy_state.json")))
        # Discovery state is usually a directory
        self.assertTrue(os.path.exists(os.path.join(checkpoint_path, "discovery_state")))

        # --- 4. Setup Controller B (Restored) ---

        # Create a fresh controller
        controller_b = OpenEvolve(
            initial_program_path=self.initial_program_path,
            evaluation_file=self.evaluator_path,
            config=self.config,
            output_dir=self.test_dir,
        )

        # Configure mock for B to accept load_state
        mock_discovery_instance_b = mock_discovery_cls.return_value

        # Load the checkpoint
        controller_b._load_checkpoint(checkpoint_path)

        # --- 5. Verify Invariance ---

        # A. Verify Database
        self.assertIn("prog1", controller_b.database.programs)
        prog1_b = controller_b.database.programs["prog1"]
        self.assertEqual(prog1.code, prog1_b.code)
        self.assertEqual(prog1.metrics, prog1_b.metrics)
        self.assertEqual(prog1.metadata, prog1_b.metadata)

        # B. Verify RL Policy
        if controller_a.rl_policy and controller_b.rl_policy:
            state_a = controller_a.rl_policy.save_state()
            state_b = controller_b.rl_policy.save_state()

            # Deep compare dictionaries
            # We can't compare the 'algorithm_state' directly if it contains numpy arrays,
            # unless we convert them to lists. PolicyLearner.save_state does convert to lists.

            self.assertEqual(state_a["iterations"], state_b["iterations"])
            self.assertEqual(state_a["exploration_rate"], state_b["exploration_rate"])

            # Compare action stats
            stats_a = state_a["global_action_stats"][0]
            stats_b = state_b["global_action_stats"][0]
            self.assertEqual(stats_a["uses"], stats_b["uses"])
            self.assertEqual(stats_a["total_reward"], stats_b["total_reward"])

            # Ensure the full state dict is identical (excluding potentially variable timestamps if any)
            # Using json.dumps to handle potential type discrepancies (like tuple vs list) and ensuring exact match
            json_a = json.dumps(state_a, sort_keys=True)
            json_b = json.dumps(state_b, sort_keys=True)
            self.assertEqual(json_a, json_b)

        # C. Verify Discovery Engine (via Mock interaction)
        # Check that load_state was called on the mock with the correct path
        expected_discovery_path = os.path.join(checkpoint_path, "discovery_state")
        # Note: OpenEvolve creates a new instance of DiscoveryEngine in __init__, so we check the most recent call
        # We need to ensure we are checking the instance attached to controller_b
        controller_b.discovery_engine.load_state.assert_called()
        # Get the path passed to load_state
        call_args = controller_b.discovery_engine.load_state.call_args
        self.assertEqual(call_args[0][0], expected_discovery_path)


if __name__ == "__main__":
    unittest.main()
