import importlib.util
import random
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
SPEC = importlib.util.spec_from_file_location(
    "run_safedial_dcgs", SCRIPTS_DIR / "run_safedial_dcgs.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class SafeDialDCGSTest(unittest.TestCase):
    def test_gold_history_uses_only_prior_gold_assistant_turns(self):
        source = [
            {"user": "u0", "bot": "g0"},
            {"user": "u1", "bot": "g1"},
            {"user": "u2", "bot": "g2"},
        ]
        self.assertEqual(
            MODULE.dcgs_gold_history(source, 2),
            [("[NO_AGENT_ACTION]", "u0"), ("g0", "u1"), ("g1", "u2")],
        )

    def test_observation_matches_episode_state_shape(self):
        self.assertEqual(
            MODULE.format_dcgs_observation(
                [("[NO_AGENT_ACTION]", "hello"), ("gold answer", "follow-up")]
            ),
            "Turn 1:\nAgent: [NO_AGENT_ACTION]\nUser: hello\n"
            "Turn 2:\nAgent: gold answer\nUser: follow-up",
        )

    def test_softmax_selection_is_normalized_and_reproducible(self):
        scores = {"a": -1.0, "b": 0.0, "c": 1.0}
        first, probabilities = MODULE.softmax_select(scores, random.Random(7))
        second, _ = MODULE.softmax_select(scores, random.Random(7))
        self.assertEqual(first, second)
        self.assertAlmostEqual(sum(probabilities.values()), 1.0)
        self.assertGreater(probabilities["c"], probabilities["b"])
        self.assertGreater(probabilities["b"], probabilities["a"])


if __name__ == "__main__":
    unittest.main()
