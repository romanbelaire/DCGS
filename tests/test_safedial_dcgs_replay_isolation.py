import copy
import json
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import safedial_dcgs_wildjailbreak as adapter
import safedial_dcgs_run_state as state
from safedial_dcgs_replay_isolation import isolated_replay
from test_safedial_dcgs_wildjailbreak import ROW, Tokenizer, execute, is_ll_generation


class ReplayIsolationTests(unittest.TestCase):
    def fixture(self, exhaust=False):
        attempts = 0

        def backend(request):
            nonlocal attempts
            if request["kind"] == "generate" and not is_ll_generation(request):
                attempts += 1
                if exhaust or attempts == 1:
                    return {"texts": ["" if attempts == 1 else "[SKIP]"]}
            return execute(request)

        config, tokenizer = adapter.configuration("vdcgs"), Tokenizer()
        return config, tokenizer, backend

    def test_retry_reproduces_failure_then_passes_without_mutating_evidence(self):
        config, tokenizer, backend = self.fixture()
        result = adapter.generate_turn(ROW, 0, 0, config, tokenizer, backend)
        record = {"turn_index": 0, "seed": 0, "generated_response": result["message"],
                  "dcgs_original": result["dcgs_original"]}
        record = json.loads(json.dumps(record))  # Match the persisted JSON trace.
        with self.assertRaisesRegex(ValueError, "selection/audit mismatch"):
            adapter.validate_turn(copy.deepcopy(record), ROW, config, tokenizer)
        saved = copy.deepcopy(record)
        with isolated_replay():
            adapter.validate_turn(record, ROW, config, tokenizer)
            adapter.validate_turn(record, ROW, config, tokenizer)
        self.assertEqual(record, saved)

    def test_tampered_response_still_rejected(self):
        config, tokenizer, backend = self.fixture()
        result = adapter.generate_turn(ROW, 0, 0, config, tokenizer, backend)
        record = {"turn_index": 0, "seed": 0, "generated_response": "tampered",
                  "dcgs_original": result["dcgs_original"]}
        record = json.loads(json.dumps(record))
        with isolated_replay(), self.assertRaisesRegex(ValueError, "selection/audit mismatch"):
            adapter.validate_turn(record, ROW, config, tokenizer)

    def test_terminal_retry_and_incomplete_trace_preserve_evidence(self):
        config, tokenizer, backend = self.fixture(exhaust=True)
        with self.assertRaises(adapter.PolicyFailure) as failed:
            adapter.generate_turn(ROW, 0, 0, config, tokenizer, backend)
        events = failed.exception.audit["events"]
        before = copy.deepcopy(events)
        original = state.replay_trace
        with isolated_replay():
            with self.assertRaises(adapter.PolicyFailure) as replayed:
                state.replay_trace(ROW, 0, 0, config, tokenizer, events)
            self.assertEqual(replayed.exception.audit, failed.exception.audit)
            with self.assertRaises(adapter.TraceIncomplete):
                state.replay_trace(ROW, 0, 0, config, tokenizer, events[:1], allow_incomplete=True)
        self.assertEqual(events, before)
        self.assertIs(state.replay_trace, original)
