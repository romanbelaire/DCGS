import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_dcgs_api_smoke as api


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.calls = []
        def infer(prompt, seed):
            self.calls.append((prompt, seed))
            return {"message": "Example", "audit_summary": {"ll_critic_calls": 1}}
        self.service = api.Service(infer, "test-secret", Path(self.tmp.name), "vdcgs", 1)
        self.payload = {"prompt": "Explain GPUs.", "request_id": "first"}
        self.auth = "Bearer test-secret"

    def test_unauthorized_and_invalid_requests_do_not_run_engine(self):
        self.assertEqual(self.service.generate(None, self.payload)[0], 401)
        for invalid in ({**self.payload, "tools": []}, {**self.payload, "seed": True},
                        {**self.payload, "request_id": "../secret"}, {**self.payload, "prompt": " "}):
            self.assertEqual(self.service.generate(self.auth, invalid)[0], 400)
        self.assertEqual(self.calls, [])

    def test_retry_is_cached_and_conflict_or_limit_does_not_generate(self):
        first = self.service.generate(self.auth, self.payload)
        self.assertEqual(first[0], 200)
        self.assertEqual(self.service.generate(self.auth, {**self.payload, "seed": 0}), first)
        self.assertEqual(self.service.generate(self.auth, {**self.payload, "prompt": "Changed"})[0], 409)
        self.assertEqual(self.service.generate(self.auth, {**self.payload, "request_id": "second"})[0], 429)
        self.assertEqual(len(self.calls), 1)
        saved = json.loads((Path(self.tmp.name) / "first.response.json").read_text())
        self.assertEqual(saved, first[1])

    def test_engine_failure_preserves_audit_and_stops_new_requests(self):
        class Failure(RuntimeError):
            audit = {"events": ["example"]}
        def fail(*args):
            raise Failure("engine failed")
        self.service.infer = fail
        result = self.service.generate(self.auth, self.payload)
        self.assertEqual(result[0], 500)
        self.assertFalse(result[1]["automatic_retry"])
        self.assertEqual(self.service.generate(self.auth, self.payload), result)
        self.assertEqual(self.service.generate(self.auth, {**self.payload, "request_id": "second"})[0], 503)
        self.assertFalse(self.service.health()["ok"])
        self.assertEqual(json.loads((Path(self.tmp.name) / "first.failed_audit.json").read_text()), Failure.audit)


class PolicyTests(unittest.TestCase):
    def test_both_methods_replay_and_summarize_original_policy(self):
        from test_safedial_dcgs_wildjailbreak import Tokenizer, execute
        import safedial_dcgs_wildjailbreak as adapter
        row = api.probe_row(api.SMOKE_PROMPT)
        for method in ("vdcgs", "rdcgs"):
            with self.subTest(method=method):
                config = adapter.configuration(method)
                result = adapter.generate_turn(row, 0, 0, config, Tokenizer(), execute)
                result = json.loads(json.dumps(result))
                adapter.validate_turn({**result, "turn_index": 0, "seed": 0,
                                       "generated_response": result["message"]}, row, config, Tokenizer())
                summary = api.audit_summary(result, config, 123)
                self.assertEqual(summary["ll_candidates_scored"], 5)
                self.assertEqual(summary["regret_enabled"], method == "rdcgs")
                self.assertGreater(summary["critic_context"]["critic_candidate_contexts"], 0)


if __name__ == "__main__":
    unittest.main()
