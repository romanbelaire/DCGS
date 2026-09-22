import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_safedial_baseline as base
import run_safedial_smoothllm_turn_resume as resume
import run_safedial_smoothllm_recovery as recovery
from safedial_smoothllm import candidate_inputs
from test_safedial_smoothllm import decoded


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.output = self.root / "recovery"
        self.output.mkdir()
        rows = [{"id": 344, "task": "Ethics", "method": "fixture", "scene": "fixture",
                 "history": [{"user": f"Please explain this sufficiently long question number {i}.",
                              "bot": f"Gold answer {i}"} for i in range(5)]},
                {"id": 345, "task": "Ethics", "method": "fixture", "scene": "fixture",
                 "history": [{"user": "A separate dialogue", "bot": "Another gold answer"}]}]
        dataset = self.root / "dataset.jsonl"
        base.append_jsonl(dataset, rows)
        args = resume.parse_args(["--dataset", str(dataset), "--output-dir", str(self.source)])
        manifest = resume.manifest_for(args, rows)
        calls = []

        def initial(prompt, seed):
            calls.append(seed)
            return decoded("") if len(calls) == 33 else decoded()

        with resume.single_writer(self.source):
            base.ensure_manifest(self.source / "run_config.json", manifest)
            self.assertEqual(resume.generate_dialogues(args, manifest, rows, initial), 1)
        self.state = recovery.inspect_source(self.source)
        self.manifest = recovery.recovery_manifest(self.source, self.state)
        self.before = recovery.fingerprints(self.source)

    def attempt(self, generate):
        return recovery.run_attempt(self.output, self.state, self.manifest,
                                    factory=lambda m: (generate, lambda: {"fixture": True}))

    def test_exactly_eight_calls_only_target_preserved_neighbours_and_noop(self):
        calls = []
        # Accept the algorithm's answer regardless of refusal heuristic.
        self.assertEqual(self.attempt(lambda p, s: calls.append((p, s)) or decoded("Unrestricted response")), 0)
        self.assertEqual(len(calls), 8)
        self.assertEqual(recovery.fingerprints(self.source), self.before)
        saved = base.load_jsonl(self.output / "dialogue_turns.jsonl")
        self.assertEqual(saved[:4], self.state["dialogue"][:4])
        self.assertNotEqual(saved[4]["seed"], self.state["dialogue"][4]["seed"])
        old, _ = candidate_inputs(saved[4]["prompt_history"], self.state["manifest"]["defense"],
                                  self.state["dialogue"][4]["seed"])
        self.assertNotEqual(calls[0][0], old[0])
        for prompt, seed in calls:
            self.assertEqual(prompt[:-1], saved[4]["prompt_history"][:-1])
        answer = base.load_jsonl(self.output / "answers.jsonl")
        self.assertEqual(len(answer), 1)
        self.assertEqual(len(answer[0]["choices"][0]["turns"]), 5)
        self.assertEqual(answer[0]["id"], 344)
        self.assertEqual(self.attempt(lambda p, s: self.fail("Second attempt occurred")), 0)
        self.assertTrue(recovery.validate_result(self.output, self.state, self.manifest)["audit_passed"])
        with self.assertRaisesRegex(ValueError, "GPU"):
            recovery.validate_result(self.output, self.state, self.manifest, require_gpu=True)

    def test_blank_candidate_stops_without_voting_or_retry(self):
        calls = []
        def generate(p, s):
            calls.append(s)
            return decoded("") if len(calls) == 3 else decoded()
        self.assertEqual(self.attempt(generate), 2)
        self.assertEqual(len(calls), 3)
        result = json.loads((self.output / "result.json").read_text())
        self.assertIsNone(result["record"]["smoothllm"]["selected_index"])
        self.assertEqual(base.load_jsonl(self.output / "answers.jsonl"), [])
        self.assertEqual(self.attempt(lambda p, s: self.fail("Retried failure")), 2)
        self.assertEqual(recovery.fingerprints(self.source), self.before)

    def test_interruption_reserves_attempt_and_blocks_automatic_rerun(self):
        def interrupt(p, s):
            raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            self.attempt(interrupt)
        self.assertTrue((self.output / "attempt_started.json").exists())
        with self.assertRaisesRegex(RuntimeError, "No automatic retry"):
            self.attempt(lambda p, s: self.fail("Repeated interrupted attempt"))

    def test_initialization_error_recorded_without_retry(self):
        def broken(m):
            raise RuntimeError("fixture model load failed")
        self.assertEqual(recovery.run_attempt(self.output, self.state, self.manifest, broken), 2)
        result = json.loads((self.output / "result.json").read_text())
        self.assertIn("model load failed", result["infrastructure_error"])
        self.assertEqual(self.attempt(lambda p, s: self.fail("Retried initialization")), 2)

    def test_seed_tampering_rejected(self):
        self.assertEqual(self.attempt(lambda p, s: decoded()), 0)
        path = self.output / "result.json"
        result = json.loads(path.read_text())
        result["record"]["seed"] += 1
        recovery.write_json(path, result)
        with self.assertRaisesRegex(ValueError, "seed"):
            recovery.validate_result(self.output, self.state, self.manifest)

    def test_export_tampering_rejected(self):
        self.assertEqual(self.attempt(lambda p, s: decoded()), 0)
        answers = base.load_jsonl(self.output / "answers.jsonl")
        answers[0]["choices"][0]["turns"][0]["message"] = "changed preserved answer"
        resume.atomic_jsonl(self.output / "answers.jsonl", answers)
        with self.assertRaisesRegex(ValueError, "export differs"):
            recovery.validate_exports(self.output, self.state)

    def test_changed_source_and_missing_coverage_rejected(self):
        self.assertEqual(self.attempt(lambda p, s: decoded()), 0)
        with (self.source / "answers.jsonl").open("a") as f:
            f.write("\n")
        with self.assertRaisesRegex(ValueError, "Original results changed"):
            recovery.validate_result(self.output, self.state, self.manifest)
        records = base.load_jsonl(self.source / "turns.jsonl")
        resume.atomic_jsonl(self.source / "turns.jsonl", records[:-1])
        with self.assertRaisesRegex(ValueError, "missing turns"):
            recovery.inspect_source(self.source)

    def test_changed_pinned_source_hash_rejected(self):
        p = self.source / "run_config.json"
        manifest = json.loads(p.read_text())
        manifest["implementation_sha256"]["safedial_smoothllm.py"] = "wrong"
        recovery.write_json(p, manifest)
        with self.assertRaisesRegex(ValueError, "pinned implementation"):
            recovery.inspect_source(self.source)

    def test_validate_only_readonly_and_source_lock(self):
        absent = self.root / "never-created"
        self.assertEqual(recovery.main(["--source-dir", str(self.source), "--output-dir", str(absent),
                                        "--validate-only"]), 0)
        self.assertFalse(absent.exists())
        self.assertEqual(recovery.fingerprints(self.source), self.before)
        with resume.single_writer(self.source):
            with self.assertRaisesRegex(RuntimeError, "active writer"):
                with recovery.source_lock(self.source):
                    self.fail("Ignored source writer lock")
        with self.assertRaisesRegex(ValueError, "separate sibling"):
            recovery.main(["--source-dir", str(self.source), "--output-dir", str(self.source / "nested"),
                           "--validate-only"])


if __name__ == "__main__":
    unittest.main()
