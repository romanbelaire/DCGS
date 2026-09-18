import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_safedial_baseline as base
import run_safedial_tpo_upstream as runner
import run_safedial_tpo_upstream_full as full
import safedial_tpo_upstream as tpo
from validate_safedial_tpo_upstream import validate_full
from test_safedial_tpo import fake_execute


class UpstreamPolicyTests(unittest.TestCase):
    def setUp(self):
        self.config = tpo.defense_config(sample_size=3)
        self.messages = [{"role": "user", "content": "test"}]

    def test_exact_upstream_extraction_including_empty_and_prompt_echo(self):
        cases = ["", "no tags", tpo.START, tpo.START+tpo.END,
                 tpo.START+"hello", tpo.START+"first<VARIABLE>echo"+tpo.START+"second",
                 tpo.END+tpo.START+"first"+tpo.END,
                 tpo.START+"first"+tpo.END+tpo.START+"second"+tpo.END]
        for path in (Path(__file__).parent / "fixtures").glob("tpo_*.json"):
            data = json.loads(path.read_text())
            if isinstance(data.get("result"), dict) and "message" in data["result"]:
                cases.append(data["result"]["message"])
        for raw in cases:
            try:
                expected = raw.split(tpo.START)[1].split(tpo.END)[0].strip()
            except IndexError:
                with self.assertRaises(IndexError): tpo.extract_update(raw)
            else:
                self.assertEqual(tpo.extract_update(raw), expected)

    def test_all_updates_skipped_retains_initial_pool_and_has_no_retries(self):
        calls = []
        def execute(req):
            calls.append(req)
            out = fake_execute(req)
            if req["stage"] == "update": out["message"] = "no tags"
            return out
        result = tpo.generate_tpo(self.messages, self.config, 42, execute)
        self.assertEqual(result["message"], "initial-0")
        self.assertEqual(len(result["tpo"]["candidates"]), 3)
        self.assertEqual(sum(bool(e.get("candidate_failure")) for e in result["tpo"]["events"]), 6)
        self.assertEqual(len(calls), 16)  # 13 generations, 3 rewards; no replacement calls.
        self.assertFalse(any("retry_attempt" in req for req in calls))
        self.assertEqual([(r["chosen_id"],r["rejected_id"]) for r in result["tpo"]["rounds"]], [(0,1),(0,1)])

    def test_empty_extracted_candidate_is_scored_and_can_win(self):
        def execute(req):
            out = fake_execute(req)
            if req["stage"] == "update": out["message"] = tpo.START+tpo.END
            if req["kind"] == "reward" and not req["messages"][-1]["content"]: out["score"] = 100
            return out
        result = tpo.generate_tpo(self.messages, self.config, 0, execute)
        self.assertEqual(result["message"], "")
        self.assertEqual(len(result["tpo"]["candidates"]), 9)
        record = {**result, "prompt_history": self.messages, "seed": 0, "generated_response": "",
                  "error": "empty_selected_answer"}
        tpo.validate_audit(record, self.config)
        record["error"] = None
        with self.assertRaisesRegex(ValueError, "outcome"): tpo.validate_audit(record, self.config)

    def test_blank_update_skipped_but_blank_initial_and_feedback_retained(self):
        def execute(req):
            out = fake_execute(req)
            if req["kind"] == "generate": out["message"] = ""
            return out
        result = tpo.generate_tpo(self.messages, self.config, 0, execute)
        self.assertEqual(len(result["tpo"]["candidates"]), 3)
        self.assertEqual(len(result["tpo"]["events"]), 16)
        self.assertEqual(result["message"], "")

    def test_backend_and_accounting_errors_are_not_skipped(self):
        for kind in ("cuda", "nan", "truncated"):
            def execute(req):
                if kind == "cuda": raise RuntimeError("CUDA out of memory")
                out = fake_execute(req)
                if kind == "nan" and req["kind"] == "reward": out["score"] = float("nan")
                if kind == "truncated": out["input_truncated"] = True
                return out
            events = []
            with self.assertRaises((RuntimeError, ValueError)):
                tpo.generate_tpo(self.messages, self.config, 0, execute, events.append)
            self.assertIn("error", events[-1])
            self.assertNotIn("candidate_failure", events[-1])


class UpstreamRunnerTests(unittest.TestCase):
    def fixture(self, root):
        data = root / "dataset.jsonl"
        rows = [{"id": i, "task": "Ethics", "method": "fixture", "scene": "fixture",
                 "history": [{"user": f"test-{i}", "bot": "reference"}]} for i in (1,2)]
        base.append_jsonl(data, rows)
        args = full.parse_args(["--dataset", str(data), "--output-dir", str(root / "out"), "--device", "cpu", "--sample-size", "3"])
        args.output_dir.mkdir()
        manifest = full.manifest_for(args, rows, {})
        base.ensure_manifest(args.output_dir / "run_config.json", manifest)
        return args, manifest, rows

    def test_skips_ledger_and_noop_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, manifest, rows = self.fixture(Path(tmp))
            def execute(req):
                out = fake_execute(req)
                if req["stage"] == "update" and req["slot"] == 1: out["message"] = "no tag"
                return out
            self.assertEqual(runner.generate_selected(args, manifest, rows, execute), 0)
            report = validate_full(args.output_dir)
            self.assertTrue(report["passed"])
            self.assertEqual(report["skipped_candidates"], 4)
            self.assertEqual(report["candidates"], 14)
            paths = [args.output_dir / name for name in ("turns.jsonl", "events.jsonl", "failures.jsonl", "answers.jsonl")]
            before = [p.read_bytes() for p in paths]
            self.assertEqual(runner.generate_selected(args, manifest, rows, lambda r: self.fail("resume called model")), 0)
            self.assertEqual(before, [p.read_bytes() for p in paths])

    def test_empty_selected_turn_continues_exports_only_good_dialogues(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, manifest, rows = self.fixture(Path(tmp))
            def execute(req):
                out = fake_execute(req)
                if req["stage"] == "initial" and req["messages"][0]["content"] == "test-1": out["message"] = ""
                if req["kind"] == "reward" and not req["messages"][-1]["content"]: out["score"] = 100
                return out
            self.assertEqual(runner.generate_selected(args, manifest, rows, execute), 2)
            report = validate_full(args.output_dir)
            self.assertTrue(report["audit_passed"] and report["processing_complete"])
            self.assertFalse(report["passed"])
            self.assertEqual((report["successful_turns"],report["failed_turns"]),(1,1))
            self.assertEqual([r["id"] for r in base.load_jsonl(args.output_dir / "answers.jsonl")], [2])
            self.assertFalse((args.output_dir / "failure.json").exists())
            self.assertEqual(runner.generate_selected(args, manifest, rows, lambda r: self.fail("retried terminal turn")), 2)

    def test_interrupt_after_skipped_event_resumes_without_regenerating_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, manifest, rows = self.fixture(Path(tmp))
            calls = []
            def execute(req):
                calls.append(req)
                if req["stage"] == "update" and req["slot"] == 1: raise KeyboardInterrupt()
                out = fake_execute(req)
                if req["stage"] == "update" and req["slot"] == 0: out["message"] = "missing"
                return out
            with self.assertRaises(KeyboardInterrupt): runner.generate_selected(args, manifest, rows, execute)
            interrupted_seed = calls[-1]["seed"]
            resumed = []
            def finish(req):
                resumed.append(req)
                out = fake_execute(req)
                if req["stage"] == "update" and req["slot"] == 0: out["message"] = "missing"
                return out
            runner.generate_selected(args, manifest, rows, finish)
            self.assertEqual(resumed[0]["seed"], interrupted_seed)
            self.assertTrue(validate_full(args.output_dir)["passed"])
            self.assertEqual(len(base.load_jsonl(args.output_dir / "failures.jsonl")), 4)

    def test_completed_events_without_turn_record_reconcile_without_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, manifest, rows = self.fixture(Path(tmp))
            runner.generate_selected(args, manifest, rows, fake_execute)
            (args.output_dir / "turns.jsonl").write_text("")
            runner.generate_selected(args, manifest, rows, lambda r: self.fail("regenerated completed events"))
            self.assertTrue(validate_full(args.output_dir)["passed"])

    def test_tampered_skip_metadata_and_ledger_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, manifest, rows = self.fixture(Path(tmp))
            def execute(req):
                out = fake_execute(req)
                if req["stage"] == "update": out["message"] = "missing"
                return out
            runner.generate_selected(args, manifest, rows, execute)
            path = args.output_dir / "failures.jsonl"
            original = path.read_bytes()
            path.write_text("")
            with self.assertRaisesRegex(ValueError, "ledger"): validate_full(args.output_dir)
            path.write_bytes(original)
            records = base.load_jsonl(args.output_dir / "turns.jsonl")
            event = next(e for e in records[0]["tpo"]["events"] if e.get("candidate_failure"))
            event["candidate_failure"]["code"] = "altered"
            runner.atomic_jsonl(args.output_dir / "turns.jsonl", records)
            with self.assertRaisesRegex(ValueError, "audit"): validate_full(args.output_dir)

    def test_fatal_failure_blocks_unchanged_resume_and_retains_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, manifest, rows = self.fixture(Path(tmp))
            def execute(req): raise RuntimeError("CUDA out of memory")
            with self.assertRaisesRegex(RuntimeError, "CUDA"): runner.generate_selected(args, manifest, rows, execute)
            failures = base.load_jsonl(args.output_dir / "failures.jsonl")
            self.assertEqual(failures[0]["category"], "fatal_execution")
            with self.assertRaisesRegex(ValueError, "Prior fatal"): runner.generate_selected(args, manifest, rows, fake_execute)

    def test_reject_retry_flags_and_previous_policy_manifest(self):
        for flags in (["--empty-generation-retries", "2"], ["--retry-errors"]):
            with self.assertRaises(ValueError): full.parse_args(flags)
        with tempfile.TemporaryDirectory() as tmp:
            args, manifest, rows = self.fixture(Path(tmp))
            old = copy.deepcopy(manifest)
            old["defense"]["implementation_version"] = 8
            with self.assertRaises(RuntimeError): base.ensure_manifest(args.output_dir / "run_config.json", old)

    def test_full_entrypoint_preserves_costs_probe_and_failure_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, manifest, rows = self.fixture(Path(tmp))
            def execute(req):
                out = fake_execute(req)
                if req["kind"] == "generate": out["message"] = ""
                return out
            backend = MagicMock(side_effect=execute)
            backend.loading = {k: {"passed": True, "trainable_parameters": 0} for k in ("actor", "reward")}
            with patch.object(full, "parse_args", return_value=args), \
                 patch.object(full, "validate_lock", return_value={}), \
                 patch.object(full, "tokenizers_and_preflight", return_value=({}, {}, {})), \
                 patch.object(full, "LocalBackend", return_value=backend), \
                 patch.object(full, "probe_reward_context", return_value={"passed": True, "sequence_tokens": 8192}):
                self.assertEqual(full.main(), 2)
            stats = json.loads((args.output_dir / "runtime_stats.json").read_text())
            self.assertEqual(stats["actual_calls"], {"generate": 26, "reward": 6})
            self.assertEqual(stats["startup_reward_probe_calls"], 1)
            self.assertIsNone(stats["failure"])
            report = validate_full(args.output_dir)
            self.assertEqual(report["failed_turns"], 2)
            self.assertTrue(report["processing_complete"] and report["audit_passed"])
            with self.assertRaises(ValueError): validate_full(args.output_dir, require_gpu=True)


if __name__ == "__main__":
    unittest.main()
