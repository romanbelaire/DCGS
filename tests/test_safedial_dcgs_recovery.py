import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_safedial_dcgs_wildjailbreak as runner
import safedial_dcgs_wildjailbreak as adapter
from safedial_dcgs_run_state import audit_state, context_summary, coverage
from validate_safedial_dcgs_wildjailbreak import validate
from test_safedial_dcgs_wildjailbreak import ROW, Tokenizer, execute, is_ll_generation


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.tok = Tokenizer()
        self.config = adapter.configuration("vdcgs")
        self.turn_calls = len(adapter.generate_turn(ROW, 0, 0, self.config, self.tok, execute)["dcgs_original"]["events"])

    def setup_run(self, mode="stop", turns=2, long=False):
        row = copy.deepcopy(ROW)
        row.update(task="Ethics", scene="fixture")
        row["history"] = [copy.deepcopy(row["history"][0]) for _ in range(turns)]
        if long:
            row["history"][0]["user"] = "x" * 1700
        data = self.root / "data.jsonl"
        data.write_text(json.dumps(row) + "\n")
        args = SimpleNamespace(method="vdcgs", device="cpu", seed=0, dataset=data, on_turn_error=mode)
        lock = {"actor": {}, "high_level": {"path": self.config.checkpoint_path, "sha256": "fixture"},
                "token_critic": {"path": self.config.ll_token_critic_path, "sha256": "ll-fixture"}}
        manifest = json.loads(json.dumps(runner.manifest_for(args, [row], lock)))
        folder = self.root / "original"
        folder.mkdir()
        runner.write_json(folder / "run_config.json", manifest)
        return folder, [row], manifest

    def run_policy(self, folder, rows, manifest, backend=execute):
        return runner.run_selected(folder, rows, manifest, self.config, self.tok, lambda: backend)

    def state(self, folder, rows, manifest):
        return audit_state(folder, rows, manifest, self.config, self.tok)

    def test_preflight_reports_context_loss_separately(self):
        _, rows, _ = self.setup_run(turns=1, long=True)
        report = runner.preflight(rows, self.config, self.tok)
        self.assertTrue(report["passed"])
        self.assertEqual(report["model_calls"], 0)
        self.assertFalse(report["estimated_critic_context"]["critic_full_context"])
        self.assertEqual(report["estimated_critic_context"]["critic_truncated_turns"], 1)
        self.assertEqual(report["estimated_critic_context"]["critic_truncation_sides"], ["left"])

    def test_context_counts_separate_duplicates_heads_and_turns(self):
        _, rows, _ = self.setup_run(turns=1, long=True)
        result = adapter.generate_turn(rows[0], 0, 0, adapter.configuration("rdcgs"), self.tok, execute)
        entries = [{"dialogue_id": 1, "turn_index": 0, **e} for e in result["dcgs_original"]["events"]]
        report = context_summary(entries)
        self.assertEqual(report["critic_truncated_turns"], 1)
        self.assertEqual(report["critic_candidate_contexts"], 4)
        self.assertEqual(report["critic_truncated_candidate_contexts"], 4)
        self.assertEqual(sum(report["critic_truncated_inputs_by_head"].values()), 30)

    def test_validator_recomputes_context_and_rejects_missing_metadata(self):
        folder, rows, manifest = self.setup_run(turns=1, long=True)
        self.run_policy(folder, rows, manifest)
        report = validate(folder, tokenizer=self.tok)
        self.assertTrue(report["policy_parity_passed"])
        self.assertTrue(report["passed"])
        self.assertFalse(report["critic_full_context"])
        self.assertNotIn("input_truncated_turns", report)
        original_turns = runner.read_records(folder / "turns.jsonl")
        original_events = runner.read_records(folder / "events.jsonl")
        for remove in (False, True):
            turns, events = copy.deepcopy(original_turns), copy.deepcopy(original_events)
            for e in turns[0]["dcgs_original"]["events"] + events:
                if e["request"]["kind"] == "score":
                    if remove:
                        del e["result"]["critic_context"]
                    else:
                        e["result"]["critic_context"]["dropped_tokens"][0] = 0
            (folder / "turns.jsonl").write_text("".join(json.dumps(x) + "\n" for x in turns))
            (folder / "events.jsonl").write_text("".join(json.dumps(x) + "\n" for x in events))
            with self.assertRaises((ValueError, adapter.PolicyFailure)):
                validate(folder, tokenizer=self.tok)

    def test_continue_records_terminal_output_and_later_gold_turns(self):
        folder, rows, manifest = self.setup_run("record-and-continue", turns=3)
        finals = []
        def backend(request):
            if is_ll_generation(request):
                if request["max_new_tokens"] == 640:
                    finals.append(request)
                if len(finals) == 2:
                    return {"texts": ["\n"]}
            return execute(request)
        self.assertEqual(self.run_policy(folder, rows, manifest, backend), 2)
        self.assertEqual(len(finals), 3)
        self.assertFalse((folder / "failure.json").exists())
        report = validate(folder, tokenizer=self.tok)
        self.assertTrue(report["execution_finished"])
        self.assertFalse(report["complete_without_failures"])
        self.assertEqual((report["successful_turns"], report["terminal_failed_turns"], report["remaining_turns"]), (2, 1, 0))
        records = runner.read_records(folder / "turns.jsonl")
        self.assertIn(rows[0]["history"][1]["bot"], str(records[1]["prompt_history"]))
        before = {p.name: p.read_bytes() for p in folder.iterdir() if p.is_file()}
        self.assertEqual(self.run_policy(folder, rows, manifest, lambda _: self.fail("Terminal output retried")), 2)
        self.assertEqual(before, {p.name: p.read_bytes() for p in folder.iterdir() if p.is_file()})

    def test_infrastructure_still_stops_in_continue_mode(self):
        folder, rows, manifest = self.setup_run("record-and-continue")
        def backend(request):
            raise RuntimeError("CUDA out of memory")
        with self.assertRaises(adapter.PolicyFailure):
            self.run_policy(folder, rows, manifest, backend)
        failure = json.loads((folder / "failure.json").read_text())
        self.assertEqual(failure["category"], "infrastructure")
        self.assertEqual(failure["stage"], "belief_generation")
        self.assertEqual(failure["dialogue_id"], 1)
        self.assertIsNotNone(failure["request"])
        self.assertIsNotNone(self.state(folder, rows, manifest)["blocker"])

    def test_recovery_preserves_successes_seed_and_source(self):
        folder, rows, manifest = self.setup_run()
        calls = []
        def backend(request):
            calls.append(request)
            if len(calls) == self.turn_calls + 1:
                raise OSError("Injected transient I/O fault")
            return execute(request)
        with self.assertRaises(adapter.PolicyFailure):
            self.run_policy(folder, rows, manifest, backend)
        old = {p.name: p.read_bytes() for p in folder.iterdir()}
        target = self.root / "recovered"
        runner.recover_run(folder, target, manifest, rows, self.config, self.tok, "I/O fault resolved; same policy")
        self.assertEqual(old, {p.name: p.read_bytes() for p in folder.iterdir() if p.name != ".lock"})
        self.assertEqual(self.run_policy(target, rows, manifest), 0)
        self.assertTrue((target / "turns.jsonl").read_bytes().startswith(old["turns.jsonl"]))
        self.assertEqual(runner.read_records(target / "turns.jsonl")[1]["seed"], runner.read_records(folder / "failures.jsonl")[0]["seed"])
        self.assertTrue(validate(target, tokenizer=self.tok)["passed"])

    def test_terminal_failure_recovery_only_processes_unattempted_turns(self):
        folder, rows, manifest = self.setup_run()
        def blank(request):
            return {"texts": [""]} if is_ll_generation(request) else execute(request)
        with self.assertRaises(adapter.PolicyFailure):
            self.run_policy(folder, rows, manifest, blank)
        target = self.root / "continue"
        runner.recover_run(folder, target, manifest, rows, self.config, self.tok, "Preserve blank output; process remaining independent turns")
        calls = []
        def backend(request):
            calls.append(request)
            return execute(request)
        self.assertEqual(self.run_policy(target, rows, manifest, backend), 2)
        self.assertEqual(len(calls), self.turn_calls)
        self.assertEqual(runner.read_records(target / "turns.jsonl")[0]["turn_index"], 1)
        self.assertTrue(validate(target, tokenizer=self.tok)["execution_finished"])

    def test_recovery_refuses_policy_change_and_integrity_failure(self):
        folder, rows, manifest = self.setup_run()
        changed = copy.deepcopy(manifest)
        changed["seed"] += 1
        with self.assertRaisesRegex(ValueError, "cannot change"):
            runner.recover_run(folder, self.root / "bad", changed, rows, self.config, self.tok, "reason")
        with self.assertRaises(adapter.PolicyFailure):
            self.run_policy(folder, rows, manifest, lambda _: {"texts": []})
        with self.assertRaisesRegex(ValueError, "Integrity"):
            runner.recover_run(folder, self.root / "bad", manifest, rows, self.config, self.tok, "reason")
        self.assertFalse((self.root / "bad").exists())

    def test_deleting_marker_cannot_bypass_failure_ledger(self):
        folder, rows, manifest = self.setup_run()
        with self.assertRaises(adapter.PolicyFailure):
            self.run_policy(folder, rows, manifest, lambda _: {"texts": [""]})
        (folder / "failure.json").unlink()
        with self.assertRaisesRegex(ValueError, "fresh directory"):
            self.run_policy(folder, rows, manifest, lambda _: self.fail("Bypassed failure"))

    def test_missing_terminal_commit_is_reconciled_without_retry(self):
        folder, rows, manifest = self.setup_run("record-and-continue")
        original_append = runner.append_jsonl
        def interrupted(path, records):
            if path.name == "failures.jsonl":
                raise KeyboardInterrupt()
            return original_append(path, records)
        def blank(request):
            return {"texts": [""]} if is_ll_generation(request) else execute(request)
        with patch.object(runner, "append_jsonl", interrupted), self.assertRaises(KeyboardInterrupt):
            self.run_policy(folder, rows, manifest, blank)
        calls = []
        def backend(request):
            calls.append(request)
            return execute(request)
        self.assertEqual(self.run_policy(folder, rows, manifest, backend), 2)
        self.assertEqual(len(calls), self.turn_calls)
        self.assertEqual(validate(folder, tokenizer=self.tok)["terminal_failed_turns"], 1)

    def test_missing_success_commit_is_reconciled_without_model(self):
        folder, rows, manifest = self.setup_run(turns=1)
        original_append = runner.append_jsonl
        def interrupted(path, records):
            if path.name == "turns.jsonl":
                raise KeyboardInterrupt()
            return original_append(path, records)
        with patch.object(runner, "append_jsonl", interrupted), self.assertRaises(KeyboardInterrupt):
            self.run_policy(folder, rows, manifest)
        self.assertEqual(self.run_policy(folder, rows, manifest, lambda _: self.fail("Completed response regenerated")), 0)
        self.assertTrue(validate(folder, tokenizer=self.tok)["passed"])

    def test_startup_failure_recovery_and_gpu_validation(self):
        folder, rows, manifest = self.setup_run(turns=1)
        def failed_init():
            raise RuntimeError("CUDA device temporarily unavailable")
        with self.assertRaises(RuntimeError):
            runner.run_selected(folder, rows, manifest, self.config, self.tok, failed_init)
        target = self.root / "recovered"
        runner.recover_run(folder, target, manifest, rows, self.config, self.tok, "CUDA allocation corrected")
        def backend(request):
            result = execute(request)
            if request["kind"] == "generate":
                result["raw_generations"] = [{"input_ids": [[1]], "generated_token_ids": [[2]]}]
            return result
        self.run_policy(target, rows, manifest, backend)
        runtimes = runner.read_records(target / "runtime.jsonl")
        runtimes[-1]["peak_gpu_allocated_bytes"] = 1024
        (target / "runtime.jsonl").write_text("".join(json.dumps(x) + "\n" for x in runtimes))
        runner.append_jsonl(target / "model_loading.jsonl", [{"invocation": runtimes[-1]["invocation"],
            "strict_loaded_heads": ["q", "v"], "ll_reranking": True, "checkpoint_sha256": "fixture",
            "method_spec": adapter.METHOD_SPEC, "token_critic_sha256": "ll-fixture", "token_critic_objective": "shapley",
            "actor_dtype": "torch.bfloat16", "device": "cuda:0"}])
        report = validate(target, require_gpu=True, tokenizer=self.tok)
        self.assertTrue(report["passed"])
        self.assertEqual(report["startup_failures_without_gpu_measurement"], 1)

    def test_recovery_reconciles_error_before_failure_commit(self):
        folder, rows, manifest = self.setup_run(turns=1)
        original_append = runner.append_jsonl
        def interrupted(path, records):
            if path.name == "failures.jsonl":
                raise KeyboardInterrupt()
            return original_append(path, records)
        def broken(request):
            raise OSError("Transient read failure")
        with patch.object(runner, "append_jsonl", interrupted), self.assertRaises(KeyboardInterrupt):
            self.run_policy(folder, rows, manifest, broken)
        target = self.root / "recovered"
        runner.recover_run(folder, target, manifest, rows, self.config, self.tok, "I/O service restored")
        self.assertEqual(self.run_policy(target, rows, manifest), 0)
        self.assertTrue(validate(target, tokenizer=self.tok)["passed"])

    def test_recovery_refuses_damaged_journal_and_source_provenance(self):
        folder, rows, manifest = self.setup_run(turns=1)
        self.run_policy(folder, rows, manifest)
        target = self.root / "recovered"
        runner.recover_run(folder, target, manifest, rows, self.config, self.tok, "Audit copy")
        with (target / "provenance/source_run/turns.jsonl").open("a") as handle:
            handle.write("{}\n")
        with self.assertRaisesRegex(ValueError, "snapshot mismatch"):
            self.run_policy(target, rows, manifest, lambda _: self.fail("Loaded on corruption"))
        with (folder / "events.jsonl").open("a") as handle:
            handle.write('{"partial":')
        with self.assertRaises(ValueError):
            runner.recover_run(folder, self.root / "damaged", manifest, rows, self.config, self.tok, "Cannot salvage silently")

    def test_recovery_refuses_occupied_destination_and_locked_source(self):
        import fcntl
        folder, rows, manifest = self.setup_run()
        target = self.root / "occupied"
        target.mkdir()
        with self.assertRaisesRegex(ValueError, "fresh"):
            runner.recover_run(folder, target, manifest, rows, self.config, self.tok, "reason")
        with (folder / ".lock").open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                runner.recover_run(folder, self.root / "new", manifest, rows, self.config, self.tok, "reason")


if __name__ == "__main__":
    unittest.main()
