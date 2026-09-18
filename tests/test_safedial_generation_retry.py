import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_safedial_baseline as base
import run_safedial_tpo as tpo_runner
import safedial_tpo as tpo
import safedial_generation_retry as retry
from validate_safedial_tpo import validate_tpo
from test_safedial_tpo import fake_execute as tpo_execute


class Interrupted(Exception):
    pass


class RetryTests(unittest.TestCase):
    def cases(self):
        # Custom DCGS coverage is preserved with its unchanged source/tests in
        # archive/scripts_cleanup_20260918/snapshot. TPO remains an active method.
        return [("tpo", tpo, tpo_runner, tpo_execute, validate_tpo)]

    def setup_run(self, runner, method, root):
        rows = [{"id": 1, "task": "Ethics", "method": "fixture", "scene": "fixture",
                 "history": [{"user": "request", "bot": "unseen reference"}]}]
        data = root / "data.jsonl"
        base.append_jsonl(data, rows)
        flags = ["--method", "rdcgs"] if method == "dcgs" else []
        args = runner.parse_args(flags + ["--dataset", str(data), "--output-dir", str(root / "out"), "--device", "cpu"])
        args.output_dir.mkdir()
        lock = {}
        for name in ("high_level", "token_critic"):
            file = root / name
            file.write_text("fixture")
            lock[name] = {"path": str(file), "sha256": base.file_sha256(file)}
        manifest = runner.manifest_for(args, rows, lock)
        base.ensure_manifest(args.output_dir / "run_config.json", manifest)
        return args, manifest, rows

    @staticmethod
    def target(request, method):
        return request["stage"] == ("response" if method == "dcgs" else "loss") and request["slot"] == 0

    def test_journal_resume_at_each_attempt_preserves_successes_and_budget(self):
        for method, algorithm, runner, fake, validate in self.cases():
            for interrupt_attempt in (1, 2, 3):
                with self.subTest(method=method, interrupt=interrupt_attempt), tempfile.TemporaryDirectory() as tmp:
                    args, manifest, rows = self.setup_run(runner, method, Path(tmp))
                    calls = []

                    def execute(request):
                        if self.target(request, method):
                            attempt = request.get("retry_attempt", 0)
                            if attempt == interrupt_attempt:
                                raise Interrupted()
                            if attempt < 2:
                                result = fake(request)
                                result["message"] = "\n" * 1024
                                result["completion_tokens"] = 1024
                                result["hit_token_cap"] = request["max_new_tokens"] == 1024
                                calls.append(copy.deepcopy(request))
                                return result
                        elif interrupt_attempt == 3 and calls and calls[-1].get("retry_attempt") == 2:
                            raise Interrupted()
                        calls.append(copy.deepcopy(request))
                        return fake(request)

                    with self.assertRaises(Interrupted):
                        runner.generate_selected(args, manifest, rows, execute)
                    saved = base.load_jsonl(args.output_dir / "events.jsonl")
                    saved_requests = [entry["event"]["request"] for entry in saved]
                    fresh = []

                    def resume(request):
                        self.assertNotIn(request, saved_requests)
                        fresh.append(copy.deepcopy(request))
                        result = fake(request)
                        if self.target(request, method) and request.get("retry_attempt", 0) < 2:
                            result.update(message="\n" * 1024, completion_tokens=1024,
                                          hit_token_cap=request["max_new_tokens"] == 1024)
                        return result

                    self.assertEqual(runner.generate_selected(args, manifest, rows, resume), 0)
                    validate(args.output_dir)
                    journal = base.load_jsonl(args.output_dir / "events.jsonl")
                    self.assertEqual(journal[:len(saved)], saved)
                    record = base.load_jsonl(args.output_dir / "turns.jsonl")[0]
                    events = record[method]["events"]
                    generated = [e for e in events if e["request"]["kind"] == "generate"]
                    self.assertEqual(record["completion_tokens"], sum(e["result"]["completion_tokens"] for e in generated))
                    self.assertEqual(record["prompt_tokens"], sum(e["result"]["prompt_tokens"] for e in generated))
                    self.assertGreaterEqual(sum(bool(e.get("error")) for e in events), 2)
                    count = len(record[method]["responses" if method == "dcgs" else "candidates"])
                    self.assertEqual(count, 5 if method == "dcgs" else 15)
                    before = {n: base.file_sha256(args.output_dir / n) for n in ("events.jsonl", "turns.jsonl", "answers.jsonl")}
                    self.assertEqual(runner.generate_selected(args, manifest, rows, lambda q: self.fail("No-op generated")), 0)
                    self.assertEqual(before, {n: base.file_sha256(args.output_dir / n) for n in before})

    def test_exhaustion_and_explicit_retry_do_not_reset_budget(self):
        for method, algorithm, runner, fake, validate in self.cases():
            with self.subTest(method=method), tempfile.TemporaryDirectory() as tmp:
                args, manifest, rows = self.setup_run(runner, method, Path(tmp))
                failed = []

                def execute(request):
                    result = fake(request)
                    if self.target(request, method):
                        failed.append(copy.deepcopy(request))
                        result["message"] = ""
                    return result

                self.assertEqual(runner.generate_selected(args, manifest, rows, execute), 2)
                self.assertEqual(len(failed), 3)
                self.assertEqual(len({q["seed"] for q in failed}), 3)
                self.assertEqual([q.get("retry_attempt", 0) for q in failed], [0, 1, 2])
                for q in failed[1:]:
                    comparable = {k: v for k, v in q.items() if k not in ("seed", "original_seed", "retry_attempt")}
                    self.assertEqual(comparable, {k: v for k, v in failed[0].items() if k != "seed"})
                args.retry_errors = True
                self.assertEqual(runner.generate_selected(args, manifest, rows, lambda q: self.fail("Budget reset")), 2)

    def test_only_empty_valid_sampled_results_are_retried(self):
        request = {"kind": "generate", "stage": "response", "temperature": 0.7,
                   "max_new_tokens": 10, "seed": 3}
        result = {"message": "\n", "completion_tokens": 1, "prompt_tokens": 5,
                  "original_prompt_tokens": 5, "input_truncated": False, "latency_seconds": 0.1}
        self.assertTrue(retry.empty_generation(request, result))
        for field, value in [("message", None), ("message", "ERROR"), ("message", "missing tags"),
                             ("error", "OOM"), ("input_truncated", True), ("completion_tokens", 0),
                             ("completion_tokens", 11), ("latency_seconds", float("nan")),
                             ("original_prompt_tokens", 6)]:
            with self.subTest(field=field, value=value):
                self.assertFalse(retry.empty_generation(request, {**result, field: value}))
        self.assertFalse(retry.empty_generation({**request, "temperature": 0}, result))
        self.assertFalse(retry.empty_generation({**request, "kind": "reward"}, result))
        for value in (-1, 1, 3, True):
            with self.assertRaises(ValueError):
                retry.retry_policy(value)

    def test_seed_collision_is_resolved(self):
        original = {"seed": 0}
        with patch.object(retry, "stable_id", return_value="0000000000000000"):
            self.assertEqual([retry.retry_request(original, i)["seed"] for i in range(3)], [0, 1, 2])

    def test_token_diagnostics_preserve_whitespace_and_actual_stop_reason(self):
        from types import SimpleNamespace
        class Tokens(list):
            def tolist(self):
                return list(self)
        tokenizer = SimpleNamespace(eos_token_id=2, decode=lambda tokens, **kw: "" if tokens == [2] else "\n" * len(tokens))
        for tokens, cap, reason in [(Tokens([2]), 2048, "eos"), (Tokens([13] * 1024), 1024, "max_new_tokens")]:
            diagnostics = retry.generation_diagnostics(tokenizer, tokens, cap)
            self.assertEqual(diagnostics["stop_reason"], reason)
            result = {**diagnostics, "completion_tokens": len(tokens), "message": diagnostics["raw_decoded_text"]}
            retry.validate_diagnostics({"max_new_tokens": cap}, result)
            retry.validate_diagnostics({"max_new_tokens": cap}, {**result, "message": result["message"].strip()}, strip_output=True)
            with self.assertRaisesRegex(ValueError, "diagnostics"):
                retry.validate_diagnostics({"max_new_tokens": cap}, {**result, "completion_tokens": len(tokens) + 1})

    def test_zero_retry_policy_preserves_legacy_failure(self):
        for method, algorithm, runner, fake, validate in self.cases():
            calls = []
            def execute(request):
                calls.append(request)
                result = fake(request)
                result["message"] = ""
                return result
            run = algorithm.generate_dcgs if method == "dcgs" else algorithm.generate_tpo
            error = algorithm.DCGSFailure if method == "dcgs" else algorithm.TPOFailure
            with self.assertRaises(error):
                run([{"role": "user", "content": "x"}], algorithm.defense_config(empty_retries=0), 0, execute)
            self.assertEqual(len(calls), 1)

    def test_runtime_errors_do_not_retry(self):
        for method, algorithm, runner, fake, validate in self.cases():
            calls = []
            def broken(request):
                calls.append(request)
                raise RuntimeError("CUDA out of memory")
            run = algorithm.generate_dcgs if method == "dcgs" else algorithm.generate_tpo
            error = algorithm.DCGSFailure if method == "dcgs" else algorithm.TPOFailure
            with self.assertRaises(error):
                run([{"role": "user", "content": "x"}], algorithm.defense_config(), 0, broken)
            self.assertEqual(len(calls), 1)

    def test_retry_audit_rejects_changed_error_seed_or_cost(self):
        for method, algorithm, runner, fake, validate in self.cases():
            def execute(request):
                result = fake(request)
                if self.target(request, method) and not request.get("retry_attempt"):
                    result["message"] = ""
                return result
            messages = [{"role": "user", "content": "x"}]
            run = algorithm.generate_dcgs if method == "dcgs" else algorithm.generate_tpo
            result = run(messages, algorithm.defense_config(), 0, execute)
            record = {**result, "prompt_history": messages, "seed": 0, "generated_response": result["message"]}
            algorithm.validate_audit(record, algorithm.defense_config())
            def altered_error(r):
                next(e for e in r[method]["events"] if e.get("error"))["error"] = "ValueError: unrelated"
            def altered_seed(r):
                next(e for e in r[method]["events"] if e["request"].get("retry_attempt"))["request"]["seed"] += 1
            for mutate in (altered_error, altered_seed, lambda r: r.update(completion_tokens=0)):
                changed = copy.deepcopy(record)
                mutate(changed)
                with self.assertRaises((ValueError, RuntimeError)):
                    algorithm.validate_audit(changed, algorithm.defense_config())
