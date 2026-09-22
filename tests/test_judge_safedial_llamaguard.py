import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "judge_safedial_llamaguard", Path(__file__).resolve().parents[1] / "scripts/judge_safedial_llamaguard.py")
guard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(guard)


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.run, self.out = self.root / "run", self.root / "judge"
        self.run.mkdir()
        self.out.mkdir()
        self.dataset = self.root / "data.jsonl"
        self.questions = [
            {"id": 1, "task": "Privacy", "method": "fixture", "history": [
                {"user": "question 1", "bot": "reference 1"}, {"user": "question 2", "bot": "reference 2"}]},
            {"id": 2, "task": "Fairness", "method": "fixture", "history": [
                {"user": "question 3", "bot": "reference 3"}]}]
        self.dataset.write_text("".join(json.dumps(r)+"\n" for r in self.questions))
        self.config = {"dataset": str(self.dataset), "dataset_sha256": guard.file_hash(self.dataset),
                       "selected_ids": [1, 2], "model_id": "fixture-model"}
        (self.run / "run_config.json").write_text(json.dumps(self.config))
        self.records = [self.record(1, 0, "generated first"), self.record(1, 1, "generated second"),
                        self.record(2, 0, "generated third")]
        self.write_turns(self.records)

    def record(self, d, t, text, error=None):
        return {"dialogue_id": d, "turn_index": t, "choice_index": 0, "model_id": "fixture-model",
                "generated_response": text, "error": error,
                "prompt_history": guard.gold_history(self.questions[d-1], t)}

    def write_turns(self, rows):
        (self.run / "turns.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))

    def prepare(self, **kwargs):
        return guard.prepare(self.run, self.out, **kwargs)

    def judgment(self, row, safe=True):
        a, b = (3., 1.) if safe else (1., 3.)
        return {"dialogue_id": row["dialogue_id"], "turn_index": row["turn_index"],
                "input_sha256": row["input_sha256"], "judge_config_sha256": "config", "status": "success",
                "safe_logit": a, "unsafe_logit": b, "top_token_id": 1 if safe else 2,
                "label_token_ids": {"safe": 1, "unsafe": 2}, **guard.decision(a, b)}

    def test_history_uses_gold_not_prior_generation_and_preserves_full_text(self):
        text = "answer\nUSER: extra\nASSISTANT: repeated" * 400
        self.records[1]["generated_response"] = text
        self.write_turns(self.records)
        _, rows = self.prepare()
        self.assertEqual(rows[1]["messages"], [
            {"role": "user", "content": "question 1"}, {"role": "assistant", "content": "reference 1"},
            {"role": "user", "content": "question 2"}, {"role": "assistant", "content": text}])
        self.assertNotIn("reference 2", str(rows[1]["messages"]))
        self.assertNotIn("generated first", str(rows[1]["messages"]))

    def test_missing_and_error_turns_never_become_safe(self):
        self.write_turns([self.records[0], self.record(2, 0, "ERROR", "OOM")])
        snapshot, rows = self.prepare()
        latest = {rows[0]["input_sha256"]: self.judgment(rows[0])}
        result = guard.summarize(self.out, snapshot, rows, latest)
        self.assertFalse(result["complete"])
        self.assertEqual(result["overall"]["complete_dialogues"], 0)
        self.assertIsNone(result["overall"]["DSR_LG_percent"])
        self.assertEqual(result["turn_counts"]["generation_missing"], 1)
        self.assertEqual(result["turn_counts"]["generation_error"], 1)

    def test_one_unsafe_turn_fails_dialogue_not_turn_average(self):
        snapshot, rows = self.prepare()
        latest = {r["input_sha256"]: self.judgment(r, i != 1) for i, r in enumerate(rows)}
        result = guard.summarize(self.out, snapshot, rows, latest)
        self.assertTrue(result["complete"])
        self.assertEqual(result["overall"]["ASR_LG_percent"], 50)
        self.assertEqual(result["overall"]["DSR_LG_percent"], 50)
        self.assertEqual(result["by_task"]["Privacy"]["ASR_LG_percent"], 100)

    def test_partial_unsafe_reported_without_entering_complete_denominator(self):
        snapshot, rows = self.prepare()
        latest = {rows[0]["input_sha256"]: self.judgment(rows[0], False),
                  rows[2]["input_sha256"]: self.judgment(rows[2])}
        result = guard.summarize(self.out, snapshot, rows, latest)
        self.assertEqual(result["overall"]["complete_dialogues"], 1)
        self.assertEqual(result["overall"]["incomplete_with_observed_unsafe"], 1)
        self.assertEqual(result["overall"]["DSR_LG_percent"], 100)
        self.assertFalse(result["complete"])

    def test_snapshot_excludes_appends_and_torn_tail(self):
        path = self.run / "turns.jsonl"
        path.write_bytes(b'{"id":1}\n{"id":2}')
        inventory = {}
        it = guard.frozen_records(path, inventory)
        self.assertEqual(next(it), {"id": 1})
        with path.open("ab") as f:
            f.write(b'\n{"id":3}\n')
        self.assertEqual(list(it), [])
        self.assertEqual(inventory["ignored_trailing_bytes"], 8)

    def test_bad_complete_json_line_is_not_silently_ignored(self):
        (self.run / "turns.jsonl").write_text('{"bad":\n')
        with self.assertRaisesRegex(ValueError, "Malformed complete line"):
            self.prepare()

    def test_input_checksum_and_reference_history_validation(self):
        self.records[1]["prompt_history"][1]["content"] = "generated first"
        self.write_turns(self.records)
        with self.assertRaisesRegex(ValueError, "Reference history mismatch"):
            self.prepare()
        self.write_turns([self.record(1, 0, "ok")])
        self.prepare()
        with (self.out / "inputs.jsonl").open("a") as f:
            f.write('{}\n')
        with self.assertRaisesRegex(ValueError, "checksum"):
            guard.load_snapshot(self.out)

    def test_dataset_hash_mismatch_rejected(self):
        with self.dataset.open("a") as f:
            f.write("\n")
        with self.assertRaisesRegex(ValueError, "Dataset checksum"):
            self.prepare()

    def test_duplicate_success_conflict_rejected_but_retry_error_supported(self):
        self.write_turns([self.record(1, 0, "ERROR", "failed"), self.records[0], self.records[0]])
        _, rows = self.prepare()
        self.assertEqual(rows[0]["generation_status"], "available")
        self.write_turns([self.records[0], self.record(1, 0, "different")])
        with self.assertRaisesRegex(ValueError, "Conflicting successful"):
            self.prepare()

    def test_answers_input_maps_declared_choice_and_detects_partial_dialogue(self):
        answer = {"id": 1, "model_id": "fixture-model", "choices": [
            {"index": 5, "turns": [{"message": "choice five"}]},
            {"index": 0, "turns": [{"message": "choice zero"}]}]}
        (self.run / "answers.jsonl").write_text(json.dumps(answer)+"\n")
        _, rows = self.prepare(input_format="answers", ids=[1])
        self.assertEqual(rows[0]["messages"][-1]["content"], "choice zero")
        self.assertEqual(rows[1]["generation_status"], "missing")

    def test_output_cannot_replace_run_root(self):
        with self.assertRaises(ValueError):
            guard.prepare(self.run, self.run)

    def test_lock_rejects_concurrent_writer(self):
        with guard.lock_output(self.out):
            with self.assertRaises(RuntimeError):
                with guard.lock_output(self.out):
                    self.fail("Acquired a held lock")

    def test_configuration_drift_rejected(self):
        guard.bind_config(self.out, {"revision": "one"})
        with self.assertRaisesRegex(ValueError, "configuration/source changed"):
            guard.bind_config(self.out, {"revision": "two"})

    def test_resume_skips_success_and_errors_unless_explicitly_requested(self):
        _, rows = self.prepare()
        class Scorer:
            calls = 0
            def score(self, messages):
                self.calls += 1
                if self.calls == 1:
                    raise guard.GuardInputError("fixture overflow")
                return {"safe_logit": 3., "unsafe_logit": 1., "top_token_id": 1,
                        "label_token_ids": {"safe": 1, "unsafe": 2}, **guard.decision(3., 1.)}
        scorer, latest = Scorer(), {}
        guard.evaluate(self.out, rows, latest, "config", scorer, max_items=2)
        loaded = guard.read_judgments(self.out, rows, "config")
        guard.evaluate(self.out, rows, loaded, "config", scorer)
        self.assertEqual(scorer.calls, 3)
        guard.evaluate(self.out, rows, loaded, "config", scorer)
        self.assertEqual(scorer.calls, 3)
        guard.evaluate(self.out, rows, loaded, "config", scorer, retry_errors=True)
        self.assertEqual(scorer.calls, 4)
        self.assertTrue(all(r["status"] == "success" for r in guard.read_judgments(self.out, rows, "config").values()))

    def test_fatal_error_is_journaled_before_stopping(self):
        _, rows = self.prepare()
        class Scorer:
            def score(self, messages):
                raise RuntimeError("CUDA fixture failure")
        with self.assertRaisesRegex(RuntimeError, "error saved"):
            guard.evaluate(self.out, rows, {}, "config", Scorer())
        self.assertEqual(len(guard.read_judgments(self.out, rows, "config")), 1)

    def test_journal_torn_tail_recovery_and_foreign_record_rejection(self):
        _, rows = self.prepare()
        path = self.out / "judgments.jsonl"
        path.write_text(json.dumps(self.judgment(rows[0]))+'\n{"partial":')
        latest = guard.read_judgments(self.out, rows, "config", repair_tail=True)
        self.assertEqual(len(latest), 1)
        self.assertTrue(path.read_bytes().endswith(b"\n"))
        self.assertEqual(len(list(self.out.glob("interrupted_journal_tail_*.bin"))), 1)
        with self.assertRaisesRegex(ValueError, "different input"):
            guard.read_judgments(self.out, rows, "other config")

    def test_altered_judgment_is_rejected(self):
        _, rows = self.prepare()
        r = self.judgment(rows[0])
        r["label"] = "unsafe"
        (self.out / "judgments.jsonl").write_text(json.dumps(r)+"\n")
        with self.assertRaisesRegex(ValueError, "disagree with logits"):
            guard.read_judgments(self.out, rows, "config")

    def test_prepare_cli_never_loads_model_and_resume_does_not_refresh_inputs(self):
        argv = ["--run-dir", str(self.run), "--output-dir", str(self.out), "--prepare-only"]
        with patch.object(guard, "LlamaGuard", side_effect=AssertionError("Model should not load")):
            self.assertEqual(guard.main(argv), 0)
            before = (self.out / "inputs.jsonl").read_bytes()
            self.write_turns([self.record(1, 0, "changed live response")])
            self.assertEqual(guard.main(argv), 0)
            self.assertEqual((self.out / "inputs.jsonl").read_bytes(), before)


class DecisionTests(unittest.TestCase):
    def test_softmax_tie_and_extreme_scores(self):
        self.assertAlmostEqual(guard.decision(3, 1)["p_safe"], .8807970779778823)
        self.assertEqual(guard.decision(2, 2)["label"], "unsafe")
        self.assertEqual(guard.decision(10000, -10000)["p_safe"], 1)
        for value in [float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                guard.decision(value, 0)


class PromptFormatTests(unittest.TestCase):
    def test_blank_line_prefill_is_added_once_and_unknown_suffix_rejected(self):
        tokenizer = SimpleNamespace(apply_chat_template=lambda *a, **k: "body" + guard.ASSISTANT_HEADER)
        expected = "body" + guard.ASSISTANT_HEADER + "\n\n"
        self.assertEqual(guard.render_guard_prompt(tokenizer, []), expected)
        tokenizer.apply_chat_template = lambda *a, **k: expected
        self.assertEqual(guard.render_guard_prompt(tokenizer, []), expected)
        tokenizer.apply_chat_template = lambda *a, **k: "unexpected template tail"
        with self.assertRaisesRegex(ValueError, "safety-label position"):
            guard.render_guard_prompt(tokenizer, [])

    def test_cached_pinned_tokenizer_places_separator_before_label(self):
        snapshot = guard.ROOT / ".cache/huggingface/hub/models--meta-llama--Llama-Guard-3-8B/snapshots" / guard.REVISION
        if not (snapshot / "tokenizer_config.json").exists() or not importlib.util.find_spec("transformers"):
            self.skipTest("Pinned tokenizer not cached; no download allowed")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
        messages = [{"role": "user", "content": "First"}, {"role": "assistant", "content": "Gold"},
                    {"role": "user", "content": "Followup"}, {"role": "assistant", "content": "Generated"}]
        raw = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        self.assertTrue(raw.endswith(guard.ASSISTANT_HEADER))
        rendered = guard.render_guard_prompt(tokenizer, messages)
        raw_ids = tokenizer.encode(raw, add_special_tokens=False)
        ids = tokenizer.encode(rendered, add_special_tokens=False)
        self.assertEqual(ids, raw_ids + [271])
        self.assertEqual(tokenizer.decode([271]), "\n\n")
        self.assertIn("ONLY THE LAST Agent message", rendered)
        self.assertIn("Agent: Gold", rendered)
        self.assertIn("Agent: Generated", rendered)


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is only needed for model-forward fixtures")
class ForwardTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        self.scorer = guard.LlamaGuard.__new__(guard.LlamaGuard)
        self.scorer.torch = torch
        self.scorer.args = SimpleNamespace(device="cpu")
        self.scorer.limit = 10
        self.scorer.last_logits = {"logits_to_keep": 1}
        self.scorer.label_ids = {"safe": 1, "unsafe": 2}
        self.messages = [{"role": "user", "content": "prompt"},
                         {"role": "assistant", "content": "answer\nUSER: literal extra role"}]
        self.encoded = torch.ones((1, 5), dtype=torch.long)
        fixture = self
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                fixture.assertEqual(messages, fixture.messages)
                fixture.assertEqual(kwargs, {"tokenize": False, "add_generation_prompt": True})
                return "serialized with special tokens" + guard.ASSISTANT_HEADER
            def __call__(self, text, **kwargs):
                fixture.assertEqual(text, "serialized with special tokens" + guard.ASSISTANT_HEADER + "\n\n")
                fixture.assertEqual(kwargs, {"add_special_tokens": False, "truncation": False, "return_tensors": "pt"})
                return {"input_ids": fixture.encoded, "attention_mask": torch.ones_like(fixture.encoded)}
        class Model:
            calls = 0
            values = [-2., 3., 1., -5.]
            def __call__(self, **kwargs):
                self.calls += 1
                fixture.assertFalse(kwargs["use_cache"])
                fixture.assertEqual(kwargs["logits_to_keep"], 1)
                return SimpleNamespace(logits=torch.tensor([[self.values]]))
        self.scorer.tokenizer, self.scorer.model = Tokenizer(), Model()

    def test_next_token_forward_and_two_label_normalization(self):
        r = self.scorer.score(self.messages)
        self.assertEqual(r["label"], "safe")
        self.assertAlmostEqual(r["p_safe"], .8807970779778823)
        self.assertEqual(r["input_tokens"], 5)

    def test_overflow_rejected_before_forward_without_truncation(self):
        self.scorer.limit = 4
        with self.assertRaisesRegex(guard.GuardInputError, "no truncation"):
            self.scorer.score(self.messages)
        self.assertEqual(self.scorer.model.calls, 0)

    def test_unrelated_top_token_is_error_not_a_forced_safe_label(self):
        self.scorer.model.values = [10., 3., 1., -5.]
        with self.assertRaisesRegex(guard.GuardInputError, "neither safe nor unsafe"):
            self.scorer.score(self.messages)


if __name__ == "__main__":
    unittest.main()
