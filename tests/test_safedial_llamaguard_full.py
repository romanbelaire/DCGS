import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import run_safedial_llamaguard_full as full


class Tokenizer:
    chat_template = "fixture-template"

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": False, "add_generation_prompt": True}
        return "safety taxonomy " + json.dumps(messages, ensure_ascii=False) + full.guard.ASSISTANT_HEADER

    def __call__(self, text, **kwargs):
        assert kwargs == {"add_special_tokens": False, "truncation": False}
        return {"input_ids": [1] * len(text)}


class FullRunTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.run, self.out = self.root / "run", self.root / "out"
        self.run.mkdir()
        self.dataset = self.root / "data.jsonl"
        self.data = [{"id": 1, "task": "Ethics", "method": "fixture", "history": [
            {"user": "first", "bot": "gold prefix " * 200},
            {"user": "followup", "bot": "excluded current gold"}]},
            {"id": 2, "task": "Privacy", "method": "fixture", "history": [
                {"user": "other", "bot": "gold"}]}]
        self.dataset.write_text("".join(json.dumps(r) + "\n" for r in self.data))
        self.config = {"dataset": str(self.dataset), "dataset_sha256": full.guard.file_hash(self.dataset),
                       "selected_ids": [1, 2], "model_id": "fixture"}
        self.write_config()
        self.records = [{"dialogue_id": r["id"], "turn_index": t, "model_id": "fixture",
                         "generated_response": "complete response " * 100,
                         "prompt_history": full.guard.gold_history(r, t)}
                        for r in self.data for t in range(len(r["history"]))]
        self.write_turns()
        self.argv = ["--run-dir", str(self.run), "--output-dir", str(self.out), "--device", "cpu"]
        self.metadata = {"model_context_limit": 16000, "model_config_sha256": "fixture-config",
                         "tokenizer_template_sha256": full.guard.digest(Tokenizer.chat_template)}

    def write_config(self):
        (self.run / "run_config.json").write_text(json.dumps(self.config))

    def write_turns(self):
        (self.run / "turns.jsonl").write_text("".join(json.dumps(r) + "\n" for r in self.records))

    def test_prepare_is_full_scope_and_loads_no_tokenizer_or_weights(self):
        with patch.object(full, "load_tokenizer", side_effect=AssertionError("must not load")), \
                patch.object(full.guard, "LlamaGuard", side_effect=AssertionError("must not load")):
            self.assertEqual(full.main(self.argv + ["--prepare-only"]), 0)
        m = json.loads((self.out / "full_run_manifest.json").read_text())
        self.assertEqual(m["expected_turns"], 3)
        self.assertTrue(m["generation_complete"])

    def test_rejects_subset_even_if_all_selected_outputs_exist(self):
        self.config["selected_ids"] = [1]
        self.write_config()
        with self.assertRaisesRegex(ValueError, "every dataset dialogue"):
            full.main(self.argv + ["--prepare-only"])

    def test_missing_generation_stops_before_tokenizer_and_can_be_explicit(self):
        self.records.pop()
        self.write_turns()
        with patch.object(full, "load_tokenizer", side_effect=AssertionError("must not load")):
            self.assertEqual(full.main(self.argv), 2)
        with self.assertRaisesRegex(ValueError, "Full-run configuration"):
            full.main(self.argv + ["--prepare-only", "--allow-incomplete-generation"])
        other = self.argv[:]
        other[3] = str(self.root / "partial")
        self.assertEqual(full.main(other + ["--prepare-only", "--allow-incomplete-generation"]), 0)
        a = json.loads((self.root / "partial" / "aggregate.json").read_text())
        self.assertEqual(a["turn_counts"]["generation_missing"], 1)

    def test_preflight_counts_complete_prefix_and_response_without_loading_weights(self):
        with patch.object(full, "load_tokenizer", return_value=(Tokenizer(), self.metadata)), \
                patch.object(full.guard, "LlamaGuard", side_effect=AssertionError("must not load")):
            self.assertEqual(full.main(self.argv + ["--preflight-only"]), 0)
        rows = json.loads((self.out / "context_lengths.json").read_text())
        expected = full.guard.gold_history(self.data[0], 1) + [
            {"role": "assistant", "content": self.records[1]["generated_response"]}]
        rendered = full.guard.render_guard_prompt(Tokenizer(), expected)
        self.assertEqual(rows[1]["input_tokens"], len(rendered))
        self.assertGreater(rows[1]["input_tokens"], 3000)
        a = json.loads((self.out / "context_preflight.json").read_text())
        self.assertEqual(a["effective_context_limit"], 16000)
        self.assertTrue(a["passed"])

    def test_overflow_aborts_all_inference_and_reports_turn_ids(self):
        with patch.object(full, "load_tokenizer", return_value=(Tokenizer(), self.metadata)), \
                patch.object(full.guard, "LlamaGuard", side_effect=AssertionError("must not load")):
            self.assertEqual(full.main(self.argv + ["--max-input-tokens", "2000"]), 2)
        a = json.loads((self.out / "context_preflight.json").read_text())
        self.assertFalse(a["passed"])
        self.assertIn((1, 1), [(r["dialogue_id"], r["turn_index"]) for r in a["overflow_turns"]])
        self.assertFalse((self.out / "judgments.jsonl").exists())

    def test_judging_resumes_without_reloading_and_rejects_config_drift(self):
        metadata = self.metadata

        class Scorer:
            def __init__(self, args, folder):
                self.limit = min(args.max_input_tokens, metadata["model_context_limit"])
                full.guard.atomic_json(folder / "runtime.json", metadata)

            def score(self, messages):
                return {**full.guard.decision(3., 1.), "safe_logit": 3., "unsafe_logit": 1.,
                        "top_token_id": 1, "label_token_ids": {"safe": 1, "unsafe": 2}}

        with patch.object(full, "load_tokenizer", return_value=(Tokenizer(), metadata)), \
                patch.object(full.guard, "LlamaGuard", Scorer):
            self.assertEqual(full.main(self.argv), 0)
        journal = (self.out / "judgments.jsonl").read_bytes()
        with patch.object(full, "load_tokenizer", side_effect=AssertionError("must not load")), \
                patch.object(full.guard, "LlamaGuard", side_effect=AssertionError("must not load")):
            self.assertEqual(full.main(self.argv), 0)
        self.assertEqual((self.out / "judgments.jsonl").read_bytes(), journal)
        with self.assertRaisesRegex(ValueError, "configuration/source changed"):
            full.main(self.argv + ["--max-input-tokens", "32768"])

    def test_partial_snapshot_does_not_ingest_new_generation_on_resume(self):
        last = self.records.pop()
        self.write_turns()
        flags = ["--prepare-only", "--allow-incomplete-generation"]
        self.assertEqual(full.main(self.argv + flags), 0)
        self.records.append(last)
        self.write_turns()
        self.assertEqual(full.main(self.argv + flags), 0)
        a = json.loads((self.out / "aggregate.json").read_text())
        self.assertEqual(a["turn_counts"]["generation_missing"], 1)


if __name__ == "__main__":
    unittest.main()
