import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_safedial_dcr_prompt_ablation as previous
import run_safedial_dcr_zephyr_prompt as zephyr
import test_safedial_dcr_prompt_ablation as prior_tests


class DCRZephyrPromptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        prior_tests.DCRPromptAblationTests.setUpClass.__func__(cls)

    def test_exact_zephyr_text_keeps_gold_history_without_reference_leakage(self):
        history = [{"user": "first", "bot": "GOLD ONE"},
                   {"user": "second", "bot": "SECRET CURRENT"},
                   {"user": "SECRET FUTURE", "bot": "future"}]
        with patch.object(self.tokenizer, "apply_chat_template", side_effect=AssertionError("wrong template")):
            messages, rendered = zephyr.render_prompt(self.tokenizer, history, 1, "zephyr")
        self.assertEqual(rendered, "<|user|>\nfirst</s>\n<|assistant|>\nGOLD ONE</s>\n"
                                   "<|user|>\nsecond</s>\n<|assistant|>\n")
        self.assertEqual([m["role"] for m in messages], ["user", "assistant", "user"])
        self.assertNotIn("SECRET", rendered)
        self.assertEqual(self.tokenizer.eos_token, "[EOS]")

    def test_private_runtime_does_not_mutate_existing_modes(self):
        original_render, original_policy = previous.render_prompt, previous.prompt_policy
        runtime, other = zephyr.build_runtime(), zephyr.build_runtime()
        self.assertIsNot(runtime, other)
        self.assertIs(previous.render_prompt, original_render)
        self.assertIs(previous.prompt_policy, original_policy)
        self.assertEqual(previous.identities("raw")[0], "safedial_dcr_prompt_ablation_raw_v1")
        with self.assertRaises(ValueError):
            previous.identities("zephyr")
        self.assertEqual(runtime.identities("zephyr"), (zephyr.PROTOCOL, zephyr.MODEL_ID))
        with self.assertRaises(ValueError):
            runtime.identities("training")
        hashes = runtime.source_hashes()
        self.assertIn("run_safedial_dcr_zephyr_prompt.py", hashes)
        self.assertIn("run_safedial_dcr_prompt_ablation.py", hashes)
        self.assertIn("run_safedial_dcr.py", hashes)

    def test_validate_only_no_model_no_output_and_marker_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps({"max_position_embeddings": 512, "vocab_size": 32}))
            dataset = root / "data.jsonl"
            dataset.write_text(json.dumps({"id": 1, "task": "Ethics", "method": "fixture", "scene": "fixture",
                "history": [{"user": "first", "bot": "hidden"}]}) + "\n")
            runtime = zephyr.build_runtime()
            with patch.object(runtime, "validate_lock", return_value={"base_path": str(root)}), \
                 patch.object(runtime, "load_tokenizer", return_value=self.tokenizer), \
                 patch("transformers.AutoModelForCausalLM.from_pretrained") as load:
                self.assertEqual(runtime.main(["--validate-only", "--ids", "1", "--dataset", str(dataset),
                    "--output-dir", str(root / "out"), "--max-new-tokens", "8"]), 0)
                load.assert_not_called()
                self.assertFalse((root / "out").exists())
            diagnostics = zephyr.marker_diagnostics(self.tokenizer)
            self.assertFalse(diagnostics["markers"]["</s>"]["registered_special_token"])
            self.assertEqual(diagnostics["generation_eos_token_id"], self.tokenizer.eos_token_id)
            self.assertEqual(diagnostics["added_stop_sequences"], [])

    def test_generation_audit_noop_resume_and_wrong_format_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "data.jsonl"
            rows = [{"id": 1, "task": "Ethics", "method": "fixture", "scene": "fixture",
                     "history": [{"user": "first", "bot": "GOLD ONE"}, {"user": "second", "bot": "hidden"}]}]
            dataset.write_text(json.dumps(rows[0]) + "\n")
            runtime = zephyr.build_runtime()
            args = zephyr.parse_args(["--ids", "1", "--dataset", str(dataset), "--device", "cpu",
                                     "--output-dir", str(root / "out"), "--max-new-tokens", "8"])
            args.output_dir.mkdir()
            tokenizer = self.tokenizer
            captured = []
            def generate(**kwargs):
                captured.append((kwargs["rendered"], kwargs["seed"]))
                count = len(tokenizer(kwargs["rendered"], add_special_tokens=False)["input_ids"])
                return {"message": "first", "completion_token_ids": [5, tokenizer.eos_token_id],
                        "completion_tokens": 2, "prompt_tokens": count, "original_prompt_tokens": count,
                        "input_truncated": False, "latency_seconds": 0.01}
            model = SimpleNamespace(generation_config=SimpleNamespace())
            lock = {"base_path": str(root)}
            checked = runtime.preflight(rows, tokenizer, {"max_position_embeddings": 512, "vocab_size": 32}, 8, "zephyr")
            with patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=model) as load, \
                 patch.object(runtime, "attach_adapter", side_effect=lambda m, _: m), \
                 patch.object(runtime, "verify_adapter_effect", return_value={"passed": True, "max_abs_logit_difference": 0.5}), \
                 patch.object(runtime, "validate_lock_data", side_effect=lambda v: v), \
                 patch.object(runtime, "load_tokenizer", return_value=tokenizer), \
                 patch.object(runtime, "generate_rendered", side_effect=generate):
                self.assertEqual(runtime.run_generation(args, rows, lock, tokenizer, checked), 0)
                report = runtime.audit_run(args.output_dir)
                self.assertEqual(report["prompt_format"], "zephyr")
                self.assertEqual(report["turn_metrics"][0]["generated_zephyr_role_markers"], 0)
                self.assertEqual(captured[1][0], "<|user|>\nfirst</s>\n<|assistant|>\nGOLD ONE</s>\n"
                                              "<|user|>\nsecond</s>\n<|assistant|>\n")
                self.assertEqual(captured[1][1], previous.baseline.turn_seed(0, previous.SEED_MODEL_ID, 1, 0, 1))
                before = {p.name: p.read_bytes() for p in args.output_dir.iterdir()}
                self.assertEqual(runtime.run_generation(args, rows, lock, tokenizer, checked), 0)
                self.assertEqual(before, {p.name: p.read_bytes() for p in args.output_dir.iterdir()})
                self.assertEqual(load.call_count, 1)
                with self.assertRaisesRegex(ValueError, "prompt format"):
                    previous.audit_run(args.output_dir)
                path = args.output_dir / "turns.jsonl"
                records = runtime.baseline.load_jsonl(path)
                records[0]["rendered_prompt"] = "first"
                path.write_text("".join(json.dumps(r) + "\n" for r in records))
                with self.assertRaisesRegex(ValueError, "rendered prompt audit"):
                    runtime.audit_run(args.output_dir)


if __name__ == "__main__":
    unittest.main()
