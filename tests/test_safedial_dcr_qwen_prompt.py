import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_safedial_dcr_prompt_ablation as previous
import run_safedial_dcr_qwen_prompt as qwen
import test_safedial_dcr_prompt_ablation as prior_tests


class DCRQwenPromptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        prior_tests.DCRPromptAblationTests.setUpClass.__func__(cls)

    def test_exact_standard_template_preserves_training_template_and_gold_history(self):
        history = [{"user": "first", "bot": "GOLD ONE"},
                   {"user": "second", "bot": "SECRET CURRENT"},
                   {"user": "SECRET FUTURE", "bot": "future"}]
        before = self.tokenizer.chat_template
        messages, rendered = qwen.render_prompt(self.tokenizer, history, 1, qwen.FORMAT)
        self.assertEqual(rendered, "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                         "<|im_start|>user\nfirst<|im_end|>\n"
                         "<|im_start|>assistant\nGOLD ONE<|im_end|>\n"
                         "<|im_start|>user\nsecond<|im_end|>\n<|im_start|>assistant\n")
        self.assertEqual(messages, previous.baseline.gold_messages(history, 1))
        self.assertNotIn("SECRET", rendered)
        self.assertEqual(self.tokenizer.chat_template, before)
        self.assertEqual(self.tokenizer.eos_token, "[EOS]")

    def test_pinned_template_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text('{"chat_template": "wrong"}')
            with patch.object(qwen, "TEMPLATE_CONFIG", path):
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    qwen.standard_template()

    def test_empty_generation_retains_actual_ids_and_unchanged_eos(self):
        import torch
        tokenizer = self.tokenizer
        captured = {}

        class FixtureModel:
            device = torch.device("cpu")

            def generate(self, **kwargs):
                captured.update(kwargs)
                return torch.cat([kwargs["input_ids"], torch.tensor([[tokenizer.eos_token_id]])], dim=1)

        with tempfile.TemporaryDirectory() as directory:
            args = qwen.parse_args(["--device", "cpu", "--output-dir", directory])
            runtime = qwen.build_runtime()
            result = runtime.generate_rendered(model=FixtureModel(), tokenizer=tokenizer,
                rendered="first", args=args, seed=11, torch_module=torch, input_limit=100)
            self.assertEqual(result["message"], "")
            self.assertEqual(captured["eos_token_id"], tokenizer.eos_token_id)
            self.assertFalse(captured["do_sample"])
            saved = previous.baseline.load_jsonl(Path(directory) / "raw_generations.jsonl")
            self.assertEqual(saved[0]["completion_token_ids"], [tokenizer.eos_token_id])
            self.assertEqual(saved[0]["completion_tokens"], 1)
            self.assertEqual(saved[0]["seed"], 11)

    def test_private_runtime_does_not_mutate_existing_modes(self):
        original_render, original_policy = previous.render_prompt, previous.prompt_policy
        runtime, other = qwen.build_runtime(), qwen.build_runtime()
        self.assertIsNot(runtime, other)
        self.assertIs(previous.render_prompt, original_render)
        self.assertIs(previous.prompt_policy, original_policy)
        self.assertEqual(previous.identities("raw")[0], "safedial_dcr_prompt_ablation_raw_v1")
        with self.assertRaises(ValueError):
            previous.identities("qwen_standard")
        self.assertEqual(runtime.identities("qwen_standard"), (qwen.PROTOCOL, qwen.MODEL_ID))
        with self.assertRaises(ValueError):
            runtime.identities("training")
        hashes = runtime.source_hashes()
        self.assertIn("run_safedial_dcr_qwen_prompt.py", hashes)
        self.assertIn("run_safedial_dcr_prompt_ablation.py", hashes)
        self.assertIn("run_safedial_dcr.py", hashes)

    def test_validate_only_no_model_no_output_and_marker_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps({"max_position_embeddings": 512, "vocab_size": 32}))
            dataset = root / "data.jsonl"
            dataset.write_text(json.dumps({"id": 1, "task": "Ethics", "method": "fixture", "scene": "fixture",
                "history": [{"user": "first", "bot": "hidden"}]}) + "\n")
            runtime = qwen.build_runtime()
            with patch.object(runtime, "validate_lock", return_value={"base_path": str(root)}), \
                 patch.object(runtime, "load_tokenizer", return_value=self.tokenizer), \
                 patch("transformers.AutoModelForCausalLM.from_pretrained") as load:
                self.assertEqual(runtime.main(["--validate-only", "--ids", "1", "--dataset", str(dataset),
                    "--output-dir", str(root / "out"), "--max-new-tokens", "8"]), 0)
                load.assert_not_called()
                self.assertFalse((root / "out").exists())
            diagnostics = qwen.marker_diagnostics(self.tokenizer)
            self.assertFalse(diagnostics["markers"]["<|im_end|>"]["registered_special_token"])
            self.assertEqual(diagnostics["generation_eos_token_id"], self.tokenizer.eos_token_id)
            self.assertEqual(diagnostics["added_stop_sequences"], [])

    def test_generation_audit_noop_resume_and_wrong_format_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "data.jsonl"
            rows = [{"id": 1, "task": "Ethics", "method": "fixture", "scene": "fixture",
                     "history": [{"user": "first", "bot": "GOLD ONE"}, {"user": "second", "bot": "hidden"}]}]
            dataset.write_text(json.dumps(rows[0]) + "\n")
            runtime = qwen.build_runtime()
            args = qwen.parse_args(["--ids", "1", "--dataset", str(dataset), "--device", "cpu",
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
            checked = runtime.preflight(rows, tokenizer, {"max_position_embeddings": 512, "vocab_size": 32}, 8, "qwen_standard")
            with patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=model) as load, \
                 patch.object(runtime, "attach_adapter", side_effect=lambda m, _: m), \
                 patch.object(runtime, "verify_adapter_effect", return_value={"passed": True, "max_abs_logit_difference": 0.5}), \
                 patch.object(runtime, "validate_lock_data", side_effect=lambda v: v), \
                 patch.object(runtime, "load_tokenizer", return_value=tokenizer), \
                 patch.object(runtime, "generate_rendered", side_effect=generate):
                self.assertEqual(runtime.run_generation(args, rows, lock, tokenizer, checked), 0)
                report = runtime.audit_run(args.output_dir)
                self.assertEqual(report["prompt_format"], "qwen_standard")
                self.assertEqual(report["turn_metrics"][0]["generated_qwen_marker_counts"]["<|im_start|>"], 0)
                self.assertEqual(captured[1][0], qwen.render_prompt(tokenizer, rows[0]["history"], 1, qwen.FORMAT)[1])
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
