import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_safedial_dcr as original
import run_safedial_dcr_prompt_ablation as experiment
import test_safedial_dcr as original_tests


class DCRPromptAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        original_tests.DCRProtocolTests.setUpClass.__func__(cls)
        # Match Qwen's tokenizer: the generic fixture otherwise emits token_type_ids.
        cls.tokenizer.model_input_names = ["input_ids", "attention_mask"]

    def test_raw_keeps_reference_history_without_template_or_future_leakage(self):
        history = [{"user": "first", "bot": "GOLD ONE"},
                   {"user": "second", "bot": "SECRET CURRENT"},
                   {"user": "SECRET FUTURE", "bot": "future"}]
        with patch.object(self.tokenizer, "apply_chat_template", side_effect=AssertionError("template called")):
            messages, rendered = experiment.render_prompt(self.tokenizer, history, 1, "raw")
            self.assertEqual(rendered, "first\n\nGOLD ONE\n\nsecond")
            self.assertEqual(messages, original.baseline.gold_messages(history, 1))
            report = experiment.preflight([{"id": 1, "history": history[:2]}], self.tokenizer,
                                         {"max_position_embeddings": 512, "vocab_size": 32}, 8, "raw")
            self.assertEqual(report["prompt_format"], "raw")

    def test_training_render_is_identical_to_original(self):
        history = [{"user": "first", "bot": "GOLD ONE"}, {"user": "second", "bot": "GOLD TWO"}]
        for index in range(2):
            self.assertEqual(experiment.render_prompt(self.tokenizer, history, index, "training"),
                             original.render_prompt(self.tokenizer, history, index))

    def test_raw_generation_never_calls_chat_template_and_retains_eos_id(self):
        import torch
        args = experiment.parse_args(["--prompt-format", "raw", "--output-dir", "unused",
                                      "--device", "cpu", "--max-new-tokens", "8"])
        rendered = "first\n\nsecond"
        tokenizer = self.tokenizer
        captured = {}
        class FixtureModel:
            device = torch.device("cpu")
            def generate(self, **kwargs):
                captured.update(kwargs)
                return torch.cat([kwargs["input_ids"], torch.tensor([[5, tokenizer.eos_token_id]])], dim=1)
        with patch.object(tokenizer, "apply_chat_template", side_effect=AssertionError("template called")):
            result = experiment.generate_rendered(model=FixtureModel(), tokenizer=tokenizer, rendered=rendered,
                        args=args, seed=0, torch_module=torch, input_limit=100)
        self.assertEqual(captured["input_ids"].tolist()[0], tokenizer(rendered, add_special_tokens=False)["input_ids"])
        self.assertFalse(captured["do_sample"])
        self.assertEqual(captured["eos_token_id"], tokenizer.eos_token_id)
        self.assertEqual(result["message"], "first")
        self.assertEqual(result["completion_token_ids"], [5, tokenizer.eos_token_id])
        self.assertEqual(result["completion_tokens"], 2)

    def test_actual_cpu_training_generation_matches_original(self):
        import torch
        from transformers import Qwen2Config, Qwen2ForCausalLM
        torch.set_num_threads(2)
        torch.manual_seed(7)
        model = Qwen2ForCausalLM(Qwen2Config(vocab_size=len(self.tokenizer), hidden_size=16,
                intermediate_size=32, num_hidden_layers=1, num_attention_heads=4,
                num_key_value_heads=2, eos_token_id=self.tokenizer.eos_token_id)).eval()
        args = experiment.parse_args(["--output-dir", "unused", "--device", "cpu", "--max-new-tokens", "4"])
        messages, rendered = experiment.render_prompt(self.tokenizer, [{"user": "first", "bot": "hidden"}], 0, "training")
        old = original.baseline.generate_one(model=model, tokenizer=self.tokenizer, messages=messages,
                    args=args, seed=0, torch_module=torch, input_limit=100)
        new = experiment.generate_rendered(model=model, tokenizer=self.tokenizer, rendered=rendered,
                    args=args, seed=0, torch_module=torch, input_limit=100)
        for key in ("message", "prompt_tokens", "original_prompt_tokens", "completion_tokens", "input_truncated"):
            self.assertEqual(old[key], new[key])

    def test_paired_runs_audit_resume_and_reject_format_or_token_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "data.jsonl"
            rows = [{"id": 2, "task": "Ethics", "method": "fixture", "scene": "fixture",
                     "history": [{"user": "first", "bot": "GOLD ONE"}, {"user": "second", "bot": "GOLD TWO"}]}]
            dataset.write_text(json.dumps(rows[0]) + "\n")
            tokenizer = self.tokenizer
            lock = {"base_path": str(root)}
            captured = []
            model = SimpleNamespace(generation_config=SimpleNamespace())
            def generate(**kwargs):
                captured.append((kwargs["args"].prompt_format, kwargs["seed"], kwargs["rendered"]))
                count = len(tokenizer(kwargs["rendered"], add_special_tokens=False)["input_ids"])
                return {"message": "first", "completion_token_ids": [5, tokenizer.eos_token_id],
                        "completion_tokens": 2, "prompt_tokens": count, "original_prompt_tokens": count,
                        "latency_seconds": 0.01, "input_truncated": False}
            with patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=model) as load, \
                 patch.object(experiment, "attach_adapter", side_effect=lambda m, _: m), \
                 patch.object(experiment, "verify_adapter_effect", return_value={"passed": True, "max_abs_logit_difference": 0.5}), \
                 patch.object(experiment, "validate_lock_data", side_effect=lambda v: v), \
                 patch.object(experiment, "load_tokenizer", return_value=tokenizer), \
                 patch.object(experiment, "generate_rendered", side_effect=generate):
                for mode in ("training", "raw"):
                    args = experiment.parse_args(["--prompt-format", mode, "--output-dir", str(root / mode),
                                                  "--dataset", str(dataset), "--device", "cpu", "--max-new-tokens", "8"])
                    args.output_dir.mkdir()
                    checked = experiment.preflight(rows, tokenizer, {"max_position_embeddings": 512, "vocab_size": 32}, 8, mode)
                    self.assertEqual(experiment.run_generation(args, rows, lock, tokenizer, checked), 0)
                    report = experiment.audit_run(args.output_dir)
                    self.assertTrue(report["prompt_format_verified"])
                    self.assertEqual(report["prompt_format"], mode)
                    self.assertTrue(report["turn_metrics"][0]["last_token_is_eos"])
                    before = {p.name: p.read_bytes() for p in args.output_dir.iterdir()}
                    self.assertEqual(experiment.run_generation(args, rows, lock, tokenizer, checked), 0)
                    self.assertEqual(before, {p.name: p.read_bytes() for p in args.output_dir.iterdir()})
                    other = copy.copy(args)
                    other.prompt_format = "raw" if mode == "training" else "training"
                    other.protocol, other.model_id = experiment.identities(other.prompt_format)
                    with self.assertRaisesRegex(RuntimeError, "different run"):
                        experiment.run_generation(other, rows, lock, tokenizer, checked)
                    path = args.output_dir / "turns.jsonl"
                    records = experiment.baseline.load_jsonl(path)
                    records[0]["completion_token_ids"] = [6, tokenizer.eos_token_id]
                    path.write_text("".join(json.dumps(r) + "\n" for r in records))
                    with self.assertRaisesRegex(ValueError, "generated token audit"):
                        experiment.audit_run(args.output_dir)
                    path.write_bytes(before["turns.jsonl"])
                    manifest_path = args.output_dir / "run_config.json"
                    manifest = json.loads(manifest_path.read_text())
                    manifest["prompt_policy"]["serialization"] = "tampered"
                    manifest_path.write_text(json.dumps(manifest))
                    with self.assertRaisesRegex(ValueError, "prompt policy"):
                        experiment.audit_run(args.output_dir)
                self.assertEqual(load.call_count, 2)
            self.assertEqual([c[1] for c in captured[:2]], [c[1] for c in captured[2:]])
            self.assertEqual(captured[3][2], "first\n\nGOLD ONE\n\nsecond")
            self.assertEqual(captured[0][1], original.baseline.turn_seed(0, original.MODEL_ID, 2, 0, 0))

    def test_both_preflights_create_no_outputs_and_load_no_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps({"max_position_embeddings": 512, "vocab_size": 32}))
            dataset = root / "data.jsonl"
            dataset.write_text(json.dumps({"id": 2, "task": "Ethics", "method": "fixture", "scene": "fixture",
                                         "history": [{"user": "first", "bot": "hidden"}]}) + "\n")
            with patch.object(experiment, "validate_lock", return_value={"base_path": str(root)}), \
                 patch.object(experiment, "load_tokenizer", return_value=self.tokenizer), \
                 patch("transformers.AutoModelForCausalLM.from_pretrained") as load:
                for mode in ("training", "raw"):
                    self.assertEqual(experiment.main(["--validate-only", "--dataset", str(dataset),
                        "--output-dir", str(root / mode), "--prompt-format", mode, "--max-new-tokens", "8"]), 0)
                    self.assertFalse((root / mode).exists())
                load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
