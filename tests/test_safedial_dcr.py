import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import prepare_safedial_dcr as artifacts
import run_safedial_dcr as dcr


class DCRProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast

        backend = Tokenizer(models.WordLevel({"[UNK]": 0, "[EOS]": 1, "USER": 2,
                            "ASSISTANT": 3, ":": 4, "first": 5, "second": 6}, unk_token="[UNK]"))
        backend.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]",
                                            eos_token="[EOS]", pad_token="[EOS]")
        tokenizer.chat_template = (
            "{% for message in messages %}{{ message['role'] | upper }}: {{ message['content'] }}\n"
            "{% endfor %}{% if add_generation_prompt %}ASSISTANT: {% endif %}"
        )
        directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(directory.cleanup)
        tokenizer.save_pretrained(directory.name)
        cls.tokenizer = artifacts.load_tokenizer({"adapter_path": directory.name})

    def test_validate_only_does_not_load_model_or_create_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps({"max_position_embeddings": 512, "vocab_size": 32}))
            dataset = root / "data.jsonl"
            dataset.write_text(json.dumps({"id": 1, "task": "Ethics", "method": "fixture", "scene": "fixture",
                                         "history": [{"user": "first", "bot": "hidden"}]}) + "\n")
            with patch.object(dcr, "validate_lock", return_value={"base_path": str(root)}), \
                 patch.object(dcr, "load_tokenizer", return_value=self.tokenizer), \
                 patch("transformers.AutoModelForCausalLM.from_pretrained") as load:
                self.assertEqual(dcr.main(["--validate-only", "--dataset", str(dataset),
                                           "--output-dir", str(root / "out"), "--max-new-tokens", "8"]), 0)
                load.assert_not_called()
                self.assertFalse((root / "out").exists())

    def test_training_template_preserves_roles_and_withholds_current_reference(self):
        history = [{"user": "first", "bot": "reference one"},
                   {"user": "second", "bot": "SECRET CURRENT REFERENCE"},
                   {"user": "SECRET FUTURE USER", "bot": "future"}]
        messages, rendered = dcr.render_prompt(self.tokenizer, history, 1)
        self.assertEqual([m["role"] for m in messages], ["user", "assistant", "user"])
        self.assertEqual(rendered, "USER: first\nASSISTANT: reference one\nUSER: second\nASSISTANT: ")
        self.assertNotIn("SECRET", rendered)

    def test_preflight_rejects_overflow_instead_of_truncating(self):
        row = {"id": 1, "history": [{"user": "word " * 40, "bot": "hidden"}]}
        config = {"max_position_embeddings": 24, "vocab_size": 151936}
        with self.assertRaisesRegex(ValueError, "context overflow"):
            dcr.preflight([row], self.tokenizer, config, 8)

    def test_end_to_end_saved_prompts_audit_resume_and_failed_retry(self):
        # Inject only model computation; run real template/history, persistence,
        # provenance, validation, retry and resume paths on a two-turn dataset.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset.jsonl"
            rows = [{"id": 1, "task": "Ethics", "method": "fixture", "scene": "fixture",
                     "history": [{"user": "first", "bot": "GOLD ONE"},
                                 {"user": "second", "bot": "GOLD TWO"}]}]
            dataset.write_text(json.dumps(rows[0]) + "\n")
            args = dcr.parse_args(["--dataset", str(dataset), "--output-dir", str(root / "out"),
                                   "--device", "cpu", "--max-new-tokens", "8"])
            args.output_dir.mkdir()
            tokenizer = self.tokenizer
            checked = dcr.preflight(rows, tokenizer, {"max_position_embeddings": 512, "vocab_size": 151936}, 8)
            lock = {"fixture": "local"}
            captured = []

            def generate(**kwargs):
                captured.append(copy.deepcopy(kwargs["messages"]))
                self.assertEqual(kwargs["args"].temperature, 0.0)
                text = tokenizer.apply_chat_template(kwargs["messages"], tokenize=False, add_generation_prompt=True)
                count = len(tokenizer(text, add_special_tokens=False)["input_ids"])
                return {"message": "GENERATED", "prompt_tokens": count, "original_prompt_tokens": count,
                        "completion_tokens": 1, "latency_seconds": 0.01, "input_truncated": False}

            from types import SimpleNamespace
            model = SimpleNamespace(generation_config=SimpleNamespace())
            lock["base_path"] = str(root)
            with patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=model) as load, \
                 patch.object(dcr, "attach_adapter", side_effect=lambda m, _: m), \
                 patch.object(dcr, "verify_adapter_effect", return_value={"passed": True, "max_abs_logit_difference": 0.5}), \
                 patch.object(dcr, "validate_lock_data", side_effect=lambda value: value), \
                 patch.object(dcr, "load_tokenizer", return_value=tokenizer), \
                 patch.object(dcr.baseline, "generate_one", side_effect=generate):
                self.assertEqual(dcr.run_generation(args, rows, lock, tokenizer, checked), 0)
                self.assertEqual(captured[1][1]["content"], "GOLD ONE")
                self.assertNotIn("GENERATED", str(captured[1]))
                self.assertTrue(dcr.audit_run(args.output_dir)["training_template_verified"])
                with self.assertRaisesRegex(ValueError, "GPU memory"):
                    dcr.audit_run(args.output_dir, require_gpu=True)
                before = {p.name: p.read_bytes() for p in args.output_dir.iterdir()}
                self.assertEqual(dcr.run_generation(args, rows, lock, tokenizer, checked), 0)
                self.assertEqual(before, {p.name: p.read_bytes() for p in args.output_dir.iterdir()})
                self.assertEqual(load.call_count, 1)
                args.max_new_tokens = 9
                with self.assertRaisesRegex(RuntimeError, "different run"):
                    dcr.run_generation(args, rows, lock, tokenizer, checked)
                args.max_new_tokens = 8
                turns_path = args.output_dir / "turns.jsonl"
                original = turns_path.read_text()
                records = dcr.baseline.load_jsonl(turns_path)
                records[1]["rendered_prompt_sha256"] = "tampered"
                turns_path.write_text("".join(json.dumps(r) + "\n" for r in records))
                with self.assertRaisesRegex(ValueError, "prompt audit"):
                    dcr.audit_run(args.output_dir)
                turns_path.write_text(original)
                # Fresh directory: first generation fails, successful retry keeps
                # the original failure evidence in the append-only attempt journal.
                args.output_dir = root / "retry"
                args.output_dir.mkdir()
                with patch.object(dcr.baseline, "generate_one", side_effect=[ValueError("injected failure"),
                           generate(messages=dcr.baseline.gold_messages(rows[0]["history"], 1), args=args)]):
                    with self.assertRaises(ValueError):
                        dcr.run_generation(args, rows, lock, tokenizer, checked)
                args.retry_errors = True
                self.assertEqual(dcr.run_generation(args, rows, lock, tokenizer, checked), 0)
                attempts = dcr.baseline.load_jsonl(args.output_dir / "attempts.jsonl")
                self.assertEqual(len(attempts), 4)
                self.assertIn("injected failure", attempts[0]["error"])

    def test_output_directory_cannot_have_two_writers(self):
        with tempfile.TemporaryDirectory() as directory:
            with dcr.output_lock(Path(directory)):
                with self.assertRaisesRegex(RuntimeError, "Another DCR process"):
                    with dcr.output_lock(Path(directory)):
                        pass


class DCRArtifactTests(unittest.TestCase):
    def test_tokenizer_and_base_tampering_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, adapter = root / "base", root / "adapter"
            base.mkdir()
            adapter.mkdir()
            for name in artifacts.BASE_FILES:
                (base / name).write_text("fixture")
            for name in artifacts.ADAPTER_FILES:
                (adapter / name).write_text("fixture")
            (adapter / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": artifacts.BASE}))
            lock = {"schema_version": 1, "method": "DCR", "base_model": artifacts.BASE,
                    "base_revision": artifacts.BASE_REVISION, "base_path": str(base), "adapter_path": str(adapter),
                    "base_sha256": {n: artifacts.sha256(base / n) for n in artifacts.BASE_FILES},
                    "sha256": {n: artifacts.sha256(adapter / n) for n in artifacts.ADAPTER_FILES}}
            with patch.object(artifacts, "BASE_WEIGHTS_SHA256", lock["base_sha256"]["model.safetensors"]):
                artifacts.validate_lock_data(lock)
                (adapter / "chat_template.jinja").write_text("changed")
                with self.assertRaisesRegex(ValueError, "hash mismatch: chat_template"):
                    artifacts.validate_lock_data(lock)
                (adapter / "chat_template.jinja").write_text("fixture")
                (base / "config.json").write_text("changed")
                with self.assertRaisesRegex(ValueError, "hash mismatch: config"):
                    artifacts.validate_lock_data(lock)

    def test_qwen_cpu_lora_loading_is_frozen_and_changes_logits(self):
        import torch
        from transformers import Qwen2Config, Qwen2ForCausalLM
        from peft import LoraConfig, get_peft_model

        torch.set_num_threads(2)
        torch.manual_seed(3)
        config = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2)
        base = Qwen2ForCausalLM(config)
        state = {k: v.clone() for k, v in base.state_dict().items()}
        adapted = get_peft_model(base, LoraConfig(r=8, lora_alpha=32, lora_dropout=0.05,
                    target_modules=["q_proj", "k_proj", "v_proj"], task_type="CAUSAL_LM"))
        with torch.no_grad():
            for name, parameter in adapted.named_parameters():
                if "lora_B" in name:
                    parameter.fill_(0.125)
        with tempfile.TemporaryDirectory() as directory:
            adapted.save_pretrained(directory)
            fresh = Qwen2ForCausalLM(config)
            fresh.load_state_dict(state)
            loaded = dcr.attach_adapter(fresh, {"adapter_path": directory})
            self.assertFalse(any(p.requires_grad for p in loaded.parameters()))
            class FixtureTokenizer:
                def apply_chat_template(self, *args, **kwargs):
                    return "USER: Hello.\nASSISTANT: "
                def __call__(self, *args, **kwargs):
                    return {"input_ids": torch.tensor([[1, 2, 3]]),
                            "attention_mask": torch.ones(1, 3, dtype=torch.long)}
            self.assertTrue(dcr.verify_adapter_effect(loaded, FixtureTokenizer(), torch)["passed"])


if __name__ == "__main__":
    unittest.main()
