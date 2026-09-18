import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_safedial_baseline as baseline
import safedial_adapters as adapters
from validate_safedial_generation import validate_run


class CATSetupTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.config = {"base_model_name_or_path": "base", "peft_type": "LORA", "task_type": "CAUSAL_LM"}
        (self.root / "adapter_config.json").write_text(json.dumps(self.config))
        (self.root / "adapter_model.safetensors").write_bytes(b"hash fixture, not loaded")
        self.lock = {"schema_version": 1, "base_model": "base", "base_revision": "pinned",
                     "adapter_path": str(self.root), "sha256": {
                         name: adapters.sha256(self.root / name)
                         for name in ("adapter_config.json", "adapter_model.safetensors")}}
        self.lock_path = self.root / "lock.json"
        self.lock_path.write_text(json.dumps(self.lock))

    def test_lock_validates_and_rejects_wrong_base_or_revision(self):
        self.assertEqual(adapters.validate_adapter_lock(self.lock_path, "base", "pinned"), self.lock)
        with self.assertRaisesRegex(ValueError, "base model/revision"):
            adapters.validate_adapter_lock(self.lock_path, "base", "other")

    def test_modified_weights_rejected(self):
        (self.root / "adapter_model.safetensors").write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            adapters.validate_adapter_lock(self.lock_path, "base", "pinned")

    def test_plain_manifest_schema_unchanged_and_adapter_included_when_requested(self):
        dataset = self.root / "data.jsonl"
        dataset.write_text("{}\n")
        with patch.object(sys, "argv", ["runner"]):
            args = baseline.parse_args()
        plain = baseline.manifest_for(args, dataset, [{"id": 1}])
        self.assertNotIn("adapter", plain)
        self.assertNotIn("verify_adapter", plain)
        args.model = "base"
        args.revision = "pinned"
        args.adapter_lock = self.lock_path
        self.assertEqual(baseline.manifest_for(args, dataset, [{"id": 1}])["adapter"], self.lock)

    def test_verification_flag_requires_adapter(self):
        with patch.object(sys, "argv", ["runner", "--verify-adapter"]):
            args = baseline.parse_args()
        with self.assertRaisesRegex(ValueError, "requires --adapter-lock"):
            baseline.validate_args(args)


class GenerationValidationTest(unittest.TestCase):
    def test_complete_history_and_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            history = [{"user": "u0", "bot": "gold0"}, {"user": "u1", "bot": "gold1"}]
            dataset = root / "dataset.jsonl"
            dataset.write_text(json.dumps({"id": 1, "task": "Ethics", "history": history}) + "\n")
            manifest = {"dataset": str(dataset), "dataset_sha256": baseline.file_sha256(dataset),
                        "selected_ids": [1], "num_choices": 1, "model_id": "cat"}
            (root / "run_config.json").write_text(json.dumps(manifest))
            (root / "answers.jsonl").write_text(json.dumps({"id": 1, "task": "Ethics", "model_id": "cat",
                "choices": [{"turns": [{"message": "a0"}, {"message": "a1"}]}]}) + "\n")
            turns = [{"dialogue_id": 1, "choice_index": 0, "turn_index": i, "model_id": "cat",
                      "generated_response": f"a{i}", "error": None,
                      "prompt_history": baseline.gold_messages(history, i)} for i in range(2)]
            path = root / "turns.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in turns))
            self.assertEqual(validate_run(root)["turns"], 2)
            with self.assertRaisesRegex(ValueError, "Adapter provenance"):
                validate_run(root, require_adapter=True)
            turns[1]["prompt_history"][1]["content"] = "generated answer instead of gold"
            path.write_text("".join(json.dumps(row) + "\n" for row in turns))
            with self.assertRaisesRegex(ValueError, "Gold-history"):
                validate_run(root)


class PeftCPUIntegrationTest(unittest.TestCase):
    def test_local_adapter_is_loaded_frozen_and_changes_logits(self):
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import LlamaConfig, LlamaForCausalLM

        torch.set_num_threads(2)
        torch.manual_seed(4)
        config = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                             num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2)
        base = LlamaForCausalLM(config)
        base_state = {key: value.clone() for key, value in base.state_dict().items()}
        source = get_peft_model(base, LoraConfig(r=2, lora_alpha=2, target_modules=["q_proj"], task_type="CAUSAL_LM"))
        with torch.no_grad():
            for name, parameter in source.named_parameters():
                if "lora_B" in name:
                    parameter.fill_(0.25)
        with tempfile.TemporaryDirectory() as directory:
            source.save_pretrained(directory)
            fresh = LlamaForCausalLM(config)
            fresh.load_state_dict(base_state)
            loaded = adapters.attach_adapter(fresh, {"adapter_path": directory})
            self.assertFalse(any(p.requires_grad for p in loaded.parameters()))

            class FixtureTokenizer:
                def apply_chat_template(self, *args, **kwargs):
                    return "fixture"

                def __call__(self, *args, **kwargs):
                    return {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.ones(1, 3, dtype=torch.long)}

            self.assertTrue(adapters.verify_adapter_effect(loaded, FixtureTokenizer(), torch)["passed"])
            with torch.no_grad():
                for name, parameter in loaded.named_parameters():
                    if "lora_B" in name:
                        parameter.zero_()
            with self.assertRaisesRegex(RuntimeError, "no measurable effect"):
                adapters.verify_adapter_effect(loaded, FixtureTokenizer(), torch)


if __name__ == "__main__":
    unittest.main()
