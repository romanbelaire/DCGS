import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_safedial_baseline as base
import run_safedial_tpo_full as full
from test_safedial_tpo import fake_execute
from validate_safedial_tpo import validate_tpo
from validate_safedial_tpo_full import validate_full


class FullTPOTests(unittest.TestCase):
    def test_native_context_bound_and_tokenizer_metadata_override(self):
        self.assertEqual(full.reward_context_limit(8192), 8192)
        for bad in (4096, None, -1, "8192"):
            with self.assertRaises(ValueError): full.reward_context_limit(bad)
        class Tokenizer:
            model_max_length = 4096
            chat_template = "fixture"
            pad_token_id = 0
            def apply_chat_template(self, *args, **kwargs): return "fixture history"
            def encode(self, *args, **kwargs): return [1] * 5000
        row = {"id": 1, "history": [{"user": "x", "bot": "y"}]}
        lock = {kind: {"path": kind} for kind in ("actor", "reward")}
        with patch("transformers.AutoTokenizer.from_pretrained", side_effect=lambda path, **kwargs: Tokenizer()), \
             patch("transformers.AutoConfig.from_pretrained", return_value=SimpleNamespace(max_position_embeddings=8192)):
            # Actor metadata stays conservative: use a fixture with sufficient actor metadata.
            def tokenizer(path, **kwargs):
                tok = Tokenizer()
                if path == "actor": tok.model_max_length = 8192
                return tok
            with patch("transformers.AutoTokenizer.from_pretrained", side_effect=tokenizer):
                toks, limits, report = full.tokenizers_and_preflight(lock, [row], {"response_tokens": 1024})
        self.assertEqual(limits["reward"], 8192)
        self.assertEqual(report["declared_tokenizer_max_tokens"]["reward"], 4096)
        self.assertEqual(report["gold_history_max_tokens"]["reward"], 5000)
        self.assertEqual(toks["reward"].model_max_length, 8192)

    def test_probe_real_tiny_cpu_model_and_config_bound(self):
        import torch
        from transformers import LlamaConfig, LlamaForSequenceClassification
        torch.set_num_threads(2)
        model = LlamaForSequenceClassification(LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                    num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, num_labels=1,
                    pad_token_id=0, max_position_embeddings=32)).eval()
        backend = SimpleNamespace(torch=torch, models={"reward": model}, args=SimpleNamespace(device="cpu"),
                                  tokenizers={"reward": SimpleNamespace(encode=lambda *a, **kw: [3])})
        report = full.probe_reward_context(backend, 32)
        self.assertTrue(report["passed"] and report["not_a_benchmark_reward"])
        self.assertEqual(report["sequence_tokens"], 32)
        with self.assertRaises(ValueError): full.probe_reward_context(backend, 33)

    def run_fixture(self, root, probe_failure=False):
        data = root / "data.jsonl"
        rows = [{"id": 1, "task": "Ethics", "method": "fixture", "scene": "fixture",
                 "history": [{"user": "test", "bot": "reference"}]}]
        base.append_jsonl(data, rows)
        args = full.parse_args(["--dataset", str(data), "--output-dir", str(root / "out"), "--device", "cpu"])
        backend = MagicMock(side_effect=fake_execute)
        backend.loading = {k: {"passed": True, "trainable_parameters": 0} for k in ("actor", "reward")}
        with patch.object(full, "parse_args", return_value=args), \
             patch.object(full, "validate_lock", return_value={}), \
             patch.object(full, "tokenizers_and_preflight", return_value=({}, {}, {})), \
             patch.object(full, "LocalBackend", return_value=backend), \
             patch.object(full, "probe_reward_context") as probe:
            if probe_failure: probe.side_effect = RuntimeError("fixture context probe failed")
            else: probe.return_value = {"passed": True, "sequence_tokens": 8192}
            status = full.main()
        return args.output_dir, status, backend

    def test_full_manifest_audit_and_separate_probe_costs(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder, status, backend = self.run_fixture(Path(tmp))
            self.assertEqual(status, 0)
            report = validate_full(folder)
            self.assertEqual(report["candidates"], 15)
            with self.assertRaisesRegex(ValueError, "source hashes"):
                validate_tpo(folder)  # Smoke validator must not silently accept full policy.
            stats = json.loads((folder / "runtime_stats.json").read_text())
            self.assertEqual(stats["actual_calls"], {"generate": 19, "reward": 15})
            self.assertEqual(stats["startup_reward_probe_calls"], 1)
            m = json.loads((folder / "run_config.json").read_text())
            m["full_run_context"]["reward_context_tokens"] = 16384
            (folder / "run_config.json").write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError, "context policy"): validate_full(folder)

    def test_failed_probe_stops_before_any_benchmark_model_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder, status, backend = self.run_fixture(Path(tmp), probe_failure=True)
            self.assertEqual(status, 2)
            backend.assert_not_called()
            probe = json.loads((folder / "context_probe.json").read_text())
            self.assertFalse(probe["passed"])
            self.assertIn("fixture context probe failed", probe["error"])
            stats = json.loads((folder / "runtime_stats.json").read_text())
            self.assertEqual(stats["actual_calls"], {"generate": 0, "reward": 0})
            self.assertEqual(stats["startup_reward_probe_calls"], 1)
            self.assertEqual(base.load_jsonl(folder / "answers.jsonl"), [])


if __name__ == "__main__":
    unittest.main()
