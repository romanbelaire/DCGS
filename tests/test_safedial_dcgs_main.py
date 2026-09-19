"""Regression checks for the main-method SafeDial adapter (no model downloads)."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import safedial_dcgs_wildjailbreak as adapter
import run_safedial_dcgs_wildjailbreak as runner
from validate_safedial_dcgs_wildjailbreak import validate
from src.value.value_function import ValueFunction
from src.value import ll_token_critic as ll_module
from test_safedial_dcgs_wildjailbreak import ROW, REPLIES, Tokenizer, execute


class MainMethodTests(unittest.TestCase):
    def test_real_q_and_regret_scoring_uses_last_nonpad_for_both_padding_sides(self):
        critic = ValueFunction.__new__(ValueFunction)
        critic.device, critic.dtype = "cpu", torch.float32
        critic.use_regret_critic = True
        critic.q_mlp_head = critic.q_min_mlp_head = critic.regret_mlp_head = torch.nn.Identity()
        for side, hidden, mask in [
            ("left", [[[100.], [2.], [9.]], [[3.], [4.], [7.]]], [[0, 1, 1], [1, 1, 1]]),
            ("right", [[[2.], [9.], [100.]], [[3.], [4.], [7.]]], [[1, 1, 0], [1, 1, 1]]),
        ]:
            with self.subTest(padding_side=side):
                tok = Mock(return_value={"input_ids": torch.zeros(2, 3, dtype=torch.long),
                                         "attention_mask": torch.tensor(mask)})
                tok.padding_side = side
                critic._forward_body = Mock(return_value=(SimpleNamespace(hidden_states=[torch.tensor(hidden)]), False))
                for method in (critic.predict_q_value, critic.predict_q_min_value, critic.predict_regret_value):
                    self.assertEqual(method(["context"] * 2, ["belief"] * 2, tok).tolist(), [9., 7.])
                self.assertEqual(tok.call_args.kwargs["max_length"], 1500)

    def test_trained_ll_scores_drive_answer_and_gold_history_stays_isolated(self):
        config = adapter.configuration("rdcgs")
        result = adapter.generate_turn(ROW, 1, 42, config, Tokenizer(), execute)
        ll = result["dcgs_original"]["events"][-1]
        self.assertEqual(ll["request"]["kind"], "score_ll")
        self.assertEqual(ll["request"]["actions"], REPLIES)
        self.assertEqual(len(ll["result"]["scores"]), 4)  # Identical candidate strings merge as in main.
        self.assertIn("PRIVATE CURRENT", ll["request"]["observation"])
        self.assertNotIn("PRIVATE NEXT", str(ll["request"]))
        self.assertEqual(result["message"], "A helpful response.")
        self.assertFalse(hasattr(config, "_ll_token_critic"))
        self.assertEqual(ll["result"]["ll_critic_context"]["truncated"], False)

    def test_replay_rejects_ll_score_coverage_and_context_tampering(self):
        config, tok = adapter.configuration("vdcgs"), Tokenizer()
        result = adapter.generate_turn(ROW, 0, 11, config, tok, execute)
        events = result["dcgs_original"]["events"]
        for corruption in ("missing_score", "nan_score", "missing_context", "changed_context"):
            with self.subTest(corruption=corruption):
                altered = copy.deepcopy(events)
                score = altered[-1]["result"]
                if corruption == "missing_score":
                    score["scores"].pop(REPLIES[0])
                elif corruption == "nan_score":
                    score["scores"][REPLIES[0]] = float("nan")
                elif corruption == "missing_context":
                    del score["ll_critic_context"]
                else:
                    score["ll_critic_context"]["encoded_sha256"][0] = "changed"
                with self.assertRaises((ValueError, adapter.PolicyFailure)):
                    adapter.replay_trace(ROW, 0, 11, config, tok, altered)

    def test_replay_preserves_seeded_argmax_ties(self):
        config, tok = adapter.configuration("rdcgs"), Tokenizer()
        def tied(request):
            if request["kind"] == "score_ll":
                return {"scores": {a: 1. for a in request["actions"]}, "objective": "td"}
            return execute(request)
        result = adapter.generate_turn(ROW, 0, 123, config, tok, tied)
        replayed = adapter.replay_trace(ROW, 0, 123, config, tok, result["dcgs_original"]["events"])
        self.assertEqual(replayed, result)

    def test_backend_uses_pinned_trained_ll_critic_and_shared_backbone(self):
        config = adapter.configuration("vdcgs")
        actor = SimpleNamespace(dtype=torch.bfloat16, device=torch.device("cpu"))
        tok = Tokenizer()
        value = SimpleNamespace(q_mlp_head=torch.nn.Linear(2, 1), v_mlp_head=torch.nn.Linear(2, 1),
                                _resolved_hidden_dims=lambda: [2])
        checkpoint = {name + "_mlp_head_state_dict": getattr(value, name + "_mlp_head").state_dict() for name in ("q", "v")}
        ll = SimpleNamespace(objective="shapley", harm_head=torch.nn.Linear(2, 1), follow_head=torch.nn.Linear(2, 1),
                             score_actions=Mock(return_value={"candidate": .75}))
        lock = {"actor": {"path": "local-actor"}, "high_level": {"path": "hl.pt", "sha256": "hl-hash"},
                "token_critic": {"path": "ll.pt", "sha256": "ll-hash"}}
        with patch.object(adapter.legacy, "initialize_dcgs", return_value=(torch, tok, SimpleNamespace(model=actor), None, value)), \
             patch.object(adapter.LLTokenCritic, "from_checkpoint", return_value=ll) as loader, \
             patch.object(torch, "load", return_value=checkpoint):
            backend = adapter.LocalBackend(config, lock)
        self.assertEqual(loader.call_args.args, ("ll.pt",))
        self.assertIs(loader.call_args.kwargs["backbone"], actor)
        self.assertIsNot(loader.call_args.kwargs["tokenizer"], tok)
        result = backend({"kind": "score_ll", "observation": "context", "selected_belief": "belief", "actions": ["candidate"]})
        ll.score_actions.assert_called_once_with("context", "belief", ["candidate"])
        self.assertEqual(result["scores"], {"candidate": .75})
        self.assertEqual(backend.loading["method_spec"]["critic_pooling"], "last_nonpad")
        self.assertTrue(backend.loading["ll_reranking"])
        self.assertEqual(backend.loading["token_critic_sha256"], "ll-hash")
        self.assertFalse(any(p.requires_grad for p in ll.harm_head.parameters()))

    def test_ll_loader_uses_supplied_tokenizer_without_hub_lookup(self):
        actor_tok = SimpleNamespace(pad_token="pad", padding_side="left")
        encoder = torch.nn.Identity()
        head = ll_module.make_head(2, 2.)
        ckpt = {"hidden_size": 2, "mlp_width_mult": 2., "objective": "shapley", "k": 1,
                "harm_head": head.state_dict(), "follow_head": head.state_dict(), "n_examples": 1}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "critic.pt"
            torch.save(ckpt, path)
            with patch.object(ll_module, "HIDDEN_SIZE", 2), \
                 patch.object(ll_module.AutoTokenizer, "from_pretrained", side_effect=AssertionError("Unexpected unpinned tokenizer lookup")):
                critic = ll_module.LLTokenCritic.from_checkpoint(str(path), "cpu", backbone=SimpleNamespace(model=encoder),
                                                                tokenizer=copy.deepcopy(actor_tok))
        self.assertIs(critic.encoder, encoder)
        self.assertEqual(critic.tokenizer.padding_side, "right")
        self.assertEqual(actor_tok.padding_side, "left")

    def test_real_backend_loads_and_scores_both_critics_on_tiny_cpu_model(self):
        from tokenizers import Tokenizer as FastTokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import MistralConfig, MistralForCausalLM, PreTrainedTokenizerFast
        from src.utils import llm_utils

        # Random tiny weights exercise real loaders/forwards without hub access.
        torch.manual_seed(0)
        hidden = 8
        actor = MistralForCausalLM(MistralConfig(vocab_size=32, hidden_size=hidden,
            intermediate_size=16, num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=1, max_position_embeddings=256, pad_token_id=0)).to(torch.bfloat16).eval()
        core = FastTokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1, "[EOS]": 2,
            "context": 3, "belief": 4, "answer": 5, "longer": 6}, unk_token="[UNK]"))
        core.pre_tokenizer = Whitespace()
        tok = PreTrainedTokenizerFast(tokenizer_object=core, pad_token="[PAD]",
                                     unk_token="[UNK]", eos_token="[EOS]", padding_side="left")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = ValueFunction(actor, hidden_size=hidden, device="cpu", dtype=torch.bfloat16,
                                  use_regret_critic=True, critic_mlp_dims=[12])
            value.save_checkpoint(str(root / "hl.pt"))
            head = ll_module.make_head(hidden, 2.)
            torch.save({"hidden_size": hidden, "mlp_width_mult": 2., "objective": "shapley", "k": 1,
                        "harm_head": head.state_dict(), "follow_head": head.state_dict(), "n_examples": 1}, root / "ll.pt")
            lock = {"actor": {"path": "tiny-local-fixture"},
                    "high_level": {"path": str(root / "hl.pt"), "sha256": "tiny-hl"},
                    "token_critic": {"path": str(root / "ll.pt"), "sha256": "tiny-ll"}}
            for method in ("vdcgs", "rdcgs"):
                with self.subTest(method=method), \
                     patch.object(ll_module, "HIDDEN_SIZE", hidden), \
                     patch.object(llm_utils, "get_model_instance", return_value=actor), \
                     patch.object(llm_utils, "get_tokenizer_instance", return_value=tok), \
                     patch.object(ll_module.AutoTokenizer, "from_pretrained", side_effect=AssertionError("Unexpected hub lookup")), \
                     patch.object(ll_module.AutoModelForCausalLM, "from_pretrained", side_effect=AssertionError("Unexpected model download")):
                    backend = adapter.LocalBackend(adapter.configuration(method, "cpu"), lock)
                    self.assertIs(backend.value.model, actor)
                    self.assertIs(backend.token_critic.encoder, actor.model)
                    self.assertEqual(backend.tokenizer.padding_side, "left")
                    self.assertEqual(backend.token_critic.tokenizer.padding_side, "right")
                    self.assertEqual(backend.loading["critic_mlp_dims"], [12])
                    expected = ["q", "v"] + (["q_min", "v_min", "regret"] if method == "rdcgs" else [])
                    self.assertEqual(backend.loading["strict_loaded_heads"], expected)
                    for name in expected:
                        self.assertFalse(any(p.requires_grad for p in getattr(backend.value, name + "_mlp_head").parameters()))
                    functions = ["predict_q_value"] + (["predict_q_min_value", "predict_regret_value"] if method == "rdcgs" else [])
                    for function in functions:
                        result = backend({"kind": "score", "function": function, "observations": ["context"] * 2,
                                          "high_level_actions": ["belief", "longer belief"], "requires_grad": False})
                        self.assertEqual(len(result["scores"]), 2)
                        self.assertTrue(torch.isfinite(torch.tensor(result["scores"])).all())
                    result = backend({"kind": "score_ll", "observation": "context", "selected_belief": "belief",
                                      "actions": ["answer", "longer answer"]})
                    self.assertEqual(set(result["scores"]), {"answer", "longer answer"})
                    self.assertTrue(torch.isfinite(torch.tensor(list(result["scores"].values()))).all())
                    self.assertFalse(any(p.requires_grad for p in backend.token_critic.harm_head.parameters()))

    def test_both_critic_artifacts_are_hash_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock = {"actor": {"path": str(root), "sha256": {}}}
            for name in ("high_level", "token_critic"):
                path = root / name
                path.write_bytes(name.encode())
                lock[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            runner.verify_artifacts(lock)
            (root / "token_critic").write_bytes(b"different checkpoint")
            with self.assertRaisesRegex(ValueError, "token_critic checkpoint mismatch"):
                runner.verify_artifacts(lock)

    def test_legacy_v2_cannot_validate_as_new_method(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "run_config.json").write_text(json.dumps({"protocol": "safedial_original_wildjailbreak_v2"}))
            with self.assertRaisesRegex(ValueError, "legacy v2"):
                validate(folder, tokenizer=Tokenizer())


if __name__ == "__main__":
    unittest.main()
