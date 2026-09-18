import copy
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_safedial_baseline as base
import run_safedial_dcgs_two_stage as runner
import safedial_dcgs_two_stage as algorithm
import safedial_dcgs_two_stage_runtime as runtime
from validate_safedial_dcgs_two_stage import validate_dcgs


def fake_execute(request):
    result = {"prompt_tokens": 12, "original_prompt_tokens": 12, "input_truncated": False, "latency_seconds": 0.01}
    if request["kind"] == "generate":
        if request["stage"] in ("nominal", "adversarial"):
            # Both sides deliberately contain the same texts; all ten IDs survive.
            text = "\n".join(f"{i}. Insight: intent {i}. Instruction: provide helpful response." for i in range(1, 6))
            result["message"] = text[3:]
        else:
            result["message"] = f"Response candidate {request['slot']} is the final answer.[/RESPONSE]"
        result["completion_tokens"] = 10
        result["hit_token_cap"] = False
    elif request["kind"] == "hl_score":
        result.update(q=float(request["slot"]), regret=float(9 - request["slot"]))
    else:
        result.update(score=20.0 if request["slot"] in (1, 3) else -1.0, action_start=5, action_end=12)
    return result


class AlgorithmTests(unittest.TestCase):
    def setUp(self):
        self.messages = [{"role": "user", "content": "first request"},
            {"role": "assistant", "content": "PRIOR_GOLD_ANSWER"}, {"role": "user", "content": "current request"}]

    def test_context_both_heads_indexed_duplicates_and_argmax_first_tie(self):
        calls = []
        def execute(req):
            calls.append(copy.deepcopy(req))
            return fake_execute(req)
        result = algorithm.generate_dcgs(self.messages, algorithm.defense_config(), 42, execute)
        audit = result["dcgs"]
        self.assertEqual(len(audit["intents"]), 10)
        self.assertEqual(len({c["text"] for c in audit["intents"]}), 5)
        self.assertEqual(len(audit["responses"]), 5)
        self.assertEqual(audit["selected_response_id"], 1)
        self.assertIn("Response candidate 1", result["message"])
        self.assertEqual(len(calls), 22)
        self.assertAlmostEqual(sum(c["probability"] for c in audit["intents"]), 1)
        for c in audit["intents"]:
            self.assertEqual(c["score"], 0.8 * c["q"] - 0.2 * c["regret"])
        belief = audit["intents"][audit["selected_intent_id"]]["text"]
        for request in calls:
            self.assertIn("PRIOR_GOLD_ANSWER", request.get("prompt", request.get("observation", "")))
            if request["stage"] == "response":
                self.assertIn(belief, request["prompt"])
            if request["kind"] == "ll_score":
                self.assertEqual(request["belief"], belief)
                self.assertIn(request["action"], [c["text"] for c in audit["responses"]])
        self.assertEqual(len({r["seed"] for r in calls if r["kind"] == "generate"}), 7)
        record = {**result, "prompt_history": self.messages, "seed": 42, "generated_response": result["message"]}
        algorithm.validate_audit(record, algorithm.defense_config())
        for mutation in (lambda r: r["dcgs"].update(selected_response_id=3),
                         lambda r: r["dcgs"]["responses"][1].update(score=-100),
                         lambda r: r["dcgs"]["events"][-1]["request"].update(belief="wrong")):
            changed = copy.deepcopy(record)
            mutation(changed)
            with self.assertRaises((ValueError, algorithm.DCGSFailure)):
                algorithm.validate_audit(changed, algorithm.defense_config())

    def test_errors_keep_raw_evidence_and_stop(self):
        for stage, field, value in (("nominal", "message", ""), ("intent_score", "q", float("nan")),
                                    ("response_score", "score", float("inf")), ("response", "input_truncated", True),
                                    ("response_score", "action_end", 13)):
            def bad(req):
                r = fake_execute(req)
                if req["stage"] == stage:
                    r[field] = value
                return r
            saved = []
            with self.subTest(stage=stage, field=field), self.assertRaises(algorithm.DCGSFailure) as caught:
                algorithm.generate_dcgs(self.messages, algorithm.defense_config(), 0, bad, saved.append)
            self.assertIsNone(caught.exception.audit["selected_response_id"])
            self.assertIn("result", saved[-1])
            self.assertIn("error", saved[-1])

    def test_all_unparseable_intents_fail_without_response_generation(self):
        def bad(req):
            self.assertEqual(req["kind"], "generate")
            r = fake_execute(req)
            r["message"] = "invalid"
            return r
        with self.assertRaisesRegex(algorithm.DCGSFailure, "No valid intent"):
            algorithm.generate_dcgs(self.messages, algorithm.defense_config(), 0, bad)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.rows = [{"id": 1, "task": "Ethics", "method": "fixture", "scene": "fixture",
            "history": [{"user": "first request", "bot": "PRIOR_GOLD_ANSWER"},
                        {"user": "current request", "bot": "SECRET_CURRENT_GOLD"}]}]
        self.dataset = self.root / "data.jsonl"
        base.append_jsonl(self.dataset, self.rows)
        self.args = runner.parse_args(["--dataset", str(self.dataset), "--output-dir", str(self.root / "out"), "--device", "cpu"])
        self.args.output_dir.mkdir()
        for name in ("hl", "ll"):
            (self.root / name).write_bytes(name.encode())
        lock = {k: {"path": str(self.root / name), "sha256": base.file_sha256(self.root / name)}
                for k, name in (("high_level", "hl"), ("token_critic", "ll"))}
        self.manifest = runner.manifest_for(self.args, self.rows, lock)
        base.ensure_manifest(self.args.output_dir / "run_config.json", self.manifest)

    def test_interruption_resume_no_leak_full_audit_and_noop(self):
        count = 0
        class Interrupted(Exception): pass
        def interrupted(req):
            nonlocal count
            if count == 8:
                raise Interrupted()
            count += 1
            return fake_execute(req)
        with self.assertRaises(Interrupted):
            runner.generate_selected(self.args, self.manifest, self.rows, interrupted)
        self.assertEqual(len(base.load_jsonl(self.args.output_dir / "events.jsonl")), 8)
        calls = []
        def execute(req):
            self.assertNotIn("SECRET_CURRENT_GOLD", json.dumps(req))
            calls.append(req)
            return fake_execute(req)
        self.assertEqual(runner.generate_selected(self.args, self.manifest, self.rows, execute), 0)
        self.assertEqual(len(calls), 36)
        report = validate_dcgs(self.args.output_dir)
        self.assertEqual(report["call_counts"], {"generate": 14, "hl_score": 20, "ll_score": 10})
        before = {n: base.file_sha256(self.args.output_dir / n) for n in ("events.jsonl", "turns.jsonl", "answers.jsonl")}
        self.assertEqual(runner.generate_selected(self.args, self.manifest, self.rows, lambda r: self.fail("No-op made a call")), 0)
        self.assertEqual(before, {n: base.file_sha256(self.args.output_dir / n) for n in before})
        with self.assertRaises(ValueError):
            validate_dcgs(self.args.output_dir, require_gpu=True)
        (self.root / "ll").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            validate_dcgs(self.args.output_dir)

    def test_failed_event_preserved_and_retry_reuses_successes(self):
        def fail(req):
            if req["kind"] == "ll_score":
                raise RuntimeError("injected score failure")
            return fake_execute(req)
        self.assertEqual(runner.generate_selected(self.args, self.manifest, self.rows, fail), 2)
        self.assertEqual(base.load_jsonl(self.args.output_dir / "answers.jsonl"), [])
        self.assertEqual(runner.generate_selected(self.args, self.manifest, self.rows, lambda r: self.fail("Unrequested retry")), 2)
        self.args.retry_errors = True
        calls = []
        self.assertEqual(runner.generate_selected(self.args, self.manifest, self.rows, lambda r: (calls.append(r), fake_execute(r))[1]), 0)
        self.assertEqual(calls[0]["kind"], "ll_score")
        validate_dcgs(self.args.output_dir)
        entries = base.load_jsonl(self.args.output_dir / "events.jsonl")
        self.assertTrue(any(e["event"].get("error") for e in entries))

    def test_manifest_and_history_changes_rejected(self):
        changed = copy.deepcopy(self.manifest)
        changed["defense"]["response_selection"] = "softmax"
        with self.assertRaises(RuntimeError):
            base.ensure_manifest(self.args.output_dir / "run_config.json", changed)
        runner.generate_selected(self.args, self.manifest, self.rows, fake_execute)
        records = base.load_jsonl(self.args.output_dir / "turns.jsonl")
        records[0]["prompt_history"][0]["content"] = "tampered"
        runner.atomic_jsonl(self.args.output_dir / "turns.jsonl", records)
        with self.assertRaisesRegex(ValueError, "gold history"):
            runner.generate_selected(self.args, self.manifest, self.rows, fake_execute)

    def test_partial_append_and_writer_lock(self):
        p = self.root / "journal.jsonl"
        p.write_bytes(b'{"ok": 1}\n{"unfinished":')
        self.assertEqual(runner.read_journal(p), [{"ok": 1}])
        self.assertEqual(next((self.root / "recovery").iterdir()).read_bytes(), b'{"unfinished":')
        with runner.single_writer(self.root):
            with self.assertRaises(RuntimeError):
                with runner.single_writer(self.root):
                    pass

    def test_completed_turn_cannot_hide_altered_journal(self):
        runner.generate_selected(self.args, self.manifest, self.rows, fake_execute)
        entries = base.load_jsonl(self.args.output_dir / "events.jsonl")
        entries[0]["event"]["result"]["message"] = "tampered generation"
        runner.atomic_jsonl(self.args.output_dir / "events.jsonl", entries)
        with self.assertRaisesRegex(ValueError, "journal differs"):
            runner.generate_selected(self.args, self.manifest, self.rows, fake_execute)


class RuntimeTests(unittest.TestCase):
    def test_separate_frozen_backbones_and_separate_trained_token_heads(self):
        import torch
        class TinyBackbone(torch.nn.Module):
            def __init__(self, dtype):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(2, dtype=dtype))
            @property
            def dtype(self): return self.weight.dtype
            @property
            def model(self): return self
        calls = []
        def load(*a, **kw):
            calls.append(kw)
            return TinyBackbone(kw["torch_dtype"]), {}
        heads = {name: torch.nn.Linear(2, 1).eval().requires_grad_(False)
                 for name in runtime.HL_HEADS + ("harm_head", "follow_head")}
        lock = {"actor": {"path": "fixture"}, "high_level": {"sha256": "hl"}, "token_critic": {"sha256": "ll"}}
        tok = lambda *a, **kw: {"input_ids": [[1], [2], [3], [4]]}
        with patch("transformers.AutoModelForCausalLM.from_pretrained", side_effect=load), \
             patch.object(runtime, "load_heads", return_value=(heads, {"objective": "shapley", "k": 5})):
            backend = runtime.LocalBackend(SimpleNamespace(device="cpu"), lock, tok, 32768)
        self.assertEqual([c["torch_dtype"] for c in calls], [torch.bfloat16, torch.float16])
        self.assertTrue(all(c["local_files_only"] for c in calls))
        self.assertFalse(any(p.requires_grad for p in backend.actor.parameters()))
        self.assertFalse(any(p.requires_grad for p in backend.ll.encoder.parameters()))
        self.assertIs(backend.ll.harm_head, heads["harm_head"])
        self.assertIs(backend.ll.follow_head, heads["follow_head"])
        self.assertIsNot(backend.ll.encoder, backend.actor.model)
        self.assertEqual(backend.loading["heads"]["token_checkpoint_sha256"], "ll")

    def test_token_span_includes_boundary_straddling_token_and_rejects_overflow(self):
        def tok(*a, **kw):
            self.assertFalse(kw["truncation"])
            return {"input_ids": [0, 1, 2, 3], "attention_mask": [1] * 4,
                    "offset_mapping": [(0, 0), (0, 2), (2, 4), (4, 6)]}
        encoded, start, end = runtime.tokenize_action_span(tok, "abc", "abcdef")
        self.assertEqual((start, end), (2, 4))
        self.assertEqual(encoded["input_ids"].shape[1], 4)
        def too_long(*a, **kw):
            return {"input_ids": [1] * 8193, "attention_mask": [1] * 8193, "offset_mapping": [(0, 4)] * 8193}
        with self.assertRaisesRegex(RuntimeError, "refusing to truncate"):
            runtime.tokenize_action_span(too_long, "abc", "abcdef")

    def test_high_context_overflow_never_reaches_model(self):
        import torch
        backend = object.__new__(runtime.LocalBackend)
        backend.torch, backend.args = torch, SimpleNamespace(device="cpu")
        backend.tokenizer = lambda *a, **kw: {"input_ids": torch.ones((1, 8193), dtype=torch.long)}
        backend.actor = None
        with self.assertRaisesRegex(ValueError, "refusing truncation"):
            backend({"kind": "hl_score", "observation": "history", "belief": "intent"})

    def test_generation_overflow_never_reaches_model(self):
        import torch
        backend = object.__new__(runtime.LocalBackend)
        backend.torch, backend.args, backend.actor_limit = torch, SimpleNamespace(device="cpu"), 10
        backend.tokenizer = SimpleNamespace(encode=lambda *a, **kw: list(range(6)))
        backend.actor = None
        with self.assertRaisesRegex(ValueError, "refusing truncation"):
            backend({"kind": "generate", "prompt": "history", "prefill": "prefill", "max_new_tokens": 1})

    def test_validate_only_never_initializes_backend(self):
        rows = [{"id": 1, "task": "Ethics", "method": "fixture", "scene": "fixture", "history": [{"user": "u", "bot": "g"}]}]
        with patch.object(base, "load_jsonl", return_value=rows), patch.object(base, "validate_dataset"), \
             patch.object(runner, "validate_lock", return_value={}), \
             patch.object(runner, "tokenizer_and_preflight", return_value=(None, 32768, {"passed": True})), \
             patch.object(runner, "LocalBackend", side_effect=AssertionError("No model loading in preflight")):
            self.assertEqual(runner.main(["--ids", "1", "--validate-only"]), 0)


if __name__ == "__main__":
    unittest.main()
