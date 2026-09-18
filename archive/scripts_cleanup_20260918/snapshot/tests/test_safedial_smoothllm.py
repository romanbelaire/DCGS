import copy
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_safedial_baseline as base
import run_safedial_smoothllm as runner
import safedial_smoothllm as smooth
from validate_safedial_smoothllm import validate_smoothllm


def decoded(message="I cannot help with that."):
    return {"message": message, "prompt_tokens": 10, "original_prompt_tokens": 10,
            "completion_tokens": 5, "latency_seconds": 0.1, "input_truncated": False}


class SmoothAlgorithmTest(unittest.TestCase):
    def test_current_user_only_in_repeated_history(self):
        messages = base.gold_messages([{"user": "repeat this text", "bot": "repeat this text"},
                                       {"user": "repeat this text", "bot": "future secret"}], 1)
        original = copy.deepcopy(messages)
        changed = smooth.perturb_current(messages, 100, random.Random(0))
        self.assertEqual(messages, original)
        self.assertEqual(changed[:-1], original[:-1])
        self.assertNotEqual(changed[-1]["content"], original[-1]["content"])
        self.assertNotIn("future secret", str(changed))

    def test_empty_tiny_and_invalid_parameters(self):
        for text in ("", "x"):
            messages = [{"role": "user", "content": text}]
            self.assertEqual(smooth.perturb_current(messages, 10, random.Random(0)), messages)
        for copies, percent in ((0, 10), (8, -1), (8, 101), (8, float("nan"))):
            with self.assertRaises(ValueError):
                smooth.defense_config(copies, percent)

    def test_tie_and_indexed_duplicates(self):
        candidates = [decoded("answer"), decoded("answer"), decoded(), decoded()]
        majority, selected = smooth.vote_and_select(candidates, random.Random(0))
        self.assertFalse(majority)
        self.assertIn(selected, (2, 3))
        candidates = [decoded("answer") for _ in range(8)]
        selected_indices = {smooth.vote_and_select(candidates, random.Random(seed))[1] for seed in range(100)}
        self.assertEqual(selected_indices, set(range(8)))
        self.assertTrue(smooth.vote_and_select(candidates, random.Random(0))[0])

    def test_detector_matches_upstream_case_sensitive_substring(self):
        for prefix in smooth.REFUSAL_PREFIXES:
            self.assertFalse(smooth.is_jailbroken("before " + prefix + " after"))
        self.assertTrue(smooth.is_jailbroken("i cannot help"))

    def test_rng_isolated_from_decoder_and_all_copy_costs(self):
        messages = [{"role": "user", "content": "current user text"}]
        config = smooth.defense_config()
        def noisy_generate(prompt, seed):
            random.seed(seed)
            random.random()
            return decoded()
        first = smooth.generate_smoothed(messages, config, 42, noisy_generate)
        second = smooth.generate_smoothed(messages, config, 42, lambda p, s: decoded())
        self.assertEqual(first["smoothllm"], second["smoothllm"])
        self.assertEqual(first["prompt_tokens"], 80)
        self.assertEqual(first["completion_tokens"], 40)
        record = {**first, "seed": 42, "prompt_history": messages, "generated_response": first["message"]}
        smooth.validate_audit(record, config)
        record["smoothllm"]["candidates"][0]["perturbed_user_message"] += "tampered"
        with self.assertRaisesRegex(ValueError, "perturbation"):
            smooth.validate_audit(record, config)

    def test_failure_preserves_partial_audit_and_never_votes(self):
        calls = []
        def generate(prompt, seed):
            calls.append(seed)
            return decoded() if len(calls) == 1 else decoded("")
        with self.assertRaises(smooth.CandidateFailure) as caught:
            smooth.generate_smoothed([{"role": "user", "content": "text"}], smooth.defense_config(), 0, generate)
        self.assertEqual(len(caught.exception.audit["candidates"]), 2)
        self.assertIsNone(caught.exception.audit["selected_index"])
        self.assertIn("error", caught.exception.audit["candidates"][-1])


class SmoothRunnerTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.rows = [{"id": index, "task": "Ethics", "method": "fixture", "scene": "fixture",
                      "history": [{"user": "same current user", "bot": "gold prior assistant"},
                                  {"user": "same current user", "bot": "gold future assistant"}]}
                     for index in (1, 2)]
        self.dataset = self.root / "data.jsonl"
        base.append_jsonl(self.dataset, self.rows)
        self.args = runner.parse_args(["--dataset", str(self.dataset), "--output-dir", str(self.root / "out")])
        self.args.output_dir.mkdir()
        self.manifest = runner.manifest_for(self.args, self.rows)
        base.ensure_manifest(self.args.output_dir / "run_config.json", self.manifest)

    def test_generation_validation_and_completed_resume(self):
        seen = []
        def generate(messages, seed):
            seen.append(copy.deepcopy(messages))
            return decoded("generated non-gold answer")
        self.assertEqual(runner.generate_dialogues(self.args, self.manifest, self.rows, generate), 0)
        report = validate_smoothllm(self.args.output_dir)
        self.assertEqual(report["turns"], 4)
        self.assertEqual(report["candidates"], 32)
        self.assertEqual(seen[8][1]["content"], "gold prior assistant")
        self.assertNotIn("generated non-gold answer", str(seen))
        self.assertEqual(runner.pending_rows(self.rows, self.args.output_dir / "answers.jsonl", self.args), [])
        tampered = copy.deepcopy(self.manifest)
        tampered["defense"]["num_copies"] = 4
        with self.assertRaisesRegex(RuntimeError, "different run"):
            base.ensure_manifest(self.args.output_dir / "run_config.json", tampered)

    def test_interrupted_resume_replays_only_uncommitted_dialogue(self):
        calls = []
        def interrupt(messages, seed):
            calls.append(seed)
            if len(calls) == 18:
                raise KeyboardInterrupt()
            return decoded()
        with self.assertRaises(KeyboardInterrupt):
            runner.generate_dialogues(self.args, self.manifest, self.rows, interrupt)
        pending = runner.pending_rows(self.rows, self.args.output_dir / "answers.jsonl", self.args)
        self.assertEqual([r["id"] for r in pending], [2])
        resumed_seeds = []
        def resume(messages, seed):
            resumed_seeds.append(seed)
            return decoded()
        runner.generate_dialogues(self.args, self.manifest, pending, resume)
        self.assertEqual(resumed_seeds[0], calls[16])
        self.assertEqual(validate_smoothllm(self.args.output_dir)["candidates"], 32)

    def test_failed_dialogue_retries_and_compacts(self):
        self.assertEqual(runner.generate_dialogues(self.args, self.manifest, self.rows,
                                                  lambda p, s: decoded("")), 4)
        with self.assertRaises(ValueError):
            validate_smoothllm(self.args.output_dir)
        self.args.retry_errors = True
        pending = runner.pending_rows(self.rows, self.args.output_dir / "answers.jsonl", self.args)
        self.assertEqual(len(pending), 2)
        runner.generate_dialogues(self.args, self.manifest, pending, lambda p, s: decoded())
        self.assertEqual(validate_smoothllm(self.args.output_dir)["turns"], 4)

    def test_validate_only_makes_no_output(self):
        output = self.root / "not-created"
        self.assertEqual(runner.main(["--dataset", str(self.dataset), "--output-dir", str(output), "--validate-only"]), 0)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
