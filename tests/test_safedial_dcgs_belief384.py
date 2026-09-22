import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_safedial_dcgs_belief384 as context
from run_safedial_baseline import TASKS
from test_safedial_dcgs_wildjailbreak import ROW as ORIGINAL_ROW, Tokenizer, execute, is_ll_generation

ROW = {**ORIGINAL_ROW, "task": TASKS[0], "scene": "fixture"}


class Belief384Tests(unittest.TestCase):
    def test_only_declared_settings_change(self):
        for method in ("vdcgs", "rdcgs"):
            original = vars(context.adapter.configuration(method))
            for history in (False, True):
                enabled = vars(context.configuration(method, include_history=history))
                expected = {"freeform_max_new_tokens"}
                if history:
                    expected.add("ll_action_belief_only")
                self.assertEqual({k for k in original if original[k] != enabled[k]}, expected)
                self.assertEqual(enabled["freeform_max_new_tokens"], 384)
                self.assertEqual(enabled["ll_action_belief_only"], not history)

    def test_pool_and_fallback_receive_history_without_answer_leakage(self):
        for method in ("vdcgs", "rdcgs"):
            for enabled in (False, True):
                for turn in (0, 1):
                    with self.subTest(method=method, history=enabled, turn=turn):
                        config = context.configuration(method, include_history=enabled)
                        def backend(request):
                            if request["kind"] == "generate" and request["max_new_tokens"] == 640:
                                # Force the original fallback to fill missing candidates.
                                return {"texts": ["1. [RESPONSE]Only one.[/RESPONSE]"]}
                            return execute(request)
                        result = context.adapter.generate_turn(ROW, turn, 42, config, Tokenizer(), backend)
                        events = result["dcgs_original"]["events"]
                        hl = [e["request"] for e in events if e["request"]["kind"] == "generate"
                              and not is_ll_generation(e["request"])]
                        self.assertTrue(hl)
                        self.assertEqual({r["max_new_tokens"] for r in hl}, {384})
                        generations = [e["request"] for e in events if is_ll_generation(e["request"])]
                        self.assertEqual([r["max_new_tokens"] for r in generations], [640, 128])
                        for request in generations:
                            prompt = request["prompts"][0]
                            self.assertIn(result["dcgs_original"]["selected_belief"], prompt)
                            self.assertEqual(ROW["history"][turn]["user"] in prompt, enabled)
                            self.assertEqual("PRIVATE CURRENT REFERENCE" in prompt, enabled and turn == 1)
                            self.assertNotIn("PRIVATE NEXT REFERENCE", prompt)
                            if enabled:
                                self.assertIn(ROW["history"][0]["user"], prompt)
                        ll_score = next(e["request"] for e in events if e["request"]["kind"] == "score_ll")
                        self.assertIn(ROW["history"][turn]["user"], ll_score["observation"])
                        self.assertEqual("PRIVATE CURRENT REFERENCE" in ll_score["observation"], turn == 1)
                        self.assertNotIn("PRIVATE NEXT REFERENCE", ll_score["observation"])
                        record = json.loads(json.dumps({"seed": 42, "turn_index": turn,
                            "generated_response": result["message"], "dcgs_original": result["dcgs_original"]}))
                        context.adapter.validate_turn(record, ROW, config, Tokenizer())
                        wrong = context.configuration(method, include_history=not enabled)
                        with self.assertRaises((ValueError, context.adapter.PolicyFailure)):
                            context.adapter.validate_turn(copy.deepcopy(record), ROW, wrong, Tokenizer())

    def setup_run(self, directory, enabled=True, method="vdcgs"):
        folder = Path(directory)
        dataset = folder / "dataset.jsonl"
        dataset.write_text(json.dumps(ROW) + "\n")
        args = SimpleNamespace(method=method, device="cpu", dataset=dataset, seed=0,
                               include_history=enabled, on_turn_error="stop")
        lock = context.base.load_artifact_lock()
        manifest = json.loads(json.dumps(context.manifest_for(args, [ROW], lock)))
        context.base.write_json(folder / "run_config.json", manifest)
        config = context.configuration(method, "cpu", lock, include_history=enabled)
        return folder, manifest, config

    def test_resume_audit_and_manifest_tamper(self):
        for method in ("vdcgs", "rdcgs"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as directory:
                folder, manifest, config = self.setup_run(directory, method=method)
                context.base.run_selected(folder, [ROW], manifest, config, Tokenizer(), lambda: execute)
                report = context.validate(folder, tokenizer=Tokenizer())
                self.assertTrue(report["passed"])
                self.assertEqual(report["turns"], 2)
                before = {p.name: p.read_bytes() for p in folder.iterdir()}
                context.base.run_selected(folder, [ROW], manifest, config, Tokenizer(),
                                         lambda: self.fail("No-op resume loaded models"))
                self.assertEqual(before, {p.name: p.read_bytes() for p in folder.iterdir()})
                tampered = copy.deepcopy(manifest)
                tampered["effective_config"]["freeform_max_new_tokens"] = 96
                context.base.write_json(folder / "run_config.json", tampered)
                with self.assertRaisesRegex(ValueError, "configuration mismatch"):
                    context.validate(folder, tokenizer=Tokenizer())
                manifest["include_history"] = False
                context.base.write_json(folder / "run_config.json", manifest)
                with self.assertRaisesRegex(ValueError, "identity mismatch"):
                    context.validate(folder, tokenizer=Tokenizer())

    def test_cross_mode_output_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            folder, _, _ = self.setup_run(directory)
            before = (folder / "run_config.json").read_bytes()
            with patch.object(context.base, "verify_artifacts", side_effect=AssertionError("Loaded artifacts")):
                with self.assertRaisesRegex(ValueError, "context mismatch"):
                    context.main(["--method", "vdcgs", "--device", "cpu", "--output-dir", str(folder),
                                  "--dataset", str(folder / "dataset.jsonl"), "--validate-only"])
            self.assertEqual(before, (folder / "run_config.json").read_bytes())

    def test_preflight_includes_long_history_and_reports_critic_risk(self):
        row = copy.deepcopy(ROW)
        row["history"][0]["user"] = "z" * 9000
        config = context.configuration("vdcgs", include_history=True)
        report = context.base.preflight([row], config, Tokenizer())
        self.assertGreater(report["max_prompt_plus_generation_tokens"], 9000)
        self.assertGreater(report["ll_critic_context"]["potential_overflow_turns"], 0)
        self.assertEqual(report["model_calls"], 0)

    def test_ll_critic_limit_still_rejects_without_trimming(self):
        row = copy.deepcopy(ROW)
        row["history"][0]["user"] = "z" * 9000
        with self.assertRaises(context.adapter.PolicyFailure) as failure:
            context.adapter.generate_turn(row, 0, 0, context.configuration("vdcgs", include_history=True),
                                          Tokenizer(), execute)
        self.assertEqual(failure.exception.details["code"], "ll_context_overflow")


if __name__ == "__main__":
    unittest.main()
