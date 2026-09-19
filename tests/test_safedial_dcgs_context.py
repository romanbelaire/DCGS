"""CPU checks for the intact LL context limit and SafeDial turn continuation."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "scripts")]
import run_safedial_dcgs_wildjailbreak as runner
import safedial_dcgs_wildjailbreak as adapter
from validate_safedial_dcgs_wildjailbreak import validate
from src.value import ll_token_critic as ll_module
from test_safedial_dcgs_wildjailbreak import ROW, Tokenizer, execute


def long_pool(request):
    if request["kind"] == "generate" and request["max_new_tokens"] == 640:
        actions = ["x" * 500, "ok", "ok", "ok", "ok"]
        text = "\n".join(f"{i}. [RESPONSE]{a}[/RESPONSE]" for i, a in enumerate(actions, 1))
        assert len(text) <= 640  # Valid within this fixture's character-token budget.
        return {"texts": [text]}
    return execute(request)


class ContextTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.tok = Tokenizer()

    def setup_run(self, method, mode, row):
        root = self.root / (method + "-" + mode)
        root.mkdir()
        data = root / "data.jsonl"
        row = copy.deepcopy(row)
        row.update(task="Ethics", scene="fixture")
        data.write_text(json.dumps(row) + "\n")
        config = adapter.configuration(method)
        args = SimpleNamespace(method=method, device="cpu", seed=0, dataset=data, on_turn_error=mode)
        lock = {"actor": {}, "high_level": {"path": config.checkpoint_path, "sha256": "fixture"},
                "token_critic": {"path": config.ll_token_critic_path, "sha256": "ll-fixture"}}
        manifest = json.loads(json.dumps(runner.manifest_for(args, [row], lock)))
        folder = root / "run"
        folder.mkdir()
        runner.write_json(folder / "run_config.json", manifest)
        return folder, [row], config, manifest

    def test_exact_limit_allowed_and_overflow_rejected_before_encoder(self):
        limit = ll_module.MAX_UNTRUNCATED_LEN
        prefix = "p"
        full = prefix + "x" * (limit - 2)  # BOS is one additional token.
        encoded, start, end = ll_module.tokenize_action_span(self.tok, prefix, full)
        self.assertEqual(encoded["input_ids"].shape[1], limit)
        self.assertEqual(end, limit)
        with self.assertRaises(ll_module.LLContextLengthError) as error:
            ll_module.tokenize_action_span(self.tok, prefix, full + "x")
        self.assertEqual((error.exception.input_tokens, error.exception.max_tokens), (limit + 1, limit))
        encoder = Mock(side_effect=AssertionError("Oversized input reached the model"))
        critic = ll_module.LLTokenCritic(None, None, encoder, self.tok, "cpu", "shapley", 1)
        with self.assertRaises(ll_module.LLContextLengthError):
            critic.score_actions("observation", "belief", ["x" * limit])
        encoder.assert_not_called()

    def test_preflight_reserves_whole_pool_budget_and_reports_risk(self):
        row = copy.deepcopy(ROW)
        row["history"] = [{"user": "x" * 7850, "bot": "reference"}]
        for method in ("vdcgs", "rdcgs"):
            with self.subTest(method=method):
                report = runner.preflight([row], adapter.configuration(method), self.tok)
                self.assertTrue(report["passed"])
                context = report["ll_critic_context"]
                self.assertEqual(context["potential_overflow_turns"], 1)
                self.assertEqual(context["synthetic_overflow_turns"], 0)
                risk = context["potential_overflow_turn_details"][0]
                self.assertEqual(risk["response_generation_budget"], 640)
                self.assertEqual(risk["belief_generation_budget"], 96)
                self.assertGreater(risk["estimated_input_tokens"], context["max_input_tokens"])
                self.assertEqual(report["model_calls"], 0)

    def test_preflight_keeps_scanning_after_synthetic_overflow(self):
        row = copy.deepcopy(ROW)
        row["history"] = [{"user": "x" * 8300, "bot": "reference"}]
        short = copy.deepcopy(ROW)
        short["id"] = 2
        short["history"] = short["history"][:1]
        report = runner.preflight([row, short], adapter.configuration("vdcgs"), self.tok)
        self.assertEqual(report["turns"], 2)
        self.assertEqual(report["ll_critic_context"]["synthetic_overflow_turns"], 1)
        self.assertEqual(report["ll_critic_context"]["potential_overflow_turns"], 1)
        self.assertEqual(report["model_calls"], 0)

    def test_full_runs_continue_and_never_retry_overflow_turn(self):
        row = copy.deepcopy(ROW)
        row["history"] = [copy.deepcopy(ROW["history"][0]) for _ in range(3)]
        row["history"][1]["user"] = "x" * 7650
        for method in ("vdcgs", "rdcgs"):
            with self.subTest(method=method):
                folder, rows, config, manifest = self.setup_run(method, "record-and-continue", row)
                pools = []
                def backend(request):
                    if request["kind"] == "generate" and request["max_new_tokens"] == 640:
                        pools.append(request)
                        if len(pools) == 2:
                            return long_pool(request)
                    return execute(request)
                code = runner.run_selected(folder, rows, manifest, config, self.tok, lambda: backend)
                self.assertEqual(code, 2)
                self.assertEqual(len(pools), 3)
                self.assertFalse((folder / "failure.json").exists())
                failure = runner.read_records(folder / "failures.jsonl")[0]
                self.assertEqual((failure["code"], failure["category"]), ("ll_context_overflow", "terminal_output"))
                self.assertGreater(failure["input_tokens"], failure["max_tokens"])
                report = validate(folder, tokenizer=self.tok)
                self.assertEqual((report["successful_turns"], report["terminal_failed_turns"], report["remaining_turns"]), (2, 1, 0))
                self.assertTrue(report["execution_finished"])
                self.assertFalse(report["passed"])
                self.assertEqual(report["ll_critic_overflow_turns"], 1)
                self.assertEqual(runner.read_records(folder / "answers.jsonl"), [])
                records = runner.read_records(folder / "turns.jsonl")
                self.assertEqual([r["turn_index"] for r in records], [0, 2])
                self.assertIn(row["history"][1]["bot"], str(records[-1]["prompt_history"]))
                before = {p.name: p.read_bytes() for p in folder.iterdir()}
                def forbidden():
                    self.fail("Completed/terminal turns caused another model load")
                self.assertEqual(runner.run_selected(folder, rows, manifest, config, self.tok, forbidden), 2)
                self.assertEqual(before, {p.name: p.read_bytes() for p in folder.iterdir()})

    def test_stop_mode_preserves_overflow_and_recovery_processes_later_turn(self):
        row = copy.deepcopy(ROW)
        row["history"][0]["user"] = "x" * 7650
        folder, rows, config, manifest = self.setup_run("vdcgs", "stop", row)
        with self.assertRaises(adapter.PolicyFailure) as error:
            runner.run_selected(folder, rows, manifest, config, self.tok, lambda: long_pool)
        self.assertEqual(error.exception.details["code"], "ll_context_overflow")
        self.assertTrue((folder / "failure.json").exists())
        target = self.root / "recovered"
        runner.recover_run(folder, target, manifest, rows, config, self.tok, "Preserve overflow; finish remaining turns")
        self.assertEqual(runner.run_selected(target, rows, manifest, config, self.tok, lambda: execute), 2)
        report = validate(target, tokenizer=self.tok)
        self.assertTrue(report["execution_finished"])
        self.assertEqual((report["ll_critic_overflow_turns"], report["successful_turns"]), (1, 1))

    def test_overflow_replay_recomputes_lengths_and_rejects_false_metadata(self):
        row = copy.deepcopy(ROW)
        row["history"] = [{"user": "x" * 7850, "bot": "reference"}]
        config = adapter.configuration("vdcgs")
        with self.assertRaises(adapter.PolicyFailure) as error:
            adapter.generate_turn(row, 0, 0, config, self.tok, long_pool)
        events = error.exception.audit["events"]
        with self.assertRaises(adapter.PolicyFailure) as replayed:
            adapter.replay_trace(row, 0, 0, config, self.tok, events)
        self.assertEqual(error.exception.details, replayed.exception.details)
        for field in ("input_tokens", "max_tokens"):
            altered = copy.deepcopy(events)
            altered[-1]["failure_details"][field] += 1
            with self.assertRaises(adapter.PolicyFailure) as rejected:
                adapter.replay_trace(row, 0, 0, config, self.tok, altered)
            self.assertEqual(rejected.exception.details["category"], "integrity")
        with self.assertRaises(adapter.PolicyFailure) as invalid:
            adapter.generate_turn(ROW, 0, 0, config, self.tok,
                lambda req: (_ for _ in ()).throw(RuntimeError("Unrelated scorer failure")) if req["kind"] == "score_ll" else execute(req))
        self.assertEqual(invalid.exception.details["category"], "integrity")

    def test_runner_rejects_unreviewed_main_execution_modes(self):
        base = adapter.configuration("vdcgs")
        for key, value in {"defender_backend": "nbf", "use_gpt_for_agents": True,
                           "paper_dsr_reward": True, "hl_greedy_q": True,
                           "raw_judge_belief_selection": True, "random_belief_selection": True,
                           "baseline_mode": True, "ground_truth_belief_selection": True}.items():
            with self.subTest(setting=key):
                changed = copy.deepcopy(base)
                setattr(changed, key, value)
                with patch.object(adapter.BaseConfig, "from_json", return_value=changed):
                    with self.assertRaisesRegex(ValueError, "configuration changed"):
                        adapter.configuration("vdcgs")


if __name__ == "__main__":
    unittest.main()
