import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_safedial_baseline as base
import run_safedial_tpo as runner
import safedial_tpo as tpo
from validate_safedial_tpo import validate_tpo


def fake_execute(request):
    result = {"prompt_tokens": 10, "original_prompt_tokens": 10, "input_truncated": False, "latency_seconds": 0.01}
    if request["kind"] == "reward":
        # Fixed reward, independent of the implementation's ranking function.
        text = request["messages"][-1]["content"]
        result["score"] = {"initial-0": 9.0, "initial-1": -5.0, "initial-2": 3.0,
                           "update-0-0": 2.0, "update-0-1": 10.0, "update-0-2": 4.0,
                           "update-1-0": 7.0, "update-1-1": 6.0, "update-1-2": 0.0}.get(text, 0.0)
    else:
        stage, iteration, slot = request["stage"], request["iteration"], request["slot"]
        text = f"initial-{slot}" if stage == "initial" else f"{stage}-{iteration}-{slot}"
        if stage == "update":
            text = tpo.START + text + tpo.END
        result.update(message=text, completion_tokens=5, hit_token_cap=False)
    return result


class TPOAlgorithmTests(unittest.TestCase):
    def setUp(self):
        self.config = tpo.defense_config(sample_size=3)
        self.messages = [{"role": "user", "content": "repeat"}, {"role": "assistant", "content": "gold previous"},
                         {"role": "user", "content": "repeat"}]

    def test_cumulative_reward_selection_and_exact_candidate_budget(self):
        result = tpo.generate_tpo(self.messages, self.config, 42, fake_execute)
        self.assertEqual(result["message"], "update-0-1")
        self.assertEqual(result["tpo"]["selected_id"], 4)
        self.assertEqual(len(result["tpo"]["candidates"]), 9)
        self.assertEqual([(r["chosen_id"], r["rejected_id"]) for r in result["tpo"]["rounds"]], [(0, 1), (4, 1)])
        # 9 response generations + 4 feedback generations, and 9 reward calls.
        self.assertEqual(len(result["tpo"]["events"]), 22)
        self.assertEqual(result["completion_tokens"], 65)
        self.assertEqual(result["prompt_tokens"], 130)
        self.assertEqual(result["reward_prompt_tokens"], 90)

    def test_gold_context_in_initial_reward_and_feedback(self):
        original = copy.deepcopy(self.messages)
        result = tpo.generate_tpo(self.messages, self.config, 0, fake_execute)
        self.assertEqual(self.messages, original)
        for event in result["tpo"]["events"]:
            req = event["request"]
            if req["stage"] == "initial":
                self.assertEqual(req["messages"], original)
            elif req["kind"] == "reward":
                self.assertEqual(req["messages"][:-1], original)
                self.assertNotIn(tpo.START, req["messages"][-1]["content"])
            else:
                self.assertIn("gold previous", str(req["messages"]))

    def test_prompt_stages_and_seed_reproducibility(self):
        first = tpo.generate_tpo(self.messages, self.config, 42, fake_execute)
        again = tpo.generate_tpo(self.messages, self.config, 42, fake_execute)
        self.assertEqual(first["tpo"], again["tpo"])
        generations = [e["request"] for e in first["tpo"]["events"] if e["request"]["kind"] == "generate"]
        self.assertEqual(len({r["seed"] for r in generations}), len(generations))
        for req in generations:
            if req["stage"] == "gradient":
                self.assertIn("<OBJECTIVE_FUNCTION>", req["messages"][1]["content"])
                self.assertEqual(req["top_p"], 0.99)
            if req["stage"] == "update":
                self.assertIn("<FEEDBACK>", req["messages"][1]["content"])
                self.assertIn("Constraint 2:", req["messages"][1]["content"])
                self.assertEqual(req["top_p"], 0.95)

    def test_duplicates_and_ties_remain_indexed(self):
        def identical(req):
            result = fake_execute(req)
            if req["kind"] == "reward": result["score"] = 1.0
            elif req["stage"] == "initial": result["message"] = "duplicate"
            return result
        result = tpo.generate_tpo(self.messages, self.config, 0, identical)
        self.assertEqual(result["tpo"]["selected_id"], 0)
        self.assertEqual(len(result["tpo"]["candidates"]), 9)

    def test_empty_nonfinite_truncated_and_malformed_fail_with_evidence(self):
        for stage, field, value in [("initial", "message", ""), ("score", "score", float("nan")),
                                    ("loss", "input_truncated", True), ("update", "message", "missing tags")]:
            def bad(req):
                out = fake_execute(req)
                if req["stage"] == stage: out[field] = value
                return out
            persisted = []
            with self.subTest(stage=stage), self.assertRaises(tpo.TPOFailure) as caught:
                tpo.generate_tpo(self.messages, self.config, 0, bad, persisted.append)
            self.assertIsNone(caught.exception.audit["selected_id"])
            self.assertIn("error", persisted[-1])
            self.assertIn("result", persisted[-1])

    def test_reject_missing_or_empty_first_improvement(self):
        for text in ["", "missing tags", tpo.END, tpo.END+tpo.START, tpo.START+tpo.END,
                     tpo.START, tpo.START+"   ", tpo.START*2+"x",
                     tpo.START+"  "+tpo.END+tpo.START+"valid later"+tpo.END]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                tpo.extract_update(text)
        self.assertEqual(tpo.extract_update("prefix "+tpo.START+"good"+tpo.END+" suffix"), "good")

    def test_missing_closing_tag_preserves_scored_and_selected_text_without_retries(self):
        body = "answer\n</LM_INPUT>\n<FEEDBACK>copied context remains visible"
        for capped in (False, True):
            calls = []
            def execute(req):
                calls.append(copy.deepcopy(req))
                out = fake_execute(req)
                if req["stage"] == "update" and req["iteration"] == 0 and req["slot"] == 1:
                    out.update(message=tpo.START+body, hit_token_cap=capped,
                               completion_tokens=req["max_new_tokens"] if capped else 5)
                if req["kind"] == "reward" and req["messages"][-1]["content"] == body:
                    out["score"] = 20.0
                return out
            with self.subTest(capped=capped):
                result = tpo.generate_tpo(self.messages, self.config, 0, execute)
                self.assertEqual(result["message"], body)
                self.assertEqual(len(calls), 22)
                self.assertEqual(len(result["tpo"]["candidates"]), 9)
                warning_events = [e for e in result["tpo"]["events"] if e.get("format_warnings")]
                self.assertEqual(len(warning_events), 1)
                self.assertEqual(warning_events[0]["format_warnings"], [tpo.MISSING_END_WARNING])
                self.assertEqual(warning_events[0]["result"]["message"], tpo.START+body)
                self.assertTrue(any(r["kind"] == "reward" and r["messages"][-1]["content"] == body for r in calls))
                record = {**result, "prompt_history": self.messages, "seed": 0, "generated_response": body}
                tpo.validate_audit(record, self.config)
                warning_events[0].pop("format_warnings")
                with self.assertRaisesRegex(ValueError, "audit mismatch"):
                    tpo.validate_audit(record, self.config)

    def test_opening_tag_in_trailing_instruction_does_not_change_complete_answer(self):
        def execute(req):
            out = fake_execute(req)
            if req["stage"] == "update":
                out["message"] += "\nSend the improved variable only between the " + tpo.START + " tags."
            return out
        result = tpo.generate_tpo(self.messages, self.config, 0, execute)
        self.assertEqual(result["message"], "update-0-1")
        self.assertEqual(len(result["tpo"]["events"]), 22)
        warnings = [e for e in result["tpo"]["events"] if e.get("format_warnings")]
        self.assertEqual(len(warnings), 6)
        self.assertTrue(all(e["format_warnings"] == [tpo.TRAILING_START_WARNING] for e in warnings))
        for event in result["tpo"]["events"]:
            if event["request"]["kind"] == "reward":
                self.assertNotIn("Send the improved variable", event["request"]["messages"][-1]["content"])
        record = {**result, "prompt_history": self.messages, "seed": 0, "generated_response": result["message"]}
        tpo.validate_audit(record, self.config)
        warnings[0].pop("format_warnings")
        with self.assertRaisesRegex(ValueError, "audit mismatch"):
            tpo.validate_audit(record, self.config)

    def test_audit_detects_tampering(self):
        result = tpo.generate_tpo(self.messages, self.config, 0, fake_execute)
        record = {**result, "prompt_history": self.messages, "seed": 0, "generated_response": result["message"]}
        tpo.validate_audit(record, self.config)
        for mutate in [lambda r: r["tpo"].update(selected_id=0),
                       lambda r: r["tpo"]["events"][0]["request"].update(seed=12),
                       lambda r: r.update(completion_tokens=1),
                       lambda r: r["tpo"]["rounds"][1].update(rejected_id=0)]:
            altered = copy.deepcopy(record); mutate(altered)
            with self.assertRaises(ValueError): tpo.validate_audit(altered, self.config)

    def test_duplicate_blocks_use_only_boundary_normalization_and_keep_first_text(self):
        block = lambda value: tpo.START + value + tpo.END
        for first, second in [("answer", "answer"), (" answer\n", " { answer } "),
                              ("{ answer }", "answer"), ("a\nb", " {a\nb} "),
                              ('{"key": 1}', '{"key": 1}')]:
            with self.subTest(first=first, second=second):
                raw = "prefix " + block(first) + " echoed instructions " + block(second) + " suffix"
                self.assertEqual(tpo.parse_update(raw), (first.strip(), [tpo.DUPLICATE_BLOCK_WARNING]))
        for first, second in [("a b", "a  b"), ("a\nb", "a b"), ("answer", "Answer"),
                              ("answer", "answer."), ("answer", "{{answer}}"),
                              ("{answer", "answer"), ("", ""), ("{}", "{ }"), (" ", "{}")]:
            with self.subTest(first=first, second=second):
                if not first.strip():
                    with self.assertRaises(ValueError):
                        tpo.parse_update(block(first) + block(second))
                else:
                    self.assertEqual(tpo.extract_update(block(first) + block(second)), first.strip())

    def test_first_improvement_ignores_later_blocks_and_misordered_tags(self):
        block = tpo.START + "answer" + tpo.END
        for raw in [block * 3, block * 2 + tpo.START, block * 2 + tpo.END,
                    tpo.END + block * 2, block + tpo.END,
                    tpo.START + "answer" + tpo.START + "different" + tpo.END,
                    tpo.START + "answer" + tpo.START + "different",
                    "prefix " + tpo.START + "answer" + tpo.START]:
            with self.subTest(raw=raw):
                self.assertEqual(tpo.parse_update(raw), ("answer", [tpo.FIRST_IMPROVEMENT_WARNING]))
        for raw in [tpo.START + block * 2, tpo.START * 2 + "answer" + tpo.END * 2]:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                tpo.parse_update(raw)

    def test_trailing_instructions_do_not_change_first_improvement(self):
        block = lambda value: tpo.START + value + tpo.END
        duplicate = block(" {answer} ") + block("answer")
        suffix = tpo.ECHOED_FORMAT_INSTRUCTION
        self.assertEqual(tpo.parse_update(duplicate + "\n\t" + suffix + " \n"),
                         ("{answer}", [tpo.DUPLICATE_BLOCK_WARNING, tpo.TRAILING_START_WARNING]))
        for tail in [suffix + " extra", "extra " + suffix, suffix.replace("ONLY", "only"),
                     suffix + suffix, tpo.START + "third answer", block("different") + suffix]:
            with self.subTest(tail=tail):
                self.assertEqual(tpo.extract_update(duplicate + tail), "{answer}")
        self.assertEqual(tpo.extract_update(block("answer") + block("different") + suffix), "answer")
        # Opening tags are literal boundaries, including quoted tags in a prefix.
        self.assertEqual(tpo.extract_update(suffix + duplicate), "tags, and nothing else.")

    def test_real_duplicate_failure_is_scored_and_audited_without_regeneration(self):
        fixture = json.loads((Path(__file__).parent / "fixtures/tpo_duplicate_blocks_252269.json").read_text())
        raw_result = fixture["result"]
        self.assertEqual(raw_result["completion_tokens"], 1024)
        self.assertTrue(raw_result["hit_token_cap"])
        raw = raw_result["message"]
        body = raw.split(tpo.START, 1)[1].split(tpo.END, 1)[0].strip()
        self.assertEqual(tpo.parse_update(raw), (body, [tpo.DUPLICATE_BLOCK_WARNING]))
        for reward in (-100.0, 100.0):
            calls = []
            def execute(req):
                calls.append(copy.deepcopy(req))
                if req["stage"] == "update" and req["iteration"] == 0 and req["slot"] == 2:
                    return copy.deepcopy(raw_result)
                out = fake_execute(req)
                if req["kind"] == "reward" and req["messages"][-1]["content"] == body:
                    out["score"] = reward
                return out
            with self.subTest(reward=reward):
                result = tpo.generate_tpo(self.messages, self.config, 0, execute)
                self.assertEqual(len(calls), 22)
                self.assertEqual(result["message"] == body, reward > 0)
                warnings = [e for e in result["tpo"]["events"] if e.get("format_warnings")]
                self.assertEqual(len(warnings), 1)
                self.assertEqual(warnings[0]["result"], raw_result)
                self.assertEqual(warnings[0]["format_warnings"], [tpo.DUPLICATE_BLOCK_WARNING])
                self.assertTrue(any(q["kind"] == "reward" and q["messages"][-1]["content"] == body for q in calls))
                record = {**result, "prompt_history": self.messages, "seed": 0,
                          "generated_response": result["message"]}
                tpo.validate_audit(record, self.config)
                warnings[0].pop("format_warnings")
                with self.assertRaisesRegex(ValueError, "audit mismatch"):
                    tpo.validate_audit(record, self.config)

    def test_real_duplicate_trailing_instruction_failure_is_scored_and_audited(self):
        fixture = json.loads((Path(__file__).parent / "fixtures/tpo_duplicate_trailing_instruction_254257.json").read_text())
        raw_result = fixture["result"]
        self.assertEqual(raw_result["completion_tokens"], 799)
        self.assertFalse(raw_result["hit_token_cap"])
        raw = raw_result["message"]
        body = raw.split(tpo.START, 1)[1].split(tpo.END, 1)[0].strip()
        self.assertEqual(tpo.parse_update(raw), (body, [tpo.DUPLICATE_BLOCK_WARNING, tpo.TRAILING_START_WARNING]))
        for reward in (-100.0, 100.0):
            calls = []
            def execute(req):
                calls.append(copy.deepcopy(req))
                if req["stage"] == "update" and req["iteration"] == 0 and req["slot"] == 2:
                    return copy.deepcopy(raw_result)
                out = fake_execute(req)
                if req["kind"] == "reward" and req["messages"][-1]["content"] == body:
                    out["score"] = reward
                return out
            with self.subTest(reward=reward):
                result = tpo.generate_tpo(self.messages, self.config, 0, execute)
                self.assertEqual(len(calls), 22)
                self.assertEqual(result["message"] == body, reward > 0)
                warnings = [e for e in result["tpo"]["events"] if e.get("format_warnings")]
                self.assertEqual(len(warnings), 1)
                self.assertEqual(warnings[0]["result"], raw_result)
                self.assertEqual(warnings[0]["format_warnings"], [tpo.DUPLICATE_BLOCK_WARNING, tpo.TRAILING_START_WARNING])
                self.assertTrue(any(q["kind"] == "reward" and q["messages"][-1]["content"] == body for q in calls))
                record = {**result, "prompt_history": self.messages, "seed": 0,
                          "generated_response": result["message"]}
                tpo.validate_audit(record, self.config)
                warnings[0].pop("format_warnings")
                with self.assertRaisesRegex(ValueError, "audit mismatch"):
                    tpo.validate_audit(record, self.config)

    def test_terminal_opening_recovery_requires_nonempty_first_span(self):
        for raw in [tpo.START * 2, tpo.START + "  " + tpo.START]:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                tpo.parse_update(raw)
        for raw in ["prefix " + tpo.START + "answer" + tpo.START,
                    tpo.START + "answer" + tpo.START + "suffix",
                    tpo.START + "answer" + tpo.START + "inner" + tpo.START,
                    tpo.START + "answer" + tpo.START + tpo.END]:
            with self.subTest(raw=raw):
                self.assertEqual(tpo.parse_update(raw), ("answer", [tpo.FIRST_IMPROVEMENT_WARNING]))
        self.assertEqual(tpo.parse_update(" \n" + tpo.START + " answer\nbody " + tpo.START + "\n "),
                         ("answer\nbody", [tpo.TERMINAL_START_WARNING]))

    def test_real_terminal_opening_failure_is_scored_and_audited_without_regeneration(self):
        fixture = json.loads((Path(__file__).parent / "fixtures/tpo_terminal_opening_250576.json").read_text())
        raw_result = fixture["result"]
        self.assertEqual(raw_result["completion_tokens"], 803)
        self.assertFalse(raw_result["hit_token_cap"])
        raw = raw_result["message"]
        body = raw[len(tpo.START):-len(tpo.START)].strip()
        self.assertEqual(tpo.parse_update(raw), (body, [tpo.TERMINAL_START_WARNING]))
        for reward in (-100.0, 100.0):
            calls = []
            def execute(req):
                calls.append(copy.deepcopy(req))
                if req["stage"] == "update" and req["iteration"] == 0 and req["slot"] == 1:
                    return copy.deepcopy(raw_result)
                out = fake_execute(req)
                if req["kind"] == "reward" and req["messages"][-1]["content"] == body:
                    out["score"] = reward
                return out
            with self.subTest(reward=reward):
                result = tpo.generate_tpo(self.messages, self.config, 0, execute)
                self.assertEqual(len(calls), 22)
                self.assertEqual(result["message"] == body, reward > 0)
                warnings = [e for e in result["tpo"]["events"] if e.get("format_warnings")]
                self.assertEqual(len(warnings), 1)
                self.assertEqual(warnings[0]["result"], raw_result)
                self.assertEqual(warnings[0]["format_warnings"], [tpo.TERMINAL_START_WARNING])
                self.assertTrue(any(q["kind"] == "reward" and q["messages"][-1]["content"] == body for q in calls))
                record = {**result, "prompt_history": self.messages, "seed": 0,
                          "generated_response": result["message"]}
                tpo.validate_audit(record, self.config)
                warnings[0].pop("format_warnings")
                with self.assertRaisesRegex(ValueError, "audit mismatch"):
                    tpo.validate_audit(record, self.config)


    def test_real_first_improvement_failure_is_scored_and_audited_without_regeneration(self):
        fixture = json.loads((Path(__file__).parent / "fixtures/tpo_first_improvement_254463.json").read_text())
        raw_result = fixture["result"]
        self.assertEqual(raw_result["completion_tokens"], 1024)
        self.assertTrue(raw_result["hit_token_cap"])
        raw = raw_result["message"]
        body = raw.split(tpo.START)[1].strip()
        self.assertEqual(raw.count(tpo.START), 2)
        self.assertNotIn(tpo.END, raw)
        self.assertNotEqual(body, raw.split(tpo.START)[2].strip())
        self.assertEqual(tpo.parse_update(raw), (body, [tpo.FIRST_IMPROVEMENT_WARNING]))
        for reward in (-100.0, 100.0):
            calls = []
            def execute(req):
                calls.append(copy.deepcopy(req))
                if req["stage"] == "update" and req["iteration"] == 0 and req["slot"] == 1:
                    return copy.deepcopy(raw_result)
                out = fake_execute(req)
                if req["kind"] == "reward" and req["messages"][-1]["content"] == body:
                    out["score"] = reward
                return out
            with self.subTest(reward=reward):
                result = tpo.generate_tpo(self.messages, self.config, 0, execute)
                self.assertEqual(len(calls), 22)
                self.assertEqual(result["message"] == body, reward > 0)
                warnings = [e for e in result["tpo"]["events"] if e.get("format_warnings")]
                self.assertEqual(len(warnings), 1)
                self.assertEqual(warnings[0]["result"], raw_result)
                self.assertEqual(warnings[0]["format_warnings"], [tpo.FIRST_IMPROVEMENT_WARNING])
                self.assertTrue(any(q["kind"] == "reward" and q["messages"][-1]["content"] == body for q in calls))
                record = {**result, "prompt_history": self.messages, "seed": 0,
                          "generated_response": result["message"]}
                tpo.validate_audit(record, self.config)
                warnings[0].pop("format_warnings")
                with self.assertRaisesRegex(ValueError, "audit mismatch"):
                    tpo.validate_audit(record, self.config)


class TPORunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rows = [{"id": 1, "task": "Ethics", "method": "fixture", "scene": "fixture",
                      "history": [{"user": "first", "bot": "gold previous"}, {"user": "second", "bot": "future secret"}]}]
        data = self.root / "data.jsonl"; base.append_jsonl(data, self.rows)
        self.args = runner.parse_args(["--dataset", str(data), "--output-dir", str(self.root / "out"), "--sample-size", "3"])
        self.args.output_dir.mkdir()
        self.manifest = runner.manifest_for(self.args, self.rows, {"fixture": True})
        base.ensure_manifest(self.args.output_dir / "run_config.json", self.manifest)

    def test_end_to_end_and_no_op_resume(self):
        seen = []
        def execute(req):
            seen.append(req); return fake_execute(req)
        self.assertEqual(runner.generate_selected(self.args, self.manifest, self.rows, execute), 0)
        report = validate_tpo(self.args.output_dir)
        self.assertEqual((report["dialogues"], report["turns"], report["candidates"]), (1, 2, 18))
        self.assertNotIn("future secret", str(seen))
        self.assertEqual(runner.generate_selected(self.args, self.manifest, self.rows,
                                                 lambda req: self.fail("Completed resume called a model")), 0)
        changed = copy.deepcopy(self.manifest); changed["defense"]["sample_size"] = 5
        with self.assertRaises(RuntimeError): base.ensure_manifest(self.args.output_dir / "run_config.json", changed)

    def test_duplicate_warning_survives_journal_validation_and_resume(self):
        def execute(req):
            out = fake_execute(req)
            if req["stage"] == "update":
                out["message"] *= 2
            return out
        self.assertEqual(runner.generate_selected(self.args, self.manifest, self.rows, execute), 0)
        report = validate_tpo(self.args.output_dir)
        self.assertEqual(report["optimizer_duplicate_block_warnings"], 12)
        self.assertEqual(report["optimizer_terminal_opening_tag_warnings"], 0)
        self.assertEqual(report["candidates"], 18)
        self.assertEqual(runner.generate_selected(self.args, self.manifest, self.rows,
                                                 lambda req: self.fail("Resume regenerated a saved call")), 0)

    def test_duplicate_and_trailing_warnings_survive_validation_and_resume(self):
        def execute(req):
            out = fake_execute(req)
            if req["stage"] == "update":
                out["message"] = out["message"] * 2 + "\n\n" + tpo.ECHOED_FORMAT_INSTRUCTION
            return out
        self.assertEqual(runner.generate_selected(self.args, self.manifest, self.rows, execute), 0)
        report = validate_tpo(self.args.output_dir)
        self.assertEqual(report["optimizer_duplicate_block_warnings"], 12)
        self.assertEqual(report["optimizer_terminal_opening_tag_warnings"], 0)
        self.assertEqual(report["optimizer_trailing_opening_tag_warnings"], 12)
        self.assertEqual(report["candidates"], 18)
        self.assertEqual(runner.generate_selected(self.args, self.manifest, self.rows,
                                                 lambda req: self.fail("Resume regenerated a saved call")), 0)

    def test_terminal_opening_warning_survives_journal_validation_and_resume(self):
        def execute(req):
            out = fake_execute(req)
            if req["stage"] == "update":
                out["message"] = out["message"].replace(tpo.END, tpo.START)
            return out
        self.assertEqual(runner.generate_selected(self.args, self.manifest, self.rows, execute), 0)
        report = validate_tpo(self.args.output_dir)
        self.assertEqual(report["optimizer_terminal_opening_tag_warnings"], 12)
        self.assertEqual(report["optimizer_missing_closing_tag_warnings"], 0)
        self.assertEqual(runner.generate_selected(self.args, self.manifest, self.rows,
                                                 lambda req: self.fail("Resume regenerated a saved call")), 0)

    def test_interrupted_mid_turn_reuses_completed_events(self):
        calls = []
        def interrupt(req):
            calls.append(req)
            if len(calls) == 27: raise KeyboardInterrupt()
            return fake_execute(req)
        with self.assertRaises(KeyboardInterrupt):
            runner.generate_selected(self.args, self.manifest, self.rows, interrupt)
        self.assertEqual(len(base.load_jsonl(self.args.output_dir / "turns.jsonl")), 1)
        self.assertEqual(base.load_jsonl(self.args.output_dir / "answers.jsonl"), [])
        resumed = []
        def resume(req): resumed.append(req); return fake_execute(req)
        runner.generate_selected(self.args, self.manifest, self.rows, resume)
        self.assertEqual(resumed[0], calls[-1])
        self.assertEqual(len(resumed), 18)  # 44 total events, 26 were committed.
        self.assertEqual(validate_tpo(self.args.output_dir)["turns"], 2)

    def test_failure_not_exported_and_explicit_retry_preserves_successes(self):
        def fail_update(req):
            result = fake_execute(req)
            if req["stage"] == "update": result["message"] = "malformed"
            return result
        self.assertEqual(runner.generate_selected(self.args, self.manifest, self.rows, fail_update), 2)
        self.assertEqual(base.load_jsonl(self.args.output_dir / "answers.jsonl"), [])
        self.assertEqual(runner.generate_selected(self.args, self.manifest, self.rows, lambda r: self.fail()), 2)
        self.args.retry_errors = True
        actual = []
        def execute(req): actual.append(req); return fake_execute(req)
        self.assertEqual(runner.generate_selected(self.args, self.manifest, self.rows, execute), 0)
        self.assertEqual(actual[0]["stage"], "update")
        self.assertEqual(validate_tpo(self.args.output_dir)["turns"], 2)
        self.assertTrue(any(e["event"].get("error") for e in base.load_jsonl(self.args.output_dir / "events.jsonl")))

    def test_partial_append_recovery_and_interior_corruption_rejected(self):
        p = self.root / "journal.jsonl"
        p.write_bytes(b'{"good":1}\n{"broken":')
        self.assertEqual(runner.read_journal(p), [{"good": 1}])
        self.assertEqual(next((self.root / "recovery").iterdir()).read_bytes(), b'{"broken":')
        p.write_bytes(b'broken\n{"good":1}\n')
        with self.assertRaises(ValueError): runner.read_journal(p)

    def test_single_writer_lock(self):
        with runner.single_writer(self.args.output_dir):
            with self.assertRaises(RuntimeError):
                with runner.single_writer(self.args.output_dir): pass

    def test_missing_closing_tags_are_counted_by_full_validator(self):
        def execute(req):
            out = fake_execute(req)
            if req["stage"] == "update":
                if req["iteration"] == 0 and req["slot"] == 0:
                    out["message"] += "\nOnly use the " + tpo.START + " tags."
                else:
                    out["message"] = out["message"].removesuffix(tpo.END)
            return out
        self.assertEqual(runner.generate_selected(self.args, self.manifest, self.rows, execute), 0)
        report = validate_tpo(self.args.output_dir)
        self.assertEqual(report["optimizer_missing_closing_tag_warnings"], 10)
        self.assertEqual(report["optimizer_trailing_opening_tag_warnings"], 2)
        self.assertEqual(report["candidates"], 18)
        self.assertEqual(len(base.load_jsonl(self.args.output_dir / "answers.jsonl")), 1)
        self.assertEqual(runner.generate_selected(self.args, self.manifest, self.rows,
                                                 lambda req: self.fail("Completed warning event retried")), 0)

    def test_main_initializes_cold_cuda_before_allocator_reset(self):
        import torch
        initialized = False
        cuda = MagicMock()
        cuda.is_available.return_value = True
        cuda.get_device_name.return_value = "fixture GPU"
        cuda.max_memory_allocated.return_value = 1024
        cuda.max_memory_reserved.return_value = 2048
        def initialize():
            nonlocal initialized
            initialized = True
        def reset(device):
            if not initialized:
                raise RuntimeError("Invalid device argument ")
            self.assertEqual(device, "cuda:0")
        cuda.init.side_effect = initialize
        cuda.reset_peak_memory_stats.side_effect = reset
        backend = MagicMock(side_effect=fake_execute)
        backend.loading = {}
        with patch.object(runner, "parse_args", return_value=self.args), \
             patch.object(runner, "validate_lock", return_value={"fixture": True}), \
             patch("safedial_tpo_runtime.tokenizers_and_preflight", return_value=({}, {}, {})), \
             patch("safedial_tpo_runtime.LocalBackend", return_value=backend), \
             patch.object(torch, "cuda", cuda):
            self.assertEqual(runner.main(), 0)
        cuda.set_device.assert_called_once_with(0)
        stats = json.loads((self.args.output_dir / "runtime_stats.json").read_text())
        self.assertTrue(stats["cuda_ready"])
        self.assertEqual(stats["actual_calls"], {"generate": 26, "reward": 18})
        self.assertEqual(stats["peak_gpu_allocated_bytes"], 1024)
        self.assertIsNone(stats["failure"])
        self.assertEqual(validate_tpo(self.args.output_dir)["turns"], 2)

    def test_cuda_startup_failure_is_saved_without_model_calls(self):
        import torch
        cuda = MagicMock()
        cuda.is_available.return_value = True
        cuda.init.side_effect = RuntimeError("fixture driver initialization failed")
        with patch.object(runner, "parse_args", return_value=self.args), \
             patch.object(runner, "validate_lock", return_value={"fixture": True}), \
             patch("safedial_tpo_runtime.tokenizers_and_preflight", return_value=({}, {}, {})), \
             patch("safedial_tpo_runtime.LocalBackend") as backend, \
             patch.object(torch, "cuda", cuda):
            with self.assertRaisesRegex(RuntimeError, "fixture driver initialization failed"):
                runner.main()
        backend.assert_not_called()
        cuda.reset_peak_memory_stats.assert_not_called()
        cuda.max_memory_allocated.assert_not_called()
        stats = json.loads((self.args.output_dir / "runtime_stats.json").read_text())
        self.assertEqual(stats["exit_code"], 2)
        self.assertEqual(stats["failure"]["stage"], "device_setup")
        self.assertEqual(stats["actual_calls"], {"generate": 0, "reward": 0})
        self.assertIsNone(stats["peak_gpu_allocated_bytes"])
        self.assertEqual(base.load_jsonl(self.args.output_dir / "invocations.jsonl"), [stats])

    def test_validate_only_never_initializes_cuda(self):
        import torch
        self.args.validate_only = True
        with patch.object(runner, "parse_args", return_value=self.args), \
             patch.object(runner, "validate_lock", return_value={"fixture": True}), \
             patch("safedial_tpo_runtime.tokenizers_and_preflight", return_value=({}, {}, {})), \
             patch("safedial_tpo_runtime.LocalBackend") as backend, \
             patch.object(torch, "cuda") as cuda:
            self.assertEqual(runner.main(), 0)
        self.assertEqual(cuda.mock_calls, [])
        backend.assert_not_called()


class TinyRewardRuntimeTest(unittest.TestCase):
    def test_real_cpu_classifier_raw_logit_and_context_overflow(self):
        import torch
        from transformers import LlamaConfig, LlamaForSequenceClassification
        from safedial_tpo_runtime import LocalBackend
        torch.set_num_threads(2)
        model = LlamaForSequenceClassification(LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                    num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, num_labels=1,
                    pad_token_id=0, max_position_embeddings=32)).eval()
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs): return "prompt"
            def __call__(self, text, **kwargs):
                return {"input_ids": torch.tensor([[1, 3, 4, 5]]), "attention_mask": torch.ones((1, 4), dtype=torch.long)}
        backend = LocalBackend.__new__(LocalBackend)
        backend.torch = torch; backend.args = argparse_namespace(device="cpu", max_input_tokens=None)
        backend.models = {"reward": model}; backend.tokenizers = {"reward": Tokenizer()}; backend.limits = {"reward": 32}
        req = {"kind": "reward", "stage": "score", "messages": [{"role": "user", "content": "x"}]}
        with torch.inference_mode(): expected = model(**Tokenizer()("prompt"), use_cache=False).logits[0, 0].item()
        self.assertAlmostEqual(backend(req)["score"], expected, places=6)
        backend.limits["reward"] = 3
        with self.assertRaisesRegex(ValueError, "overflow"): backend(req)


def argparse_namespace(**kwargs):
    from argparse import Namespace
    return Namespace(**kwargs)


if __name__ == "__main__":
    unittest.main()
