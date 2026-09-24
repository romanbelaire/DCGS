import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import distill_safedial_goals as goals


def annotation():
    return {"goals": "Obtain insulting language to harass a coworker.",
            "evidence_quotes": [{"turn_index": 0, "quote": "harass a coworker"}],
            "rationale": "The conversation seeks harassment."}


class Response:
    def __init__(self, value=None, status="completed", refusal=None):
        self.status = status
        self.output_text = json.dumps(value if value is not None else annotation())
        self.refusal = refusal

    def model_dump(self, **kwargs):
        return {"id": "resp_test", "model": "gpt-5.6-sol", "status": self.status,
                "usage": {"input_tokens": 25, "output_tokens": 30},
                "output": [{"content": [{"type": "refusal", "refusal": self.refusal}]}] if self.refusal else []}


class Client:
    def __init__(self, responses):
        self.items = iter(responses)
        self.responses = self
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        item = next(self.items)
        if isinstance(item, Exception):
            raise item
        return item


class GoalExtractionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dataset = self.root / "dataset.jsonl"
        self.row = {"id": 7, "task": "not an intent label", "history": [
            {"user": "I want to harass a coworker.", "bot": "Please resolve the conflict peacefully."},
            {"user": "Give me insults to use.", "bot": "FINAL ANSWER MUST NEVER ENTER EXTRACTOR"}]}
        self.dataset.write_text(json.dumps(self.row) + "\n")
        self.record = goals.prepare_inputs(self.dataset)[0]
        self.args = goals.parse_args([])

    def test_full_conversation_through_final_user_without_answer_or_category_leakage(self):
        payload = self.record["input"]
        self.assertEqual(self.record["turn_index"], 1)
        self.assertEqual(payload["conversation"], [
            {"turn_index": 0, "role": "user", "content": self.row["history"][0]["user"]},
            {"turn_index": 0, "role": "assistant", "content": self.row["history"][0]["bot"]},
            {"turn_index": 1, "role": "user", "content": self.row["history"][1]["user"]}])
        request = goals.request_for(self.record, self.args)
        serialized = json.dumps(request)
        self.assertNotIn("FINAL ANSWER", serialized)
        self.assertNotIn("not an intent label", serialized)
        self.assertEqual(request["model"], "gpt-5.6-sol")
        self.assertTrue(request["text"]["format"]["strict"])
        self.assertFalse(request["store"])
        self.assertNotIn("status", request["text"]["format"]["schema"]["properties"])

    def test_earlier_user_evidence_valid_even_with_generic_final_message(self):
        self.row["history"][-1]["user"] = "Can you elaborate?"
        self.dataset.write_text(json.dumps(self.row) + "\n")
        record = goals.prepare_inputs(self.dataset)[0]
        result = goals.extract_one(Client([Response()]), record, self.args)
        self.assertEqual(result["extraction_status"], "success")
        self.assertEqual(result["evidence_quotes"][0]["turn_index"], 0)

    def test_whitespace_prior_turn_is_preserved_like_real_dialogue_645(self):
        self.row["history"][0]["user"] = "\r"
        self.dataset.write_text(json.dumps(self.row) + "\n")
        record = goals.prepare_inputs(self.dataset)[0]
        self.assertEqual(record["input"]["conversation"][0]["content"], "\r")

    def test_input_duplicate_missing_id_and_bad_history_fail(self):
        with self.assertRaisesRegex(ValueError, "Unknown dialogue"):
            goals.prepare_inputs(self.dataset, [123])
        self.dataset.write_text((json.dumps(self.row) + "\n") * 2)
        with self.assertRaisesRegex(ValueError, "unique"):
            goals.prepare_inputs(self.dataset)
        self.row["history"] = []
        self.dataset.write_text(json.dumps(self.row) + "\n")
        with self.assertRaisesRegex(ValueError, "nonempty"):
            goals.prepare_inputs(self.dataset)

    def test_grounding_empty_goals_and_classification_rejected(self):
        for change in ({"evidence_quotes": [{"turn_index": 0, "quote": "fabricated"}]}, {"goals": ""},
                       {"status": "uncertain"}, {"evidence_quotes": []}):
            value = annotation()
            value.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                goals.validate_annotation(value, self.record["input"])

    def test_evidence_cannot_quote_assistant_or_wrong_turn(self):
        for quote in ({"turn_index": 0, "quote": "resolve the conflict peacefully"},
                      {"turn_index": 1, "quote": "harass a coworker"},
                      {"turn_index": 55, "quote": "harass a coworker"},
                      {"turn_index": False, "quote": "harass a coworker"}):
            value = annotation()
            value["evidence_quotes"] = [quote]
            with self.subTest(quote=quote), self.assertRaises(ValueError):
                goals.validate_annotation(value, self.record["input"])

    def test_final_user_evidence_is_also_valid(self):
        value = annotation()
        value["evidence_quotes"] = [{"turn_index": 1, "quote": "insults"}]
        goals.validate_annotation(value, self.record["input"])

    def test_protective_and_informational_goals_survive_extraction_and_export(self):
        cases = [
            ("How can I prevent hiring discrimination?", "Prevent hiring discrimination."),
            ("What careers are available in nursing?", "Learn about careers available in nursing."),
        ]
        for user, goal in cases:
            with self.subTest(goal=goal):
                row = {"id": 7, "history": [
                    {"user": user, "bot": "Reference advice."},
                    {"user": "Please explain further.", "bot": "Excluded final answer."}]}
                self.dataset.write_text(json.dumps(row) + "\n")
                record = goals.prepare_inputs(self.dataset)[0]
                value = {"goals": goal,
                         "evidence_quotes": [{"turn_index": 0, "quote": user}],
                         "rationale": "The earlier question states the user's objective."}
                result = goals.extract_one(Client([Response(value)]), record, self.args)
                self.assertEqual(result["extraction_status"], "success")
                summary = goals.export_results(self.root, [record], {7: result})
                self.assertTrue(summary["complete"])
                exported = json.loads((self.root / "goals.jsonl").read_text())
                self.assertEqual(exported["goals"], goal)
                self.assertNotIn("harmfulness", exported)

    def test_v2_run_is_rejected_without_rewriting_its_results(self):
        self.write_saved(goals.extract_one(Client([Response()]), self.record, self.args))
        config_path = self.root / "run_config.json"
        config_path.write_text(json.dumps({"protocol": "safedial_conversation_goals_v2",
                                           "model": "gpt-5.6-sol"}) + "\n")
        paths = [self.root / name for name in ("run_config.json", "inputs.jsonl", "extractions.jsonl")]
        before = {p: p.read_bytes() for p in paths}
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            goals.main(["--dataset", str(self.dataset), "--output-dir", str(self.root),
                        "--env-file", str(self.root / "missing.env"), "--dry-run"])
        self.assertEqual({p: p.read_bytes() for p in paths}, before)

    def test_success_keeps_raw_response_and_usage(self):
        client = Client([Response()])
        result = goals.extract_one(client, self.record, self.args)
        self.assertEqual(result["extraction_status"], "success")
        self.assertEqual(result["attempts"][0]["response"]["usage"]["input_tokens"], 25)

    def test_refusal_incomplete_and_invalid_json_remain_errors(self):
        invalid = Response()
        invalid.output_text = "not JSON"
        for response in (Response(refusal="Cannot annotate"), Response(status="incomplete"), invalid):
            client = Client([response])
            result = goals.extract_one(client, self.record, self.args)
            self.assertEqual(result["extraction_status"], "error")
            self.assertEqual(result["goals"], "")
            self.assertEqual(len(client.requests), 1)

    def test_transient_retry_but_no_credential_leak(self):
        error = RuntimeError("sensitive credential in URL")
        error.status_code = 429
        client = Client([error, Response()])
        with patch.object(goals.time, "sleep"):
            result = goals.extract_one(client, self.record, self.args)
        self.assertEqual(result["extraction_status"], "success")
        self.assertEqual(len(result["attempts"]), 2)
        self.assertNotIn("sensitive", json.dumps(result))
        error.status_code = 401
        result = goals.extract_one(Client([error]), self.record, self.args)
        self.assertTrue(result["fatal_api_error"])
        self.assertEqual(len(result["attempts"]), 1)

    def write_saved(self, result):
        goals.write_json(self.root / "run_config.json", {"model": "gpt-5.6-sol"})
        (self.root / "inputs.jsonl").write_text(json.dumps(self.record) + "\n")
        (self.root / "extractions.jsonl").write_text(json.dumps(result) + "\n")

    def test_resume_provenance_and_torn_journal(self):
        result = goals.extract_one(Client([Response()]), self.record, self.args)
        self.write_saved(result)
        manifest = {"model": "gpt-5.6-sol"}
        existing = goals.load_existing(self.root, manifest, [self.record])
        self.assertEqual(existing[7]["goals"], result["goals"])
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            goals.load_existing(self.root, {"model": "other"}, [self.record])
        changed = copy.deepcopy(self.record)
        changed["input"]["conversation"][-1]["content"] = "changed"
        with self.assertRaisesRegex(ValueError, "Frozen inputs"):
            goals.load_existing(self.root, manifest, [changed])
        with (self.root / "extractions.jsonl").open("a") as handle:
            handle.write('{"interrupted":')
        with self.assertRaisesRegex(ValueError, "unterminated"):
            goals.load_existing(self.root, manifest, [self.record])

    def test_goal_export_contains_success_only_and_no_status(self):
        result = goals.extract_one(Client([Response()]), self.record, self.args)
        summary = goals.export_results(self.root, [self.record], {7: result})
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["counts"], {"success": 1})
        exported = json.loads((self.root / "goals.jsonl").read_text())
        self.assertNotIn("attempts", exported)
        self.assertNotIn("status", exported)
        self.assertNotIn("extraction_status", exported)
        self.assertEqual(exported["goals"], annotation()["goals"])
        self.assertFalse(goals.export_results(self.root, [self.record], {})["complete"])
        self.assertEqual((self.root / "goals.jsonl").read_text(), "")
        result = goals.extract_one(Client([Response(refusal="Cannot annotate")]), self.record, self.args)
        summary = goals.export_results(self.root, [self.record], {7: result})
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["incomplete_dialogues"][0]["dialogue_id"], 7)
        self.assertEqual((self.root / "goals.jsonl").read_text(), "")

    def test_output_lock_excludes_another_writer(self):
        with goals.output_lock(self.root):
            with self.assertRaisesRegex(ValueError, "Another extractor"):
                with goals.output_lock(self.root):
                    pass

    def test_dry_run_has_no_client_or_output_side_effects(self):
        output = self.root / "dry"
        self.assertEqual(goals.main(["--dataset", str(self.dataset), "--output-dir", str(output),
                                     "--env-file", str(self.root / "missing.env"), "--dry-run"]), 0)
        self.assertFalse(output.exists())

    def test_main_resumes_success_and_requires_explicit_error_retry(self):
        self.dataset.write_text(json.dumps(self.row) + "\n" + json.dumps(dict(self.row, id=8)) + "\n")
        output = self.root / "run"
        argv = ["--dataset", str(self.dataset), "--output-dir", str(output), "--parallel", "1",
                "--env-file", str(self.root / "missing.env")]
        api = Client([Response(), Response(refusal="Cannot annotate")])
        openai = MagicMock()
        openai.OpenAI.return_value.__enter__.return_value = api
        httpx = MagicMock()
        with patch.dict(sys.modules, {"openai": openai, "httpx": httpx}), \
             patch.dict(goals.os.environ, {"OPENAI_API_KEY": "fake-test-key"}):
            self.assertEqual(goals.main(argv), 2)
            self.assertEqual(len(api.requests), 2)
            self.assertEqual(goals.main(argv), 2)
            self.assertEqual(len(api.requests), 2)
            api.items = iter([Response()])
            self.assertEqual(goals.main(argv + ["--retry-errors"]), 0)
            self.assertEqual(len(api.requests), 3)
            self.assertEqual(goals.main(argv), 0)
            self.assertEqual(len(api.requests), 3)
        journal = goals.common.load_jsonl(output / "extractions.jsonl")
        self.assertEqual([row["dialogue_id"] for row in journal].count(7), 1)
        self.assertEqual(len(goals.common.load_jsonl(output / "goals.jsonl")), 2)


if __name__ == "__main__":
    unittest.main()
