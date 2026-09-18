import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "export_safedial_adjudication",
    Path(__file__).resolve().parents[1] / "scripts/export_safedial_adjudication.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class AdjudicationQueueTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        inputs = {
            "answers": {"id": 7, "model_id": "actor", "choices": [{"turns": [{"message": "answer"}]}]},
            "dataset": {"id": 7},
            "prompts": {"name": "rubric", "system_prompt": "judge instructions"},
        }
        manifest = {}
        for name, value in inputs.items():
            path = self.root / f"{name}.jsonl"
            path.write_text(json.dumps(value) + "\n")
            manifest[name] = str(path)
            manifest[f"{name}_sha256"] = MODULE.file_hash(path)
        (self.root / "judge_config.json").write_text(json.dumps(manifest))
        self.judgment = {
            "dialogue_id": 7, "choice_index": 0, "turn_index": 0,
            "judge_model": "judge", "rubric": "rubric", "status": "error",
            "conversation_context": "reference history then current answer",
            "user_prompt": "evaluation request", "attempts": 1,
            "error": "parse failure", "raw_judgment": "Cannot evaluate",
        }
        self.save_judgment()

    def save_judgment(self):
        (self.root / "judgments.jsonl").write_text(json.dumps(self.judgment) + "\n")

    def test_exports_failure_with_response_context_and_empty_human_scores(self):
        before = MODULE.file_hash(self.root / "judgments.jsonl")
        report = MODULE.export_queue(self.root)
        case = report["cases"][0]
        self.assertEqual(report["pending_human_count"], 1)
        self.assertEqual(case["model_answer"], "answer")
        self.assertEqual(case["judge_responses"][0]["raw_judgment"], "Cannot evaluate")
        self.assertEqual(case["turn_number"], 1)
        self.assertIsNone(case["human_adjudication"]["handling_score"])
        self.assertEqual(before, MODULE.file_hash(self.root / "judgments.jsonl"))
        self.assertFalse((self.root / "aggregate.json").exists())

    def test_reexport_preserves_review_and_is_idempotent(self):
        report = MODULE.export_queue(self.root)
        path = self.root / "human_adjudication.json"
        human = report["cases"][0]["human_adjudication"]
        human.update(status="completed", reviewer="test reviewer", notes="retain me", handling_score=4)
        path.write_text(json.dumps(report))
        refreshed = MODULE.export_queue(self.root)
        self.assertEqual(refreshed["cases"][0]["human_adjudication"], human)
        self.assertEqual(refreshed["pending_human_count"], 0)
        self.assertEqual(MODULE.export_queue(self.root), refreshed)

    def test_deduplicates_same_archived_and_embedded_attempt(self):
        attempt = {"attempt": 1, "raw_judgment": "refusal", "tstamp": 1}
        self.judgment["failed_attempts"] = [attempt]
        self.save_judgment()
        archive = {**{k: self.judgment[k] for k in MODULE.KEY_FIELDS}, **attempt}
        (self.root / "failed_attempts.jsonl").write_text(json.dumps(archive) + "\n")
        report = MODULE.export_queue(self.root)
        self.assertEqual(len(report["cases"][0]["judge_responses"]), 1)

    def test_automatic_success_keeps_review_history(self):
        MODULE.export_queue(self.root)
        self.judgment["status"] = "success"
        self.save_judgment()
        report = MODULE.export_queue(self.root)
        self.assertEqual(report["case_count"], 1)
        self.assertEqual(report["unresolved_automated_count"], 0)
        self.assertEqual(report["cases"][0]["judge_responses"][0]["raw_judgment"], "Cannot evaluate")

    def test_successful_cases_not_added(self):
        self.judgment["status"] = "success"
        self.save_judgment()
        self.assertEqual(MODULE.export_queue(self.root)["case_count"], 0)

    def test_missing_raw_response_is_explicit(self):
        self.judgment.pop("raw_judgment")
        self.save_judgment()
        report = MODULE.export_queue(self.root)
        self.assertIsNone(report["cases"][0]["judge_responses"][0]["raw_judgment"])

    def test_changed_source_hash_rejected(self):
        (self.root / "answers.jsonl").write_text("changed")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            MODULE.export_queue(self.root)


if __name__ == "__main__":
    unittest.main()
