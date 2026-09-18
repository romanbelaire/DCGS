import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock


SPEC = importlib.util.spec_from_file_location(
    "judge_safedial", Path(__file__).resolve().parents[1] / "scripts/judge_safedial.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def response(content, finish_reason="stop", refusal=None):
    return NS(
        id="test-response", model="test-model",
        choices=[NS(message=NS(content=content, refusal=refusal), finish_reason=finish_reason)],
        usage=NS(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


class JudgeDiagnosticsTest(unittest.TestCase):
    def run_request(self, outcomes, diagnostics_path=None):
        client = NS(chat=NS(completions=NS(create=Mock(side_effect=outcomes))))
        result = MODULE.judge_request(
            client=client,
            question={"id": 1, "task": "Fairness", "history": [{"user": "fixture", "bot": "gold"}]},
            answer={"model_id": "actor", "choices": [{"turns": [{"message": "answer"}]}]},
            prompt={"name": "fair-evaluation", "system_prompt": "rubric", "prompt_template": "{dialogue}"},
            args=NS(choice_index=0, judge_model="gpt-4o-mini", temperature=0.7,
                    max_tokens=2048, seed=None, max_retries=len(outcomes), retry_backoff=0),
            turn_index=0, diagnostics_path=diagnostics_path,
        )
        return result, client.chat.completions.create

    def test_success_preserves_scores_and_request_settings(self):
        result, create = self.run_request([response("Score:[[1]]/[[2]]/[[3]]")])
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["handling_score"], 2)
        self.assertEqual(result["failed_attempts"], [])
        self.assertEqual(create.call_args.kwargs["max_tokens"], 2048)
        self.assertEqual(create.call_args.kwargs["temperature"], 0.7)
        self.assertNotIn("response_format", create.call_args.kwargs)

    def test_parse_failure_retains_raw_response_and_usage(self):
        result, _ = self.run_request([response("unfinished score", "length")])
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["raw_judgment"], "unfinished score")
        self.assertEqual(result["finish_reason"], "length")
        self.assertEqual(result["usage"]["total_tokens"], 15)
        self.assertIsNone(result["identification_score"])

    def test_refusal_retained_without_inventing_scores(self):
        result, _ = self.run_request([response(None, refusal="Cannot evaluate this content")])
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["refusal"], "Cannot evaluate this content")
        self.assertIsNone(result["handling_score"])

    def test_api_failure_does_not_reuse_previous_response(self):
        result, _ = self.run_request([response("malformed"), RuntimeError("network fixture")])
        self.assertIsNone(result["raw_judgment"])
        self.assertIsNone(result["usage"]["total_tokens"])
        self.assertEqual(result["failed_attempts"][0]["raw_judgment"], "malformed")
        self.assertEqual(len(result["failed_attempts"]), 2)

    def test_diagnostics_survive_success_and_subsequent_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "failed_attempts.jsonl"
            result, _ = self.run_request([response("bad"), response("Score:1/2/3")], path)
            self.assertEqual(result["status"], "success")
            self.assertEqual(len(result["failed_attempts"]), 1)
            self.run_request([response("another failure")], path)
            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([r["raw_judgment"] for r in records], ["bad", "another failure"])
            self.assertTrue(all(r["turn_index"] == 0 for r in records))


if __name__ == "__main__":
    unittest.main()
