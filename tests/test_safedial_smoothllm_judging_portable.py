"""Native Windows/Linux snapshot preparation; no network or paid API calls."""
import copy
import gzip
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_safedial_smoothllm_judging as prep


def jsonl(rows):
    return "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode()


class PortableJudgingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.snapshot, self.target = self.root / "snapshot", self.root / "prepared"
        self.snapshot.mkdir()
        self.dataset, self.prompts = self.root / "dataset.jsonl", self.root / "prompts.jsonl"
        rows = [{"id": i, "task": "Fairness", "method": "fixture",
                 "history": [{"user": "fixture", "bot": "gold"}]} for i in (1, 344)]
        self.dataset.write_bytes(jsonl(rows))
        self.prompts.write_bytes(jsonl([{"name": name, "system_prompt": "rubric",
                                        "prompt_template": "{dialogue}"}
                                       for name in set(prep.judge.TASK_TO_PROMPT.values())]))
        manifest = {"dataset_sha256": prep.base.file_sha256(self.dataset),
                    "protocol": prep.base.PROTOCOL, "num_choices": 1, "temperature": 0,
                    "defense": prep.defense_config(8, 10.0), "seed": 7, "model_id": "fixture",
                    "selected_ids": [1, 344], "implementation_sha256": {
                        name: prep.sha((prep.ROOT / "scripts" / name).read_bytes().replace(b"\r\n", b"\n"))
                        for name in prep.SOURCE_NAMES}}
        run_id = prep.base.stable_id(json.dumps(manifest, sort_keys=True))
        self.answers, self.statuses = [], []
        for i in (1, 344):
            turn = {"role": "assistant", "message": "answer" if i == 1 else "ERROR"}
            if i == 344:
                turn["error"] = "empty candidate"
            self.answers.append({"id": i, "model_id": "fixture", "task": "Fairness", "method": "fixture",
                                 "answer_id": prep.base.stable_id(run_id, i),
                                 "choices": [{"index": 0, "turns": [turn]}]})
            self.statuses.append({"dialogue_id": i, "turn_index": 0, "choice_index": 0,
                                  "seed": prep.base.turn_seed(7, "fixture", i, 0, 0),
                                  "generated_response_sha256": prep.sha(turn["message"].encode()),
                                  "error": turn.get("error")})
        self.index = {"snapshot_finished": "fixture", "files": []}
        self.prefix = "safedial_baseline/" + prep.RUN + "/"
        self.put(self.prefix + "run_config.json", json.dumps(manifest).encode() + b"\n")
        self.put(self.prefix + "answers.jsonl.gz", jsonl(self.answers))
        self.put(self.prefix + "turn_status.jsonl.gz", jsonl(self.statuses))
        reference = {"dataset_sha256": manifest["dataset_sha256"],
                     "prompts_sha256": prep.base.file_sha256(self.prompts)}
        self.put(prep.REFERENCE_CONFIG, json.dumps(reference).encode() + b"\n")

    def put(self, relative, data):
        path = self.snapshot / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        stored = gzip.compress(data, mtime=0) if relative.endswith(".gz") else data
        path.write_bytes(stored)
        self.index["files"] = [item for item in self.index["files"] if item["path"] != relative]
        self.index["files"].append({"path": relative, "sha256": prep.sha(stored),
                                    "uncompressed_sha256": prep.sha(data)})
        (self.snapshot / "snapshot.json").write_bytes(json.dumps(self.index).encode())

    def prepare(self):
        return prep.prepare_snapshot(self.snapshot, self.target, self.dataset, self.prompts)

    def test_subset_provenance_resume_and_source_preservation(self):
        before = {p.relative_to(self.snapshot): p.read_bytes() for p in self.snapshot.rglob("*") if p.is_file()}
        report = self.prepare()
        self.assertEqual((report["included_dialogues"], report["included_turns"], report["expected_turns"]), (1, 1, 2))
        self.assertEqual(report["excluded_dialogue_ids"], [344])
        self.assertTrue(report["snapshot_answer_audit_passed"])
        self.assertFalse(report["full_generation_audit_passed"])
        self.assertEqual(prep.judge.load_jsonl(self.target / "answers.jsonl"), self.answers[:1])
        self.assertEqual(self.prepare(), report)
        self.assertEqual(before, {p.relative_to(self.snapshot): p.read_bytes() for p in self.snapshot.rglob("*") if p.is_file()})
        (self.target / "answers.jsonl").write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Prepared input changed"):
            self.prepare()

    def test_corrupt_snapshot_and_answer_status_disagreement_refused(self):
        path = self.snapshot / self.prefix / "answers.jsonl.gz"
        path.write_bytes(path.read_bytes() + b"corrupt")
        with self.assertRaisesRegex(ValueError, "Snapshot checksum mismatch"):
            self.prepare()
        changed = copy.deepcopy(self.answers)
        changed[0]["choices"][0]["turns"][0]["message"] = "changed"
        self.put(self.prefix + "answers.jsonl.gz", jsonl(changed))
        with self.assertRaisesRegex(ValueError, "answer/status mismatch"):
            self.prepare()
        self.assertFalse((self.target / "answers.jsonl").exists())

    def test_missing_status_and_unreviewed_failure_refused(self):
        self.put(self.prefix + "turn_status.jsonl.gz", jsonl(self.statuses[:-1]))
        with self.assertRaisesRegex(ValueError, "missing turn statuses"):
            self.prepare()
        self.statuses[0]["error"] = "unexpected failure"
        self.answers[0]["choices"][0]["turns"][0]["error"] = "unexpected failure"
        self.put(self.prefix + "turn_status.jsonl.gz", jsonl(self.statuses))
        self.put(self.prefix + "answers.jsonl.gz", jsonl(self.answers))
        with self.assertRaisesRegex(ValueError, "Unreviewed failed dialogues"):
            self.prepare()

    def test_windows_line_endings_and_benchmark_mismatch(self):
        for entry in self.index["files"]:
            if not entry["path"].endswith(".gz"):
                path = self.snapshot / entry["path"]
                path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
        self.prepare()
        self.dataset.write_bytes(self.dataset.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "Benchmark hash mismatch"):
            self.prepare()

    def test_fetch_verifies_hash_before_writing(self):
        path = self.root / "download.jsonl"
        with patch.object(prep.urllib.request, "urlopen", return_value=io.BytesIO(b"wrong")):
            with self.assertRaisesRegex(ValueError, "Downloaded benchmark hash mismatch"):
                prep.benchmark_file(path, prep.sha(b"right"), prep.DATASET_REL, True)
        self.assertFalse(path.exists())
        with patch.object(prep.urllib.request, "urlopen", return_value=io.BytesIO(b"right")) as fetch:
            prep.benchmark_file(path, prep.sha(b"right"), prep.DATASET_REL, True)
        self.assertEqual(path.read_bytes(), b"right")
        self.assertIn(prep.BENCHMARK_REVISION, fetch.call_args.args[0])

    def test_lock_excludes_other_process_and_releases(self):
        path = self.root / "judge.lock"
        code = ("import sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); "
                "from run_safedial_smoothllm_judging import file_lock; "
                "lock = file_lock(Path(sys.argv[2])); lock.__enter__(); lock.__exit__(None, None, None)")
        command = [sys.executable, "-B", "-c", code, str(prep.ROOT / "scripts"), str(path)]
        with prep.file_lock(path):
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Another process holds the lock", result.stderr)
        self.assertEqual(subprocess.run(command, capture_output=True).returncode, 0)

    def test_dry_run_only_invokes_judge_validation(self):
        args = ["--snapshot", str(self.snapshot), "--output-dir", str(self.target),
                "--dataset", str(self.dataset), "--prompts", str(self.prompts), "--dry-run"]
        with patch.object(prep.subprocess, "run") as run, patch("sys.stdout", new=io.StringIO()):
            self.assertEqual(prep.main(args), 0)
        run.assert_called_once()
        self.assertIn("--dry-run", run.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
