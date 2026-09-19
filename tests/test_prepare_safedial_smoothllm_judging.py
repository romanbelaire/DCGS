"""CPU-only subset audit; no inference, network, or judge calls."""
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_safedial_smoothllm_judging as prep
import run_safedial_baseline as base
import run_safedial_smoothllm_turn_resume as runner
import test_safedial_smoothllm_turn_resume as fixtures
from test_safedial_smoothllm import decoded


class JudgingPreparationTests(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.TurnResumeTests()
        fixture.setUp()
        self.addCleanup(fixture.temp.cleanup)
        self.fixture = fixture
        self.source = fixture.args.output_dir
        self.target = fixture.root / "prepared"

    def fail_one_turn(self):
        calls = []
        def generate(messages, seed):
            calls.append(seed)
            return decoded("") if len(calls) == 9 else decoded()
        self.assertEqual(self.fixture.run_fixture(generate), 1)

    def prepare(self, allowed=(1,)):
        # Importing or calling model/API code is not needed by preparation.
        with patch.dict(sys.modules, {"torch": None, "transformers": None, "openai": None}):
            return prep.prepare(self.source, self.target, allowed_failed_ids=allowed)

    def test_failed_dialogue_excluded_with_provenance_and_unchanged_source(self):
        self.fail_one_turn()
        before = {name: (self.source / name).read_bytes() for name in prep.SOURCE_NAMES}
        report = self.prepare()
        self.assertEqual((report["included_dialogues"], report["included_turns"], report["expected_turns"]), (1, 2, 4))
        self.assertEqual(report["excluded_dialogue_ids"], [1])
        self.assertFalse(report["generation_complete_without_failures"])
        self.assertEqual(report["failed_turns"][0]["turn_index"], 1)
        original = base.load_jsonl(self.source / "answers.jsonl")
        self.assertEqual(base.load_jsonl(self.target / "answers.jsonl"), [original[1]])
        self.assertEqual(before, {name: (self.source / name).read_bytes() for name in prep.SOURCE_NAMES})
        prepared = {p.name: p.read_bytes() for p in self.target.iterdir()}
        self.assertEqual(self.prepare(), report)
        self.assertEqual(prepared, {p.name: p.read_bytes() for p in self.target.iterdir()})

    def test_repaired_dialogue_included_but_existing_judging_input_never_replaced(self):
        self.fail_one_turn()
        self.prepare()
        original = (self.target / "answers.jsonl").read_bytes()
        self.fixture.args.retry_errors = True
        self.assertEqual(self.fixture.run_fixture(), 0)
        with self.assertRaisesRegex(ValueError, "Prepared input changed"):
            self.prepare()
        self.assertEqual((self.target / "answers.jsonl").read_bytes(), original)
        self.target = self.fixture.root / "new-prepared"
        report = self.prepare()
        self.assertEqual(report["excluded_dialogue_ids"], [])
        self.assertTrue(report["generation_complete_without_failures"])
        self.assertEqual(report["included_dialogues"], 2)

    def test_unreviewed_failure_and_missing_turns_refused(self):
        self.fail_one_turn()
        with self.assertRaisesRegex(ValueError, "Unreviewed failed dialogues"):
            self.prepare(allowed=(344,))
        records = base.load_jsonl(self.source / "turns.jsonl")
        runner.atomic_jsonl(self.source / "turns.jsonl", records[:-1])
        with self.assertRaisesRegex(ValueError, "Generation is still incomplete"):
            self.prepare()
        self.assertFalse((self.target / "answers.jsonl").exists())

    def test_answer_or_candidate_tampering_refused(self):
        self.fixture.run_fixture()
        path = self.source / "answers.jsonl"
        answers = base.load_jsonl(path)
        changed = copy.deepcopy(answers)
        changed[0]["choices"][0]["turns"][0]["message"] = "changed"
        runner.atomic_jsonl(path, changed)
        with self.assertRaisesRegex(ValueError, "Answer/turn journal mismatch"):
            self.prepare()
        runner.atomic_jsonl(path, answers)
        path = self.source / "turns.jsonl"
        records = base.load_jsonl(path)
        records[0]["smoothllm"]["candidates"][0]["perturbed_user_message"] = "changed"
        runner.atomic_jsonl(path, records)
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertFalse((self.target / "answers.jsonl").exists())

    def test_source_dataset_and_source_lock_checked_before_export(self):
        self.fixture.run_fixture()
        with runner.single_writer(self.source):
            with self.assertRaisesRegex(RuntimeError, "Another writer"):
                self.prepare()
        manifest_path = self.source / "run_config.json"
        original = manifest_path.read_text()
        manifest = json.loads(original)
        manifest["implementation_sha256"]["safedial_smoothllm.py"] = "changed"
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "source hashes changed"):
            self.prepare()
        manifest_path.write_text(original)
        with self.fixture.dataset.open("a") as handle:
            handle.write("\n")
        with self.assertRaisesRegex(ValueError, "Dataset hash mismatch"):
            self.prepare()
        self.assertFalse((self.target / "answers.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
