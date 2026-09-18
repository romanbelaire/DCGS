import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_safedial_baseline as base
import run_safedial_smoothllm as legacy
import run_safedial_smoothllm_turn_resume as runner
from test_safedial_smoothllm import decoded
from validate_safedial_smoothllm_turn_resume import validate_turn_resume


class TurnResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rows = [{"id": i, "task": "Ethics", "method": "fixture", "scene": "fixture",
                      "history": [{"user": "first user", "bot": "gold previous"},
                                  {"user": "second user", "bot": "gold future"}]} for i in (1, 2)]
        self.dataset = self.root / "data.jsonl"
        base.append_jsonl(self.dataset, self.rows)
        self.args = runner.parse_args(["--dataset", str(self.dataset), "--output-dir", str(self.root / "out")])
        self.args.output_dir.mkdir()
        self.manifest = runner.manifest_for(self.args, self.rows)
        base.ensure_manifest(self.args.output_dir / "run_config.json", self.manifest)

    def run_fixture(self, generate=None):
        return runner.generate_dialogues(self.args, self.manifest, self.rows, generate or (lambda p, s: decoded()))

    def test_successful_turn_survives_mid_dialogue_interruption(self):
        calls = []
        def interrupt(messages, seed):
            calls.append(seed)
            if len(calls) == 10:
                raise KeyboardInterrupt()
            return decoded()
        with self.assertRaises(KeyboardInterrupt):
            self.run_fixture(interrupt)
        saved = base.load_jsonl(self.args.output_dir / "turns.jsonl")
        self.assertEqual([(r['dialogue_id'], r['turn_index']) for r in saved], [(1, 0)])
        self.assertEqual(base.load_jsonl(self.args.output_dir / "answers.jsonl"), [])
        resumed = []
        self.assertEqual(self.run_fixture(lambda p, s: resumed.append(s) or decoded()), 0)
        self.assertEqual(len(resumed), 24)
        self.assertEqual(resumed[0], calls[8])
        self.assertEqual(base.load_jsonl(self.args.output_dir / "turns.jsonl")[0], saved[0])
        self.assertEqual(validate_turn_resume(self.args.output_dir)["candidates"], 32)

    def test_one_failed_turn_retries_eight_copies_only_and_preserves_audit(self):
        calls = []
        def fail(messages, seed):
            calls.append(seed)
            return decoded("") if len(calls) == 9 else decoded()
        self.assertEqual(self.run_fixture(fail), 1)
        before = base.load_jsonl(self.args.output_dir / "turns.jsonl")
        self.args.retry_errors = True
        retry = []
        self.assertEqual(self.run_fixture(lambda p, s: retry.append(s) or decoded()), 0)
        self.assertEqual(len(retry), 8)
        after = base.load_jsonl(self.args.output_dir / "turns.jsonl")
        for i in (0, 2, 3):
            self.assertEqual(before[i], after[i])
        archives = list((self.args.output_dir / "provenance/turn_journals").glob('*.jsonl'))
        self.assertEqual(len(archives), 1)
        self.assertEqual(len(base.load_jsonl(archives[0])), 5)
        self.assertTrue(base.load_jsonl(archives[0])[1]["error"])
        self.assertEqual(validate_turn_resume(self.args.output_dir)["turns"], 4)

    def test_error_without_retry_does_not_regenerate_or_report_success(self):
        self.assertEqual(self.run_fixture(lambda p, s: decoded("")), 4)
        self.assertEqual(self.run_fixture(lambda p, s: self.fail('Retried without flag')), 4)
        with patch.dict(sys.modules, {'torch': None, 'transformers': None}):
            self.assertEqual(runner.run_locked(self.args, self.manifest, self.rows), 2)

    def test_repeated_failure_keeps_good_turns_and_continues_missing_turns(self):
        n = []
        def fail_then_interrupt(messages, seed):
            n.append(seed)
            if len(n) == 10:
                raise KeyboardInterrupt()
            return decoded("") if len(n) == 9 else decoded()
        with self.assertRaises(KeyboardInterrupt):
            self.run_fixture(fail_then_interrupt)
        self.args.retry_errors = True
        retry = []
        def still_bad(messages, seed):
            retry.append(seed)
            return decoded("") if len(retry) == 1 else decoded()
        self.assertEqual(self.run_fixture(still_bad), 1)
        self.assertEqual(len(retry), 17)  # One failure + two missing turns of eight copies.
        self.assertEqual(len(base.load_jsonl(self.args.output_dir / 'turns.jsonl')), 4)

    def test_stale_answer_export_rebuilt_without_calls_and_no_op_is_stable(self):
        self.assertEqual(self.run_fixture(), 0)
        (self.args.output_dir / 'answers.jsonl').write_text('{"broken":')
        self.assertEqual(self.run_fixture(lambda p, s: self.fail('Completed turn regenerated')), 0)
        self.assertEqual(validate_turn_resume(self.args.output_dir)['turns'], 4)
        before = {n: (self.args.output_dir / n).read_bytes() for n in ['answers.jsonl','turns.jsonl']}
        self.assertEqual(self.run_fixture(lambda p, s: self.fail('No-op called model')), 0)
        self.assertEqual(before, {n: (self.args.output_dir / n).read_bytes() for n in before})

    def test_changed_history_seed_or_candidate_rejected_before_calls(self):
        self.run_fixture()
        path = self.args.output_dir / 'turns.jsonl'
        originals = base.load_jsonl(path)
        for mutate in [lambda r: r.update(seed=999),
                       lambda r: r['prompt_history'][0].update(content='changed'),
                       lambda r: r['smoothllm']['candidates'][0].update(perturbed_user_message='changed'),
                       lambda r: r.update(run_id='foreign')]:
            records = copy.deepcopy(originals)
            mutate(records[0]); runner.atomic_jsonl(path, records)
            with self.assertRaises(ValueError):
                self.run_fixture(lambda p, s: self.fail('Called model before checking saved data'))
        runner.atomic_jsonl(path, originals)

    def test_partial_last_append_recovered_and_interior_corruption_rejected(self):
        path = self.args.output_dir / 'journal.jsonl'
        path.write_bytes(b'{"good":1}\n{"broken":')
        self.assertEqual(runner.read_journal(path), [{'good':1}])
        self.assertEqual(next((path.parent / 'recovery').iterdir()).read_bytes(), b'{"broken":')
        path.write_bytes(b'broken\n{"good":1}\n')
        with self.assertRaises(ValueError):
            runner.read_journal(path)

    def legacy_fixture(self):
        source = self.root / 'legacy'
        source.mkdir()
        args = legacy.parse_args(['--dataset',str(self.dataset),'--output-dir',str(source)])
        manifest = legacy.manifest_for(args,self.rows)
        base.ensure_manifest(source/'run_config.json',manifest)
        n=[]
        def fail_one(p,s):
            n.append(s)
            return decoded('') if len(n)==9 else decoded()
        self.assertEqual(legacy.generate_dialogues(args,manifest,self.rows,fail_one),1)
        return source

    def test_legacy_import_locked_preserves_raw_records_and_retries_one_turn(self):
        source = self.legacy_fixture()
        hashes = {p.name:base.file_sha256(p) for p in source.iterdir() if p.is_file()}
        self.args.output_dir = self.root / 'imported'
        self.args.output_dir.mkdir()
        self.args.resume_from = source
        self.args.retry_errors = True
        with runner.single_writer(source):
            with self.assertRaisesRegex(RuntimeError,'Another writer'):
                runner.prepare_output(self.args,self.manifest,self.rows)
        runner.prepare_output(self.args,self.manifest,self.rows)
        old=base.load_jsonl(source/'turns.jsonl'); new=base.load_jsonl(self.args.output_dir/'turns.jsonl')
        for a,b in zip(old,new):
            a.update(run_id=b['run_id'],answer_id=b['answer_id'])
            self.assertEqual(a,b)
        for name,digest in hashes.items():
            self.assertEqual(base.file_sha256(source/name),digest)
            self.assertEqual(base.file_sha256(self.args.output_dir/'provenance/legacy'/name),digest)
        calls=[]
        self.assertEqual(self.run_fixture(lambda p,s:calls.append(s) or decoded()),0)
        self.assertEqual(len(calls),8)
        self.assertEqual(validate_turn_resume(self.args.output_dir)['turns'],4)
        runner.prepare_output(self.args,self.manifest,self.rows)  # Does not import again.
        self.assertEqual(self.run_fixture(lambda p,s:self.fail('Regenerated after import')),0)

    def test_import_rejects_changed_configuration_and_same_directory(self):
        source=self.legacy_fixture()
        self.args.output_dir=self.root/'imported'; self.args.output_dir.mkdir()
        self.args.resume_from=source
        self.args.seed=1
        with self.assertRaisesRegex(ValueError,'configuration'):
            runner.prepare_output(self.args,runner.manifest_for(self.args,self.rows),self.rows)
        with self.assertRaisesRegex(ValueError,'separate output'):
            runner.parse_args(['--resume-from',str(source),'--output-dir',str(source)])

    def test_validate_only_creates_no_output_or_model(self):
        output=self.root/'not-created'
        with patch.dict(sys.modules, {'torch':None,'transformers':None}):
            self.assertEqual(runner.main(['--dataset',str(self.dataset),'--output-dir',str(output),'--validate-only']),0)
        self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
