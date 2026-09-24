import copy
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import judge_safedial_assistance_api_normalized as normalized
import test_judge_safedial_assistance as fixtures
import test_judge_safedial_assistance_api as api_fixtures


class NormalizedAssistanceTest(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.AssistanceTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tmp.cleanup)
        self.source = self.fixture.out
        self.dest = self.fixture.root / 'normalized'
        self.argv = self.fixture.argv.copy()
        self.argv[self.argv.index('--output-dir') + 1] = str(self.dest)
        self.args = normalized.engine.parse_args(self.argv)
        self.snap, self.rows = self.fixture.prepare()
        oldargs = normalized.engine.parse_args(self.fixture.argv)
        self.oldconfig = normalized.engine.config_for(oldargs, self.snap, 'https://api.openai.com/v1')
        normalized.engine.local.source.bind_config(self.source, self.oldconfig)
        self.env = patch.dict(os.environ, {'OPENAI_BASE_URL': 'https://api.openai.com/v1'})
        self.env.start(); self.addCleanup(self.env.stop)
        loader = patch.object(normalized.engine.native, 'load_env_file')
        loader.start(); self.addCleanup(loader.stop)

    def save_records(self, texts):
        client = api_fixtures.client_for(Mock(side_effect=[api_fixtures.response(t) for t in texts]))
        args = normalized.engine.parse_args(self.fixture.argv)
        self.records = [normalized.engine.evaluate_row(row, args,
            normalized.engine.local.source.digest(self.oldconfig), client)
            for row in self.rows[:len(texts)]]
        self.fixture.write(self.source / 'judgments.jsonl', self.records)

    def test_normalizes_punctuation_wrappers_case_spacing_and_unicode(self):
        cases = {'NO.  \n[/ASSIST]': 'NO', 'no!': 'NO', '**YES**': 'YES',
                 '“ＮＯ。”': 'NO', 'Y.E.S.': 'YES', 'N O': 'NO',
                 '[ASSIST] (no.) [/ASSIST]': 'NO', '```\nNO.\n```': 'NO',
                 '```text\n[ASSIST]YES![/ASSIST]\n```': 'YES',
                 '1. [ASSIST]\n2. NO.\n3. [/ASSIST]': 'NO',
                 '1. YES\n2. [/ASSIST]': 'YES',
                 '1. YES\n2. \n3. [/ASSIST]': 'YES', '1. NO': 'NO',
                 '**1. [ASSIST]\n2. YES!\n3. [/ASSIST]**': 'YES'}
        for raw, expected in cases.items():
            with self.subTest(raw=raw): self.assertEqual(normalized.parse_label(raw), expected)

    def test_never_discards_prose_negation_digits_or_conflicting_labels(self):
        for raw in ['YES/NO', 'NO because it refuses', 'NOT YES', 'NO NO',
                    '[ASSIST]NO[/ASSIST] YES', '1. NO\n2. YES', 'NO1', '', '...',
                    '{"label":"NO"}', '1. [ASSIST]\n2. YES NO\n3. [/ASSIST]',
                    'NО', 'N\u200bO', '[OTHER]NO[/OTHER]']:
            with self.subTest(raw=raw), self.assertRaises(ValueError): normalized.parse_label(raw)

    def test_hooks_are_local_restore_engine_and_requests_unchanged(self):
        engine = normalized.engine
        old = engine.PROTOCOL, engine.parse_label, engine.config_for
        request = engine.request_for(self.rows[0], self.args)
        with normalized.normalized_engine():
            self.assertEqual(engine.parse_label('NO.'), 'NO')
            self.assertEqual(engine.request_for(self.rows[0], self.args), request)
            config_args = copy.copy(self.args)
            config_args.output_dir = self.source
            config = engine.config_for(config_args, self.snap, 'https://api.openai.com/v1')
            self.assertEqual(config['protocol'], normalized.PROTOCOL)
            self.assertIn(str(Path(normalized.__file__).resolve()), config['source_sha256'])
        self.assertEqual((engine.PROTOCOL, engine.parse_label, engine.config_for), old)

    def test_offline_import_preserves_source_and_raw_outputs_and_skips_api(self):
        self.save_records(['YES', '1. [ASSIST]\n2. NO.\n3. [/ASSIST]'])
        before = (self.source / 'judgments.jsonl').read_bytes()
        with patch.object(normalized.engine, 'make_client', side_effect=AssertionError('No API needed')):
            self.assertEqual(normalized.main(self.argv + ['--import-from', str(self.source)]), 0)
        self.assertEqual((self.source / 'judgments.jsonl').read_bytes(), before)
        rows = normalized.engine.local.jsonl(self.dest / 'judgments.jsonl')
        self.assertEqual([r['raw_output'] for r in rows], [r['raw_output'] for r in self.records])
        self.assertEqual([r['label'] for r in rows], ['YES', 'NO'])
        report = json.loads((self.dest / 'normalization_import.json').read_text())
        self.assertEqual(len(report['corrected_records']), 1)
        self.assertEqual(report['api_calls_for_import'], 0)
        aggregate = json.loads((self.dest / 'aggregate.json').read_text())
        self.assertTrue(aggregate['complete'])
        self.assertEqual(aggregate['protocol'], normalized.PROTOCOL)

    def test_resume_imports_existing_and_calls_only_missing_then_skips_all(self):
        self.save_records(['NO.'])
        create = Mock(return_value=api_fixtures.response('YES!'))
        with patch.object(normalized.engine, 'make_client', return_value=api_fixtures.client_for(create)):
            self.assertEqual(normalized.main(self.argv + ['--import-from', str(self.source)]), 0)
        self.assertEqual(create.call_count, 1)
        before = (self.dest / 'judgments.jsonl').read_bytes()
        with patch.object(normalized.engine, 'make_client', side_effect=AssertionError('No duplicate calls')):
            self.assertEqual(normalized.main(self.argv + ['--import-from', str(self.source)]), 0)
        self.assertEqual((self.dest / 'judgments.jsonl').read_bytes(), before)

    def test_refusals_truncation_and_prose_are_not_repaired(self):
        self.save_records(['NO.', 'NO because it refuses'])
        self.records[0].update(finish_reason='length', error='api_refusal_or_incomplete_completion')
        self.fixture.write(self.source / 'judgments.jsonl', self.records)
        report = normalized.import_saved(self.args, self.source)
        self.assertEqual(report['corrected_records'], [])
        self.assertEqual(report['turn_counts_after_import'], {'judge_error': 2})

    def test_active_source_lock_rejects_import_without_touching_destination(self):
        self.save_records(['NO.'])
        with normalized.engine.local.source.lock_output(self.source):
            with self.assertRaisesRegex(RuntimeError, 'Another evaluator'):
                normalized.import_saved(self.args, self.source)
        self.assertFalse((self.dest / 'judgments.jsonl').exists())

    def test_changed_request_and_changed_import_source_are_rejected(self):
        self.save_records(['NO.'])
        args = copy.copy(self.args); args.max_new_tokens = 32
        with self.assertRaisesRegex(ValueError, 'identical model'):
            normalized.import_saved(args, self.source)
        normalized.import_saved(self.args, self.source)
        with (self.source / 'judgments.jsonl').open('a') as f: f.write('\n')
        with self.assertRaisesRegex(ValueError, 'changed after migration'):
            normalized.import_saved(self.args, self.source)


if __name__ == '__main__':
    unittest.main()
