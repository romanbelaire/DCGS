import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_safedial_api as api
import run_safedial_baseline as base
from validate_safedial_generation import validate_run
from openai.types.chat import ChatCompletion


def response(text='generated', model='gpt-4o-2024-08-06', finish='stop', refusal=None):
    return ChatCompletion.model_validate(dict(
        id='chatcmpl-test', created=0, object='chat.completion', model=model,
        choices=[dict(index=0, finish_reason=finish,
                      message=dict(role='assistant', content=text, refusal=refusal))],
        usage=dict(prompt_tokens=20, completion_tokens=3, total_tokens=23),
        system_fingerprint='test-fingerprint'))


class APITests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.row = dict(id=1, task='Ethics', method='test', scene='test', history=[
            dict(user='first user', bot='GOLD first'), dict(user='second user', bot='GOLD second')])
        self.dataset = root / 'dataset.jsonl'
        self.dataset.write_text(json.dumps(self.row) + '\n')
        self.args = api.parse_args(['--dataset', str(self.dataset), '--output-dir', str(root / 'out'),
                                    '--parallel', '1', '--retry-backoff', '0'])
        self.args.output_dir.mkdir()
        base.ensure_manifest(self.args.output_dir / 'run_config.json', api.manifest_for(self.args, [self.row]))

    def client(self, values):
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=Mock(side_effect=values))))

    def test_gold_history_decoding_and_provenance(self):
        client = self.client([response('MODEL first'), response('MODEL second', finish='length')])
        self.assertEqual(api.generate(client, [self.row], self.args), 0)
        request = client.chat.completions.create.call_args.kwargs
        self.assertEqual(request['messages'], [dict(role='user', content='first user'),
                                              dict(role='assistant', content='GOLD first'),
                                              dict(role='user', content='second user')])
        self.assertEqual(request['model'], self.args.model)
        self.assertEqual(request['max_completion_tokens'], 1024)
        self.assertEqual(request['temperature'], 0.7)
        self.assertEqual(request['top_p'], 1.0)
        self.assertFalse(request['store'])
        records = base.load_jsonl(self.args.output_dir / 'turns.jsonl')
        self.assertTrue(records[1]['output_truncated'])
        self.assertEqual(records[0]['usage']['total_tokens'], 23)
        self.assertTrue(validate_run(self.args.output_dir)['passed'])

    def test_resume_partial_dialogue_without_repeating_paid_turn(self):
        client = self.client([response('saved'), ValueError('failed')])
        self.assertEqual(api.generate(client, [self.row], self.args), 2)
        self.assertEqual(base.load_jsonl(self.args.output_dir / 'answers.jsonl'), [])
        client = self.client([response('remaining')])
        self.assertEqual(api.generate(client, [self.row], self.args), 0)
        self.assertEqual(client.chat.completions.create.call_count, 1)
        self.assertEqual(validate_run(self.args.output_dir)['turns'], 2)
        client = self.client([])
        self.assertEqual(api.generate(client, [self.row], self.args), 0)
        client.chat.completions.create.assert_not_called()

    def test_snapshot_mismatch_and_empty_unfiltered_fail_without_fabrication(self):
        for bad in (response(model='other'), response(text=None)):
            record, audits = api.request_turn(self.client([bad]), self.row, 0, self.args)
            self.assertIsNone(record)
            self.assertIsNotNone(audits[0]['error'])
            self.assertEqual(audits[0]['response']['usage']['total_tokens'], 23)

    def test_filters_continue_preserve_partial_text_and_do_not_retry(self):
        for text in ('partial answer', None):
            with self.subTest(text=text):
                self.args.output_dir = Path(self.tmp.name) / ('partial' if text else 'empty')
                self.args.output_dir.mkdir()
                second = dict(self.row, id=2)
                rows = [self.row, second]
                client = self.client([response(text=text, finish='content_filter'), response(), response(), response()])
                self.assertEqual(api.generate(client, rows, self.args), 0)
                self.assertEqual(client.chat.completions.create.call_count, 4)
                filtered = json.loads((self.args.output_dir / 'filtered_cases.json').read_text())
                self.assertEqual(filtered['count'], 1)
                self.assertEqual(filtered['cases'][0]['partial_response'], text)
                self.assertEqual(filtered['cases'][0]['status'], 'unresolved_content_filter')
                self.assertEqual([a['id'] for a in base.load_jsonl(self.args.output_dir / 'answers.jsonl')], [2])
                report = api.validate_output(rows, self.args)
                self.assertTrue(report['passed'])
                self.assertTrue(report['processing_complete'])
                self.assertFalse(report['complete'])
                self.assertEqual(report['successful_turns'], 3)
                client = self.client([])
                self.assertEqual(api.generate(client, rows, self.args), 0)
                client.chat.completions.create.assert_not_called()

    def test_legacy_filtered_audit_recovers_without_repeating_call(self):
        audit = dict(dialogue_id=1, turn_index=0, requested_model=self.args.model,
                     error='ValueError', response=response(text='retained partial', finish='content_filter').model_dump(mode='json'))
        base.append_jsonl(self.args.output_dir / 'api_attempts.jsonl', [audit])
        client = self.client([response()])
        self.assertEqual(api.generate(client, [self.row], self.args), 0)
        self.assertEqual(client.chat.completions.create.call_count, 1)
        self.assertEqual(client.chat.completions.create.call_args.kwargs['messages'][-1]['content'], 'second user')
        self.assertEqual(api.validate_output([self.row], self.args)['filtered_turns'], 1)

    def test_filter_does_not_hide_later_fatal_failure_or_missing_coverage(self):
        client = self.client([response(finish='content_filter'), ValueError('fatal')])
        self.assertEqual(api.generate(client, [self.row], self.args), 2)
        report = api.validate_output([self.row], self.args)
        self.assertFalse(report['passed'])
        self.assertEqual(report['pending_turns'], 1)

    def test_manifest_migration_preserves_old_manifest_and_rejects_setting_changes(self):
        current = api.manifest_for(self.args, [self.row])
        legacy = dict(current, runner_sha256=api.LEGACY_RUNNER_SHA256)
        path = self.args.output_dir / 'run_config.json'
        path.write_text(json.dumps(legacy))
        self.args.temperature = 0.2
        with self.assertRaises(ValueError):
            api.ensure_api_manifest(self.args, [self.row], migrate=True)
        self.assertEqual(json.loads(path.read_text()), legacy)
        self.args.temperature = 0.7
        api.ensure_api_manifest(self.args, [self.row], migrate=True)
        self.assertEqual(json.loads(path.read_text()), current)
        self.assertEqual(json.loads((self.args.output_dir / 'provenance/run_config.before-filter-repair.json').read_text()), legacy)
        api.ensure_api_manifest(self.args, [self.row], migrate=True)

    def test_explicit_refusal_text_is_preserved(self):
        record, _ = api.request_turn(self.client([response(text=None, refusal='I cannot help.')]),
                                     self.row, 0, self.args)
        self.assertEqual(record['generated_response'], 'I cannot help.')
        self.assertEqual(record['refusal'], 'I cannot help.')

    def test_retry_transient_but_stop_authentication_and_redact_exception(self):
        class StatusError(Exception):
            def __init__(self, status):
                self.status_code = status
                super().__init__('secret-should-not-be-persisted')
        client = self.client([StatusError(429), response()])
        record, audits = api.request_turn(client, self.row, 0, self.args)
        self.assertIsNotNone(record)
        self.assertEqual(len(audits), 2)
        self.assertNotIn('secret-should-not-be-persisted', json.dumps(audits))
        client = self.client([StatusError(401), response()])
        record, _ = api.request_turn(client, self.row, 0, self.args)
        self.assertIsNone(record)
        self.assertEqual(client.chat.completions.create.call_count, 1)

    def test_changed_manifest_rejected(self):
        self.args.temperature = 0.2
        with self.assertRaises(RuntimeError):
            base.ensure_manifest(self.args.output_dir / 'run_config.json', api.manifest_for(self.args, [self.row]))

    def test_interrupted_append_preserved_and_recovered(self):
        path = self.args.output_dir / 'tail.jsonl'
        path.write_bytes(b'{"valid": true}\n{"partial":')
        self.assertEqual(api.read_records(path), [dict(valid=True)])
        tails = list(path.parent.glob('tail.jsonl.interrupted-*'))
        self.assertEqual(len(tails), 1)
        self.assertEqual(tails[0].read_bytes(), b'{"partial":')
        path.write_bytes(b'bad\n')
        with self.assertRaises(ValueError):
            api.read_records(path)


if __name__ == '__main__':
    unittest.main()
