import copy
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import judge_safedial_assistance_api as api
import test_judge_safedial_assistance as fixtures


def response(text='NO[/ASSIST]', finish='stop', refusal=None):
    return SimpleNamespace(id='response-1', model='gpt-4o-mini-test', system_fingerprint='fingerprint',
        usage=SimpleNamespace(prompt_tokens=123, completion_tokens=5, total_tokens=128),
        choices=[SimpleNamespace(finish_reason=finish,
                                 message=SimpleNamespace(content=text, refusal=refusal))])


def client_for(fn):
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fn)), close=Mock())


class AssistanceAPITest(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.AssistanceTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tmp.cleanup)
        self.args = api.parse_args(self.fixture.argv)
        self.out = self.args.output_dir
        self.snap, self.rows = self.fixture.prepare()

    def test_exact_existing_prompt_and_suffix_without_added_instructions(self):
        for row in self.rows:
            request = api.request_for(row, self.args)
            self.assertEqual(request, {'model': 'gpt-4o-mini', 'temperature': 0.0, 'max_tokens': 24,
                'messages': [{'role': 'user', 'content': api.local.render_prompt(
                    row['goal'], row['user_message'], row['response']) + '[ASSIST]'}]})
        self.assertEqual(self.args.parallel, 8)

    def test_success_preserves_response_provenance_usage_and_strict_label(self):
        create = Mock(return_value=response('[ASSIST]\nYES\n[/ASSIST]'))
        row = api.evaluate_row(self.rows[0], self.args, 'hash', client_for(create))
        self.assertEqual((row['status'], row['label'], row['attempts']), ('success', 'YES', 1))
        self.assertEqual(row['usage']['total_tokens'], 128)
        self.assertEqual(row['response_id'], 'response-1')
        self.assertEqual(row['judge_config_sha256'], 'hash')
        self.assertEqual(row['input_sha256'], self.rows[0]['input_sha256'])

    def test_bad_format_refusal_and_truncation_never_become_no(self):
        cases = [response('Explaining rather than labeling'), response(''),
                 response('YES NO'), response('NO', finish='length'),
                 response('NO', refusal='refused'), response(None, finish='content_filter')]
        for result in cases:
            with self.subTest(result=result):
                create = Mock(return_value=result)
                row = api.evaluate_row(self.rows[0], self.args, 'hash', client_for(create))
                self.assertEqual(row['status'], 'error')
                self.assertNotIn('label', row)
                self.assertEqual(create.call_count, 1)

    def test_numbered_prompt_format_is_accepted_but_ambiguous_variants_are_not(self):
        for label in ['YES', 'NO']:
            raw = f'1. [ASSIST]\n2. {label}\n3. [/ASSIST]'
            self.assertEqual(api.parse_label(raw), label)
            create = Mock(return_value=response(raw))
            record = api.evaluate_row(self.rows[0], self.args, 'hash', client_for(create))
            self.assertEqual(record['status'], 'success')
            self.assertEqual(record['raw_output'], raw)
            self.fixture.write(self.out / 'judgments.jsonl', [record])
            self.assertEqual(api.read_judgments(self.out, self.rows, 'hash')[self.rows[0]['input_sha256']]['label'], label)
        for bad in ['1. [ASSIST]\n2. YES NO\n3. [/ASSIST]',
                    '1. [ASSIST]\n2. NO because...\n3. [/ASSIST]',
                    '1. [ASSIST]\n2. NO\n3. [/ASSIST] YES',
                    '2. [ASSIST]\n1. NO\n3. [/ASSIST]', '1. NO']:
            with self.subTest(raw=bad), self.assertRaises(ValueError):
                api.parse_label(bad)

    def test_api_journal_rejects_inconsistent_completion_and_duplicate_success(self):
        row = api.evaluate_row(self.rows[0], self.args, 'hash', client_for(Mock(return_value=response())))
        for changes in [{'finish_reason': 'length'}, {'refusal': 'refused'}, {'label': 'YES'},
                        {'input_sha256': 'wrong'}, {'judge_config_sha256': 'wrong'}]:
            self.fixture.write(self.out / 'judgments.jsonl', [dict(row, **changes)])
            with self.assertRaises(ValueError): api.read_judgments(self.out, self.rows, 'hash')
        self.fixture.write(self.out / 'judgments.jsonl', [row, row])
        with self.assertRaisesRegex(ValueError, 'overwrites'):
            api.read_judgments(self.out, self.rows, 'hash')

    def test_transient_api_retry_keeps_attempts_and_identical_request(self):
        self.args.retry_backoff = 0
        create = Mock(side_effect=[TimeoutError('secret must not appear'), response('YES')])
        row = api.evaluate_row(self.rows[0], self.args, 'hash', client_for(create))
        self.assertEqual((row['status'], row['attempts']), ('success', 2))
        self.assertEqual(row['failed_attempts'][0]['type'], 'TimeoutError')
        self.assertNotIn('secret', json.dumps(row))
        self.assertEqual(create.call_args_list[0], create.call_args_list[1])

    def test_nontransient_and_exhausted_requests_remain_errors(self):
        class APIError(Exception):
            status_code = 401
        for exc, expected in [(APIError('secret'), 1), (ConnectionError('secret'), 3)]:
            self.args.retry_backoff = 0
            create = Mock(side_effect=exc)
            row = api.evaluate_row(self.rows[0], self.args, 'hash', client_for(create))
            self.assertEqual(row['status'], 'error')
            self.assertEqual(create.call_count, expected)
            self.assertNotIn('secret', json.dumps(row))

    def test_eight_workers_with_serial_durable_journal_and_complete_aggregate(self):
        barrier = threading.Barrier(8, timeout=10)
        lock = threading.Lock()
        active = peak = 0
        def create(**kwargs):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            barrier.wait()
            with lock:
                active -= 1
            return response()
        rows = []
        for i in range(16):
            row = dict(self.rows[0], dialogue_id=i + 1, input_sha256=f'input-{i}')
            rows.append(row)
        latest = {}
        api.run_api(self.args, rows, latest, 'hash', client_for(create))
        self.assertEqual(peak, 8)
        saved = api.read_judgments(self.out, rows, 'hash')
        self.assertEqual(len(saved), 16)
        self.assertTrue(api.summarize(self.out, rows, saved)['complete'])
        runtime = json.loads((self.out / 'runtime.json').read_text())
        self.assertEqual(runtime['parallel_workers'], 8)
        self.assertEqual(runtime['journal_records_written'], 16)

    def test_resume_skips_success_and_errors_without_explicit_retry(self):
        latest = {self.rows[0]['input_sha256']: self.fixture.success(self.rows[0], config='hash'),
                  self.rows[1]['input_sha256']: dict(self.fixture.success(self.rows[1], config='hash'),
                                                    status='error', raw_output='bad')}
        create = Mock(return_value=response('YES'))
        api.run_api(self.args, self.rows, latest, 'hash', client_for(create))
        create.assert_not_called()
        self.args.retry_errors = True
        api.run_api(self.args, self.rows, latest, 'hash', client_for(create))
        self.assertEqual(create.call_count, 1)
        self.assertEqual(latest[self.rows[0]['input_sha256']]['label'], 'NO')
        self.assertEqual(latest[self.rows[1]['input_sha256']]['label'], 'YES')

    def test_offline_preflight_is_no_network_and_rejects_oversize_without_clipping(self):
        with patch.object(api, 'make_client', side_effect=AssertionError('network forbidden')):
            self.assertEqual(api.main(self.fixture.argv + ['--preflight-only']), 0)
        oversized = dict(self.rows[0], response='é' * 70000)
        report = api.preflight(self.args, [oversized])
        self.assertFalse(report['passed'])
        self.assertEqual(len(oversized['response']), 70000)
        self.assertFalse(report['truncation'])

    def test_api_config_rejects_local_or_changed_endpoint_and_preserves_hashes(self):
        config = api.config_for(self.args, self.snap, 'https://api.openai.com/v1')
        api.local.source.bind_config(self.out, config)
        other = api.config_for(self.args, self.snap, 'https://example.invalid/v1')
        with self.assertRaises(ValueError): api.local.source.bind_config(self.out, other)
        self.assertNotEqual(config['protocol'], api.local.PROTOCOL)
        local_config = api.local.config_for(self.fixture.args, self.snap)
        with self.assertRaises(ValueError): api.local.source.bind_config(self.out, local_config)

    def test_main_roundtrip_complete_resume_makes_no_more_calls(self):
        create = Mock(return_value=response())
        client = client_for(create)
        with patch.object(api, 'make_client', return_value=client):
            self.assertEqual(api.main(self.fixture.argv), 0)
        self.assertEqual(create.call_count, 2)
        with patch.object(api, 'make_client', side_effect=AssertionError('must skip complete')):
            self.assertEqual(api.main(self.fixture.argv), 0)
            self.assertEqual(api.main(self.fixture.argv + ['--aggregate-only']), 0)
        report = json.loads((self.out / 'aggregate.json').read_text())
        self.assertEqual(report['protocol'], api.PROTOCOL)
        self.assertEqual(report['judge_model'], 'gpt-4o-mini')
        self.assertEqual(report['turn_counts'], {'NO': 2})

    def test_source_missing_response_preserves_incomplete_coverage(self):
        rows = copy.deepcopy(self.rows)
        rows[0]['generation_status'] = 'missing'
        latest = {rows[1]['input_sha256']: self.fixture.success(rows[1])}
        report = api.summarize(self.out, rows, latest)
        self.assertFalse(report['complete'])
        self.assertTrue(report['available_responses_judged'])
        self.assertEqual(report['turn_counts'], {'generation_missing': 1, 'NO': 1})


if __name__ == '__main__':
    unittest.main()
