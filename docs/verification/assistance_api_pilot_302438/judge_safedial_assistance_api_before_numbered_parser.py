#!/usr/bin/env python3
"""GPT-4o-mini assistance judge with the unchanged local DCGS prompt.

Uses frozen generation/goals and strict labels from the local runner without
changing its source or manifests. API execution belongs in user-submitted jobs.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import judge_safedial as native
import judge_safedial_assistance as local

MODEL = 'gpt-4o-mini'
PROTOCOL = 'safedial_dcgs_api_assistance_same_prompt_v1'
CONTEXT_LIMIT = 128000


def request_for(row, args):
    # Same text, including the final prefill. The API supplies chat framing;
    # this is not a literal assistant-token prefill or the native scoring rubric.
    return {'model': MODEL, 'messages': [{'role': 'user', 'content':
            local.render_prompt(row['goal'], row['user_message'], row['response']) + local.PREFILL}],
            'temperature': 0.0, 'max_tokens': args.max_new_tokens}


def config_for(args, snapshot, endpoint):
    files = [Path(__file__), Path(local.__file__), Path(native.__file__),
             Path(local.source.__file__), Path(local.goals.__file__), local.ORIGINAL]
    return {'protocol': PROTOCOL, 'model': MODEL, 'backend': 'chat.completions',
            'endpoint_sha256': hashlib.sha256(endpoint.encode()).hexdigest(),
            'prompt_template': local.render_prompt('{goal}', '{user}', '{response}') + local.PREFILL,
            'messages': 'one user message; no added system instruction; ASSIST suffix is literal user text',
            'temperature': 0.0, 'max_new_tokens': args.max_new_tokens,
            'parallel': args.parallel, 'timeout': args.timeout, 'max_attempts': args.max_attempts,
            'retry_backoff': args.retry_backoff, 'retry_policy': 'transient API errors only; no format/refusal retries',
            'parser': 'strict single YES/NO; refusals, filtering and truncated completions remain errors',
            'truncation': 'none', 'context_limit': CONTEXT_LIMIT,
            'context_audit': 'conservative UTF-8 byte upper bound plus 32 framing tokens; no tokenizer download',
            'snapshot_sha256': local.source.file_hash(args.output_dir / 'frozen/snapshot.json'),
            'source_sha256': {str(p.resolve()): local.source.file_hash(p) for p in files}}


def preflight(args, rows):
    # A byte-level tokenizer cannot use more ordinary text tokens than bytes.
    # Reserve extra space for this single user message's API role/framing tokens.
    records = []
    for row in rows:
        if row['generation_status'] != 'available':
            continue
        upper = len(request_for(row, args)['messages'][0]['content'].encode('utf-8')) + 32
        records.append({'dialogue_id': row['dialogue_id'], 'turn_index': row['turn_index'],
                        'input_token_upper_bound': upper,
                        'overflow': upper + args.max_new_tokens > CONTEXT_LIMIT})
    report = {'passed': bool(records) and not any(r['overflow'] for r in records),
              'available_turns': len(records), 'context_limit': CONTEXT_LIMIT,
              'max_input_token_upper_bound': max((r['input_token_upper_bound'] for r in records), default=0),
              'counting': 'UTF-8 bytes plus 32 framing tokens; conservative bound, not exact token count',
              'overflow_turns': [r for r in records if r['overflow']], 'truncation': False}
    local.source.atomic_json(args.output_dir / 'preflight.json', report)
    return report


def summarize(folder, rows, latest):
    report = local.aggregate(folder, rows, latest)
    report.update(protocol=PROTOCOL, judge_model=MODEL, backend='chat.completions')
    local.source.atomic_json(folder / 'aggregate.json', report)
    return report


def transient_error(exc):
    status = getattr(exc, 'status_code', None)
    return (status in (408, 409, 429) or isinstance(status, int) and status >= 500 or
            isinstance(exc, (TimeoutError, ConnectionError)) or
            type(exc).__name__ in ('APITimeoutError', 'APIConnectionError'))


def evaluate_row(row, args, config_hash, client):
    record = {k: row[k] for k in ('dialogue_id', 'turn_index', 'input_sha256')}
    record.update(judge_config_sha256=config_hash, status='error', raw_output='',
                  recorded_at=local.source.now(), failed_attempts=[])
    for attempt in range(1, args.max_attempts + 1):
        record['attempts'] = attempt
        started = time.monotonic()
        try:
            response = client.chat.completions.create(**request_for(row, args))
        except Exception as exc:
            # Exception strings can include endpoint credentials or request data.
            failure = {'type': type(exc).__name__, 'status_code': getattr(exc, 'status_code', None),
                       'latency_seconds': time.monotonic() - started, 'attempt': attempt}
            record['failed_attempts'].append(failure)
            record['error'] = 'api_request_failed'
            if not transient_error(exc) or attempt == args.max_attempts:
                return record
            time.sleep(args.retry_backoff * 2 ** (attempt - 1))
            continue
        record.update(response_id=getattr(response, 'id', None), response_model=getattr(response, 'model', None),
                      system_fingerprint=getattr(response, 'system_fingerprint', None),
                      usage=native.usage_dict(response), latency_seconds=time.monotonic() - started)
        if not response.choices:
            record['error'] = 'empty_api_choices'
            return record
        choice = response.choices[0]
        record.update(raw_output=choice.message.content or '', finish_reason=choice.finish_reason,
                      refusal=getattr(choice.message, 'refusal', None))
        if record['refusal'] or record['finish_reason'] != 'stop':
            record['error'] = 'api_refusal_or_incomplete_completion'
            return record
        try:
            label = local.parse_label(record['raw_output'])
        except ValueError as exc:
            record['error'] = str(exc)
        else:
            record.update(status='success', label=label, error=None)
        return record


def run_api(args, rows, latest, config_hash, client):
    pending = iter(r for r in rows if r['generation_status'] == 'available' and
                   (r['input_sha256'] not in latest or
                    args.retry_errors and latest[r['input_sha256']]['status'] == 'error'))
    started = time.monotonic()
    written = 0
    # At most parallel requests are in flight; only this thread writes the journal.
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = set()
        def submit_one():
            row = next(pending, None)
            if row is not None:
                futures.add(pool.submit(evaluate_row, row, args, config_hash, client))
        try:
            for _ in range(args.parallel):
                submit_one()
            with (args.output_dir / 'judgments.jsonl').open('a', encoding='utf-8') as journal:
                while futures:
                    done, futures = wait(futures, return_when=FIRST_COMPLETED)
                    for future in done:
                        record = future.result()
                        journal.write(local.source.canonical(record) + '\n')
                        journal.flush()
                        os.fsync(journal.fileno())
                        latest[record['input_sha256']] = record
                        written += 1
                        print(f"dialogue={record['dialogue_id']} turn={record['turn_index']} "
                              f"status={record['status']} label={record.get('label', '')}", flush=True)
                    if written and written // 100 != (written - len(done)) // 100:
                        summarize(args.output_dir, rows, latest)
                    for _ in done:
                        submit_one()
        finally:
            for future in futures:
                future.cancel()
            local.source.atomic_json(args.output_dir / 'runtime.json',
                {'model': MODEL, 'backend': 'chat.completions', 'parallel_workers': args.parallel,
                 'elapsed_seconds': time.monotonic() - started, 'journal_records_written': written})


def make_client(args, endpoint):
    import httpx
    from openai import OpenAI
    key = os.getenv('OPENAI_API_KEY', '').strip()
    if not key:
        raise ValueError('OPENAI_API_KEY is missing; use the existing .env or exported environment')
    native.validate_url('OPENAI_BASE_URL', endpoint)
    if not args.ignore_environment_proxy:
        for name in native.PROXY_VARIABLES:
            value = os.getenv(name, '').strip()
            if value:
                native.validate_url(name, value)
    return OpenAI(api_key=key, base_url=endpoint, max_retries=0, timeout=args.timeout,
                  http_client=httpx.Client(timeout=args.timeout, trust_env=not args.ignore_environment_proxy))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--goals-dir', type=Path, default=local.DEFAULT_GOALS)
    p.add_argument('--dataset', type=Path)
    p.add_argument('--ids', type=int, nargs='+')
    p.add_argument('--input-format', choices=['auto', 'turns', 'answers'], default='auto')
    p.add_argument('--choice-index', type=int, default=0)
    p.add_argument('--parallel', type=int, default=8)
    p.add_argument('--max-new-tokens', type=int, default=24)
    p.add_argument('--max-attempts', type=int, default=3)
    p.add_argument('--retry-backoff', type=float, default=2.0)
    p.add_argument('--timeout', type=float, default=120.0)
    p.add_argument('--env-file', type=Path, default=Path('.env'))
    p.add_argument('--ignore-environment-proxy', action='store_true')
    p.add_argument('--retry-errors', action='store_true')
    p.add_argument('--llamaguard-dir', type=Path)
    modes = p.add_mutually_exclusive_group()
    modes.add_argument('--prepare-only', action='store_true')
    modes.add_argument('--preflight-only', action='store_true')
    modes.add_argument('--aggregate-only', action='store_true')
    args = p.parse_args(argv)
    if (min(args.parallel, args.max_new_tokens, args.max_attempts, args.timeout) <= 0 or
            args.max_new_tokens > 16384 or args.retry_backoff < 0 or args.choice_index < 0):
        p.error('Invalid workers, token budget, retry settings, timeout or choice index')
    args.run_dir = args.run_dir.resolve(); args.output_dir = args.output_dir.resolve()
    args.goals_dir = args.goals_dir.resolve()
    if (args.output_dir == args.run_dir or args.output_dir in args.run_dir.parents or
            args.output_dir == args.goals_dir or args.output_dir in args.goals_dir.parents or
            args.goals_dir in args.output_dir.parents):
        p.error('Output must be separate from generation and goal artifacts')
    if args.ids is not None:
        args.ids = sorted(args.ids)
    return args


def main(argv=None):
    args = parse_args(argv)
    native.load_env_file(args.env_file)
    endpoint = os.getenv('OPENAI_BASE_URL', '').strip() or 'https://api.openai.com/v1'
    with local.source.lock_output(args.output_dir):
        if args.aggregate_only and not (args.output_dir / 'frozen').exists():
            raise ValueError('Aggregate-only requires existing frozen inputs')
        snapshot, rows = local.prepare(args)
        config = config_for(args, snapshot, endpoint)
        local.source.bind_config(args.output_dir, config)
        latest = local.read_judgments(args.output_dir, rows, local.source.digest(config))
        report = summarize(args.output_dir, rows, latest)
        if args.prepare_only:
            print(json.dumps(report)); return 0
        if not args.aggregate_only:
            audit = preflight(args, rows)
            if not audit['passed'] or args.preflight_only:
                print(json.dumps(audit)); return 0 if audit['passed'] else 2
            pending = any(r['generation_status'] == 'available' and
                          (r['input_sha256'] not in latest or
                           args.retry_errors and latest[r['input_sha256']]['status'] == 'error') for r in rows)
            if pending:
                client = make_client(args, endpoint)
                try:
                    run_api(args, rows, latest, local.source.digest(config), client)
                finally:
                    client.close()
                    report = summarize(args.output_dir, rows, latest)
        if args.llamaguard_dir:
            report = local.combine_guard(args.output_dir, rows, latest, args.llamaguard_dir)
            report.update(protocol=PROTOCOL + '_guard_join_v1', assistance_judge_model=MODEL)
            local.source.atomic_json(args.output_dir / 'combined_aggregate.json', report)
        print(json.dumps(report))
        return 0 if report['complete'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
