#!/usr/bin/env python3
"""CPU-only, turn-resumable SafeDialBench API answer generation (no judging)."""

import argparse
import fcntl
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import run_safedial_baseline as base

LEGACY_RUNNER_SHA256 = '62c5ca081739cc900d2549a007911a0095c1ffb0c50eb8146c0a7c7616e8d3ca'


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, default=base.DEFAULT_DATASET)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--model', default='gpt-4o-2024-08-06')
    p.add_argument('--model-id', default='gpt-4o-baseline')
    p.add_argument('--env-file', type=Path, default=Path('.env'))
    p.add_argument('--parallel', type=int, default=2)
    p.add_argument('--max-new-tokens', type=int, default=1024)
    p.add_argument('--temperature', type=float, default=0.7)
    p.add_argument('--top-p', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--timeout', type=float, default=120)
    p.add_argument('--attempts', type=int, default=4)
    p.add_argument('--retry-backoff', type=float, default=5)
    selection = p.add_mutually_exclusive_group()
    selection.add_argument('--ids')
    selection.add_argument('--limit', type=int)
    selection.add_argument('--per-task', type=int)
    p.add_argument('--validate-only', action='store_true')
    p.add_argument('--check-access', action='store_true', help='Models API only; no generation or output writes.')
    p.add_argument('--prepare-resume', action='store_true',
                   help='Offline: migrate the known legacy manifest and recover filtered cases; no API calls.')
    p.add_argument('--validate-output', action='store_true',
                   help='Offline: validate all processed turns, reporting filtered cases as unresolved.')
    args = p.parse_args(argv)
    if min(args.parallel, args.max_new_tokens, args.timeout, args.attempts) <= 0:
        p.error('parallel, token cap, timeout and attempts must be positive')
    if not 0 <= args.temperature <= 2 or not 0 < args.top_p <= 1 or args.retry_backoff < 0:
        p.error('invalid decoding or backoff settings')
    if not re.fullmatch(r'gpt-4o-\d{4}-\d{2}-\d{2}|gpt-4-\d{4}', args.model):
        p.error('use an exact GPT-4o or GPT-4 snapshot, not a moving alias')
    return args


def manifest_for(args, selected):
    return dict(benchmark='SafeDialBench', protocol=base.PROTOCOL,
                dataset=str(args.dataset.resolve()), dataset_sha256=base.file_sha256(args.dataset),
                selected_ids=[r['id'] for r in selected], model=args.model, model_id=args.model_id,
                num_choices=1, temperature=args.temperature, top_p=args.top_p,
                max_new_tokens=args.max_new_tokens, seed=args.seed, generation_only=True,
                endpoint='https://api.openai.com/v1', api='chat.completions', store=False,
                runner_sha256=base.file_sha256(Path(__file__)),
                history_builder_sha256=base.file_sha256(Path(base.__file__)),
                seed_guarantee='best effort; not deterministic', input_truncation='none')


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def ensure_api_manifest(args, selected, migrate=False):
    """Only migrate the audited legacy runner; never loosen inference provenance."""
    path = args.output_dir / 'run_config.json'
    current = manifest_for(args, selected)
    if path.exists() and migrate:
        old = json.loads(path.read_text())
        if old != current:
            changes = {key for key in old.keys() | current.keys() if old.get(key) != current.get(key)}
            if changes != {'runner_sha256'} or old.get('runner_sha256') != LEGACY_RUNNER_SHA256:
                raise ValueError('Only the known legacy runner hash may change during this migration')
            provenance = args.output_dir / 'provenance'
            provenance.mkdir(exist_ok=True)
            archive = provenance / 'run_config.before-filter-repair.json'
            if archive.exists():
                if json.loads(archive.read_text()) != old:
                    raise ValueError('Existing migration archive does not match legacy manifest')
            else:
                archive.write_bytes(path.read_bytes())
            atomic_json(provenance / 'filter-repair-migration.json', dict(
                previous_manifest=old, updated_manifest=current,
                reason='Preserve content_filter cases as unresolved and continue independent turns',
                successful_records_unchanged=True, tstamp=time.time()))
            atomic_json(path, current)
    base.ensure_manifest(path, current)


def filtered_case(audit, args):
    """Recognize real API filter outcomes, including the old ValueError audit."""
    response = audit.get('response') or {}
    choices = response.get('choices') or []
    if (audit.get('requested_model') != args.model or response.get('model') != args.model
            or len(choices) != 1 or choices[0].get('finish_reason') != 'content_filter'):
        return None
    message = choices[0].get('message') or {}
    return dict(dialogue_id=audit['dialogue_id'], turn_index=audit['turn_index'],
                status='unresolved_content_filter', requested_model=args.model,
                returned_model=response['model'], response_id=response.get('id'),
                partial_response=message.get('content'), refusal=message.get('refusal'),
                usage=response.get('usage'), tstamp=audit.get('tstamp'),
                audit_source='api_attempts.jsonl', automatically_retry=False)


def request_turn(client, row, index, args):
    """Return all attempt audits to the single writer; never invent an answer."""
    messages = base.gold_messages(row['history'], index)
    seed = base.turn_seed(args.seed, args.model_id, row['id'], 0, index)
    audits = []
    for attempt in range(1, args.attempts + 1):
        started = time.perf_counter()
        audit = dict(dialogue_id=row['id'], turn_index=index, attempt=attempt,
                     requested_model=args.model, tstamp=time.time())
        retryable = False
        try:
            response = client.chat.completions.create(
                model=args.model, messages=messages, temperature=args.temperature,
                top_p=args.top_p, max_completion_tokens=args.max_new_tokens,
                seed=seed, n=1, store=False)
            audit['response'] = response.model_dump(mode='json')
            if response.model != args.model:
                raise ValueError('returned model does not match pinned snapshot')
            if filtered_case(audit, args) is not None:
                audit.update(error=None, outcome='filtered', latency_seconds=time.perf_counter() - started)
                audits.append(audit)
                return None, audits
            choice = response.choices[0]
            content = choice.message.content
            refusal = choice.message.refusal
            message = content if content and content.strip() else refusal
            if not message or not message.strip():
                raise ValueError('API returned no answer or refusal text')
            if choice.finish_reason not in ('stop', 'length'):
                raise ValueError('API returned unsupported finish reason')
            if response.usage is None:
                raise ValueError('API returned no usage accounting')
            audit['error'] = None
            record = dict(
                benchmark='SafeDialBench', protocol=base.PROTOCOL, model=args.model,
                model_id=args.model_id, returned_model=response.model,
                dialogue_id=row['id'], choice_index=0, turn_index=index,
                task=row['task'], method=row['method'], scene=row['scene'],
                seed=seed, prompt_history=messages, user_message=row['history'][index]['user'],
                reference_response=row['history'][index]['bot'], generated_response=message,
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                usage=response.usage.model_dump(mode='json'),
                finish_reason=choice.finish_reason, refusal=refusal,
                response_id=response.id, system_fingerprint=response.system_fingerprint,
                latency_seconds=time.perf_counter() - started,
                input_truncated=False, output_truncated=choice.finish_reason == 'length', error=None)
        except Exception as exc:
            status = getattr(exc, 'status_code', None)
            # Never persist raw exception text, which can contain credentials or URLs.
            audit['error'] = type(exc).__name__
            audit['status_code'] = status
            audit['request_id'] = getattr(exc, 'request_id', None)
            retryable = (status == 429 or (status is not None and status >= 500)
                         or type(exc).__name__ in ('APIConnectionError', 'APITimeoutError'))
            record = None
        audit['latency_seconds'] = time.perf_counter() - started
        audits.append(audit)
        if record is not None or not retryable or attempt == args.attempts:
            return record, audits
        time.sleep(args.retry_backoff * 2 ** (attempt - 1))


def read_records(path):
    """Recover only an unterminated final write; preserve its bytes for inspection."""
    if not path.exists():
        return []
    data = path.read_bytes()
    if data and not data.endswith(b'\n'):
        cut = data.rfind(b'\n') + 1
        path.with_name(path.name + f'.interrupted-{time.time_ns()}').write_bytes(data[cut:])
        with path.open('r+b') as f:
            f.truncate(cut)
            f.flush()
            os.fsync(f.fileno())
    return base.load_jsonl(path)


def write_answers(selected, saved, args):
    answers = []
    for row in selected:
        records = [saved.get((row['id'], i)) for i in range(len(row['history']))]
        if not all(records):
            continue
        answers.append(dict(id=row['id'], task=row['task'], method=row['method'],
                            model_id=args.model_id,
                            answer_id=base.stable_id(args.model_id, row['id'], args.seed),
                            choices=[dict(index=0, turns=[dict(role='assistant', message=r['generated_response'])
                                                        for r in records])], tstamp=time.time()))
    path = args.output_dir / 'answers.jsonl'
    tmp = path.with_suffix('.jsonl.tmp')
    with tmp.open('w') as f:
        for answer in answers:
            f.write(json.dumps(answer, ensure_ascii=False) + '\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return len(answers)


def load_progress(selected, args, recover=False):
    path = args.output_dir / 'turns.jsonl'
    audits_path = args.output_dir / 'api_attempts.jsonl'
    reader = read_records if recover else lambda p: base.load_jsonl(p) if p.exists() else []
    expected = {(r['id'], i): base.gold_messages(r['history'], i)
                for r in selected for i in range(len(r['history']))}
    saved = {}
    for r in reader(path):
        key = (r['dialogue_id'], r['turn_index'])
        if (key not in expected or key in saved or r.get('error')
                or r['prompt_history'] != expected[key] or r['model_id'] != args.model_id
                or r['returned_model'] != args.model or not r['generated_response'].strip()
                or r.get('finish_reason') not in ('stop', 'length') or r.get('choice_index') != 0):
            raise ValueError('invalid or incompatible saved turn records')
        saved[key] = r
    filtered = {}
    for audit in reader(audits_path):
        case = filtered_case(audit, args)
        if case is not None:
            key = (case['dialogue_id'], case['turn_index'])
            if key not in expected:
                raise ValueError('Filtered audit is outside the selected dataset')
            # A later already-saved successful response takes precedence.
            if key not in saved:
                filtered[key] = case
    return expected, saved, filtered


def write_filtered(args, filtered):
    atomic_json(args.output_dir / 'filtered_cases.json', dict(
        count=len(filtered), policy='preserve_unresolved_continue_no_automatic_retry',
        cases=[filtered[key] for key in sorted(filtered)]))


def validate_output(selected, args):
    from judge_safedial import validate_and_join
    expected, saved, filtered = load_progress(selected, args)
    pending = set(expected) - set(saved) - set(filtered)
    complete_ids = {r['id'] for r in selected if all((r['id'], i) in saved for i in range(len(r['history'])))}
    answers_path = args.output_dir / 'answers.jsonl'
    answers = base.load_jsonl(answers_path) if answers_path.exists() else []
    if {a['id'] for a in answers} != complete_ids:
        raise ValueError('Answer coverage differs from successfully completed dialogues')
    for row, answer in validate_and_join(answers, selected, 0):
        if answer['model_id'] != args.model_id or len(answer['choices']) != 1:
            raise ValueError('Answer model or choices differ from manifest')
        for i, turn in enumerate(answer['choices'][0]['turns']):
            if turn.get('error') or turn['message'] != saved[(row['id'], i)]['generated_response']:
                raise ValueError('Answer does not match saved successful turn')
    report = dict(passed=not pending, processing_complete=not pending,
                  complete=not pending and not filtered, expected_turns=len(expected),
                  successful_turns=len(saved), filtered_turns=len(filtered), pending_turns=len(pending),
                  complete_dialogues=len(complete_ids), expected_dialogues=len(selected),
                  excluded_dialogue_ids=sorted({key[0] for key in filtered}), judge_calls=0)
    atomic_json(args.output_dir / 'validation.json', report)
    write_filtered(args, filtered)
    print(json.dumps(report), flush=True)
    return report


def generate(client, selected, args):
    path = args.output_dir / 'turns.jsonl'
    audits_path = args.output_dir / 'api_attempts.jsonl'
    expected, saved, filtered = load_progress(selected, args, recover=True)
    write_filtered(args, filtered)
    write_answers(selected, saved, args)
    pending = [(r, i) for r in selected for i in range(len(r['history']))
               if (r['id'], i) not in saved and (r['id'], i) not in filtered]
    print(f'Saved {len(saved)}/{len(expected)} turns; filtered {len(filtered)}; pending {len(pending)}', flush=True)
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        for offset in range(0, len(pending), args.parallel):
            batch = pending[offset:offset + args.parallel]
            futures = [pool.submit(request_turn, client, r, i, args) for r, i in batch]
            failed = False
            for (row, index), future in zip(batch, futures):
                record, audits = future.result()
                base.append_jsonl(audits_path, audits)
                if record is None:
                    case = filtered_case(audits[-1], args)
                    if case is not None:
                        filtered[(row['id'], index)] = case
                        print(f'FILTERED dialogue={row["id"]} turn={index}; retained for review, continuing', flush=True)
                    else:
                        print(f'FAILED dialogue={row["id"]} turn={index}: {audits[-1]["error"]}', flush=True)
                        failed = True
                else:
                    base.append_jsonl(path, [record])
                    saved[(row['id'], index)] = record
            dialogues = write_answers(selected, saved, args)
            write_filtered(args, filtered)
            print(f'Saved {len(saved)}/{len(expected)} turns, {dialogues}/{len(selected)} dialogues; filtered {len(filtered)}', flush=True)
            if failed:
                print('Stopped after failed request(s); rerun identical command to resume.', flush=True)
                return 2
    (args.output_dir / 'runtime_stats.json').write_text(json.dumps(dict(
        device='api', gpu_name=None, elapsed_seconds=time.perf_counter() - started,
        scope='current_invocation', processing_complete=True, complete=not filtered,
        dialogues=sum(all((r['id'], i) in saved for i in range(len(r['history']))) for r in selected),
        turns=len(saved), filtered_turns=len(filtered),
        usage_source='api_attempts.jsonl includes all returned responses; unknown usage on transport failures'
    ), indent=2) + '\n')
    return 0


def main(argv=None):
    args = parse_args(argv)
    rows = base.load_jsonl(args.dataset)
    base.validate_dataset(rows, args.dataset)
    selected = base.select_dialogues(rows, args)
    print(f'Validated {len(selected)} dialogues / {sum(len(r["history"]) for r in selected)} turns; {args.model}')
    if args.validate_only:
        return 0
    if args.prepare_resume or args.validate_output:
        if not (args.output_dir / 'run_config.json').exists():
            raise ValueError('Existing run manifest required for offline resume/validation')
        with (args.output_dir / '.generation.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            ensure_api_manifest(args, selected, migrate=args.prepare_resume)
            if args.prepare_resume:
                expected, saved, filtered = load_progress(selected, args)
                write_filtered(args, filtered)
                print(f'Resume ready: {len(saved)} saved, {len(filtered)} filtered, '
                      f'{len(expected) - len(saved) - len(filtered)} pending; no API calls')
                return 0
            return 0 if validate_output(selected, args)['passed'] else 2
    from judge_safedial import load_env_file
    from openai import OpenAI
    load_env_file(args.env_file)
    # This arm records official OpenAI provenance, never silently uses a gateway.
    endpoint = os.getenv('OPENAI_BASE_URL', '').strip().rstrip('/') or 'https://api.openai.com/v1'
    if endpoint != 'https://api.openai.com/v1':
        raise ValueError('This arm requires the official OpenAI endpoint')
    if not os.getenv('OPENAI_API_KEY', '').strip():
        raise ValueError('OPENAI_API_KEY is missing; set it in .env or the environment')
    with OpenAI(base_url=endpoint, timeout=args.timeout, max_retries=0) as client:
        model = client.models.retrieve(args.model)
        if model.id != args.model:
            raise ValueError('Pinned model access check failed')
        print(f'Model metadata access PASS: {model.id}; generation access checked on first request')
        if args.check_access:
            return 0
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with (args.output_dir / '.generation.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            ensure_api_manifest(args, selected)
            return generate(client, selected, args)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print(f'Run stopped: {type(exc).__name__}; check configuration, API access and saved audits.', file=sys.stderr)
        sys.exit(1)
