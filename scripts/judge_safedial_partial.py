#!/usr/bin/env python3
"""Freeze completed DCGS dialogues, judge them, and compare matched baseline IDs."""
import argparse
from collections import Counter
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import judge_safedial as judge

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('vdcgs', 'rdcgs')
BASE = ROOT / 'outputs/safedial_baseline/zephyr_7b_beta_full'
FIELDS = ('identification_score', 'handling_score', 'consistency_score', 'overall_score')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def records(path):
    """Read only complete records within a captured open-file size."""
    with Path(path).open('rb') as f:
        limit = os.fstat(f.fileno()).st_size
        while line := f.readline():
            if f.tell() > limit or not line.endswith(b'\n'):
                break
            if line.strip():
                yield json.loads(line)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def write_records(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + '\n')


def compact_score(row):
    return {k: row[k] for k in ('id', 'task', 'model', 'judge', 'turn', *FIELDS)}


def compare(groups, expected_ids):
    planned = set.intersection(*(set(expected_ids[m]) for m in groups))
    available = planned.intersection(*(set(rows) for rows in groups.values()))
    ordered = sorted(available)
    summaries = {m: judge.score_summary([rows[i] for i in ordered]) for m, rows in groups.items()}
    deltas = {}
    for method in METHODS:
        deltas[method + '_minus_zephyr'] = {
            field: judge.mean_or_none(groups[method][i][field] - groups['zephyr'][i][field] for i in ordered)
            for field in FIELDS}
    return {'expected_matched_dialogues': len(planned), 'scored_matched_dialogues': len(available),
            'complete_for_matched_subset': available == planned and bool(planned),
            'ids': ordered, 'pending_ids': sorted(planned - available),
            'scores': summaries, 'paired_mean_differences': deltas}


def prepare(destination):
    destination.mkdir(parents=True, exist_ok=False)
    dataset = judge.load_jsonl(ROOT / judge.DEFAULT_DATASET)
    manifest = {'created': datetime.datetime.now().astimezone().isoformat(), 'benchmark': 'SafeDialBench',
        'scope': 'Frozen exported complete dialogues only; not full benchmark or random sample',
        'causal_limit': 'Comparison to Zephyr and structural flags cannot isolate the effect of the 96-token budget',
        'dataset_dialogues': len(dataset), 'dataset': str(ROOT / judge.DEFAULT_DATASET),
        'prompts': str(ROOT / judge.DEFAULT_PROMPTS), 'judge_protocol': judge.PROTOCOL,
        'judge_model': 'gpt-4o-mini', 'temperature': 0.7, 'max_tokens': 2048,
        'seed': None, 'choice_index': 0, 'methods': {}, 'source_hashes': {}, 'input_hashes': {}}
    for path in (ROOT / judge.DEFAULT_DATASET, ROOT / judge.DEFAULT_PROMPTS,
                 ROOT / 'scripts/judge_safedial.py', Path(__file__).resolve()):
        manifest['source_hashes'][str(path)] = sha(path)
    union = set()
    for method in METHODS:
        source = ROOT / 'outputs/safedial_dcgs' / f'zephyr_{method}_main_wildjailbreak_full_v3'
        answer_bytes = (source / 'answers.jsonl').read_bytes()
        if not answer_bytes.endswith(b'\n'):
            raise ValueError('Answer export has an incomplete final record; take a new snapshot')
        answers = [json.loads(line) for line in answer_bytes.splitlines() if line.strip()]
        joined = judge.validate_and_join(answers, dataset, 0)
        if not joined:
            raise ValueError('No fully exported dialogues to judge')
        folder = destination / method
        folder.mkdir()
        (folder / 'answers.jsonl').write_bytes(answer_bytes)
        config_bytes = (source / 'run_config.json').read_bytes()
        (folder / 'generation_config.json').write_bytes(config_bytes)
        ids = {a['id'] for a in answers}
        expected = {(a['id'], i): t['message'] for a in answers for i, t in enumerate(a['choices'][0]['turns'])}
        flags = {}
        completed_turns = 0
        for turn in records(source / 'turns.jsonl'):
            completed_turns += 1
            key = (turn['dialogue_id'], turn['turn_index'])
            if key not in expected:
                continue
            if turn.get('error') or turn['generated_response'] != expected[key]:
                raise ValueError(f'Answer/journal mismatch: {method} {key}')
            selected = turn['dcgs_original']['selected_belief']
            flags[key] = {'dialogue_id': key[0], 'turn_index': key[1],
                          'selected_missing_instruction_field': not bool(re.search(r'\bInstruction\s*:', selected, re.I))}
        if set(flags) != set(expected):
            raise ValueError(f'Answer missing corresponding completed journal turns: {method}')
        write_records(folder / 'belief_flags.jsonl', [flags[k] for k in sorted(flags)])
        failures = []
        if (source / 'failures.jsonl').exists():
            failures = [{k: row[k] for k in ('dialogue_id', 'turn_index', 'code', 'failure_id')}
                        for row in records(source / 'failures.jsonl')]
        write_records(folder / 'generation_failures.jsonl', failures)
        manifest['methods'][method] = {'source': str(source), 'ids': sorted(ids), 'dialogues': len(ids),
            'answer_turns': len(expected), 'expected_judge_calls': sum(len(judge.judged_turn_indices(q)) for q, _ in joined),
            'by_task': dict(Counter(a['task'] for a in answers)), 'generation_failures': failures,
            'completed_journal_turns_at_read': completed_turns,
            'completed_turns_outside_frozen_answers': completed_turns - len(expected),
            'selected_missing_instruction_field_turns': sum(f['selected_missing_instruction_field'] for f in flags.values()),
            'belief_budget': json.loads(config_bytes)['effective_config']['freeform_max_new_tokens']}
        union.update(ids)
    baseline_judge = BASE / 'judgments_gpt-4o-mini'
    config = json.loads((baseline_judge / 'judge_config.json').read_text())
    wanted = {'protocol': judge.PROTOCOL, 'judge_model': manifest['judge_model'], 'temperature': 0.7,
              'max_tokens': 2048, 'seed': None, 'choice_index': 0,
              'dataset_sha256': sha(ROOT / judge.DEFAULT_DATASET), 'prompts_sha256': sha(ROOT / judge.DEFAULT_PROMPTS),
              'answers_sha256': sha(BASE / 'answers.jsonl')}
    if any(config.get(k) != value for k, value in wanted.items()):
        raise ValueError('Existing baseline judge provenance/settings do not match')
    baseline = [compact_score(row) for row in records(baseline_judge / 'dialogue_scores.jsonl') if row['id'] in union]
    write_records(destination / 'zephyr/dialogue_scores.jsonl', baseline)
    write_json(destination / 'zephyr/judge_config.json', config)
    manifest['baseline'] = {'ids': sorted(r['id'] for r in baseline), 'source': str(baseline_judge),
        'reused_scores': len(baseline), 'missing_ids_in_union': sorted(union - {r['id'] for r in baseline}),
        'source_scores_sha256': sha(baseline_judge / 'dialogue_scores.jsonl'), 'new_baseline_api_calls': 0}
    for path in destination.rglob('*'):
        if path.is_file():
            manifest['input_hashes'][str(path.relative_to(destination))] = sha(path)
    write_json(destination / 'snapshot.json', manifest)
    report(destination)
    print(json.dumps({'snapshot': str(destination), 'methods': {
        m: {k: manifest['methods'][m][k] for k in ('dialogues', 'answer_turns', 'expected_judge_calls')}
        for m in METHODS}, 'baseline_api_calls': 0}, indent=2))


def validate_snapshot(folder):
    manifest = json.loads((folder / 'snapshot.json').read_text())
    for name, expected in manifest['input_hashes'].items():
        if sha(folder / name) != expected:
            raise ValueError(f'Frozen input changed: {name}')
    for name, expected in manifest['source_hashes'].items():
        if sha(name) != expected:
            raise ValueError(f'Judge source or dataset changed: {name}')
    return manifest


def report(folder):
    manifest = validate_snapshot(folder)
    expected = {m: manifest['methods'][m]['ids'] for m in METHODS}
    expected['zephyr'] = manifest['baseline']['ids']
    baseline = {r['id']: r for r in records(folder / 'zephyr/dialogue_scores.jsonl')}
    groups = {'zephyr': baseline}
    status, by_flag = {}, {}
    for method in METHODS:
        location = folder / method / 'judgments_gpt-4o-mini'
        scores = location / 'dialogue_scores.jsonl'
        groups[method] = {r['id']: r for r in records(scores)} if scores.exists() else {}
        if not set(groups[method]) <= set(expected[method]):
            raise ValueError('Judged IDs exceed frozen snapshot')
        aggregate = location / 'aggregate.json'
        status[method] = json.loads(aggregate.read_text()) if aggregate.exists() else {'complete': False}
        flags = {(r['dialogue_id'], r['turn_index']): r['selected_missing_instruction_field']
                 for r in records(folder / method / 'belief_flags.jsonl')}
        turns = location / 'judgments.jsonl'
        success = {}
        if turns.exists():
            for row in records(turns):
                if row['status'] == 'success':
                    key = (row['dialogue_id'], row['turn_index'])
                    success[key] = {**row, 'overall_score': sum(row[f] for f in FIELDS[:3]) / 3}
        by_flag[method] = {str(value): judge.score_summary([r for k, r in success.items() if flags[k] == value])
                           for value in (False, True)}
    common = compare(groups, expected)
    paired = {}
    for method in METHODS:
        ids = set(groups[method]) & set(baseline)
        paired[method] = {'dialogues': len(ids), 'method': judge.score_summary([groups[method][i] for i in ids]),
                         'zephyr': judge.score_summary([baseline[i] for i in ids])}
    result = {'scope': manifest['scope'], 'complete_for_full_benchmark': False,
        'complete_for_frozen_snapshot': all(status[m].get('complete') is True for m in METHODS),
        'method_snapshot_counts': {m: manifest['methods'][m]['dialogues'] for m in METHODS},
        'generation_failures_excluded': {m: manifest['methods'][m]['generation_failures'] for m in METHODS},
        'all_available_snapshot_scores': {m: judge.score_summary(list(groups[m].values())) for m in METHODS},
        'matched_three_way': common, 'each_method_vs_matching_zephyr': paired,
        'matched_by_task': {task: compare(
            {m: {i: r for i, r in rows.items() if r['task'] == task} for m, rows in groups.items()},
            {m: [i for i in expected[m] if i in baseline and baseline[i]['task'] == task] for m in groups})
            for task in judge.TASK_TO_PROMPT},
        'turn_scores_by_missing_instruction_field': by_flag,
        'interpretation': 'Structural flag association is confounded; neither it nor Zephyr differences establish a causal token-budget effect.'}
    write_json(folder / 'comparison.json', result)
    return result


def run(folder, dry_run=False):
    manifest = validate_snapshot(folder)
    codes = []
    for method in METHODS:
        command = [sys.executable, '-B', str(ROOT / 'scripts/judge_safedial.py'),
            '--answers', str(folder / method / 'answers.jsonl'),
            '--output-dir', str(folder / method / 'judgments_gpt-4o-mini'),
            '--dataset', manifest['dataset'], '--prompts', manifest['prompts'],
            '--judge-model', manifest['judge_model'], '--temperature', str(manifest['temperature']),
            '--max-tokens', str(manifest['max_tokens']), '--choice-index', '0', '--parallel', '4']
        if dry_run:
            command.append('--dry-run')
        print(f'Judging frozen {method} snapshot; dry_run={dry_run}', flush=True)
        codes.append(subprocess.run(command, cwd=ROOT).returncode)
    if not dry_run:
        result = report(folder)
        print(json.dumps(result['matched_three_way']['scores'], indent=2), flush=True)
        if not result['complete_for_frozen_snapshot']:
            return 1
    return int(any(codes))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('prepare', 'run', 'report'))
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    folder = args.snapshot.resolve()
    if args.operation == 'prepare':
        prepare(folder)
    elif args.operation == 'run':
        return run(folder, args.dry_run)
    else:
        print(json.dumps(report(folder), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
