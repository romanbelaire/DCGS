"""Prepare a frozen 384-token snapshot for the existing partial judge runner.

No API calls or generation. Preserve this file as snapshot provenance.
"""
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path('/common/home/users/d/darrius.ng.2024/projects/RL-Defense/DCGS')
sys.path.insert(0, str(ROOT / 'scripts'))
import judge_safedial as judge
import judge_safedial_partial as partial
from run_safedial_baseline import gold_messages

DEST = ROOT / 'outputs/safedial_partial_judging/dcgs_384tokens_20260922_v1'


def main():
    dataset_path = ROOT / judge.DEFAULT_DATASET
    prompts_path = ROOT / judge.DEFAULT_PROMPTS
    dataset = judge.load_jsonl(dataset_path)
    questions = {row['id']: row for row in dataset}
    # Create once; never refresh a prepared snapshot while judging it.
    DEST.mkdir(parents=True, exist_ok=False)
    provenance = DEST / 'preparation.py'
    provenance.write_bytes(Path(__file__).read_bytes())
    manifest = {
        'created': datetime.now().astimezone().isoformat(),
        'benchmark': 'SafeDialBench',
        'scope': 'Frozen completed 384-token belief-only dialogues; not full benchmark or random sample',
        'causal_limit': 'Completed-prefix selection and failures may bias results; comparisons with Zephyr do not isolate a belief-budget effect',
        'dataset_dialogues': len(dataset), 'dataset': str(dataset_path),
        'prompts': str(prompts_path), 'judge_protocol': judge.PROTOCOL,
        'judge_model': 'gpt-4o-mini', 'temperature': 0.7, 'max_tokens': 2048,
        'seed': None, 'choice_index': 0, 'methods': {},
        'source_hashes': {}, 'input_hashes': {},
    }
    for path in (dataset_path, prompts_path, Path(judge.__file__), Path(partial.__file__)):
        manifest['source_hashes'][str(path)] = partial.sha(path)
    union = set()
    for method in partial.METHODS:
        source = ROOT / f'outputs/safedial_dcgs/zephyr_{method}_main_belief-only_belief384_full_v1'
        config_bytes = (source / 'run_config.json').read_bytes()
        config = json.loads(config_bytes)
        expected_model = f'zephyr-7b-beta-{method}-main-belief384-belief-only-v1'
        if (config['method'] != method or config['model_id'] != expected_model
                or config['effective_config']['freeform_max_new_tokens'] != 384
                or config['include_history'] is not False
                or config['dataset_sha256'] != partial.sha(dataset_path)):
            raise ValueError(f'Unexpected source identity: {method}')
        # Open/read one atomic answer export; generation can replace it later.
        answer_bytes = (source / 'answers.jsonl').read_bytes()
        if not answer_bytes.endswith(b'\n'):
            raise ValueError(f'Incomplete answer export: {method}')
        answers = [json.loads(line) for line in answer_bytes.splitlines() if line.strip()]
        joined = judge.validate_and_join(answers, dataset, 0)
        if not joined or any(a['model_id'] != expected_model for a in answers):
            raise ValueError(f'Empty or wrong-model exported answers: {method}')
        expected = {(a['id'], i): t['message'] for a in answers
                    for i, t in enumerate(a['choices'][0]['turns'])}
        if any(not text.strip() for text in expected.values()):
            raise ValueError(f'Blank exported response: {method}')
        folder = DEST / method
        folder.mkdir()
        (folder / 'answers.jsonl').write_bytes(answer_bytes)
        (folder / 'generation_config.json').write_bytes(config_bytes)
        flags = {}
        completed_turns = 0
        # This reader captures a file-size boundary and ignores partial tails.
        for turn in partial.records(source / 'turns.jsonl'):
            completed_turns += 1
            key = (turn['dialogue_id'], turn['turn_index'])
            if key not in expected:
                continue
            if (key in flags or turn.get('error')
                    or turn['generated_response'] != expected[key]
                    or turn['model_id'] != expected_model
                    or turn['prompt_history'] != gold_messages(questions[key[0]]['history'], key[1])):
                raise ValueError(f'Duplicate or mismatched exported turn: {method} {key}')
            selected = turn['dcgs_original']['selected_belief']
            flags[key] = {'dialogue_id': key[0], 'turn_index': key[1],
                          'selected_missing_instruction_field': not bool(re.search(r'\bInstruction\s*:', selected, re.I))}
        if set(flags) != set(expected):
            raise ValueError(f'Export missing journal evidence: {method}')
        failures = []
        if (source / 'failures.jsonl').exists():
            failures = [{k: r[k] for k in ('dialogue_id', 'turn_index', 'code', 'failure_id')}
                        for r in partial.records(source / 'failures.jsonl')]
        if any((r['dialogue_id'], r['turn_index']) in expected for r in failures):
            raise ValueError(f'Failed turn in complete export: {method}')
        partial.write_records(folder / 'belief_flags.jsonl', [flags[k] for k in sorted(flags)])
        partial.write_records(folder / 'generation_failures.jsonl', failures)
        ids = {a['id'] for a in answers}
        manifest['methods'][method] = {
            'source': str(source), 'ids': sorted(ids), 'dialogues': len(ids),
            'answer_turns': len(expected),
            'expected_judge_calls': sum(len(judge.judged_turn_indices(q)) for q, _ in joined),
            'by_task': dict(Counter(a['task'] for a in answers)),
            'generation_failures': failures, 'completed_journal_turns_at_read': completed_turns,
            'completed_turns_outside_frozen_answers': completed_turns - len(expected),
            'selected_missing_instruction_field_turns': sum(f['selected_missing_instruction_field'] for f in flags.values()),
            'belief_budget': 384, 'source_config_sha256': hashlib.sha256(config_bytes).hexdigest(),
            'source_answer_snapshot_sha256': hashlib.sha256(answer_bytes).hexdigest(),
        }
        union.update(ids)
    baseline_judge = partial.BASE / 'judgments_gpt-4o-mini'
    baseline_config = json.loads((baseline_judge / 'judge_config.json').read_text())
    wanted = {'protocol': judge.PROTOCOL, 'judge_model': 'gpt-4o-mini',
              'temperature': 0.7, 'max_tokens': 2048, 'seed': None, 'choice_index': 0,
              'dataset_sha256': partial.sha(dataset_path), 'prompts_sha256': partial.sha(prompts_path),
              'answers_sha256': partial.sha(partial.BASE / 'answers.jsonl')}
    if any(baseline_config.get(k) != value for k, value in wanted.items()):
        raise ValueError('Cached Zephyr judge provenance/settings mismatch')
    baseline = [partial.compact_score(r) for r in partial.records(baseline_judge / 'dialogue_scores.jsonl')
                if r['id'] in union]
    partial.write_records(DEST / 'zephyr/dialogue_scores.jsonl', baseline)
    partial.write_json(DEST / 'zephyr/judge_config.json', baseline_config)
    manifest['baseline'] = {'ids': sorted(r['id'] for r in baseline), 'source': str(baseline_judge),
        'reused_scores': len(baseline), 'missing_ids_in_union': sorted(union - {r['id'] for r in baseline}),
        'source_scores_sha256': partial.sha(baseline_judge / 'dialogue_scores.jsonl'),
        'new_baseline_api_calls': 0}
    for path in DEST.rglob('*'):
        if path.is_file():
            manifest['input_hashes'][str(path.relative_to(DEST))] = partial.sha(path)
    partial.write_json(DEST / 'snapshot.json', manifest)
    result = partial.report(DEST)
    print(json.dumps({'snapshot': str(DEST), 'methods': {
        m: {k: manifest['methods'][m][k] for k in ('dialogues', 'answer_turns', 'expected_judge_calls', 'by_task')}
        for m in partial.METHODS}, 'matched_dialogues': result['matched_three_way']['expected_matched_dialogues'],
        'baseline_missing': manifest['baseline']['missing_ids_in_union']}, indent=2))


if __name__ == '__main__':
    main()
