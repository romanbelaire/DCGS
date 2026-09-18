#!/usr/bin/env python3
"""Read-only snapshot of saved SafeDial answers, judgments and compact progress.

Run from a source checkout or pass --source. The destination must be fresh.
Active files are captured independently, not as a transactional run checkpoint.
Full model-call journals, weights and credentials are intentionally excluded.
"""
import argparse
import collections
import datetime
import gzip
import hashlib
import json
from pathlib import Path


def read_jsonl_prefix(path):
    with path.open('rb') as handle:
        data = handle.read(path.stat().st_size)
    end = data.rfind(b'\n') + 1
    complete = data[:end]
    records = [json.loads(line) for line in complete.splitlines() if line.strip()]
    return complete, records, len(data) - end


def snapshot(source, destination):
    destination.mkdir(parents=True, exist_ok=False)
    started = datetime.datetime.now().astimezone().isoformat()
    inventory, runs = [], []

    def save(relative, data, source_path=None, details=None):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if len(data) > 1_000_000 and target.suffix == '.jsonl':
            target = target.with_suffix('.jsonl.gz')
            stored = gzip.compress(data, compresslevel=6, mtime=0)
        else:
            stored = data
        target.write_bytes(stored)
        inventory.append({'path': str(target.relative_to(destination)), 'bytes': len(stored),
                          'sha256': hashlib.sha256(stored).hexdigest(),
                          'uncompressed_sha256': hashlib.sha256(data).hexdigest(),
                          'source': str(source_path.relative_to(source)) if source_path else None,
                          **(details or {})})

    for family in ('safedial_baseline', 'safedial_dcgs'):
        for folder in sorted((source / 'outputs' / family).iterdir()):
            if not folder.is_dir() or not (folder / 'run_config.json').exists():
                continue
            relative = Path(family) / folder.name
            manifest = json.loads((folder / 'run_config.json').read_text())
            row = {'family': family, 'run': folder.name, 'selected_dialogues': len(manifest.get('selected_ids', [])),
                   'snapshot_kind': 'independent_file_prefixes; not_a_resume_checkpoint'}
            # Capture answer files before journals: an active run may advance between reads.
            candidates = [folder / 'answers.jsonl']
            candidates += sorted(f for f in folder.glob('*.json') if f.name != 'answers.jsonl')
            candidates += sorted(f for f in folder.glob('*.jsonl') if f.name not in
                                 ('answers.jsonl', 'turns.jsonl', 'events.jsonl', 'attempts.jsonl'))
            for judgments in sorted(folder.glob('judg*')):
                if judgments.is_dir():
                    candidates += sorted(f for f in judgments.iterdir() if f.suffix in ('.json', '.jsonl') and f.is_file())
            for f in candidates:
                if not f.exists():
                    continue
                if f.suffix == '.jsonl':
                    data, records, trailing = read_jsonl_prefix(f)
                    details = {'records': len(records), 'incomplete_tail_bytes_omitted': trailing}
                    if f.name == 'answers.jsonl':
                        row['exported_dialogues'] = len(records)
                        row['exported_error_turns'] = sum(bool(t.get('error')) for a in records
                                                        for c in a.get('choices', []) for t in c.get('turns', []))
                else:
                    data = f.read_bytes()
                    # Abort rather than publish a partially rewritten JSON document.
                    obj = json.loads(data)
                    details = {}
                    if f.name == 'aggregate.json':
                        row.setdefault('judge_aggregates', {})[f.parent.name] = obj
                    if f.name in ('validation.json', 'coverage.json', 'status.json'):
                        row[f.name] = obj
                save(relative / f.relative_to(folder), data, f, details)
            f = folder / 'turns.jsonl'
            if f.exists():
                _, records, trailing = read_jsonl_prefix(f)
                latest = {}
                for record in records:
                    latest[(record['dialogue_id'], record.get('choice_index', 0), record['turn_index'])] = record
                compact = []
                for key, record in sorted(latest.items()):
                    item = {k: record[k] for k in ('dialogue_id', 'choice_index', 'turn_index', 'seed', 'error',
                            'completion_tokens', 'prompt_tokens', 'latency_seconds', 'tstamp') if k in record}
                    item['generated_response_sha256'] = hashlib.sha256(str(record.get('generated_response')).encode()).hexdigest()
                    if 'tpo' in record:
                        item['candidate_count'] = len(record['tpo'].get('candidates', []))
                        item['skipped_candidates'] = sum(bool(e.get('candidate_failure')) for e in record['tpo'].get('events', []))
                    compact.append(item)
                data = ''.join(json.dumps(r, sort_keys=True) + '\n' for r in compact).encode()
                save(relative / 'turn_status.jsonl', data, f, {'derived': True, 'source_records': len(records),
                     'incomplete_tail_bytes_omitted': trailing})
                row.update(unique_recorded_turns=len(latest), successful_recorded_turns=sum(not r.get('error') for r in latest.values()),
                           error_recorded_turns=sum(bool(r.get('error')) for r in latest.values()))
            row['generation_complete_claim'] = 'only_if_saved_validation_or_coverage_explicitly_confirms; counts_alone_are_not_validation'
            runs.append(row)
    summary = {'snapshot_started': started, 'snapshot_finished': datetime.datetime.now().astimezone().isoformat(),
               'scope': 'all_local_SafeDial_run_directories; historical_and_active_runs_kept_separate',
               'limitations': ['Live files captured independently; counts may differ across files.',
                              'No new GPU audit or judge execution performed by snapshot.',
                              'Full per-call journals and original turns remain local; compact turn_status is not a replay journal.',
                              'A run with no validation is not established complete; historical failures are retained.'],
               'runs': runs, 'files': inventory}
    (destination / 'snapshot.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    lines = ['# SafeDialBench saved results', '', f'Captured {started}.', '',
             'These are saved results, including historical and still-running experiments. '
             'They are not all final benchmark scores. Consult each saved validation and judge aggregate.', '',
             '| Run | Exported dialogues | Unique recorded turns | Recorded turn errors |',
             '| --- | ---: | ---: | ---: |']
    for row in runs:
        lines.append(f"| {row['run']} | {row.get('exported_dialogues', 0)} | {row.get('unique_recorded_turns', 0)} | {row.get('error_recorded_turns', 0)} |")
    lines += ['', 'Large JSONL files use standard gzip compression. `snapshot.json` lists file hashes, '
              'sources, per-run metadata, and saved judge aggregates.', '',
              'Answer exports and journals are captured independently while jobs may advance. '
              'The compact `turn_status` files deduplicate repeated turn attempts using the latest record. '
              'Exported dialogues can contain errors in legacy runners; check the error fields. '
              'Original call journals, model weights, caches, credentials, and the external dataset are not included. '
              'This snapshot cannot be used as a resumable execution directory.', '']
    (destination / 'README.md').write_text('\n'.join(lines))
    print(json.dumps({'runs': len(runs), 'files': len(inventory), 'stored_bytes': sum(f['bytes'] for f in inventory),
                      'destination': str(destination)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    snapshot(args.source.resolve(), args.destination.resolve())
