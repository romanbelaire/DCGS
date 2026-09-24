#!/usr/bin/env python3
"""Normalized assistance labels with an audited, offline import of API v2 runs.

The v2 engine stays unchanged while old jobs run. Hooks are process-local and
the v3 manifest pins both files. Only labels change, never requests or raw text.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import shutil
import sys
import unicodedata

import judge_safedial_assistance_api as engine

PROTOCOL = 'safedial_dcgs_api_assistance_same_prompt_v3'
PARSER = 'NFKC/case normalization; remove label punctuation/symbols/whitespace and known ASSIST/numbered/fence wrappers; require exactly YES or NO'
BASE_CONFIG = engine.config_for


def parse_label(raw):
    text = unicodedata.normalize('NFKC', raw).strip()
    # A fenced label may optionally declare a plain-text language.
    fence = re.fullmatch(r'```(?:text|plaintext)?\s*\n(.*?)\n\s*```', text,
                         re.IGNORECASE | re.DOTALL)
    if fence:
        text = fence.group(1)
    # Remove decorative punctuation around wrappers, preserving tag brackets.
    def decoration(char):
        return char.isspace() or char not in '[]/' and unicodedata.category(char)[0] in 'PS'
    while text and decoration(text[0]):
        text = text[1:]
    while text and decoration(text[-1]):
        text = text[:-1]
    numbered = re.fullmatch(
        r'1[.)]\s*\[ASSIST\]\s*2[.)]\s*(.*?)\s*3[.)]\s*\[/ASSIST\]',
        text, re.IGNORECASE | re.DOTALL)
    if numbered:
        text = numbered.group(1)
    else:
        # Some API replies number the continuation rather than the whole
        # ASSIST block (e.g. "1. NO\n2. [/ASSIST]"). Strip only the
        # prompt's small enumerator prefixes at line starts, never body digits.
        text = re.sub(r'(?m)^[ \t]*[1-3][.)][ \t]*', '', text)
        text = re.sub(r'\[/?ASSIST\]', '', text, flags=re.IGNORECASE)
    # Keep letters, digits and unknown control characters. Prose, negations,
    # duplicate/conflicting labels and unknown body digits still fail.
    label = ''.join(c for c in text if not c.isspace() and
                    unicodedata.category(c)[0] not in 'PS').upper()
    if label not in ('YES', 'NO'):
        raise ValueError('Expected exactly YES or NO after label normalization')
    return label


def config_for(args, snapshot, endpoint):
    config = BASE_CONFIG(args, snapshot, endpoint)
    config.update(protocol=PROTOCOL, parser=PARSER)
    config['source_sha256'][str(Path(__file__).resolve())] = engine.local.source.file_hash(__file__)
    return config


@contextmanager
def normalized_engine():
    previous = engine.PROTOCOL, engine.parse_label, engine.config_for
    engine.PROTOCOL, engine.parse_label, engine.config_for = PROTOCOL, parse_label, config_for
    try:
        yield
    finally:
        engine.PROTOCOL, engine.parse_label, engine.config_for = previous


def import_saved(args, source_dir):
    """Import frozen v2 responses, preserving all raw text and API provenance.

    Both directories are locked. An active source is rejected; queue the new
    suite with afterany on the old job. Existing destination calls are never
    replaced on a subsequent resume.
    """
    source_dir = source_dir.resolve()
    destination = args.output_dir
    if (not source_dir.is_dir() or source_dir == destination or
            source_dir in destination.parents or destination in source_dir.parents):
        raise ValueError('Import requires separate existing source and destination directories')
    engine.native.load_env_file(args.env_file)
    endpoint = os.getenv('OPENAI_BASE_URL', '').strip() or 'https://api.openai.com/v1'
    with engine.local.source.lock_output(source_dir), engine.local.source.lock_output(destination):
        oldconfig = json.loads((source_dir / 'judge_config.json').read_text())
        if oldconfig['protocol'] != 'safedial_dcgs_api_assistance_same_prompt_v2':
            raise ValueError('Import source must use the original API v2 protocol')
        for name, expected in oldconfig['source_sha256'].items():
            if engine.local.source.file_hash(name) != expected:
                raise ValueError('Original source-code pin changed; refuse unverified import')
        source_args = argparse.Namespace(**vars(args))
        source_args.output_dir = source_dir
        snapshot, rows = engine.local.load_snapshot(source_args)
        if engine.local.source.file_hash(source_dir / 'frozen/snapshot.json') != oldconfig['snapshot_sha256']:
            raise ValueError('Original judge configuration does not match frozen snapshot')
        oldhash = engine.local.source.digest(oldconfig)
        original = engine.read_judgments(source_dir, rows, oldhash)
        inventory = {n: engine.local.source.file_hash(source_dir / n) for n in
                     ('judge_config.json', 'frozen/snapshot.json', 'frozen/inputs.jsonl')}
        inventory['judgments.jsonl'] = (engine.local.source.file_hash(source_dir / 'judgments.jsonl')
                                      if (source_dir / 'judgments.jsonl').exists() else None)
        origin = {'source_dir': str(source_dir), 'source_files_sha256': inventory}
        target_snapshot = destination / 'frozen'
        if not target_snapshot.exists():
            # No change to source inputs or their provenance; output paths are
            # not part of the frozen input contract.
            if any((destination / n).exists() for n in ('judge_config.json', 'judgments.jsonl')):
                raise ValueError('Destination artifacts exist without frozen inputs')
            staging = destination / '.import-frozen'
            if staging.exists():
                shutil.rmtree(staging)
            shutil.copytree(source_dir / 'frozen', staging)
            staging.rename(target_snapshot)
        target_contract, target_rows = engine.local.load_snapshot(args)
        if target_contract != snapshot or target_rows != rows:
            raise ValueError('Destination frozen inputs differ from import source')
        config = config_for(args, snapshot, endpoint)
        permitted = {'protocol', 'parser', 'source_sha256'}
        if ({k: v for k, v in config.items() if k not in permitted} !=
                {k: v for k, v in oldconfig.items() if k not in permitted}):
            raise ValueError('Import must retain identical model, prompt, request and execution settings')
        engine.local.source.bind_config(destination, config)
        newhash = engine.local.source.digest(config)
        manifest_path = destination / 'normalization_import.json'
        if manifest_path.exists():
            imported = json.loads(manifest_path.read_text())
            if imported['origin'] != origin or imported['destination_config_sha256'] != newhash:
                raise ValueError('Import source or configuration changed after migration')
            with normalized_engine():
                engine.read_judgments(destination, rows, newhash)
            return imported
        migrated, corrected = [], []
        for row in rows:
            record = original.get(row['input_sha256'])
            if record is None:
                continue
            updated = dict(record, judge_config_sha256=newhash)
            updated['imported_from'] = {**origin, 'original_status': record['status'],
                                       'original_error': record.get('error'),
                                       'operation': 'offline label normalization; original raw response preserved'}
            if record.get('finish_reason') == 'stop' and not record.get('refusal'):
                try:
                    label = parse_label(record['raw_output'])
                except ValueError:
                    label = None
                if label is not None:
                    if record['status'] == 'success' and record['label'] != label:
                        raise ValueError('Normalization changed an existing successful label')
                    updated.update(status='success', label=label, error=None)
                    if record['status'] != 'success':
                        corrected.append({'dialogue_id': row['dialogue_id'], 'turn_index': row['turn_index'],
                                          'label': label, 'raw_output': record['raw_output']})
            migrated.append(updated)
        journal_path = destination / 'judgments.jsonl'
        expected_text = ''.join(engine.local.source.canonical(r) + '\n' for r in migrated)
        if journal_path.exists():
            if journal_path.read_text() != expected_text:
                raise ValueError('Destination already contains other judgments; refuse overwrite')
        else:
            engine.local.write_rows(journal_path, migrated)
        with normalized_engine():
            latest = engine.read_judgments(destination, rows, newhash)
            report = engine.summarize(destination, rows, latest)
        imported = {'origin': origin, 'destination_config_sha256': newhash,
                    'imported_records': len(migrated), 'corrected_records': corrected,
                    'turn_counts_after_import': report['turn_counts'],
                    'raw_outputs_unchanged': True, 'api_calls_for_import': 0}
        engine.local.source.atomic_json(manifest_path, imported)
        return imported


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--import-from', type=Path)
    parser.add_argument('--import-only', action='store_true')
    options, remaining = parser.parse_known_args(argv)
    args = engine.parse_args(remaining)
    if options.import_only and not options.import_from:
        raise ValueError('--import-only requires --import-from')
    # Validate original journal with its original parser before installing hooks.
    if options.import_from:
        report = import_saved(args, options.import_from)
        print(json.dumps({'imported_records': report['imported_records'],
                          'normalized_errors': len(report['corrected_records']),
                          'turn_counts_after_import': report['turn_counts_after_import']}), flush=True)
        if options.import_only:
            return 0
    with normalized_engine():
        return engine.main(remaining)


if __name__ == '__main__':
    raise SystemExit(main())
