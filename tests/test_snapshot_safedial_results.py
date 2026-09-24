import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location(
    'snapshot_safedial_results', Path(__file__).resolve().parents[1] / 'scripts/snapshot_safedial_results.py')
snapshot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(snapshot)


class SnapshotTests(unittest.TestCase):
    def test_compact_prefix_keeps_latest_complete_turn_and_omits_tail(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'turns.jsonl'
            old = {'dialogue_id': 1, 'turn_index': 0, 'generated_response': '', 'error': 'empty'}
            new = {**old, 'generated_response': 'answer', 'error': None,
                   'tpo': {'candidates': [1, 2], 'events': [{'candidate_failure': True}]}}
            tail = b'{"dialogue_id": 2'
            path.write_bytes((json.dumps(old) + '\n' + json.dumps(new) + '\n').encode() + tail)
            compact, count, omitted = snapshot.compact_turn_prefix(path)
            self.assertEqual((count, omitted, len(compact)), (2, len(tail), 1))
            self.assertIsNone(compact[0]['error'])
            self.assertEqual(compact[0]['candidate_count'], 2)
            self.assertEqual(compact[0]['skipped_candidates'], 1)
            self.assertEqual(compact[0]['generated_response_sha256'], hashlib.sha256(b'answer').hexdigest())
            self.assertNotIn('tpo', compact[0])

    def test_malformed_complete_line_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'turns.jsonl'
            path.write_bytes(b'{broken}\n')
            with self.assertRaises(json.JSONDecodeError):
                snapshot.compact_turn_prefix(path)

    def test_evaluator_snapshot_preserves_records_and_hashes_without_locks(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / 'source'
            for family in ('safedial_baseline', 'safedial_dcgs'):
                (source / 'outputs' / family).mkdir(parents=True)
            for family in ('llamaguard', 'assistance', 'safedial_goals', 'safedial_partial_judging'):
                folder = source / 'outputs' / family / 'run'
                folder.mkdir(parents=True)
                (folder / 'config.json').write_text('{"complete": false}\n')
                (folder / '.judge.lock').write_text('not an artifact')
                (folder / '.env').write_text('EXAMPLE=excluded')
                (folder / 'records.jsonl').write_text(json.dumps({'raw': 'x' * 1_000_001}) + '\n{unfinished')
            destination = Path(temporary) / 'snapshot'
            snapshot.snapshot(source, destination)
            report = json.loads((destination / 'snapshot.json').read_text())
            self.assertEqual(len(report['files']), 8)
            for entry in report['files']:
                stored = (destination / entry['path']).read_bytes()
                self.assertEqual(hashlib.sha256(stored).hexdigest(), entry['sha256'])
                data = gzip.decompress(stored) if entry['path'].endswith('.gz') else stored
                self.assertEqual(hashlib.sha256(data).hexdigest(), entry['uncompressed_sha256'])
                if entry['path'].endswith('.gz'):
                    self.assertEqual(entry['records'], 1)
                    self.assertEqual(entry['incomplete_tail_bytes_omitted'], len(b'{unfinished'))
                    self.assertEqual(len(json.loads(data)['raw']), 1_000_001)
            self.assertFalse(list(destination.rglob('.env')))
            self.assertFalse(list(destination.rglob('.judge.lock')))


if __name__ == '__main__':
    unittest.main()
