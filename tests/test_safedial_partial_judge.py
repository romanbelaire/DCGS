import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import judge_safedial_partial as partial


def score(value):
    return {field: value for field in partial.FIELDS}


class PartialJudgeTests(unittest.TestCase):
    def test_comparison_uses_identical_ids(self):
        groups = {'vdcgs': {1: score(100), 2: score(1)}, 'rdcgs': {2: score(2), 3: score(100)},
                  'zephyr': {1: score(4), 2: score(3), 3: score(6)}}
        expected = {method: list(rows) for method, rows in groups.items()}
        result = partial.compare(groups, expected)
        self.assertEqual(result['ids'], [2])
        self.assertEqual(result['scores']['vdcgs']['overall_mean'], 1)
        self.assertEqual(result['paired_mean_differences']['vdcgs_minus_zephyr']['overall_score'], -2)
        self.assertTrue(result['complete_for_matched_subset'])

    def test_missing_scores_do_not_claim_completion(self):
        groups = {'vdcgs': {1: score(1)}, 'rdcgs': {}, 'zephyr': {1: score(2)}}
        result = partial.compare(groups, {method: [1] for method in groups})
        self.assertFalse(result['complete_for_matched_subset'])
        self.assertEqual(result['pending_ids'], [1])
        self.assertIsNone(result['scores']['vdcgs']['overall_mean'])

    def test_journal_snapshot_excludes_partial_and_later_records(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'journal.jsonl'
            path.write_bytes(b'{"id":1}\n{"id":2}')
            rows = partial.records(path)
            self.assertEqual(next(rows), {'id': 1})
            with path.open('ab') as f:
                f.write(b'\n{"id":3}\n')
            self.assertEqual(list(rows), [])

    def test_changed_snapshot_is_rejected_before_judging(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root/'answers.jsonl';path.write_text('{"id":1}\n')
            (root/'snapshot.json').write_text(json.dumps({'input_hashes': {'answers.jsonl': partial.sha(path)},
                                                        'source_hashes': {}}))
            partial.validate_snapshot(root)
            path.write_text('{"id":2}\n')
            with self.assertRaises(ValueError): partial.validate_snapshot(root)


if __name__ == '__main__':
    unittest.main()
