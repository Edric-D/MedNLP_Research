"""Regression checks for E1 window coverage and document-level decoding."""
import unittest
from types import SimpleNamespace

from e1_head_only import LABELS, decode_document, make_windows, evaluate_standardized_records


class E1Tests(unittest.TestCase):
    def test_windows_cover_each_training_token_once(self):
        record = {'input_ids': list(range(12)), 'bio_tags': ['O'] * 12,
                  'offset_mapping': [(i, i + 1) for i in range(12)]}
        windows = make_windows([record], SimpleNamespace(cls_token_id=101, sep_token_id=102), 8, 2)
        supervised = []
        for window in windows:
            supervised.extend(window['start'] + i for i, tag in enumerate(window['labels'][1:-1]) if tag != -100)
        self.assertEqual(supervised, list(range(12)))
        self.assertEqual([(w['start'], w['end']) for w in windows], [(0, 6), (4, 10), (8, 12)])

    def test_illegal_i_is_repaired_and_types_remain_distinct(self):
        record = {'record_id': 'doc', 'pmid': '1', 'text': 'aa bb cc',
                  'offset_mapping': [(0, 2), (3, 5), (6, 8)],
                  'entities': [{'start': 0, 'end': 5, 'type': 'DiseaseClass'},
                               {'start': 6, 'end': 8, 'type': 'Modifier'}]}
        tags = [LABELS.index(x) for x in ['I-DiseaseClass', 'I-DiseaseClass', 'I-Modifier']]
        prediction, illegal = decode_document(record, tags)
        self.assertEqual(illegal, 2)
        self.assertEqual(prediction['entities'], record['entities'])
        self.assertEqual(evaluate_standardized_records([record], [prediction])['typed_exact']['f1'], 1.0)

    def test_invalid_stride_is_rejected(self):
        with self.assertRaises(ValueError):
            make_windows([], None, 8, 6)


if __name__ == '__main__':
    unittest.main()
