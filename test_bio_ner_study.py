"""Regression checks for MLM budget, mask grouping, and document isolation."""
import random
import unittest

from bio_ner_study import make_mlm_splits, mask_row


class TokenizerFixture:
    cls_token_id = 101
    sep_token_id = 102
    mask_token_id = 103

    def __len__(self):
        return 128

    def __call__(self, text, **kwargs):
        class Encoding(dict):
            def word_ids(self):
                return [0, 0, 1, 2, 3, 4, 5, 6, 7]

        return Encoding(input_ids=list(range(10, 19)))


def selected_positions(masked):
    return {index for index, label in enumerate(masked["labels"]) if label != -100}


class StudyMaskingTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = TokenizerFixture()
        self.row = {
            "input_ids": [101] + list(range(10, 50)) + [102],
            "words": [list(range(i, i + 2)) for i in range(1, 41, 2)],
            "entities": [[3, 4], [7, 8, 9, 10]],
            "entity_tokens": {3, 4, 7, 8, 9, 10},
        }

    def test_masks_are_reproducible_and_budgeted(self):
        for strategy in ("token", "word", "entity"):
            with self.subTest(strategy=strategy):
                a = mask_row(self.row, strategy, self.tokenizer, random.Random(42))
                b = mask_row(self.row, strategy, self.tokenizer, random.Random(42))
                self.assertEqual(a, b)
                masked, count, _ = a
                positions = selected_positions(masked)
                self.assertEqual(count, 6)
                self.assertEqual(len(positions), count)
                self.assertTrue(positions <= set(range(1, 41)))
                for index in positions:
                    self.assertEqual(masked["labels"][index], self.row["input_ids"][index])

    def test_grouped_strategies_keep_words_and_mentions_complete(self):
        for strategy, groups in (("word", self.row["words"]), ("entity", self.row["entities"])):
            for seed in range(10):
                masked, _, targeted = mask_row(self.row, strategy, self.tokenizer, random.Random(seed))
                positions = selected_positions(masked)
                for group in groups:
                    self.assertIn(len(positions.intersection(group)), (0, len(group)))
                if strategy == "entity":
                    self.assertGreater(targeted, 0)

    def test_mlm_deduplication_and_blocks_independent_of_entities(self):
        def record(pmid, entity):
            return dict(
                pmid=pmid, text=pmid, input_ids=list(range(10, 19)),
                entities=[{"start": 0, "end": 5}] if entity else [],
                bio_tags=["O", "O", "O", "B-Modifier", "I-Modifier", "I-Modifier",
                          "O", "O", "O"] if entity else ["O"] * 9,
            )

        annotated = [record("1", True), record("1", True), record("2", True)]
        baseline = [record("1", False), record("2", False)]
        blocks, groups = make_mlm_splits(annotated, self.tokenizer, 42, block_size=7)
        no_entities, baseline_groups = make_mlm_splits(baseline, self.tokenizer, 42, block_size=7)
        self.assertEqual(groups, baseline_groups)
        self.assertEqual(set(groups["training"] + groups["validation"]), {"1", "2"})
        self.assertFalse(set(groups["training"]) & set(groups["validation"]))
        for split in blocks:
            self.assertEqual([r["input_ids"] for r in blocks[split]],
                             [r["input_ids"] for r in no_entities[split]])
            self.assertTrue(all(not r["entities"] for r in blocks[split]))
            self.assertEqual(sum(len(r["entity_tokens"]) for r in blocks[split]), 3)


if __name__ == "__main__":
    unittest.main()
