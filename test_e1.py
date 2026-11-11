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


class E1SearchTests(unittest.TestCase):
    def test_scheduler_boundaries(self):
        from e1_head_only import learning_rate_factor
        self.assertEqual(learning_rate_factor(0, 100, 'constant'), 1)
        self.assertEqual(learning_rate_factor(0, 100, 'linear'), 1)
        self.assertEqual(learning_rate_factor(100, 100, 'linear'), 0)
        self.assertEqual(learning_rate_factor(0, 100, 'linear_warmup'), 0)
        self.assertEqual(learning_rate_factor(10, 100, 'linear_warmup'), 1)
        self.assertAlmostEqual(learning_rate_factor(55, 100, 'linear_warmup'), 0.5)

    def test_weights_ignore_overlap_and_missing_classes(self):
        import torch
        from e1_head_only import training_class_weights
        weights = training_class_weights([{'labels': [-100, 0, 0, 1, -100]}, {'labels': [-100, -100, 1, -100]}], torch)
        self.assertEqual(weights[:2].tolist(), [1.0, 1.0])
        self.assertTrue((weights[2:] == 0).all())

    def test_frozen_encoder_and_head_dropout(self):
        import torch
        from transformers import BertConfig, BertForTokenClassification
        from e1_head_only import configure_e1
        model = BertForTokenClassification(BertConfig(vocab_size=32, hidden_size=16,
            intermediate_size=32, num_attention_heads=2, num_hidden_layers=1, num_labels=9))
        configure_e1(model, 0.2)
        before = {name: value.detach().clone() for name, value in model.named_parameters()}
        model.train()
        model.base_model.eval()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
        model(input_ids=torch.tensor([[1, 2, 3]]), labels=torch.tensor([[-100, 1, -100]])).loss.backward()
        optimizer.step()
        self.assertEqual(model.dropout.p, 0.2)
        self.assertFalse(model.base_model.training)
        self.assertTrue(model.dropout.training)
        self.assertFalse(torch.equal(before['classifier.weight'], model.classifier.weight))
        for name, parameter in model.base_model.named_parameters():
            self.assertTrue(torch.equal(before[f'bert.{name}'], parameter))
            self.assertIsNone(parameter.grad)

    def fake_experiment(self, folder, batch_improves):
        from dataclasses import asdict
        from pathlib import Path
        from unittest.mock import patch
        from e1_head_only import E1Config, run_experiment
        calls, test_calls = [], []

        def trial(config, output, data):
            calls.append((str(output), config))
            score = 0.4 + (0.01 if config.learning_rate == 0.0015 else 0)
            score += 0.01 if config.dropout == 0.2 else 0
            score += 0.01 if config.loss == 'weighted_cross_entropy' else 0
            score += 0.01 if config.scheduler == 'linear_warmup' else 0
            score += 0.001 if batch_improves and config.batch_size == 16 else 0
            score += 0.01 if config.weight_decay == 0.05 else 0
            return {'config': asdict(config), 'path': str(output), 'validation_f1': score,
                    'validation': {'typed_exact': dict.fromkeys(['precision', 'recall', 'f1'], score),
                                   'typed_macro': {'f1': score}},
                    'training_seconds': 2, 'validation_seconds': 1, 'best_epoch': 1}

        def test(config, selected, tokenizer, data):
            self.assertTrue((Path(folder) / 'selected_config.json').exists())
            self.assertTrue((Path(folder) / '5.1.5 Seeds and final results/validation_selection.json').exists())
            self.assertEqual([c.seed for p, c in calls if 'seed_' in p], [42, 9444, 2026])
            test_calls.append(config.seed)
            return {'seed': config.seed, 'metrics': selected['validation']}

        with patch('e1_head_only.load_e1_split', return_value=([], [])) as load:
            result = run_experiment(E1Config(output_dir=Path(folder)), trial_runner=trial,
                prepare_data=lambda config: (None, [], [], [], []), test_runner=test, render=False)
            load.assert_called_once()
            self.assertEqual(load.call_args.args[2], 'testing')
        self.assertEqual(test_calls, [42, 9444, 2026])
        return result, calls

    def test_stages_propagate_best_parameters_and_any_batch_gain_opens_gate(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            result, calls = self.fake_experiment(folder, True)
        config = result['selected_config']
        self.assertEqual([config[key] for key in ['learning_rate', 'dropout', 'loss', 'scheduler', 'batch_size', 'weight_decay']],
                         [0.0015, 0.2, 'weighted_cross_entropy', 'linear_warmup', 16, 0.05])
        self.assertEqual(result['summary']['seeds'], [42, 9444, 2026])
        self.assertEqual(result['summary']['validation']['f1']['std'], 0)
        self.assertEqual(len(calls), 22)
        self.assertTrue(all(c.seed == 9444 for path, c in calls if 'seed_' not in path))

    def test_no_batch_gain_skips_weight_decay(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            result, calls = self.fake_experiment(folder, False)
        self.assertEqual(result['selected_config']['batch_size'], 8)
        self.assertEqual(result['selected_config']['weight_decay'], 0.01)
        self.assertFalse(any('Weight decay/trial' in path for path, _ in calls))

    def test_smoke_cannot_report_formal_test_scores(self):
        from e1_head_only import E1Config, run_experiment
        with self.assertRaisesRegex(ValueError, 'Smoke'):
            run_experiment(E1Config(limit_docs=2))

class E1RecoveryTests(unittest.TestCase):
    def test_encoder_chunking_preserves_head_loss_and_gradients(self):
        import copy
        import torch
        from transformers import BertConfig, BertForTokenClassification
        from e1_head_only import configure_e1, forward_e1_logits
        torch.manual_seed(9444)
        original = BertForTokenClassification(BertConfig(vocab_size=32, hidden_size=16,
            intermediate_size=32, num_attention_heads=2, num_hidden_layers=1, num_labels=9))
        configure_e1(original, 0.2)
        chunked = copy.deepcopy(original)
        inputs = {'input_ids': torch.randint(0, 32, (5, 7)),
                  'attention_mask': torch.ones(5, 7, dtype=torch.long)}
        targets = torch.randint(0, 9, (5, 7))
        targets[:, 0] = -100
        criterion = torch.nn.CrossEntropyLoss(weight=torch.arange(1, 10).float())
        outputs, gradients = [], []
        for model, size in [(original, 5), (chunked, 2)]:
            model.train()
            model.base_model.eval()
            torch.manual_seed(2026)
            logits = model(**inputs).logits if size == 5 else forward_e1_logits(model, inputs, size)
            criterion(logits.reshape(-1, 9), targets.reshape(-1)).backward()
            outputs.append(logits.detach())
            gradients.append(model.classifier.weight.grad)
            self.assertTrue(all(p.grad is None for p in model.base_model.parameters()))
        torch.testing.assert_close(outputs[0], outputs[1], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(gradients[0], gradients[1], rtol=1e-5, atol=1e-6)

    def test_recovery_reuses_complete_trials_and_restarts_partial_trial(self):
        import json
        import tempfile
        from pathlib import Path
        from dataclasses import asdict, replace
        from unittest.mock import Mock
        from e1_head_only import E1Config, run_or_reuse_trial, save_json
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            config = E1Config(resume=True)
            complete, partial = folder / 'complete', folder / 'partial'
            result = {'config': asdict(config), 'validation_f1': 0.42}
            save_json(complete / 'summary.json', result)
            for name in ('best_head.pt', 'validation_metrics.json', 'validation_predictions.json', 'history.json'):
                (complete / name).touch()
            runner = Mock(return_value={'new': True})
            reused = run_or_reuse_trial(config, complete, None, runner)
            self.assertEqual(reused['validation_f1'], 0.42)
            runner.assert_not_called()
            with self.assertRaisesRegex(ValueError, 'differs'):
                run_or_reuse_trial(replace(config, learning_rate=0.1), complete, None, runner)
            partial.mkdir()
            (partial / 'history.json').write_text('[]')
            self.assertEqual(run_or_reuse_trial(config, partial, None, runner), {'new': True})
            self.assertFalse(partial.exists())
            self.assertEqual(len(list(folder.glob('partial.interrupted_*'))), 1)
            runner.assert_called_once_with(config, partial, None)


if __name__ == '__main__':
    unittest.main()
