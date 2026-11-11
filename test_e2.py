"""Check E2 freezing, optimizer groups, and updates without downloading a model."""
import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import BertConfig, BertForTokenClassification

from e2_partial import configure_e2, set_e2_training_mode, run_sweep


def tiny_model(layers=12):
    return BertForTokenClassification(BertConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_attention_heads=2, num_hidden_layers=layers,
        max_position_embeddings=32, num_labels=9,
    ))


class E2Tests(unittest.TestCase):
    def test_optimizer_updates_only_last_four_layers_and_head(self):
        torch.manual_seed(42)
        model = tiny_model()
        groups = configure_e2(model, 2e-5, 1e-4)
        self.assertEqual([g['lr'] for g in groups], [2e-5, 1e-4])
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        expected = {name for name, p in model.named_parameters()
                    if name.startswith('classifier.') or
                    any(name.startswith(f'bert.encoder.layer.{i}.') for i in range(8, 12))}
        actual = {name for name, p in model.named_parameters() if p.requires_grad}
        self.assertEqual(actual, expected)
        self.assertEqual(sum(len(g['params']) for g in groups), len(expected))
        set_e2_training_mode(model)
        optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
        loss = model(input_ids=torch.tensor([[1, 2, 3, 4]]),
                     attention_mask=torch.ones(1, 4, dtype=torch.long),
                     labels=torch.tensor([[-100, 1, 2, -100]])).loss
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        optimizer.step()
        changed = set()
        for name, parameter in model.named_parameters():
            if not torch.equal(before[name], parameter):
                changed.add(name)
            if name not in expected:
                self.assertIsNone(parameter.grad)
                self.assertTrue(torch.equal(before[name], parameter), name)
        self.assertTrue(changed <= expected)
        self.assertIn('classifier.weight', changed)
        for layer in range(8, 12):
            self.assertTrue(any(name.startswith(f'bert.encoder.layer.{layer}.') for name in changed))

    def test_dropout_mode_matches_frozen_scope(self):
        model = tiny_model()
        configure_e2(model)
        model.eval()  # Simulate returning from validation.
        set_e2_training_mode(model)
        self.assertFalse(model.bert.embeddings.training)
        self.assertTrue(all(not layer.training for layer in model.bert.encoder.layer[:8]))
        self.assertTrue(all(layer.training for layer in model.bert.encoder.layer[8:]))
        self.assertTrue(model.dropout.training)
        self.assertTrue(model.classifier.training)

    def test_non_twelve_layer_backbone_is_rejected(self):
        with self.assertRaisesRegex(ValueError, '12-layer'):
            configure_e2(tiny_model(layers=6))


class E2SweepTests(unittest.TestCase):
    def run_fake_sweep(self, directory, validation_only=False):
        args = SimpleNamespace(output_dir=directory, encoder_learning_rates=[1e-5, 2e-5, 3e-5],
                               head_learning_rate=1e-4, seed=42, limit_docs=None,
                               validation_only=validation_only, test=False, evaluate_checkpoint=None)
        calls = []

        def fake_trial(trial):
            calls.append(trial)
            trial.output_dir.mkdir(parents=True)
            if trial.evaluate_checkpoint:
                self.assertFalse(validation_only)
                saved = json.loads((directory / 'selection.json').read_text())
                self.assertEqual(len(saved['candidates']), 3)
                self.assertEqual(saved['selected']['encoder_learning_rate'], 2e-5)
                self.assertEqual(trial.evaluate_checkpoint, directory.resolve() / '2e-05/best_checkpoint')
                self.assertTrue(trial.test)
                metrics = {'typed_exact': {'precision': 0.1, 'recall': 0.1, 'f1': 0.1},
                           'typed_macro': {'f1': 0.1}}
                (trial.output_dir / 'test_metrics.json').write_text(json.dumps(metrics))
                return
            self.assertFalse(trial.test)
            score = {1e-5: 0.4, 2e-5: 0.8, 3e-5: 0.6}[trial.encoder_learning_rate]
            metrics = {'typed_exact': {'precision': score, 'recall': score, 'f1': score},
                       'typed_macro': {'f1': score}}
            for name, value in {
                'summary.json': {'best_epoch': 1, 'best_validation_f1': score, 'training_seconds': 1.0},
                'validation_metrics.json': metrics,
                'history.json': [{'epoch': 1, 'training_loss': 0.5, 'validation': metrics}],
            }.items():
                (trial.output_dir / name).write_text(json.dumps(value))
            (trial.output_dir / 'best_checkpoint').mkdir()
            (trial.output_dir / 'best_checkpoint/model.safetensors').write_bytes(b'test-only dummy')

        with patch('e2_partial.run_trial', side_effect=fake_trial), patch('e2_partial.release_device_cache'):
            result = run_sweep(args)
        return result, calls

    def test_selection_is_persisted_before_only_selected_model_reaches_test(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / 'search'
            result, calls = self.run_fake_sweep(directory)
            self.assertEqual(len(calls), 4)
            self.assertEqual(result['selection']['selected']['encoder_learning_rate'], 2e-5)
            self.assertEqual(result['test']['typed_exact']['f1'], 0.1)
            self.assertTrue((directory / 'learning_curves.csv').is_file())
            self.assertTrue((directory / 'summary.md').is_file())

    def test_validation_only_never_evaluates_test(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, calls = self.run_fake_sweep(Path(temporary) / 'search', validation_only=True)
            self.assertEqual(len(calls), 3)
            self.assertIsNone(result['test'])


if __name__ == '__main__':
    unittest.main()
