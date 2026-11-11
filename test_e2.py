"""Check E2 freezing, optimizer groups, and updates without downloading a model."""
import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import BertConfig, BertForTokenClassification

from e2_partial import configure_e2, set_e2_training_mode, run_experiment, E2Config, inherit_e1_config


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


class E2PipelineTests(unittest.TestCase):
    def fixture(self, directory):
        from e1_head_only import E1Config, save_json
        from dataclasses import asdict
        source = directory / 'e1_selected_config.json'
        save_json(source, asdict(E1Config(learning_rate=7.5e-4)))
        return source

    def test_e1_settings_are_inherited_without_old_machine_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            source = self.fixture(folder)
            source_values = json.loads(source.read_text())
            source_values.update(root='/old/server', dropout=0.2, batch_size=16)
            source.write_text(json.dumps(source_values))
            config = inherit_e1_config(E2Config(root=folder, e1_config_path=source))
            self.assertEqual(config.root, folder)
            self.assertEqual(config.dropout, 0.2)
            self.assertEqual(config.batch_size, 16)
            self.assertEqual(config.head_learning_rate, 7.5e-4)
            self.assertEqual(config.seed, 9444)

    def fake_pipeline(self, folder, test_enabled=True):
        from dataclasses import asdict
        source = self.fixture(folder)
        calls, tests = [], []
        output = folder / 'e2'

        def train(config, path, data):
            calls.append(config)
            score = 0.4 + (0.1 if config.encoder_learning_rate == 2e-5 else 0)
            score += 0.02 if config.head_learning_rate == 2e-4 else 0
            return {'config': asdict(config), 'path': str(path), 'validation_f1': score,
                    'validation': {'typed_exact': dict.fromkeys(['precision', 'recall', 'f1'], score),
                                   'typed_macro': {'f1': score}}, 'training_seconds': 1.0}

        def test(config, trial, tokenizer, data):
            self.assertTrue((output / 'selected_config.json').exists())
            self.assertTrue((output / '5.2.3 Seeds and final results/validation_selection.json').exists())
            self.assertEqual(len(calls), 9)
            tests.append(config.seed)
            return {'seed': config.seed, 'metrics': trial['validation']}

        with patch('e2_partial.load_e1_split', return_value=([], [])) as load:
            result = run_experiment(E2Config(output_dir=output, e1_config_path=source,
                                             evaluate_test=test_enabled, limit_docs=None if test_enabled else 2),
                                    trial_runner=train, prepare_data=lambda config: (None, [], [], [], []),
                                    test_runner=test, render=False)
            if test_enabled:
                self.assertEqual(load.call_args.args[2], 'testing')
                load.assert_called_once()
            else:
                load.assert_not_called()
        return result, calls, tests

    def test_searches_propagate_rates_and_test_all_three_seeds_after_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, calls, tests = self.fake_pipeline(Path(temporary))
        self.assertEqual([config.encoder_learning_rate for config in calls[:3]], [1e-5, 2e-5, 3e-5])
        self.assertTrue(all(config.head_learning_rate == 7.5e-4 for config in calls[:3]))
        self.assertEqual([config.head_learning_rate for config in calls[3:6]], [5e-5, 1e-4, 2e-4])
        self.assertTrue(all(config.encoder_learning_rate == 2e-5 for config in calls[3:]))
        self.assertTrue(all(config.seed == 9444 for config in calls[:6]))
        self.assertEqual([config.seed for config in calls[6:]], [42, 9444, 2026])
        self.assertTrue(all(config.head_learning_rate == 2e-4 for config in calls[6:]))
        self.assertEqual(tests, [42, 9444, 2026])
        self.assertAlmostEqual(result['summary']['validation']['f1']['mean'], 0.52)
        self.assertEqual(result['summary']['validation']['f1']['std'], 0)

    def test_smoke_never_loads_test_and_discards_temporary_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result, calls, tests = self.fake_pipeline(root, False)
            self.assertFalse((root / 'e2').exists())
            self.assertFalse(tests)

    def test_micro_batch_gradients_match_full_weighted_batch(self):
        import copy
        from e2_partial import backward_e2_batch
        torch.manual_seed(9444)
        model = tiny_model()
        configure_e2(model)
        model.eval()
        chunked = copy.deepcopy(model)
        inputs = {'input_ids': torch.randint(0, 32, (5, 7)),
                  'attention_mask': torch.ones(5, 7, dtype=torch.long),
                  'labels': torch.randint(0, 9, (5, 7))}
        inputs['labels'][0, :4] = -100
        weights = torch.arange(1, 10).float()
        logits = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask']).logits
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 9), inputs['labels'].reshape(-1), weight=weights)
        loss.backward()
        numerator, denominator = backward_e2_batch(chunked, inputs, weights, 2)
        self.assertAlmostEqual(numerator / denominator, loss.item(), places=6)
        for (name, expected), (_, actual) in zip(model.named_parameters(), chunked.named_parameters()):
            if expected.requires_grad:
                torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-4, atol=1e-6)
            else:
                self.assertIsNone(actual.grad)

    def test_checkpoint_restores_all_trainable_layers_and_rejects_partial_state(self):
        from e2_partial import save_trainable_checkpoint, restore_trainable_checkpoint
        model = tiny_model()
        configure_e2(model)
        expected = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'best_trainable.pt'
            save_trainable_checkpoint(model, path)
            with torch.no_grad():
                for p in model.parameters():
                    if p.requires_grad:
                        p.add_(1)
            restore_trainable_checkpoint(model, path)
            for name, p in model.named_parameters():
                if name in expected:
                    torch.testing.assert_close(p, expected[name])
            torch.save({'classifier.bias': model.classifier.bias.detach()}, path)
            with self.assertRaisesRegex(ValueError, 'scope'):
                restore_trainable_checkpoint(model, path)


class E2FastModeTests(unittest.TestCase):
    def test_cuda_uses_full_batches_and_larger_evaluation_batches(self):
        from e2_partial import execution_batch_sizes
        config = E2Config()
        self.assertEqual(execution_batch_sizes(config, torch.device('cuda')), (8, 32))
        self.assertEqual(execution_batch_sizes(config, torch.device('mps')), (2, 2))

    def test_aligned_padding_keeps_masks_and_loss_labels_correct(self):
        from e2_partial import make_e2_collator
        collate = make_e2_collator(SimpleNamespace(pad_token_id=0), torch.device('cpu'), align=True)
        batch = [{'input_ids': [101, 1, 102], 'labels': [-100, 1, -100]},
                 {'input_ids': [101, 2, 3, 4, 102], 'labels': [-100, 1, 2, 0, -100]}]
        _, inputs = collate(batch)
        self.assertEqual(tuple(inputs['input_ids'].shape), (2, 8))
        self.assertEqual(inputs['attention_mask'].sum(1).tolist(), [3, 5])
        self.assertTrue((inputs['labels'][0, 3:] == -100).all())
        self.assertTrue((inputs['labels'][1, 5:] == -100).all())

    def test_bfloat16_autocast_has_finite_loss_and_encoder_gradients(self):
        from e2_partial import backward_e2_batch
        model = tiny_model()
        configure_e2(model)
        set_e2_training_mode(model)
        inputs = {'input_ids': torch.randint(0, 32, (2, 8)),
                  'attention_mask': torch.ones(2, 8, dtype=torch.long),
                  'labels': torch.randint(0, 9, (2, 8))}
        numerator, denominator = backward_e2_batch(model, inputs, None, 8,
                                                    amp_dtype=torch.bfloat16, return_tensors=True)
        self.assertTrue(torch.isfinite(numerator))
        self.assertEqual(denominator.item(), 16)
        self.assertTrue(torch.isfinite(model.classifier.weight.grad).all())
        for layer in model.bert.encoder.layer[8:]:
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in layer.parameters()))


if __name__ == '__main__':
    unittest.main()
