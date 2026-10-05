"""Offline regressions and a tiny training/save/load workflow; these are not thesis scores."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from transformers import BatchEncoding, CLIPConfig, CLIPModel, RobertaConfig, RobertaModel

from datasets.authenticheck_data import ASPECTS, build_targets, fingerprint, parse_annotations, read_splits
from datasets.clip_cache import ClipFeatureCache
from datasets.multimodal_dataset import MultiModalDataset, multimodal_collate_fn
from inference import format_aspects, predict_frame
from models.encoders.clip_encoder import FrozenCLIPEncoder
from models.encoders.text_encoder import TextEncoder
from models.fusion.cross_attention import CrossAttention
from models.model import CLIPCACG
from runtime import load_checkpoint, load_config
from training.engine import aspect_sentiment_loss
from training.train import main as train


class TinyTokenizer:
    def __call__(self, text, max_length, **kwargs):
        values = [2] + [5 + ord(char) % 25 for char in text[:max_length - 2]] + [3]
        padding = max_length - len(values)
        return {'input_ids': torch.tensor([values + [0] * padding]),
                'attention_mask': torch.tensor([[1] * len(values) + [0] * padding])}


class TinyImageEncoder(nn.Module):
    output_dim = 8

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.AdaptiveAvgPool2d((2, 2)))
        self.requires_grad_(False)
        self.calls = 0

    def forward(self, images):
        self.calls += 1
        return self.network(images).flatten(2).transpose(1, 2)


class FakeFrozenEncoder:
    constructions = 0
    dimension = 8

    def __init__(self, name, revision, device, cache_dir):
        type(self).constructions += 1
        self.revision = revision

    def encode(self, text, images):
        vector = np.arange(1, 9, dtype=np.float32)
        vector /= np.linalg.norm(vector)
        return {'text': vector, 'image': vector.copy() if images else np.zeros(8, dtype=np.float32),
                'clip_score': 1.0 if images else 0.0}


def config_for(directory):
    config = copy.deepcopy(load_config())
    config['model'].update(embedding_dim=8, attention_heads=2, dropout=0.0,
                           clip_model='offline-fixture', clip_revision='test-revision')
    config['data'].update(image_size=8, max_length=8)
    config['training'].update(epochs=1, batch_size=2, lr=0.001, num_workers=0)
    for name in config['output']:
        config['output'][name] = str(directory / name)
    return config


def tiny_model(config):
    backbone = RobertaModel(RobertaConfig(vocab_size=32, hidden_size=8, intermediate_size=16,
                                         num_hidden_layers=1, num_attention_heads=2,
                                         max_position_embeddings=32, pad_token_id=0))
    text = TextEncoder(dimension=8, backbone=backbone)
    return CLIPCACG(config, clip_dimension=8, text_encoder=text, image_encoder=TinyImageEncoder())


def fixture_frame():
    rows = []
    for product, partition in enumerate(['train', 'train', 'validation', 'test']):
        for number in range(4):
            rows.append({'review_id': f'r{product}-{number}', 'product_id': f'p{product}',
                         'review_text': f'product {product} review {number}', 'review_image_urls': '',
                         'ground_truth': 'authentic', 'star_rating': 3, 'image_urls': [],
                         'label': 0 if number != 3 else 1,
                         'annotations': {ASPECTS[number]: {number % 3}} if number != 3 else {},
                         'partition': partition, 'fold': 0 if partition == 'train' else -1})
    return pd.DataFrame(rows)


class AdaptationTests(unittest.TestCase):
    def test_mixed_polarities_are_preserved_and_absence_is_not_neutral(self):
        annotations = parse_annotations(json.dumps([
            {'category': 'design', 'text': 'comfortable', 'sentiment': 'positive'},
            {'category': 'design', 'text': 'hard to grip', 'sentiment': 'negative'},
            {'category': 'design', 'text': 'fits well', 'sentiment': 'positive'},
            {'category': 'performance', 'text': 'fast', 'sentiment': 'positive'},
        ]))
        aspects, sentiments = build_targets(annotations)
        self.assertEqual(aspects.sum(), 1)
        np.testing.assert_allclose(sentiments[ASPECTS.index('design')], [0.5, 0, 0.5])
        self.assertEqual(sentiments[ASPECTS.index('value')].sum(), 0)

    def test_sentiment_loss_ignores_unannotated_aspects(self):
        logits = torch.randn(1, 6, 3, requires_grad=True)
        outputs = {'aspect_logits': torch.zeros(1, 6, requires_grad=True), 'sentiment_logits': logits}
        targets = torch.zeros(1, 6, 3)
        targets[0, 0, [0, 2]] = 0.5
        aspect_sentiment_loss(outputs, torch.tensor([[1., 0, 0, 0, 0, 0]]), targets).backward()
        self.assertGreater(float(logits.grad[0, 0].abs().sum()), 0)
        self.assertEqual(float(logits.grad[0, 1:].abs().sum()), 0)

    def test_shared_manifest_rejects_product_leakage(self):
        frame = fixture_frame()
        frame.loc[frame.partition.eq('validation'), 'product_id'] = 'p0'
        payload = {'version': 'product-70-15-15-oof5-v1', 'dataset_sha256': fingerprint(frame),
                   'records': frame[['review_id', 'product_id', 'partition', 'fold']].to_dict('records')}
        raw = frame.drop(columns=['partition', 'fold'])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'splits.json'
            path.write_text(json.dumps(payload), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Product leakage'):
                read_splits(raw, path)

    def test_cache_is_reused_and_invalidated_by_changed_review_text(self):
        with tempfile.TemporaryDirectory() as directory:
            config = config_for(Path(directory))
            frame = fixture_frame().head(2)
            cache = ClipFeatureCache(config)
            cache.prepare(frame, encoder_factory=FakeFrozenEncoder)
            before = FakeFrozenEncoder.constructions
            self.assertEqual(cache.prepare(frame, encoder_factory=FakeFrozenEncoder)['computed'], 0)
            self.assertEqual(FakeFrozenEncoder.constructions, before)
            entry = cache.get(frame.iloc[0])
            self.assertEqual(entry['clip_score'], 0)
            self.assertFalse(entry['image'].any())
            changed = frame.iloc[0].to_dict() | {'review_text': 'different review'}
            with self.assertRaises(FileNotFoundError):
                cache.get(changed)

    def test_missing_images_are_masked_and_do_not_call_the_visual_encoder(self):
        with tempfile.TemporaryDirectory() as directory:
            config = config_for(Path(directory))
            frame = fixture_frame().head(2)
            cache = ClipFeatureCache(config)
            cache.prepare(frame, encoder_factory=FakeFrozenEncoder)
            dataset = MultiModalDataset(frame, TinyTokenizer(), cache, config)
            batch = multimodal_collate_fn([dataset[0], dataset[1]])
            self.assertFalse(batch['image_mask'].any())
            model = tiny_model(config).eval()
            inputs = {key: batch[key] for key in ['input_ids', 'attention_mask', 'images', 'image_mask', 'clip_text', 'clip_image']}
            with torch.no_grad():
                result = model(**inputs)
                inputs['images'] = torch.randn_like(inputs['images']) * 100
                repeated = model(**inputs)
            self.assertEqual(model.image_encoder.calls, 0)
            self.assertTrue(torch.isfinite(result['sentiment_logits']).all())
            torch.testing.assert_close(result['aspect_logits'], repeated['aspect_logits'])

    def test_attention_uses_multiple_regions_and_is_sensitive_to_the_query(self):
        torch.manual_seed(42)
        attention = CrossAttention(dim=8, heads=2, dropout=0).eval()
        image = torch.randn(1, 4, 8)
        text = torch.randn(1, 3, 8)
        args = (image, torch.ones(1, 3, dtype=torch.bool), torch.ones(1, 4, dtype=torch.bool), torch.zeros(1, 8))
        attention(text, *args)
        first, _ = attention.text_to_image(text, image, image, need_weights=False)
        second, _ = attention.text_to_image(text * 3, image, image, need_weights=False)
        self.assertFalse(torch.allclose(first, second))

    def test_real_photos_and_missing_photos_collate_and_backpropagate_together(self):
        with tempfile.TemporaryDirectory() as directory:
            config = config_for(Path(directory))
            frame = fixture_frame().head(2).copy()
            urls = ['https://example.test/red.png', 'https://example.test/green.png']
            frame.at[0, 'image_urls'] = urls
            cache = ClipFeatureCache(config)
            cache.images.directory.mkdir(parents=True)
            for url, color in zip(urls, ['red', 'green']):
                Image.new('RGB', (8, 8), color).save(cache.images.path_for(url), format='PNG')
            cache.prepare(frame, encoder_factory=FakeFrozenEncoder)
            dataset = MultiModalDataset(frame, TinyTokenizer(), cache, config)
            batch = multimodal_collate_fn([dataset[0], dataset[1]])
            self.assertEqual(batch['image_mask'].tolist(), [[True, True], [False, False]])
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = tiny_model(config).to(device)
            with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                outputs = model(**{name: batch[name].to(device) for name in ['input_ids', 'attention_mask', 'images', 'image_mask', 'clip_text', 'clip_image']})
                loss = aspect_sentiment_loss(outputs, batch['aspect_targets'].to(device), batch['sentiment_targets'].to(device))
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertEqual(model.image_encoder.calls, 1)
            self.assertGreater(float(model.image_proj.weight.grad.abs().sum()), 0)
            self.assertTrue(all(parameter.grad is None for parameter in model.image_encoder.parameters()))

    def test_frozen_clip_extracts_without_gradients_and_zeros_missing_images(self):
        config = CLIPConfig(projection_dim=8,
                            text_config={'vocab_size': 32, 'hidden_size': 8, 'intermediate_size': 16,
                                         'num_hidden_layers': 1, 'num_attention_heads': 2,
                                         'max_position_embeddings': 8, 'eos_token_id': 3, 'bos_token_id': 2},
                            vision_config={'hidden_size': 8, 'intermediate_size': 16,
                                           'num_hidden_layers': 1, 'num_attention_heads': 2,
                                           'image_size': 8, 'patch_size': 4})
        model = CLIPModel(config)

        class Processor:
            def __call__(self, text=None, images=None, **kwargs):
                if text is not None:
                    return BatchEncoding({'input_ids': torch.tensor([[2, 5, 3]]),
                                          'attention_mask': torch.ones(1, 3, dtype=torch.long)})
                return BatchEncoding({'pixel_values': torch.ones(len(images), 3, 8, 8)})

        with patch('models.encoders.clip_encoder.CLIPModel.from_pretrained', return_value=model), \
             patch('models.encoders.clip_encoder.CLIPProcessor.from_pretrained', return_value=Processor()):
            encoder = FrozenCLIPEncoder('offline-fixture', 'test-revision', 'cpu', '.')
            self.assertFalse(any(parameter.requires_grad for parameter in encoder.model.parameters()))
            self.assertFalse(encoder.model.training)
            self.assertEqual(encoder.encode('review', [])['clip_score'], 0)
            output = encoder.encode('review', [Image.new('RGB', (8, 8))])
            self.assertEqual(output['image'].shape, (8,))
            self.assertTrue(np.isfinite(output['clip_score']))

    def test_training_checkpoint_round_trip_and_json_use_only_review_context(self):
        with tempfile.TemporaryDirectory() as directory:
            config = config_for(Path(directory))
            frame = fixture_frame()
            cache = ClipFeatureCache(config)
            cache.prepare(frame, encoder_factory=FakeFrozenEncoder)
            model = tiny_model(config)
            checkpoint_path = train(frame, config, tokenizer=TinyTokenizer(), model=model)
            checkpoint = load_checkpoint(checkpoint_path)
            expected = set(frame.loc[frame.partition.eq('train') & frame.label.eq(0), 'review_id'])
            self.assertEqual(set(checkpoint['training_metadata']['fit_review_ids']), expected)
            self.assertFalse(set(checkpoint['training_metadata']['fit_review_ids']) & set(frame.loc[frame.partition.eq('test'), 'review_id']))
            selected = frame[frame.partition.eq('test')].drop(columns=['annotations', 'label', 'ground_truth'])
            output, raw, _ = predict_frame(selected, checkpoint_path, tokenizer=TinyTokenizer(), model=tiny_model(config))
            self.assertEqual(len(output['reviews']), len(selected))
            forced = format_aspects('unseen context', np.ones(6), raw['sentiment_probabilities'][0], 0.5)
            self.assertEqual(len(forced), 6)
            self.assertEqual(set(forced[0]), {'category', 'text', 'sentiment'})
            self.assertTrue(all(item['text'] == 'unseen context' for item in forced))


if __name__ == '__main__':
    unittest.main()
