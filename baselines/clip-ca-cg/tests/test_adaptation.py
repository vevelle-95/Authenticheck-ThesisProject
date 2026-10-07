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

from datasets.authenticheck_data import ASPECTS, build_targets, fingerprint, load_experiment, load_reviews, parse_annotations, read_splits
from datasets.clip_cache import ClipFeatureCache
from datasets.image_store import ImageStore
from datasets.multimodal_dataset import MultiModalDataset, multimodal_collate_fn
from inference import BaselinePredictor, format_aspects, predict_frame
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
    def test_loader_preserves_duplicate_reviews_and_requires_unique_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            frame = fixture_frame()
            frame['aspect_annotations'] = '[]'
            frame.loc[4, 'review_text'] = frame.loc[0, 'review_text']
            path = Path(directory) / 'reviews.csv'
            frame.drop(columns=['annotations', 'label', 'image_urls', 'partition', 'fold']).to_csv(path, index=False)
            loaded = load_reviews(path, require_annotations=True)
            self.assertEqual(len(loaded), len(frame))
            self.assertEqual(loaded.loc[0, 'review_text'], loaded.loc[4, 'review_text'])
            frame.loc[4, 'review_id'] = frame.loc[0, 'review_id']
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, 'unique'):
                load_reviews(path, require_annotations=True)

    def test_duplicate_text_must_stay_in_one_partition_and_oof_fold(self):
        rows = []
        for product in range(9):
            partition = 'train' if product < 7 else 'validation' if product == 7 else 'test'
            fold = product % 5 if partition == 'train' else -1
            rows.append({'review_id': f'r{product}', 'product_id': f'p{product}',
                         'review_text': f'Unique opinion {product}', 'partition': partition, 'fold': fold})
        base = pd.DataFrame(rows)
        # Different products in the same fold may legitimately share review text.
        base.loc[5, 'review_text'] = 'UNIQUE  opinion\n0'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'splits.json'
            for leak in (None, 'partitions', 'OOF folds'):
                with self.subTest(leak=leak):
                    frame = base.copy()
                    if leak is not None:
                        frame.loc[7 if leak == 'partitions' else 1, 'review_text'] = 'unique\topinion 0'
                    payload = {'version': 'product-70-15-15-oof5-v1', 'dataset_sha256': fingerprint(frame),
                               'records': frame[['review_id', 'product_id', 'partition', 'fold']].to_dict('records')}
                    path.write_text(json.dumps(payload), encoding='utf-8')
                    raw = frame.drop(columns=['partition', 'fold'])
                    if leak is None:
                        self.assertEqual(len(read_splits(raw, path)), len(frame))
                    else:
                        with self.assertRaisesRegex(ValueError, 'Duplicate review text leakage between ' + leak):
                            read_splits(raw, path)

    def test_image_downloads_do_not_contact_private_addresses(self):
        with tempfile.TemporaryDirectory() as directory:
            images = ImageStore(Path(directory))
            addresses = [(2, 1, 6, '', ('127.0.0.1', 80))]
            with patch('datasets.image_store.socket.getaddrinfo', return_value=addresses), \
                 patch('datasets.image_store.requests.get') as download:
                self.assertIsNone(images.load('http://127.0.0.1/private.jpg'))
                self.assertIsNone(images.load('https://example.com/private.jpg'))
                download.assert_not_called()

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
            predictor = BaselinePredictor(checkpoint_path, tokenizer=TinyTokenizer(), model=tiny_model(config))
            with patch('inference.load_checkpoint', side_effect=AssertionError('Requests must reuse loaded weights')):
                first, _, _ = predictor.predict(selected)
                second, _, _ = predictor.predict(selected)
            self.assertEqual(first, output)
            self.assertEqual(second, first)
            forced = format_aspects('unseen context', np.ones(6), raw['sentiment_probabilities'][0], 0.5)
            self.assertEqual(len(forced), 6)
            self.assertEqual(set(forced[0]), {'category', 'text', 'sentiment'})
            self.assertTrue(all(item['text'] == 'unseen context' for item in forced))


class SharedAugmentationTests(unittest.TestCase):
    def test_both_loaders_use_identical_cohorts_fingerprints_and_mixed_targets(self):
        # Import the shared module through the same project path as the baseline loader.
        import sys
        project_root = Path(__file__).resolve().parents[3]
        if str(project_root) not in sys.path:
            sys.path.append(str(project_root))
        import model_data as own_data
        from training_augmentation import AUGMENTATION_COLUMNS

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            annotations = [{'category': 'product_quality', 'text': 'fragile', 'sentiment': 'negative'},
                           {'category': 'product_quality', 'text': 'sturdy', 'sentiment': 'positive'}]
            rows = [dict(review_id=f'r{product}-{label}', product_id=f'p{product}',
                         product_title='Product', product_description='Listing context',
                         review_text=f'Product {product} feedback {label}: fragile but sturdy',
                         review_image_urls='[]', star_rating=3,
                         ground_truth=('authentic', 'deceptive', 'liv', 'irrelevant')[label],
                         aspect_annotations=json.dumps(annotations) if label == 0 else '[]')
                    for product in range(30) for label in range(4)]
            csv, splits, augmented = directory / 'real.csv', directory / 'splits.json', directory / 'augmentations.csv'
            pd.DataFrame(rows).to_csv(csv, index=False)
            original = own_data.prepare_splits(own_data.load_reviews(csv, True), splits)
            source = original[original.partition.eq('train') & original.label.eq(0)].iloc[0]
            draft = dict(review_id='synthetic-1', source_review_id=source.review_id,
                         review_text='It is fragile in places, yet sturdy elsewhere.', ground_truth='authentic',
                         aspect_annotations=json.dumps(annotations), image_mode='none', image_source_review_id='',
                         augmentation_method='paraphrase', review_status='approved', reviewed_by='human',
                         review_notes='Mixed opinions and exact evidence checked')
            pd.DataFrame([draft], columns=AUGMENTATION_COLUMNS).to_csv(augmented, index=False)
            own = own_data.load_experiment(csv, splits, augmented, True)
            baseline = load_experiment(csv, splits, augmented)
            self.assertEqual(own_data.fingerprint(own), fingerprint(baseline))
            for column in ('review_id', 'product_id', 'review_text', 'partition', 'fold', 'source_review_id', 'image_urls'):
                self.assertEqual(own[column].tolist(), baseline[column].tolist())
            self.assertEqual(baseline.iloc[-1].annotations, {'product_quality': {0, 2}})
            self.assertEqual({a['sentiment'] for a in own.iloc[-1].annotations}, {0, 2})
            self.assertEqual(len(baseline[baseline.partition.eq('test')]), len(original[original.partition.eq('test')]))
            # Changing accepted synthetic content invalidates checkpoint provenance.
            draft['review_text'] = 'Still fragile here and sturdy there.'
            pd.DataFrame([draft], columns=AUGMENTATION_COLUMNS).to_csv(augmented, index=False)
            changed = load_experiment(csv, splits, augmented)
            self.assertNotEqual(fingerprint(changed), fingerprint(baseline))
            # Explicit experimental mode includes drafts without inventing approval.
            draft['review_status'], draft['reviewed_by'] = 'pending', ''
            pd.DataFrame([draft], columns=AUGMENTATION_COLUMNS).to_csv(augmented, index=False)
            self.assertEqual(len(load_experiment(csv, splits, augmented)), len(original))
            pending_own = own_data.load_experiment(csv, splits, augmented, True, allow_unreviewed=True)
            pending_baseline = load_experiment(csv, splits, augmented, allow_unreviewed=True)
            self.assertEqual(own_data.fingerprint(pending_own), fingerprint(pending_baseline))
            self.assertEqual(pending_baseline.iloc[-1].review_status, 'pending')
            self.assertEqual(pending_baseline.iloc[-1].reviewed_by, '')
            self.assertEqual(len(pending_baseline), len(original) + 1)
            self.assertNotEqual(fingerprint(pending_baseline), fingerprint(changed))


if __name__ == '__main__':
    unittest.main()
