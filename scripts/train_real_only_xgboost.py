"""Finish the real-only Stage 1 experiment using frozen reference visual features.

Run after real-only OOF generation. Only original training/validation reviews
are processed; no images are downloaded and no test predictions are generated.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import xgboost as xgb

from model_contract import CLASS_NAMES, FEATURE_COLUMNS, INPUT_VERSION, normalize_rating, quality_input_text
from model_data import fingerprint, load_experiment
from scripts.compare_roberta_validation import metrics_with_classes, validate_encoder
from stage1.extract_6d_features import build_features
from stage1.train_roberta import predict_probabilities
from stage2.train_xgboost import train_classifier, validate_features


class FrozenVisualFeatures:
    """Serve verified cached image scores while replacing all text probabilities."""

    def __init__(self, rows, cached, validation_probabilities):
        self.rows = iter(rows)
        self.cached = cached.set_index('review_id')
        self.validation_probabilities = validation_probabilities

    def extract(self, text, image_urls, star_rating, probabilities=None):
        row = next(self.rows)
        if (text != quality_input_text(row) or image_urls != row['image_urls']
                or normalize_rating(star_rating) != normalize_rating(row['star_rating'])):
            raise ValueError('Cached feature row order or inputs changed')
        cached = self.cached.loc[row['review_id']]
        if probabilities is None:
            probabilities = self.validation_probabilities[row['review_id']]
        values = np.asarray(probabilities, dtype=float)
        if values.shape != (4,) or not np.isfinite(values).all() or (values < 0).any() or not np.isclose(values.sum(), 1):
            raise ValueError('Invalid new text probabilities')
        return (values.tolist() + [float(cached.dim5_clip_sim), normalize_rating(star_rating)],
                cached.best_image_url or None, json.loads(cached.image_scores))


def load_reference_cache(path, assigned):
    metadata = json.loads(path.with_suffix('.metadata.json').read_text(encoding='utf-8'))
    if metadata.get('input_version') != INPUT_VERSION or metadata.get('dataset_sha256') != fingerprint(assigned):
        raise ValueError('Reference visual-feature cache belongs to different inputs or data')
    cache = pd.read_csv(path, dtype={'review_id': str, 'product_id': str}, keep_default_na=False)
    validate_features(cache, assigned)
    by_id = assigned.set_index('review_id')
    for row in cache.itertuples():
        original = by_id.loc[row.review_id]
        if not np.isclose(row.dim6_star_rating, normalize_rating(original.star_rating), atol=1e-8):
            raise ValueError(f'{row.review_id}: cached rating differs from the CSV')
        scores = json.loads(row.image_scores)
        if not isinstance(scores, list):
            raise ValueError('Cached image scores must be a list')
        if not scores:
            if row.dim5_clip_sim != 0 or row.best_image_url:
                raise ValueError('Missing usable images must have zero similarity and no best URL')
        else:
            if any(not isinstance(pair, list) or len(pair) != 2
                   or pair[1] not in original.image_urls or not np.isfinite(pair[0])
                   or abs(pair[0]) > 1.00001 for pair in scores):
                raise ValueError(f'{row.review_id}: image scores do not match the input URLs')
            score, url = max(scores, key=lambda pair: pair[0])
            if not np.isclose(score, row.dim5_clip_sim, atol=1e-8) or url != row.best_image_url:
                raise ValueError('Cached best image and similarity disagree')
    return cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='data/test_reviews.csv')
    parser.add_argument('--splits', default='data/splits_dataset.json')
    parser.add_argument('--bundle', type=Path, default=Path('models/real_only_v1'))
    parser.add_argument('--oof', type=Path, default=Path('data/real_only_v1/oof_probabilities.csv'))
    parser.add_argument('--features', type=Path, default=Path('data/real_only_v1/6d_features.csv'))
    parser.add_argument('--reference', type=Path, default=Path('models/quality_balanced_v5'))
    parser.add_argument('--reference-features', type=Path, default=Path('data/quality_balanced_v5/features/6d_features.csv'))
    parser.add_argument('--reference-augmentations', default='data/augmented_reviews.csv')
    parser.add_argument('--allow-unreviewed-reference-augmentations', action='store_true')
    parser.add_argument('--report', type=Path, default=Path('reports/generated/real_only_v1/stage1_validation_comparison.json'))
    args = parser.parse_args()
    output = args.bundle / 'xgboost_meta_classifier.json'
    for path in (output, output.with_suffix('.metadata.json'), args.features,
                 args.features.with_suffix('.metadata.json'), args.report, args.report.with_suffix('.md')):
        if path.exists():
            raise FileExistsError(f'Preserve existing experiment output: {path}')

    original = load_experiment(args.data, args.splits)
    reference_assigned = load_experiment(
        args.data, args.splits, args.reference_augmentations,
        allow_unreviewed=args.allow_unreviewed_reference_augmentations)
    candidate_metadata = validate_encoder(args.bundle / 'dost_roberta', original)
    validate_encoder(args.reference / 'dost_roberta', reference_assigned)
    oof_metadata = json.loads(args.oof.with_suffix('.metadata.json').read_text(encoding='utf-8'))
    if oof_metadata.get('quality_weighting', 'uniform') != candidate_metadata.get('quality_weighting', 'uniform'):
        raise ValueError('OOF and final encoder weighting policies differ')
    cached = load_reference_cache(args.reference_features, reference_assigned)
    source = original[original.partition.ne('test')]
    cache = cached[cached.review_id.isin(source.review_id)].copy()
    if set(cache.review_id) != set(source.review_id):
        raise ValueError('Frozen cache must cover all original training/validation reviews')
    reference_model = xgb.XGBClassifier()
    reference_model.load_model(args.reference / 'xgboost_meta_classifier.json')
    validation = original[original.partition.eq('validation')].copy()
    cached_validation = cache.set_index('review_id').loc[validation.review_id]
    reference_predictions = reference_model.predict(cached_validation[list(FEATURE_COLUMNS)]).astype(int)
    reference_probabilities = reference_model.predict_proba(cached_validation[list(FEATURE_COLUMNS)])
    reference_metadata = json.loads((args.reference / 'xgboost_meta_classifier.metadata.json').read_text(encoding='utf-8'))
    reference_metrics = metrics_with_classes(validation.label, reference_predictions)
    if (set(reference_metadata['fit_review_ids']) != set(reference_assigned.loc[reference_assigned.partition.eq('train'), 'review_id'])
            or set(reference_metadata['validation_review_ids']) != set(validation.review_id)
            or reference_metadata['dataset_sha256'] != fingerprint(reference_assigned)
            or reference_metadata['validation_metrics']['confusion_matrix'] != reference_metrics['confusion_matrix']
            or not np.isclose(reference_metadata['validation_metrics']['macro_f1'], reference_metrics['macro_f1'])):
        raise ValueError('Reference classifier or validation scores no longer match the frozen experiment')

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    model = AutoModelForSequenceClassification.from_pretrained(args.bundle / 'dost_roberta', local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(args.bundle / 'dost_roberta', local_files_only=True)
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    probabilities = predict_probabilities(model, tokenizer, validation)
    text_metrics = metrics_with_classes(validation.label, probabilities.argmax(axis=1))
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    extractor = FrozenVisualFeatures(source.to_dict('records'), cache,
                                     dict(zip(validation.review_id, probabilities.tolist())))
    features = build_features(original, args.oof, args.features, args.bundle / 'dost_roberta', extractor=extractor)
    validate_features(features, original)
    metadata_path = args.features.with_suffix('.metadata.json')
    feature_metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    feature_metadata.update(visual_features_source=str(args.reference_features),
                            visual_source_file_sha256=hashlib.sha256(args.reference_features.read_bytes()).hexdigest(),
                            frozen_visual_features=True, image_downloads_performed=False,
                            final_encoder=str(args.bundle / 'dost_roberta'))
    metadata_path.write_text(json.dumps(feature_metadata, indent=2), encoding='utf-8')
    candidate_model = train_classifier(
        features, original, output, quality_weighting=candidate_metadata.get('quality_weighting', 'uniform'),
        sample_weighting='row-balanced')
    candidate_validation = features.set_index('review_id').loc[validation.review_id]
    candidate_predictions = candidate_model.predict(candidate_validation[list(FEATURE_COLUMNS)]).astype(int)
    candidate_probabilities = candidate_model.predict_proba(candidate_validation[list(FEATURE_COLUMNS)])
    candidate_metrics = metrics_with_classes(validation.label, candidate_predictions)
    report = {
        'partition': 'validation', 'review_count': len(validation),
        'product_count': int(validation.product_id.nunique()), 'class_order': list(CLASS_NAMES),
        'validation_sha256': fingerprint(validation), 'test_predictions_generated': False,
        'candidate_promoted': False, 'target': 'ground_truth', 'reference_directory': str(args.reference),
        'candidate_directory': str(args.bundle), 'reference_xgboost': reference_metrics,
        'candidate_xgboost': candidate_metrics, 'candidate_roberta_only': text_metrics,
        'candidate_minus_reference': {key: candidate_metrics[key] - reference_metrics[key]
                                    for key in ('accuracy', 'macro_precision', 'macro_recall', 'macro_f1')},
        'frozen_visual_features_source': str(args.reference_features),
        'limitations': [
            'One seed and four validation products; no statistical significance is established.',
            'No test evaluation, ABSA evaluation or deployment was performed.',
            'Original-only training and six OOF epochs differ from the reference augmentation/epoch protocol.',
            'Recorded visual scores preserve the earlier image-fetch outcomes; images were not downloaded again.',
        ],
        'records': [
            {'review_id': row.review_id, 'product_id': row.product_id, 'review_text': row.review_text,
             'text_label': CLASS_NAMES[row.text_target], 'ground_truth': CLASS_NAMES[row.label],
             'reference_prediction': CLASS_NAMES[int(reference_predictions[i])],
             'candidate_prediction': CLASS_NAMES[int(candidate_predictions[i])],
             'candidate_roberta_prediction': CLASS_NAMES[int(probabilities[i].argmax())],
             'reference_probabilities': reference_probabilities[i].tolist(),
             'candidate_probabilities': candidate_probabilities[i].tolist()}
            for i, row in enumerate(validation.itertuples())
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    lines = ['# Real-only Stage 1 validation comparison', '',
             f'{len(validation)} original validation reviews; all results below use final ground_truth.', '',
             '| Model | Accuracy | Macro-F1 |', '| --- | ---: | ---: |']
    for name, result in [('Reference XGBoost', reference_metrics), ('Real-only XGBoost', candidate_metrics),
                         ('Real-only RoBERTa alone', text_metrics)]:
        lines.append(f'| {name} | {result["accuracy"]*100:.2f}% | {result["macro_f1"]*100:.2f}% |')
    lines.extend(['', '| Class | Support | Reference precision / recall / F1 | Real-only precision / recall / F1 |',
                  '| --- | ---: | --- | --- |'])
    for name in CLASS_NAMES:
        ref, cand = reference_metrics['per_class'][name], candidate_metrics['per_class'][name]
        fmt = lambda item: ' / '.join(f'{item[key]*100:.2f}%' for key in ('precision', 'recall', 'f1'))
        lines.append(f'| {name} | {ref["support"]} | {fmt(ref)} | {fmt(cand)} |')
    lines.extend(['', 'The candidate is not deployed. Read class errors as well as the overall score.', '',
                  *report['limitations'], ''])
    args.report.with_suffix('.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('reference_xgboost', 'candidate_xgboost', 'candidate_roberta_only')}, indent=2))
    print(f'Saved comparison to {args.report}', flush=True)


if __name__ == '__main__':
    main()
