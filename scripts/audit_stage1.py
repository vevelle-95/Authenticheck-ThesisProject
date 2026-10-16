"""Diagnose Stage 1 using saved train/validation features, without test predictions.

Candidates are diagnostic fits only: this script never replaces a trained bundle.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight

from model_contract import CLASS_NAMES, FEATURE_COLUMNS, INPUT_VERSION, ROBERTA_TARGET_COLUMN
from model_data import fingerprint, load_experiment
from model_metrics import classification_metrics
from stage2.train_xgboost import validate_features
from stage2.evaluate_models import check_training_provenance


def audit_oof_predictions(assigned, oof, metadata, confidence_threshold=0.99):
    """Audit cross-fitted text predictions, keeping originals and drafts separate."""
    training = assigned[assigned.partition.eq('train')].copy()
    if (metadata.get('dataset_sha256') != fingerprint(assigned)
            or metadata.get('input_version') != INPUT_VERSION
            or metadata.get('target_column') != ROBERTA_TARGET_COLUMN):
        raise ValueError('OOF metadata does not match the dataset and text_label contract')
    folds = metadata.get('folds', [])
    if len(folds) != 5 or {row.get('fold') for row in folds} != set(range(5)):
        raise ValueError('OOF audit requires provenance for all five folds')
    for record in folds:
        fit = training[training.fold.ne(record['fold'])]
        held = training[training.fold.eq(record['fold'])]
        for key, expected in [('fit_review_ids', set(fit.review_id)), ('predicted_review_ids', set(held.review_id)),
                              ('fit_product_ids', set(fit.product_id)), ('predicted_product_ids', set(held.product_id))]:
            if set(record.get(key, [])) != expected:
                raise ValueError(f'OOF fold {record["fold"]}: {key} mismatch')
        if set(fit.product_id) & set(held.product_id):
            raise ValueError('OOF fitting and prediction products overlap')
    if oof.review_id.duplicated().any() or set(oof.review_id) != set(training.review_id):
        raise ValueError('OOF predictions must cover training reviews exactly once')
    merged = training.merge(oof, on=['review_id', 'product_id', 'fold'], validate='one_to_one')
    if len(merged) != len(training) or not all(
            row.probability_source == f'oof_fold_{row.fold}' for row in merged.itertuples()):
        raise ValueError('OOF row product/fold provenance mismatch')
    probabilities = merged[list(FEATURE_COLUMNS[:4])].apply(pd.to_numeric, errors='raise')
    import numpy as np
    values = probabilities.to_numpy()
    if (not np.isfinite(values).all() or (values < 0).any()
            or not np.allclose(values.sum(axis=1), 1, atol=1e-5)):
        raise ValueError('Invalid OOF probability vectors')
    merged['prediction'] = values.argmax(axis=1)
    merged['confidence'] = values.max(axis=1)
    merged['wrong'] = merged.prediction.ne(merged.text_target)
    merged['high_confidence_wrong'] = merged.wrong & merged.confidence.ge(confidence_threshold)
    merged['source_review_id'] = merged.get('source_review_id', pd.Series('', index=merged.index)).fillna('')
    merged['population'] = merged.source_review_id.map(lambda source: 'synthetic' if source else 'original')
    merged['source_family'] = merged.source_review_id.where(merged.source_review_id.ne(''), merged.review_id)

    def summary(rows):
        return {'count': len(rows), 'metrics_against_text_label': classification_metrics(
                    rows.text_target, rows.prediction, range(4)),
                'metrics_against_ground_truth': classification_metrics(rows.label, rows.prediction, range(4)),
                'error_count': int(rows.wrong.sum()),
                'high_confidence_error_count': int(rows.high_confidence_wrong.sum()),
                'text_label_differs_from_ground_truth_count': int(rows.text_target.ne(rows.label).sum())}

    populations = {name: summary(rows) for name, rows in [('all', merged),
                   ('original', merged[merged.population.eq('original')]),
                   ('synthetic', merged[merged.population.eq('synthetic')])]}
    per_fold = [{'fold': int(fold), 'original': summary(rows[rows.population.eq('original')]),
                 'synthetic': summary(rows[rows.population.eq('synthetic')])} for fold, rows in merged.groupby('fold')]
    sources = assigned.set_index('review_id')
    families = []
    for source_id, rows in merged.groupby('source_family'):
        synthetic = rows[rows.population.eq('synthetic')]
        if synthetic.empty:
            continue
        source = sources.loc[source_id]
        families.append({'source_review_id': source_id, 'product_id': str(source.product_id),
                         'fold': int(source.fold), 'text_label': CLASS_NAMES[int(source.text_target)],
                         'ground_truth': CLASS_NAMES[int(source.label)], 'source_review_text': source.review_text,
                         'synthetic_count': len(synthetic), 'family_count': len(rows),
                         'synthetic_error_count': int(synthetic.wrong.sum()),
                         'synthetic_high_confidence_error_count': int(synthetic.high_confidence_wrong.sum())})
    families.sort(key=lambda row: (-row['synthetic_count'], row['source_review_id']))
    errors = []
    for row in merged[merged.wrong].sort_values(['population', 'fold', 'review_id']).to_dict('records'):
        source = sources.loc[row['source_family']]
        errors.append({key: row[key] for key in ('review_id', 'product_id', 'review_text', 'population')} | {
            'fold': int(row['fold']), 'source_review_id': row['source_family'],
            'product_title': row.get('product_title', ''), 'product_description': row.get('product_description', ''),
            'source_review_text': source.review_text, 'review_status': row.get('review_status', ''),
            'text_label': CLASS_NAMES[int(row['text_target'])], 'ground_truth': CLASS_NAMES[int(row['label'])],
            'oof_prediction': CLASS_NAMES[int(row['prediction'])], 'confidence': float(row['confidence']),
            'high_confidence_wrong': bool(row['high_confidence_wrong']),
            'probabilities': {label: float(row[column]) for label, column in zip(CLASS_NAMES, FEATURE_COLUMNS[:4])}})
    # Fixed, reproducible review sample: up to three confident errors per true
    # text class per population, spread over source families before repeats.
    samples = []
    for population in ('original', 'synthetic'):
        for label in CLASS_NAMES:
            candidates = [row for row in errors if row['population'] == population
                          and row['text_label'] == label and row['high_confidence_wrong']]
            seen, selected = set(), []
            for row in candidates:
                if row['source_review_id'] not in seen:
                    selected.append(row)
                    seen.add(row['source_review_id'])
                if len(selected) == 3:
                    break
            samples.extend(selected)
    synthetic_count = int(merged.population.eq('synthetic').sum())
    return {'scope': 'saved OOF predictions on training rows only; no refitting or test predictions',
            'class_order': list(CLASS_NAMES), 'primary_target': ROBERTA_TARGET_COLUMN,
            'dataset_sha256': fingerprint(assigned), 'high_confidence_threshold': confidence_threshold,
            'populations': populations, 'per_fold': per_fold, 'source_families': families,
            'source_concentration': {'synthetic_rows': synthetic_count, 'augmented_original_sources': len(families),
                'largest_family_synthetic_count': families[0]['synthetic_count'] if families else 0,
                'top_five_sources_fraction_of_synthetic_rows':
                    sum(row['synthetic_count'] for row in families[:5]) / synthetic_count if synthetic_count else None},
            'review_sample': samples, 'oof_errors': errors,
            'limitations': ['Labels are preserved; a model disagreement does not prove a label is wrong.',
                           'Synthetic variants are correlated, not independent additional human reviews.',
                           'OOF models and the final encoder use different training products; their probability distributions can differ.',
                           'No causal claim about augmentation follows from this observational audit.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='data/test_reviews.csv')
    parser.add_argument('--splits', default='data/splits_dataset.json')
    parser.add_argument('--augmentations', default='data/augmented_reviews.csv')
    parser.add_argument('--allow-unreviewed-augmentations', action='store_true')
    parser.add_argument('--features', default='data/quality_balanced_v5/features/6d_features.csv')
    parser.add_argument('--bundle', default='models/quality_balanced_v5')
    parser.add_argument('--oof-only', action='store_true', help='Audit saved OOF predictions; skip validation fits')
    parser.add_argument('--oof', default='data/quality_balanced_v5/features/oof_probabilities.csv')
    parser.add_argument('--output', default='reports/generated/stage1_validation_audit.json')
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f'Preserve the earlier audit; choose a new --output: {output}')
    assigned = load_experiment(args.data, args.splits, args.augmentations,
                               allow_unreviewed=args.allow_unreviewed_augmentations)
    if args.oof_only:
        metadata = json.loads(Path(args.oof).with_suffix('.metadata.json').read_text(encoding='utf-8'))
        report = audit_oof_predictions(assigned, pd.read_csv(args.oof, dtype={'review_id': str, 'product_id': str}), metadata)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
        print(json.dumps({'populations': report['populations'], 'source_concentration': report['source_concentration']}))
        print(f'Saved OOF audit: {output}')
        return
    check_training_provenance(assigned, args.bundle)
    metadata = json.loads(Path(args.features).with_suffix('.metadata.json').read_text(encoding='utf-8'))
    if metadata.get('dataset_sha256') != fingerprint(assigned) or metadata.get('input_version') != INPUT_VERSION:
        raise ValueError('Feature metadata does not match this experiment')
    features = validate_features(pd.read_csv(args.features, dtype={'review_id': str, 'product_id': str}), assigned)
    train = features[features.partition.eq('train')].copy()
    val = features[features.partition.eq('validation')].copy()
    original_ids = set(pd.read_csv(args.data, dtype=str, keep_default_na=False).review_id)
    columns = list(FEATURE_COLUMNS)
    target = val.ground_truth.to_numpy()
    probabilities = val[columns[:4]].to_numpy()
    current = xgb.XGBClassifier()
    current.load_model(Path(args.bundle) / 'xgboost_meta_classifier.json')
    current_predictions = current.predict(val[columns]).astype(int)
    text_predictions = probabilities.argmax(axis=1)
    baselines = {
        'roberta_only_against_ground_truth': classification_metrics(target, text_predictions, range(4)),
        'current_saved_xgboost': classification_metrics(target, current_predictions, range(4)),
    }
    groups = {'probabilities': columns[:4], 'probabilities_rating': columns[:4] + columns[5:],
              'probabilities_clip': columns[:5], 'all_six': columns}
    # Small, declared grid; no test-set evaluation or parameter selection here.
    settings = [
        {'n_estimators': 100, 'max_depth': 2, 'learning_rate': 0.05},
        {'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.1},
        {'n_estimators': 100, 'max_depth': 2, 'learning_rate': 0.05,
         'min_child_weight': 10, 'reg_lambda': 10, 'gamma': 1},
    ]
    trials = []
    for population in ('original_and_synthetic', 'original_only'):
        fit = train if population == 'original_and_synthetic' else train[train.review_id.isin(original_ids)]
        if set(fit.ground_truth) != set(range(4)):
            raise ValueError(f'{population} must contain all four training classes')
        for group, selected in groups.items():
            for parameters in settings:
                model = xgb.XGBClassifier(**parameters, random_state=42, eval_metric='mlogloss', n_jobs=2)
                model.fit(fit[selected], fit.ground_truth,
                          sample_weight=compute_sample_weight('balanced', fit.ground_truth))
                predicted = model.predict(val[selected]).astype(int)
                trials.append({'training_population': population, 'training_rows': len(fit),
                               'feature_group': group, 'settings': parameters,
                               'metrics': classification_metrics(target, predicted, range(4))})
    trials.sort(key=lambda trial: trial['metrics']['macro_f1'], reverse=True)
    lookup = assigned.set_index('review_id')
    errors = []
    for index, row in enumerate(val.to_dict('records')):
        if int(target[index]) == int(current_predictions[index]):
            continue
        source = lookup.loc[row['review_id']]
        errors.append({'review_id': row['review_id'], 'product_id': row['product_id'],
                       'review_text': source.review_text, 'ground_truth': CLASS_NAMES[int(target[index])],
                       'text_label': CLASS_NAMES[int(source.text_target)],
                       'xgboost_prediction': CLASS_NAMES[int(current_predictions[index])],
                       'roberta_prediction': CLASS_NAMES[int(text_predictions[index])],
                       'features': {name: float(row[name]) for name in columns}})
    distribution = []
    for name, frame in [('train_original', train[train.review_id.isin(original_ids)]),
                        ('train_synthetic', train[~train.review_id.isin(original_ids)]), ('validation', val)]:
        for label, rows in frame.groupby('ground_truth'):
            distribution.append({'cohort': name, 'ground_truth': CLASS_NAMES[int(label)], 'count': len(rows),
                                 'zero_clip_fraction': float(rows[columns[4]].eq(0).mean()),
                                 'mean_clip': float(rows[columns[4]].mean()),
                                 'mean_normalized_rating': float(rows[columns[5]].mean()),
                                 'mean_max_text_probability': float(rows[columns[:4]].max(axis=1).mean())})
    report = {'scope': 'validation only; no test predictions; production bundle unchanged',
              'class_order': list(CLASS_NAMES), 'dataset_sha256': fingerprint(assigned),
              'validation_review_ids': val.review_id.tolist(), 'validation_baselines': baselines,
              'candidate_trials': trials, 'feature_distribution': distribution, 'validation_errors': errors,
              'limitations': [
                  'Original-only candidates still use OOF encoders fitted with eligible synthetic training rows.',
                  'Candidate ranking uses one small validation set; improvement is not proof of generalization.',
                  'Saved image scores reflect image availability at extraction time; URLs can later fail.',
              ]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print('VALIDATION baselines:', json.dumps(baselines))
    for trial in trials[:5]:
        print(json.dumps(trial))
    print(f'Saved validation audit: {output}')


if __name__ == '__main__':
    main()
