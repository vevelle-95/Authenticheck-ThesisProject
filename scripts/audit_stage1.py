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

from model_contract import CLASS_NAMES, FEATURE_COLUMNS, INPUT_VERSION
from model_data import fingerprint, load_experiment
from model_metrics import classification_metrics
from stage2.train_xgboost import validate_features
from stage2.evaluate_models import check_training_provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='data/test_reviews.csv')
    parser.add_argument('--splits', default='data/splits_dataset.json')
    parser.add_argument('--augmentations', default='data/augmented_reviews.csv')
    parser.add_argument('--allow-unreviewed-augmentations', action='store_true')
    parser.add_argument('--features', default='data/quality_balanced_v5/features/6d_features.csv')
    parser.add_argument('--bundle', default='models/quality_balanced_v5')
    parser.add_argument('--output', default='reports/generated/stage1_validation_audit.json')
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f'Preserve the earlier audit; choose a new --output: {output}')
    assigned = load_experiment(args.data, args.splits, args.augmentations,
                               allow_unreviewed=args.allow_unreviewed_augmentations)
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
