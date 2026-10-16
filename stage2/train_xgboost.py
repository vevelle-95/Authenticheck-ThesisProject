"""Fit XGBoost on OOF features and select settings using validation products."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight
from model_contract import (
    CLASS_NAMES, DEFAULT_BUNDLE, DEFAULT_DATA_PATH, DEFAULT_FEATURES_PATH,
    DEFAULT_SPLITS_PATH, FEATURE_COLUMNS, INPUT_VERSION,
)
from model_data import fingerprint, load_experiment
from training_augmentation import augmentation_summary
from model_metrics import classification_metrics

XGB_WEIGHTING_POLICIES = ('row-balanced', 'source-class-balanced')


def training_sample_weights(train, assigned, policy='row-balanced'):
    """Keep equal class mass, optionally giving equal mass to sources per class."""
    if policy not in XGB_WEIGHTING_POLICIES:
        raise ValueError('Unknown XGBoost sample weighting policy')
    if train.review_id.duplicated().any() or assigned.review_id.duplicated().any():
        raise ValueError('Weighting requires unique review IDs')
    rows = assigned.set_index('review_id').loc[train.review_id].copy()
    if not rows.partition.eq('train').all() or not np.array_equal(rows.label.to_numpy(), train.ground_truth.to_numpy()):
        raise ValueError('Sample weights must use aligned training rows and ground_truth')
    if policy == 'row-balanced':
        return compute_sample_weight('balanced', train.ground_truth), None
    sources = rows.get('source_review_id', pd.Series('', index=rows.index)).fillna('')
    families = sources.where(sources.ne(''), rows.index.to_series())
    if not set(families).issubset(set(rows.index[sources.eq('')])):
        raise ValueError('Every source family must include its original training review')
    for column in ('label', 'product_id', 'fold'):
        if rows.groupby(families)[column].nunique().gt(1).any():
            raise ValueError(f'Source family {column} mismatch')
    sizes = families.value_counts()
    family_classes = pd.DataFrame({'family': families, 'label': rows.label}).drop_duplicates('family')
    counts = family_classes.groupby('label').size()
    if set(counts.index) != set(range(4)):
        raise ValueError('All four ground_truth classes need original training sources')
    # Each class gets N/4 mass; each original family within that class gets
    # N/(4*source_count) mass, split equally among its original and variants.
    weights = len(rows) / (4 * rows.label.map(counts) * families.map(sizes))
    summary = {'training_rows': len(rows), 'source_family_count': len(sizes),
               'largest_family_rows': int(sizes.max()), 'min_row_weight': float(weights.min()),
               'max_row_weight': float(weights.max()), 'mean_row_weight': float(weights.mean()),
               'classes': {CLASS_NAMES[label]: {'source_families': int(counts[label]),
                    'training_rows': int(rows.label.eq(label).sum()),
                    'total_weight': float(weights[rows.label.eq(label)].sum())} for label in range(4)}}
    return weights.to_numpy(dtype=float), summary


def validate_features(features, assigned):
    expected = assigned[assigned.partition.ne("test")]
    required = {"review_id", "product_id", "partition", "fold", "ground_truth", "probability_source", *FEATURE_COLUMNS}
    if required.difference(features.columns):
        raise ValueError(f"Missing feature fields: {sorted(required.difference(features.columns))}")
    if features.review_id.duplicated().any() or set(features.review_id) != set(expected.review_id):
        raise ValueError("Feature table must cover training/validation reviews exactly once, excluding test")
    matched = features.merge(
        expected[["review_id", "product_id", "partition", "fold", "label"]],
        on=["review_id", "product_id", "partition", "fold"], validate="one_to_one",
    )
    if len(matched) != len(expected) or not matched.ground_truth.eq(matched.label).all():
        raise ValueError("Features disagree with the shared split/ground-truth manifest")
    for row in matched.itertuples():
        source = f"oof_fold_{row.fold}" if row.partition == "train" else "final_roberta"
        if row.probability_source != source:
            raise ValueError("Training requires OOF probabilities; validation requires final-RoBERTa probabilities")
    values = matched[list(FEATURE_COLUMNS)].apply(pd.to_numeric, errors="raise").to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("Features must be finite")
    if (values[:, :4] < 0).any() or not np.allclose(values[:, :4].sum(axis=1), 1, atol=1e-5):
        raise ValueError("Invalid probability features")
    if (abs(values[:, 4]) > 1.00001).any() or ((values[:, 5] < 0) | (values[:, 5] > 1)).any():
        raise ValueError("Invalid cosine similarity or normalized rating")
    return matched


def train_classifier(features, assigned, output, *, quality_weighting='uniform', sample_weighting='row-balanced'):
    features = validate_features(features, assigned)
    train = features[features.partition.eq("train")]
    validation = features[features.partition.eq("validation")]
    if set(train.ground_truth) != {0, 1, 2, 3} or validation.empty:
        raise ValueError("All training classes and a nonempty validation partition are required")
    output = Path(output)
    if sample_weighting != 'row-balanced' and (output.exists() or output.with_suffix('.metadata.json').exists()):
        raise FileExistsError('Use a fresh output for the source/class-balanced XGBoost candidate')
    weights, weight_summary = training_sample_weights(train, assigned, sample_weighting)
    best = None
    trials = []
    for depth in (2, 4, 6):
        for rate in (0.05, 0.1):
            settings = {"n_estimators": 100, "max_depth": depth, "learning_rate": rate}
            model = xgb.XGBClassifier(**settings, random_state=42, eval_metric="mlogloss", n_jobs=2)
            model.fit(
                train[list(FEATURE_COLUMNS)], train.ground_truth,
                sample_weight=weights,
            )
            metrics = classification_metrics(
                validation.ground_truth.to_numpy(), model.predict(validation[list(FEATURE_COLUMNS)]), range(4),
            )
            trials.append({"settings": settings, "metrics": metrics})
            if best is None or metrics["macro_f1"] > best[0]:
                best = (metrics["macro_f1"], model, settings, metrics)
    output.parent.mkdir(parents=True, exist_ok=True)
    best[1].save_model(output)
    predictions = best[1].predict(validation[list(FEATURE_COLUMNS)]).astype(int)
    probabilities = best[1].predict_proba(validation[list(FEATURE_COLUMNS)])
    output.with_suffix(".metadata.json").write_text(json.dumps({
        "augmentation_usage": augmentation_summary(assigned[assigned.review_id.isin(train.review_id)]),
        "input_version": INPUT_VERSION, "dataset_sha256": fingerprint(assigned),
        "class_order": list(CLASS_NAMES), "feature_columns": list(FEATURE_COLUMNS),
        "quality_weighting": quality_weighting,
        "xgboost_sample_weighting": sample_weighting, "source_weight_summary": weight_summary,
        "settings": best[2], "validation_metrics": best[3], "validation_trials": trials,
        "fit_review_ids": train.review_id.tolist(), "validation_review_ids": validation.review_id.tolist(),
        "validation_predictions": [{'review_id': row.review_id, 'product_id': row.product_id,
            'ground_truth': CLASS_NAMES[int(row.ground_truth)], 'prediction': CLASS_NAMES[int(prediction)],
            'probabilities': values.tolist()} for row, prediction, values in zip(validation.itertuples(), predictions, probabilities)],
    }, indent=2), encoding="utf-8")
    print(json.dumps(best[3], indent=2))
    return best[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--splits", default=str(DEFAULT_SPLITS_PATH))
    parser.add_argument("--augmentations", help="Approved training-only augmentation CSV; use the same file throughout the experiment")
    parser.add_argument("--allow-unreviewed-augmentations", action="store_true", help="Experimental run: include pending drafts without marking them approved")
    parser.add_argument("--features", default=str(DEFAULT_FEATURES_PATH))
    parser.add_argument("--output", default=str(DEFAULT_BUNDLE / "xgboost_meta_classifier.json"))
    parser.add_argument('--sample-weighting', choices=XGB_WEIGHTING_POLICIES, default='row-balanced')
    args = parser.parse_args()
    assigned = load_experiment(args.data, args.splits, args.augmentations,
                               allow_unreviewed=args.allow_unreviewed_augmentations)
    metadata = json.loads(Path(args.features).with_suffix(".metadata.json").read_text(encoding="utf-8"))
    if metadata.get("dataset_sha256") != fingerprint(assigned) or metadata.get("input_version") != INPUT_VERSION:
        raise ValueError("Feature metadata does not match this experiment")
    train_classifier(pd.read_csv(args.features, dtype={"review_id": str, "product_id": str}), assigned, args.output,
                     quality_weighting=metadata.get('quality_weighting', 'uniform'), sample_weighting=args.sample_weighting)


if __name__ == "__main__":
    main()
