"""Quality metrics and shared ten-aspect multi-label ABSA evaluation."""

import hashlib
import json

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, multilabel_confusion_matrix

from model_contract import ASPECTS, CLASS_NAMES, POLARITIES, ABSA_EVALUATION_PROTOCOL, SENTIMENT_TARGET_POLICY, sentiment_indices


def classification_metrics(targets, predictions, labels):
    if len(targets) == 0:
        return {"count": 0, "accuracy": None, "macro_precision": None, "macro_recall": None, "macro_f1": None}
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predictions, labels=list(labels), zero_division=0,
    )
    return {
        "count": len(targets), "accuracy": float(accuracy_score(targets, predictions)),
        "macro_precision": float(precision.mean()), "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()), "support": support.astype(int).tolist(),
        "confusion_matrix": confusion_matrix(targets, predictions, labels=list(labels)).tolist(),
    }


def aspect_filtering_ablation(gold, filtered, unfiltered):
    """Paired product/aspect errors, weighting each review equally within an aspect.

    Each distinct polarity contributes once to a review/aspect mean. Undefined
    scores remain None; summaries use the intersection of defined model scores
    and gold scores, and explicitly report exclusions and missing predictions.
    """
    def index(rows):
        result = {row['id']: row for row in rows}
        if len(result) != len(rows):
            raise ValueError('Aggregation review IDs must be unique')
        return result

    indexed = [index(rows) for rows in (gold, filtered, unfiltered)]
    if not all(set(rows) == set(indexed[0]) for rows in indexed):
        raise ValueError('Aggregation requires identical review cohorts')
    for review_id in indexed[0]:
        if len({rows[review_id]['product_id'] for rows in indexed}) != 1:
            raise ValueError('Aggregation product IDs disagree')
        if any(rows[review_id]['label'] not in CLASS_NAMES for rows in indexed):
            raise ValueError('Invalid quality label in aggregation')
        if indexed[1][review_id]['label'] != indexed[2][review_id]['label']:
            raise ValueError('Filtered and unfiltered quality predictions disagree')

    products = sorted({row['product_id'] for row in gold})

    def aggregates(rows, authentic_only):
        values = {}
        for row in rows:
            if authentic_only and row['label'] != 'authentic':
                continue
            polarities = {}
            for item in row['aspectSentiment']:
                category, sentiment = item['category'], item['sentiment'].lower()
                if category not in ASPECTS or sentiment not in ('negative', 'neutral', 'positive'):
                    raise ValueError('Invalid aspect/sentiment in aggregation')
                polarities.setdefault(category, set()).add({'negative': -1, 'neutral': 0, 'positive': 1}[sentiment])
            for category, distinct in polarities.items():
                values.setdefault((row['product_id'], category), []).append(sum(distinct) / len(distinct))
        return values

    cohorts = [aggregates(gold, True), aggregates(filtered, True), aggregates(unfiltered, False)]
    def pairs(row):
        return {(item['category'], item['sentiment'].lower()) for item in row['aspectSentiment']}
    for review_id in indexed[0]:
        expected = pairs(indexed[2][review_id]) if indexed[1][review_id]['label'] == 'authentic' else set()
        if pairs(indexed[1][review_id]) != expected:
            raise ValueError('Filtered predictions must apply only the Stage 1 gate to the same ABSA predictions')
    pairs = []
    for product in products:
        for category in ASPECTS:
            entries = [cohort.get((product, category), []) for cohort in cohorts]
            truth, with_filter, without_filter = [sum(values) / len(values) if values else None for values in entries]
            reasons = [name for name, score in zip(
                ('no_ground_truth', 'no_filtered_prediction', 'no_unfiltered_prediction'),
                (truth, with_filter, without_filter)) if score is None]
            eligible = not reasons
            filtered_error = abs(with_filter - truth) if truth is not None and with_filter is not None else None
            unfiltered_error = abs(without_filter - truth) if truth is not None and without_filter is not None else None
            pairs.append({'product_id': product, 'category': category,
                          'ground_truth': truth, 'filtered': with_filter, 'unfiltered': without_filter,
                          'ground_truth_review_count': len(entries[0]), 'filtered_review_count': len(entries[1]),
                          'unfiltered_review_count': len(entries[2]),
                          'filtered_absolute_error': filtered_error, 'unfiltered_absolute_error': unfiltered_error,
                          'paired_eligible': eligible, 'exclusion_reasons': reasons,
                          'improvement': unfiltered_error - filtered_error if eligible else None})

    def mean(values):
        return sum(values) / len(values) if values else None

    summaries = []
    for category in ASPECTS:
        rows = [row for row in pairs if row['category'] == category]
        referenced = [row for row in rows if row['ground_truth'] is not None]
        eligible = [row for row in rows if row['paired_eligible']]
        summaries.append({'category': category, 'product_count': len(rows),
                          'ground_truth_product_count': len(referenced), 'paired_product_count': len(eligible),
                          'excluded_ground_truth_product_count': len(referenced) - len(eligible),
                          'no_ground_truth_product_count': len(rows) - len(referenced),
                          'missing_filtered_prediction_count': sum(row['filtered'] is None for row in referenced),
                          'missing_unfiltered_prediction_count': sum(row['unfiltered'] is None for row in referenced),
                          'filtered_without_ground_truth_count': sum(row['ground_truth'] is None and row['filtered'] is not None for row in rows),
                          'unfiltered_without_ground_truth_count': sum(row['ground_truth'] is None and row['unfiltered'] is not None for row in rows),
                          'filtered_mae': mean([row['filtered_absolute_error'] for row in eligible]),
                          'unfiltered_mae': mean([row['unfiltered_absolute_error'] for row in eligible]),
                          'improvement': mean([row['improvement'] for row in eligible])})
    eligible = [row for row in pairs if row['paired_eligible']]
    evaluable_aspects = [row for row in summaries if row['paired_product_count']]
    return {'protocol': 'product-aspect-equal-review-v1', 'aspects': list(ASPECTS),
            'score_range': [-1, 1], 'aggregation_unit': 'mean_distinct_polarities_per_review_then_mean_reviews_per_aspect',
            'baseline': 'same trained ABSA model without Stage 1 filtering',
            'exclusion_policy': 'Paired MAEs use only product/aspect rows with all three scores defined; missing scores are null, not zero.',
            'per_product_aspect': pairs, 'by_aspect': summaries,
            'overall': {'product_count': len(products), 'product_aspect_count': len(pairs),
                        'ground_truth_product_aspect_count': sum(row['ground_truth'] is not None for row in pairs),
                        'paired_product_aspect_count': len(eligible), 'evaluable_aspect_count': len(evaluable_aspects),
                        'pooled_product_aspect_mae': {
                            'filtered': mean([row['filtered_absolute_error'] for row in eligible]),
                            'unfiltered': mean([row['unfiltered_absolute_error'] for row in eligible]),
                            'improvement': mean([row['improvement'] for row in eligible])},
                        'macro_mae_across_aspects': {
                            'filtered': mean([row['filtered_mae'] for row in evaluable_aspects]),
                            'unfiltered': mean([row['unfiltered_mae'] for row in evaluable_aspects]),
                            'improvement': mean([row['improvement'] for row in evaluable_aspects])}},
            'statistical_tests': 'Not performed. Product/aspect rows within a product are not independent product samples.'}


def multilabel_metrics(targets, predictions, labels):
    """One-vs-rest counts; accuracy means exact agreement of the whole label set."""
    labels = list(labels)
    truth = np.asarray(targets, dtype=int).reshape(-1, len(labels))
    predicted = np.asarray(predictions, dtype=int).reshape(-1, len(labels))
    if truth.shape != predicted.shape or not np.isin(truth, [0, 1]).all() or not np.isin(predicted, [0, 1]).all():
        raise ValueError("Multilabel targets/predictions must be matching binary matrices")
    if not len(truth):
        return {"count": 0, "accuracy": None, "macro_precision": None, "macro_recall": None,
                "macro_f1": None, "micro_f1": None, "support": [0] * len(labels),
                "labels": labels, "per_label": {}, "accuracy_definition": "exact_label_set_match"}
    precision, recall, f1, support = precision_recall_fscore_support(truth, predicted, average=None, zero_division=0)
    matrices = multilabel_confusion_matrix(truth, predicted)
    per_label = {}
    for index, label in enumerate(labels):
        tn, fp, fn, tp = map(int, matrices[index].ravel())
        per_label[label] = {"precision": float(precision[index]), "recall": float(recall[index]),
                            "f1": float(f1[index]), "support": int(support[index]),
                            "tp": tp, "fp": fp, "fn": fn, "tn": tn}
    return {"count": len(truth), "accuracy": float(accuracy_score(truth, predicted)),
            "accuracy_definition": "exact_label_set_match", "labels": labels,
            "macro_precision": float(precision.mean()), "macro_recall": float(recall.mean()),
            "macro_f1": float(f1.mean()),
            "micro_f1": float(precision_recall_fscore_support(truth, predicted, average="micro", zero_division=0)[2]),
            "support": support.astype(int).tolist(), "per_label": per_label,
            "multilabel_confusion_matrices": matrices.tolist()}


def sentiment_predictions(probabilities, threshold):
    values = np.asarray(probabilities, dtype=float).reshape(-1, len(POLARITIES))
    result = np.zeros_like(values, dtype=int)
    for index, row in enumerate(values):
        result[index, sentiment_indices(row, threshold)] = 1
    return result


def select_sentiment_threshold(targets, probabilities):
    if not len(targets):
        raise ValueError("Validation needs annotated Authentic review/category polarity instances")
    candidates = [round(value, 2) for value in np.arange(0.1, 0.91, 0.05)]
    return max(candidates, key=lambda value: (
        multilabel_metrics(targets, sentiment_predictions(probabilities, value), POLARITIES)["macro_f1"],
        -abs(value - 0.5),
    ))


def _pair_set(items):
    pairs = set()
    for item in items:
        category, sentiment = item["category"], str(item["sentiment"]).lower()
        if category not in ASPECTS or sentiment not in [label.lower() for label in POLARITIES]:
            raise ValueError("Invalid aspect/sentiment pair in ABSA evaluation")
        pairs.add((category, sentiment))
    return pairs


def _score_absa_records(records):
    detection_gold, detection_pred, sentiment_gold, sentiment_pred, joint_gold, joint_pred = [], [], [], [], [], []
    polarities = [label.lower() for label in POLARITIES]
    for record in records:
        gold, predicted = _pair_set(record["gold_pairs"]), _pair_set(record["predicted_pairs"])
        conditioned = _pair_set(record["gold_category_predictions"])
        categories = {pair[0] for pair in gold}
        detected = set(record["predicted_categories"])
        if not detected <= set(ASPECTS) or {pair[0] for pair in predicted} != detected:
            raise ValueError("Predicted categories and aspect/sentiment pairs disagree")
        if {pair[0] for pair in conditioned} != categories:
            raise ValueError("Gold-category sentiment predictions must cover exactly the gold aspects")
        detection_gold.append([int(category in categories) for category in ASPECTS])
        detection_pred.append([int(category in detected) for category in ASPECTS])
        for category in ASPECTS:
            if category in categories:
                sentiment_gold.append([int((category, polarity) in gold) for polarity in polarities])
                sentiment_pred.append([int((category, polarity) in conditioned) for polarity in polarities])
        joint_gold.append([int((category, polarity) in gold) for category in ASPECTS for polarity in polarities])
        joint_pred.append([int((category, polarity) in predicted) for category in ASPECTS for polarity in polarities])
    return {"category_detection": multilabel_metrics(detection_gold, detection_pred, ASPECTS),
            "gold_category_sentiment": multilabel_metrics(sentiment_gold, sentiment_pred, POLARITIES),
            "aspect_sentiment_pairs": multilabel_metrics(joint_gold, joint_pred,
                [f"{category}/{polarity}" for category in ASPECTS for polarity in polarities])}


def absa_evaluation_report(records, *, aspect_threshold, sentiment_threshold):
    """Identical standalone ABSA scoring for both models, without a quality gate."""
    if len({row["review_id"] for row in records}) != len(records):
        raise ValueError("ABSA evaluation review IDs must be unique")
    ordered = sorted(records, key=lambda row: row["review_id"])
    cohort = [{key: row[key] for key in ("review_id", "product_id", "review_text", "gold_pairs")}
              for row in ordered]
    for row, source in zip(cohort, ordered):
        row["gold_pairs"] = sorted(_pair_set(row["gold_pairs"]))
        row["image_urls"] = source.get("image_urls", [])
    signature = hashlib.sha256(json.dumps(cohort, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    products = sorted({row["product_id"] for row in ordered})
    return {"protocol": ABSA_EVALUATION_PROTOCOL, "sentiment_target_policy": SENTIMENT_TARGET_POLICY,
            "aspects": list(ASPECTS), "sentiments": [value.lower() for value in POLARITIES],
            "scope": "standalone ABSA on ground-truth Authentic reviews; no predicted quality filter",
            "cohort_sha256": signature, "review_count": len(records), "product_count": len(products),
            "aspect_threshold": float(aspect_threshold), "sentiment_threshold": float(sentiment_threshold),
            "metrics": _score_absa_records(ordered),
            "per_product_metrics": [{"product_id": product, **_score_absa_records(
                [row for row in ordered if row["product_id"] == product])} for product in products],
            "records": ordered}
