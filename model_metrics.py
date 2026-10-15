"""Quality metrics and shared ten-aspect multi-label ABSA evaluation."""

import hashlib
import json

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, multilabel_confusion_matrix

from model_contract import ASPECTS, POLARITIES, ABSA_EVALUATION_PROTOCOL, SENTIMENT_TARGET_POLICY, sentiment_indices


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
