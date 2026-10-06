"""Shared six-aspect detection and gold-category polarity metrics."""

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix

from datasets.authenticheck_data import ASPECTS, SENTIMENTS
from training.engine import move_inputs


@torch.no_grad()
def collect_predictions(model, dataloader, device):
    model.eval()
    detection, sentiments, gold_aspects, gold_sentiments = [], [], [], []
    for batch in dataloader:
        outputs = model(**move_inputs(batch, device))
        detection.append(torch.sigmoid(outputs["aspect_logits"]).float().cpu().numpy())
        sentiments.append(torch.softmax(outputs["sentiment_logits"], dim=-1).float().cpu().numpy())
        if "aspect_targets" in batch:
            gold_aspects.append(batch["aspect_targets"].numpy())
            gold_sentiments.append(batch["sentiment_targets"].numpy())
    if not detection:
        raise ValueError("Evaluation/prediction partition is empty")
    return {
        "aspect_probabilities": np.concatenate(detection),
        "sentiment_probabilities": np.concatenate(sentiments),
        "aspect_targets": np.concatenate(gold_aspects) if gold_aspects else None,
        "sentiment_targets": np.concatenate(gold_sentiments) if gold_sentiments else None,
    }


def compute_metrics(outputs, threshold):
    gold = outputs["aspect_targets"]
    predicted = outputs["aspect_probabilities"] >= threshold
    precision, recall, f1, support = precision_recall_fscore_support(
        gold, predicted, average=None, zero_division=0,
    )
    truth, guesses = [], []
    for row, targets in enumerate(outputs["sentiment_targets"]):
        for aspect, distribution in enumerate(targets):
            for polarity in np.flatnonzero(distribution):
                truth.append(int(polarity))
                guesses.append(int(outputs["sentiment_probabilities"][row, aspect].argmax()))
    if truth:
        p, r, f, s = precision_recall_fscore_support(truth, guesses, labels=[0, 1, 2], zero_division=0)
        polarity = {
            "count": len(truth), "accuracy": float(accuracy_score(truth, guesses)),
            "macro_precision": float(p.mean()), "macro_recall": float(r.mean()),
            "macro_f1": float(f.mean()), "support": s.tolist(),
            "confusion_matrix": confusion_matrix(truth, guesses, labels=[0, 1, 2]).tolist(),
        }
    else:
        polarity = {"count": 0, "accuracy": None, "macro_f1": 0.0}
    return {
        "aspects": list(ASPECTS), "sentiments": list(SENTIMENTS), "threshold": float(threshold),
        "category_detection": {
            "macro_precision": float(precision.mean()), "macro_recall": float(recall.mean()),
            "macro_f1": float(f1.mean()),
            "per_category": {
                category: {"precision": float(precision[i]), "recall": float(recall[i]),
                           "f1": float(f1[i]), "support": int(support[i])}
                for i, category in enumerate(ASPECTS)
            },
        },
        "gold_category_sentiment": polarity,
    }


def select_threshold(outputs):
    candidates = [(compute_metrics(outputs, value), float(value)) for value in np.linspace(0.1, 0.9, 17)]
    metrics, threshold = max(candidates, key=lambda pair: (
        pair[0]["category_detection"]["macro_f1"], -abs(pair[1] - 0.5),
    ))
    return threshold, metrics
