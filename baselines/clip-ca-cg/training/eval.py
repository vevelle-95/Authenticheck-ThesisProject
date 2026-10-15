"""Ten-aspect detection and multi-label polarity metrics shared with AuthentiCheck."""

import numpy as np
import torch

from datasets.authenticheck_data import ASPECTS, SENTIMENTS
from model_metrics import multilabel_metrics, sentiment_predictions, select_sentiment_threshold, absa_evaluation_report
from training.engine import move_inputs


@torch.no_grad()
def collect_predictions(model, dataloader, device):
    model.eval()
    detection, sentiments, gold_aspects, gold_sentiments = [], [], [], []
    for batch in dataloader:
        outputs = model(**move_inputs(batch, device))
        detection.append(torch.sigmoid(outputs["aspect_logits"]).float().cpu().numpy())
        sentiments.append(torch.sigmoid(outputs["sentiment_logits"]).float().cpu().numpy())
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


def compute_metrics(outputs, threshold, sentiment_threshold=0.5):
    gold = outputs["aspect_targets"]
    detection = multilabel_metrics(gold, outputs["aspect_probabilities"] >= threshold, ASPECTS)
    detection["per_category"] = detection["per_label"]
    active = gold.astype(bool)
    polarity = multilabel_metrics(outputs["sentiment_targets"][active],
        sentiment_predictions(outputs["sentiment_probabilities"][active], sentiment_threshold), SENTIMENTS)
    return {
        "aspects": list(ASPECTS), "sentiments": list(SENTIMENTS), "threshold": float(threshold),
        "sentiment_threshold": float(sentiment_threshold),
        "category_detection": detection,
        "gold_category_sentiment": polarity,
    }


def select_threshold(outputs):
    active = outputs["aspect_targets"].astype(bool)
    sentiment_threshold = select_sentiment_threshold(outputs["sentiment_targets"][active],
                                                     outputs["sentiment_probabilities"][active])
    candidates = [(compute_metrics(outputs, value, sentiment_threshold), float(value)) for value in np.linspace(0.1, 0.9, 17)]
    metrics, threshold = max(candidates, key=lambda pair: (
        pair[0]["category_detection"]["macro_f1"], -abs(pair[1] - 0.5),
    ))
    return threshold, metrics


def evaluation_report(frame, outputs, checkpoint):
    records = []
    masks = sentiment_predictions(outputs["sentiment_probabilities"], checkpoint["sentiment_threshold"])
    masks = masks.reshape(-1, len(ASPECTS), len(SENTIMENTS))
    for index, row in enumerate(frame.to_dict("records")):
        gold = outputs["sentiment_targets"][index].astype(bool)
        detected = outputs["aspect_probabilities"][index] >= checkpoint["threshold"]
        def pairs(mask):
            return [{"category": category, "sentiment": sentiment}
                    for aspect, category in enumerate(ASPECTS)
                    for polarity, sentiment in enumerate(SENTIMENTS) if mask[aspect, polarity]]
        records.append({"review_id": row["review_id"], "product_id": row["product_id"],
                        "image_urls": row.get("image_urls", []),
                        "review_text": row["review_text"], "gold_pairs": pairs(gold),
                        "predicted_categories": [category for aspect, category in enumerate(ASPECTS) if detected[aspect]],
                        "predicted_pairs": pairs(masks[index] & detected[:, None]),
                        "gold_category_predictions": pairs(masks[index] & gold.any(-1)[:, None])})
    return absa_evaluation_report(records, aspect_threshold=checkpoint["threshold"],
                                 sentiment_threshold=checkpoint["sentiment_threshold"])
