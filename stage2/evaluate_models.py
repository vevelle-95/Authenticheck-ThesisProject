"""Final held-out evaluation of our models and our filtering ablation only."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np

from model_contract import ASPECTS, CLASS_NAMES, DEFAULT_BUNDLE, DEFAULT_SPLITS_PATH, POLARITIES, SENSORY_POLICY, SENTIMENT_TARGET_POLICY, quality_input_text
from model_data import fingerprint, load_experiment
from training_augmentation import augmentation_summary
from model_metrics import classification_metrics
from stage2.fine_tune_absa import detection_metrics
from stage2.online_inference import OnlineInference, aggregate_products


def check_training_provenance(assigned, bundle):
    bundle = Path(bundle)
    training = assigned[assigned.partition.eq("train")]
    validation = assigned[assigned.partition.eq("validation")]
    for path, authentic_only in (
        (bundle / "dost_roberta" / "training_metadata.json", False),
        (bundle / "xgboost_meta_classifier.metadata.json", False),
        (bundle / "absa_model" / "training_metadata.json", True),
    ):
        metadata = json.loads(path.read_text(encoding="utf-8"))
        fit = training[training.label.eq(0)] if authentic_only else training
        val = validation[validation.label.eq(0)] if authentic_only else validation
        if set(metadata.get("fit_review_ids", [])) != set(fit.review_id) or set(metadata.get("validation_review_ids", [])) != set(val.review_id):
            raise ValueError(f"{path}: training/validation membership does not match the split manifest")
        if "dataset_sha256" in metadata and metadata["dataset_sha256"] != fingerprint(assigned):
            raise ValueError(f"{path}: dataset fingerprint mismatch")
        if "training_sha256" in metadata and metadata["training_sha256"] != fingerprint(fit):
            raise ValueError(f"{path}: training contents or augmentation lineage changed")


def evaluate_own_models(test, pipe):
    pipe.check_artifacts()
    absa, tokenizer = pipe._load_absa()
    aspect_predictions = absa.predict(test.review_text.tolist(), tokenizer)
    filtered, unfiltered, gold, quality_predictions, text_predictions = [], [], [], [], []
    polarity_gold, polarity_predicted = [], []
    per_review_polarity = {}
    det_gold, det_predicted, det_eligible = [], [], []
    for record, aspects in zip(test.to_dict("records"), aspect_predictions):
        features, _, _ = pipe.extractor.extract(quality_input_text(record), record["image_urls"], record["star_rating"])
        verdict, probabilities = pipe.classify(features)
        text_verdict = int(np.argmax(features[:4]))
        quality_predictions.append(verdict)
        text_predictions.append(text_verdict)
        common = {
            "id": record["review_id"], "product_id": record["product_id"], "text": record["review_text"],
            "label": CLASS_NAMES[verdict], "ground_truth": CLASS_NAMES[record["label"]],
            "probabilities": dict(zip(CLASS_NAMES, probabilities)), "features": features,
        }
        unfiltered.append({**common, "aspectSentiment": aspects})
        filtered.append({**common, "aspectSentiment": aspects if verdict == 0 else []})
        gold.append({
            "id": record["review_id"], "product_id": record["product_id"], "label": CLASS_NAMES[record["label"]],
            "aspectSentiment": [
                {"category": annotation["category"], "sentiment": POLARITIES[annotation["sentiment"]],
                 "text": " ".join(dict.fromkeys(annotation["evidence"]))}
                for annotation in record["annotations"]
            ] if record["label"] == 0 else [],
        })
        if record["label"] == 0:
            categories = [annotation["category"] for annotation in record["annotations"]]
            probabilities = absa.predict_sentiment([record["review_text"]] * len(categories), categories, tokenizer)
            targets = [annotation["sentiment"] for annotation in record["annotations"]]
            predictions = [int(np.argmax(values)) for values in probabilities]
            polarity_gold.extend(targets)
            polarity_predicted.extend(predictions)
            per_review_polarity[record["review_id"]] = (targets, predictions)
            det_gold.append([int(category in categories) for category in ASPECTS])
            detected = {aspect["category"] for aspect in aspects}
            det_predicted.append([int(category in detected) for category in ASPECTS])
            det_eligible.append([1] * len(ASPECTS))
    by_id = {row["id"]: index for index, row in enumerate(filtered)}
    per_product = []
    for product, rows in test.groupby("product_id"):
        indices = [by_id[review_id] for review_id in rows.review_id]
        targets = rows.label.tolist()
        pairs = [per_review_polarity[review_id] for review_id in rows.review_id if review_id in per_review_polarity]
        per_product.append({
            "product_id": product,
            "quality_xgboost": classification_metrics(targets, [quality_predictions[index] for index in indices], range(4)),
            "quality_text_only": classification_metrics(targets, [text_predictions[index] for index in indices], range(4)),
            "gold_category_sentiment": classification_metrics(
                [value for pair in pairs for value in pair[0]], [value for pair in pairs for value in pair[1]], range(3),
            ),
        })
    filtered_scores = {row["product_id"]: row["aggregateSentiment"] for row in aggregate_products(filtered)}
    unfiltered_scores = {row["product_id"]: row["aggregateSentiment"] for row in aggregate_products(unfiltered, authentic_only=False)}
    ground_truth_scores = {row["product_id"]: row["aggregateSentiment"] for row in aggregate_products(gold)}
    errors = []
    for product, truth in ground_truth_scores.items():
        with_filter, without_filter = filtered_scores[product], unfiltered_scores[product]
        defined = truth is not None and with_filter is not None and without_filter is not None
        errors.append({
            "product_id": product, "ground_truth": truth, "filtered": with_filter, "unfiltered": without_filter,
            "filtered_absolute_error": abs(with_filter - truth) if truth is not None and with_filter is not None else None,
            "unfiltered_absolute_error": abs(without_filter - truth) if truth is not None and without_filter is not None else None,
            "paired_eligible": defined,
            "improvement": abs(without_filter - truth) - abs(with_filter - truth) if defined else None,
        })
    return {
        "partition": "test", "review_count": len(test), "product_count": test.product_id.nunique(),
        "quality_xgboost": classification_metrics(test.label.tolist(), quality_predictions, range(4)),
        "quality_text_only": classification_metrics(test.label.tolist(), text_predictions, range(4)),
        "gold_category_sentiment": classification_metrics(polarity_gold, polarity_predicted, range(3)),
        "category_detection": detection_metrics(np.asarray(det_gold), np.asarray(det_predicted), np.asarray(det_eligible), 0.5) if det_gold else None,
        "per_product_metrics": per_product, "filtering_ablation": errors,
        "excluded_ablation_products": sum(not row["paired_eligible"] for row in errors),
        "filtered_reviews": filtered, "unfiltered_reviews": unfiltered,
        "aggregation_unit": "one_review_category_polarity_group",
        "sensory_domain_policy": SENSORY_POLICY,
        "sentiment_target_policy": SENTIMENT_TARGET_POLICY,
        "statistical_tests": "Deferred until the thesis statistical protocol is finalized; no external comparison model is used.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--splits", default=str(DEFAULT_SPLITS_PATH))
    parser.add_argument("--augmentations", help="Approved training-only augmentation CSV; use the same file throughout the experiment")
    parser.add_argument("--allow-unreviewed-augmentations", action="store_true", help="Experimental run: include pending drafts without marking them approved")
    parser.add_argument("--bundle", default=str(DEFAULT_BUNDLE))
    parser.add_argument("--output", default="results/own_model_test.json")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("A final test report already exists at this path; use a new experiment output")
    assigned = load_experiment(args.data, args.splits, args.augmentations, require_annotations=True,
                               allow_unreviewed=args.allow_unreviewed_augmentations)
    check_training_provenance(assigned, args.bundle)
    bundle = Path(args.bundle)
    pipe = OnlineInference(bundle / "dost_roberta", bundle / "xgboost_meta_classifier.json", bundle / "absa_model")
    result = evaluate_own_models(assigned[assigned.partition.eq("test")], pipe)
    result["augmentation_usage"] = augmentation_summary(assigned)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(f"Saved held-out results to {output}")


if __name__ == "__main__":
    main()
