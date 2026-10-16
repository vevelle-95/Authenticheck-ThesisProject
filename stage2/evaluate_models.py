"""Final held-out evaluation of our models and our filtering ablation only."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np

from model_contract import ASPECTS, CLASS_NAMES, DEFAULT_BUNDLE, DEFAULT_SPLITS_PATH, INPUT_VERSION, POLARITIES, ROBERTA_TARGET_COLUMN, SENSORY_POLICY, SENTIMENT_TARGET_POLICY, quality_input_text, sentiment_indices
from model_data import fingerprint, load_experiment
from training_augmentation import augmentation_summary
from model_metrics import classification_metrics, absa_evaluation_report, aspect_filtering_ablation
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
        if path.parent.name == "dost_roberta" and (
                metadata.get("input_version") != INPUT_VERSION
                or metadata.get("target_column") != ROBERTA_TARGET_COLUMN):
            raise ValueError(f"{path}: RoBERTa input contract or text_label target does not match")
        fit = training[training.label.eq(0)] if authentic_only else training
        val = validation[validation.label.eq(0)] if authentic_only else validation
        if set(metadata.get("fit_review_ids", [])) != set(fit.review_id) or set(metadata.get("validation_review_ids", [])) != set(val.review_id):
            raise ValueError(f"{path}: training/validation membership does not match the split manifest")
        if "dataset_sha256" in metadata and metadata["dataset_sha256"] != fingerprint(assigned):
            raise ValueError(f"{path}: dataset fingerprint mismatch")
        if "training_sha256" in metadata and metadata["training_sha256"] != fingerprint(fit):
            raise ValueError(f"{path}: training contents or augmentation lineage changed")


def add_aspect_ablation_from_report(report):
    """Re-score frozen predictions; never reload models, labels, images or splits."""
    if report.get('partition') != 'test':
        raise ValueError('Expected a held-out test report')
    filtered, unfiltered = report['filtered_reviews'], report['unfiltered_reviews']
    if len(unfiltered) != report['review_count']:
        raise ValueError('Saved review count does not match predictions')
    authentic = {row['id']: row for row in unfiltered if row['ground_truth'] == 'authentic'}
    records = report['absa_evaluation']['records']
    if len({row['review_id'] for row in records}) != len(records) or {row['review_id'] for row in records} != set(authentic):
        raise ValueError('Saved gold annotations must cover ground-truth Authentic reviews exactly once')
    annotations = {row['review_id']: row for row in records}
    for review_id, row in authentic.items():
        if annotations[review_id]['product_id'] != row['product_id']:
            raise ValueError('Saved gold product IDs disagree with predictions')
    gold = [{'id': row['id'], 'product_id': row['product_id'], 'label': row['ground_truth'],
             'aspectSentiment': annotations[row['id']]['gold_pairs'] if row['ground_truth'] == 'authentic' else []}
            for row in unfiltered]
    return {**report, 'aspect_filtering_ablation': aspect_filtering_ablation(gold, filtered, unfiltered)}


def evaluate_own_models(test, pipe):
    pipe.check_artifacts()
    absa, tokenizer = pipe._load_absa()
    aspect_predictions = absa.predict(test.review_text.tolist(), tokenizer)
    filtered, unfiltered, gold, quality_predictions, text_predictions = [], [], [], [], []
    absa_records = []
    for record, aspects in zip(test.to_dict("records"), aspect_predictions):
        features, _, _ = pipe.extractor.extract(quality_input_text(record), record["image_urls"], record["star_rating"])
        verdict, probabilities = pipe.classify(features)
        text_verdict = int(np.argmax(features[:4]))
        quality_predictions.append(verdict)
        text_predictions.append(text_verdict)
        common = {
            "id": record["review_id"], "product_id": record["product_id"], "text": record["review_text"],
            "label": CLASS_NAMES[verdict], "ground_truth": CLASS_NAMES[record["label"]],
            "text_label": CLASS_NAMES[record["text_target"]], "roberta_prediction": CLASS_NAMES[text_verdict],
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
            categories = [category for category in ASPECTS if any(
                annotation["category"] == category for annotation in record["annotations"])]
            probabilities = absa.predict_sentiment([record["review_text"]] * len(categories), categories, tokenizer)
            absa_records.append({
                "review_id": record["review_id"], "product_id": record["product_id"],
                "review_text": record["review_text"],
                "image_urls": record["image_urls"],
                "gold_pairs": [{"category": annotation["category"], "sentiment": POLARITIES[annotation["sentiment"]].lower()}
                               for annotation in record["annotations"]],
                "predicted_categories": [category for category in ASPECTS if any(item["category"] == category for item in aspects)],
                "predicted_pairs": [{"category": item["category"], "sentiment": item["sentiment"].lower()} for item in aspects],
                "gold_category_predictions": [{"category": category, "sentiment": POLARITIES[polarity].lower()}
                    for category, values in zip(categories, probabilities)
                    for polarity in sentiment_indices(values, absa.sentiment_threshold)],
            })
    absa_report = absa_evaluation_report(absa_records, aspect_threshold=absa.threshold,
                                          sentiment_threshold=absa.sentiment_threshold)
    absa_products = {row["product_id"]: row for row in absa_report["per_product_metrics"]}
    by_id = {row["id"]: index for index, row in enumerate(filtered)}
    per_product = []
    for product, rows in test.groupby("product_id"):
        indices = [by_id[review_id] for review_id in rows.review_id]
        targets = rows.label.tolist()
        per_product.append({
            "product_id": product,
            "quality_xgboost": classification_metrics(targets, [quality_predictions[index] for index in indices], range(4)),
            "quality_text_only": classification_metrics(targets, [text_predictions[index] for index in indices], range(4)),
            "quality_roberta_text_target": classification_metrics(rows.text_target.tolist(), [text_predictions[index] for index in indices], range(4)),
            "gold_category_sentiment": absa_products.get(product, {}).get("gold_category_sentiment"),
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
        "quality_roberta_text_target": classification_metrics(test.text_target.tolist(), text_predictions, range(4)),
        "quality_metric_targets": {
            "quality_xgboost": "ground_truth",
            "quality_text_only": "ground_truth (RoBERTa-only ablation of final classification)",
            "quality_roberta_text_target": ROBERTA_TARGET_COLUMN,
        },
        "gold_category_sentiment": absa_report["metrics"]["gold_category_sentiment"],
        "category_detection": absa_report["metrics"]["category_detection"],
        "aspect_sentiment_pairs": absa_report["metrics"]["aspect_sentiment_pairs"],
        "absa_evaluation": absa_report,
        "per_product_metrics": per_product, "filtering_ablation": errors,
        "aspect_filtering_ablation": aspect_filtering_ablation(gold, filtered, unfiltered),
        "excluded_ablation_products": sum(not row["paired_eligible"] for row in errors),
        "filtered_reviews": filtered, "unfiltered_reviews": unfiltered,
        "aggregation_unit": "one_review_category_polarity_group",
        "sensory_domain_policy": SENSORY_POLICY,
        "sentiment_target_policy": SENTIMENT_TARGET_POLICY,
        "statistical_tests": "Deferred until the thesis statistical protocol is finalized; no external comparison model is used.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data")
    source.add_argument("--from-report", help="Add per-aspect errors to saved test predictions without model inference")
    parser.add_argument("--splits", default=str(DEFAULT_SPLITS_PATH))
    parser.add_argument("--augmentations", help="Approved training-only augmentation CSV; use the same file throughout the experiment")
    parser.add_argument("--allow-unreviewed-augmentations", action="store_true", help="Experimental run: include pending drafts without marking them approved")
    parser.add_argument("--bundle", default=str(DEFAULT_BUNDLE))
    parser.add_argument("--output", default="results/own_model_test.json")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("A final test report already exists at this path; use a new experiment output")
    if args.from_report:
        result = add_aspect_ablation_from_report(json.loads(Path(args.from_report).read_text(encoding='utf-8')))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
        print(f"Saved report with per-aspect errors to {output}; existing predictions unchanged")
        return
    assigned = load_experiment(args.data, args.splits, args.augmentations, require_annotations=True,
                               allow_unreviewed=args.allow_unreviewed_augmentations)
    check_training_provenance(assigned, args.bundle)
    bundle = Path(args.bundle)
    pipe = OnlineInference(bundle / "dost_roberta", bundle / "xgboost_meta_classifier.json", bundle / "absa_model")
    result = evaluate_own_models(assigned[assigned.partition.eq("test")], pipe)
    result["augmentation_usage"] = augmentation_summary(assigned)
    result["dataset_sha256"] = fingerprint(assigned)
    for partition in ("train", "validation"):
        cohort = assigned[assigned.partition.eq(partition) & assigned.label.eq(0)]
        result["absa_evaluation"][f"{partition}_review_ids"] = sorted(cohort.review_id.tolist())
        result["absa_evaluation"][f"{partition}_sha256"] = fingerprint(cohort)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(f"Saved held-out results to {output}")


if __name__ == "__main__":
    main()
