"""Compare AuthentiCheck filtering with CLIP-CA-CG ABSA on all mixed test reviews.

This implements the product-level aggregation comparison for SOP 3.  Human
ground truth is aggregated only from ground-truth Authentic reviews.  The saved
AuthentiCheck report contributes ABSA outputs only when its Stage 1 classifier
predicted Authentic.  The single-stage CLIP-CA-CG baseline predicts ABSA for
every review in the same mixed held-out test partition.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from datasets.authenticheck_data import ASPECTS, SENTIMENTS, fingerprint, load_experiment
from datasets.multimodal_dataset import MultiModalDataset, multimodal_collate_fn
from inference import BaselinePredictor, format_aspects
from runtime import default_checkpoint, load_config, resolve_path, write_json
from training.eval import collect_predictions


SENTIMENT_VALUES = {"negative": -1, "neutral": 0, "positive": 1}


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_own_report(report, test, dataset_sha256):
    if report.get("partition") != "test":
        raise ValueError("AuthentiCheck report must be a held-out test report")
    if report.get("dataset_sha256") != dataset_sha256:
        raise ValueError("AuthentiCheck report was evaluated on a different dataset")
    rows = report.get("filtered_reviews", [])
    expected = set(test.review_id)
    actual = {row.get("id") for row in rows}
    if len(rows) != len(actual) or actual != expected:
        raise ValueError("AuthentiCheck report does not cover exactly the shared test reviews")
    products = dict(zip(test.review_id, test.product_id))
    if any(products[row["id"]] != row.get("product_id") for row in rows):
        raise ValueError("AuthentiCheck report product IDs do not match the shared test set")
    return {row["id"]: row for row in rows}


def aggregate(rows, *, authentic_only=False):
    """Mean distinct predicted polarities per review/aspect, then per product."""
    values = defaultdict(list)
    for row in rows:
        if authentic_only and row["label"] != "authentic":
            continue
        by_aspect = defaultdict(set)
        for item in row.get("aspectSentiment", []):
            category = item.get("category")
            sentiment = str(item.get("sentiment", "")).lower()
            if category not in ASPECTS or sentiment not in SENTIMENT_VALUES:
                raise ValueError("Invalid aspect or sentiment in aggregation input")
            by_aspect[category].add(SENTIMENT_VALUES[sentiment])
        for category, polarities in by_aspect.items():
            values[(row["product_id"], category)].append(float(np.mean(sorted(polarities))))
    return {key: float(np.mean(scores)) for key, scores in values.items()}


def mean(values):
    return float(np.mean(values)) if values else None


def compare(gold, left_system, right_system, products, *, left_name, right_name):
    truth = aggregate(gold, authentic_only=True)
    left = aggregate(left_system)
    right = aggregate(right_system)
    rows = []
    for product in sorted(products):
        for category in ASPECTS:
            ground_truth = truth.get((product, category))
            left_score = left.get((product, category))
            right_score = right.get((product, category))
            reasons = [name for name, value in (
                ("no_ground_truth", ground_truth),
                (f"no_{left_name}_prediction", left_score),
                (f"no_{right_name}_prediction", right_score),
            ) if value is None]
            paired = not reasons
            left_error = abs(left_score - ground_truth) if left_score is not None and ground_truth is not None else None
            right_error = abs(right_score - ground_truth) if right_score is not None and ground_truth is not None else None
            rows.append({
                "product_id": product, "category": category,
                "ground_truth": ground_truth,
                left_name: left_score, right_name: right_score,
                f"{left_name}_absolute_error": left_error,
                f"{right_name}_absolute_error": right_error,
                "paired_eligible": paired, "exclusion_reasons": reasons,
                f"{left_name}_improvement": right_error - left_error if paired else None,
            })
    by_aspect = []
    for category in ASPECTS:
        category_rows = [row for row in rows if row["category"] == category]
        paired = [row for row in category_rows if row["paired_eligible"]]
        referenced = [row for row in category_rows if row["ground_truth"] is not None]
        by_aspect.append({
            "category": category,
            "product_count": len(category_rows),
            "ground_truth_product_count": len(referenced),
            "paired_product_count": len(paired),
            f"{left_name}_mae": mean([row[f"{left_name}_absolute_error"] for row in paired]),
            f"{right_name}_mae": mean([row[f"{right_name}_absolute_error"] for row in paired]),
            f"{left_name}_improvement": mean([row[f"{left_name}_improvement"] for row in paired]),
        })
    paired = [row for row in rows if row["paired_eligible"]]
    evaluable = [row for row in by_aspect if row["paired_product_count"]]
    return {
        "aggregation_unit": "mean_distinct_polarities_per_review_then_mean_reviews_per_product_aspect",
        "score_range": [-1, 1],
        "exclusion_policy": "Paired MAEs include only product/aspect rows with defined human, AuthentiCheck, and baseline aggregates; undefined scores are null, never zero.",
        "per_product_aspect": rows,
        "by_aspect": by_aspect,
        "overall": {
            "product_count": len(products),
            "product_aspect_count": len(rows),
            "paired_product_aspect_count": len(paired),
            f"{left_name}_mae": mean([row[f"{left_name}_absolute_error"] for row in paired]),
            f"{right_name}_mae": mean([row[f"{right_name}_absolute_error"] for row in paired]),
            f"{left_name}_improvement": mean([row[f"{left_name}_improvement"] for row in paired]),
            "macro_mae_across_evaluable_aspects": {
                left_name: mean([row[f"{left_name}_mae"] for row in evaluable]),
                right_name: mean([row[f"{right_name}_mae"] for row in evaluable]),
                f"{left_name}_improvement": mean([row[f"{left_name}_improvement"] for row in evaluable]),
            },
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--own-report", default="../../results/real_only_v1_test.json")
    parser.add_argument("--config")
    parser.add_argument("--csv")
    parser.add_argument("--splits")
    parser.add_argument("--checkpoint", help="Defaults to best.pt in the configured checkpoint directory")
    parser.add_argument("--output", default="outputs/product_aggregation_comparison.json")
    args = parser.parse_args()
    output_path = resolve_path(args.output)
    if output_path.exists():
        raise FileExistsError(f"Report already exists: {output_path}; choose another filename")

    config = load_config(args.config)
    assigned = load_experiment(
        resolve_path(args.csv or config["data"]["csv"]),
        resolve_path(args.splits or config["data"]["splits"]),
    )
    dataset_sha256 = fingerprint(assigned)
    test = assigned[assigned.partition.eq("test")].copy()
    own_by_id = validate_own_report(load_json(resolve_path(args.own_report)), test, dataset_sha256)
    checkpoint_path = resolve_path(args.checkpoint) if args.checkpoint else default_checkpoint(config)
    predictor = BaselinePredictor(checkpoint_path)
    if predictor.checkpoint["training_metadata"]["dataset_sha256"] != dataset_sha256:
        raise ValueError("CLIP-CA-CG checkpoint was trained on a different dataset")

    loader = DataLoader(
        MultiModalDataset(test, predictor.tokenizer, predictor.cache, predictor.config),
        batch_size=predictor.config["training"]["batch_size"], shuffle=False,
        num_workers=predictor.config["training"]["num_workers"],
        collate_fn=multimodal_collate_fn,
    )
    raw = collect_predictions(predictor.model, loader, predictor.device)
    threshold = predictor.checkpoint["threshold"]
    sentiment_threshold = predictor.checkpoint["sentiment_threshold"]
    baseline = []
    for index, row in enumerate(test.to_dict("records")):
        baseline.append({
            "id": row["review_id"], "product_id": row["product_id"], "label": "absa_only",
            "aspectSentiment": format_aspects(
                row["review_text"], raw["aspect_probabilities"][index], raw["sentiment_probabilities"][index],
                threshold, sentiment_threshold,
            ),
        })
    gold = [{
        "id": row["review_id"], "product_id": row["product_id"],
        "label": "authentic" if row["label"] == 0 else "not_authentic",
        "aspectSentiment": [
            {"category": category, "sentiment": SENTIMENTS[polarity]}
            for category, polarities in row["annotations"].items() for polarity in polarities
        ] if row["label"] == 0 else [],
    } for row in test.to_dict("records")]
    own = [{
        "id": row["id"], "product_id": row["product_id"], "label": row["label"],
        "aspectSentiment": row.get("aspectSentiment", []),
    } for row in own_by_id.values()]
    own_absa_only = [{
        "id": row["id"], "product_id": row["product_id"], "label": "absa_only",
        "aspectSentiment": row.get("aspectSentiment", []),
    } for row in load_json(resolve_path(args.own_report))["unfiltered_reviews"]]
    filtering_comparison = compare(
        gold, own, baseline, test.product_id.unique(),
        left_name="authenticheck_filtered", right_name="clip_ca_cg_all_reviews",
    )
    absa_only_comparison = compare(
        gold, own_absa_only, baseline, test.product_id.unique(),
        left_name="authenticheck_absa_only", right_name="clip_ca_cg_absa_only",
    )
    usable = [len(predictor.cache.get(row)["usable_urls"]) for row in test.to_dict("records")]
    report = {
        "protocol": "product-aspect-filtered-authenticheck-vs-absa-only-clip-ca-cg-v1",
        "scope": "mixed held-out test reviews; ground truth aggregates use only human-authentic reviews",
        "interpretation": (
            "AuthentiCheck predictions are filtered by its saved Stage 1 XGBoost label. "
            "CLIP-CA-CG is an ABSA-only baseline and therefore predicts every mixed test review."
        ),
        "dataset_sha256": dataset_sha256,
        "review_count": len(test), "product_count": int(test.product_id.nunique()),
        "ground_truth_authentic_review_count": int(test.label.eq(0).sum()),
        "authenticheck_predicted_authentic_review_count": sum(row["label"] == "authentic" for row in own),
        "clip_ca_cg_input_review_count": len(baseline),
        "clip_ca_cg_image_availability": {
            "reviews_with_usable_buyer_images": sum(count > 0 for count in usable),
            "reviews_without_usable_buyer_images": sum(count == 0 for count in usable),
        },
        "filtering_vs_clip_ca_cg": filtering_comparison,
        "absa_only_vs_clip_ca_cg": absa_only_comparison,
    }
    write_json(output_path, report, overwrite=False)
    overall = filtering_comparison["overall"]
    print("Product aggregation comparison saved:", output_path)
    print("Paired product/aspect rows:", overall["paired_product_aspect_count"])
    print("AuthentiCheck filtered MAE:", overall["authenticheck_filtered_mae"])
    print("CLIP-CA-CG ABSA-only MAE:", overall["clip_ca_cg_all_reviews_mae"])


if __name__ == "__main__":
    main()
