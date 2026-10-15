"""Pair frozen AuthentiCheck/baseline ABSA reports on identical ten-aspect cohorts."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_contract import ABSA_EVALUATION_PROTOCOL, ASPECTS, POLARITIES, SENTIMENT_TARGET_POLICY
from model_metrics import absa_evaluation_report


def compare_reports(own_report, baseline_report):
    evaluations = []
    for name, report in (("AuthentiCheck", own_report), ("CLIP-CA-CG adaptation", baseline_report)):
        evaluation = report.get("absa_evaluation", {})
        if (report.get("partition") != "test" or evaluation.get("protocol") != ABSA_EVALUATION_PROTOCOL
                or evaluation.get("sentiment_target_policy") != SENTIMENT_TARGET_POLICY
                or evaluation.get("aspects") != list(ASPECTS)
                or evaluation.get("sentiments") != [value.lower() for value in POLARITIES]):
            raise ValueError(f"{name}: need a new held-out ten-aspect multi-label evaluation report")
        fresh = absa_evaluation_report(evaluation["records"], aspect_threshold=evaluation["aspect_threshold"],
                                       sentiment_threshold=evaluation["sentiment_threshold"])
        if fresh["cohort_sha256"] != evaluation.get("cohort_sha256") or fresh["metrics"] != evaluation.get("metrics"):
            raise ValueError(f"{name}: stored metrics/cohort do not match the raw evaluation records")
        evaluations.append((evaluation, fresh))
    (own, own_scored), (baseline, baseline_scored) = evaluations
    if own["cohort_sha256"] != baseline["cohort_sha256"]:
        raise ValueError("ABSA test review IDs, products, review text, image URLs or gold targets differ")
    if not own_scored["review_count"]:
        raise ValueError("Cannot compare empty ABSA test cohorts")
    for partition in ("train", "validation"):
        if (not own.get(f"{partition}_review_ids") or not own.get(f"{partition}_sha256")
                or own[f"{partition}_review_ids"] != baseline.get(f"{partition}_review_ids")
                or own[f"{partition}_sha256"] != baseline.get(f"{partition}_sha256")):
            raise ValueError(f"ABSA {partition} review IDs/content differ or provenance is missing")
    paired = []
    baseline_products = {row["product_id"]: row for row in baseline_scored["per_product_metrics"]}
    metric_names = ("category_detection", "gold_category_sentiment", "aspect_sentiment_pairs")
    for product in own_scored["per_product_metrics"]:
        other = baseline_products[product["product_id"]]
        row = {"product_id": product["product_id"]}
        for metric in metric_names:
            a, b = product[metric]["macro_f1"], other[metric]["macro_f1"]
            eligible = a is not None and b is not None
            row[metric] = {"authenticheck": product[metric], "baseline": other[metric],
                           "paired_eligible": eligible, "macro_f1_difference": a - b if eligible else None}
        paired.append(row)
    return {"protocol": ABSA_EVALUATION_PROTOCOL, "partition": "test", "aspects": list(ASPECTS),
            "cohort_sha256": own["cohort_sha256"], "review_count": own_scored["review_count"],
            "product_count": own_scored["product_count"],
            "metrics": {"authenticheck": own_scored["metrics"], "baseline": baseline_scored["metrics"]},
            "per_product_pairs": paired,
            "records": {"authenticheck": own_scored["records"], "baseline": baseline_scored["records"]},
            "statistical_tests": "Not performed; use paired product results under the finalized thesis statistical protocol"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--own", required=True, help="AuthentiCheck stage2/evaluate_models.py JSON report")
    parser.add_argument("--baseline", required=True, help="Baseline evaluate.py JSON report")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    path = Path(args.output)
    if path.exists():
        raise FileExistsError(f"Comparison already exists: {path}; choose another filename")
    result = compare_reports(json.loads(Path(args.own).read_text(encoding="utf-8")),
                             json.loads(Path(args.baseline).read_text(encoding="utf-8")))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(f"Compared {result['review_count']} Authentic test reviews across {result['product_count']} products: {path}")


if __name__ == "__main__":
    main()
