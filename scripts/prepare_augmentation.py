"""Validate augmentation drafts or export training sources for the LLM prompt."""

import argparse
from collections import Counter
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd

from model_contract import ASPECTS, CLASS_NAMES, DEFAULT_DATA_PATH, DEFAULT_SPLITS_PATH, POLARITIES
from model_data import fingerprint, load_experiment
from training_augmentation import append_augmentations, image_scopes


def balancing_report(original, combined, draft):
    """Count distinct review/category/polarity targets, rather than raw spans."""
    real_train = original[original.partition.eq("train")]
    active_train = combined[combined.partition.eq("train")]

    def quality_counts(rows):
        counts = rows.label.value_counts()
        return {name: int(counts.get(label, 0)) for label, name in enumerate(CLASS_NAMES)}

    def aspect_counts(rows):
        counts = Counter()
        for annotations in rows.loc[rows.label.eq(0), "annotations"]:
            counts.update({(item["category"], item["sentiment"]) for item in annotations})
        return counts

    original_aspects, active_aspects = aspect_counts(real_train), aspect_counts(active_train)
    # Projection is an in-memory structural check, never a saved approval.
    projection_draft = draft.copy()
    pending = projection_draft.review_status.eq("pending")
    projection_draft.loc[pending, "review_status"] = "approved"
    projection_draft.loc[pending, "reviewed_by"] = "projection-only"
    projection_draft.loc[pending & projection_draft.review_notes.eq(""), "review_notes"] = "Projection only; human review required"
    try:
        projected = append_augmentations(original, projection_draft)
        projected_train = projected[projected.partition.eq("train")]
        projected_quality, projected_aspects = quality_counts(projected_train), aspect_counts(projected_train)
        projection_error = None
    except ValueError as error:
        projected_quality, projected_aspects = None, None
        projection_error = str(error)
    minority_sources = real_train[real_train.label.isin((1, 3))]
    source_review = [
        {"source_review_id": row.review_id, "ground_truth": CLASS_NAMES[int(row.label)],
         "product_id": row.product_id, "fold": int(row.fold),
         "review_text": row.review_text, "star_rating": int(row.star_rating),
         "check": "Confirm corroborating deceptive indicators; rating/image mismatch alone is insufficient"
                  if row.label == 1 else "Confirm primary text is off-topic for product/seller context"}
        for row in minority_sources.itertuples()
    ]
    sparse_sources = []
    for row in real_train[real_train.label.eq(0)].itertuples():
        pairs = {(item["category"], item["sentiment"]) for item in row.annotations}
        sparse = sorted(pair for pair in pairs if original_aspects[pair] < 10)
        if sparse:
            sparse_sources.append({
                "source_review_id": row.review_id, "product_id": row.product_id,
                "fold": int(row.fold), "product_title": getattr(row, "product_title", ""),
                "review_text": row.review_text, "aspect_annotations": json.loads(row.aspect_annotations),
                "sparse_targets": [{"category": category, "sentiment": POLARITIES[index].lower()}
                                   for category, index in sparse],
                "check": "Confirm every target against the fixed taxonomy before multiplying it. "
                         "Accuracy-of-description needs a reviewer-stated advertised/received comparison; "
                         "seller_service needs concrete seller-controlled conduct.",
            })
    return {
        "scope": "Original training only; real validation/test excluded from generation",
        "original_dataset_sha256": fingerprint(original),
        "active_dataset_sha256": fingerprint(combined),
        "counts_unit": "ABSA counts distinct reviews containing each category/polarity, not repeated evidence spans",
        "original_training_quality": quality_counts(real_train),
        "active_training_quality": quality_counts(active_train),
        "projected_training_quality_if_all_pending_pass_human_review": projected_quality,
        "projection_validation_error": projection_error,
        "draft_status_counts": draft.review_status.value_counts().to_dict(),
        "pending_quality_counts": draft.loc[draft.review_status.eq("pending"), "ground_truth"].value_counts().to_dict(),
        "source_variants_drafted": draft.source_review_id.value_counts().to_dict(),
        "original_training_fold_quality": [
            {"fold": int(fold), "quality_counts": quality_counts(rows),
             "product_count": int(rows.product_id.nunique())}
            for fold, rows in real_train.groupby("fold")
        ],
        "authentic_training_aspect_sentiments": [
            {"category": category, "sentiment": polarity.lower(),
             "original_review_count": original_aspects[(category, index)],
             "active_review_count": active_aspects[(category, index)],
             "projected_review_count_if_all_pending_approved": projected_aspects[(category, index)] if projected_aspects is not None else None,
             "original_product_count": int(real_train.loc[
                 real_train.label.eq(0) & real_train.annotations.map(
                     lambda annotations: any(item["category"] == category and item["sentiment"] == index
                                             for item in annotations)), "product_id"].nunique())}
            for category in ASPECTS for index, polarity in enumerate(POLARITIES)
        ],
        "minority_source_label_review": source_review,
        "sparse_aspect_source_review": sparse_sources,
        "batch_policy": {
            "drafts_per_batch": 20, "suggested_max_variants_per_original_source": 3,
            "source_selection": "Cover source products and OOF folds; do not copy just the first rows",
            "human_review": "Source label, preserved meaning, exact evidence and selected images before approval",
            "targets": "Increase minority support gradually; do not promise equal classes or final accuracy",
            "independent_information": "Paraphrases add wording variation, not independent products or real experiences",
        },
        "pending_draft_review": [
            {"review_id": row.review_id, "source_review_id": row.source_review_id,
             "ground_truth": row.ground_truth, "review_notes": row.review_notes}
            for row in draft[draft.review_status.eq("pending")].itertuples()
        ],
    }


def diverse_sources(rows, limit):
    """Rotate through product groups from every fold before taking more siblings."""
    by_fold = {}
    for (fold, _), group in rows.groupby(["fold", "product_id"], sort=True):
        by_fold.setdefault(fold, []).append(group.sort_values("review_id").to_dict("records"))
    selected = []
    while by_fold and len(selected) < limit:
        for fold in sorted(list(by_fold)):
            if len(selected) == limit:
                break
            queue = by_fold[fold].pop(0)
            selected.append(queue.pop(0))
            if queue:
                by_fold[fold].append(queue)
            if not by_fold[fold]:
                del by_fold[fold]
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--splits", default=str(DEFAULT_SPLITS_PATH))
    parser.add_argument("--augmentations", default="data/augmented_reviews.csv")
    parser.add_argument("--check-drafts", action="store_true", help="Check pending rows structurally without approving or saving them")
    parser.add_argument("--audit-images", action="store_true", help="Print URLs shared between original partitions or OOF folds")
    parser.add_argument("--export-sources", help="Write source/context and image audit JSON for LLM generation")
    parser.add_argument("--balance-report", help="Write training quality/aspect counts and minority-source review notes as JSON")
    parser.add_argument("--labels", nargs="+", choices=CLASS_NAMES, default=["deceptive", "irrelevant", "authentic"])
    parser.add_argument("--limit-per-class", type=int, default=5)
    args = parser.parse_args()
    original = load_experiment(args.data, args.splits, require_annotations=True)
    combined = load_experiment(args.data, args.splits, args.augmentations, require_annotations=True)
    draft = pd.read_csv(args.augmentations, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if args.check_drafts:
        checked = draft.copy()
        pending = checked.review_status.eq("pending")
        checked.loc[pending, "review_status"] = "approved"
        checked.loc[pending, "reviewed_by"] = "structural-check-only"
        checked.loc[pending & checked.review_notes.eq(""), "review_notes"] = "Structural check; human approval still required"
        append_augmentations(original, checked)
        print("Draft structure/lineage/image checks passed. No rows were approved or saved.")
    scopes = image_scopes(original)
    shared = [
        {"url": url, "scopes": [{"partition": p, "fold": f} for p, f in sorted(groups)]}
        for url, groups in scopes.items() if len(groups) > 1
    ]
    summary = {
        "draft_status_counts": draft.review_status.value_counts().to_dict(),
        "accepted_augmentation_count": len(combined) - len(original),
        "partition_counts": combined.partition.value_counts().to_dict(),
        "training_quality_counts": {
            CLASS_NAMES[int(label)]: int(count)
            for label, count in combined.loc[combined.partition.eq("train"), "label"].value_counts().items()
        },
        "cross_partition_image_url_count": sum(len({group["partition"] for group in row["scopes"]}) > 1 for row in shared),
        "cross_scope_image_url_count": len(shared),
        "image_audit_scope": "Exact URL reuse only; photos/availability and identical content at different URLs require separate inspection",
    }
    if args.audit_images:
        summary["cross_scope_image_urls"] = shared
    print(json.dumps(summary, indent=2))
    if args.balance_report:
        path = Path(args.balance_report)
        if path.exists():
            raise FileExistsError(f"Balance report already exists: {path}; choose another filename")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(balancing_report(original, combined, draft), indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved training balance report to {path}")
    if args.export_sources:
        if args.limit_per_class < 1:
            raise ValueError("limit-per-class must be positive")
        sources = []
        train = original[original.partition.eq("train")]
        for quality in args.labels:
            group = train[train.label.eq(CLASS_NAMES.index(quality))]
            for row in diverse_sources(group, args.limit_per_class):
                safe = bool(row["image_urls"]) and all(scopes[url] == {("train", int(row["fold"]))} for url in row["image_urls"])
                sources.append({
                    "source_review_id": row["review_id"], "product_id": row["product_id"],
                    "product_title": row.get("product_title", ""), "product_description": row.get("product_description", ""),
                    "review_text": row["review_text"], "ground_truth": quality,
                    "aspect_annotations": json.loads(row["aspect_annotations"]) if row["label"] == 0 else [],
                    "star_rating": int(row["star_rating"]), "fold": int(row["fold"]),
                    "image_mode": "source" if safe else "none",
                    "review_image_urls": row["image_urls"] if safe else [],
                    "requires_source_label_review": quality == "deceptive",
                })
        path = Path(args.export_sources)
        if path.exists():
            raise FileExistsError(f"Source pack already exists: {path}; choose another filename")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"sources": sources, "image_audit": shared}, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Exported {len(sources)} training sources to {path}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(f"ERROR: {error}") from error
