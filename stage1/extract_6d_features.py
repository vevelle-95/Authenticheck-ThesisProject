"""Build XGBoost train/validation features; test features belong to final evaluation."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
from stage1 import config
from stage1.features import FeatureExtractor
from model_contract import FEATURE_COLUMNS, INPUT_VERSION
from model_data import fingerprint, load_reviews, read_splits


def build_features(assigned, oof_path, output, model_dir, extractor=None):
    oof_path = Path(oof_path)
    oof = pd.read_csv(oof_path, dtype={"review_id": str, "product_id": str})
    metadata = json.loads(oof_path.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    if metadata.get("dataset_sha256") != fingerprint(assigned) or metadata.get("input_version") != INPUT_VERSION:
        raise ValueError("OOF features belong to a different dataset/input contract")
    training = assigned[assigned.partition.eq("train")]
    folds = metadata.get("folds", [])
    if len(folds) != 5 or {item.get("fold") for item in folds} != set(range(5)):
        raise ValueError("OOF metadata must include all five fitting/prediction cohorts")
    for item in folds:
        fit = training[training.fold.ne(item["fold"])]
        held = training[training.fold.eq(item["fold"])]
        if (set(item.get("fit_review_ids", [])) != set(fit.review_id)
                or set(item.get("predicted_review_ids", [])) != set(held.review_id)
                or set(item.get("fit_product_ids", [])) != set(fit.product_id)
                or set(item.get("predicted_product_ids", [])) != set(held.product_id)):
            raise ValueError("OOF provenance does not match the training folds")
    if extractor is None:
        final_metadata = json.loads(Path(model_dir, "training_metadata.json").read_text(encoding="utf-8"))
        validation = assigned[assigned.partition.eq("validation")]
        if (set(final_metadata.get("fit_review_ids", [])) != set(training.review_id)
                or set(final_metadata.get("validation_review_ids", [])) != set(validation.review_id)
                or final_metadata.get("training_sha256") != fingerprint(training)):
            raise ValueError("Final RoBERTa checkpoint does not match this training/validation experiment")
    if oof.review_id.duplicated().any() or set(oof.review_id) != set(training.review_id):
        raise ValueError("OOF features must cover training reviews exactly once")
    aligned = training[["review_id", "product_id", "fold"]].merge(
        oof, on=["review_id", "product_id", "fold"], validate="one_to_one",
    )
    if len(aligned) != len(training) or not aligned.apply(
        lambda row: row.probability_source == f"oof_fold_{row.fold}", axis=1,
    ).all():
        raise ValueError("OOF product/fold provenance mismatch")
    extractor = extractor or FeatureExtractor(model_dir)
    lookup = aligned.set_index("review_id")
    records = []
    for row in assigned[assigned.partition.ne("test")].to_dict("records"):
        probabilities = lookup.loc[row["review_id"], list(FEATURE_COLUMNS[:4])].tolist() if row["partition"] == "train" else None
        features, best, scores = extractor.extract(
            row["review_text"], row["image_urls"], row["star_rating"], probabilities,
        )
        records.append({
            "review_id": row["review_id"], "product_id": row["product_id"],
            "partition": row["partition"], "fold": row["fold"], "ground_truth": row["label"],
            "probability_source": f"oof_fold_{row['fold']}" if row["partition"] == "train" else "final_roberta",
            **dict(zip(FEATURE_COLUMNS, features)), "best_image_url": best,
            "image_scores": json.dumps(scores),
        })
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output, index=False)
    output.with_suffix(".metadata.json").write_text(json.dumps({
        "input_version": INPUT_VERSION, "dataset_sha256": fingerprint(assigned),
    }, indent=2), encoding="utf-8")
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(config.DATA_PATH))
    parser.add_argument("--splits", default=str(config.SPLITS_PATH))
    parser.add_argument("--oof", default=str(config.DATA_DIR / "oof_probabilities.csv"))
    parser.add_argument("--output", default=str(config.FEATURES_PATH))
    parser.add_argument("--model", default=str(config.MODEL_DIR))
    args = parser.parse_args()
    assigned = read_splits(load_reviews(args.data), args.splits)
    result = build_features(assigned, args.oof, args.output, args.model)
    print(f"Saved {len(result)} train/validation feature rows. Test products were not processed.")


if __name__ == "__main__":
    main()
