"""Validated records and persistent product-disjoint splits used by both stages."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from model_contract import SEED, map_quality, normalize_rating, parse_annotations, parse_image_urls

SPLIT_VERSION = "product-70-15-15-oof5-v1"


def load_reviews(path, require_annotations=False):
    frame = pd.read_csv(path, encoding="utf-8-sig", skipinitialspace=True, keep_default_na=False,
                        dtype={"review_id": str, "product_id": str})
    required = {"review_id", "product_id", "review_text", "review_image_urls", "star_rating", "ground_truth"}
    if require_annotations:
        required.add("aspect_annotations")
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing CSV fields: {sorted(missing)}. test_reviews.csv is a fixture; supply the finalized schema for training.")
    if frame.empty:
        raise ValueError("The dataset is empty")
    for column in ("review_id", "product_id", "review_text"):
        frame[column] = frame[column].astype(str).str.strip()
        if frame[column].eq("").any():
            raise ValueError(f"{column} cannot be blank")
    if frame.review_id.duplicated().any():
        raise ValueError("review_id must uniquely identify each review")
    if frame.review_text.str.casefold().duplicated().any():
        raise ValueError("Duplicate review text must be resolved before partitioning")
    frame["label"] = frame.ground_truth.map(map_quality)
    frame["normalized_rating"] = frame.star_rating.map(normalize_rating)
    frame["image_urls"] = frame.review_image_urls.map(parse_image_urls)
    if frame.image_urls.map(len).eq(0).any():
        raise ValueError("Multimodal training/evaluation requires a matched review image for every record")
    if "product_category" not in frame:
        frame["product_category"] = ""
    if "product_description" not in frame:
        frame["product_description"] = ""
    if require_annotations:
        frame["annotations"] = [
            parse_annotations(row.aspect_annotations, row.review_text, row.product_category)
            if row.label == 0 else [] for row in frame.itertuples()
        ]
    return frame.reset_index(drop=True)


def fingerprint(frame):
    columns = sorted(column for column in frame if column not in {
        "label", "normalized_rating", "image_urls", "annotations", "partition", "fold",
    })
    records = frame.sort_values("review_id")[columns].astype(str).to_dict("records")
    return hashlib.sha256(json.dumps(records, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _balanced_groups(frame, fractions, seed):
    """Search seeded group assignments; proportions are approximate at product level."""
    products = frame.product_id.unique()
    if len(products) < 3:
        raise ValueError("At least three distinct products are required for product-disjoint partitions")
    groups = [frame.index[frame.product_id.eq(product)].to_numpy() for product in products]
    all_counts = np.bincount(frame.label, minlength=4)
    if (all_counts == 0).any():
        raise ValueError("All four review-quality classes are required")
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(1000):
        order = rng.permutation(len(groups))
        sizes = np.array([len(groups[i]) for i in order])
        cumulative = sizes.cumsum()
        cut1 = int(np.argmin(abs(cumulative[:-2] - len(frame) * fractions[0]))) + 1
        options = np.arange(cut1 + 1, len(groups))
        cut2 = int(options[np.argmin(abs(cumulative[options - 1] - len(frame) * sum(fractions[:2])))])
        segments = [order[:cut1], order[cut1:cut2], order[cut2:]]
        indices = [np.concatenate([groups[i] for i in segment]) for segment in segments]
        if any(set(frame.loc[index, "label"]) != {0, 1, 2, 3} for index in indices):
            continue
        train = frame.loc[indices[0]]
        if train.product_id.nunique() < 5 or train.groupby("label").product_id.nunique().min() < 5:
            continue
        score = sum(
            abs(len(index) / len(frame) - fraction)
            + np.abs(np.bincount(frame.loc[index, "label"], minlength=4) / all_counts - fraction).mean()
            for index, fraction in zip(indices, fractions)
        )
        if best is None or score < best[0]:
            best = (score, indices)
    if best is None:
        raise ValueError("Cannot create class-covered product splits with five OOF folds. More class-diverse product groups are needed; no row-split fallback is permitted.")
    return best[1]


def prepare_splits(frame, path, seed=SEED):
    path = Path(path)
    if path.exists():
        return read_splits(frame, path)
    train_index, val_index, test_index = _balanced_groups(frame, (0.7, 0.15, 0.15), seed)
    assigned = frame.copy()
    assigned["partition"] = ""
    assigned["fold"] = -1
    for name, indices in zip(("train", "validation", "test"), (train_index, val_index, test_index)):
        assigned.loc[indices, "partition"] = name
    train = assigned.loc[train_index]
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (fit, held) in enumerate(splitter.split(train, train.label, train.product_id)):
        if set(train.iloc[fit].label) != {0, 1, 2, 3}:
            raise ValueError(f"OOF fold {fold} has missing fitting classes; collect more diverse product groups")
        assigned.loc[train.iloc[held].index, "fold"] = fold
    payload = {
        "version": SPLIT_VERSION, "seed": seed, "dataset_sha256": fingerprint(frame),
        "records": assigned[["review_id", "product_id", "partition", "fold"]].to_dict("records"),
        "counts": assigned.partition.value_counts().to_dict(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return read_splits(frame, path)


def read_splits(frame, path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("version") != SPLIT_VERSION or payload.get("dataset_sha256") != fingerprint(frame):
        raise ValueError("Split manifest does not match this dataset/version. Create a new manifest for a new experiment.")
    records = pd.DataFrame(payload["records"])
    if records.review_id.duplicated().any() or set(records.review_id) != set(frame.review_id):
        raise ValueError("Split manifest must assign every review exactly once")
    joined = frame.merge(records, on=["review_id", "product_id"], validate="one_to_one")
    if len(joined) != len(frame) or set(joined.partition) != {"train", "validation", "test"}:
        raise ValueError("Invalid partition coverage")
    if joined.groupby("product_id").partition.nunique().max() != 1:
        raise ValueError("Product leakage between partitions")
    training = joined[joined.partition.eq("train")]
    if set(training.fold) != set(range(5)) or training.groupby("product_id").fold.nunique().max() != 1:
        raise ValueError("Invalid or leaking OOF fold assignments")
    if not joined.loc[~joined.partition.eq("train"), "fold"].eq(-1).all():
        raise ValueError("Validation/test records cannot belong to training folds")
    return joined
