"""Evaluate a frozen selected checkpoint on the shared held-out test partition."""

import argparse

from datasets.authenticheck_data import fingerprint, load_reviews, read_splits
from inference import predict_frame
from runtime import load_checkpoint, load_config, resolve_path, write_json
from training.eval import compute_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--csv")
    parser.add_argument("--splits")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best.pt")
    parser.add_argument("--output", default="outputs/test_metrics.json")
    args = parser.parse_args()
    output_path = resolve_path(args.output)
    if output_path.exists():
        raise FileExistsError(f"Report already exists: {output_path}; choose another filename")
    config = load_config(args.config)
    assigned = read_splits(
        load_reviews(resolve_path(args.csv or config["data"]["csv"]), require_annotations=True),
        resolve_path(args.splits or config["data"]["splits"]),
    )
    checkpoint_path = resolve_path(args.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path)
    metadata = checkpoint["training_metadata"]
    if metadata["dataset_sha256"] != fingerprint(assigned):
        raise ValueError("Evaluation dataset differs from the training dataset")
    for partition, name in [("train", "fit_review_ids"), ("validation", "validation_review_ids")]:
        expected = set(assigned.loc[assigned.partition.eq(partition) & assigned.label.eq(0), "review_id"])
        if set(metadata[name]) != expected:
            raise ValueError(f"Checkpoint {partition} membership differs from shared splits")
    test = assigned[assigned.partition.eq("test") & assigned.label.eq(0)]
    predictions, raw, _ = predict_frame(test, checkpoint_path)
    report = {
        "model_version": checkpoint["model_version"], "partition": "test",
        "review_count": len(test), "product_count": test.product_id.nunique(),
        "scope": "standalone ABSA on ground-truth Authentic reviews, six shared aspects",
        "metrics": compute_metrics(raw, checkpoint["threshold"]), "predictions": predictions,
    }
    write_json(output_path, report, overwrite=False)
    print("Held-out report saved:", output_path)


if __name__ == "__main__":
    main()
