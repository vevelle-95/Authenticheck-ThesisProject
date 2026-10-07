"""Validate the shared dataset or train the adapted six-aspect baseline."""

import argparse
import json

from datasets.authenticheck_data import load_experiment
from runtime import load_config, resolve_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--csv")
    parser.add_argument("--splits")
    parser.add_argument("--augmentations", help="Training-only CSV; paths resolve from the baseline folder")
    parser.add_argument("--allow-unreviewed-augmentations", action="store_true", help="Experimental run: include pending drafts without marking them approved")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    augmentation_path = args.augmentations or config["data"].get("augmentations")
    assigned = load_experiment(
        resolve_path(args.csv or config["data"]["csv"]),
        resolve_path(args.splits or config["data"]["splits"]),
        resolve_path(augmentation_path) if augmentation_path else None,
        allow_unreviewed=args.allow_unreviewed_augmentations,
    )
    summary = {
        name: {"reviews": len(rows), "products": rows.product_id.nunique(),
               "authentic_reviews": int(rows.label.eq(0).sum())}
        for name, rows in assigned.groupby("partition")
    }
    print(json.dumps(summary, indent=2))
    from training_augmentation import augmentation_summary
    print("Augmentation usage:", json.dumps(augmentation_summary(assigned)))
    if args.validate_only:
        return
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    from training.train import main as train_main
    print("Best checkpoint:", train_main(assigned, config))


if __name__ == "__main__":
    main()
