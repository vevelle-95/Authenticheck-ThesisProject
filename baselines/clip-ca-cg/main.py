"""Validate the shared dataset or train the adapted six-aspect baseline."""

import argparse
import json

from datasets.authenticheck_data import load_reviews, read_splits
from runtime import load_config, resolve_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--csv")
    parser.add_argument("--splits")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    assigned = read_splits(
        load_reviews(resolve_path(args.csv or config["data"]["csv"]), require_annotations=True),
        resolve_path(args.splits or config["data"]["splits"]),
    )
    summary = {
        name: {"reviews": len(rows), "products": rows.product_id.nunique(),
               "authentic_reviews": int(rows.label.eq(0).sum())}
        for name, rows in assigned.groupby("partition")
    }
    print(json.dumps(summary, indent=2))
    if args.validate_only:
        return
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    from training.train import main as train_main
    print("Best checkpoint:", train_main(assigned, config))


if __name__ == "__main__":
    main()
