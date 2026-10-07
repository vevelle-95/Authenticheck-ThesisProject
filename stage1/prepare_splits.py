"""Validate the final training schema and persist product-disjoint partitions."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model_contract import DEFAULT_DATA_PATH, DEFAULT_SPLITS_PATH
from model_data import load_reviews, prepare_splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--splits", default=str(DEFAULT_SPLITS_PATH))
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    frame = load_reviews(args.data, require_annotations=True)
    if args.validate_only:
        print(f"Validated {len(frame)} reviews. No training or partitioning performed.")
        return
    assigned = prepare_splits(frame, args.splits)
    print(json.dumps(assigned.groupby("partition").agg(reviews=("review_id", "count"), products=("product_id", "nunique")).to_dict(), indent=2))


if __name__ == "__main__":
    main()
