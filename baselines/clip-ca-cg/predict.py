"""Predict aspect JSON from the existing CSV or a single new review."""

import argparse

import pandas as pd

from datasets.authenticheck_data import load_reviews, parse_image_urls, read_splits
from inference import predict_frame
from runtime import load_config, resolve_path, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--csv")
    parser.add_argument("--splits")
    parser.add_argument("--partition", choices=["train", "validation", "test", "all"], default="test")
    parser.add_argument("--text", help="A single review; its output is the aspect list directly")
    parser.add_argument("--image-url", action="append", default=[])
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best.pt")
    parser.add_argument("--prepare-cache", action="store_true")
    parser.add_argument("--output", default="outputs/predictions.json")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.text is not None:
        if not args.text.strip():
            parser.error("--text cannot be blank")
        frame = pd.DataFrame([{
            "review_id": "manual_review", "review_text": args.text.strip(),
            "image_urls": parse_image_urls(args.image_url),
        }])
    else:
        frame = load_reviews(resolve_path(args.csv or config["data"]["csv"]))
        if args.partition != "all":
            frame = read_splits(frame, resolve_path(args.splits or config["data"]["splits"]))
            frame = frame[frame.partition.eq(args.partition)]
    output, _, _ = predict_frame(frame, resolve_path(args.checkpoint), prepare_cache=args.prepare_cache)
    payload = output["reviews"][0]["aspects"] if args.text is not None else output
    write_json(resolve_path(args.output), payload)
    print("Predictions saved:", resolve_path(args.output))


if __name__ == "__main__":
    main()
