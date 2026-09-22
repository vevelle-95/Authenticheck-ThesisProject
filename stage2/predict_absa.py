"""Stage 2 - ABSA inference for authenticated reviews.

Reads a CSV of review texts (must include an `authentic` or `label_stage1`
column for filtering to verified authentic reviews only) and writes a
structured per-aspect sentiment profile for each authentic review.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer

import absa_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser(description="Run ABSA inference on authenticated reviews")
    parser.add_argument("--data", default=str(absa_model.DEFAULT_DATA_PATH), help="CSV of reviews to analyze")
    parser.add_argument("--model-dir", default=str(absa_model.DEFAULT_ABSA_MODEL_DIR), help="Trained ABSA model directory")
    parser.add_argument("--output", default=str(absa_model.DEFAULT_RESULTS_PATH), help="JSON output path")
    parser.add_argument("--max-length", type=int, default=absa_model.ABSA_MAX_LENGTH)
    parser.add_argument("--threshold", type=float, default=0.5, help="Aspect detection threshold")
    parser.add_argument("--authentic-only", action="store_true", help="Only analyze reviews classified as authentic")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: Data file not found at {data_path}.")
        return
    if not (Path(args.model_dir) / "absa_config.json").exists():
        print(f"ERROR: Trained ABSA model not found at {args.model_dir}. Run fine_tune_absa.py first.")
        return

    print(f"1. Loading reviews from {data_path}...")
    df = pd.read_csv(data_path, encoding="utf-8-sig", skipinitialspace=True)
    text_col = "text" if "text" in df.columns else "review_text"

    if args.authentic_only:
        if "authentic" in df.columns:
            df = df[df["authentic"].astype(str).str.lower().isin(["1", "true", "yes", "authentic"])]
        elif "label_stage1" in df.columns:
            df = df[df["label_stage1"] == 0]
        else:
            print("Warning: no authenticity column found; analyzing all reviews.")
        print(f"   Restricting to authentic reviews: {len(df)} remaining.")

    print(f"2. Loading ABSA model from {args.model_dir}...")
    model = absa_model.ABSAHeadModel.from_pretrained(args.model_dir).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(model.encoder_name)

    print("3. Generating aspect-level sentiment profiles...")
    encodings = tokenizer(
        list(df[text_col]),
        padding="max_length",
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
    )
    results = model.predict(
        encodings["input_ids"].to(DEVICE),
        encodings["attention_mask"].to(DEVICE),
        threshold=args.threshold,
    )

    output = [
        {"text": record[text_col], "aspect_sentiment": aspects}
        for record, aspects in zip(df.to_dict("records"), results)
    ]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"Success! Per-aspect sentiment profiles saved to {out_path}")


if __name__ == "__main__":
    main()