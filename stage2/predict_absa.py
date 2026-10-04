"""Apply fixed-category ABSA to reviews already classified as Authentic."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
import torch
from transformers import AutoTokenizer
from stage2 import absa_model
from model_contract import map_quality


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--model-dir", default=str(absa_model.DEFAULT_ABSA_MODEL_DIR))
    parser.add_argument("--output", default=str(absa_model.DEFAULT_RESULTS_PATH))
    parser.add_argument("--threshold", type=float, default=None, help="Diagnostic override; default is saved validation-selected threshold")
    parser.add_argument("--authentic-only", action="store_true", help="Filter using predicted label/label_stage1; never use ground truth as an online gate")
    args = parser.parse_args()
    frame = pd.read_csv(args.data, keep_default_na=False)
    text_col = "review_text" if "review_text" in frame else "text"
    if text_col not in frame or frame[text_col].astype(str).str.strip().eq("").any():
        raise ValueError("Nonempty review_text/text is required")
    if args.authentic_only:
        label_col = next((name for name in ("label", "label_stage1") if name in frame), None)
        if label_col is None:
            raise ValueError("Authentic-only inference needs predicted label or label_stage1")
        frame = frame[frame[label_col].map(map_quality).eq(0)]
    if frame.empty:
        raise ValueError("No Authentic reviews remain")
    model = absa_model.ABSAHeadModel.from_pretrained(args.model_dir)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model.tokenizer_dir, local_files_only=True)
    results = model.predict(
        frame[text_col].astype(str).tolist(), tokenizer,
        threshold=args.threshold,
    )
    output = [{
        "review_id": row.get("review_id", ""), "product_id": row.get("product_id", ""),
        "text": row[text_col], "aspectSentiment": aspects,
    } for row, aspects in zip(frame.to_dict("records"), results)]
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
