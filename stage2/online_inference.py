"""Stage 2 - Online inference phase (Revised Section 3.4, items 3-9).

Loads the saved artifacts and runs, in order:
  4. Feature extraction: DOST-RoBERTa P_text, M-CLIP S_clip, normalized R_star.
  5. Deterministic (parameter-free) concatenation into the 6D vector X.
  6. XGBoost classification into the four quality classes.
  7. Filtering: only Authentic reviews propagate onward.
  8. ABSA aspect detection and aspect-conditioned sentiment on authentic reviews.
  9. Structured output (classification results, aspect sentiment, adjusted
     rating). Dashboard/graph visualization is intentionally excluded for now.
"""

import argparse
import json
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests
import torch
import torch.nn.functional as F
import xgboost as xgb
from PIL import Image
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import absa_model

DEFAULT_XGB_PATH = absa_model.MODELS_DIR / "xgboost_meta_classifier.json"
DEFAULT_OUTPUT_PATH = absa_model.DATA_DIR / "online_inference_results.json"

CLASS_NAMES = ["Authentic", "Deceptive", "Low Informational Value", "Irrelevant"]
STAR_SCALE_MIN = 1
STAR_SCALE_MAX = 5


def normalize_rating(rating):
    if rating is None or pd.isna(rating):
        rating = 3.0
    return max(0.0, min(1.0, (float(rating) - STAR_SCALE_MIN) / (STAR_SCALE_MAX - STAR_SCALE_MIN)))


def load_image(url):
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception:
        return Image.new("RGB", (224, 224), color="white")


class OnlineInference:
    def __init__(self, roberta_model, xgb_path, absa_dir, threshold=0.5):
        self.roberta_model = Path(roberta_model)
        self.xgb_path = Path(xgb_path)
        self.absa_dir = Path(absa_dir)
        self.threshold = threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._text_tokenizer = None
        self._text_model = None
        self._clip_text = None
        self._clip_image = None
        self._xgb = None
        self._absa = None
        self._absa_tokenizer = None

    def check_artifacts(self):
        missing = []
        if not (self.roberta_model / "config.json").exists():
            missing.append(f"Stage 1 RoBERTa classifier not found at {self.roberta_model}")
        if not self.xgb_path.exists():
            missing.append(f"XGBoost classifier not found at {self.xgb_path}")
        if not (self.absa_dir / "absa_config.json").exists():
            missing.append(f"ABSA model not found at {self.absa_dir}")
        if missing:
            raise FileNotFoundError(" | ".join(missing))

    def _load_text(self):
        if self._text_model is None:
            self._text_tokenizer = AutoTokenizer.from_pretrained(str(self.roberta_model))
            self._text_model = AutoModelForSequenceClassification.from_pretrained(
                str(self.roberta_model)
            ).to(self.device)
            self._text_model.eval()
        return self._text_tokenizer, self._text_model

    def _load_clip(self):
        if self._clip_text is None:
            self._clip_text = SentenceTransformer("clip-ViT-B-32-multilingual-v1")
            self._clip_image = SentenceTransformer("clip-ViT-B-32")
        return self._clip_text, self._clip_image

    def _load_xgb(self):
        if self._xgb is None:
            self._xgb = xgb.XGBClassifier()
            self._xgb.load_model(str(self.xgb_path))
        return self._xgb

    def _load_absa(self):
        if self._absa is None:
            self._absa = absa_model.ABSAHeadModel.from_pretrained(self.absa_dir).to(self.device)
            self._absa.eval()
            self._absa_tokenizer = AutoTokenizer.from_pretrained(self._absa.encoder_name)
        return self._absa, self._absa_tokenizer

    def extract_features(self, text, img_url, star_rating):
        tokenizer, model = self._load_text()
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=absa_model.ABSA_MAX_LENGTH).to(self.device)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = F.softmax(logits, dim=-1).squeeze(0).tolist()

        text_enc, image_enc = self._load_clip()
        image = load_image(img_url)
        img_emb = image_enc.encode(image, convert_to_tensor=True)
        txt_emb = text_enc.encode(text, convert_to_tensor=True)
        s_clip = util.cos_sim(img_emb, txt_emb).item()

        r_star = normalize_rating(star_rating)
        return probs + [s_clip, r_star], probs, s_clip, r_star

    def classify(self, features):
        probs = self._load_xgb().predict_proba([features])[0].tolist()
        verdict_idx = int(max(range(len(probs)), key=lambda i: probs[i]))
        return verdict_idx, probs

    def aspect_sentiment(self, text):
        absa, tokenizer = self._load_absa()
        enc = tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=absa_model.ABSA_MAX_LENGTH,
        ).to(self.device)
        return absa.predict(enc["input_ids"], enc["attention_mask"], threshold=self.threshold)[0]

    def run(self, df):
        self.check_artifacts()
        text_col = "text" if "text" in df.columns else "review_text"

        results = []
        verdicts = []
        for record in df.to_dict("records"):
            text = record[text_col]
            img_url = record.get("image_url", "")
            star_rating = record.get("star_rating", None)

            features, p_text, s_clip, r_star = self.extract_features(text, img_url, star_rating)
            verdict_idx, probs = self.classify(features)

            entry = {
                "text": text,
                "star_rating": star_rating,
                "classification": {
                    "verdict": CLASS_NAMES[verdict_idx],
                    "probabilities": {name: round(p, 4) for name, p in zip(CLASS_NAMES, probs)},
                },
                "features": {
                    "p_text": {
                        name: round(p, 4)
                        for name, p in zip(
                            ["p_authentic", "p_deceptive", "p_liv", "p_irrelevant"], p_text
                        )
                    },
                    "s_clip": round(s_clip, 4),
                    "r_star": round(r_star, 4),
                },
            }

            if verdict_idx == 0:
                entry["aspect_sentiment"] = self.aspect_sentiment(text)
            verdicts.append(CLASS_NAMES[verdict_idx])
            results.append(entry)

        summary = build_summary(df, verdicts)
        return {"summary": summary, "reviews": results}


def build_summary(df, verdicts):
    df = df.copy()
    df["verdict"] = verdicts
    authentic = df[df["verdict"] == "Authentic"]
    summary = {
        "review_count": len(df),
        "verdict_counts": df["verdict"].value_counts().to_dict(),
        "authentic_count": len(authentic),
        "authentic_share": round(len(authentic) / len(df), 4) if len(df) else None,
        "authenticity_adjusted_rating": (
            float(round(authentic["star_rating"].mean(), 4)) if len(authentic) else None
        ),
    }
    if "product_id" in df.columns:
        per_product = []
        for pid, group in df.groupby("product_id"):
            auth = group[group["verdict"] == "Authentic"]
            per_product.append({
                "product_id": pid,
                "review_count": len(group),
                "authentic_share": round(len(auth) / len(group), 4),
                "authenticity_adjusted_rating": (
                    float(round(auth["star_rating"].mean(), 4)) if len(auth) else None
                ),
            })
        summary["per_product"] = per_product
    return summary


def main():
    parser = argparse.ArgumentParser(description="AuthentiCheck online inference phase (items 3-9, no visualization)")
    parser.add_argument("--data", required=True, help="CSV of new reviews (text, image_url, star_rating)")
    parser.add_argument("--roberta-model", default=str(absa_model.STAGE1_MODEL_DIR), help="Fine-tuned Stage 1 DOST-RoBERTa classifier directory")
    parser.add_argument("--xgb-path", default=str(DEFAULT_XGB_PATH), help="Trained XGBoost meta-classifier")
    parser.add_argument("--absa-dir", default=str(absa_model.DEFAULT_ABSA_MODEL_DIR), help="Trained ABSA heads directory")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="JSON output path")
    parser.add_argument("--threshold", type=float, default=0.5, help="ABSA aspect detection threshold")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: Input data not found at {data_path}.")
        return

    print("1. Loading input reviews...")
    df = pd.read_csv(data_path, encoding="utf-8-sig", skipinitialspace=True)
    print(f"   Loaded {len(df)} reviews.")

    print("2. Initializing inference components...")
    pipe = OnlineInference(args.roberta_model, args.xgb_path, args.absa_dir, threshold=args.threshold)
    try:
        pipe.check_artifacts()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")

    print("3. Running classification -> filtering -> ABSA...")
    output = pipe.run(df)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"Success! Results saved to {out_path}")


if __name__ == "__main__":
    main()