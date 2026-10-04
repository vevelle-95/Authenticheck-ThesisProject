
import argparse
import ipaddress
import json
import socket
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

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
API_CLASS_NAMES = ["authentic", "deceptive", "liv", "irrelevant"]
STAR_SCALE_MIN = 1
STAR_SCALE_MAX = 5
STAGE1_MAX_LENGTH = 128
MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_IMAGES_PER_REVIEW = 5


def normalize_rating(rating):
    if rating is None or pd.isna(rating):
        rating = 3.0
    return max(0.0, min(1.0, (float(rating) - STAR_SCALE_MIN) / (STAR_SCALE_MAX - STAR_SCALE_MIN)))


def parse_image_urls(value):
    if value is None or (not isinstance(value, (list, tuple)) and pd.isna(value)):
        return []
    if isinstance(value, (list, tuple)):
        values = value
    else:
        text = str(value).strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
            values = parsed if isinstance(parsed, list) else [parsed]
        except json.JSONDecodeError:
            values = [part.strip() for part in text.split("|")]
    return list(dict.fromkeys(str(item).strip() for item in values if str(item).strip()))[:MAX_IMAGES_PER_REVIEW]


def image_url_is_safe(url):
    try:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            return False
        addresses = socket.getaddrinfo(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80))
        for address in addresses:
            ip = ipaddress.ip_address(address[4][0])
            if not ip.is_global:
                return False
        return True
    except (OSError, ValueError):
        return False


def load_image(url):
    if not image_url_is_safe(url):
        return None
    try:
        response = requests.get(
            url,
            timeout=(3, 7),
            stream=True,
            allow_redirects=False,
            headers={"User-Agent": "AuthentiCheck/1.0"},
        )
        response.raise_for_status()
        if not response.headers.get("Content-Type", "").lower().startswith("image/"):
            return None
        content_length = int(response.headers.get("Content-Length", "0") or 0)
        if content_length > MAX_IMAGE_BYTES:
            return None
        chunks = []
        total = 0
        for chunk in response.iter_content(64 * 1024):
            total += len(chunk)
            if total > MAX_IMAGE_BYTES:
                return None
            chunks.append(chunk)
        image = Image.open(BytesIO(b"".join(chunks)))
        image.load()
        return image.convert("RGB")
    except (requests.RequestException, OSError, ValueError):
        return None


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
        if not (self.absa_dir / "absa_config.json").exists() or not (self.absa_dir / "model.pt").exists():
            missing.append(f"ABSA model not found at {self.absa_dir}")
        if missing:
            raise FileNotFoundError(" | ".join(missing))

    def preload(self):
        self.check_artifacts()
        self._load_text()
        self._load_clip()
        self._load_xgb()
        self._load_absa()

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
            self._absa_tokenizer = AutoTokenizer.from_pretrained(self._absa.encoder_name, use_fast=True)
        return self._absa, self._absa_tokenizer

    def extract_features(self, product_description, text, image_urls, star_rating):
        tokenizer, model = self._load_text()
        inputs = tokenizer(
            str(product_description or ""),
            str(text),
            return_tensors="pt",
            padding="max_length",
            truncation="only_first",
            max_length=STAGE1_MAX_LENGTH,
        ).to(self.device)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = F.softmax(logits, dim=-1).squeeze(0).tolist()
        if len(probs) != 4:
            raise ValueError("Stage 1 model must output exactly four probabilities.")

        s_clip = 0.0
        best_image_url = ""
        urls = parse_image_urls(image_urls)
        if urls:
            text_enc, image_enc = self._load_clip()
            context = f"{product_description}\n{text}" if product_description else str(text)
            txt_emb = text_enc.encode(context, convert_to_tensor=True)
            scores = []
            for image_url in urls:
                image = load_image(image_url)
                if image is None:
                    continue
                img_emb = image_enc.encode(image, convert_to_tensor=True)
                scores.append((util.cos_sim(img_emb, txt_emb).item(), image_url))
            if scores:
                s_clip, best_image_url = max(scores, key=lambda item: item[0])

        r_star = normalize_rating(star_rating)
        return probs + [s_clip, r_star], probs, s_clip, r_star, best_image_url

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
            return_offsets_mapping=True,
        ).to(self.device)
        return absa.predict(
            enc["input_ids"],
            enc["attention_mask"],
            offset_mapping=enc["offset_mapping"],
            tokenizer=tokenizer,
            texts=[text],
            threshold=self.threshold,
        )[0]

    def run(self, df):
        self.check_artifacts()
        text_col = "text" if "text" in df.columns else "review_text"

        results = []
        verdicts = []
        for record in df.to_dict("records"):
            text = str(record[text_col])
            product_description = str(record.get("product_description", "") or "")
            image_urls = record.get("image_urls", record.get("image_url", ""))
            star_rating = record.get("star_rating", None)

            features, p_text, s_clip, r_star, best_image_url = self.extract_features(
                product_description, text, image_urls, star_rating
            )
            verdict_idx, probs = self.classify(features)
            label = API_CLASS_NAMES[verdict_idx]

            entry = {
                "id": str(record.get("review_id", "")),
                "text": text,
                "starRating": None if pd.isna(star_rating) else float(star_rating),
                "label": label,
                "confidence": round(float(max(probs)), 4),
                "probabilities": {name: round(p, 4) for name, p in zip(API_CLASS_NAMES, probs)},
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
                "signals": [
                    "DOST-RoBERTa product/review evidence",
                    "M-CLIP buyer-media similarity" if best_image_url else "No usable buyer image",
                    "Normalized star-rating evidence",
                ],
            }

            if verdict_idx == 0:
                entry["aspectSentiment"] = self.aspect_sentiment(text)
            verdicts.append(CLASS_NAMES[verdict_idx])
            results.append(entry)

        summary = build_summary(df, verdicts, results)
        return {**summary, "reviews": results}


def build_summary(df, verdicts, results=None):
    df = df.copy()
    df["verdict"] = verdicts
    authentic = df[df["verdict"] == "Authentic"]
    verdict_counts = df["verdict"].value_counts().to_dict()
    rated_authentic = pd.to_numeric(authentic.get("star_rating", pd.Series(dtype=float)), errors="coerce").dropna()
    aspects, sentiment_counts = aggregate_aspects(results or [])
    return {
        "reviewCount": len(df),
        "authenticShare": round(len(authentic) / len(df) * 100, 2) if len(df) else 0.0,
        "verifiedRating": float(round(rated_authentic.mean(), 2)) if len(rated_authentic) else None,
        "counts": {
            "authentic": int(verdict_counts.get("Authentic", 0)),
            "deceptive": int(verdict_counts.get("Deceptive", 0)),
            "liv": int(verdict_counts.get("Low Informational Value", 0)),
            "irrelevant": int(verdict_counts.get("Irrelevant", 0)),
        },
        "sentimentCounts": sentiment_counts,
        "aspects": aspects,
    }


def aggregate_aspects(results):
    grouped = {}
    sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
    for review in results:
        for item in review.get("aspectSentiment", []):
            name = str(item.get("aspect", "")).strip()
            sentiment = str(item.get("sentiment", "neutral")).lower()
            if not name or sentiment not in sentiment_counts:
                continue
            sentiment_counts[sentiment] += 1
            key = name.casefold()
            bucket = grouped.setdefault(key, {"name": name, "mentions": 0, "positive": 0})
            bucket["mentions"] += 1
            bucket["positive"] += int(sentiment == "positive")
    aspects = [
        {
            "name": bucket["name"],
            "mentions": bucket["mentions"],
            "positivePercent": round(bucket["positive"] / bucket["mentions"] * 100, 2),
        }
        for bucket in grouped.values()
    ]
    aspects.sort(key=lambda item: (-item["mentions"], item["name"].casefold()))
    return aspects[:12], sentiment_counts


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
        print("Required artifacts are missing. Train Stage 1 (train_roberta.py), Stage 2 (train_xgboost.py),")
        print("and ABSA (fine_tune_absa.py) first, or point --roberta-model/--xgb-path/--absa-dir at existing artifacts.")
        return

    print("3. Running classification -> filtering -> ABSA...")
    output = pipe.run(df)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"Success! Results saved to {out_path}")


if __name__ == "__main__":
    main()
