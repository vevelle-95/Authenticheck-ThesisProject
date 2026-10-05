"""Our multimodal review gate followed by fixed-category ABSA and aggregation."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
import torch
import xgboost as xgb
from transformers import AutoTokenizer

from stage2 import absa_model
from stage1.features import FeatureExtractor
from model_contract import ASPECTS, CLASS_NAMES, FEATURE_COLUMNS, INPUT_VERSION, POLARITIES, SENSORY_POLICY, SENTIMENT_TARGET_POLICY, TAXONOMY_VERSION, parse_image_urls

DEFAULT_XGB_PATH = absa_model.MODELS_DIR / "xgboost_meta_classifier.json"
DEFAULT_OUTPUT_PATH = absa_model.DATA_DIR / "online_inference_results.json"
API_CLASS_NAMES = list(CLASS_NAMES)


class OnlineInference:
    def __init__(self, roberta_model, xgb_path, absa_dir, threshold=None):
        self.roberta_model = Path(roberta_model)
        self.xgb_path = Path(xgb_path)
        self.absa_dir = Path(absa_dir)
        self.threshold = threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.extractor = FeatureExtractor(self.roberta_model)
        self._xgb = self._absa = self._absa_tokenizer = None

    def check_artifacts(self):
        required = [
            self.roberta_model / "config.json",
            self.xgb_path, self.xgb_path.with_suffix(".metadata.json"),
            self.absa_dir / "absa_config.json", self.absa_dir / "model.pt",
            self.absa_dir / "encoder" / "config.json", self.absa_dir / "tokenizer" / "tokenizer_config.json",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if not any((self.roberta_model / name).is_file() for name in ("model.safetensors", "pytorch_model.bin")):
            missing.append(str(self.roberta_model / "model.safetensors"))
        if missing:
            raise FileNotFoundError("Train our v2 bundle first. Missing artifacts: " + ", ".join(missing))
        text = json.loads((self.roberta_model / "config.json").read_text(encoding="utf-8"))
        meta = json.loads(self.xgb_path.with_suffix(".metadata.json").read_text(encoding="utf-8"))
        absa = json.loads((self.absa_dir / "absa_config.json").read_text(encoding="utf-8"))
        if text.get("input_contract_version") != INPUT_VERSION or meta.get("input_version") != INPUT_VERSION:
            raise ValueError("Stage 1 checkpoint contract mismatch; old feature/model bundles require retraining")
        if meta.get("feature_columns") != list(FEATURE_COLUMNS) or meta.get("class_order") != list(CLASS_NAMES):
            raise ValueError("XGBoost feature/class order mismatch")
        if absa.get("model_version") != absa_model.MODEL_VERSION or absa.get("taxonomy_version") != TAXONOMY_VERSION or absa.get("aspects") != list(ASPECTS):
            raise ValueError("ABSA bundle does not implement the fixed ten-category taxonomy")
        if absa.get("sensory_domain_policy") != SENSORY_POLICY:
            raise ValueError("ABSA sensory policy mismatch; retrain without product-category gating")
        if absa.get("sentiment_target_policy") != SENTIMENT_TARGET_POLICY:
            raise ValueError("ABSA sentiment target policy mismatch; retrain with mixed-polarity support")

    def preload(self):
        self.check_artifacts()
        self.extractor.load_text()
        self.extractor.load_clip()
        self._load_xgb()
        self._load_absa()

    def _load_xgb(self):
        if self._xgb is None:
            self._xgb = xgb.XGBClassifier()
            self._xgb.load_model(self.xgb_path)
        return self._xgb

    def _load_absa(self):
        if self._absa is None:
            self._absa = absa_model.ABSAHeadModel.from_pretrained(self.absa_dir).to(self.device).eval()
            self._absa_tokenizer = AutoTokenizer.from_pretrained(self._absa.tokenizer_dir, local_files_only=True)
        return self._absa, self._absa_tokenizer

    def extract_features(self, product_description, text, image_urls, star_rating):
        # Description is retained in the API for annotation/reporting, not model input.
        features, best, scores = self.extractor.extract(text, image_urls, star_rating)
        return features, features[:4], features[4], features[5], best

    def classify(self, features):
        frame = pd.DataFrame([features], columns=FEATURE_COLUMNS)
        probabilities = self._load_xgb().predict_proba(frame)[0].tolist()
        return int(max(range(4), key=lambda index: probabilities[index])), probabilities

    def aspect_sentiment(self, text, product_category=""):
        model, tokenizer = self._load_absa()
        return model.predict([text], tokenizer, threshold=self.threshold)[0]

    def run(self, df):
        self.check_artifacts()

        title_col = "product_title" if "product_title" in df else "prod_title"
        description_col = "product_description" if "product_description" in df else "prod_description"
        text_col = "review_text" if "review_text" in df else "text"

        if not all(c in df.columns for c in (title_col, description_col, text_col)):
            raise ValueError("product title, description and review cols are ALL required!")

        results, verdicts = [], []
        for row in df.to_dict("records"):
            title = row[title_col]
            description = row[description_col]
            review_text = row[text_col]

            text = " ".join(
                str(v).strip() for v in (title, description, review_text)
                if v is not None and str(v).strip()
            )

            urls = row.get("review_image_urls", row.get("image_urls", row.get("image_url")))

            features, p_text, similarity, rating, best = self.extract_features(
                row.get("product_description", ""), text, urls, row.get("star_rating"),
            )

            verdict, probabilities = self.classify(features)
            entry = {
                "id": str(row.get("review_id", row.get("id", ""))),
                "product_id": str(row.get("product_id", "")),
                "text": text, "starRating": float(row["star_rating"]), "label": CLASS_NAMES[verdict],
                "confidence": round(max(probabilities), 4),
                "probabilities": dict(zip(CLASS_NAMES, (round(value, 4) for value in probabilities))),
                "features": {"p_text": dict(zip(CLASS_NAMES, p_text)), "s_clip": similarity, "r_star": rating},
                "signals": ["DOST-RoBERTa review-text evidence",
                            "M-CLIP matched buyer-image similarity" if best else "No usable buyer image; CLIP score is 0",
                            "Normalized star rating"],
                "bestImageUrl": best,
            }
            if verdict == 0:
                entry["aspectSentiment"] = self.aspect_sentiment(text)

            results.append(entry)
            verdicts.append(CLASS_NAMES[verdict])

        return {**build_summary(df, verdicts, results), "reviews": results}


def valid_aspects(review):
    for item in review.get("aspectSentiment", []):
        category = item.get("category", item.get("aspect"))
        sentiment = str(item.get("sentiment", "")).lower()
        if category not in ASPECTS or sentiment not in {"positive", "neutral", "negative"}:
            raise ValueError("Invalid fixed-category sentiment output")
        yield category, sentiment


def aggregate_products(results, authentic_only=True):
    groups = {}
    for review in results:
        product = str(review.get("product_id", ""))
        group = groups.setdefault(product, {"product_id": product, "values": [], "reviewCount": 0})
        group["reviewCount"] += 1
        if authentic_only and review.get("label") != "authentic":
            continue
        group["values"].extend({"negative": -1, "neutral": 0, "positive": 1}[polarity]
                               for _, polarity in valid_aspects(review))
    return [{
        "product_id": product, "reviewCount": group["reviewCount"], "aspectCount": len(group["values"]),
        "aggregateSentiment": sum(group["values"]) / len(group["values"]) if group["values"] else None,
    } for product, group in groups.items()]


def aggregate_aspects(results):
    groups = {}
    counts = {"positive": 0, "neutral": 0, "negative": 0}
    for review in results:
        if review.get("label") != "authentic":
            continue
        for category, polarity in valid_aspects(review):
            counts[polarity] += 1
            bucket = groups.setdefault(category, {"name": category, "mentions": 0, "positive": 0})
            bucket["mentions"] += 1
            bucket["positive"] += int(polarity == "positive")
    return [{
        "name": category, "mentions": groups[category]["mentions"],
        "positivePercent": round(100 * groups[category]["positive"] / groups[category]["mentions"], 2),
    } for category in ASPECTS if category in groups], counts


def build_summary(df, verdicts, results=None):
    results = results or []
    counts = {name: verdicts.count(name) for name in CLASS_NAMES}
    authentic_ratings = [row["starRating"] for row in results if row.get("label") == "authentic"]
    aspects, sentiments = aggregate_aspects(results)
    product_summaries = aggregate_products(results)
    aspect_count = sum(row["aspectCount"] for row in product_summaries)
    overall = (sum(row["aggregateSentiment"] * row["aspectCount"] for row in product_summaries
                   if row["aggregateSentiment"] is not None) / aspect_count) if aspect_count else None

    return {
        "reviewCount": len(df), 
        "authenticShare": 100 * counts["authentic"] / len(df) if len(df) else 0,
        "verifiedRating": sum(authentic_ratings) / len(authentic_ratings) if authentic_ratings else None,
        "counts": counts, 
        "sentimentCounts": sentiments, 
        "aspects": aspects,
        "aggregateSentiment": overall, 
        "perProduct": product_summaries,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--roberta-model", default=str(absa_model.STAGE1_MODEL_DIR))
    parser.add_argument("--xgb-path", default=str(DEFAULT_XGB_PATH))
    parser.add_argument("--absa-dir", default=str(absa_model.DEFAULT_ABSA_MODEL_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()
    pipe = OnlineInference(args.roberta_model, args.xgb_path, args.absa_dir, args.threshold)
    output = pipe.run(pd.read_csv(args.data, keep_default_na=False))
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
