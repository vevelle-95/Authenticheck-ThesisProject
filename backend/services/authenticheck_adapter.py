"""Normalize the existing online inference pipeline for comparison."""

from functools import lru_cache
from pathlib import Path
import sys

from .taxonomy import category_for, normalize_sentiment


@lru_cache(maxsize=1)
def _pipeline():
    stage2 = Path(__file__).resolve().parents[2] / "stage2"
    if str(stage2) not in sys.path:
        sys.path.insert(0, str(stage2))
    import absa_model
    from online_inference import DEFAULT_XGB_PATH, OnlineInference

    return OnlineInference(
        roberta_model=absa_model.STAGE1_MODEL_DIR,
        xgb_path=DEFAULT_XGB_PATH,
        absa_dir=absa_model.DEFAULT_ABSA_MODEL_DIR,
    )


def run_authenticheck(reviews, product_title, product_description):
    import pandas as pd

    pipeline = _pipeline()
    pipeline.check_artifacts()
    records = [{
        "review_id": review["id"], 
        "text": review["text"],
        "image_urls": review.get("image_urls") or ([review["image_url"]] if review.get("image_url") else []),
        "product_description": product_description,
        "product_title": product_title,
        "star_rating": review["star_rating"],
    } for review in reviews]
    output = pipeline.run(pd.DataFrame.from_records(records))
    if len(output.get("reviews", [])) != len(reviews):
        raise RuntimeError("AuthentiCheck returned an unexpected review count")
    normalized = []
    for review, result in zip(reviews, output["reviews"]):
        aspects = []
        for aspect in result.get("aspectSentiment", []):
            evidence = aspect.get("aspect", "")
            category = category_for(evidence)
            if category:
                aspects.append({
                    "category": category, "evidence": evidence,
                    "sentiment": normalize_sentiment(aspect["sentiment"]),
                    "sentiment_source": "authenticheck_absa",
                })
        normalized.append({
            "id": review["id"],
            "classification": result.get("label"),
            "aspects": aspects,
        })
    return normalized
