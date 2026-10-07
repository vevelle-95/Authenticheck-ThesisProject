"""Contracts shared by our training and inference code (no comparison adapters)."""

import json
import math
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_BUNDLE = PROJECT_ROOT / "models" / "own_model_v2"
BASE_MODEL = "dost-asti/RoBERTa-tl-cased"
INPUT_VERSION = "product-context-six-features-v3-missing-image-zero"
MODEL_VERSION = "authenticheck-2.1"
TAXONOMY_VERSION = "fixed-ten-v1"
SENSORY_POLICY = "annotation-guidelines-only"
SENTIMENT_TARGET_POLICY = "category-polarity-distribution-v1"
MAX_LENGTH = 128
SEED = 42
CLASS_NAMES = ("authentic", "deceptive", "liv", "irrelevant")
POLARITIES = ("Negative", "Neutral", "Positive")
ASPECTS = (
    "product_quality", "functionality", "performance", "design", "aesthetics",
    "sensory_experience", "value", "packaging", "seller_service", "accuracy_of_description",
)
ASPECT_DEFINITIONS = {
    "product_quality": "Overall build, material quality, durability, or craftsmanship.",
    "functionality": "Whether the product has the expected features or capabilities.",
    "performance": "How well functions work: speed, responsiveness, lag, accuracy, sound, battery performance, results, effectiveness, or reliability.",
    "design": "Ergonomics, form, layout, fit, comfort, feel in the hand, non-beauty texture, or ease of use.",
    "aesthetics": "Purely visual appeal: color, appearance, style, or look.",
    "sensory_experience": "Beauty, skincare, and personal care only: smell, fragrance, texture, consistency, or feel on the skin.",
    "value": "Worth its price: sulit, affordable, expensive, or overpriced.",
    "packaging": "Box condition, wrapping, protection, presentation, or secure packing.",
    "seller_service": "Concrete seller responsiveness, communication, assistance, support, or problem resolution; exclude general thanks or praise.",
    "accuracy_of_description": "Reviewer-stated agreement or mismatch with advertised size, color, variation, specifications, or features.",
}
FEATURE_COLUMNS = (
    "dim1_prob_auth", "dim2_prob_dec", "dim3_prob_liv", "dim4_prob_irr",
    "dim5_clip_sim", "dim6_star_rating",
)
QUALITY_ALIASES = {name: i for i, name in enumerate(CLASS_NAMES)}
QUALITY_ALIASES.update({name: 2 for name in (
    "vague", "low informational value", "low_informational_value", "low-value", "low_value",
)})


def get_inference_bundle():
    """Select the API bundle without changing the default training destination."""
    configured = os.getenv("AUTHENTICHECK_MODEL_BUNDLE", "").strip()
    bundle = Path(configured).expanduser() if configured else DEFAULT_BUNDLE
    if not bundle.is_absolute():
        bundle = PROJECT_ROOT / bundle
    return bundle.resolve()


def quality_input_text(record):
    """Use the same title + description + buyer review for every quality input.

    Missing listing context is allowed; the buyer review itself must be present.
    Only text fields are included, never labels, IDs, stars, or annotations.
    """
    parts = (
        record.get("product_title", record.get("prod_title", "")),
        record.get("product_description", record.get("prod_description", "")),
        record.get("review_text", record.get("text", "")),
    )
    cleaned = []
    for index, value in enumerate(parts):
        if value is None or (isinstance(value, float) and math.isnan(value)):
            value = ""
        if not isinstance(value, str):
            raise ValueError("Product title, description, and review text must be strings")
        value = " ".join(value.split())
        if index == 2 and not value:
            raise ValueError("Review text cannot be empty")
        if value:
            cleaned.append(value)
    return " ".join(cleaned)


def _label(value, names, aliases):
    normalized = str(value).strip().lower()
    if normalized in aliases:
        return aliases[normalized]
    try:
        number = float(normalized)
    except (ValueError, TypeError) as error:
        raise ValueError(f"Unknown label {value!r}; expected {list(names)}") from error
    if not math.isfinite(number) or not number.is_integer() or int(number) not in range(len(names)):
        raise ValueError(f"Invalid label {value!r}; expected {list(names)}")
    return int(number)


def map_quality(value):
    return _label(value, CLASS_NAMES, QUALITY_ALIASES)


def map_polarity(value):
    return _label(value, POLARITIES, {name.lower(): i for i, name in enumerate(POLARITIES)})


def normalize_rating(value):
    try:
        rating = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("A numeric star_rating from 1 to 5 is required.") from error
    if not math.isfinite(rating) or not 1 <= rating <= 5 or not rating.is_integer():
        raise ValueError(f"star_rating must be an integer from 1 to 5, received {value!r}")
    return (rating - 1) / 4


def parse_image_urls(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, (list, tuple)):
        values = value
    else:
        value = str(value).strip()
        if not value:
            return []
        if value.startswith("["):
            values = json.loads(value)
            if not isinstance(values, list):
                raise ValueError("Image URLs must be a JSON list or pipe-separated URLs")
        else:
            values = value.split("|")
    if any(not isinstance(item, str) for item in values):
        raise ValueError("Each image URL must be a string")
    return list(dict.fromkeys(item.strip() for item in values if item.strip()))


def aspect_prompt(category):
    return f"{category}: {ASPECT_DEFINITIONS[category]}"


def parse_annotations(value, text, product_category=""):
    """Keep each category/polarity and its evidence; metadata is not required.

    Evidence may paraphrase the review. Mixed category polarities remain separate.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        raise ValueError("aspect_annotations must be an explicit JSON list (use [] for none)")
    annotations = json.loads(value) if isinstance(value, str) else value
    if not isinstance(annotations, list):
        raise ValueError("aspect_annotations must be a JSON list")
    grouped = {}
    for annotation in annotations:
        if not isinstance(annotation, dict):
            raise ValueError("Each aspect annotation must be an object")
        category = annotation.get("category")
        if category not in ASPECTS:
            raise ValueError(f"Unknown aspect category {category!r}; expected {list(ASPECTS)}")
        evidence = annotation.get("text")
        if not isinstance(evidence, str) or not evidence.strip():
            raise ValueError("Annotation text must be a nonempty string")
        polarity = map_polarity(annotation.get("sentiment"))
        entry = grouped.setdefault((category, polarity), {"category": category, "sentiment": polarity, "evidence": []})
        if evidence not in entry["evidence"]:
            entry["evidence"].append(evidence)
    return [grouped[(category, polarity)] for category in ASPECTS for polarity in range(len(POLARITIES))
            if (category, polarity) in grouped]
