"""Shared, explicit projection of aspect phrases onto the thesis taxonomy."""

import re

CATEGORIES = (
    "aesthetics", "product_quality", "accuracy_of_description",
    "design", "value", "seller_service",
)

# These rules are a transparent adapter, not a trained aspect classifier.
TERMS = {
    "aesthetics": ("color", "colour", "finish", "look", "appearance", "style", "kulay", "ganda"),
    "product_quality": ("quality", "durable", "broken", "defect", "material", "performance", "battery", "sound", "matibay", "sira"),
    "accuracy_of_description": ("description", "listing", "advertised", "pictured", "as described", "wrong item", "not as shown", "iba sa", "mali ang item"),
    "design": ("design", "shape", "fit", "layout", "size", "ergonomic", "comfortable", "comfort", "sukat"),
    "value": ("price", "cost", "worth", "value", "expensive", "cheap", "presyo", "sulit", "mahal"),
    "seller_service": ("seller", "delivery", "shipping", "courier", "fulfillment", "fulfilment", "packaging", "parcel", "refund", "customer service", "late", "delayed", "dumating", "padala"),
}


def category_for(text):
    normalized = str(text).lower()
    matches = []
    for category, words in TERMS.items():
        for word in words:
            match = re.search(r"(?<!\w)" + re.escape(word) + r"(?!\w)", normalized)
            if match:
                matches.append((match.start(), category))
    return min(matches)[1] if matches else None


def normalize_sentiment(value):
    label = str(value).strip().lower()
    if label not in {"positive", "negative", "neutral"}:
        raise ValueError("Sentiment must be positive, negative, or neutral")
    return label
