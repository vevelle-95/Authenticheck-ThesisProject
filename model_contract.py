"""AuthentiCheck settings and contracts shared by training and inference."""

import json
import math
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_DATA_PATH = DATA_DIR / "test_reviews.csv"
DEFAULT_SPLITS_PATH = DATA_DIR / "splits_dataset.json"
DEFAULT_BUNDLE = PROJECT_ROOT / "models" / "own_model_v2"
DEFAULT_ROBERTA_DIR = DEFAULT_BUNDLE / "dost_roberta"
DEFAULT_OOF_PATH = DATA_DIR / "oof_probabilities.csv"
DEFAULT_FEATURES_PATH = DATA_DIR / "6d_features.csv"
DEFAULT_WORK_DIR = DATA_DIR / "training_v2"

# Training defaults: CLI options override these for an individual experiment.
BASE_MODEL = "dost-asti/RoBERTa-tl-cased"
EPOCHS = 3
BATCH_SIZE = 2
LEARNING_RATE = 2e-5
MAX_LENGTH = 128
SEED = 42

# Shared label, feature, and artifact compatibility contracts.
INPUT_VERSION = "product-context-six-features-v5-text-target"
ROBERTA_TARGET_COLUMN = "text_label"
QUALITY_TEXT_POLICY = "review-first-bounded-context-v1"
QUALITY_CONTEXT_TOKENS = 28
QUALITY_TITLE_TOKENS = 10
MODEL_VERSION = "authenticheck-2.2"
TAXONOMY_VERSION = "fixed-ten-v1"
SENSORY_POLICY = "annotation-guidelines-only"
SENTIMENT_TARGET_POLICY = "category-polarity-multilabel-v2"
ABSA_EVALUATION_PROTOCOL = "ten-aspect-multilabel-v2"
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
    """Keep field boundaries so tokenization can protect the buyer review.

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
        cleaned.append(value)
    title, description, review = cleaned
    return f"Review: {review}\nProduct title: {title}\nProduct description: {description}"


def quality_input_parts(text):
    """Read our three-field format; plain strings remain review-only inputs."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Review text cannot be empty")
    lines = text.splitlines()
    prefixes = ("Review: ", "Product title: ", "Product description: ")
    if len(lines) == 3 and all(line.startswith(prefix) for line, prefix in zip(lines, prefixes)):
        review, title, description = [line[len(prefix):] for line, prefix in zip(lines, prefixes)]
    else:
        review, title, description = " ".join(text.split()), "", ""
    if not review.strip():
        raise ValueError("Review text cannot be empty")
    return review, title, description


def _token_count(tokenizer, text):
    return len(tokenizer(text, add_special_tokens=False, truncation=False, verbose=False)["input_ids"])


def _token_prefix(tokenizer, text, limit):
    """Keep original characters up to a token boundary, without decode changes."""
    if not text or limit <= 0:
        return ""
    encoded = tokenizer(text, add_special_tokens=False, truncation=True, max_length=limit,
                        return_offsets_mapping=True, return_attention_mask=False,
                        return_token_type_ids=False)
    end = max((end for _, end in encoded["offset_mapping"]), default=0)
    prefix = text[:end].rstrip()
    # Cutting inside a word can change its BPE segmentation. Recheck the prefix.
    while prefix and _token_count(tokenizer, prefix) > limit:
        prefix = prefix[:-1].rstrip()
    return prefix


def quality_input_pairs(tokenizer, texts, max_length=MAX_LENGTH):
    """Bound context before encoding (review, context) pairs with either encoder.

    At 128 tokens RoBERTa reserves at least 96 review tokens, up to 28 context
    tokens and four pair separators. Shorter context frees more review space.
    Each encoder's own tokenizer counts tokens. No listing can remove the review.
    """
    if not isinstance(max_length, int) or max_length < 8:
        raise ValueError("Quality max_length must be at least 8")
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("Review-preserving tokenization requires a fast tokenizer with character offsets")
    tokenizer.truncation_side = "right"
    content_limit = max_length - tokenizer.num_special_tokens_to_add(pair=True)
    if content_limit < 3:
        raise ValueError("Quality token budget must fit review and product context")
    context_limit = min(QUALITY_CONTEXT_TOKENS, max(2, content_limit // 4), content_limit - 1)
    title_limit = min(QUALITY_TITLE_TOKENS, max(1, context_limit * QUALITY_TITLE_TOKENS // QUALITY_CONTEXT_TOKENS))
    pairs = []
    for text in texts:
        review, title, description = quality_input_parts(text)
        title = _token_prefix(tokenizer, title, title_limit)
        description = _token_prefix(tokenizer, description, context_limit - _token_count(tokenizer, title))
        context = _token_prefix(tokenizer, " ".join(part for part in (title, description) if part), context_limit)
        review = _token_prefix(tokenizer, review, content_limit - _token_count(tokenizer, context))
        if not review:
            raise ValueError("Tokenization removed the review; inspect the review/tokenizer")
        pairs.append((review, context))
    return pairs


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


def sentiment_indices(probabilities, threshold=0.5):
    """Keep every supported polarity; a present aspect gets at least one label."""
    if not 0 < threshold < 1:
        raise ValueError("Sentiment threshold must lie strictly between 0 and 1")
    if len(probabilities) != len(POLARITIES) or any(
            not math.isfinite(float(value)) or not 0 <= value <= 1 for value in probabilities):
        raise ValueError("Expected three finite sentiment probabilities from 0 to 1")
    selected = [index for index, value in enumerate(probabilities) if value >= threshold]
    return selected or [max(range(len(POLARITIES)), key=lambda index: probabilities[index])]


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
