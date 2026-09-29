from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

MODEL_DIR = PROJECT_ROOT / "models" / "dost_roberta"
DATA_PATH = DATA_DIR / "test_reviews.csv"
FEATURES_PATH = DATA_DIR / "6d_features.csv"

BASE_MODEL = "dost-asti/RoBERTa-tl-cased"

TEXT_LABEL_COLUMN = "text_label"
GROUND_TRUTH_COLUMN = "ground_truth"
PRODUCT_DESCRIPTION_COLUMN = "product_description"
REVIEW_IMAGE_URLS_COLUMN = "review_image_urls"

MAX_LENGTH = 128
TEST_SPLIT = 0.2

LEARNING_RATE = 2e-5
BATCH_SIZE = 2
EPOCHS = 3

LABEL_MAP = {
    "authentic": 0,
    "deceptive": 1,
    "vague": 2,
    "low informational value": 2, #just in case lang
    "low_informational_value": 2, #just in case lang
    "low-value": 2, #just in case lang
    "low_value": 2, #just in case lang
    "irrelevant": 3,
}


def map_label(value):
    if value is None:
        raise ValueError("Label cannot be empty.")

    normalized = str(value).strip().lower()

    if normalized in LABEL_MAP:
        return LABEL_MAP[normalized]

    try:
        numeric = int(float(normalized))
    except ValueError as error:
        raise ValueError(
            f"Unknown label {value!r}. "
            f"Expected one of: {sorted(LABEL_MAP)}"
        ) from error

    if numeric not in range(4):
        raise ValueError(
            f"Numeric label must be 0, 1, 2, or 3: {value!r}"
        )

    return numeric