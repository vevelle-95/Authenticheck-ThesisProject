"""Our model defaults; validation/test membership lives in one saved manifest."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_contract import BASE_MODEL, DATA_DIR, DEFAULT_BUNDLE, MAX_LENGTH, QUALITY_ALIASES, map_quality

MODEL_DIR = DEFAULT_BUNDLE / "dost_roberta"
DATA_PATH = DATA_DIR / "test_reviews.csv"
SPLITS_PATH = DATA_DIR / "splits.json"
FEATURES_PATH = DATA_DIR / "6d_features.csv"
GROUND_TRUTH_COLUMN = "ground_truth"
TEXT_LABEL_COLUMN = "text_label"  # Source metadata; never a training target.
PRODUCT_TITLE_COLUMN = "product_title"  # Quality model and CLIP context.
PRODUCT_DESCRIPTION_COLUMN = "product_description"  # Quality model and CLIP context.
REVIEW_IMAGE_URLS_COLUMN = "review_image_urls"
TRAIN_FRACTION, VALIDATION_FRACTION, TEST_FRACTION = 0.70, 0.15, 0.15
OOF_FOLDS = 5
LEARNING_RATE = 2e-5
BATCH_SIZE = 2
EPOCHS = 3
LABEL_MAP = QUALITY_ALIASES
map_label = map_quality
