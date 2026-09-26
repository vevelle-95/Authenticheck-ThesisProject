# --- Model & Data Paths ---
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models" / "dost_roberta"
DATA_PATH = DATA_DIR / "test_reviews.csv"
FEATURES_PATH = DATA_DIR / "6d_features.csv"
BASE_MODEL = "dost-asti/RoBERTa-tl-cased"

# --- Data Processing ---
MAX_LENGTH = 128
TEST_SPLIT = 0.2

# --- Training Hyperparameters ---
LEARNING_RATE = 2e-5
BATCH_SIZE = 2
EPOCHS = 3