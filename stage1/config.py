# --- Model & Data Paths ---
DATA_PATH = "data/test_reviews.csv"
MODEL_DIR = "models/dost_roberta"
BASE_MODEL = "dost-asti/RoBERTa-tl-cased"

# --- Data Processing ---
MAX_LENGTH = 128
TEST_SPLIT = 0.2

# --- Training Hyperparameters ---
LEARNING_RATE = 2e-5
BATCH_SIZE = 2
EPOCHS = 3