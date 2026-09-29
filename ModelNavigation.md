# AuthentiCheck: Navigation & Testing Guide

## 📂 Quick File Layout
- `data/` — Central folder for input CSVs and `6d_features.csv`
- `models/` — Central folder for saved model weights (`xgboost_meta_classifier.json`, etc.)
- `stage1/` — Feature extraction scripts (`extract_6d_features.py`, `train_roberta.py`, `config.py`)
- `stage2/` — Meta-classifier and ABSA scripts (`train_xgboost.py`, `fine_tune_absa.py`, `predict_absa.py`)

---

## 🚀 How to Run & Test

### 1. Setup Environment (Root Directory)
powershell:
    python -m venv .venv
    .venv\Scripts\Activate.ps1
    pip install -r requirements.txt

### 2. Run Stage 1
    cd stage1
    python train_roberta.py
    python extract_6d_features.py

### 3. Run Stage 2
    cd stage2
    python train_xgboost.py

    PA-ADD NALANG DITO @JRMAPS

### 4. Run ABSA (Open-Ended Aspect Extraction + Sentiment — Section 3.4, item 8)
    cd stage2
    python fine_tune_absa.py --data ../data/samples/absa_reviews.csv --output ../models/absa_model --authentic-only
    python predict_absa.py --data ../data/samples/absa_reviews.csv --model-dir ../models/absa_model --authentic-only
    # expected CSV (data/absa_reviews.csv):
    #   text, label_stage1, aspect_spans
    #   aspect_spans is JSON, for example:
    #   [{"text":"build quality","sentiment":2},{"text":"finish","sentiment":2}]
    #   sentiment values: 0=Negative, 1=Neutral, 2=Positive
    # existing fixed-aspect ABSA checkpoints must be retrained for the span format

### 5. Run Online Inference (Section 3.4, items 3-9; no dashboards/graphs yet)
    cd stage2
    python online_inference.py --data <new_reviews.csv>
    # input CSV: text, image_url, star_rating (optional product_id for per-product aggregates)
    # outputs JSON: classification verdicts (item 9.1), generated aspect phrases with sentiment (9.2), adjusted rating (9.3)
    # requires artifacts: models/dost_roberta, models/xgboost_meta_classifier.json, models/absa_model

### Quick-Test a Prediction via Terminal
    python -c "import xgboost as xgb, pandas as pd; model = xgb.XGBClassifier(); model.load_model('models/xgboost_meta_classifier.json'); print(model.predict_proba([[0.85, 0.05, 0.05, 0.05, 0.78, 5]]))"

    each number corresponds to a vector (authentic, deceptive, liv, irrelevant, clip score, star rating)

    output should also be an array of percentage corresponding to each of the classes (authentic, deceptive, liv, irrelevant)