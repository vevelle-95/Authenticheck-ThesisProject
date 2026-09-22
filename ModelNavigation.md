# AuthentiCheck: Navigation & Testing Guide

## 📂 Quick File Layout
- `data/` — Central folder for input CSVs and `6d_features.csv`
- `models/` — Central folder for saved model weights (`xgboost_meta_classifier.json`, etc.)
- `stage1/` — Feature extraction scripts (`extract_6d_features.py`, `train_roberta.py`, `config.py`)
- `stage2/` — Meta-classifier script (`train_xgboost.py`)

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

### Quick-Test a Prediction via Terminal
    python -c "import xgboost as xgb, pandas as pd; model = xgb.XGBClassifier(); model.load_model('models/xgboost_meta_classifier.json'); print(model.predict_proba([[0.85, 0.05, 0.05, 0.05, 0.78, 5]]))"

    each number corresponds to a vector (authentic, deceptive, liv, irrelevant, clip score, star rating)

    output should also be an array of percentage corresponding to each of the classes (authentic, deceptive, liv, irrelevant)