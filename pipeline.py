import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
STAGE1 = ROOT / "stage1"
STAGE2 = ROOT / "stage2"


def run_script(folder, script, *arguments):
    command = [
        sys.executable,
        script,
        *arguments,
    ]

    subprocess.run(
        command,
        cwd=folder,
        check=True,
    )


def train_stage1():
    run_script(STAGE1, "train_roberta.py")
    run_script(STAGE1, "extract_6d_features.py")


def train_xgboost():
    run_script(STAGE2, "train_xgboost.py")


def train_absa():
    run_script(
        STAGE2,
        "fine_tune_absa.py",
        "--data",
        str(ROOT / "data" / "test_reviews.csv"),
        "--output",
        str(ROOT / "models" / "absa_model"),
        "--authentic-only",
    )


def train_all():
    train_stage1()
    train_xgboost()
    train_absa()

def main():
    print("Starting AuthentiCheck training pipeline...")
    print("1. Training Stage 1 and extracting features...")
    train_stage1()

    print("2. Training XGBoost...")
    train_xgboost()

    print("3. Training ABSA...")
    train_absa()

    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()