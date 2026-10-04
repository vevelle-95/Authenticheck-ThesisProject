"""Prepare, cross-fit, train, and validate our models. Final testing is separate."""

import argparse
from importlib.metadata import version
import json
import subprocess
import sys
from pathlib import Path

from model_contract import ASPECTS, BASE_MODEL, DEFAULT_BUNDLE, INPUT_VERSION, SENSORY_POLICY, SENTIMENT_TARGET_POLICY, TAXONOMY_VERSION
from model_data import load_reviews, prepare_splits

ROOT = Path(__file__).resolve().parent


def run_script(script, *arguments):
    subprocess.run([sys.executable, str(ROOT / script), *map(str, arguments)], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(ROOT / "data" / "test_reviews.csv"))
    parser.add_argument("--splits", default=str(ROOT / "data" / "splits.json"))
    parser.add_argument("--output", default=str(DEFAULT_BUNDLE))
    parser.add_argument("--work-dir", default=str(ROOT / "data" / "training_v2"))
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    args.data = str(Path(args.data).resolve())
    args.splits = str(Path(args.splits).resolve())
    args.output = str(Path(args.output).resolve())
    args.work_dir = str(Path(args.work_dir).resolve())
    if Path(args.base_model).is_dir():
        args.base_model = str(Path(args.base_model).resolve())
    frame = load_reviews(args.data, require_annotations=True)
    if args.validate_only:
        print(f"Validated {len(frame)} records; weights and partitions were not changed.")
        return
    assigned = prepare_splits(frame, args.splits)
    print(assigned.partition.value_counts().to_string(), flush=True)
    if args.prepare_only:
        return
    bundle, work = Path(args.output), Path(args.work_dir)
    common = ["--data", args.data, "--splits", args.splits]
    fitting = ["--epochs", args.epochs, "--batch-size", args.batch_size, "--max-length", args.max_length]
    run_script("stage1/generate_oof_features.py", *common, *fitting, "--base-model", args.base_model,
               "--output", work / "oof_probabilities.csv", "--fold-dir", bundle / "oof")
    run_script("stage1/train_roberta.py", *common, *fitting, "--base-model", args.base_model,
               "--output", bundle / "dost_roberta")
    run_script("stage1/extract_6d_features.py", *common, "--oof", work / "oof_probabilities.csv",
               "--model", bundle / "dost_roberta", "--output", work / "6d_features.csv")
    run_script("stage2/train_xgboost.py", *common, "--features", work / "6d_features.csv",
               "--output", bundle / "xgboost_meta_classifier.json")
    run_script("stage2/fine_tune_absa.py", *common, *fitting, "--encoder", args.base_model,
               "--output", bundle / "absa_model")
    bundle.mkdir(parents=True, exist_ok=True)
    (bundle / "manifest.json").write_text(json.dumps({
        "input_version": INPUT_VERSION, "taxonomy_version": TAXONOMY_VERSION,
        "sensory_domain_policy": SENSORY_POLICY,
        "sentiment_target_policy": SENTIMENT_TARGET_POLICY,
        "aspects": list(ASPECTS), "splits": str(Path(args.splits).resolve()),
        "test_evaluated": False,
        "model_version": "authenticheck-2.0",
        "environment": {name: version(name) for name in (
            "torch", "transformers", "datasets", "scikit-learn", "xgboost", "sentence-transformers",
        )},
    }, indent=2), encoding="utf-8")
    print(f"Training/validation complete: {bundle}. Run stage2/evaluate_models.py separately for final testing.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(f"ERROR: {error}") from error
