"""Prepare, cross-fit, train, and validate our models. Final testing is separate."""

import argparse
from importlib.metadata import version
import json
import subprocess
import sys
from pathlib import Path

from model_contract import (
    ASPECTS, BASE_MODEL, BATCH_SIZE, DEFAULT_BUNDLE, DEFAULT_DATA_PATH,
    DEFAULT_SPLITS_PATH, DEFAULT_WORK_DIR, EPOCHS, INPUT_VERSION, MAX_LENGTH,
    MODEL_VERSION, SENSORY_POLICY, SENTIMENT_TARGET_POLICY, TAXONOMY_VERSION,
)
from model_data import load_experiment, load_reviews, prepare_splits
from training_augmentation import augmentation_summary

ROOT = Path(__file__).resolve().parent


def run_script(script, *arguments):
    subprocess.run([sys.executable, str(ROOT / script), *map(str, arguments)], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--splits", default=str(DEFAULT_SPLITS_PATH))
    parser.add_argument("--augmentations", help="Optional approved training-only augmentation CSV")
    parser.add_argument("--allow-unreviewed-augmentations", action="store_true", help="Experimental run: include pending drafts without marking them approved")
    parser.add_argument("--output", default=str(DEFAULT_BUNDLE))
    parser.add_argument("--work-dir", default=str(DEFAULT_WORK_DIR))
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.allow_unreviewed_augmentations and not args.augmentations:
        parser.error("--allow-unreviewed-augmentations requires --augmentations")
    args.data = str(Path(args.data).resolve())
    args.splits = str(Path(args.splits).resolve())
    if args.augmentations:
        args.augmentations = str(Path(args.augmentations).resolve())
    args.output = str(Path(args.output).resolve())
    args.work_dir = str(Path(args.work_dir).resolve())
    if Path(args.base_model).is_dir():
        args.base_model = str(Path(args.base_model).resolve())
    frame = load_reviews(args.data, require_annotations=True)
    if args.validate_only:
        if args.augmentations:
            frame = load_experiment(args.data, args.splits, args.augmentations, require_annotations=True,
                                    allow_unreviewed=args.allow_unreviewed_augmentations)
        print("Augmentation usage:", json.dumps(augmentation_summary(frame)))
        print(f"Validated {len(frame)} records; weights and partitions were not changed.")
        return
    assigned = prepare_splits(frame, args.splits)
    if args.augmentations:
        assigned = load_experiment(args.data, args.splits, args.augmentations, require_annotations=True,
                                   allow_unreviewed=args.allow_unreviewed_augmentations)
    print("Augmentation usage:", json.dumps(augmentation_summary(assigned)), flush=True)
    print(assigned.partition.value_counts().to_string(), flush=True)
    if args.prepare_only:
        return
    bundle, work = Path(args.output), Path(args.work_dir)
    common = ["--data", args.data, "--splits", args.splits]
    if args.augmentations:
        common += ["--augmentations", args.augmentations]
    if args.allow_unreviewed_augmentations:
        common += ["--allow-unreviewed-augmentations"]
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
        "augmentations": args.augmentations,
        "augmentation_policy": ("experimental-unreviewed-training-only-source-product-and-fold-v1"
                                if args.allow_unreviewed_augmentations else "approved-training-only-source-product-and-fold-v1"),
        "augmentation_usage": augmentation_summary(assigned),
        "model_version": MODEL_VERSION,
        "quality_text_input": "product_title + product_description + review_text",
        "absa_text_input": "review_text",
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
