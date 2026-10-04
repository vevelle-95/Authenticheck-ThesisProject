"""Cross-fit RoBERTa on five training folds, never on validation/test products."""

import argparse
import gc
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
from stage1 import config
from stage1.train_roberta import fit_roberta, predict_probabilities
from model_contract import FEATURE_COLUMNS, INPUT_VERSION, SEED
from model_data import fingerprint, load_reviews, read_splits


def generate_oof(assigned, output, fold_dir, *, base_model=config.BASE_MODEL,
                 epochs=config.EPOCHS, batch_size=config.BATCH_SIZE, max_length=config.MAX_LENGTH,
                 fit=fit_roberta, predict=predict_probabilities):
    training = assigned[assigned.partition.eq("train")].copy()
    if set(training.fold) != set(range(5)):
        raise ValueError("Five saved training folds are required")
    rows, provenance = [], []
    for fold in range(5):
        fitting = training[training.fold.ne(fold)]
        held = training[training.fold.eq(fold)]
        if set(fitting.product_id) & set(held.product_id):
            raise ValueError("Product overlap inside an OOF fold")
        print(f"OOF fold {fold + 1}/5: fitting {len(fitting)}, predicting {len(held)}", flush=True)
        model, tokenizer = fit(
            fitting, None, Path(fold_dir) / f"fold_{fold}", base_model=base_model,
            epochs=epochs, batch_size=batch_size, max_length=max_length, seed=SEED,
        )
        values = predict(model, tokenizer, held)
        if np.asarray(values).shape != (len(held), 4):
            raise ValueError("OOF predictor returned an unexpected shape")
        for record, probabilities in zip(held.to_dict("records"), values):
            rows.append({
                "review_id": record["review_id"], "product_id": record["product_id"], "fold": fold,
                "probability_source": f"oof_fold_{fold}",
                **dict(zip(FEATURE_COLUMNS[:4], map(float, probabilities))),
            })
        provenance.append({
            "fold": fold, "fit_review_ids": fitting.review_id.tolist(),
            "predicted_review_ids": held.review_id.tolist(),
            "fit_product_ids": sorted(set(fitting.product_id)),
            "predicted_product_ids": sorted(set(held.product_id)),
        })
        del model, tokenizer
        gc.collect()
    frame = pd.DataFrame(rows)
    if frame.review_id.duplicated().any() or set(frame.review_id) != set(training.review_id):
        raise ValueError("Every training review must receive exactly one OOF prediction")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    output.with_suffix(".metadata.json").write_text(json.dumps({
        "input_version": INPUT_VERSION, "dataset_sha256": fingerprint(assigned),
        "folds": provenance,
    }, indent=2), encoding="utf-8")
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(config.DATA_PATH))
    parser.add_argument("--splits", default=str(config.SPLITS_PATH))
    parser.add_argument("--output", default=str(config.DATA_DIR / "oof_probabilities.csv"))
    parser.add_argument("--fold-dir", default=str(config.MODEL_DIR.parent / "oof"))
    parser.add_argument("--base-model", default=config.BASE_MODEL)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=config.MAX_LENGTH)
    args = parser.parse_args()
    assigned = read_splits(load_reviews(args.data), args.splits)
    generate_oof(assigned, args.output, args.fold_dir, base_model=args.base_model,
                 epochs=args.epochs, batch_size=args.batch_size, max_length=args.max_length)


if __name__ == "__main__":
    main()
