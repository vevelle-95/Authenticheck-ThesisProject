"""Evaluate sentiment-star mismatch with fold-held-out ABSA features; never score test."""

import gc
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_sample_weight
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, set_seed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from diagnostics.option_b_missing_image import image_score_records
from model_contract import (
    ASPECTS, BASE_MODEL, BATCH_SIZE, CLASS_NAMES, DEFAULT_DATA_PATH,
    FEATURE_COLUMNS, LEARNING_RATE, POLARITIES, SEED,
)
from model_data import fingerprint, load_reviews
from stage2.absa_model import ABSAHeadModel
from stage2.fine_tune_absa import batch_forward, build_targets

BUNDLE = ROOT / "models" / "retrain_real_v3"
FEATURES = ROOT / "data" / "retrain_real_v3" / "features" / "6d_features.csv"
OUTPUT = ROOT / "diagnostics" / "candidates"
PREDICTIONS_PATH = OUTPUT / "sentiment_star_mismatch_predictions.csv"
FEATURES_PATH = OUTPUT / "sentiment_star_mismatch_features.csv"
REPORT_PATH = OUTPUT / "sentiment_star_mismatch_report.json"
VARIANT_COLUMNS = {
    "mismatch_7d": (*FEATURE_COLUMNS, "dim7_sentiment_star_mismatch"),
    "mismatch_6d": (
        "dim1_prob_auth", "dim2_prob_dec", "dim3_prob_liv", "dim4_prob_irr",
        "dim5_sentiment_star_mismatch", "dim6_star_rating",
    ),
}


def product_macro_f1(frame, truth, prediction):
    scores = []
    for indices in frame.groupby("product_id", sort=False).indices.values():
        indices = np.asarray(indices)
        scores.append(float(f1_score(
            truth[indices], prediction[indices], labels=range(4),
            average="macro", zero_division=0,
        )))
    return float(np.mean(scores)) if scores else None


def metric_summary(frame, truth, prediction):
    if not len(truth):
        return None
    precision, recall, class_f1, support = precision_recall_fscore_support(
        truth, prediction, labels=range(4), zero_division=0,
    )
    authentic = truth == 0
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(class_f1)),
        "class_f1": {CLASS_NAMES[i]: float(class_f1[i]) for i in range(4)},
        "support": {CLASS_NAMES[i]: int(support[i]) for i in range(4)},
        "authentic_recall": float(np.mean(prediction[authentic] == 0)) if authentic.any() else None,
        "mean_product_macro_f1": product_macro_f1(frame, truth, prediction),
    }


def gain_by_feature(model):
    scores = model.get_booster().get_score(importance_type="gain")
    names = model.get_booster().feature_names or []
    return {name: float(scores.get(name, 0.0)) for name in names}


def make_xgb(settings):
    return xgb.XGBClassifier(
        **settings, random_state=42, eval_metric="mlogloss", n_jobs=2,
    )


def sentiment_score(model, texts, tokenizer, batch_size=16):
    aspect_probabilities = model.aspect_scores(texts, tokenizer, batch_size).numpy()
    thresholds = np.asarray([
        model.aspect_thresholds.get(name, model.threshold) for name in ASPECTS
    ])
    present = aspect_probabilities >= thresholds
    selected = np.argwhere(present)
    review_scores = np.full(len(texts), 0.5, dtype=np.float64)
    if not len(selected):
        return review_scores, int(len(texts))
    selected_texts = [texts[int(row)] for row, _ in selected]
    categories = [ASPECTS[int(aspect)] for _, aspect in selected]
    probabilities = np.asarray(model.predict_sentiment(
        selected_texts, categories, tokenizer, batch_size,
    ), dtype=np.float64)
    if probabilities.shape != (len(selected), len(POLARITIES)):
        raise ValueError("ABSA sentiment output does not match the three-polarity contract")
    totals = probabilities.sum(axis=1, keepdims=True)
    if not np.isfinite(probabilities).all() or (totals <= 0).any():
        raise ValueError("ABSA emitted invalid sigmoid probabilities")
    normalized = probabilities / totals
    aspect_scores = normalized[:, POLARITIES.index("Positive")] + (
        0.5 * normalized[:, POLARITIES.index("Neutral")]
    )
    per_review = [[] for _ in texts]
    for (row, _), score in zip(selected, aspect_scores):
        per_review[int(row)].append(float(score))
    for row, values in enumerate(per_review):
        if values:
            review_scores[row] = float(np.mean(values))
    return review_scores, int(sum(not values for values in per_review))


def fit_fold_absa(training, tokenizer, absa_config, epochs, fold):
    set_seed(SEED)
    dataset = build_targets(training, tokenizer, absa_config["max_length"])
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    encoder = AutoModel.from_pretrained(BASE_MODEL, local_files_only=True)
    model = ABSAHeadModel(
        BASE_MODEL, absa_config["dropout"], encoder=encoder,
        threshold=absa_config["threshold"],
        sentiment_threshold=absa_config["sentiment_threshold"],
        max_length=absa_config["max_length"],
        aspect_thresholds=absa_config.get("aspect_thresholds", {}),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    for epoch in range(epochs):
        model.train()
        losses = []
        for batch in loader:
            optimizer.zero_grad()
            aspect_logits, sentiment_logits, targets, sentiments, eligible = batch_forward(
                model, batch, device,
            )
            loss = model.compute_loss(
                aspect_logits, sentiment_logits, targets, sentiments, eligible,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        print(
            f"ABSA fold={fold} epoch={epoch + 1}/{epochs} "
            f"loss={float(np.mean(losses)):.4f}",
            flush=True,
        )
    return model


def ensure_fold_polarities(training):
    polarities = {
        int(annotation["sentiment"])
        for annotations in training.annotations
        for annotation in annotations
    }
    if polarities != {0, 1, 2}:
        raise ValueError(
            "A cross-fit ABSA training fold lacks one or more polarity labels"
        )


def main():
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metadata = json.loads(
        (BUNDLE / "xgboost_meta_classifier.metadata.json").read_text(encoding="utf-8")
    )
    absa_metadata = json.loads(
        (BUNDLE / "absa_model" / "training_metadata.json").read_text(encoding="utf-8")
    )
    absa_config = json.loads(
        (BUNDLE / "absa_model" / "absa_config.json").read_text(encoding="utf-8")
    )
    reviews = load_reviews(DEFAULT_DATA_PATH, require_annotations=True)
    if fingerprint(reviews) != metadata["dataset_sha256"]:
        raise ValueError("Dataset fingerprint differs from the saved Stage 1 bundle")
    train_ids = set(metadata["fit_review_ids"])
    validation_ids = set(metadata["validation_review_ids"])
    if train_ids & validation_ids or not train_ids.issubset(set(reviews.review_id)) or not validation_ids.issubset(set(reviews.review_id)):
        raise ValueError("Saved train/validation membership is invalid")
    rows = pd.read_csv(FEATURES, dtype={"review_id": str, "product_id": str})
    if set(rows.partition) != {"train", "validation"}:
        raise ValueError("Only train and validation features are allowed")
    if rows.review_id.duplicated().any() or set(rows.review_id) != train_ids | validation_ids:
        raise ValueError("Feature IDs do not exactly match the saved train/validation split")
    rows["has_image"] = rows.image_scores.map(
        lambda value: int(bool(image_score_records(value)))
    )
    assigned = reviews.loc[reviews.review_id.isin(train_ids | validation_ids)].copy()
    assigned["partition"] = np.where(
        assigned.review_id.isin(train_ids), "train", "validation",
    )
    fold_by_id = rows.set_index("review_id")["fold"]
    assigned["fold"] = assigned.review_id.map(fold_by_id)
    if assigned.groupby("product_id").partition.nunique().max() != 1:
        raise ValueError("Saved train/validation partitions split product groups")
    aligned = assigned.set_index("review_id").loc[rows.review_id].reset_index()
    for column in ("product_id", "partition", "ground_truth"):
        expected = "label" if column == "ground_truth" else column
        if not np.array_equal(
            rows[column].to_numpy(dtype=str),
            aligned[expected].to_numpy(dtype=str),
        ):
            raise ValueError(f"Cached and assigned {column} values differ")
    cached_folds = pd.to_numeric(rows["fold"], errors="coerce").fillna(-1).to_numpy(dtype=int)
    assigned_folds = pd.to_numeric(aligned["fold"], errors="coerce").fillna(-1).to_numpy(dtype=int)
    if not np.array_equal(cached_folds, assigned_folds):
        raise ValueError("Cached and assigned product fold IDs differ")
    if rows.loc[rows.partition.eq("train"), "probability_source"].str.startswith(
        "oof_fold_"
    ).sum() != int(rows.partition.eq("train").sum()):
        raise ValueError("Stage 1 training text probabilities are not all OOF")

    train_mask = rows.partition.eq("train").to_numpy()
    validation_mask = rows.partition.eq("validation").to_numpy()
    train_rows = rows.loc[train_mask].reset_index(drop=True)
    validation_rows = rows.loc[validation_mask].reset_index(drop=True)
    train_annotations = aligned.loc[train_mask].reset_index(drop=True)
    validation_annotations = aligned.loc[validation_mask].reset_index(drop=True)
    folds = sorted(train_rows.fold.astype(int).unique().tolist())
    if len(folds) != 5:
        raise ValueError(f"Expected five saved product folds, found {folds}")
    if train_rows.groupby("product_id").fold.nunique().max() != 1:
        raise ValueError("Saved ABSA folds split product groups")
    if set(train_rows.review_id) & set(validation_rows.review_id):
        raise ValueError("Train and validation IDs overlap")

    fit_ids = set(absa_metadata["fit_review_ids"])
    if fit_ids != set(train_annotations.loc[train_annotations.label.eq(0), "review_id"]):
        raise ValueError("Saved final ABSA checkpoint fit IDs differ from train Authentic rows")
    if fit_ids & set(validation_rows.review_id):
        raise ValueError("Validation IDs were used to fit the saved ABSA checkpoint")

    fold_training_frames = {}
    for fold in folds:
        held_ids = set(train_rows.loc[train_rows.fold.astype(int).eq(fold), "review_id"])
        fit = train_annotations.partition.eq("train") & ~train_annotations.review_id.isin(held_ids)
        fold_training = train_annotations.loc[fit & train_annotations.label.eq(0)].copy()
        if held_ids & set(fold_training.review_id):
            raise ValueError(f"Fold {fold} ABSA model would score a fit review")
        ensure_fold_polarities(fold_training)
        fold_training_frames[fold] = fold_training

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, local_files_only=True,
    )
    oof_scores = np.full(len(train_rows), np.nan, dtype=np.float64)
    empty_aspect_counts = {}
    for fold in folds:
        held = train_rows.fold.astype(int).eq(fold).to_numpy()
        fold_training = fold_training_frames[fold]
        model = fit_fold_absa(
            fold_training, tokenizer, absa_config, int(absa_metadata["best_epoch"]), fold,
        )
        held_indices = np.flatnonzero(held)
        held_texts = train_annotations.loc[held, "review_text"].tolist()
        scores, empty_count = sentiment_score(model, held_texts, tokenizer)
        oof_scores[held_indices] = scores
        empty_aspect_counts[str(fold)] = empty_count
        del model
        gc.collect()
        print(
            f"ABSA fold={fold} complete fit_authentic={len(fold_training)} "
            f"held_rows={len(held_indices)} no_predicted_aspect={empty_count}",
            flush=True,
        )
    if not np.isfinite(oof_scores).all():
        raise ValueError("Cross-fitted ABSA scores are incomplete")

    final_model = ABSAHeadModel.from_pretrained(BUNDLE / "absa_model").to(device)
    final_tokenizer = AutoTokenizer.from_pretrained(
        final_model.tokenizer_dir, local_files_only=True,
    )
    validation_texts = validation_annotations.review_text.tolist()
    validation_scores, validation_empty = sentiment_score(
        final_model, validation_texts, final_tokenizer,
    )
    if not np.isfinite(validation_scores).all():
        raise ValueError("Validation ABSA scores are incomplete")
    sentiment_scores = np.full(len(rows), np.nan, dtype=np.float64)
    sentiment_scores[train_mask] = oof_scores
    sentiment_scores[validation_mask] = validation_scores
    stars = rows["dim6_star_rating"].to_numpy(dtype=np.float64)
    if not np.isfinite(stars).all() or (stars < 0).any() or (stars > 1).any():
        raise ValueError("Normalized star ratings must be finite values in [0, 1]")
    mismatch = np.abs(sentiment_scores - stars)
    if not np.isfinite(mismatch).all() or (mismatch < 0).any() or (mismatch > 1).any():
        raise ValueError("Mismatch values must be finite values in [0, 1]")

    rows["sentiment_score"] = sentiment_scores
    rows["sentiment_star_mismatch"] = mismatch
    rows["dim7_sentiment_star_mismatch"] = mismatch
    rows["dim5_sentiment_star_mismatch"] = mismatch
    current_columns = list(FEATURE_COLUMNS)
    columns_7d = list(VARIANT_COLUMNS["mismatch_7d"])
    columns_6d = list(VARIANT_COLUMNS["mismatch_6d"])
    feature_matrices = {
        "current": rows[current_columns],
        "mismatch_7d": rows[columns_7d],
        "mismatch_6d": rows[columns_6d],
    }
    y_train = train_rows.ground_truth.to_numpy(dtype=int)
    y_validation = validation_rows.ground_truth.to_numpy(dtype=int)
    train_global = np.flatnonzero(train_mask)
    validation_global = np.flatnonzero(validation_mask)
    text_probabilities = rows[list(FEATURE_COLUMNS[:4])].to_numpy(dtype=np.float64)
    cv_predictions = {
        "text_only": text_probabilities[train_mask].argmax(axis=1),
        "current": np.full(len(train_rows), -1, dtype=int),
        "mismatch_7d": np.full(len(train_rows), -1, dtype=int),
        "mismatch_6d": np.full(len(train_rows), -1, dtype=int),
    }
    cv_gains = {name: [] for name in feature_matrices}
    settings = metadata["settings"]
    for fold in folds:
        held = train_rows.fold.astype(int).eq(fold).to_numpy()
        fit = ~held
        for name, matrix in feature_matrices.items():
            model = make_xgb(settings)
            model.fit(
                matrix.iloc[train_global[fit]], y_train[fit],
                sample_weight=compute_sample_weight("balanced", y_train[fit]),
            )
            cv_predictions[name][held] = model.predict(
                matrix.iloc[train_global[held]]
            ).astype(int)
            cv_gains[name].append(gain_by_feature(model))
    if any((prediction < 0).any() for prediction in cv_predictions.values()):
        raise ValueError("Grouped CV predictions are incomplete")

    validation_predictions = {
        "text_only": text_probabilities[validation_mask].argmax(axis=1),
    }
    current_model = xgb.XGBClassifier()
    current_model.load_model(BUNDLE / "xgboost_meta_classifier.json")
    validation_predictions["current"] = current_model.predict(
        pd.DataFrame(
            rows.loc[validation_mask, current_columns].to_numpy(),
            columns=current_columns,
        )
    ).astype(int)
    validation_models = {}
    validation_gains = {"current": gain_by_feature(current_model)}
    for name in ("mismatch_7d", "mismatch_6d"):
        model = make_xgb(settings)
        matrix = feature_matrices[name]
        model.fit(
            matrix.iloc[train_global], y_train,
            sample_weight=compute_sample_weight("balanced", y_train),
        )
        validation_predictions[name] = model.predict(
            matrix.iloc[validation_global]
        ).astype(int)
        validation_gains[name] = gain_by_feature(model)
        validation_models[name] = model

    metric_names = ("text_only", "current", "mismatch_7d", "mismatch_6d")
    cv_summary = {
        name: metric_summary(train_rows, y_train, cv_predictions[name])
        for name in metric_names
    }
    validation_summary = {
        name: metric_summary(
            validation_rows, y_validation, validation_predictions[name],
        )
        for name in metric_names
    }
    cv_by_image, validation_by_image = {}, {}
    for has_image, label in ((1, "has_image"), (0, "no_usable_image")):
        cv_subset = train_rows.has_image.to_numpy(dtype=int) == has_image
        val_subset = validation_rows.has_image.to_numpy(dtype=int) == has_image
        cv_by_image[label] = {
            name: metric_summary(
                train_rows.loc[cv_subset].reset_index(drop=True),
                y_train[cv_subset], cv_predictions[name][cv_subset],
            )
            for name in metric_names
        }
        validation_by_image[label] = {
            name: metric_summary(
                validation_rows.loc[val_subset].reset_index(drop=True),
                y_validation[val_subset], validation_predictions[name][val_subset],
            )
            for name in metric_names
        }

    gain_summary = {
        name: {
            "grouped_cv_mean_gain": {
                column: float(np.mean([fold_gain.get(column, 0.0) for fold_gain in fold_gains]))
                for column in feature_matrices[name].columns
            } if fold_gains else {},
            "validation_gain": validation_gains.get(name, {}),
        }
        for name, fold_gains in cv_gains.items()
    }
    gain_summary["current"] = {"validation_gain": validation_gains["current"]}
    reference = max(
        validation_summary["text_only"]["mean_product_macro_f1"],
        validation_summary["current"]["mean_product_macro_f1"],
    )
    candidates = [
        name for name in ("mismatch_6d", "mismatch_7d")
        if validation_summary[name]["mean_product_macro_f1"] > reference
    ]
    winner = max(
        candidates,
        key=lambda name: validation_summary[name]["mean_product_macro_f1"],
        default=None,
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    prediction_cache = rows[
        ["review_id", "product_id", "partition", "fold", "ground_truth",
         "probability_source", "has_image", "sentiment_score",
         "dim6_star_rating", "sentiment_star_mismatch"]
    ].copy()
    prediction_cache.to_csv(PREDICTIONS_PATH, index=False)
    rows.to_csv(FEATURES_PATH, index=False)
    report = {
        "status": "exploratory; no test rows scored; methodology approval still required",
        "dataset_sha256": metadata["dataset_sha256"],
        "feature_rows": {"train": len(train_rows), "validation": len(validation_rows)},
        "device": str(device),
        "absa_oof": {
            "folds": folds, "best_epoch": int(absa_metadata["best_epoch"]),
            "optimizer": "AdamW", "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE, "fit_only_on_train_authentic": True,
            "per_fold_no_predicted_aspect": empty_aspect_counts,
            "validation_no_predicted_aspect": validation_empty,
            "sentiment_aggregation": (
                "For predicted-present aspects, normalize the three ABSA sigmoid scores "
                "to sum to one; compute P(Positive)+0.5*P(Neutral); average by review. "
                "Use 0.5 if no aspects are predicted."
            ),
        },
        "grouped_cv_note": (
            "Product-grouped XGBoost CV over row-OOF ABSA features using the saved folds. "
            "This is not nested ABSA/XGBoost CV."
        ),
        "xgboost_settings": settings,
        "xgboost_weighting": metadata["xgboost_sample_weighting"],
        "cv": cv_summary,
        "cv_by_image": cv_by_image,
        "validation": validation_summary,
        "validation_by_image": validation_by_image,
        "feature_gain": gain_summary,
        "winner": winner,
        "winner_criterion": "validation mean product macro-F1 strictly exceeds both text-only and current XGBoost",
    }
    if winner is not None:
        model_path = OUTPUT / f"sentiment_star_{winner}_xgb.json"
        metadata_path = OUTPUT / f"sentiment_star_{winner}_xgb.metadata.json"
        validation_models[winner].save_model(model_path)
        metadata_path.write_text(json.dumps({
            "status": report["status"],
            "dataset_sha256": metadata["dataset_sha256"],
            "variant": winner,
            "feature_columns": list(VARIANT_COLUMNS[winner]),
            "settings": settings,
            "xgboost_weighting": metadata["xgboost_sample_weighting"],
            "sentiment_source": "fold-held-out Stage 2 ABSA; final saved ABSA checkpoint on validation",
            "sentiment_aggregation": report["absa_oof"]["sentiment_aggregation"],
            "validation_metrics": validation_summary[winner],
        }, indent=2), encoding="utf-8")
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("SENTIMENT-STAR MISMATCH — TRAIN/VALIDATION ONLY")
    print(f"rows: train={len(train_rows)} validation={len(validation_rows)}; folds={folds}")
    print("CV grouped products; metrics shown as macro-F1 / mean product macro-F1:")
    for name in metric_names:
        item = cv_summary[name]
        print(f"  {name}: {item['macro_f1']:.4f} / {item['mean_product_macro_f1']:.4f}")
    print("validation:")
    for name in metric_names:
        item = validation_summary[name]
        print(f"  {name}: {item['macro_f1']:.4f} / {item['mean_product_macro_f1']:.4f}")
    print(f"winner: {winner or 'none'}")
    print(f"saved: {REPORT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
