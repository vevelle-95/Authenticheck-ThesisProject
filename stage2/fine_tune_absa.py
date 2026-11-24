"""Train independent fixed-category ABSA; choose checkpoint/threshold on validation."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
from sklearn.metrics import f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, set_seed

from stage2 import absa_model
from model_contract import (
    ASPECTS, BASE_MODEL, BATCH_SIZE, DEFAULT_DATA_PATH, DEFAULT_SPLITS_PATH,
    EPOCHS, LEARNING_RATE, SEED, SENSORY_POLICY, SENTIMENT_TARGET_POLICY, sentiment_indices,
)
from model_data import fingerprint, load_experiment
from training_augmentation import augmentation_summary
from model_metrics import multilabel_metrics, select_sentiment_threshold, sentiment_predictions
from model_contract import POLARITIES

MIN_ASPECT_VALIDATION_POSITIVES = 15
MIN_ASPECT_VALIDATION_NEGATIVES = 15
MIN_ASPECT_VALIDATION_PRODUCTS = 3
MIN_ASPECT_POSITIVES_PER_PRODUCT = 3
MIN_ASPECT_F1_GAIN = 0.05


def build_targets(frame, tokenizer, max_length):
    if frame.empty:
        raise ValueError("ABSA needs nonempty ground-truth Authentic training and validation partitions")
    texts = frame.review_text.tolist()
    enc = absa_model.tokenize_detection(tokenizer, texts, max_length)
    targets = torch.zeros(len(frame), len(ASPECTS))
    sentiments = torch.zeros(len(frame), len(ASPECTS), 3)
    eligible = torch.ones_like(targets)
    pair_texts = [text for text in texts for _ in ASPECTS]
    pair_categories = list(ASPECTS) * len(texts)
    pair_enc = absa_model.tokenize_sentiment(tokenizer, pair_texts, pair_categories, max_length)
    for index, row in enumerate(frame.to_dict("records")):
        for annotation in row["annotations"]:
            category = ASPECTS.index(annotation["category"])
            targets[index, category] = 1
            sentiments[index, category, annotation["sentiment"]] = 1
    return TensorDataset(
        enc["input_ids"], enc["attention_mask"], targets,
        pair_enc["input_ids"].reshape(len(frame), len(ASPECTS), -1),
        pair_enc["attention_mask"].reshape(len(frame), len(ASPECTS), -1), sentiments, eligible,
    )


def batch_forward(model, batch, device):
    ids, mask, targets, pair_ids, pair_mask, sentiments, eligible = [tensor.to(device) for tensor in batch]
    active = sentiments.sum(-1).gt(0)
    aspect_logits, sentiment_logits = model(ids, mask, pair_ids[active], pair_mask[active])
    return aspect_logits, sentiment_logits, targets, sentiments[active], eligible


def detection_metrics(targets, probabilities, eligible, threshold, aspect_thresholds=None):
    cutoffs = np.asarray([aspect_thresholds.get(category, threshold) for category in ASPECTS]
                         if aspect_thresholds else [threshold] * len(ASPECTS))
    predictions = (probabilities >= cutoffs) & eligible.astype(bool)
    precision, recall, f1, support = precision_recall_fscore_support(
        targets.astype(int), predictions.astype(int), average=None, zero_division=0,
    )
    return {
        "macro_precision": float(precision.mean()), "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()), "support": dict(zip(ASPECTS, map(int, support))),
    }


def select_threshold(targets, probabilities, eligible):
    candidates = [round(value, 2) for value in np.arange(0.1, 0.91, 0.05)]
    return max(candidates, key=lambda value: (
        detection_metrics(targets, probabilities, eligible, value)["macro_f1"], -abs(value - 0.5),
    ))


def select_aspect_thresholds(targets, probabilities, eligible, default, product_ids=None):
    """Use an aspect override only with enough validation support and a real F1 gain."""
    candidates = [round(value, 2) for value in np.arange(0.1, 0.91, 0.05)]
    selected = {}
    for index, category in enumerate(ASPECTS):
        active = eligible[:, index].astype(bool)
        truth = targets[active, index].astype(int)
        scores = probabilities[active, index]
        positives = int(truth.sum())
        if (positives < MIN_ASPECT_VALIDATION_POSITIVES or
                len(truth) - positives < MIN_ASPECT_VALIDATION_NEGATIVES):
            continue
        if product_ids is not None:
            active_products = np.asarray(product_ids)[active]
            positive_products = {product for product in active_products[truth.astype(bool)]}
            if sum(int(np.sum((active_products == product) & truth.astype(bool))) >=
                   MIN_ASPECT_POSITIVES_PER_PRODUCT for product in positive_products) < MIN_ASPECT_VALIDATION_PRODUCTS:
                continue
        baseline = f1_score(truth, scores >= default, zero_division=0)
        best = max(candidates, key=lambda value: (
            f1_score(truth, scores >= value, zero_division=0), -abs(value - default),
        ))
        if product_ids is not None:
            loses_product = any(
                np.any(scores[active_products == product] >= default) and
                not np.any(scores[active_products == product] >= best)
                for product in positive_products
            )
            if loses_product:
                continue
        if f1_score(truth, scores >= best, zero_division=0) >= baseline + MIN_ASPECT_F1_GAIN:
            selected[category] = float(best)
    return selected


@torch.inference_mode()
def evaluate_loader(model, loader, device, tune=False, fp16=False, product_ids=None):
    model.eval()
    target_rows, probabilities, eligible_rows = [], [], []
    sentiment_targets, sentiment_probabilities, joint_targets = [], [], []
    for batch in loader:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=fp16):
            logits, sentiments, targets, polarities, eligible = batch_forward(model, batch, device)
        target_rows.extend(targets.cpu().tolist())
        probabilities.extend(torch.sigmoid(logits).cpu().tolist())
        eligible_rows.extend(eligible.cpu().tolist())
        joint_targets.extend(batch[5].tolist())
        if sentiments is not None:
            sentiment_targets.extend(polarities.cpu().tolist())
            sentiment_probabilities.extend(torch.sigmoid(sentiments).float().cpu().tolist())
    targets, probabilities, eligible = map(np.asarray, (target_rows, probabilities, eligible_rows))
    if product_ids is not None and len(product_ids) != len(targets):
        raise ValueError("Validation product IDs must match the loader order")
    threshold = select_threshold(targets, probabilities, eligible) if tune else model.threshold
    aspect_thresholds = (select_aspect_thresholds(targets, probabilities, eligible, threshold, product_ids)
                         if tune else model.aspect_thresholds)
    sentiment_threshold = select_sentiment_threshold(sentiment_targets, sentiment_probabilities) if tune else model.sentiment_threshold
    cutoffs = np.asarray([aspect_thresholds.get(category, threshold) for category in ASPECTS])
    detected = (probabilities >= cutoffs) & eligible.astype(bool)
    joint_predictions = np.zeros_like(np.asarray(joint_targets), dtype=int)
    cursor = 0
    for batch in loader:
        count = len(batch[0])
        selected = np.flatnonzero(detected[cursor:cursor + count].reshape(-1))
        if len(selected):
            pair_ids = batch[3].flatten(0, 1)[selected].to(device)
            pair_mask = batch[4].flatten(0, 1)[selected].to(device)
            values = []
            for ids, mask in zip(pair_ids.split(8), pair_mask.split(8)):
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=fp16):
                    logits = model.sentiment_head(model._pooled(ids, mask))
                values.extend(torch.sigmoid(logits).float().cpu().tolist())
            for flat_index, scores in zip(selected, values):
                row, aspect = divmod(int(flat_index), len(ASPECTS))
                joint_predictions[cursor + row, aspect, sentiment_indices(scores, sentiment_threshold)] = 1
        cursor += count
    if cursor != len(targets):
        raise ValueError("Validation loader changed order between detection and sentiment passes")
    pair_labels = [f"{category}/{polarity.lower()}" for category in ASPECTS for polarity in POLARITIES]
    coverage = None
    if product_ids is not None:
        gold_product_aspects = {(product_ids[row], ASPECTS[index])
                                for row, index in zip(*np.where(targets.astype(bool)))}
        predicted_product_aspects = {(product_ids[row], ASPECTS[index])
                                     for row, index in zip(*np.where(detected))}
        coverage = {"gold_product_aspect_count": len(gold_product_aspects),
                    "predicted_gold_product_aspect_count": len(gold_product_aspects & predicted_product_aspects)}
    return {
        "threshold": threshold,
        "aspect_thresholds": aspect_thresholds,
        "sentiment_threshold": sentiment_threshold,
        "detection": detection_metrics(targets, probabilities, eligible, threshold, aspect_thresholds),
        "sentiment": multilabel_metrics(sentiment_targets,
            sentiment_predictions(sentiment_probabilities, sentiment_threshold), POLARITIES),
        "aspect_sentiment_pairs": multilabel_metrics(
            np.asarray(joint_targets).reshape(-1, len(pair_labels)),
            joint_predictions.reshape(-1, len(pair_labels)), pair_labels),
        "product_aspect_coverage": coverage,
    }


def train_absa(assigned, output, *, encoder=BASE_MODEL, epochs=EPOCHS, batch_size=BATCH_SIZE,
               learning_rate=LEARNING_RATE, max_length=absa_model.ABSA_MAX_LENGTH, low_memory=False):
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    training = assigned[assigned.partition.eq("train") & assigned.label.eq(0)]
    validation = assigned[assigned.partition.eq("validation") & assigned.label.eq(0)]
    polarities = {annotation["sentiment"] for annotations in training.annotations for annotation in annotations}
    if polarities != {0, 1, 2}:
        raise ValueError("ABSA fitting requires Authentic training examples for Negative, Neutral, and Positive; validate the fixture without training instead")
    set_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(encoder)
    train_dataset = build_targets(training, tokenizer, max_length)
    val_dataset = build_targets(validation, tokenizer, max_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = absa_model.ABSAHeadModel(encoder, max_length=max_length).to(device)
    fp16 = low_memory and device.type == "cuda"
    if low_memory:
        model.encoder.config.use_cache = False
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        from transformers import Adafactor
        optimizer = Adafactor(model.parameters(), lr=learning_rate, relative_step=False,
                              scale_parameter=False, warmup_init=False)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=fp16)
    history, best = [], float("-inf")
    for epoch in range(epochs):
        model.train()
        losses = []
        print(f"ABSA epoch {epoch + 1}/{epochs}: {len(train_loader)} batches", flush=True)
        for step, batch in enumerate(train_loader, 1):
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=fp16):
                logits, sentiments, targets, polarities, eligible = batch_forward(model, batch, device)
                loss = model.compute_loss(logits, sentiments, targets, polarities, eligible)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
            if step == 1 or step % 50 == 0 or step == len(train_loader):
                print(f"ABSA epoch {epoch + 1}/{epochs}: batch {step}/{len(train_loader)}, loss={losses[-1]:.4f}", flush=True)
        metrics = evaluate_loader(model, val_loader, device, tune=True, fp16=fp16,
                                  product_ids=validation.product_id.tolist())
        sentiment_f1 = metrics["sentiment"]["macro_f1"]
        if sentiment_f1 is None:
            raise ValueError("Validation needs annotated Authentic review/category polarity instances")
        score = metrics["aspect_sentiment_pairs"]["macro_f1"]
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), **metrics})
        print(json.dumps({"epoch": epoch + 1, "loss": history[-1]["loss"],
                          "detection_macro_f1": metrics["detection"]["macro_f1"],
                          "pair_macro_f1": score,
                          "product_aspect_coverage": metrics["product_aspect_coverage"],
                          "aspect_thresholds": metrics["aspect_thresholds"]}), flush=True)
        if score > best:
            best = score
            model.threshold = metrics["threshold"]
            model.aspect_thresholds = metrics["aspect_thresholds"]
            model.sentiment_threshold = metrics["sentiment_threshold"]
            model.save_pretrained(output, tokenizer)
            best_epoch = epoch + 1
            best_metrics = metrics
    Path(output, "training_metadata.json").write_text(json.dumps({
        "augmentation_usage": augmentation_summary(training),
        "dataset_sha256": fingerprint(assigned), "seed": SEED,
        "low_memory_training": low_memory, "fp16": fp16,
        "optimizer": "adafactor" if low_memory else "adamw",
        "sensory_domain_policy": SENSORY_POLICY,
        "sentiment_target_policy": SENTIMENT_TARGET_POLICY,
        "fit_review_ids": training.review_id.tolist(), "validation_review_ids": validation.review_id.tolist(),
        "best_epoch": best_epoch, "selection_metric": "validation_aspect_sentiment_pair_macro_f1",
        "aspect_threshold_calibration": {
            "min_positive_support": MIN_ASPECT_VALIDATION_POSITIVES,
            "min_negative_support": MIN_ASPECT_VALIDATION_NEGATIVES,
            "min_supported_products": MIN_ASPECT_VALIDATION_PRODUCTS,
            "min_positives_per_product": MIN_ASPECT_POSITIVES_PER_PRODUCT,
            "min_f1_gain": MIN_ASPECT_F1_GAIN,
        },
        "validation_metrics": best_metrics, "history": history,
        "aggregation_unit": "one_review_category_polarity_group",
    }, indent=2), encoding="utf-8")
    return best_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--splits", default=str(DEFAULT_SPLITS_PATH))
    parser.add_argument("--augmentations", help="Approved training-only augmentation CSV; use the same file throughout the experiment")
    parser.add_argument("--allow-unreviewed-augmentations", action="store_true", help="Experimental run: include pending drafts without marking them approved")
    parser.add_argument("--encoder", default=BASE_MODEL, help="Independent pretrained encoder, not the Stage 1 classifier")
    parser.add_argument("--output", default=str(absa_model.DEFAULT_ABSA_MODEL_DIR))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=absa_model.ABSA_MAX_LENGTH)
    parser.add_argument("--authentic-only", action="store_true", help="Accepted for compatibility; ABSA always trains on ground-truth Authentic reviews")
    parser.add_argument("--eval-only", action="store_true", help="Evaluate validation only; final testing uses evaluate_models.py")
    parser.add_argument("--calibrate-output", help="Save a separately calibrated copy of --output using validation only")
    parser.add_argument("--low-memory", action="store_true")
    parser.add_argument("--gpu-memory-gb", type=float)
    args = parser.parse_args()
    from gpu_memory import configure_gpu_memory
    configure_gpu_memory(args.gpu_memory_gb)
    assigned = load_experiment(args.data, args.splits, args.augmentations, require_annotations=True,
                               allow_unreviewed=args.allow_unreviewed_augmentations)
    if args.eval_only and args.calibrate_output:
        parser.error("--eval-only and --calibrate-output are separate operations")
    if args.eval_only or args.calibrate_output:
        model = absa_model.ABSAHeadModel.from_pretrained(args.output)
        tokenizer = AutoTokenizer.from_pretrained(model.tokenizer_dir, local_files_only=True)
        validation = assigned[assigned.partition.eq("validation") & assigned.label.eq(0)]
        loader = DataLoader(build_targets(validation, tokenizer, model.max_length), batch_size=args.batch_size)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        before = evaluate_loader(model, loader, device, product_ids=validation.product_id.tolist())
        if args.eval_only:
            print(json.dumps(before, indent=2))
            return
        source_metadata = json.loads((Path(args.output) / "training_metadata.json").read_text(encoding="utf-8"))
        if source_metadata.get("dataset_sha256") != fingerprint(assigned):
            raise ValueError("Saved ABSA model and current dataset/splits do not match")
        after = evaluate_loader(model, loader, device, tune=True,
                                product_ids=validation.product_id.tolist())
        before_score = before["aspect_sentiment_pairs"]["macro_f1"]
        after_score = after["aspect_sentiment_pairs"]["macro_f1"]
        print(json.dumps({"saved_joint_macro_f1": before_score, "calibrated_joint_macro_f1": after_score,
                          "saved_product_aspect_coverage": before["product_aspect_coverage"],
                          "calibrated_product_aspect_coverage": after["product_aspect_coverage"],
                          "global_threshold": after["threshold"], "aspect_thresholds": after["aspect_thresholds"],
                          "sentiment_threshold": after["sentiment_threshold"]}, indent=2))
        if (after_score <= before_score or
                after["product_aspect_coverage"]["predicted_gold_product_aspect_count"] <
                before["product_aspect_coverage"]["predicted_gold_product_aspect_count"]):
            print("Validation pair F1 or product-aspect coverage did not improve; no calibrated checkpoint was saved")
            return
        destination = Path(args.calibrate_output)
        if destination.exists():
            raise FileExistsError(f"Calibration destination already exists: {destination}")
        model.threshold = float(after["threshold"])
        model.aspect_thresholds = model._validate_aspect_thresholds(after["aspect_thresholds"])
        model.sentiment_threshold = float(after["sentiment_threshold"])
        model.save_pretrained(destination, tokenizer)
        source_metadata.update({
            "calibration_source": str(Path(args.output).resolve()),
            "calibration_selection_metric": "validation_aspect_sentiment_pair_macro_f1",
            "calibration_before_joint_macro_f1": before_score,
            "validation_metrics": after,
        })
        (destination / "training_metadata.json").write_text(
            json.dumps(source_metadata, indent=2), encoding="utf-8")
        print(f"Saved validation-calibrated ABSA checkpoint to {destination}")
    else:
        train_absa(assigned, args.output, encoder=args.encoder, epochs=args.epochs,
                   batch_size=args.batch_size, max_length=args.max_length, low_memory=args.low_memory)


if __name__ == "__main__":
    main()
