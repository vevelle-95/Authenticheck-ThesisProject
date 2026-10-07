"""Train independent fixed-category ABSA; choose checkpoint/threshold on validation."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, set_seed

from stage2 import absa_model
from model_contract import (
    ASPECTS, BASE_MODEL, BATCH_SIZE, DEFAULT_DATA_PATH, DEFAULT_SPLITS_PATH,
    EPOCHS, LEARNING_RATE, SEED, SENSORY_POLICY, SENTIMENT_TARGET_POLICY,
)
from model_data import fingerprint, load_reviews, read_splits
from model_metrics import classification_metrics


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


def detection_metrics(targets, probabilities, eligible, threshold):
    predictions = (probabilities >= threshold) & eligible.astype(bool)
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


@torch.inference_mode()
def evaluate_loader(model, loader, device, tune=False):
    model.eval()
    target_rows, probabilities, eligible_rows, sentiment_targets, sentiment_predictions = [], [], [], [], []
    for batch in loader:
        logits, sentiments, targets, polarities, eligible = batch_forward(model, batch, device)
        target_rows.extend(targets.cpu().tolist())
        probabilities.extend(torch.sigmoid(logits).cpu().tolist())
        eligible_rows.extend(eligible.cpu().tolist())
        if sentiments is not None:
            for alternatives, prediction in zip(polarities.cpu(), sentiments.argmax(-1).cpu()):
                labels = alternatives.nonzero().flatten().tolist()
                sentiment_targets.extend(labels)
                sentiment_predictions.extend([int(prediction)] * len(labels))
    targets, probabilities, eligible = map(np.asarray, (target_rows, probabilities, eligible_rows))
    threshold = select_threshold(targets, probabilities, eligible) if tune else model.threshold
    return {
        "threshold": threshold,
        "detection": detection_metrics(targets, probabilities, eligible, threshold),
        "sentiment": classification_metrics(sentiment_targets, sentiment_predictions, range(3)),
    }


def train_absa(assigned, output, *, encoder=BASE_MODEL, epochs=EPOCHS, batch_size=BATCH_SIZE,
               learning_rate=LEARNING_RATE, max_length=absa_model.ABSA_MAX_LENGTH):
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    history, best = [], float("-inf")
    for epoch in range(epochs):
        model.train()
        losses = []
        for batch in train_loader:
            optimizer.zero_grad()
            logits, sentiments, targets, polarities, eligible = batch_forward(model, batch, device)
            loss = model.compute_loss(logits, sentiments, targets, polarities, eligible)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        metrics = evaluate_loader(model, val_loader, device, tune=True)
        sentiment_f1 = metrics["sentiment"]["macro_f1"]
        if sentiment_f1 is None:
            raise ValueError("Validation needs annotated Authentic review/category polarity instances")
        score = (metrics["detection"]["macro_f1"] + sentiment_f1) / 2
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), **metrics})
        print(json.dumps(history[-1]), flush=True)
        if score > best:
            best = score
            model.threshold = metrics["threshold"]
            model.save_pretrained(output, tokenizer)
            best_epoch = epoch + 1
            best_metrics = metrics
    Path(output, "training_metadata.json").write_text(json.dumps({
        "dataset_sha256": fingerprint(assigned), "seed": SEED,
        "sensory_domain_policy": SENSORY_POLICY,
        "sentiment_target_policy": SENTIMENT_TARGET_POLICY,
        "fit_review_ids": training.review_id.tolist(), "validation_review_ids": validation.review_id.tolist(),
        "best_epoch": best_epoch, "selection_metric": "mean_detection_and_gold_category_sentiment_macro_f1",
        "validation_metrics": best_metrics, "history": history,
        "aggregation_unit": "one_review_category_polarity_group",
    }, indent=2), encoding="utf-8")
    return best_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--splits", default=str(DEFAULT_SPLITS_PATH))
    parser.add_argument("--encoder", default=BASE_MODEL, help="Independent pretrained encoder, not the Stage 1 classifier")
    parser.add_argument("--output", default=str(absa_model.DEFAULT_ABSA_MODEL_DIR))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=absa_model.ABSA_MAX_LENGTH)
    parser.add_argument("--authentic-only", action="store_true", help="Accepted for compatibility; ABSA always trains on ground-truth Authentic reviews")
    parser.add_argument("--eval-only", action="store_true", help="Evaluate validation only; final testing uses evaluate_models.py")
    args = parser.parse_args()
    assigned = read_splits(load_reviews(args.data, require_annotations=True), args.splits)
    if args.eval_only:
        model = absa_model.ABSAHeadModel.from_pretrained(args.output)
        tokenizer = AutoTokenizer.from_pretrained(model.tokenizer_dir, local_files_only=True)
        validation = assigned[assigned.partition.eq("validation") & assigned.label.eq(0)]
        loader = DataLoader(build_targets(validation, tokenizer, model.max_length), batch_size=args.batch_size)
        print(json.dumps(evaluate_loader(model, loader, torch.device("cpu")), indent=2))
    else:
        train_absa(assigned, args.output, encoder=args.encoder, epochs=args.epochs,
                   batch_size=args.batch_size, max_length=args.max_length)


if __name__ == "__main__":
    main()
