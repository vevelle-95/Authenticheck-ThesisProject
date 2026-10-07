"""Train quality classification on product title, description, and buyer review."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model_contract import (
    BASE_MODEL, BATCH_SIZE, CLASS_NAMES, DEFAULT_DATA_PATH, DEFAULT_ROBERTA_DIR,
    DEFAULT_SPLITS_PATH, EPOCHS, INPUT_VERSION, LEARNING_RATE, MAX_LENGTH, SEED,
    quality_input_text,
)
from model_data import fingerprint, load_experiment
from training_augmentation import augmentation_summary
from model_metrics import classification_metrics


def tokenize_reviews(tokenizer, texts, max_length=MAX_LENGTH, **kwargs):
    return tokenizer(list(texts), padding="max_length", truncation=True, max_length=max_length, **kwargs)


def fit_roberta(train_frame, validation_frame, output, *, base_model=BASE_MODEL,
                epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
                max_length=MAX_LENGTH, seed=SEED):
    import torch
    from datasets import Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, set_seed

    if set(train_frame.label) != {0, 1, 2, 3}:
        raise ValueError("RoBERTa fitting requires all four quality classes")
    if epochs < 1 or batch_size < 1 or max_length < 8:
        raise ValueError("epochs/batch_size must be positive; max_length must be at least 8")
    set_seed(seed)
    output = Path(output)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=4, ignore_mismatched_sizes=True,
        id2label=dict(enumerate(CLASS_NAMES)), label2id={name: i for i, name in enumerate(CLASS_NAMES)},
    )
    model.config.input_contract_version = INPUT_VERSION
    model.config.review_max_length = max_length

    def dataset(frame):
        texts = [quality_input_text(row) for row in frame.to_dict("records")]
        encodings = tokenize_reviews(tokenizer, texts, max_length)
        return Dataset.from_dict({**encodings, "labels": frame.label.tolist()})

    def metrics(prediction):
        logits = prediction.predictions[0] if isinstance(prediction.predictions, tuple) else prediction.predictions
        result = classification_metrics(prediction.label_ids, logits.argmax(axis=-1), range(4))
        return {key: result[key] for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1")}

    validating = validation_frame is not None and len(validation_frame) > 0
    arguments = TrainingArguments(
        output_dir=str(output / "checkpoints"), learning_rate=learning_rate,
        per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs, eval_strategy="epoch" if validating else "no",
        save_strategy="epoch" if validating else "no", save_total_limit=1,
        load_best_model_at_end=validating, metric_for_best_model="macro_f1",
        greater_is_better=True, use_cpu=not torch.cuda.is_available(), seed=seed, data_seed=seed,
        report_to="none", dataloader_pin_memory=torch.cuda.is_available(),
    )
    trainer = Trainer(
        model=model, args=arguments, train_dataset=dataset(train_frame),
        eval_dataset=dataset(validation_frame) if validating else None,
        processing_class=tokenizer, compute_metrics=metrics,
    )
    trainer.train()
    trainer.save_model(str(output))
    tokenizer.save_pretrained(output)
    validation_metrics = trainer.evaluate() if validating else None
    metadata = {
        "augmentation_usage": augmentation_summary(train_frame),
        "input_version": INPUT_VERSION, "max_length": max_length, "seed": seed,
        "base_model": str(base_model), "epochs": epochs,
        "text_input": "product_title + product_description + review_text",
        "fit_review_ids": train_frame.review_id.tolist(),
        "validation_review_ids": validation_frame.review_id.tolist() if validating else [],
        "training_sha256": fingerprint(train_frame), "validation_metrics": validation_metrics,
    }
    (output / "training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return model, tokenizer


def predict_probabilities(model, tokenizer, frame, batch_size=16, max_length=None):
    import numpy as np
    import torch

    max_length = max_length or model.config.review_max_length
    model.eval()
    probabilities = []
    for start in range(0, len(frame), batch_size):
        enc = tokenize_reviews(
            tokenizer,
            [quality_input_text(row) for row in frame.iloc[start:start + batch_size].to_dict("records")],
            max_length, return_tensors="pt",
        ).to(next(model.parameters()).device)
        with torch.inference_mode():
            probabilities.extend(torch.softmax(model(**enc).logits, dim=-1).cpu().tolist())
    result = np.asarray(probabilities).reshape(-1, 4)
    if not np.isfinite(result).all() or not np.allclose(result.sum(axis=1), 1, atol=1e-5):
        raise ValueError("Invalid four-class probabilities")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--splits", default=str(DEFAULT_SPLITS_PATH))
    parser.add_argument("--augmentations", help="Approved training-only augmentation CSV; use the same file throughout the experiment")
    parser.add_argument("--allow-unreviewed-augmentations", action="store_true", help="Experimental run: include pending drafts without marking them approved")
    parser.add_argument("--output", default=str(DEFAULT_ROBERTA_DIR))
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    args = parser.parse_args()
    assigned = load_experiment(args.data, args.splits, args.augmentations,
                               allow_unreviewed=args.allow_unreviewed_augmentations)
    fit_roberta(
        assigned[assigned.partition.eq("train")], assigned[assigned.partition.eq("validation")],
        args.output, base_model=args.base_model, epochs=args.epochs,
        batch_size=args.batch_size, max_length=args.max_length,
    )
    print(f"Saved validation-selected RoBERTa to {args.output}; test products were not used.")


if __name__ == "__main__":
    main()
