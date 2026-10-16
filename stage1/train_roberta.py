"""Train quality classification on product title, description, and buyer review."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model_contract import (
    BASE_MODEL, BATCH_SIZE, CLASS_NAMES, DEFAULT_DATA_PATH, DEFAULT_ROBERTA_DIR,
    DEFAULT_SPLITS_PATH, EPOCHS, INPUT_VERSION, LEARNING_RATE, MAX_LENGTH, SEED,
    QUALITY_CONTEXT_TOKENS, QUALITY_TEXT_POLICY, QUALITY_TITLE_TOKENS, ROBERTA_TARGET_COLUMN,
    quality_input_pairs, quality_input_text,
)
from model_data import fingerprint, load_experiment
from training_augmentation import augmentation_summary
from model_metrics import classification_metrics


def tokenize_reviews(tokenizer, texts, max_length=MAX_LENGTH, **kwargs):
    pairs = quality_input_pairs(tokenizer, texts, max_length)
    reviews = [review for review, _ in pairs]
    contexts = [context for _, context in pairs]
    return tokenizer(reviews, contexts, padding="max_length", truncation="only_first",
                     max_length=max_length, return_token_type_ids=False, **kwargs)


def validate_resume_checkpoint(checkpoint, output, train_frame, validation_frame, *,
                               base_model, epochs, batch_size, max_length, seed, low_memory):
    """Resume only the same experiment; device changes do not change its targets."""
    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    if checkpoint.parent != output / "checkpoints":
        raise ValueError("Resume checkpoint must be inside this output bundle's checkpoints directory")
    required = ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth", "config.json")
    missing = [name for name in required if not (checkpoint / name).is_file()]
    if not any((checkpoint / name).is_file() for name in ("model.safetensors", "pytorch_model.bin")):
        missing.append("model weights")
    if missing:
        raise ValueError(f"Incomplete resume checkpoint: {missing}")
    previous = json.loads((output / "run_metadata.json").read_text(encoding="utf-8"))
    validating = validation_frame is not None and len(validation_frame) > 0
    expected = {
        "input_version": INPUT_VERSION, "target_column": ROBERTA_TARGET_COLUMN,
        "quality_text_policy": QUALITY_TEXT_POLICY, "training_sha256": fingerprint(train_frame),
        "validation_sha256": fingerprint(validation_frame) if validating else None,
        "base_model": str(base_model), "epochs": epochs, "batch_size": batch_size,
        "max_length": max_length, "seed": seed, "low_memory_training": low_memory,
    }
    differences = [key for key, value in expected.items() if previous.get(key) != value]
    if differences:
        raise ValueError(f"Resume experiment differs in {differences}; preserve data, settings and optimizer")
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    if config.get("input_contract_version") != INPUT_VERSION or config.get("quality_target_column") != ROBERTA_TARGET_COLUMN:
        raise ValueError("Resume checkpoint has an incompatible text input/target contract")
    state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    if not 0 < state.get("global_step", 0) < state.get("max_steps", 0):
        raise ValueError("Checkpoint must describe a partially completed training run")
    return checkpoint, previous, state


def fit_roberta(train_frame, validation_frame, output, *, base_model=BASE_MODEL,
                epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
                max_length=MAX_LENGTH, seed=SEED, low_memory=False, resume_from_checkpoint=None):
    import torch
    from datasets import Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, set_seed

    if set(train_frame.text_target) != {0, 1, 2, 3}:
        raise ValueError("RoBERTa fitting requires all four text_label classes")
    if epochs < 1 or batch_size < 1 or max_length < 8:
        raise ValueError("epochs/batch_size must be positive; max_length must be at least 8")
    set_seed(seed)
    output = Path(output)
    resume_state = previous_run = None
    if resume_from_checkpoint:
        resume_from_checkpoint, previous_run, resume_state = validate_resume_checkpoint(
            resume_from_checkpoint, output, train_frame, validation_frame, base_model=base_model,
            epochs=epochs, batch_size=batch_size, max_length=max_length, seed=seed, low_memory=low_memory)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=4, ignore_mismatched_sizes=True,
        id2label=dict(enumerate(CLASS_NAMES)), label2id={name: i for i, name in enumerate(CLASS_NAMES)},
    )
    model.config.input_contract_version = INPUT_VERSION
    model.config.review_max_length = max_length
    model.config.quality_text_policy = QUALITY_TEXT_POLICY
    model.config.quality_context_tokens = QUALITY_CONTEXT_TOKENS
    model.config.quality_title_tokens = QUALITY_TITLE_TOKENS
    model.config.quality_target_column = ROBERTA_TARGET_COLUMN
    if low_memory:
        model.config.use_cache = False
    memory_options = {"gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "fp16": torch.cuda.is_available(), "optim": "adafactor"} if low_memory else {}

    def dataset(frame):
        texts = [quality_input_text(row) for row in frame.to_dict("records")]
        encodings = tokenize_reviews(tokenizer, texts, max_length)
        return Dataset.from_dict({**encodings, "labels": frame.text_target.tolist()})

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
        **memory_options,
    )
    print(f"RoBERTa device: {arguments.device}; epochs={epochs}; batch_size={batch_size}", flush=True)
    trainer = Trainer(
        model=model, args=arguments, train_dataset=dataset(train_frame),
        eval_dataset=dataset(validation_frame) if validating else None,
        processing_class=tokenizer, compute_metrics=metrics,
    )
    output.mkdir(parents=True, exist_ok=True)
    run_metadata = {"input_version": INPUT_VERSION, "training_sha256": fingerprint(train_frame),
        "target_column": ROBERTA_TARGET_COLUMN,
        "validation_sha256": fingerprint(validation_frame) if validating else None,
        "epochs": epochs, "batch_size": batch_size, "max_length": max_length,
        "quality_text_policy": QUALITY_TEXT_POLICY,
        "seed": seed, "base_model": str(base_model), "low_memory_training": low_memory,
        "fp16": low_memory and torch.cuda.is_available(),
        "device": str(arguments.device), "torch_version": torch.__version__,
        "resume_from_checkpoint": str(resume_from_checkpoint) if resume_from_checkpoint else None}
    if resume_state:
        (output / "resume_metadata.json").write_text(json.dumps({
            "checkpoint": str(resume_from_checkpoint), "global_step": resume_state["global_step"],
            "epoch": resume_state.get("epoch"), "previous_run": previous_run,
            "device": str(arguments.device), "torch_version": torch.__version__,
        }, indent=2), encoding="utf-8")
        print(f"Resuming RoBERTa from epoch {resume_state.get('epoch')}, step {resume_state['global_step']} on {arguments.device}", flush=True)
    (output / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")
    if resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=str(resume_from_checkpoint))
    else:
        trainer.train()
    trainer.save_model(str(output))
    tokenizer.save_pretrained(output)
    validation_metrics = trainer.evaluate() if validating else None
    metadata = {
        "augmentation_usage": augmentation_summary(train_frame),
        "input_version": INPUT_VERSION, "max_length": max_length, "seed": seed,
        "target_column": ROBERTA_TARGET_COLUMN,
        "base_model": str(base_model), "epochs": epochs,
        "resume_from_checkpoint": str(resume_from_checkpoint) if resume_from_checkpoint else None,
        "resumed_global_step": resume_state["global_step"] if resume_state else None,
        "device": str(arguments.device), "torch_version": torch.__version__,
        "low_memory_training": low_memory, "fp16": low_memory and torch.cuda.is_available(),
        "optimizer": "adafactor" if low_memory else "adamw_torch",
        "text_input": "review_text paired with bounded product_title + product_description",
        "quality_text_policy": QUALITY_TEXT_POLICY,
        "context_token_limit": QUALITY_CONTEXT_TOKENS, "title_token_limit": QUALITY_TITLE_TOKENS,
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
    parser.add_argument("--low-memory", action="store_true")
    parser.add_argument("--gpu-memory-gb", type=float)
    parser.add_argument("--resume-from-checkpoint", help="Continue this bundle's saved model/optimizer/scheduler state")
    parser.add_argument("--require-cuda", action="store_true", help="Fail instead of silently training on CPU")
    args = parser.parse_args()
    from gpu_memory import configure_gpu_memory
    configure_gpu_memory(args.gpu_memory_gb)
    if args.require_cuda:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in this Python environment; install CUDA-enabled PyTorch before training")
    assigned = load_experiment(args.data, args.splits, args.augmentations,
                               allow_unreviewed=args.allow_unreviewed_augmentations)
    fit_roberta(
        assigned[assigned.partition.eq("train")], assigned[assigned.partition.eq("validation")],
        args.output, base_model=args.base_model, epochs=args.epochs,
        batch_size=args.batch_size, max_length=args.max_length, low_memory=args.low_memory,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    print(f"Saved validation-selected RoBERTa to {args.output}; test products were not used.")


if __name__ == "__main__":
    main()
