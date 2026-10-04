import os

import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

import config


DATA_PATH = config.DATA_PATH
MODEL_DIR = config.MODEL_DIR


def main():
    if not DATA_PATH.exists():
        print(f"ERROR: Dataset not found at {DATA_PATH}.")
        return

    print("1. Loading dataset...")
    df = pd.read_csv(
        DATA_PATH,
        encoding="utf-8-sig",
        skipinitialspace=True,
    )

    required_columns = {
        "review_text",
        "product_description",
        "text_label",
    }
    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    df["review_text"] = df["review_text"].fillna("").astype(str)
    df["product_description"] = (
        df["product_description"]
        .fillna("")
        .astype(str)
    )

    # Stage 1 is trained exclusively against text_label.
    df["labels"] = df["text_label"].map(config.map_label)

    if df["labels"].isna().any():
        raise ValueError("One or more text_label values could not be mapped.")

    df["labels"] = df["labels"].astype("int64")

    print(f"   Loaded {len(df)} reviews.")
    print(f"   Stage 1 label counts:\n{df['labels'].value_counts().sort_index()}")

    print("2. Loading DOST-ASTI RoBERTa...")
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL)

    model = AutoModelForSequenceClassification.from_pretrained(
        config.BASE_MODEL,
        num_labels=4,
        ignore_mismatched_sizes=True,
    )

    dataset = Dataset.from_pandas(
        df[
            [
                "product_description",
                "review_text",
                "labels",
            ]
        ],
        preserve_index=False,
    )

    def tokenize_function(examples):
        # Product description is the first sequence and review is the second.
        # The tokenizer inserts the model's separator tokens automatically.
        return tokenizer(
            examples["product_description"],
            examples["review_text"],
            padding="max_length",
            truncation="only_first",
            max_length=config.MAX_LENGTH,
        )

    print("3. Tokenizing product descriptions and reviews...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=[
            "product_description",
            "review_text",
        ],
    )
    tokenized_dataset.set_format("torch")

    split_dataset = tokenized_dataset.train_test_split(
        test_size=config.TEST_SPLIT,
        seed=42,
    )

    print("4. Starting training...")
    os.makedirs(MODEL_DIR, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(MODEL_DIR),
        eval_strategy="epoch",
        learning_rate=config.LEARNING_RATE,
        per_device_train_batch_size=config.BATCH_SIZE,
        num_train_epochs=config.EPOCHS,
        save_strategy="no",
        use_cpu=not torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
        processing_class=tokenizer,
    )

    trainer.train()

    print(f"5. Saving Stage 1 model to {MODEL_DIR}...")
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    print("Stage 1 training complete.")


if __name__ == "__main__":
    main()
