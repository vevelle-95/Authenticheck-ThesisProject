import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

import config # config for the model
    
# Define paths
DATA_PATH = config.DATA_PATH # to be changed depending sa dataset natin
MODEL_DIR = config.MODEL_DIR

    # Numerical Labels:
    # 0 - Authentic
    # 1 - Deceptive
    # 2 - Low Informational Value
    # 3 - Irrelevant

def main():
    if not config.DATA_PATH.exists():
        print(f"ERROR: Dataset not found at {config.DATA_PATH}.")
        print("Create data/test_reviews.csv with 'review_text' and 'label_stage1' (0-3) columns first.")
        return
    print("1. Loading dataset...")
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig", skipinitialspace=True)
    print(f"   Loaded {len(df)} reviews.")

    print("2. Downloading & Loading DOST-ASTI RoBERTa base model...")
    # Using the official DOST-ASTI RoBERTa Tagalog base model from config
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL)
    
    # We have 4 classes: Authentic(0), Deceptive(1), LIV(2), Irrelevant(3)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.BASE_MODEL, 
        num_labels=4,
        ignore_mismatched_sizes=True
    )

    # Convert Pandas dataframe to Hugging Face Dataset format
    hf_dataset = Dataset.from_pandas(df[['review_text', 'label_stage1']])
    
    # Tokenization function using config max_length
    def tokenize_function(examples):
        return tokenizer(examples["review_text"], padding="max_length", truncation=True, max_length=config.MAX_LENGTH)
        
    print("3. Tokenizing dataset...")
    tokenized_datasets = hf_dataset.map(tokenize_function, batched=True)
    
    # Rename 'label_stage1' to 'labels' because HF Trainer expects a 'labels' column
    tokenized_datasets = tokenized_datasets.rename_column("label_stage1", "labels")
    tokenized_datasets.set_format("torch")

    # Split into train/val using config test_split
    split_ds = tokenized_datasets.train_test_split(test_size=config.TEST_SPLIT, seed=42)
    train_data = split_ds["train"]
    val_data = split_ds["test"]

    print("4. Starting Training Loop...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(config.MODEL_DIR),
        eval_strategy="epoch",
        learning_rate=config.LEARNING_RATE,
        per_device_train_batch_size=config.BATCH_SIZE, 
        num_train_epochs=config.EPOCHS,
        save_strategy="no",
        use_cpu=not torch.cuda.is_available() # Failsafe to use CPU if no GPU
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        processing_class=tokenizer,
    )

    trainer.train()

    print(f"5. Saving fine-tuned model to {MODEL_DIR}...")
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    print("Pipeline Complete! Model is ready for extraction.")

if __name__ == "__main__":
    main()