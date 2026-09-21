import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# Define paths
DATA_PATH = "stage1/data/test_reviews.csv" #to be changed depending sa dataset natin
MODEL_DIR = "stage1/models/dost_roberta"
os.makedirs(MODEL_DIR, exist_ok=True)

def main():
    print("1. Loading dataset...")
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig", skipinitialspace=True) #strips windows whitespace or any newline characters
    print(f"   Loaded {len(df)} reviews.")

    print("2. Downloading & Loading DOST-ASTI RoBERTa base model...")
    # Using the official DOST-ASTI RoBERTa Tagalog base model
    model_name = "dost-asti/RoBERTa-tl-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # We have 4 classes: Authentic(0), Deceptive(1), LIV(2), Irrelevant(3)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=4,
        ignore_mismatched_sizes=True
    )

    # Convert Pandas dataframe to Hugging Face Dataset format
    hf_dataset = Dataset.from_pandas(df[['review_text', 'label_stage1']])
    
    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(examples["review_text"], padding="max_length", truncation=True, max_length=128)
        
    print("3. Tokenizing dataset...")
    tokenized_datasets = hf_dataset.map(tokenize_function, batched=True)
    
    # Rename 'label_stage1' to 'labels' because HF Trainer expects a 'labels' column
    tokenized_datasets = tokenized_datasets.rename_column("label_stage1", "labels")
    tokenized_datasets.set_format("torch")

    # Split into train/val (for this dummy test, 80/20 split)
    split_ds = tokenized_datasets.train_test_split(test_size=0.2, seed=42)
    train_data = split_ds["train"]
    val_data = split_ds["test"]

    print("4. Starting Training Loop...")
    training_args = TrainingArguments(
        output_dir="./temp_results",
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=2, # Small batch size for testing
        num_train_epochs=3,
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