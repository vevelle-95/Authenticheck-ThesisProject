"""Fine-tune DOST-RoBERTa for open-ended aspect extraction and sentiment.

The training CSV uses an `aspect_spans` JSON column. Each entry contains the
literal aspect phrase and its sentiment, allowing one review to supervise
multiple generated aspect mentions without a fixed aspect vocabulary.
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer

import absa_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_aspect_spans(value):
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid aspect_spans JSON: {stripped}") from error
    if not isinstance(value, list):
        raise ValueError("aspect_spans must contain a JSON list")

    spans = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("Each aspect_spans entry must be an object")
        aspect_text = str(item.get("text", item.get("aspect", ""))).strip()
        if not aspect_text:
            raise ValueError("Each aspect_spans entry needs a text value")
        sentiment_value = item.get("sentiment", item.get("polarity"))
        spans.append((aspect_text, absa_model.polarity_index(sentiment_value)))
    return spans


def locate_spans(text, spans):
    located = []
    for aspect_text, sentiment in spans:
        match = re.search(re.escape(aspect_text), text, flags=re.IGNORECASE)
        if match is None:
            raise ValueError(f"Aspect phrase {aspect_text!r} was not found in review: {text!r}")
        located.append((match.start(), match.end(), sentiment))
    located.sort(key=lambda item: item[0])
    for previous, current in zip(located, located[1:]):
        if current[0] < previous[1]:
            raise ValueError("Overlapping aspect phrases are not supported")
    return located


def build_span_targets(df, tokenizer, text_col, max_length):
    if "aspect_spans" not in df.columns:
        raise ValueError("ABSA training data requires an aspect_spans column")
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("A fast tokenizer is required for aspect span alignment")

    texts = []
    for value in df[text_col]:
        if pd.isna(value) or not str(value).strip():
            raise ValueError("ABSA training data contains an empty review")
        texts.append(str(value))

    encodings = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offsets = encodings["offset_mapping"]
    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]
    aspect_targets = torch.full(
        (len(texts), input_ids.shape[1]),
        absa_model.SENTIMENT_IGNORE,
        dtype=torch.long,
    )
    sentiment_targets = torch.full(
        (len(texts), input_ids.shape[1]),
        absa_model.SENTIMENT_IGNORE,
        dtype=torch.long,
    )

    for row_index, text in enumerate(texts):
        valid = attention_mask[row_index].bool() & (offsets[row_index, :, 1] > offsets[row_index, :, 0])
        aspect_targets[row_index, valid] = 0
        located = locate_spans(text, parse_aspect_spans(df.iloc[row_index]["aspect_spans"]))
        for start_char, end_char, sentiment in located:
            token_indices = [
                token_index
                for token_index in range(input_ids.shape[1])
                if bool(valid[token_index])
                and int(offsets[row_index, token_index, 0]) < end_char
                and int(offsets[row_index, token_index, 1]) > start_char
            ]
            if not token_indices:
                raise ValueError(f"Aspect phrase {text[start_char:end_char]!r} was truncated without tokens")
            if any(int(aspect_targets[row_index, token_index]) != 0 for token_index in token_indices):
                raise ValueError(f"Overlapping aspect phrases are not supported: {text[start_char:end_char]!r}")
            aspect_targets[row_index, token_indices[0]] = 1
            if len(token_indices) > 1:
                aspect_targets[row_index, token_indices[1:]] = 2
            sentiment_targets[row_index, token_indices] = sentiment

    return input_ids, attention_mask, aspect_targets, sentiment_targets, encodings["offset_mapping"]


def evaluate(model, aspect_targets, sentiment_targets, aspect_logits, sentiment_logits):
    aspect_mask = aspect_targets != absa_model.SENTIMENT_IGNORE
    aspect_predictions = torch.argmax(aspect_logits, dim=-1)
    aspect_accuracy = float(
        accuracy_score(
            aspect_targets[aspect_mask].cpu().numpy(),
            aspect_predictions[aspect_mask].cpu().numpy(),
        )
    ) if aspect_mask.any() else 0.0
    aspect_f1 = float(
        f1_score(
            (aspect_targets[aspect_mask] > 0).cpu().numpy(),
            (aspect_predictions[aspect_mask] > 0).cpu().numpy(),
            zero_division=0,
        )
    ) if aspect_mask.any() else 0.0

    sentiment_mask = sentiment_targets != absa_model.SENTIMENT_IGNORE
    sentiment_predictions = torch.argmax(sentiment_logits, dim=-1)
    if sentiment_mask.any():
        sentiment_accuracy = float(
            accuracy_score(
                sentiment_targets[sentiment_mask].cpu().numpy(),
                sentiment_predictions[sentiment_mask].cpu().numpy(),
            )
        )
        sentiment_f1 = float(
            f1_score(
                sentiment_targets[sentiment_mask].cpu().numpy(),
                sentiment_predictions[sentiment_mask].cpu().numpy(),
                average="macro",
                zero_division=0,
            )
        )
    else:
        sentiment_accuracy = 0.0
        sentiment_f1 = 0.0
    return {
        "aspect_accuracy": aspect_accuracy,
        "aspect_f1": aspect_f1,
        "sentiment_accuracy": sentiment_accuracy,
        "sentiment_f1": sentiment_f1,
    }


def print_metrics(metrics):
    print("\n--- OPEN-ENDED ASPET EXTRACTION ---")
    print(f"   Token accuracy: {metrics['aspect_accuracy']:.3f}")
    print(f"   Aspect-token F1: {metrics['aspect_f1']:.3f}")
    print("\n--- ASPECT SENTIMENT ---")
    print(f"   Accuracy: {metrics['sentiment_accuracy']:.3f}")
    print(f"   Macro F1: {metrics['sentiment_f1']:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune DOST-RoBERTa for open-ended ABSA")
    parser.add_argument("--data", default=str(absa_model.DEFAULT_DATA_PATH), help="CSV with text and aspect_spans")
    parser.add_argument("--encoder", default=None, help="Encoder checkpoint; defaults to Stage 1 or DOST-TL")
    parser.add_argument("--output", default=str(absa_model.DEFAULT_ABSA_MODEL_DIR), help="Output model directory")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=absa_model.ABSA_MAX_LENGTH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-only", action="store_true", help="Evaluate a saved model without fine-tuning")
    parser.add_argument("--authentic-only", action="store_true", help="Train only on reviews classified as authentic")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: Dataset not found at {data_path}. Create an ABSA span-annotated CSV first.")
        return

    print(f"1. Loading dataset from {data_path}...")
    df = pd.read_csv(data_path, encoding="utf-8-sig", skipinitialspace=True)
    text_col = "text" if "text" in df.columns else "review_text"
    if text_col not in df.columns:
        print("ERROR: ABSA data requires a text or review_text column.")
        return
    print(f"   Loaded {len(df)} reviews.")

    if args.authentic_only:
        try:
            df = absa_model.filter_authentic_reviews(df)
        except ValueError as error:
            print(f"ERROR: {error}")
            return
        print(f"   Authentic reviews retained: {len(df)}.")

    if df.empty:
        print("ERROR: No reviews remain for ABSA training.")
        return
    if not args.eval_only and len(df) < 5:
        print("ERROR: At least five reviews are required for a train/validation split.")
        return

    encoder = args.encoder
    if encoder is None and args.eval_only:
        config_path = Path(args.output) / "absa_config.json"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as file:
                encoder = json.load(file).get("encoder")
    encoder = encoder or absa_model.resolve_encoder()
    print(f"2. Loading encoder checkpoint ({encoder})...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(encoder, use_fast=True)
    except Exception as error:
        print(f"ERROR: Could not load tokenizer: {error}")
        return
    try:
        input_ids, attention_mask, aspect_targets, sentiment_targets, _ = build_span_targets(
            df,
            tokenizer,
            text_col,
            args.max_length,
        )
    except ValueError as error:
        print(f"ERROR: {error}")
        return

    if args.eval_only:
        print(f"3. Loading saved ABSA model from {args.output}...")
        try:
            model = absa_model.ABSAHeadModel.from_pretrained(args.output).to(DEVICE)
        except Exception as error:
            print(f"ERROR: Could not load saved ABSA model: {error}")
            return
        model.eval()
        with torch.no_grad():
            aspect_logits, sentiment_logits = model(input_ids.to(DEVICE), attention_mask.to(DEVICE))
        metrics = evaluate(
            model,
            aspect_targets,
            sentiment_targets,
            aspect_logits.cpu(),
            sentiment_logits.cpu(),
        )
        print_metrics(metrics)
        return

    print(f"3. Building token targets from {df.shape[0]} reviews...")
    torch.manual_seed(args.seed)
    model = absa_model.ABSAHeadModel(encoder).to(DEVICE)
    split = train_test_split(
        input_ids,
        attention_mask,
        aspect_targets,
        sentiment_targets,
        test_size=0.2,
        random_state=args.seed,
    )
    train_ids, val_ids, train_mask, val_mask, train_aspects, val_aspects, train_sentiments, val_sentiments = split
    train_dataset = TensorDataset(train_ids, train_mask, train_aspects, train_sentiments)
    val_dataset = TensorDataset(val_ids, val_mask, val_aspects, val_sentiments)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print("4. Starting open-ended ABSA fine-tuning...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch_ids, batch_mask, batch_aspects, batch_sentiments = [value.to(DEVICE) for value in batch]
            optimizer.zero_grad()
            aspect_logits, sentiment_logits = model(batch_ids, batch_mask)
            loss = model.compute_loss(
                aspect_logits,
                sentiment_logits,
                batch_aspects,
                batch_sentiments,
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"   Epoch {epoch}/{args.epochs} - loss: {total_loss / len(train_loader):.4f}")

    print("5. Evaluating extracted spans and sentiment...")
    model.eval()
    all_aspect_targets = []
    all_sentiment_targets = []
    all_aspect_logits = []
    all_sentiment_logits = []
    with torch.no_grad():
        for batch in val_loader:
            batch_ids, batch_mask, batch_aspects, batch_sentiments = [value.to(DEVICE) for value in batch]
            aspect_logits, sentiment_logits = model(batch_ids, batch_mask)
            all_aspect_targets.append(batch_aspects.cpu())
            all_sentiment_targets.append(batch_sentiments.cpu())
            all_aspect_logits.append(aspect_logits.cpu())
            all_sentiment_logits.append(sentiment_logits.cpu())
    metrics = evaluate(
        model,
        torch.cat(all_aspect_targets),
        torch.cat(all_sentiment_targets),
        torch.cat(all_aspect_logits),
        torch.cat(all_sentiment_logits),
    )
    print_metrics(metrics)

    print(f"6. Saving ABSA model to {args.output}...")
    model.save_pretrained(args.output)
    print("Done. Use predict_absa.py to extract generated aspect phrases and sentiments.")


if __name__ == "__main__":
    main()
