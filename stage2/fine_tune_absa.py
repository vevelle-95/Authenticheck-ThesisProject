"""Stage 2 - ABSA fine-tuning/evaluation for the aspect detection and
aspect-conditioned sentiment heads.

Reuses the fine-tuned DOST-RoBERTa encoder from Stage 1 and fine-tunes the
whole model (encoder + randomly initialized heads) on ABSA-annotated reviews.

Expected CSV format (`data/absa_reviews.csv` by default):
  - `text` (or `review_text`): the review text.
  - `aspect_<name>` for each aspect in absa_model.ASPECTS: 0 (absent) or 1 (present).
  - `sentiment_<name>` for each aspect: 0 (Negative), 1 (Neutral), 2 (Positive),
    or NaN / -1 when the aspect is absent.
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import absa_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL_INDEX = {label: i for i, label in enumerate(absa_model.POLARITY_LABELS)}


def build_targets(df, aspects):
    aspect_targets = []
    sentiment_targets = []
    for _, row in df.iterrows():
        row_aspects = []
        row_sentiments = []
        for name in aspects:
            col_aspect = f"aspect_{name}"
            col_sent = f"sentiment_{name}"
            present = int(row[col_aspect])
            row_aspects.append(float(present))
            sent = pd.to_numeric(row[col_sent], errors="coerce")
            if present == 1 and pd.notna(sent) and int(sent) in LABEL_INDEX.values():
                row_sentiments.append(float(int(sent)))
            else:
                row_sentiments.append(float(absa_model.SENTIMENT_IGNORE))
        aspect_targets.append(row_aspects)
        sentiment_targets.append(row_sentiments)
    return torch.tensor(aspect_targets, dtype=torch.float32), torch.tensor(sentiment_targets, dtype=torch.float32)


def evaluate(model, aspect_targets, sentiment_targets, aspect_logits, sentiment_logits):
    model.eval()
    aspect_preds = (torch.sigmoid(aspect_logits) >= 0.5).numpy().astype(int)
    y_true = aspect_targets.numpy().astype(int)
    per_aspect = []
    for i, name in enumerate(model.aspects):
        per_aspect.append({
            "aspect": name,
            "precision": precision_score(y_true[:, i], aspect_preds[:, i], zero_division=0),
            "recall": recall_score(y_true[:, i], aspect_preds[:, i], zero_division=0),
            "f1": f1_score(y_true[:, i], aspect_preds[:, i], zero_division=0),
        })

    sent_probs = F.softmax(sentiment_logits, dim=-1)
    sent_preds = torch.argmax(sent_probs, dim=-1).numpy()
    s_true = sentiment_targets.numpy().astype(int)
    mask = s_true != absa_model.SENTIMENT_IGNORE
    sent_acc = accuracy_score(s_true[mask], sent_preds[mask])
    sent_macro_f1 = f1_score(s_true[mask], sent_preds[mask], average="macro", zero_division=0)

    aspect_macro_f1 = sum(p["f1"] for p in per_aspect) / len(per_aspect)
    return {
        "aspect_macro_f1": aspect_macro_f1,
        "aspect_per_aspect": per_aspect,
        "sentiment_accuracy": sent_acc,
        "sentiment_macro_f1": sent_macro_f1,
    }


def main():
    parser = argparse.ArgumentParser(description="Fine-tune the RoBERTa-based ABSA heads")
    parser.add_argument("--data", default=str(absa_model.DEFAULT_DATA_PATH), help="CSV of annotated reviews")
    parser.add_argument("--encoder", default=None, help="Encoder checkpoint; defaults to fine-tuned Stage 1 model")
    parser.add_argument("--output", default=str(absa_model.DEFAULT_ABSA_MODEL_DIR), help="Output model directory")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=absa_model.ABSA_MAX_LENGTH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-only", action="store_true", help="Evaluate a saved model without fine-tuning")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: Dataset not found at {data_path}. Create an ABSA-annotated CSV first.")
        return

    print(f"1. Loading dataset from {data_path}...")
    df = pd.read_csv(data_path, encoding="utf-8-sig", skipinitialspace=True)
    text_col = "text" if "text" in df.columns else "review_text"
    print(f"   Loaded {len(df)} reviews.")

    encoder = args.encoder or absa_model.resolve_encoder()
    print(f"2. Loading encoder checkpoint ({encoder})...")
    tokenizer = AutoTokenizer.from_pretrained(encoder)
    model = absa_model.ABSAHeadModel(encoder).to(DEVICE)

    print(f"3. Building targets for aspects: {model.aspects}...")
    aspect_targets, sentiment_targets = build_targets(df, model.aspects)

    encodings = tokenizer(
        list(df[text_col]),
        padding="max_length",
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
    )

    if args.eval_only:
        print(f"4. Loading saved ABSA model from {args.output}...")
        model = absa_model.ABSAHeadModel.from_pretrained(args.output).to(DEVICE)
        model.eval()
        with torch.no_grad():
            aspect_logits, sentiment_logits = model(encodings["input_ids"], encodings["attention_mask"])
        metrics = evaluate(model, aspect_targets, sentiment_targets, aspect_logits.cpu(), sentiment_logits.cpu())
        print_metrics(metrics)
        return

    split = train_test_split(
        encodings["input_ids"], encodings["attention_mask"], aspect_targets, sentiment_targets,
        test_size=0.2, random_state=args.seed,
    )
    train_ids, val_ids, train_mask, val_mask, train_asp, val_asp, train_sent, val_sent = split

    train_ds = TensorDataset(train_ids, train_mask, train_asp, train_sent)
    val_ds = TensorDataset(val_ids, val_mask, val_asp, val_sent)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print("4. Starting fine-tuning...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_dl:
            input_ids, attn_mask, aspect_t, sentiment_t = [b.to(DEVICE) for b in batch]
            optimizer.zero_grad()
            aspect_logits, sentiment_logits = model(input_ids, attn_mask)
            loss = model.compute_loss(aspect_logits, sentiment_logits, aspect_t, sentiment_t)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"   Epoch {epoch}/{args.epochs} - loss: {total_loss / len(train_dl):.4f}")

    print("5. Evaluating on validation split...")
    model.eval()
    all_asp_gt, all_sent_gt = [], []
    all_aspect_logits, all_sentiment_logits = [], []
    with torch.no_grad():
        for batch in val_dl:
            input_ids, attn_mask, aspect_t, sentiment_t = [b.to(DEVICE) for b in batch]
            aspect_logits, sentiment_logits = model(input_ids, attn_mask)
            all_aspect_logits.append(aspect_logits.cpu())
            all_sentiment_logits.append(sentiment_logits.cpu())
            all_asp_gt.append(aspect_t.cpu())
            all_sent_gt.append(sentiment_t.cpu())
    metrics = evaluate(
        model,
        torch.cat(all_asp_gt),
        torch.cat(all_sent_gt),
        torch.cat(all_aspect_logits),
        torch.cat(all_sentiment_logits),
    )
    print_metrics(metrics)

    print(f"6. Saving ABSA model to {args.output}...")
    model.save_pretrained(args.output)
    print("Done. Use predict_absa.py to generate per-aspect sentiment profiles.")


def print_metrics(metrics):
    print("\n--- ASPECT DETECTION ---")
    for row in metrics["aspect_per_aspect"]:
        print(f"   {row['aspect']:<10} precision={row['precision']:.3f} recall={row['recall']:.3f} f1={row['f1']:.3f}")
    print(f"   Macro F1: {metrics['aspect_macro_f1']:.3f}")
    print("\n--- ASPECT-CONDITIONED SENTIMENT ---")
    print(f"   Accuracy: {metrics['sentiment_accuracy']:.3f}")
    print(f"   Macro F1: {metrics['sentiment_macro_f1']:.3f}")


if __name__ == "__main__":
    main()