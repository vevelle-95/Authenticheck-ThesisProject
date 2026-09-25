"""RoBERTa-based open-ended aspect extraction and sentiment analysis.

The encoder produces token representations. An aspect head learns BIO tags for
aspect mentions, and a sentiment head learns polarity for tokens belonging to
those mentions. At inference, predicted spans are decoded from the review text
and each decoded aspect span receives its own sentiment distribution.
"""

import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
STAGE1_MODEL_DIR = MODELS_DIR / "dost_roberta"
DEFAULT_ABSA_MODEL_DIR = MODELS_DIR / "absa_model"
DEFAULT_DATA_PATH = DATA_DIR / "absa_reviews.csv"
DEFAULT_RESULTS_PATH = DATA_DIR / "absa_sentiment_results.json"
BASE_MODEL = "dost-asti/RoBERTa-tl-cased"
ABSA_MAX_LENGTH = 128
MODEL_VERSION = "span-v1"

ASPECT_TOKENS = ["O", "B-ASPECT", "I-ASPECT"]
POLARITY_LABELS = ["Negative", "Neutral", "Positive"]
SENTIMENT_IGNORE = -100


def polarity_index(value):
    if isinstance(value, str):
        normalized = value.strip().lower()
        named = {
            "negative": 0,
            "neutral": 1,
            "positive": 2,
        }
        if normalized in named:
            return named[normalized]
    try:
        numeric = int(float(value))
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid sentiment label: {value!r}") from error
    if numeric not in range(len(POLARITY_LABELS)):
        raise ValueError(f"Sentiment label must be 0, 1, or 2: {value!r}")
    return numeric


def filter_authentic_reviews(df):
    if "authentic" in df.columns:
        mask = df["authentic"].astype(str).str.strip().str.lower().isin(["1", "true", "yes", "authentic"])
    elif "label_stage1" in df.columns:
        labels = df["label_stage1"].astype(str).str.strip().str.lower()
        numeric_mask = pd.to_numeric(df["label_stage1"], errors="coerce").eq(0)
        mask = numeric_mask | labels.isin(["0", "authentic", "true", "yes"])
    else:
        raise ValueError("Authenticity filtering requires an 'authentic' or 'label_stage1' column.")
    return df.loc[mask].reset_index(drop=True)


def resolve_encoder():
    if (STAGE1_MODEL_DIR / "config.json").exists():
        return str(STAGE1_MODEL_DIR)
    return BASE_MODEL


def extract_aspect_spans(labels, valid_mask, offsets=None):
    spans = []
    start = None
    for index, label in enumerate(labels.tolist()):
        if not bool(valid_mask[index]):
            label = 0
        if label == 1:
            if start is not None:
                spans.append((start, index - 1))
            start = index
        elif label == 2:
            if start is None:
                start = index
        elif start is not None:
            spans.append((start, index))
            start = None
    if start is not None:
        spans.append((start, len(labels) - 1))

    if offsets is None:
        return spans
    return [
        (start, end)
        for start, end in spans
        if int(offsets[end][1]) > int(offsets[start][0])
    ]


class ABSAHeadModel(nn.Module):
    def __init__(self, encoder_name_or_dir, dropout=0.1):
        super().__init__()

        self.encoder_name = encoder_name_or_dir
        self.encoder_local = Path(encoder_name_or_dir).is_dir()
        self.encoder = AutoModel.from_pretrained(encoder_name_or_dir)
        self.hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(p=dropout)
        self.aspect_head = nn.Linear(self.hidden_size, len(ASPECT_TOKENS))
        self.sentiment_head = nn.Linear(self.hidden_size, len(POLARITY_LABELS))

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        hidden = self.dropout(hidden)
        return self.aspect_head(hidden), self.sentiment_head(hidden)

    def compute_loss(self, aspect_logits, sentiment_logits, aspect_targets, sentiment_targets):
        aspect_loss = F.cross_entropy(
            aspect_logits.reshape(-1, len(ASPECT_TOKENS)),
            aspect_targets.reshape(-1).long(),
            ignore_index=SENTIMENT_IGNORE,
        )
        sentiment_mask = sentiment_targets != SENTIMENT_IGNORE
        if sentiment_mask.any():
            sentiment_loss = F.cross_entropy(
                sentiment_logits.reshape(-1, len(POLARITY_LABELS))[sentiment_mask.reshape(-1)],
                sentiment_targets.reshape(-1).long()[sentiment_mask.reshape(-1)],
            )
        else:
            sentiment_loss = torch.tensor(0.0, device=aspect_logits.device, dtype=aspect_logits.dtype)
        return aspect_loss + sentiment_loss

    @torch.no_grad()
    def predict(self, input_ids, attention_mask, offset_mapping=None, tokenizer=None, texts=None, threshold=0.5):
        self.eval()
        if isinstance(texts, str):
            texts = [texts]
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        hidden = self.dropout(hidden)
        aspect_logits = self.aspect_head(hidden)
        aspect_probs = F.softmax(aspect_logits, dim=-1)
        aspect_labels = torch.argmax(aspect_probs, dim=-1)
        input_ids_cpu = input_ids.detach().cpu()
        attention_cpu = attention_mask.detach().cpu().bool()
        if offset_mapping is None:
            offsets_cpu = None
            valid_mask = attention_cpu
        else:
            offsets_cpu = offset_mapping.detach().cpu()
            valid_mask = attention_cpu & (offsets_cpu[:, :, 1] > offsets_cpu[:, :, 0])

        results = []
        for row_index in range(input_ids_cpu.shape[0]):
            row_aspects = []
            spans = extract_aspect_spans(
                aspect_labels[row_index].detach().cpu(),
                valid_mask[row_index],
                offsets_cpu[row_index] if offsets_cpu is not None else None,
            )
            for start, end in spans:
                span_probabilities = aspect_probs[row_index, start : end + 1, 1:]
                aspect_confidence = float(span_probabilities.max().item())
                if aspect_confidence < threshold:
                    continue
                token_ids = input_ids_cpu[row_index, start : end + 1]
                if texts is not None and offsets_cpu is not None:
                    original_text = str(texts[row_index])
                    start_char = int(offsets_cpu[row_index, start, 0])
                    end_char = int(offsets_cpu[row_index, end, 1])
                    aspect = original_text[start_char:end_char].strip()
                elif tokenizer is None:
                    aspect = f"span_{start}_{end}"
                else:
                    aspect = tokenizer.decode(
                        token_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    ).strip()
                if not aspect:
                    continue
                span_hidden = hidden[row_index, start : end + 1].mean(dim=0)
                span_sentiment_probs = F.softmax(self.sentiment_head(span_hidden), dim=-1)
                sentiment_index = int(torch.argmax(span_sentiment_probs).item())
                row_aspects.append({
                    "aspect": aspect,
                    "sentiment": POLARITY_LABELS[sentiment_index],
                    "sentiment_confidence": round(float(span_sentiment_probs[sentiment_index].item()), 4),
                    "sentiment_probabilities": {
                        label: round(float(span_sentiment_probs[index].item()), 4)
                        for index, label in enumerate(POLARITY_LABELS)
                    },
                    "aspect_confidence": round(aspect_confidence, 4),
                    "start_token": start,
                    "end_token": end,
                })
            results.append(row_aspects)
        return results

    def save_pretrained(self, save_dir):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), save_dir / "model.pt")
        with open(save_dir / "absa_config.json", "w", encoding="utf-8") as file:
            json.dump(
                {
                    "model_version": MODEL_VERSION,
                    "hidden_size": self.hidden_size,
                    "encoder": self.encoder_name,
                    "encoder_local": self.encoder_local,
                },
                file,
            )

    @classmethod
    def from_pretrained(cls, model_dir, **kwargs):
        model_dir = Path(model_dir)
        with open(model_dir / "absa_config.json", "r", encoding="utf-8") as file:
            saved = json.load(file)
        if saved.get("model_version") != MODEL_VERSION:
            raise ValueError(
                f"Unsupported ABSA model version {saved.get('model_version')!r}; "
                f"expected {MODEL_VERSION!r}. Retrain with fine_tune_absa.py."
            )
        encoder = saved.get("encoder") or resolve_encoder()
        if saved.get("encoder_local", False) and not Path(encoder).exists():
            encoder = resolve_encoder()
        model = cls(encoder, **kwargs)
        model.load_state_dict(torch.load(model_dir / "model.pt", map_location=torch.device("cpu")))
        return model
