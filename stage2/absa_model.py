"""RoBERTa-based Aspect-Based Sentiment Analysis (ABSA) model.

Implements the two ABSA heads described in Chapter 8 of the thesis:
  - Aspect detection head: linear(H -> K) + per-aspect sigmoid.
  - Aspect-conditioned sentiment head: linear(H -> 3) + softmax.

Both heads operate on token representations produced by the same
fine-tuned DOST-RoBERTa encoder used in Stage 1, so the encoder weights are
reused from `models/dost_roberta` (falling back to the DOST-ASTI base model).
"""

import json
from pathlib import Path

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

ASPECTS = ["quality", "price", "delivery", "other"]
POLARITY_LABELS = ["Negative", "Neutral", "Positive"]
SENTIMENT_IGNORE = -100


def resolve_encoder():
    if (STAGE1_MODEL_DIR / "config.json").exists():
        return str(STAGE1_MODEL_DIR)
    return BASE_MODEL


class ABSAHeadModel(nn.Module):
    def __init__(self, encoder_name_or_dir, aspects=None, dropout=0.1):
        super().__init__()
        self.aspects = list(aspects) if aspects is not None else list(ASPECTS)
        self.num_aspects = len(self.aspects)

        self.encoder_name = encoder_name_or_dir
        self.encoder_local = Path(encoder_name_or_dir).is_dir()
        self.encoder = AutoModel.from_pretrained(encoder_name_or_dir)
        self.hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(p=dropout)
        self.aspect_head = nn.Linear(self.hidden_size, self.num_aspects)
        self.aspect_embeddings = nn.Embedding(self.num_aspects, self.hidden_size)
        self.sentiment_head = nn.Linear(self.hidden_size, len(POLARITY_LABELS))

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = hidden.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)

        aspect_logits = self.aspect_head(pooled)

        aspect_idx = torch.arange(self.num_aspects, device=pooled.device)
        aspect_emb = self.aspect_embeddings(aspect_idx)
        conditioned = pooled.unsqueeze(1) + aspect_emb.unsqueeze(0)
        sentiment_logits = self.sentiment_head(conditioned)

        return aspect_logits, sentiment_logits

    def compute_loss(self, aspect_logits, sentiment_logits, aspect_targets, sentiment_targets):
        aspect_loss = F.binary_cross_entropy_with_logits(aspect_logits, aspect_targets)

        flat_logits = sentiment_logits.reshape(-1, len(POLARITY_LABELS))
        flat_targets = sentiment_targets.view(-1).long()
        valid = flat_targets != SENTIMENT_IGNORE
        if valid.any():
            sentiment_loss = F.cross_entropy(flat_logits[valid], flat_targets[valid], reduction="mean")
        else:
            sentiment_loss = torch.tensor(0.0, device=aspect_logits.device)

        return aspect_loss + sentiment_loss

    @torch.no_grad()
    def predict(self, input_ids, attention_mask, threshold=0.5):
        self.eval()
        aspect_logits, sentiment_logits = self.forward(input_ids, attention_mask)

        aspect_probs = torch.sigmoid(aspect_logits)
        sentiment_probs = F.softmax(sentiment_logits, dim=-1)
        sentiment_idx = torch.argmax(sentiment_probs, dim=-1)

        results = []
        for sample_probs, sample_sent_probs, sample_sent_idx in zip(
            aspect_probs, sentiment_probs, sentiment_idx
        ):
            aspects = []
            for i, name in enumerate(self.aspects):
                probs = sample_probs[i].item()
                entry = {
                    "aspect": name,
                    "present": bool(probs >= threshold),
                    "aspect_probability": round(probs, 4),
                }
                if entry["present"]:
                    entry["sentiment"] = POLARITY_LABELS[sample_sent_idx[i].item()]
                    entry["sentiment_confidence"] = round(sample_sent_probs[i].max().item(), 4)
                    entry["sentiment_probabilities"] = {
                        label: round(sample_sent_probs[i][j].item(), 4)
                        for j, label in enumerate(POLARITY_LABELS)
                    }
                aspects.append(entry)
            results.append(aspects)
        return results

    def save_pretrained(self, save_dir):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), save_dir / "model.pt")
        with open(save_dir / "absa_config.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "aspects": self.aspects,
                    "hidden_size": self.hidden_size,
                    "encoder": self.encoder_name,
                    "encoder_local": self.encoder_local,
                },
                f,
            )

    @classmethod
    def from_pretrained(cls, model_dir, **kwargs):
        model_dir = Path(model_dir)
        with open(model_dir / "absa_config.json", "r", encoding="utf-8") as f:
            saved = json.load(f)
        encoder = saved.get("encoder") or resolve_encoder()
        if saved.get("encoder_local", False) and not Path(encoder).exists():
            encoder = resolve_encoder()
        model = cls(encoder, aspects=saved["aspects"], **kwargs)
        model.encoder_name = saved.get("encoder", encoder)
        model.encoder_local = saved.get("encoder_local", False)
        model.load_state_dict(torch.load(model_dir / "model.pt", map_location=torch.device("cpu")))
        return model