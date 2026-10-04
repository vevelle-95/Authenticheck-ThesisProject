"""Fixed ten-category detection and category-conditioned three-way sentiment."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

from model_contract import (
    ASPECTS, BASE_MODEL, DATA_DIR, DEFAULT_BUNDLE, MAX_LENGTH, POLARITIES,
    SENSORY_POLICY, SENTIMENT_TARGET_POLICY, TAXONOMY_VERSION, aspect_prompt, map_polarity,
)

MODELS_DIR = DEFAULT_BUNDLE
STAGE1_MODEL_DIR = DEFAULT_BUNDLE / "dost_roberta"
DEFAULT_ABSA_MODEL_DIR = DEFAULT_BUNDLE / "absa_model"
DEFAULT_DATA_PATH = DATA_DIR / "test_reviews.csv"
DEFAULT_RESULTS_PATH = DATA_DIR / "absa_sentiment_results.json"
ABSA_MAX_LENGTH = MAX_LENGTH
MODEL_VERSION = "fixed-category-v2"
POLARITY_LABELS = list(POLARITIES)
SENTIMENT_IGNORE = -100
polarity_index = map_polarity


def resolve_encoder():
    return BASE_MODEL


def tokenize_detection(tokenizer, texts, max_length):
    return tokenizer(list(texts), padding="max_length", truncation=True,
                     max_length=max_length, return_tensors="pt")


def tokenize_sentiment(tokenizer, texts, categories, max_length):
    if len(texts) != len(categories):
        raise ValueError("Each sentiment instance requires a target category")
    return tokenizer(
        [aspect_prompt(category) for category in categories], list(texts),
        padding="max_length", truncation="only_second", max_length=max_length, return_tensors="pt",
    )


class ABSAHeadModel(nn.Module):
    def __init__(self, encoder_name_or_dir=BASE_MODEL, dropout=0.1, *, encoder=None,
                 threshold=0.5, max_length=ABSA_MAX_LENGTH):
        super().__init__()
        if not 0 < threshold < 1:
            raise ValueError("Aspect threshold must lie strictly between 0 and 1")
        self.encoder_name = str(encoder_name_or_dir)
        self.encoder = encoder if encoder is not None else AutoModel.from_pretrained(encoder_name_or_dir)
        self.hidden_size = self.encoder.config.hidden_size
        self.dropout_rate = dropout
        self.dropout = nn.Dropout(dropout)
        self.aspect_head = nn.Linear(self.hidden_size, len(ASPECTS))
        self.sentiment_head = nn.Linear(self.hidden_size, len(POLARITIES))
        self.threshold = threshold
        self.max_length = max_length
        self.tokenizer_dir = None

    def _pooled(self, input_ids, attention_mask):
        return self.dropout(self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0])

    def forward(self, input_ids, attention_mask, aspect_input_ids=None, aspect_attention_mask=None):
        detection = self.aspect_head(self._pooled(input_ids, attention_mask))
        sentiment = None
        if aspect_input_ids is not None and len(aspect_input_ids):
            sentiment = self.sentiment_head(self._pooled(aspect_input_ids, aspect_attention_mask))
        return detection, sentiment

    def compute_loss(self, aspect_logits, sentiment_logits, aspect_targets, sentiment_targets, eligible):
        loss = F.binary_cross_entropy_with_logits(aspect_logits, aspect_targets, reduction="none")
        loss = (loss * eligible).sum() / eligible.sum().clamp_min(1)
        if sentiment_logits is not None and sentiment_targets.numel():
            if sentiment_targets.ndim == 2:
                sentiment_targets = sentiment_targets / sentiment_targets.sum(-1, keepdim=True).clamp_min(1)
            else:
                sentiment_targets = sentiment_targets.long()
            loss = loss + F.cross_entropy(sentiment_logits, sentiment_targets)
        return loss

    @torch.inference_mode()
    def predict_sentiment(self, texts, categories, tokenizer, batch_size=16):
        self.eval()
        outputs = []
        device = next(self.parameters()).device
        for start in range(0, len(texts), batch_size):
            enc = tokenize_sentiment(tokenizer, texts[start:start + batch_size],
                                     categories[start:start + batch_size], self.max_length).to(device)
            outputs.extend(torch.softmax(self.sentiment_head(self._pooled(
                enc["input_ids"], enc["attention_mask"],
            )), dim=-1).cpu().tolist())
        return outputs

    @torch.inference_mode()
    def predict(self, texts, tokenizer, product_categories=None, threshold=None, batch_size=16):
        # Legacy metadata is accepted but never used to suppress a category.
        self.eval()
        if isinstance(texts, str):
            texts = [texts]
        threshold = self.threshold if threshold is None else threshold
        if not 0 < threshold < 1:
            raise ValueError("Aspect threshold must lie strictly between 0 and 1")
        device = next(self.parameters()).device
        results = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            enc = tokenize_detection(tokenizer, batch, self.max_length).to(device)
            detection = torch.sigmoid(self.aspect_head(self._pooled(
                enc["input_ids"], enc["attention_mask"],
            ))).cpu()
            candidates = []
            rows = [[] for _ in batch]
            for row, scores in enumerate(detection):
                for index, category in enumerate(ASPECTS):
                    if float(scores[index]) >= threshold:
                        candidates.append((row, category, float(scores[index])))
            probabilities = self.predict_sentiment(
                [batch[row] for row, _, _ in candidates],
                [category for _, category, _ in candidates], tokenizer, batch_size,
            )
            for (row, category, confidence), values in zip(candidates, probabilities):
                polarity = max(range(3), key=lambda index: values[index])
                rows[row].append({
                    "category": category, "aspect": category, "sentiment": POLARITIES[polarity],
                    "aspect_confidence": round(confidence, 4),
                    "sentiment_confidence": round(values[polarity], 4),
                    "sentiment_probabilities": dict(zip(POLARITIES, (round(value, 4) for value in values))),
                })
            results.extend(rows)
        return results

    def save_pretrained(self, save_dir, tokenizer):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.encoder.config.save_pretrained(save_dir / "encoder")
        tokenizer.save_pretrained(save_dir / "tokenizer")
        torch.save(self.state_dict(), save_dir / "model.pt")
        config = {
            "model_version": MODEL_VERSION, "taxonomy_version": TAXONOMY_VERSION,
            "sensory_domain_policy": SENSORY_POLICY,
            "sentiment_target_policy": SENTIMENT_TARGET_POLICY,
            "aspects": list(ASPECTS), "polarities": list(POLARITIES), "encoder": self.encoder_name,
            "hidden_size": self.hidden_size, "dropout": self.dropout_rate,
            "threshold": self.threshold, "max_length": self.max_length,
        }
        (save_dir / "absa_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    @classmethod
    def from_pretrained(cls, model_dir):
        model_dir = Path(model_dir)
        saved = json.loads((model_dir / "absa_config.json").read_text(encoding="utf-8"))
        if saved.get("model_version") != MODEL_VERSION:
            raise ValueError(f"ABSA checkpoint is {saved.get('model_version')!r}; retrain for {MODEL_VERSION}. Old span weights remain intact.")
        if saved.get("taxonomy_version") != TAXONOMY_VERSION or saved.get("aspects") != list(ASPECTS) or saved.get("polarities") != list(POLARITIES):
            raise ValueError("ABSA taxonomy/polarity order mismatch")
        if saved.get("sensory_domain_policy") != SENSORY_POLICY:
            raise ValueError("ABSA sensory policy mismatch; retrain without product-category gating")
        if saved.get("sentiment_target_policy") != SENTIMENT_TARGET_POLICY:
            raise ValueError("ABSA sentiment target policy mismatch; retrain with mixed-polarity support")
        config = AutoConfig.from_pretrained(model_dir / "encoder", local_files_only=True)
        model = cls(saved["encoder"], saved["dropout"], encoder=AutoModel.from_config(config),
                    threshold=saved["threshold"], max_length=saved["max_length"])
        model.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True))
        model.tokenizer_dir = model_dir / "tokenizer"
        return model
