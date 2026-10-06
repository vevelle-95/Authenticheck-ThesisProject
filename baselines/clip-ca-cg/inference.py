"""Native category JSON and isolated checkpoint inference; no keyword projection."""

import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from datasets.authenticheck_data import ASPECTS, SENTIMENTS
from datasets.clip_cache import ClipFeatureCache
from datasets.multimodal_dataset import MultiModalDataset, multimodal_collate_fn
from models.model import CLIPCACG
from runtime import load_checkpoint, resolve_path, select_device
from training.eval import collect_predictions


def format_aspects(review_text, aspect_probabilities, sentiment_probabilities, threshold):
    return [
        {"category": category, "text": review_text,
         "sentiment": SENTIMENTS[int(np.argmax(sentiment_probabilities[index]))]}
        for index, category in enumerate(ASPECTS) if aspect_probabilities[index] >= threshold
    ]


class BaselinePredictor:
    """Load a trained checkpoint once and reuse it across prediction requests."""

    def __init__(self, checkpoint_path, *, tokenizer=None, model=None):
        self.checkpoint = load_checkpoint(checkpoint_path)
        self.config = self.checkpoint["config"]
        self.cache = ClipFeatureCache(self.config)
        if self.cache.metadata is not None:
            self.validate_cache()
        self.device = select_device(self.config)
        if tokenizer is None:
            tokenizer_path = resolve_path(self.config["output"]["checkpoint_dir"]) / "tokenizer"
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        self.tokenizer = tokenizer
        if model is None:
            model = CLIPCACG(self.config, self.checkpoint["clip_dimension"], pretrained=False,
                            text_config=self.checkpoint["text_backbone_config"])
        model.load_state_dict(self.checkpoint["model_state_dict"])
        self.model = model.to(self.device).eval()

    def validate_cache(self):
        if self.cache.metadata is None:
            raise FileNotFoundError("Missing CLIP encoder metadata; restore outputs/clip_cache/encoder.json or prepare CLIP features before starting the API")
        expected = self.checkpoint["training_metadata"]["clip_encoder"]
        if self.cache.metadata != expected:
            raise ValueError("Prediction CLIP cache must use the exact encoder revision/configuration used for training")

    def predict(self, frame, prepare_cache=False):
        if prepare_cache:
            self.cache.prepare(frame)
        self.validate_cache()
        loader = DataLoader(MultiModalDataset(frame, self.tokenizer, self.cache, self.config),
                            batch_size=self.config["training"]["batch_size"], shuffle=False,
                            num_workers=self.config["training"]["num_workers"],
                            collate_fn=multimodal_collate_fn)
        raw = collect_predictions(self.model, loader, self.device)
        rows = [
            {"id": row["review_id"], "aspects": format_aspects(
                row["review_text"], raw["aspect_probabilities"][index],
                raw["sentiment_probabilities"][index], self.checkpoint["threshold"],
            )}
            for index, row in enumerate(frame.to_dict("records"))
        ]
        return {"reviews": rows}, raw, self.checkpoint


def predict_frame(frame, checkpoint_path, prepare_cache=False, *, tokenizer=None, model=None):
    predictor = BaselinePredictor(checkpoint_path, tokenizer=tokenizer, model=model)
    return predictor.predict(frame, prepare_cache=prepare_cache)
