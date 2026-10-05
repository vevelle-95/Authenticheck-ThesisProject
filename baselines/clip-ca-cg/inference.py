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


def predict_frame(frame, checkpoint_path, prepare_cache=False, *, tokenizer=None, model=None):
    checkpoint = load_checkpoint(checkpoint_path)
    config = checkpoint["config"]
    cache = ClipFeatureCache(config)
    if prepare_cache:
        cache.prepare(frame)
    expected = checkpoint["training_metadata"]["clip_encoder"]
    if cache.metadata != expected:
        raise ValueError("Prediction CLIP cache must use the exact encoder revision/configuration used for training")
    device = select_device(config)
    if tokenizer is None:
        tokenizer_path = resolve_path(config["output"]["checkpoint_dir"]) / "tokenizer"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    if model is None:
        model = CLIPCACG(config, checkpoint["clip_dimension"], pretrained=False,
                        text_config=checkpoint["text_backbone_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    loader = DataLoader(MultiModalDataset(frame, tokenizer, cache, config),
                        batch_size=config["training"]["batch_size"], shuffle=False,
                        num_workers=config["training"]["num_workers"],
                        collate_fn=multimodal_collate_fn)
    raw = collect_predictions(model, loader, device)
    rows = [
        {"id": row["review_id"], "aspects": format_aspects(
            row["review_text"], raw["aspect_probabilities"][index],
            raw["sentiment_probabilities"][index], checkpoint["threshold"],
        )}
        for index, row in enumerate(frame.to_dict("records"))
    ]
    return {"reviews": rows}, raw, checkpoint
