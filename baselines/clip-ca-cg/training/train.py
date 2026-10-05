"""Train on ground-truth Authentic training reviews; select checkpoints on validation."""

import copy
import importlib.metadata
import json
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from datasets.authenticheck_data import ASPECTS, MODEL_VERSION, SENTIMENTS, fingerprint
from datasets.clip_cache import ClipFeatureCache
from datasets.multimodal_dataset import MultiModalDataset, multimodal_collate_fn
from models.model import CLIPCACG
from runtime import resolve_path, select_device, write_json
from training.engine import Trainer
from training.eval import collect_predictions, select_threshold


def main(assigned, config, base_dir=None, *, tokenizer=None, model=None):
    seed = config["training"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    train = assigned[assigned.partition.eq("train") & assigned.label.eq(0)]
    validation = assigned[assigned.partition.eq("validation") & assigned.label.eq(0)]
    if train.empty or validation.empty:
        raise ValueError("Both training and validation need ground-truth Authentic reviews")
    if config["training"]["epochs"] < 1:
        raise ValueError("epochs must be at least 1")
    config = copy.deepcopy(config)
    cache = ClipFeatureCache(config)
    for row in train.to_dict("records") + validation.to_dict("records"):
        cache.get(row)
    device = select_device(config)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(config["model"]["text_model"],
                                                  cache_dir=str(resolve_path(config["output"]["model_cache"])))
    model = model if model is not None else CLIPCACG(config, cache.dimension)
    model.to(device)
    loaders = [
        DataLoader(MultiModalDataset(frame, tokenizer, cache, config),
                   batch_size=config["training"]["batch_size"], shuffle=shuffle,
                   num_workers=config["training"]["num_workers"], drop_last=False,
                   collate_fn=multimodal_collate_fn, pin_memory=device.type == "cuda")
        for frame, shuffle in [(train, True), (validation, False)]
    ]
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                 lr=float(config["training"]["lr"]), weight_decay=0.01)
    trainer = Trainer(model, optimizer, device,
                      config["training"]["gradient_accumulation_steps"],
                      config["training"]["mixed_precision"])
    directory = resolve_path(config["output"]["checkpoint_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    if hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(directory / "tokenizer")
    metadata = {
        "dataset_sha256": fingerprint(assigned),
        "fit_review_ids": train.review_id.tolist(),
        "validation_review_ids": validation.review_id.tolist(),
        "fit_product_ids": sorted(train.product_id.unique().tolist()),
        "validation_product_ids": sorted(validation.product_id.unique().tolist()),
        "authentic_only": True, "seed": seed,
        "clip_encoder": cache.metadata,
        "library_versions": {name: importlib.metadata.version(name)
                             for name in ("torch", "torchvision", "transformers", "numpy", "pandas")},
        "input": "review_text and buyer images; no quality labels, stars, or listing metadata",
        "sentiment_targets": "distribution over distinct annotated polarities per category",
        "evidence_policy": "full review context at prediction; no learned span extraction",
    }
    best_score, history = -1.0, []
    for epoch in range(1, config["training"]["epochs"] + 1):
        started = time.perf_counter()
        loss = trainer.train_one_epoch(loaders[0])
        validation_outputs = collect_predictions(model, loaders[1], device)
        threshold, metrics = select_threshold(validation_outputs)
        score = (metrics["category_detection"]["macro_f1"]
                 + metrics["gold_category_sentiment"]["macro_f1"]) / 2
        entry = {"epoch": epoch, "loss": loss, "selection_score": score,
                 "seconds": time.perf_counter() - started, "validation": metrics}
        history.append(entry)
        print(json.dumps(entry), flush=True)
        if score > best_score:
            best_score = score
            checkpoint = {
                "model_version": MODEL_VERSION, "aspects": list(ASPECTS), "sentiments": list(SENTIMENTS),
                "model_state_dict": model.state_dict(), "config": config,
                "text_backbone_config": model.text_encoder.backbone.config.to_dict(),
                "clip_dimension": cache.dimension, "threshold": threshold,
                "epoch": epoch, "validation_metrics": metrics, "training_metadata": metadata,
            }
            temporary = directory / "best.pt.tmp"
            torch.save(checkpoint, temporary)
            temporary.replace(directory / "best.pt")
        write_json(directory / "training_history.json", history)
    write_json(directory / "training_metadata.json", metadata)
    return directory / "best.pt"
