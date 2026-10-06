"""Aspect BCE plus masked soft-label sentiment loss, with CUDA AMP and accumulation."""

import torch
import torch.nn.functional as F

MODEL_INPUTS = ("input_ids", "attention_mask", "images", "image_mask", "clip_text", "clip_image")


def move_inputs(batch, device):
    return {name: batch[name].to(device) for name in MODEL_INPUTS}


def aspect_sentiment_loss(outputs, aspect_targets, sentiment_targets):
    detection = F.binary_cross_entropy_with_logits(outputs["aspect_logits"], aspect_targets)
    active = sentiment_targets.sum(-1) > 0
    losses = -(sentiment_targets * F.log_softmax(outputs["sentiment_logits"], dim=-1)).sum(-1)
    sentiment = losses[active].mean() if active.any() else outputs["sentiment_logits"].sum() * 0
    return detection + sentiment


class Trainer:
    def __init__(self, model, optimizer, device, accumulation_steps=1, mixed_precision=True):
        self.model = model
        self.optimizer = optimizer
        self.device = torch.device(device)
        self.accumulation = max(1, accumulation_steps)
        self.amp = self.device.type == "cuda" and mixed_precision
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)

    def train_one_epoch(self, dataloader):
        if not len(dataloader):
            raise ValueError("Training partition is empty")
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        total, count = 0.0, 0
        for index, batch in enumerate(dataloader):
            with torch.autocast(device_type=self.device.type, enabled=self.amp):
                outputs = self.model(**move_inputs(batch, self.device))
                loss = aspect_sentiment_loss(outputs, batch["aspect_targets"].to(self.device),
                                              batch["sentiment_targets"].to(self.device))
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            group_start = (index // self.accumulation) * self.accumulation
            group_size = min(self.accumulation, len(dataloader) - group_start)
            self.scaler.scale(loss / group_size).backward()
            if (index + 1) % self.accumulation == 0 or index + 1 == len(dataloader):
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
            size = len(batch["review_ids"])
            total += float(loss.detach()) * size
            count += size
        return total / count
