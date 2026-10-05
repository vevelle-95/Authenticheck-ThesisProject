"""Adapted CLIP-CA-CG: cached CLIP, RoBERTa/Bi-GRU, regions, attention, aspect heads."""

import torch
from torch import nn

from datasets.authenticheck_data import ASPECTS, SENTIMENTS
from models.encoders.text_encoder import TextEncoder
from models.fusion.cross_attention import CrossAttention
from models.fusion.gating import Gating
from runtime import resolve_path


def masked_mean(sequence, mask):
    weights = mask.unsqueeze(-1).to(sequence.dtype)
    return (sequence * weights).sum(1) / weights.sum(1).clamp_min(1)


class CLIPCACG(nn.Module):
    def __init__(self, config, clip_dimension=512, pretrained=True, text_config=None,
                 text_encoder=None, image_encoder=None):
        super().__init__()
        settings = config["model"]
        if settings.get("image_model", "resnet50") != "resnet50":
            raise ValueError("This implementation supports image_model: resnet50")
        dim = settings["embedding_dim"]
        if dim % 2 or dim % settings["attention_heads"]:
            raise ValueError("embedding_dim must be even and divisible by attention_heads")
        cache_dir = str(resolve_path(config["output"]["model_cache"]))
        self.text_encoder = text_encoder if text_encoder is not None else TextEncoder(
            settings["text_model"], dim, settings.get("freeze_text_encoder", False),
            cache_dir, pretrained, text_config,
        )
        if image_encoder is None:
            from models.encoders.image_encoder import ImageEncoder
            image_encoder = ImageEncoder(settings.get("freeze_image_encoder", True),
                                         cache_dir, pretrained)
        self.image_encoder = image_encoder
        self.image_proj = nn.Linear(self.image_encoder.output_dim, dim)
        self.clip_joint = nn.Linear(clip_dimension, dim, bias=False)
        self.cross_attn = CrossAttention(dim, settings["attention_heads"], settings["dropout"])
        self.gate = Gating(dim, settings["dropout"])
        self.dropout = nn.Dropout(settings["dropout"])
        self.aspect_head = nn.Linear(dim, len(ASPECTS))
        self.aspect_embeddings = nn.Embedding(len(ASPECTS), dim)
        self.sentiment_head = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(),
            nn.Dropout(settings["dropout"]), nn.Linear(dim, len(SENTIMENTS)),
        )

    def forward(self, input_ids, attention_mask, images, image_mask, clip_text, clip_image):
        text_mask = attention_mask.bool()
        text = self.text_encoder(input_ids, attention_mask)
        batch, count, channels, height, width = images.shape
        real = image_mask.bool()
        if real.ndim == 3:
            real = real.squeeze(-1)
        flat_images = images.reshape(batch * count, channels, height, width)
        valid = real.flatten()
        if valid.any():
            extracted = self.image_encoder(flat_images[valid])
            regions = extracted.shape[1]
            visual = extracted.new_zeros((batch * count, regions, extracted.shape[-1]))
            visual[valid] = extracted
            visual = visual.reshape(batch, count * regions, -1)
            region_mask = real.unsqueeze(-1).expand(-1, -1, regions).reshape(batch, -1)
        else:
            visual = text.new_zeros((batch, 1, self.image_encoder.output_dim))
            region_mask = real.new_zeros((batch, 1))
        present = region_mask.any(1)
        image = self.image_proj(visual) * region_mask.unsqueeze(-1)
        joint = self.clip_joint(clip_text * clip_image) * present[:, None]
        original_text = masked_mean(text, text_mask)
        original_image = masked_mean(image, region_mask)
        text, image = self.cross_attn(text, image, text_mask, region_mask, joint)
        fused = self.dropout(self.gate(masked_mean(text, text_mask),
                                      masked_mean(image, region_mask), joint, present,
                                      original_text, original_image))
        conditioned = fused[:, None, :] + self.aspect_embeddings.weight[None, :, :]
        return {
            "aspect_logits": self.aspect_head(fused),
            "sentiment_logits": self.sentiment_head(conditioned),
        }
