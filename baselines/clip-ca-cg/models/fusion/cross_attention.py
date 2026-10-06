"""Residual cross-attention over real tokens/regions with explicit padding masks."""

import torch
from torch import nn


class CrossAttention(nn.Module):
    def __init__(self, dim=256, heads=4, dropout=0.1):
        super().__init__()
        self.text_to_image = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.image_to_text = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.text_norm = nn.LayerNorm(dim)
        self.image_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, image, text_mask, image_mask, joint):
        present = image_mask.any(1)
        text_result = text.clone()
        image_result = torch.zeros_like(image)
        # Never submit an entirely masked key sequence to MultiheadAttention.
        if present.any():
            t, i = text[present], image[present]
            context = joint[present, None, :]
            delta_t, _ = self.text_to_image(
                t, i + context, i + context,
                key_padding_mask=~image_mask[present], need_weights=False,
            )
            delta_i, _ = self.image_to_text(
                i, t + context, t + context,
                key_padding_mask=~text_mask[present], need_weights=False,
            )
            text_result[present] = (t + self.dropout(delta_t)).to(text_result.dtype)
            image_result[present] = self.image_norm(i + self.dropout(delta_i)).to(image_result.dtype)
        return (self.text_norm(text_result) * text_mask.unsqueeze(-1),
                image_result * image_mask.unsqueeze(-1))
