"""Joint-feature gating with original text/image context branches."""

import torch
from torch import nn


class Gating(nn.Module):
    def __init__(self, dim=256, dropout=0.1):
        super().__init__()
        self.gate = nn.Linear(dim * 2, dim)
        self.mix = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.Dropout(dropout))
        self.image_context = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU())
        self.text_context = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU())
        self.balance = nn.Linear(dim * 2, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, text, image, joint, present, original_text, original_image):
        weight = torch.sigmoid(self.gate(torch.cat([text, image], dim=-1)))
        interaction = weight * text + (1 - weight) * image
        mixed = self.mix(torch.cat([interaction, joint], dim=-1))
        image_branch = self.image_context(torch.cat([original_image, mixed], dim=-1))
        text_branch = self.text_context(torch.cat([original_text, mixed], dim=-1))
        balance = torch.sigmoid(self.balance(torch.cat([image_branch, text_branch], dim=-1)))
        fused = balance * image_branch + (1 - balance) * text_branch
        return self.norm(torch.where(present[:, None], fused, text_branch))
