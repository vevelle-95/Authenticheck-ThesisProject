"""ResNet50 spatial regions; frozen by default, with no fabricated buyer images."""

from pathlib import Path

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


class ImageEncoder(nn.Module):
    output_dim = 2048

    def __init__(self, frozen=True, cache_dir=None, pretrained=True):
        super().__init__()
        backbone = resnet50(weights=None)
        if pretrained:
            weights = ResNet50_Weights.DEFAULT
            state = torch.hub.load_state_dict_from_url(
                weights.url, model_dir=str(Path(cache_dir) / "resnet"), progress=True,
                check_hash=True, map_location="cpu",
            )
            backbone.load_state_dict(state)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.frozen = frozen
        self.backbone.requires_grad_(not frozen)

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.backbone.eval()
        return self

    def forward(self, images):
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.frozen):
            return self.backbone(images).flatten(2).transpose(1, 2)
