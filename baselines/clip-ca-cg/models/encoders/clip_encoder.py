"""Frozen pretrained CLIP used only when building the feature cache."""

import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor


class FrozenCLIPEncoder:
    def __init__(self, model_name, revision, device, cache_dir):
        self.device = torch.device(device)
        options = {'revision': revision, 'cache_dir': str(cache_dir)}
        self.model = CLIPModel.from_pretrained(model_name, **options).to(self.device)
        self.model.requires_grad_(False)
        self.model.eval()
        self.processor = CLIPProcessor.from_pretrained(model_name, **options)
        self.dimension = self.model.config.projection_dim
        self.revision = self.model.config._commit_hash or revision

    @torch.no_grad()
    def encode(self, text, images):
        tokens = self.processor(text=[text], padding=True, truncation=True,
                                max_length=self.model.config.text_config.max_position_embeddings,
                                return_tensors='pt').to(self.device)
        text_vector = F.normalize(self.model.get_text_features(**tokens).float(), dim=-1)[0]
        if images:
            pixels = self.processor(images=images, return_tensors='pt').to(self.device)
            image_vectors = F.normalize(self.model.get_image_features(**pixels).float(), dim=-1)
            image_vector = F.normalize(image_vectors.mean(0), dim=0)
            score = float((image_vectors @ text_vector).max())
        else:
            image_vector = torch.zeros_like(text_vector)
            score = 0.0
        return {
            'text': text_vector.cpu().numpy().astype(np.float32),
            'image': image_vector.cpu().numpy().astype(np.float32),
            'clip_score': score,
        }
