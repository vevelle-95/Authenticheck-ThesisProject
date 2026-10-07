"""The exact six-feature implementation shared by offline and online inference."""

import ipaddress
import json
import socket
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import requests
from PIL import Image

from model_contract import CLASS_NAMES, INPUT_VERSION, normalize_rating, parse_image_urls

MAX_IMAGE_BYTES = 5 * 1024 * 1024


def image_url_is_safe(url):
    try:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            return False
        return all(ipaddress.ip_address(address[4][0]).is_global for address in
                   socket.getaddrinfo(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)))
    except (OSError, ValueError):
        return False


def load_image(url):
    if not image_url_is_safe(url):
        return None
    try:
        with requests.get(url, timeout=(3, 7), stream=True, allow_redirects=False) as response:
            if response.status_code != 200 or not response.headers.get("Content-Type", "").lower().startswith("image/"):
                return None
            if int(response.headers.get("Content-Length", "0") or 0) > MAX_IMAGE_BYTES:
                return None
            buffer = BytesIO()
            for chunk in response.iter_content(64 * 1024):
                buffer.write(chunk)
                if buffer.tell() > MAX_IMAGE_BYTES:
                    return None
            buffer.seek(0)
            image = Image.open(buffer)
            image.load()
            return image.convert("RGB")
    except (requests.RequestException, OSError, ValueError):
        return None


class FeatureExtractor:
    def __init__(self, model_dir, image_loader=load_image):
        self.model_dir = Path(model_dir)
        self.image_loader = image_loader
        self.tokenizer = self.model = self.clip_text = self.clip_image = None

    def load_text(self):
        if self.model is None:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
            if getattr(self.model.config, "input_contract_version", None) != INPUT_VERSION:
                raise ValueError("Stage 1 input contract mismatch; retrain with product title + description + review text")
            if [self.model.config.id2label[i] for i in range(4)] != list(CLASS_NAMES):
                raise ValueError("Stage 1 class order mismatch")
            self.model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
        return self.tokenizer, self.model

    def load_clip(self):
        if self.clip_text is None:
            from sentence_transformers import SentenceTransformer
            self.clip_text = SentenceTransformer("sentence-transformers/clip-ViT-B-32-multilingual-v1")
            self.clip_image = SentenceTransformer("sentence-transformers/clip-ViT-B-32")
        return self.clip_text, self.clip_image

    def text_probabilities(self, text):
        import torch
        from stage1.train_roberta import tokenize_reviews
        tokenizer, model = self.load_text()
        enc = tokenize_reviews(tokenizer, [text], model.config.review_max_length, return_tensors="pt")
        enc = enc.to(next(model.parameters()).device)
        with torch.inference_mode():
            return torch.softmax(model(**enc).logits, dim=-1)[0].cpu().tolist()

    def visual_features(self, text, urls):
        import torch
        images = [(url, self.image_loader(url)) for url in parse_image_urls(urls)]
        images = [(url, image) for url, image in images if image is not None]
        if not images:
            return 0.0, None, []
        text_encoder, image_encoder = self.load_clip()
        with torch.inference_mode():
            text_embedding = text_encoder.encode(text, convert_to_tensor=True)
            scores = []
            for url, image in images:
                image_embedding = image_encoder.encode(image, convert_to_tensor=True)
                score = torch.nn.functional.cosine_similarity(
                    image_embedding.reshape(1, -1), text_embedding.reshape(1, -1),
                ).item()
                scores.append((float(score), url))
        if not all(np.isfinite(score) and -1.00001 <= score <= 1.00001 for score, _ in scores):
            raise ValueError("Invalid CLIP cosine similarity")
        score, best = max(scores, key=lambda item: item[0])
        return score, best, scores

    def extract(self, text, image_urls, star_rating, probabilities=None):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Review text cannot be empty")
        rating = normalize_rating(star_rating)
        values = np.asarray(self.text_probabilities(text) if probabilities is None else probabilities, dtype=float)
        if values.shape != (4,) or not np.isfinite(values).all() or (values < 0).any() or not np.isclose(values.sum(), 1, atol=1e-5):
            raise ValueError("Expected four finite class probabilities summing to one")
        similarity, best, scores = self.visual_features(text, image_urls)
        return values.tolist() + [similarity, rating], best, scores
