"""Review-only tokens, real cached buyer photos, frozen CLIP vectors, and aspect targets."""

import torch
from torch.utils.data import Dataset

from datasets.authenticheck_data import build_targets


def image_transform(size):
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((size, size)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


class MultiModalDataset(Dataset):
    def __init__(self, frame, tokenizer, cache, config, transform=None):
        self.records = frame.to_dict("records")
        self.tokenizer = tokenizer
        self.cache = cache
        self.config = config
        self.size = config["data"]["image_size"]
        self.transform = transform if transform is not None else image_transform(self.size)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        tokens = self.tokenizer(
            row["review_text"], padding="max_length", truncation=True,
            max_length=self.config["data"]["max_length"], return_tensors="pt",
        )
        cached = self.cache.get(row)
        photos = []
        for url in cached["usable_urls"]:
            image = self.cache.images.load(url, download=False)
            if image is None:
                raise FileNotFoundError("Cached image bytes missing; rebuild features with --force")
            photos.append(self.transform(image))
        item = {
            "review_id": row["review_id"], "review_text": row["review_text"],
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "images": torch.stack(photos) if photos else torch.empty(0, 3, self.size, self.size),
            "clip_text": torch.from_numpy(cached["text"]),
            "clip_image": torch.from_numpy(cached["image"]),
        }
        if "annotations" in row:
            aspects, sentiments = build_targets(row["annotations"])
            item["aspect_targets"] = torch.from_numpy(aspects)
            item["sentiment_targets"] = torch.from_numpy(sentiments)
        return item


def multimodal_collate_fn(items):
    if not items:
        raise ValueError("Cannot collate an empty batch")
    maximum = max(1, max(item["images"].shape[0] for item in items))
    padded, masks = [], []
    for item in items:
        images = item["images"]
        count, channels, height, width = images.shape
        padding = images.new_zeros((maximum - count, channels, height, width))
        padded.append(torch.cat([images, padding]))
        mask = torch.zeros(maximum, dtype=torch.bool)
        mask[:count] = True
        masks.append(mask)
    result = {
        name: torch.stack([item[name] for item in items])
        for name in ("input_ids", "attention_mask", "clip_text", "clip_image")
    }
    result.update(images=torch.stack(padded), image_mask=torch.stack(masks),
                  review_ids=[item["review_id"] for item in items],
                  review_texts=[item["review_text"] for item in items])
    if "aspect_targets" in items[0]:
        for name in ("aspect_targets", "sentiment_targets"):
            result[name] = torch.stack([item[name] for item in items])
    return result
