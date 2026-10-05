"""Measure the default full-sized training architecture using random inputs/weights.

No encoder downloads, dataset processing, or trained checkpoints are produced.
This measures training memory, not predictive performance or an epoch's duration.
"""

import argparse
import time

import torch
from transformers import RobertaConfig

from models.model import CLIPCACG
from runtime import load_config, resolve_path, write_json
from training.engine import aspect_sentiment_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--images', type=int)
    parser.add_argument('--output', default='outputs/resource_check.json')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('This GPU resource check needs CUDA-enabled PyTorch')
    config = load_config(args.config)
    batch = args.batch_size or config['training']['batch_size']
    photos = args.images or config['data']['max_images']
    tokens = config['data']['max_length']
    accumulation = config['training']['gradient_accumulation_steps']
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = CLIPCACG(config, pretrained=False, text_config=RobertaConfig().to_dict()).to('cuda')
    model.train()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=2e-5)
    amp = config['training']['mixed_precision']
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    inputs = {
        'input_ids': torch.randint(3, 50265, (batch, tokens), device='cuda'),
        'attention_mask': torch.ones(batch, tokens, dtype=torch.long, device='cuda'),
        'images': torch.randn(batch, photos, 3, config['data']['image_size'], config['data']['image_size'], device='cuda'),
        'image_mask': torch.ones(batch, photos, dtype=torch.bool, device='cuda'),
        'clip_text': torch.randn(batch, 512, device='cuda'),
        'clip_image': torch.randn(batch, 512, device='cuda'),
    }
    aspects = torch.zeros(batch, 6, device='cuda')
    aspects[:, 0] = 1
    sentiments = torch.zeros(batch, 6, 3, device='cuda')
    sentiments[:, 0, 2] = 1
    for _ in range(accumulation):
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=amp):
            outputs = model(**inputs)
            loss = aspect_sentiment_loss(outputs, aspects, sentiments)
        scaler.scale(loss / accumulation).backward()
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize()
    if not torch.isfinite(loss):
        raise FloatingPointError('Non-finite benchmark loss')
    report = {
        'benchmark': 'random full-size encoders and inputs; resource check only',
        'gpu': torch.cuda.get_device_name(0), 'batch_size': batch,
        'images_per_review': photos, 'text_tokens': tokens, 'microbatches': accumulation,
        'peak_allocated_GiB': round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        'peak_reserved_GiB': round(torch.cuda.max_memory_reserved() / 1024**3, 3),
        'seconds_including_initialization': round(time.perf_counter() - started, 3),
        'finite_loss': True,
    }
    write_json(resolve_path(args.output), report)
    print(report)


if __name__ == '__main__':
    main()
