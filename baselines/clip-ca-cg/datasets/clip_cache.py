"""Content-keyed frozen features with pinned encoder provenance."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from datasets.image_store import ImageStore
from runtime import resolve_path, select_device, write_json


class ClipFeatureCache:
    def __init__(self, config):
        self.config = config
        self.directory = resolve_path(config['output']['clip_cache'])
        self.metadata_path = self.directory / 'encoder.json'
        self.signature = {
            'model': config['model']['clip_model'],
            'requested_revision': config['model'].get('clip_revision', 'main'),
            'max_images': config['data'].get('max_images', 5),
        }
        self.metadata = json.loads(self.metadata_path.read_text(encoding='utf-8')) if self.metadata_path.exists() else None
        if self.metadata and self.metadata['signature'] != self.signature:
            raise ValueError('CLIP configuration changed. Use a new output.clip_cache directory.')
        self.images = ImageStore(resolve_path(config['output']['image_cache']))

    @property
    def dimension(self):
        if not self.metadata:
            raise FileNotFoundError('Prepare CLIP features first: python cache_clip_features.py')
        return self.metadata['dimension']

    def path_for(self, row):
        content = {
            'signature': self.signature,
            'resolved_revision': self.metadata['resolved_revision'] if self.metadata else None,
            'text': row['review_text'],
            'urls': row['image_urls'][:self.signature['max_images']],
        }
        key = hashlib.sha256(json.dumps(content, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        return self.directory / (key + '.npz')

    def get(self, row):
        self.dimension
        path = self.path_for(row)
        if not path.is_file():
            raise FileNotFoundError(f"Missing CLIP features for {row['review_id']}. Run cache_clip_features.py for this partition.")
        with np.load(path, allow_pickle=False) as entry:
            vectors = {'text': entry['text'].copy(), 'image': entry['image'].copy(),
                       'usable_urls': entry['usable_urls'].tolist(), 'clip_score': float(entry['clip_score'])}
        for name in ('text', 'image'):
            if vectors[name].shape != (self.dimension,) or not np.isfinite(vectors[name]).all():
                raise ValueError(f'Invalid cached CLIP {name} vector: {path}')
        if not vectors['usable_urls'] and (np.any(vectors['image']) or vectors['clip_score'] != 0):
            raise ValueError('Missing images must have zero CLIP image features and score')
        return vectors

    def prepare(self, frame, force=False, encoder_factory=None):
        missing = [row for row in frame.to_dict('records')
                   if force or not self.metadata or not self.path_for(row).exists()]
        if not missing:
            return {'computed': 0, 'reused': len(frame)}
        if encoder_factory is None:
            from models.encoders.clip_encoder import FrozenCLIPEncoder
            encoder_factory = FrozenCLIPEncoder
        revision = self.metadata['resolved_revision'] if self.metadata else self.signature['requested_revision']
        encoder = encoder_factory(self.signature['model'], revision, select_device(self.config),
                                  resolve_path(self.config['output']['model_cache']))
        self.directory.mkdir(parents=True, exist_ok=True)
        if self.metadata and (encoder.dimension != self.dimension or encoder.revision != revision):
            raise ValueError('Cached CLIP encoder revision/dimension mismatch')
        self.metadata = {'signature': self.signature, 'resolved_revision': encoder.revision,
                         'dimension': encoder.dimension, 'frozen': True}
        write_json(self.metadata_path, self.metadata)
        no_images = 0
        for index, row in enumerate(missing, 1):
            usable, images = [], []
            for url in row['image_urls'][:self.signature['max_images']]:
                image = self.images.load(url)
                if image is not None:
                    usable.append(url)
                    images.append(image)
            vectors = encoder.encode(row['review_text'], images)
            path = self.path_for(row)
            temporary = path.with_suffix('.tmp')
            with temporary.open('wb') as stream:
                np.savez_compressed(stream, **vectors, usable_urls=np.asarray(usable, dtype=str))
            temporary.replace(path)
            no_images += not bool(usable)
            print(f'Cached CLIP {index}/{len(missing)}: {row["review_id"]}; usable images: {len(usable)}', flush=True)
        del encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {'computed': len(missing), 'reused': len(frame) - len(missing), 'without_usable_images': no_images}
