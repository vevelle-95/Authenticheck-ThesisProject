"""Read the existing CSV and verify AuthentiCheck's product-based split manifest."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ASPECTS = (
    'aesthetics', 'product_quality', 'accuracy_of_description',
    'design', 'value', 'seller_service',
)
ALL_ASPECTS = set(ASPECTS) | {'functionality', 'performance', 'sensory_experience', 'packaging'}
SENTIMENTS = ('negative', 'neutral', 'positive')
MODEL_VERSION = 'adapted-clip-ca-cg-aspects-v1'
SPLIT_VERSION = 'product-70-15-15-oof5-v1'


def parse_image_urls(value):
    if isinstance(value, list):
        values = value
    else:
        raw = str(value or '').strip()
        values = json.loads(raw) if raw.startswith('[') else raw.split('|')
    if not isinstance(values, list) or any(not isinstance(item, str) for item in values):
        raise ValueError('review_image_urls must be a JSON list or pipe-separated URLs')
    return list(dict.fromkeys(item.strip() for item in values if item.strip()))


def parse_annotations(raw):
    try:
        annotations = json.loads(raw) if isinstance(raw, str) else raw
    except (ValueError, TypeError) as error:
        raise ValueError('Authentic reviews require JSON aspect_annotations; use [] for none') from error
    if not isinstance(annotations, list):
        raise ValueError('aspect_annotations must be a JSON list')
    grouped = {}
    for annotation in annotations:
        if not isinstance(annotation, dict) or annotation.get('category') not in ALL_ASPECTS:
            raise ValueError('Unknown or invalid aspect annotation')
        category = annotation['category']
        sentiment = str(annotation.get('sentiment', '')).strip().lower()
        if sentiment not in SENTIMENTS:
            raise ValueError(f'Invalid sentiment: {sentiment}')
        if not isinstance(annotation.get('text'), str) or not annotation['text'].strip():
            raise ValueError('Annotation text must be nonempty; paraphrases are accepted')
        if category in ASPECTS:
            grouped.setdefault(category, set()).add(SENTIMENTS.index(sentiment))
    return grouped


def load_reviews(path, require_annotations=False):
    frame = pd.read_csv(path, encoding='utf-8-sig', skipinitialspace=True,
                        keep_default_na=False, dtype={'review_id': str, 'product_id': str})
    required = {'review_id', 'review_text', 'review_image_urls'}
    if require_annotations:
        required |= {'product_id', 'ground_truth', 'aspect_annotations', 'star_rating'}
    if required - set(frame):
        raise ValueError(f'Missing CSV columns: {sorted(required - set(frame))}')
    if frame.empty:
        raise ValueError('CSV is empty')
    for name in ('review_id', 'review_text', 'product_id'):
        if name in frame:
            frame[name] = frame[name].astype(str).str.strip()
            if frame[name].eq('').any():
                raise ValueError(f'{name} cannot be blank')
    if frame.review_id.duplicated().any():
        raise ValueError('review_id must be unique')
    if require_annotations and frame.review_text.str.casefold().duplicated().any():
        raise ValueError('Duplicate review text must be resolved before training')
    if 'product_description' not in frame:
        frame['product_description'] = ''
    frame['image_urls'] = frame.review_image_urls.map(parse_image_urls)
    if require_annotations:
        quality = {'authentic': 0, 'deceptive': 1, 'liv': 2, 'vague': 2, 'irrelevant': 3}
        frame['label'] = frame.ground_truth.str.strip().str.lower().map(quality)
        if frame.label.isna().any():
            raise ValueError('Unknown ground_truth review-quality label')
        frame['annotations'] = [parse_annotations(row.aspect_annotations) if row.label == 0 else {}
                                for row in frame.itertuples()]
    return frame.reset_index(drop=True)


def fingerprint(frame):
    columns = sorted(column for column in frame if column not in {
        'label', 'normalized_rating', 'image_urls', 'annotations', 'partition', 'fold',
    })
    records = frame.sort_values('review_id')[columns].astype(str).to_dict('records')
    return hashlib.sha256(json.dumps(records, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_splits(frame, path):
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    if payload.get('version') != SPLIT_VERSION or payload.get('dataset_sha256') != fingerprint(frame):
        raise ValueError('CSV and shared split manifest do not match; regenerate splits in AuthentiCheck')
    records = pd.DataFrame(payload['records'])
    if records.review_id.duplicated().any() or set(records.review_id) != set(frame.review_id):
        raise ValueError('Every review must occur exactly once in the split manifest')
    joined = frame.merge(records, on=['review_id', 'product_id'], validate='one_to_one')
    if len(joined) != len(frame) or set(joined.partition) != {'train', 'validation', 'test'}:
        raise ValueError('Invalid split coverage')
    if joined.groupby('product_id').partition.nunique().max() != 1:
        raise ValueError('Product leakage between partitions')
    return joined


def build_targets(annotations):
    aspects = np.zeros(len(ASPECTS), dtype=np.float32)
    sentiments = np.zeros((len(ASPECTS), len(SENTIMENTS)), dtype=np.float32)
    for index, category in enumerate(ASPECTS):
        polarities = annotations.get(category, set())
        if polarities:
            aspects[index] = 1
            for polarity in polarities:
                sentiments[index, polarity] = 1 / len(polarities)
    return aspects, sentiments
