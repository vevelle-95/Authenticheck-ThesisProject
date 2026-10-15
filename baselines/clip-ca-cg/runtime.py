"""Configuration and checkpoint contracts for the adapted baseline."""

import json
from pathlib import Path

import torch
import yaml

from datasets.authenticheck_data import ASPECTS, MODEL_VERSION, SENTIMENTS, SENTIMENT_TARGET_POLICY

BASE_DIR = Path(__file__).resolve().parent


def load_config(path=None):
    with Path(path or BASE_DIR / 'configs/config.yaml').open(encoding='utf-8') as stream:
        return yaml.safe_load(stream)


def resolve_path(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (BASE_DIR / path).resolve()


def default_checkpoint(config=None):
    config = load_config() if config is None else config
    return resolve_path(Path(config['output']['checkpoint_dir']) / 'best.pt')


def select_device(config):
    requested = config['training'].get('device', 'auto')
    if requested == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable. Run setup_environment.ps1 or select cpu.')
    return torch.device('cuda' if requested == 'auto' and torch.cuda.is_available()
                        else 'cpu' if requested == 'auto' else requested)


def load_checkpoint(path, device='cpu'):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get('model_version') != MODEL_VERSION:
        raise ValueError('Old checkpoints are incompatible; retrain the ten-aspect multi-label adaptation')
    if tuple(checkpoint.get('aspects', [])) != ASPECTS or tuple(checkpoint.get('sentiments', [])) != SENTIMENTS:
        raise ValueError('Checkpoint taxonomy or sentiment order mismatch')
    if checkpoint.get('sentiment_target_policy') != SENTIMENT_TARGET_POLICY:
        raise ValueError('Checkpoint sentiment policy mismatch; retrain with multi-label targets')
    if not 0 < checkpoint.get('threshold', 0) < 1 or not 0 < checkpoint.get('sentiment_threshold', 0) < 1:
        raise ValueError('Checkpoint must contain validation-selected aspect and sentiment thresholds')
    return checkpoint


def write_json(path, payload, overwrite=True):
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f'Report already exists: {path}; select a new output filename')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
