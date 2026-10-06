"""Configuration and checkpoint contracts for the adapted baseline."""

import json
from pathlib import Path

import torch
import yaml

from datasets.authenticheck_data import ASPECTS, MODEL_VERSION, SENTIMENTS

BASE_DIR = Path(__file__).resolve().parent


def load_config(path=None):
    with Path(path or BASE_DIR / 'configs/config.yaml').open(encoding='utf-8') as stream:
        return yaml.safe_load(stream)


def resolve_path(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (BASE_DIR / path).resolve()


def select_device(config):
    requested = config['training'].get('device', 'auto')
    if requested == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable. Run setup_environment.ps1 or select cpu.')
    return torch.device('cuda' if requested == 'auto' and torch.cuda.is_available()
                        else 'cpu' if requested == 'auto' else requested)


def load_checkpoint(path, device='cpu'):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get('model_version') != MODEL_VERSION:
        raise ValueError('Old holistic checkpoints are incompatible; train the adapted aspect model')
    if tuple(checkpoint.get('aspects', [])) != ASPECTS or tuple(checkpoint.get('sentiments', [])) != SENTIMENTS:
        raise ValueError('Checkpoint taxonomy or sentiment order mismatch')
    return checkpoint


def write_json(path, payload, overwrite=True):
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f'Report already exists: {path}; select a new output filename')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
