"""Serve the isolated compact test model at localhost:8001 (media omitted)."""
import os
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["AUTHENTICHECK_MODEL_VERSION"] = "development-compact-v3-context-and-stars"
os.environ["AUTHENTICHECK_PRELOAD_MODELS"] = "0"

import torch
import uvicorn
import main as api
from stage1.features import FeatureExtractor
from stage2.online_inference import OnlineInference


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    torch.set_num_threads(2)
    bundle = ROOT / "models" / "development_v3"
    api.pipeline = OnlineInference(bundle / "dost_roberta", bundle / "xgboost_meta_classifier.json", bundle / "absa_model", development_mode=True)
    api.pipeline.extractor = FeatureExtractor(bundle / "dost_roberta", image_loader=lambda _url: None)
    api.pipeline.check_artifacts()
    print("DEVELOPMENT ONLY: compact encoder, draft/synthetic data, image inputs omitted.", flush=True)
    uvicorn.run(api.app, host="127.0.0.1", port=args.port)
