"""Validate and serve a selected compatible AuthentiCheck bundle on port 8000."""
import argparse
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', required=True)
    p.add_argument('--port', type=int, default=8000)
    p.add_argument('--check', action='store_true')
    p.add_argument('--gpu-memory-gb', type=float)
    args = p.parse_args()
    bundle = Path(args.bundle).resolve()
    os.environ['AUTHENTICHECK_MODEL_BUNDLE'] = str(bundle)
    from gpu_memory import configure_gpu_memory
    configure_gpu_memory(args.gpu_memory_gb)
    from stage2.online_inference import OnlineInference
    model = OnlineInference(bundle / 'dost_roberta', bundle / 'xgboost_meta_classifier.json', bundle / 'absa_model')
    model.check_artifacts()
    print(f'Artifact compatibility checks passed: {bundle}', flush=True)
    if args.check:
        print('Model loading is checked at service startup.', flush=True)
        return
    import uvicorn
    uvicorn.run('main:app', host='127.0.0.1', port=args.port)


if __name__ == '__main__':
    main()
