"""Package the inference bundle, runtime, checksums, and transfer instructions."""
import hashlib
import json
import sys
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
BUNDLE = ROOT / 'models/defense_pilot_gpu'
OUTPUT = ROOT / 'output/Authenticheck-defense-pilot-models.zip'


def main():
    from stage2.online_inference import OnlineInference
    OnlineInference(BUNDLE / 'dost_roberta', BUNDLE / 'xgboost_meta_classifier.json', BUNDLE / 'absa_model').check_artifacts()
    files = [BUNDLE / name for name in ('manifest.json', 'xgboost_meta_classifier.json', 'xgboost_meta_classifier.metadata.json')]
    files += [p for p in (BUNDLE / 'dost_roberta').iterdir() if p.is_file() and p.name != 'training_args.bin']
    files += [p for p in (BUNDLE / 'absa_model').rglob('*') if p.is_file()]
    for name in ('main.py', 'model_contract.py', 'model_data.py', 'model_metrics.py', 'training_augmentation.py',
                 'gpu_memory.py', 'requirements.txt', 'scripts/serve_local.py'):
        files.append(ROOT / name)
    for folder in ('stage1', 'stage2'):
        files += list((ROOT / folder).glob('*.py'))
    files = sorted(set(files))
    checksums = {}
    for p in files:
        digest = hashlib.sha256()
        with p.open('rb') as stream:
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                digest.update(chunk)
        checksums[p.relative_to(ROOT).as_posix()] = digest.hexdigest()
    readme = '''AuthentiCheck portable pilot model package

Extract this ZIP into a NEW directory. Python 3.11 is recommended.
Run these commands in PowerShell from the extracted directory:

python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/serve_local.py --bundle models/defense_pilot_gpu --check
.\.venv\Scripts\python.exe scripts/serve_local.py --bundle models/defense_pilot_gpu

Check in another terminal:
Invoke-RestMethod http://127.0.0.1:8000/ready

The API uses http://127.0.0.1:8000/analyze. This ZIP includes runtime code;
no training dataset or GPU environment is required to run inference.
It does not include the browser extension. Use the project's extension and
configure its model API endpoint to the address above.

CPU inference works without an NVIDIA GPU, but may be slower. For CUDA,
install a compatible CUDA-enabled PyTorch build. Optional --gpu-memory-gb 3
is for CUDA devices only; its allocator cap does not limit other apps.
First startup needs internet to download the multilingual/image CLIP encoders
and dependencies. Those external encoders are NOT included in this ZIP.
Keep the resulting Hugging Face cache for later use.

This is the one-epoch experimental pilot, using 1,948 pending augmentation
rows. Quality validation macro F1 is approximately 0.131; held-out evaluation
was not recorded as completed. Treat predictions as experimental.
The adapted Lu et al. model is NOT included: its checkpoint is unavailable.
Original absolute training paths in metadata are historical provenance;
serving selects the extracted bundle via --bundle.

manifest.json inside the bundle records the training library versions.
FILES.sha256.json contains checksums for all packaged source/model files.
'''
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
        for p in files:
            print('Packing ' + p.relative_to(ROOT).as_posix(), flush=True)
            archive.write(p, p.relative_to(ROOT).as_posix())
        archive.writestr('TRANSFER_README.txt', readme)
        archive.writestr('FILES.sha256.json', json.dumps(checksums, indent=2))
    with zipfile.ZipFile(temporary) as archive:
        for name, expected in checksums.items():
            digest = hashlib.sha256()
            with archive.open(name) as stream:
                for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                    digest.update(chunk)
            if digest.hexdigest() != expected:
                raise ValueError('ZIP checksum mismatch: ' + name)
    temporary.replace(OUTPUT)
    digest = hashlib.sha256()
    with OUTPUT.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    OUTPUT.with_suffix('.zip.sha256').write_text(digest.hexdigest() + '  ' + OUTPUT.name + '\n', encoding='utf-8')
    print(f'Verified {len(files)} files. ZIP: {OUTPUT} ({OUTPUT.stat().st_size / 1e6:.1f} MB)', flush=True)


if __name__ == '__main__':
    main()
