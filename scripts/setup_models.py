"""Install and verify AuthentiCheck model artifacts without committing weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "models" / "manifest.json"


def read_manifest() -> dict:
    with MANIFEST_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(manifest: dict) -> list[str]:
    missing = [item for item in manifest["required"] if not (ROOT / item).is_file()]
    for alternatives in manifest.get("weightAlternatives", []):
        if not any((ROOT / item).is_file() for item in alternatives):
            missing.append("one of: " + ", ".join(alternatives))
    return missing


def safe_extract(archive: Path) -> None:
    root = ROOT.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (ROOT / member.filename).resolve()
            if root not in target.parents and target != root:
                raise ValueError(f"Unsafe path in model archive: {member.filename}")
        bundle.extractall(ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description="Install or verify AuthentiCheck model artifacts.")
    parser.add_argument("--archive", type=Path, help="Existing model bundle ZIP.")
    parser.add_argument("--verify-only", action="store_true", help="Only check the expected local files.")
    args = parser.parse_args()
    manifest = read_manifest()

    if not args.verify_only:
        archive = args.archive
        temporary_dir = None
        if archive is None:
            url = manifest.get("bundle", {}).get("downloadUrl")
            if not url:
                print("No model bundle URL is configured. Use --archive or update models/manifest.json.")
                return 2
            temporary_dir = tempfile.TemporaryDirectory(prefix="authenticheck-models-")
            archive = Path(temporary_dir.name) / "models.zip"
            print(f"Downloading model bundle from {url} ...")
            with urllib.request.urlopen(url, timeout=60) as response, archive.open("wb") as output:
                shutil.copyfileobj(response, output)

        archive = archive.resolve()
        if not archive.is_file():
            print(f"Model bundle not found: {archive}")
            return 2
        expected = manifest.get("bundle", {}).get("sha256")
        if expected and sha256(archive).lower() != expected.lower():
            print("Model bundle checksum does not match models/manifest.json.")
            return 2
        safe_extract(archive)
        if temporary_dir is not None:
            temporary_dir.cleanup()

    missing = verify(manifest)
    if missing:
        print("Model setup is incomplete:")
        for item in missing:
            print(f"  - {item}")
        return 1
    print(f"AuthentiCheck model artifacts are ready ({manifest['modelVersion']}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
