"""Download the model files the app needs into the models/ folder.

Run from the project root with:  python scripts/download_models.py
Files that are already present and valid are skipped.
"""

import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

MODELS = [
    {
        "name": "Hand landmarker (MediaPipe)",
        "file": "hand_landmarker.task",
        "urls": [
            "https://storage.googleapis.com/mediapipe-models/"
            "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        ],
        "min_bytes": 1_000_000,
        "sha256": None,
    },
    {
        "name": "Face detector (MediaPipe BlazeFace, short range)",
        "file": "blaze_face_short_range.tflite",
        "urls": [
            "https://storage.googleapis.com/mediapipe-models/"
            "face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
        ],
        "min_bytes": 20_000,
        "sha256": None,
    },
    {
        "name": "Facial expression classifier (ONNX Model Zoo, FER+)",
        "file": "emotion-ferplus-8.onnx",
        "urls": [
            "https://github.com/onnx/models/raw/main/validated/"
            "vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx",
            "https://huggingface.co/onnxmodelzoo/emotion-ferplus-8/resolve/main/"
            "emotion-ferplus-8.onnx",
        ],
        "min_bytes": 30_000_000,
        "sha256": "a2a2ba6a335a3b29c21acb6272f962bd3d47f84952aaffa03b60986e04efa61c",
    },
]


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_valid(path, entry):
    """A file is valid if it is big enough and (when known) matches its checksum."""
    if not path.exists() or path.stat().st_size < entry["min_bytes"]:
        return False
    if entry["sha256"] and sha256_of(path) != entry["sha256"]:
        return False
    return True


def fetch(url, destination):
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (model downloader)"})
    with urllib.request.urlopen(request, timeout=60) as response, open(destination, "wb") as out:
        shutil.copyfileobj(response, out)


def download(entry):
    target = MODELS_DIR / entry["file"]
    if is_valid(target, entry):
        print(f"[skip] {entry['name']}: already present.")
        return True

    partial = target.with_name(target.name + ".part")
    for url in entry["urls"]:
        print(f"[get ] {entry['name']}\n       from {url}")
        try:
            fetch(url, partial)
        except Exception as error:
            print(f"       failed: {error}")
            partial.unlink(missing_ok=True)
            continue

        if is_valid(partial, entry):
            partial.replace(target)
            checked = ", SHA-256 verified" if entry["sha256"] else ""
            print(f"[ ok ] saved {target.name} ({target.stat().st_size / 1_000_000:.1f} MB{checked})")
            return True

        print("       the downloaded file failed the size/checksum check; trying the next source.")
        partial.unlink(missing_ok=True)

    print(f"[FAIL] {entry['name']} could not be downloaded.", file=sys.stderr)
    return False


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    results = [download(entry) for entry in MODELS]
    if all(results):
        print("All models are ready.")
        return 0
    print("Some downloads failed. See the messages above.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())