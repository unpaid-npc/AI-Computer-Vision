"""Download the MediaPipe hand landmarker model into the models/ folder.

Run from the project root with:  python scripts/download_models.py
"""

import sys
import urllib.request
from pathlib import Path

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
TARGET = MODELS_DIR / "hand_landmarker.task"
MIN_EXPECTED_BYTES = 1_000_000  # a real model is several MB; smaller means a bad download


def main():
    MODELS_DIR.mkdir(exist_ok=True)

    if TARGET.exists() and TARGET.stat().st_size > MIN_EXPECTED_BYTES:
        print(f"Model already present: {TARGET}")
        return 0

    print(f"Downloading hand landmarker model to {TARGET} ...")
    partial = TARGET.with_name(TARGET.name + ".part")
    try:
        urllib.request.urlretrieve(MODEL_URL, partial)
    except Exception as error:
        print(f"ERROR: download failed: {error}", file=sys.stderr)
        partial.unlink(missing_ok=True)
        return 1

    if partial.stat().st_size < MIN_EXPECTED_BYTES:
        print("ERROR: the downloaded file is too small to be a valid model.", file=sys.stderr)
        partial.unlink(missing_ok=True)
        return 1

    partial.replace(TARGET)
    print(f"Done. Saved {TARGET.stat().st_size / 1_000_000:.1f} MB.")
    return 0


if __name__ == "__main__":
    sys.exit(main())