# AI Computer Vision: Hand Tracking and Facial Expression Recognition

A real-time computer vision application that uses a webcam to track hands, recognise basic hand gestures, and predict facial expressions. It is built with Python, OpenCV, and MediaPipe.

> **Status: work in progress (Day 4 of 7).** Webcam capture and real-time hand tracking work. Facial expression prediction works as a standalone test (`python -m scripts.try_expression`) but is not yet combined with the main app. Setup: install the requirements, run `python scripts/download_models.py` once, then `python -m src.app`.

## Planned features

| Feature | Status |
|---|---|
| Webcam capture with OpenCV | Done |
| Hand landmark tracking (21 landmarks, 1 to 2 hands) | Done |
| Gesture recognition (open palm, fist, peace sign, thumbs-up) | Planned |
| Facial expression prediction (CPU-friendly pretrained model) | Tested standalone; integration planned |
| Combined real-time application with FPS and status display | Planned |
| User interface with feature toggles | Planned |

## Note on facial expression recognition

Expression output is a *predicted facial expression* from a pretrained model. It is not a measurement of a person's actual emotional state.

## Environment

- Windows, Python 3.12
- Dependencies pinned in `requirements.txt` (OpenCV, MediaPipe, NumPy)
- CPU-only by design, to run on an 8 GB RAM laptop

## Setup

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Project structure

```
src/        application modules
tests/      automated tests
models/     downloaded model files (not committed)
screenshots/
```

## Models and attribution

- Hand landmarks and face detection: Google MediaPipe models (downloaded by `scripts/download_models.py`).
- Facial expression: the FER+ model from the ONNX Model Zoo, based on "Training Deep Networks for Facial Expression Recognition with Crowd-Sourced Label Distribution" (arXiv:1608.01041).

## Known limitations

- Expression output is a prediction from a small pretrained model. It can be wrong, especially with poor lighting, tilted faces, glasses, or subtle expressions, and its accuracy may differ between people.

## Licence

To be added.

onnxruntime==<the version pip showed>