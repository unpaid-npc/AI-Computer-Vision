# AI Computer Vision: Hand Tracking and Facial Expression Recognition

A real-time computer vision application that uses a webcam to track hands, recognise basic hand gestures, and predict facial expressions. It is built with Python, OpenCV, and MediaPipe.

> **Status: work in progress (Day 1 of 7).** Nothing is runnable yet. The environment and project structure are set up.

## Planned features

| Feature | Status |
|---|---|
| Webcam capture with OpenCV | Planned |
| Hand landmark tracking (21 landmarks, 1 to 2 hands) | Planned |
| Gesture recognition (open palm, fist, peace sign, thumbs-up) | Planned |
| Facial expression prediction (CPU-friendly pretrained model) | Planned |
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

## Licence

To be added.