# AI Computer Vision: Hand Tracking and Facial Expression Recognition

A real-time computer vision application that uses your webcam to track hands, recognise basic hand gestures, and predict facial expressions. It runs entirely on the CPU, is built with Python, OpenCV, MediaPipe, ONNX Runtime and PySide6, and was developed on an 8 GB RAM laptop.


## Features

| Feature | Status |
|---|---|
| Webcam capture with OpenCV, with camera error handling | Done |
| Hand landmark tracking (21 landmarks, up to 2 hands, Left/Right labels) | Done |
| Gesture recognition: Open Palm, Fist, Peace, Thumbs Up | Done (rule-based, see limitations) |
| Facial expression prediction (8 categories, with confidence) | Done (limited accuracy, see limitations) |
| Combined real-time pipeline with FPS and detection status | Done |
| Desktop interface (PySide6) with Start/Stop and per-feature switches | Done |
| Lightweight OpenCV window with keyboard controls | Done |
| Automated unit tests (56) | Done |
| GPU acceleration | Not implemented (CPU only by design) |

Expression output is a **predicted facial expression** from a small pretrained model. It is not a measurement of what a person actually feels.

## Requirements

- Windows (developed and tested on Windows only)
- Python 3.12 (tested with 3.12.10)
- A webcam
- About 45 MB of disk space for models, downloaded once

## Quick start

Run these in PowerShell:

```powershell
git clone https://github.com/unpaid-npc/AI-Computer-Vision.git
cd AI-Computer-Vision
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/download_models.py
python -m src.gui
```

If PowerShell refuses to run the activation script, run `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` once and try again. The model download needs an internet connection; after that the app works offline.

## Usage

| Command | What it opens |
|---|---|
| `python -m src.gui` | Desktop window with Start/Stop buttons, feature checkboxes, live detections, and performance figures |
| `python -m src.app` | Lightweight OpenCV window controlled with the keys below |
| `python -m scripts.try_expression` | Standalone test of facial expression recognition only |

### OpenCV window controls

| Key | Action |
|---|---|
| Q or Esc | Quit |
| H | Toggle hand tracking |
| G | Toggle gesture recognition |
| E | Toggle expression analysis |

## Gestures

| Gesture | Status | How it is detected |
|---|---|---|
| Open Palm | Implemented | All four fingers extended |
| Fist | Implemented | No fingers extended, thumb tucked in |
| Peace | Implemented | Index and middle extended, ring and little finger curled |
| Thumbs Up | Implemented | Fingers curled, thumb extended and pointing upward |

Not implemented: pointing, thumbs-down, OK sign, and any other gesture. These show no label.

Finger positions are judged from distances between landmarks, so Open Palm, Fist, and Peace work at any hand rotation (covered by unit tests). Thumbs Up needs the thumb to point upward in the image. A short majority vote over recent frames reduces flicker.

## How it works

```
Webcam frame
   |
   v
VisionPipeline.process(frame)
   |-- HandTracker (MediaPipe HandLandmarker) --> gesture rules --> smoothing
   |-- ExpressionWorker (background thread, about 4 analyses per second)
   |        face detector (MediaPipe) --> crop --> FER+ model (ONNX Runtime)
   v
Annotated frame + FrameStats --> PySide6 window (gui.py) or OpenCV window (app.py)
```

The expression analysis runs on its own thread, so a slow analysis never freezes the video or the hand tracking.

### Project structure

```
src/
  gui.py                   PySide6 interface
  app.py                   OpenCV window front end
  pipeline.py              VisionPipeline, FrameStats, FpsCounter
  camera.py                camera opening and troubleshooting hints
  hand_tracking.py         MediaPipe hand landmarks and drawing
  gesture_recognition.py   rule-based gesture classifier and smoothing
  expression_detection.py  face detection and expression classifier
  expression_worker.py     background thread for expression analysis
  config.py                all settings
scripts/
  download_models.py       downloads the model files into models/
  try_expression.py        standalone expression test
tests/                     unit tests
models/                    downloaded model files (not committed)
screenshots/
```

## Configuration

All settings live in `src/config.py`. The ones you are most likely to change:

| Setting | Purpose |
|---|---|
| `CAMERA_INDEX` | Which camera to use (0 is usually the built-in webcam) |
| `SWAP_HANDEDNESS` | Flip the Left/Right labels if they appear reversed on your machine |
| `MAX_HANDS` | Number of hands to track (1 or 2) |
| `EXPRESSION_INTERVAL_SEC` | How often a frame is handed to the expression analyser |
| `EXPRESSION_SMOOTHING` | Blend recent expression results to reduce flicker (0 disables) |
| `SHOW_GESTURE_DEBUG` | Show the raw finger states under each hand (useful for tuning gestures) |

## Performance

Informal measurements on an HP Pavilion Gaming laptop (AMD Ryzen 5 5600H, 8 GB RAM), CPU only, 640x480 camera input, with hands, gestures and expression all enabled:

- The display stayed at about 30 FPS, which appears to be the webcam's frame rate.
- Per-frame processing was roughly 20 ms with a hand in view.
- One expression analysis took roughly 13 to 14 ms, running about four times per second on a background thread.

These are not formal benchmarks. Results will differ on other machines.

## Running the tests

```powershell
pip install -r requirements-dev.txt
python -m pytest
```

The 56 tests cover the gesture rules (including hand rotation), gesture smoothing, expression helper functions, the pipeline's status text and FPS counter, and the interface's helper functions. They use synthetic data and need no camera or model files. The camera, the models, and the windows themselves were checked manually.

## Troubleshooting

| Problem | What to try |
|---|---|
| "Could not open camera" | Close other apps using the camera (Teams, Zoom, browser tabs), allow desktop apps in Windows Settings > Privacy & security > Camera, or change `CAMERA_INDEX` |
| "Model file not found" | Run `python scripts/download_models.py` |
| Left and Right labels are reversed | Toggle `SWAP_HANDEDNESS` in `src/config.py` |
| Many `INFO` and `WARNING` lines in the terminal | These come from MediaPipe's own logging and are harmless |

## Limitations

- **Expression output is a prediction, not a measure of real emotion.** The model is small and was trained on low-resolution web images. It is more reliable for clear, exaggerated expressions than subtle ones, and its accuracy can vary with lighting, head angle, glasses, and between people.
- The face detector can fire on things that are not faces. For example, it detected the back of a head during testing.
- Only the largest detected face is analysed.
- Gestures are simple geometric rules tuned on one webcam. Partly curled fingers, strong foreshortening, and poor lighting can cause misses. Thumbs Up depends on the thumb pointing upward.
- Tested on Windows with Python 3.12 only. CPU only.
- The application processes frames in memory. It does not save or upload images; the only network access is the one-time model download.

## Future improvements

- Evaluate a stronger expression model (for example HSEmotion; not yet tested here) and compare accuracy
- More gestures (pointing, thumbs-down, OK sign) or a trained gesture classifier
- Support for several faces
- Packaged Windows executable
- Testing on macOS and Linux
- Optional GPU acceleration

## Models and attribution

The model files are not stored in this repository. They are downloaded from their original sources by `scripts/download_models.py`.

| Model | Source |
|---|---|
| Hand landmarker | Google MediaPipe |
| Face detector (BlazeFace, short range) | Google MediaPipe |
| Facial expression classifier (FER+) | ONNX Model Zoo, trained on the FER+ annotations described in "Training Deep Networks for Facial Expression Recognition with Crowd-Sourced Label Distribution" (arXiv:1608.01041). The Model Zoo lists its licence as MIT. |

Libraries: OpenCV, MediaPipe, NumPy, ONNX Runtime, PySide6 (LGPL-3.0, GPL-2.0, or GPL-3.0; installed from PyPI and not bundled), and pytest. Each is used under its own licence; see the individual project pages.

## Licence

The code in this repository is released under the MIT Licence; see [LICENSE](LICENSE). Third-party models and libraries keep their own licences.