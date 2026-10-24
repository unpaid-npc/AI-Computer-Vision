"""Central configuration for the application.

All tunable settings live here so they are easy to find and change.
"""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
HAND_MODEL_PATH = MODELS_DIR / "hand_landmarker.task"
FACE_MODEL_PATH = MODELS_DIR / "blaze_face_short_range.tflite"
EXPRESSION_MODEL_PATH = MODELS_DIR / "emotion-ferplus-8.onnx"

# Camera settings
CAMERA_INDEX = 0          # 0 is usually the built-in laptop webcam
FRAME_WIDTH = 640         # requested width (the camera may choose the closest match)
FRAME_HEIGHT = 480        # requested height
MIRROR_VIEW = True        # flip horizontally so it behaves like a mirror

# Window and controls
WINDOW_TITLE = "AI Computer Vision"
QUIT_KEYS = (ord("q"), 27)                        # 'q' or the Esc key
TOGGLE_HANDS_KEYS = (ord("h"), ord("H"))          # switch hand tracking on/off
TOGGLE_EXPRESSION_KEYS = (ord("e"), ord("E"))     # switch expression analysis on/off

# Hand tracking
ENABLE_HAND_TRACKING = True
MAX_HANDS = 2
MIN_DETECTION_CONFIDENCE = 0.5
MIN_PRESENCE_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
# The frame is mirrored before detection and the labels came out reversed on the
# development machine, so they are swapped back here.
SWAP_HANDEDNESS = True

# Gesture recognition
ENABLE_GESTURES = True
GESTURE_SMOOTHING_WINDOW = 5      # majority vote over this many recent frames
SHOW_GESTURE_DEBUG = True         # show the raw finger states (T I M R P) under each hand

# Facial expression recognition
ENABLE_EXPRESSION = True
FACE_MIN_DETECTION_CONFIDENCE = 0.5
FACE_CROP_PADDING = 0.15          # extra margin around the face box before cropping
EXPRESSION_INTERVAL_SEC = 0.25   # hand a frame to the analyser at most 4 times per second
EXPRESSION_SMOOTHING = 0.5        # 0 = no smoothing; higher = steadier but slower to change
EXPRESSION_STALE_SEC = 1.5        # ignore a result older than this
ONNX_NUM_THREADS = 2              # CPU threads for the expression model

# Robustness
MAX_CONSECUTIVE_READ_FAILURES = 30        # stop if this many camera reads in a row fail
MAX_CONSECUTIVE_INFERENCE_ERRORS = 30     # turn a feature off after this many failures
FPS_SMOOTHING = 0.9                       # closer to 1.0 = steadier FPS number