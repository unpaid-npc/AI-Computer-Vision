"""Central configuration for the application.

All tunable settings live here so they are easy to find and change.
"""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
HAND_MODEL_PATH = MODELS_DIR / "hand_landmarker.task"

# Camera settings
CAMERA_INDEX = 0          # 0 is usually the built-in laptop webcam
FRAME_WIDTH = 640         # requested width (the camera may choose the closest match)
FRAME_HEIGHT = 480        # requested height
MIRROR_VIEW = True        # flip horizontally so it behaves like a mirror

# Window and controls
WINDOW_TITLE = "AI Computer Vision"
QUIT_KEYS = (ord("q"), 27)                  # 'q' or the Esc key
TOGGLE_HANDS_KEYS = (ord("h"), ord("H"))    # switch hand tracking on/off

# Hand tracking
ENABLE_HAND_TRACKING = True
MAX_HANDS = 2
MIN_DETECTION_CONFIDENCE = 0.5
MIN_PRESENCE_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
# The frame is mirrored before detection, so the Left/Right labels should match
# your real hands. If they appear reversed on your machine, set this to True.
SWAP_HANDEDNESS = False

# Robustness
MAX_CONSECUTIVE_READ_FAILURES = 30        # stop if this many camera reads in a row fail
MAX_CONSECUTIVE_INFERENCE_ERRORS = 30     # turn hand tracking off after this many failures
FPS_SMOOTHING = 0.9                       # closer to 1.0 = steadier FPS number