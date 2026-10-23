"""Central configuration for the application.

All tunable settings live here so they are easy to find and change.
"""

# Camera settings
CAMERA_INDEX = 0          # 0 is usually the built-in laptop webcam
FRAME_WIDTH = 640         # requested width (the camera may choose the closest match)
FRAME_HEIGHT = 480        # requested height
MIRROR_VIEW = True        # flip horizontally so it behaves like a mirror

# Window and controls
WINDOW_TITLE = "AI Computer Vision"
QUIT_KEYS = (ord("q"), 27)   # 'q' or the Esc key

# Robustness
MAX_CONSECUTIVE_READ_FAILURES = 30   # stop if this many frames in a row fail
FPS_SMOOTHING = 0.9                  # closer to 1.0 = steadier FPS number