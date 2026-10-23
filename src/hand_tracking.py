"""Hand landmark detection using the MediaPipe Tasks HandLandmarker.

This module detects hands and draws them. It does not classify gestures;
that is the job of gesture_recognition.py (Day 6).
"""

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

# Standard connections between the 21 hand landmarks (index 0 is the wrist).
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index finger
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle finger
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring finger
    (13, 17), (17, 18), (18, 19), (19, 20),  # little finger
    (0, 17),                                  # palm edge
)


@dataclass(frozen=True)
class DetectedHand:
    """One detected hand."""

    label: str        # "Left" or "Right"
    score: float      # confidence of the Left/Right label, 0.0 to 1.0
    landmarks: tuple  # 21 tuples of (x, y, z), x and y normalised to 0..1


class HandTracker:
    """Wraps the MediaPipe HandLandmarker for use with OpenCV BGR frames."""

    def __init__(self, model_path, max_hands=2, min_detection_confidence=0.5,
                 min_presence_confidence=0.5, min_tracking_confidence=0.5,
                 swap_handedness=False):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Hand model not found at {model_path}.\n"
                "  Download it by running:  python scripts/download_models.py"
            )

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = HandLandmarker.create_from_options(options)
        self._swap = swap_handedness
        self._last_timestamp_ms = 0

    def detect(self, frame_bgr):
        """Return a list of DetectedHand for one frame (empty if no hands)."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Video mode needs timestamps that always increase.
        timestamp_ms = int(time.monotonic() * 1000)
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms

        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        hands = []
        for index, landmarks in enumerate(result.hand_landmarks):
            label, score = "Unknown", 0.0
            if index < len(result.handedness) and result.handedness[index]:
                category = result.handedness[index][0]
                label, score = category.category_name, category.score
                if self._swap and label in ("Left", "Right"):
                    label = "Right" if label == "Left" else "Left"
            points = tuple((lm.x, lm.y, lm.z) for lm in landmarks)
            hands.append(DetectedHand(label=label, score=score, landmarks=points))
        return hands

    def close(self):
        """Release the model."""
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def draw_hands(frame, hands):
    """Draw landmarks, connections, and the Left/Right label for each hand."""
    height, width = frame.shape[:2]
    for hand in hands:
        points = [(int(x * width), int(y * height)) for x, y, _ in hand.landmarks]
        for start, end in HAND_CONNECTIONS:
            cv2.line(frame, points[start], points[end], (0, 255, 0), 2)
        for point in points:
            cv2.circle(frame, point, 4, (0, 0, 255), -1)
        wrist_x, wrist_y = points[0]
        cv2.putText(frame, f"{hand.label} {hand.score:.0%}", (wrist_x - 30, wrist_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)