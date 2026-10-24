"""Hand gesture recognition from the 21 MediaPipe hand landmarks.

Classification uses simple geometric rules on landmark coordinates. It needs no
camera and no model, so it can be unit-tested with fake landmarks.

Implemented gestures: Open Palm, Fist, Peace, Thumbs Up.
Anything else is reported as "Unknown" (shown as no gesture).

Landmark indices (MediaPipe): 0 wrist; thumb 1-4; index 5-8; middle 9-12;
ring 13-16; pinky 17-20. In each finger the order is MCP, PIP, DIP, TIP.
"""

import math
from collections import Counter, deque
from dataclasses import dataclass

import cv2

GESTURE_OPEN_PALM = "Open Palm"
GESTURE_FIST = "Fist"
GESTURE_PEACE = "Peace"
GESTURE_THUMBS_UP = "Thumbs Up"
GESTURE_UNKNOWN = "Unknown"

LANDMARK_COUNT = 21
WRIST = 0
THUMB_MCP = 2
THUMB_TIP = 4
INDEX_MCP = 5
MIDDLE_MCP = 9
PINKY_MCP = 17
FINGER_JOINTS = ((6, 8), (10, 12), (14, 16), (18, 20))  # (PIP, TIP): index, middle, ring, pinky

# Tunable thresholds.
FINGER_EXTENDED_RATIO = 1.1     # tip-to-wrist must exceed this x (PIP-to-wrist)
THUMB_EXTENDED_RATIO = 1.2      # thumb-tip-to-pinky-base must exceed this x palm width
THUMB_UP_MIN_VERTICAL = 0.6     # share of the thumb's length that must point upward


@dataclass(frozen=True)
class GestureResult:
    """The gesture recognised for one hand."""

    name: str      # one of the GESTURE_* constants
    fingers: tuple  # (thumb, index, middle, ring, pinky) -> True if extended


def _distance(a, b, aspect_ratio):
    """Distance between two (x, y, z) landmarks, correcting for non-square frames."""
    return math.hypot((a[0] - b[0]) * aspect_ratio, a[1] - b[1])


def _check(landmarks):
    if len(landmarks) != LANDMARK_COUNT:
        raise ValueError(f"Expected {LANDMARK_COUNT} landmarks, got {len(landmarks)}.")

def _palm_is_degenerate(landmarks, aspect_ratio):
    """True if the palm has no measurable size, so no gesture can be judged."""
    palm_width = _distance(landmarks[INDEX_MCP], landmarks[PINKY_MCP], aspect_ratio)
    palm_length = _distance(landmarks[WRIST], landmarks[MIDDLE_MCP], aspect_ratio)
    return palm_width < 1e-6 or palm_length < 1e-6


def finger_states(landmarks, aspect_ratio=1.0):
    """Return (thumb, index, middle, ring, pinky), True where the finger is extended.

    `aspect_ratio` is frame width / height, because landmark x and y are
    normalised separately.
    """
    _check(landmarks)
    wrist = landmarks[WRIST]
    palm_width = _distance(landmarks[INDEX_MCP], landmarks[PINKY_MCP], aspect_ratio)
    palm_length = _distance(wrist, landmarks[MIDDLE_MCP], aspect_ratio)
    if palm_width < 1e-6 or palm_length < 1e-6:
        return (False, False, False, False, False)

    thumb = (_distance(landmarks[THUMB_TIP], landmarks[PINKY_MCP], aspect_ratio)
             > THUMB_EXTENDED_RATIO * palm_width)
    others = tuple(
        _distance(wrist, landmarks[tip], aspect_ratio)
        > FINGER_EXTENDED_RATIO * _distance(wrist, landmarks[pip], aspect_ratio)
        for pip, tip in FINGER_JOINTS
    )
    return (thumb,) + others


def _thumb_points_up(landmarks, aspect_ratio):
    """True if the thumb (base to tip) points mostly upward in the image."""
    base = landmarks[THUMB_MCP]
    tip = landmarks[THUMB_TIP]
    dx = (tip[0] - base[0]) * aspect_ratio
    dy = tip[1] - base[1]  # image y grows downward, so "up" is negative
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return False
    return (-dy) / length >= THUMB_UP_MIN_VERTICAL


def classify_gesture(landmarks, aspect_ratio=1.0):
    """Classify one hand's 21 landmarks into a GestureResult."""
    _check(landmarks)
    if _palm_is_degenerate(landmarks, aspect_ratio):
        return GestureResult(name=GESTURE_UNKNOWN, fingers=(False,) * 5)
    fingers = finger_states(landmarks, aspect_ratio)
    thumb, index, middle, ring, pinky = fingers
    four = (index, middle, ring, pinky)

    name = GESTURE_UNKNOWN
    if all(four):
        name = GESTURE_OPEN_PALM
    elif index and middle and not ring and not pinky:
        name = GESTURE_PEACE
    elif not any(four):
        if thumb and _thumb_points_up(landmarks, aspect_ratio):
            name = GESTURE_THUMBS_UP
        elif not thumb:
            name = GESTURE_FIST
    return GestureResult(name=name, fingers=fingers)


def hand_keys(labels):
    """Give each hand a stable key from its Left/Right label.

    If two hands share a label, the later one gets "#2" appended.
    """
    counts = {}
    keys = []
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
        keys.append(label if counts[label] == 1 else f"{label}#{counts[label]}")
    return keys


class GestureSmoother:
    """Reduces flicker: reports the most common gesture over the last few frames."""

    def __init__(self, window=5):
        if window < 1:
            raise ValueError("window must be at least 1")
        self._window = window
        self._history = {}

    def update(self, raw):
        """`raw` maps hand key -> this frame's gesture name. Returns the smoothed names.

        Hands missing from `raw` are forgotten, so a returning hand starts fresh.
        """
        for key in list(self._history):
            if key not in raw:
                del self._history[key]

        smoothed = {}
        for key, name in raw.items():
            history = self._history.setdefault(key, deque(maxlen=self._window))
            history.append(name)
            counts = Counter(history)
            top = max(counts.values())
            # On a tie, prefer the most recent of the tied gestures.
            smoothed[key] = next(n for n in reversed(history) if counts[n] == top)
        return smoothed


def draw_gestures(frame, hands, gestures, show_debug=False):
    """Draw each hand's gesture name (and optionally its raw finger states) near the wrist."""
    height, width = frame.shape[:2]
    for hand, gesture in zip(hands, gestures):
        wrist_x = int(hand.landmarks[WRIST][0] * width)
        wrist_y = int(hand.landmarks[WRIST][1] * height)
        text_x = min(max(wrist_x - 30, 5), width - 160)
        text_y = min(max(wrist_y + 30, 45), height - 80)
        if gesture.name != GESTURE_UNKNOWN:
            cv2.putText(frame, gesture.name, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        if show_debug:
            flags = "".join("1" if extended else "0" for extended in gesture.fingers)
            cv2.putText(frame, f"TIMRP {flags}", (text_x, text_y + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)