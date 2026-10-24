"""Unit tests for gesture classification, using synthetic hand landmarks."""

import math

import pytest

from src.gesture_recognition import (
    GESTURE_FIST,
    GESTURE_OPEN_PALM,
    GESTURE_PEACE,
    GESTURE_THUMBS_UP,
    GESTURE_UNKNOWN,
    GestureSmoother,
    classify_gesture,
    hand_keys,
)

# Hand-local frame: wrist at (0, 0), fingers point up (negative y), like image coordinates.
FINGER_BASES = {
    "index": (-0.03, -0.195),
    "middle": (0.0, -0.20),
    "ring": (0.03, -0.195),
    "pinky": (0.06, -0.18),
}
# (IP joint, tip) for each thumb pose.
THUMB_POSES = {
    "folded": ((-0.035, -0.12), (-0.005, -0.15)),
    "out": ((-0.095, -0.09), (-0.14, -0.10)),
    "up": ((-0.06, -0.12), (-0.065, -0.17)),
}
STRAIGHT_OFFSETS = (-0.09, -0.145, -0.19)   # PIP, DIP, TIP relative to the finger base
CURLED_OFFSETS = (-0.03, 0.01, 0.05)

ALL_FINGERS = ("index", "middle", "ring", "pinky")


def make_hand(extended=(), thumb="folded", rotation_deg=0.0):
    """Build 21 landmarks. `extended` names the straight fingers among index..pinky."""
    points = [(0.0, 0.0), (-0.03, -0.04), (-0.05, -0.08)]
    points.extend(THUMB_POSES[thumb])
    for name in ALL_FINGERS:
        base_x, base_y = FINGER_BASES[name]
        offsets = STRAIGHT_OFFSETS if name in extended else CURLED_OFFSETS
        points.append((base_x, base_y))
        points.extend((base_x, base_y + dy) for dy in offsets)

    angle = math.radians(rotation_deg)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    return [
        (0.5 + x * cos_a - y * sin_a, 0.5 + x * sin_a + y * cos_a, 0.0)
        for x, y in points
    ]


def test_open_palm():
    assert classify_gesture(make_hand(ALL_FINGERS, "out")).name == GESTURE_OPEN_PALM


def test_fist():
    assert classify_gesture(make_hand((), "folded")).name == GESTURE_FIST


def test_peace():
    assert classify_gesture(make_hand(("index", "middle"), "folded")).name == GESTURE_PEACE


def test_peace_with_thumb_out_is_still_peace():
    assert classify_gesture(make_hand(("index", "middle"), "out")).name == GESTURE_PEACE


def test_thumbs_up():
    assert classify_gesture(make_hand((), "up")).name == GESTURE_THUMBS_UP


def test_thumb_sticking_out_sideways_is_not_a_gesture():
    assert classify_gesture(make_hand((), "out")).name == GESTURE_UNKNOWN


def test_pointing_is_not_implemented_so_unknown():
    assert classify_gesture(make_hand(("index",), "folded")).name == GESTURE_UNKNOWN


def test_thumbs_down_is_not_thumbs_up():
    upside_down = make_hand((), "up", rotation_deg=180)
    assert classify_gesture(upside_down).name != GESTURE_THUMBS_UP


ROTATION_CASES = [
    (GESTURE_OPEN_PALM, ALL_FINGERS, "out"),
    (GESTURE_FIST, (), "folded"),
    (GESTURE_PEACE, ("index", "middle"), "folded"),
]


@pytest.mark.parametrize("rotation", [0, 45, 90, 135, 180, 270])
@pytest.mark.parametrize("expected, extended, thumb", ROTATION_CASES)
def test_finger_based_gestures_ignore_hand_rotation(expected, extended, thumb, rotation):
    hand = make_hand(extended, thumb, rotation_deg=rotation)
    assert classify_gesture(hand).name == expected


def test_wrong_number_of_landmarks_raises():
    with pytest.raises(ValueError):
        classify_gesture([(0.0, 0.0, 0.0)] * 20)


def test_degenerate_hand_does_not_crash():
    result = classify_gesture([(0.5, 0.5, 0.0)] * 21)
    assert result.name == GESTURE_UNKNOWN


def test_smoother_returns_the_majority():
    smoother = GestureSmoother(window=5)
    result = {}
    for name in (GESTURE_FIST, GESTURE_FIST, GESTURE_OPEN_PALM):
        result = smoother.update({"Right": name})
    assert result == {"Right": GESTURE_FIST}


def test_smoother_prefers_the_latest_on_a_tie():
    smoother = GestureSmoother(window=4)
    smoother.update({"Right": GESTURE_FIST})
    assert smoother.update({"Right": GESTURE_OPEN_PALM}) == {"Right": GESTURE_OPEN_PALM}


def test_smoother_forgets_a_hand_that_disappears():
    smoother = GestureSmoother(window=5)
    for _ in range(3):
        smoother.update({"Right": GESTURE_FIST})
    smoother.update({})  # the hand left the frame
    assert smoother.update({"Right": GESTURE_PEACE}) == {"Right": GESTURE_PEACE}


def test_smoother_rejects_a_zero_window():
    with pytest.raises(ValueError):
        GestureSmoother(window=0)


def test_hand_keys_disambiguates_duplicate_labels():
    assert hand_keys(["Right", "Left", "Right"]) == ["Right", "Left", "Right#2"]