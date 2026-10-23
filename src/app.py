"""Application entry point (Day 3: webcam + hand tracking).

Run from the project root with:  python -m src.app
"""

import sys
import time

import cv2

from src import config
from src.hand_tracking import HandTracker, draw_hands


def open_camera(index):
    """Try to open the camera, preferring DirectShow on Windows.

    Returns an opened cv2.VideoCapture, or None if every backend failed.
    """
    backends = []
    if hasattr(cv2, "CAP_DSHOW"):
        backends.append(("DirectShow", cv2.CAP_DSHOW))
    backends.append(("default", cv2.CAP_ANY))

    for name, backend in backends:
        capture = cv2.VideoCapture(index, backend)
        if capture.isOpened():
            print(f"Camera {index} opened using the {name} backend.")
            return capture
        capture.release()
    return None


def create_hand_tracker():
    """Create the hand tracker, or print a helpful message and return None."""
    try:
        return HandTracker(
            config.HAND_MODEL_PATH,
            max_hands=config.MAX_HANDS,
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_presence_confidence=config.MIN_PRESENCE_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
            swap_handedness=config.SWAP_HANDEDNESS,
        )
    except FileNotFoundError as error:
        print(f"ERROR: {error}", file=sys.stderr)
    except Exception as error:
        print(f"ERROR: Could not start the hand tracker: {error}", file=sys.stderr)
    return None


def draw_overlay(frame, fps, hands_enabled, hand_count):
    """Draw FPS, hand status, and key hints on the frame."""
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    if hands_enabled:
        text = f"Hands detected: {hand_count}"
        color = (0, 255, 0) if hand_count else (0, 200, 255)
    else:
        text = "Hand tracking: OFF"
        color = (160, 160, 160)
    cv2.putText(frame, text, (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.putText(frame, "Q/Esc: quit    H: toggle hand tracking", (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def window_closed():
    """Return True if the user closed the window with the X button."""
    try:
        return cv2.getWindowProperty(config.WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


def main():
    tracker = None
    if config.ENABLE_HAND_TRACKING:
        tracker = create_hand_tracker()
        if tracker is None:
            return 1

    capture = open_camera(config.CAMERA_INDEX)
    if capture is None:
        print(
            f"ERROR: Could not open camera {config.CAMERA_INDEX}.\n"
            "  - Close other apps that may be using the camera (Teams, Zoom, browser tabs).\n"
            "  - Check Windows Settings > Privacy & security > Camera, and allow desktop apps.\n"
            "  - Try a different CAMERA_INDEX in src/config.py.",
            file=sys.stderr,
        )
        if tracker is not None:
            tracker.close()
        return 1

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    fps = 0.0
    previous_time = time.perf_counter()
    read_failures = 0
    inference_errors = 0
    hands_enabled = tracker is not None
    window_shown = False

    try:
        while True:
            success, frame = capture.read()

            if not success or frame is None:
                read_failures += 1
                if read_failures >= config.MAX_CONSECUTIVE_READ_FAILURES:
                    print("ERROR: The camera stopped delivering frames.", file=sys.stderr)
                    return 1
                cv2.waitKey(10)
                continue
            read_failures = 0

            if config.MIRROR_VIEW:
                frame = cv2.flip(frame, 1)

            hands = []
            if hands_enabled:
                try:
                    hands = tracker.detect(frame)
                    inference_errors = 0
                except Exception as error:
                    inference_errors += 1
                    if inference_errors == 1:
                        print(f"WARNING: hand tracking error: {error}", file=sys.stderr)
                    if inference_errors >= config.MAX_CONSECUTIVE_INFERENCE_ERRORS:
                        print("ERROR: Hand tracking keeps failing, so it was turned off.",
                              file=sys.stderr)
                        hands_enabled = False
                draw_hands(frame, hands)

            now = time.perf_counter()
            elapsed = now - previous_time
            previous_time = now
            if elapsed > 0:
                instant_fps = 1.0 / elapsed
                fps = instant_fps if fps == 0.0 else (
                    config.FPS_SMOOTHING * fps + (1 - config.FPS_SMOOTHING) * instant_fps
                )

            draw_overlay(frame, fps, hands_enabled, len(hands))
            cv2.imshow(config.WINDOW_TITLE, frame)
            window_shown = True

            key = cv2.waitKey(1) & 0xFF
            if key in config.QUIT_KEYS:
                break
            if key in config.TOGGLE_HANDS_KEYS and tracker is not None:
                hands_enabled = not hands_enabled
                inference_errors = 0
            if window_shown and window_closed():
                break
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        capture.release()
        if tracker is not None:
            tracker.close()
        cv2.destroyAllWindows()
        print("Camera released. Goodbye.")

    return 0


if __name__ == "__main__":
    sys.exit(main())