"""Application entry point (Day 5: webcam + hand tracking + facial expression).

Run from the project root with:  python -m src.app
"""

import sys
import time

import cv2

from src import config
from src.expression_detection import ExpressionRecognizer, draw_expression
from src.expression_worker import ExpressionWorker
from src.hand_tracking import HandTracker, draw_hands

FONT = cv2.FONT_HERSHEY_SIMPLEX
GREEN = (0, 255, 0)
ORANGE = (0, 200, 255)
YELLOW = (0, 255, 255)
GREY = (160, 160, 160)
WHITE = (255, 255, 255)


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
    """Create the hand tracker, or print a warning and return None."""
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
        print(f"WARNING: hand tracking disabled. {error}", file=sys.stderr)
    except Exception as error:
        print(f"WARNING: hand tracking disabled. Could not start it: {error}", file=sys.stderr)
    return None


def create_expression_recognizer():
    """Create the expression recognizer, or print a warning and return None."""
    try:
        return ExpressionRecognizer(
            config.FACE_MODEL_PATH,
            config.EXPRESSION_MODEL_PATH,
            min_face_confidence=config.FACE_MIN_DETECTION_CONFIDENCE,
            padding=config.FACE_CROP_PADDING,
            num_threads=config.ONNX_NUM_THREADS,
        )
    except FileNotFoundError as error:
        print(f"WARNING: expression recognition disabled. {error}", file=sys.stderr)
    except Exception as error:
        print(f"WARNING: expression recognition disabled. Could not start it: {error}",
              file=sys.stderr)
    return None


def draw_overlay(frame, fps, status_lines):
    """Draw FPS, status lines, and key hints on the frame."""
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), FONT, 0.8, GREEN, 2)
    for row, (text, color) in enumerate(status_lines, start=1):
        cv2.putText(frame, text, (10, 30 + row * 30), FONT, 0.65, color, 2)
    height = frame.shape[0]
    cv2.putText(frame, "Expressions are predictions, not real emotions",
                (10, height - 40), FONT, 0.45, (200, 200, 200), 1)
    cv2.putText(frame, "Q/Esc: quit    H: hands    E: expression",
                (10, height - 15), FONT, 0.5, WHITE, 1)


def window_closed():
    """Return True if the user closed the window with the X button."""
    try:
        return cv2.getWindowProperty(config.WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


def main():
    hand_tracker = create_hand_tracker() if config.ENABLE_HAND_TRACKING else None
    recognizer = create_expression_recognizer() if config.ENABLE_EXPRESSION else None

    capture = open_camera(config.CAMERA_INDEX)
    if capture is None:
        print(
            f"ERROR: Could not open camera {config.CAMERA_INDEX}.\n"
            "  - Close other apps that may be using the camera (Teams, Zoom, browser tabs).\n"
            "  - Check Windows Settings > Privacy & security > Camera, and allow desktop apps.\n"
            "  - Try a different CAMERA_INDEX in src/config.py.",
            file=sys.stderr,
        )
        if hand_tracker is not None:
            hand_tracker.close()
        if recognizer is not None:
            recognizer.close()
        return 1

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    worker = None
    if recognizer is not None:
        worker = ExpressionWorker(recognizer, smoothing=config.EXPRESSION_SMOOTHING)

    fps = 0.0
    previous_time = time.perf_counter()
    last_submit = 0.0
    read_failures = 0
    hand_errors = 0
    last_expression_error = ""
    hands_enabled = hand_tracker is not None
    expression_enabled = worker is not None
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

            now = time.perf_counter()
            elapsed = now - previous_time
            previous_time = now
            if elapsed > 0:
                instant_fps = 1.0 / elapsed
                fps = instant_fps if fps == 0.0 else (
                    config.FPS_SMOOTHING * fps + (1 - config.FPS_SMOOTHING) * instant_fps
                )

            # 1. Give the expression worker a clean copy of the frame (before anything is drawn on it).
            if expression_enabled and now - last_submit >= config.EXPRESSION_INTERVAL_SEC:
                last_submit = now
                worker.submit(frame)

            status_lines = []

            # 2. Hand tracking.
            if hand_tracker is None:
                status_lines.append(("Hands: unavailable", GREY))
            elif not hands_enabled:
                status_lines.append(("Hands: OFF", GREY))
            else:
                hands = []
                try:
                    hands = hand_tracker.detect(frame)
                    hand_errors = 0
                except Exception as error:
                    hand_errors += 1
                    if hand_errors == 1:
                        print(f"WARNING: hand tracking error: {error}", file=sys.stderr)
                    if hand_errors >= config.MAX_CONSECUTIVE_INFERENCE_ERRORS:
                        print("ERROR: Hand tracking keeps failing, so it was turned off.",
                              file=sys.stderr)
                        hands_enabled = False
                draw_hands(frame, hands)
                status_lines.append(
                    (f"Hands detected: {len(hands)}", GREEN if hands else ORANGE)
                )

            # 3. Latest expression result (from the worker thread).
            if worker is None:
                status_lines.append(("Expression: unavailable", GREY))
            elif not expression_enabled:
                status_lines.append(("Expression: OFF", GREY))
            else:
                snapshot = worker.latest()
                fresh = (snapshot.finished_at > 0.0
                         and time.monotonic() - snapshot.finished_at <= config.EXPRESSION_STALE_SEC)
                if not fresh:
                    status_lines.append(("Expression: analysing...", ORANGE))
                else:
                    if snapshot.error:
                        if snapshot.error != last_expression_error:
                            print(f"WARNING: expression analysis error: {snapshot.error}",
                                  file=sys.stderr)
                            last_expression_error = snapshot.error
                    else:
                        last_expression_error = ""

                    if snapshot.consecutive_errors >= config.MAX_CONSECUTIVE_INFERENCE_ERRORS:
                        print("ERROR: Expression analysis keeps failing, so it was turned off.",
                              file=sys.stderr)
                        expression_enabled = False
                        status_lines.append(("Expression: stopped (errors)", ORANGE))
                    elif snapshot.result is None:
                        status_lines.append(("Expression: no face detected", ORANGE))
                    else:
                        draw_expression(frame, snapshot.result)
                        status_lines.append((
                            f"Expression: {snapshot.result.label} "
                            f"({snapshot.result.confidence:.0%})", GREEN))
                    status_lines.append((f"Analysis: {snapshot.analysis_ms:.0f} ms", YELLOW))

            draw_overlay(frame, fps, status_lines)
            cv2.imshow(config.WINDOW_TITLE, frame)
            window_shown = True

            key = cv2.waitKey(1) & 0xFF
            if key in config.QUIT_KEYS:
                break
            if key in config.TOGGLE_HANDS_KEYS and hand_tracker is not None:
                hands_enabled = not hands_enabled
                hand_errors = 0
            if key in config.TOGGLE_EXPRESSION_KEYS and worker is not None:
                expression_enabled = not expression_enabled
                last_expression_error = ""
            if window_shown and window_closed():
                break
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        if worker is not None:
            worker.stop()
        if recognizer is not None:
            recognizer.close()
        if hand_tracker is not None:
            hand_tracker.close()
        capture.release()
        cv2.destroyAllWindows()
        print("Camera released. Goodbye.")

    return 0


if __name__ == "__main__":
    sys.exit(main())