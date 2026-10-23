"""Webcam application entry point (Day 2: capture and display only).

Run from the project root with:  python -m src.app
"""

import sys
import time

import cv2

from src import config


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


def draw_overlay(frame, fps):
    """Draw the FPS counter and quit hint on the frame."""
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, "Press Q or Esc to quit", (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def window_closed():
    """Return True if the user closed the window with the X button."""
    try:
        return cv2.getWindowProperty(config.WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


def main():
    capture = open_camera(config.CAMERA_INDEX)
    if capture is None:
        print(
            f"ERROR: Could not open camera {config.CAMERA_INDEX}.\n"
            "  - Close other apps that may be using the camera (Teams, Zoom, browser tabs).\n"
            "  - Check Windows Settings > Privacy & security > Camera, and allow desktop apps.\n"
            "  - Try a different CAMERA_INDEX in src/config.py.",
            file=sys.stderr,
        )
        return 1

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    fps = 0.0
    previous_time = time.perf_counter()
    failures = 0
    window_shown = False

    try:
        while True:
            success, frame = capture.read()

            if not success or frame is None:
                failures += 1
                if failures >= config.MAX_CONSECUTIVE_READ_FAILURES:
                    print("ERROR: The camera stopped delivering frames.", file=sys.stderr)
                    return 1
                cv2.waitKey(10)
                continue
            failures = 0

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

            draw_overlay(frame, fps)
            cv2.imshow(config.WINDOW_TITLE, frame)
            window_shown = True

            key = cv2.waitKey(1) & 0xFF
            if key in config.QUIT_KEYS:
                break
            if window_shown and window_closed():
                break
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        capture.release()
        cv2.destroyAllWindows()
        print("Camera released. Goodbye.")

    return 0


if __name__ == "__main__":
    sys.exit(main())