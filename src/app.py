"""OpenCV window front end: webcam + hand tracking + gestures + facial expression.

Run from the project root with:  python -m src.app
"""

import sys

import cv2

from src import config
from src.camera import camera_help, open_camera
from src.pipeline import (
    LEVEL_INFO,
    LEVEL_MUTED,
    LEVEL_OK,
    LEVEL_WARN,
    FpsCounter,
    VisionPipeline,
)

FONT = cv2.FONT_HERSHEY_SIMPLEX
LEVEL_COLORS = {
    LEVEL_OK: (0, 255, 0),
    LEVEL_WARN: (0, 200, 255),
    LEVEL_INFO: (0, 255, 255),
    LEVEL_MUTED: (160, 160, 160),
}


def draw_overlay(frame, fps, stats):
    """Draw FPS, the status lines, and key hints on the frame."""
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), FONT, 0.8, (0, 255, 0), 2)
    lines = stats.summary_lines()
    lines.append((f"Processing: {stats.process_ms:.0f} ms", LEVEL_INFO))
    for row, (text, level) in enumerate(lines, start=1):
        cv2.putText(frame, text, (10, 30 + row * 30), FONT, 0.65, LEVEL_COLORS[level], 2)

    height = frame.shape[0]
    cv2.putText(frame, "Expressions are predictions, not real emotions",
                (10, height - 40), FONT, 0.45, (200, 200, 200), 1)
    cv2.putText(frame, "Q/Esc: quit   H: hands   G: gestures   E: expression",
                (10, height - 15), FONT, 0.5, (255, 255, 255), 1)


def window_closed():
    """Return True if the user closed the window with the X button."""
    try:
        return cv2.getWindowProperty(config.WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


def main():
    pipeline = VisionPipeline()

    capture = open_camera(config.CAMERA_INDEX)
    if capture is None:
        print(f"ERROR: Could not open camera {config.CAMERA_INDEX}.\n"
              f"{camera_help(config.CAMERA_INDEX)}", file=sys.stderr)
        pipeline.close()
        return 1

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    fps_counter = FpsCounter(config.FPS_SMOOTHING)
    read_failures = 0
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

            fps = fps_counter.tick()
            stats = pipeline.process(frame)
            draw_overlay(frame, fps, stats)
            cv2.imshow(config.WINDOW_TITLE, frame)
            window_shown = True

            key = cv2.waitKey(1) & 0xFF
            if key in config.QUIT_KEYS:
                break
            if key in config.TOGGLE_HANDS_KEYS:
                pipeline.set_hands_enabled(not pipeline.hands_enabled)
            if key in config.TOGGLE_GESTURES_KEYS:
                pipeline.set_gestures_enabled(not pipeline.gestures_enabled)
            if key in config.TOGGLE_EXPRESSION_KEYS:
                pipeline.set_expression_enabled(not pipeline.expression_enabled)
            if window_shown and window_closed():
                break
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        pipeline.close()
        capture.release()
        cv2.destroyAllWindows()
        print("Camera released. Goodbye.")

    return 0


if __name__ == "__main__":
    sys.exit(main())