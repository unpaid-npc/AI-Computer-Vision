"""Standalone test for facial expression recognition (Day 4).

Runs the recognizer on the webcam a few times per second and reports how long
each analysis takes. It runs on the main thread, so the video may stutter
slightly at each analysis; Day 5 moves it to a background thread.

Run from the project root with:  python -m scripts.try_expression
"""

import sys
import time

import cv2

from src import config
from src.app import open_camera
from src.expression_detection import ExpressionRecognizer, draw_expression

WINDOW = "Expression test"


def window_closed():
    try:
        return cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


def put_text(frame, text, row, color=(0, 255, 0)):
    cv2.putText(frame, text, (10, 30 + row * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def main():
    try:
        recognizer = ExpressionRecognizer(
            config.FACE_MODEL_PATH,
            config.EXPRESSION_MODEL_PATH,
            min_face_confidence=config.FACE_MIN_DETECTION_CONFIDENCE,
            padding=config.FACE_CROP_PADDING,
            num_threads=config.ONNX_NUM_THREADS,
        )
    except FileNotFoundError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"ERROR: Could not start the expression recognizer: {error}", file=sys.stderr)
        return 1

    capture = open_camera(config.CAMERA_INDEX)
    if capture is None:
        print("ERROR: Could not open the camera.", file=sys.stderr)
        recognizer.close()
        return 1
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    result = None
    last_run = 0.0
    last_ms = 0.0
    timings = []
    warned = False
    fps = 0.0
    previous = time.perf_counter()
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

            now = time.perf_counter()
            elapsed = now - previous
            previous = now
            if elapsed > 0:
                instant = 1.0 / elapsed
                fps = instant if fps == 0.0 else (
                    config.FPS_SMOOTHING * fps + (1 - config.FPS_SMOOTHING) * instant
                )

            # Analyse only a few times per second, not on every frame.
            if now - last_run >= config.EXPRESSION_INTERVAL_SEC:
                last_run = now
                try:
                    started = time.perf_counter()
                    result = recognizer.analyze(frame)
                    last_ms = (time.perf_counter() - started) * 1000
                    timings.append(last_ms)
                except Exception as error:
                    result = None
                    if not warned:
                        print(f"WARNING: analysis failed: {error}", file=sys.stderr)
                        warned = True

            put_text(frame, f"FPS: {fps:.1f}", 0)
            put_text(frame, f"Analysis time: {last_ms:.0f} ms", 1, (0, 255, 255))
            if result is not None:
                draw_expression(frame, result)
                ranked = sorted(result.probabilities.items(), key=lambda item: item[1], reverse=True)
                for row, (label, probability) in enumerate(ranked[:3], start=2):
                    put_text(frame, f"{label}: {probability:.0%}", row, (255, 255, 255))
            else:
                put_text(frame, "No face detected", 2, (0, 200, 255))

            cv2.putText(frame, "Predicted expression, not a measure of real emotion",
                        (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(frame, "Press Q or Esc to quit", (10, frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow(WINDOW, frame)
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
        recognizer.close()
        cv2.destroyAllWindows()
        if timings:
            steady = timings[1:] if len(timings) > 1 else timings  # first call includes warm-up
            print(f"Analyses run: {len(timings)}")
            print(f"Average analysis time: {sum(steady) / len(steady):.1f} ms "
                  f"(max {max(timings):.1f} ms)")
        print("Camera released. Goodbye.")

    return 0


if __name__ == "__main__":
    sys.exit(main())