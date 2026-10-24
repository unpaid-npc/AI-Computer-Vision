"""The vision pipeline: hand tracking, gestures, and facial expression, one frame at a time.

Both the OpenCV window (app.py) and the Qt interface (gui.py) use this module, so
the computer-vision logic lives in one place and the user interfaces only handle
display and input.
"""

import sys
import time
from dataclasses import dataclass, replace

from src import config
from src.expression_detection import ExpressionRecognizer, draw_expression
from src.expression_worker import ExpressionWorker
from src.gesture_recognition import (
    GESTURE_UNKNOWN,
    GestureSmoother,
    classify_gesture,
    draw_gestures,
    hand_keys,
)
from src.hand_tracking import HandTracker, draw_hands

# Severity of a status line, so each interface can choose its own colours.
LEVEL_OK = "ok"
LEVEL_WARN = "warn"
LEVEL_INFO = "info"
LEVEL_MUTED = "muted"


class FpsCounter:
    """Smoothed frames-per-second counter. Call tick() once per frame."""

    def __init__(self, smoothing=0.9):
        self._smoothing = smoothing
        self._previous = None
        self.fps = 0.0

    def tick(self, now=None):
        now = time.perf_counter() if now is None else now
        if self._previous is not None:
            elapsed = now - self._previous
            if elapsed > 0:
                instant = 1.0 / elapsed
                self.fps = instant if self.fps == 0.0 else (
                    self._smoothing * self.fps + (1.0 - self._smoothing) * instant
                )
        self._previous = now
        return self.fps


@dataclass(frozen=True)
class FrameStats:
    """What the pipeline found in one frame, plus timing."""

    hands_state: str = "unavailable"        # "unavailable" | "off" | "on"
    hand_count: int = 0
    gestures_enabled: bool = False
    gestures: tuple = ()                    # e.g. ("Right Fist",); only implemented gestures
    expression_state: str = "unavailable"   # "unavailable" | "off" | "analysing" | "no_face" | "ok" | "stopped"
    expression_label: str = ""
    expression_confidence: float = 0.0
    expression_ms: float = 0.0              # how long the last expression analysis took
    hand_ms: float = 0.0                    # hand tracking + gestures for this frame
    process_ms: float = 0.0                 # the whole process() call for this frame

    def summary_lines(self):
        """Return a list of (text, level) status lines for display."""
        lines = []

        if self.hands_state == "unavailable":
            lines.append(("Hands: unavailable", LEVEL_MUTED))
        elif self.hands_state == "off":
            lines.append(("Hands: OFF", LEVEL_MUTED))
        else:
            lines.append((f"Hands detected: {self.hand_count}",
                          LEVEL_OK if self.hand_count else LEVEL_WARN))
            if not self.gestures_enabled:
                lines.append(("Gestures: OFF", LEVEL_MUTED))
            elif self.gestures:
                lines.append(("Gesture: " + ", ".join(self.gestures), LEVEL_OK))
            else:
                lines.append(("Gesture: none", LEVEL_MUTED))

        state = self.expression_state
        if state == "unavailable":
            lines.append(("Expression: unavailable", LEVEL_MUTED))
        elif state == "off":
            lines.append(("Expression: OFF", LEVEL_MUTED))
        elif state == "analysing":
            lines.append(("Expression: analysing...", LEVEL_WARN))
        else:
            if state == "ok":
                lines.append((f"Expression: {self.expression_label} "
                              f"({self.expression_confidence:.0%})", LEVEL_OK))
            elif state == "no_face":
                lines.append(("Expression: no face detected", LEVEL_WARN))
            else:
                lines.append(("Expression: stopped (errors)", LEVEL_WARN))
            lines.append((f"Analysis: {self.expression_ms:.0f} ms", LEVEL_INFO))
        return lines


class VisionPipeline:
    """Runs hand tracking, gesture recognition, and expression analysis on frames.

    Call process(frame) for every camera frame. It draws on the frame in place and
    returns a FrameStats summary. The feature switches are plain booleans, so they
    can be changed from the interface thread while process() runs on another.
    """

    def __init__(self):
        self.warnings = []   # human-readable messages about features that could not start
        self._hand_tracker = self._create_hand_tracker() if config.ENABLE_HAND_TRACKING else None
        self._recognizer = self._create_recognizer() if config.ENABLE_EXPRESSION else None
        self._worker = (
            ExpressionWorker(self._recognizer, smoothing=config.EXPRESSION_SMOOTHING)
            if self._recognizer is not None else None
        )
        self._smoother = GestureSmoother(window=config.GESTURE_SMOOTHING_WINDOW)
        self._hands_enabled = self._hand_tracker is not None
        self._gestures_enabled = config.ENABLE_GESTURES
        self._expression_enabled = self._worker is not None
        self._hand_errors = 0
        self._last_submit = 0.0
        self._last_expression_error = ""
        self._closed = False

    # ----- feature switches -------------------------------------------------

    @property
    def hands_available(self):
        return self._hand_tracker is not None

    @property
    def expression_available(self):
        return self._worker is not None

    @property
    def hands_enabled(self):
        return self._hands_enabled

    @property
    def gestures_enabled(self):
        return self._gestures_enabled

    @property
    def expression_enabled(self):
        return self._expression_enabled

    def set_hands_enabled(self, enabled):
        self._hands_enabled = bool(enabled) and self._hand_tracker is not None
        self._hand_errors = 0

    def set_gestures_enabled(self, enabled):
        self._gestures_enabled = bool(enabled)

    def set_expression_enabled(self, enabled):
        self._expression_enabled = bool(enabled) and self._worker is not None
        self._last_expression_error = ""

    # ----- processing -------------------------------------------------------

    def process(self, frame):
        """Analyse one BGR frame, draw the results onto it, and return FrameStats."""
        started = time.perf_counter()

        # Give the expression worker a clean copy of the frame (before anything is drawn on it).
        if (self._worker is not None and self._expression_enabled
                and started - self._last_submit >= config.EXPRESSION_INTERVAL_SEC):
            self._last_submit = started
            self._worker.submit(frame)

        hands_state, hand_count, gesture_names, hand_ms = self._process_hands(frame)
        expr_state, expr_label, expr_conf, expr_ms = self._process_expression(frame)

        return FrameStats(
            hands_state=hands_state,
            hand_count=hand_count,
            gestures_enabled=self._gestures_enabled,
            gestures=gesture_names,
            expression_state=expr_state,
            expression_label=expr_label,
            expression_confidence=expr_conf,
            expression_ms=expr_ms,
            hand_ms=hand_ms,
            process_ms=(time.perf_counter() - started) * 1000.0,
        )

    def close(self):
        """Stop the worker thread and release the models. Safe to call twice."""
        if self._closed:
            return
        self._closed = True
        if self._worker is not None:
            self._worker.stop()
        if self._recognizer is not None:
            self._recognizer.close()
        if self._hand_tracker is not None:
            self._hand_tracker.close()

    # ----- internals --------------------------------------------------------

    def _warn(self, message):
        print(f"WARNING: {message}", file=sys.stderr)
        self.warnings.append(message)

    def _create_hand_tracker(self):
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
            self._warn(f"hand tracking disabled. {error}")
        except Exception as error:
            self._warn(f"hand tracking disabled. Could not start it: {error}")
        return None

    def _create_recognizer(self):
        try:
            return ExpressionRecognizer(
                config.FACE_MODEL_PATH,
                config.EXPRESSION_MODEL_PATH,
                min_face_confidence=config.FACE_MIN_DETECTION_CONFIDENCE,
                padding=config.FACE_CROP_PADDING,
                num_threads=config.ONNX_NUM_THREADS,
            )
        except FileNotFoundError as error:
            self._warn(f"expression recognition disabled. {error}")
        except Exception as error:
            self._warn(f"expression recognition disabled. Could not start it: {error}")
        return None

    def _recognize_gestures(self, hands, frame):
        """Classify each hand's gesture and smooth the result over recent frames."""
        if not hands:
            self._smoother.update({})
            return []
        height, width = frame.shape[:2]
        aspect_ratio = width / height
        raw = [classify_gesture(hand.landmarks, aspect_ratio) for hand in hands]
        keys = hand_keys([hand.label for hand in hands])
        smoothed = self._smoother.update({key: result.name for key, result in zip(keys, raw)})
        return [replace(result, name=smoothed[key]) for key, result in zip(keys, raw)]

    def _process_hands(self, frame):
        """Returns (state, hand_count, gesture_names, milliseconds)."""
        if self._hand_tracker is None:
            return "unavailable", 0, (), 0.0
        if not self._hands_enabled:
            self._smoother.update({})
            return "off", 0, (), 0.0

        started = time.perf_counter()
        hands, gestures = [], []
        try:
            hands = self._hand_tracker.detect(frame)
            if self._gestures_enabled:
                gestures = self._recognize_gestures(hands, frame)
            self._hand_errors = 0
        except Exception as error:
            self._hand_errors += 1
            if self._hand_errors == 1:
                print(f"WARNING: hand tracking error: {error}", file=sys.stderr)
            if self._hand_errors >= config.MAX_CONSECUTIVE_INFERENCE_ERRORS:
                self._warn("hand tracking keeps failing, so it was turned off.")
                self._hands_enabled = False

        draw_hands(frame, hands)
        if gestures:
            draw_gestures(frame, hands, gestures, config.SHOW_GESTURE_DEBUG)
        names = tuple(
            f"{hand.label} {gesture.name}"
            for hand, gesture in zip(hands, gestures)
            if gesture.name != GESTURE_UNKNOWN
        )
        return "on", len(hands), names, (time.perf_counter() - started) * 1000.0

    def _process_expression(self, frame):
        """Returns (state, label, confidence, analysis_ms) from the worker's latest result."""
        if self._worker is None:
            return "unavailable", "", 0.0, 0.0
        if not self._expression_enabled:
            return "off", "", 0.0, 0.0

        snapshot = self._worker.latest()
        stale = time.monotonic() - snapshot.finished_at > config.EXPRESSION_STALE_SEC
        if snapshot.finished_at <= 0.0 or stale:
            return "analysing", "", 0.0, 0.0

        if snapshot.error:
            if snapshot.error != self._last_expression_error:
                print(f"WARNING: expression analysis error: {snapshot.error}", file=sys.stderr)
                self._last_expression_error = snapshot.error
        else:
            self._last_expression_error = ""

        if snapshot.consecutive_errors >= config.MAX_CONSECUTIVE_INFERENCE_ERRORS:
            self._warn("expression analysis keeps failing, so it was turned off.")
            self._expression_enabled = False
            return "stopped", "", 0.0, snapshot.analysis_ms
        if snapshot.result is None:
            return "no_face", "", 0.0, snapshot.analysis_ms

        draw_expression(frame, snapshot.result)
        return "ok", snapshot.result.label, snapshot.result.confidence, snapshot.analysis_ms