"""Runs facial expression analysis on a background thread.

The main loop keeps drawing the webcam image and tracking hands at full speed.
It hands the newest frame to this worker a few times per second and always
shows the most recent finished result, so slow analysis never freezes the video.
"""

import threading
import time
from dataclasses import dataclass, replace

from src.expression_detection import EXPRESSION_LABELS


@dataclass(frozen=True)
class ExpressionSnapshot:
    """The worker's latest state, safe to read from the main thread."""

    result: object            # ExpressionResult, or None if no face was found
    finished_at: float        # time.monotonic() when analysis finished; 0.0 = none yet
    analysis_ms: float        # how long the last analysis took
    error: str                # message if the last analysis failed, else ""
    consecutive_errors: int   # analyses in a row that failed
    analyses: int             # total analyses completed


def smooth_result(previous, current, weight):
    """Blend two results' probabilities to reduce flicker.

    `weight` is the share kept from the previous result (0 disables smoothing).
    """
    if previous is None or weight <= 0.0:
        return current
    blended = {
        label: weight * previous.probabilities[label]
        + (1.0 - weight) * current.probabilities[label]
        for label in EXPRESSION_LABELS
    }
    best = max(blended, key=blended.get)
    return replace(current, label=best, confidence=blended[best], probabilities=blended)


class ExpressionWorker:
    """Owns a background thread that runs an ExpressionRecognizer."""

    def __init__(self, recognizer, smoothing=0.5):
        self._recognizer = recognizer
        self._smoothing = smoothing
        self._lock = threading.Lock()
        self._frame = None
        self._snapshot = ExpressionSnapshot(None, 0.0, 0.0, "", 0, 0)
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._previous = None  # only touched by the worker thread
        self._thread = threading.Thread(target=self._run, name="expression-worker", daemon=True)
        self._thread.start()

    def submit(self, frame_bgr):
        """Give the worker a frame. Never blocks; replaces a frame not yet started."""
        with self._lock:
            self._frame = frame_bgr.copy()
        self._wake.set()

    def latest(self):
        """Return the most recent ExpressionSnapshot."""
        with self._lock:
            return self._snapshot

    def stop(self):
        """Stop the thread and wait briefly for it to finish."""
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout=3.0)

    def _run(self):
        consecutive_errors = 0
        analyses = 0
        while True:
            self._wake.wait()
            if self._stop.is_set():
                return
            with self._lock:
                frame = self._frame
                self._frame = None
                self._wake.clear()
            if frame is None:
                continue

            started = time.perf_counter()
            error = ""
            try:
                result = self._recognizer.analyze(frame)
                consecutive_errors = 0
            except Exception as exc:  # keep the thread alive whatever happens
                result = None
                error = str(exc) or exc.__class__.__name__
                consecutive_errors += 1
            analysis_ms = (time.perf_counter() - started) * 1000.0

            if result is None:
                self._previous = None
            else:
                result = smooth_result(self._previous, result, self._smoothing)
                self._previous = result

            analyses += 1
            snapshot = ExpressionSnapshot(
                result=result,
                finished_at=time.monotonic(),
                analysis_ms=analysis_ms,
                error=error,
                consecutive_errors=consecutive_errors,
                analyses=analyses,
            )
            with self._lock:
                self._snapshot = snapshot