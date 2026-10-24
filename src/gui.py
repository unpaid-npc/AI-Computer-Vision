"""PySide6 interface: live video, feature switches, detections, and performance.

Run from the project root with:  python -m src.gui

The camera and all analysis run on a background thread (CameraWorker), so the
window stays responsive. The vision work itself is done by VisionPipeline.
"""

import html
import sys
import threading

import cv2
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

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

LEVEL_HTML_COLORS = {
    LEVEL_OK: "#3fb950",
    LEVEL_WARN: "#d29922",
    LEVEL_INFO: "#58a6ff",
    LEVEL_MUTED: "#8b949e",
}

PLACEHOLDER_TEXT = "Camera stopped.\nPress \"Start camera\" to begin."

STYLE = """
QMainWindow, QWidget#root { background: #0f1115; }
QWidget { color: #e6edf3; font-family: "Segoe UI"; font-size: 10pt; }
QLabel#video { background: #000000; border: 1px solid #2a2f3a; border-radius: 8px;
               color: #8b949e; font-size: 12pt; }
QLabel#title { font-size: 15pt; font-weight: 600; }
QLabel#subtitle, QLabel#note { color: #8b949e; }
QGroupBox { background: #161a22; border: 1px solid #2a2f3a; border-radius: 8px;
            margin-top: 14px; padding: 14px 10px 10px 10px; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 4px; color: #8b949e; }
QPushButton { background: #232834; border: 1px solid #343b4a; border-radius: 6px;
              padding: 8px 12px; font-weight: 600; }
QPushButton:hover { background: #2b3142; }
QPushButton:disabled { color: #5b6270; background: #1a1e27; border-color: #2a2f3a; }
QPushButton#start { background: #238636; border-color: #2ea043; }
QPushButton#start:hover { background: #2ea043; }
QPushButton#start:disabled { background: #1a1e27; border-color: #2a2f3a; }
QPushButton#stop { background: #b62324; border-color: #da3633; }
QPushButton#stop:hover { background: #da3633; }
QPushButton#stop:disabled { background: #1a1e27; border-color: #2a2f3a; }
QCheckBox { spacing: 8px; padding: 3px 0; font-weight: 400; }
QCheckBox:disabled { color: #5b6270; }
"""


def frame_to_qimage(frame_bgr):
    """Convert an OpenCV BGR frame to a QImage that owns its pixel data."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    return QImage(rgb.data, width, height, 3 * width, QImage.Format.Format_RGB888).copy()


def lines_to_html(lines):
    """Turn (text, level) status lines into coloured rich text."""
    return "<br>".join(
        f'<span style="color:{LEVEL_HTML_COLORS[level]}">{html.escape(text)}</span>'
        for text, level in lines
    )


class CameraWorker(QThread):
    """Reads the camera and runs the pipeline on a background thread."""

    result_ready = Signal(QImage, object, float)   # annotated frame, FrameStats, FPS
    failed = Signal(str)

    def __init__(self, pipeline, parent=None):
        super().__init__(parent)
        self._pipeline = pipeline
        self._stop_requested = threading.Event()
        # At most two frames may be waiting for the window to draw them.
        self._slots = threading.Semaphore(2)

    def request_stop(self):
        self._stop_requested.set()

    def frame_consumed(self):
        self._slots.release()

    def run(self):
        capture = open_camera(config.CAMERA_INDEX)
        if capture is None:
            self.failed.emit(
                f"Could not open camera {config.CAMERA_INDEX}.\n{camera_help(config.CAMERA_INDEX)}"
            )
            return

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        fps_counter = FpsCounter(config.FPS_SMOOTHING)
        read_failures = 0

        try:
            while not self._stop_requested.is_set():
                success, frame = capture.read()
                if not success or frame is None:
                    read_failures += 1
                    if read_failures >= config.MAX_CONSECUTIVE_READ_FAILURES:
                        self.failed.emit("The camera stopped delivering frames.")
                        break
                    self.msleep(10)
                    continue
                read_failures = 0

                if config.MIRROR_VIEW:
                    frame = cv2.flip(frame, 1)

                fps = fps_counter.tick()
                stats = self._pipeline.process(frame)
                if self._slots.acquire(blocking=False):
                    self.result_ready.emit(frame_to_qimage(frame), stats, fps)
        except Exception as error:  # keep the window alive and report the problem
            self.failed.emit(f"Unexpected error: {error}")
        finally:
            capture.release()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Computer Vision: Hand Tracking and Facial Expression")
        self._pipeline = VisionPipeline()
        self._worker = None
        self._closing = False
        self._build_ui()
        self._set_message("\n".join(self._pipeline.warnings), LEVEL_WARN)
        self._warning_count = len(self._pipeline.warnings)
        self._reset_readouts()

        # Keep the checkboxes in step with the pipeline (it can switch a feature off after errors).
        self._sync_timer = QTimer(self)
        self._sync_timer.timeout.connect(self._sync_controls)
        self._sync_timer.start(250)
        self._sync_controls()

    # ----- layout -----------------------------------------------------------

    def _build_ui(self):
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        self.video_label = QLabel(PLACEHOLDER_TEXT)
        self.video_label.setObjectName("video")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        layout.addWidget(self.video_label, stretch=1)

        panel = QWidget()
        panel.setFixedWidth(310)
        side = QVBoxLayout(panel)
        side.setContentsMargins(0, 0, 0, 0)
        side.setSpacing(12)

        title = QLabel("AI Computer Vision")
        title.setObjectName("title")
        subtitle = QLabel("Hand tracking, gestures and expression prediction")
        subtitle.setObjectName("subtitle")
        subtitle.setWordWrap(True)
        side.addWidget(title)
        side.addWidget(subtitle)

        buttons = QHBoxLayout()
        self.start_button = QPushButton("Start camera")
        self.start_button.setObjectName("start")
        self.stop_button = QPushButton("Stop camera")
        self.stop_button.setObjectName("stop")
        self.stop_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_camera)
        self.stop_button.clicked.connect(self.stop_camera)
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.stop_button)
        side.addLayout(buttons)

        features = QGroupBox("Features")
        features_layout = QVBoxLayout(features)
        self.hands_box = QCheckBox("Hand tracking")
        self.gestures_box = QCheckBox("Gesture recognition")
        self.expression_box = QCheckBox("Facial expression")
        for box in (self.hands_box, self.gestures_box, self.expression_box):
            features_layout.addWidget(box)
        self.hands_box.setChecked(self._pipeline.hands_enabled)
        self.gestures_box.setChecked(self._pipeline.gestures_enabled)
        self.expression_box.setChecked(self._pipeline.expression_enabled)
        # Connect after the initial state is set, so setting it does not trigger the switches.
        self.hands_box.toggled.connect(self._pipeline.set_hands_enabled)
        self.gestures_box.toggled.connect(self._pipeline.set_gestures_enabled)
        self.expression_box.toggled.connect(self._pipeline.set_expression_enabled)
        if not self._pipeline.hands_available:
            self.hands_box.setToolTip("Model file missing. Run: python scripts/download_models.py")
        if not self._pipeline.expression_available:
            self.expression_box.setToolTip("Model file missing. Run: python scripts/download_models.py")
        side.addWidget(features)

        detections = QGroupBox("Detections")
        detections_layout = QVBoxLayout(detections)
        self.detections_label = QLabel()
        self.detections_label.setTextFormat(Qt.TextFormat.RichText)
        self.detections_label.setWordWrap(True)
        detections_layout.addWidget(self.detections_label)
        side.addWidget(detections)

        performance = QGroupBox("Performance")
        form = QFormLayout(performance)
        self.fps_value = QLabel()
        self.frame_value = QLabel()
        self.hand_value = QLabel()
        self.expression_value = QLabel()
        form.addRow("FPS", self.fps_value)
        form.addRow("Frame processing", self.frame_value)
        form.addRow("Hands + gestures", self.hand_value)
        form.addRow("Expression analysis", self.expression_value)
        side.addWidget(performance)

        self.message_label = QLabel()
        self.message_label.setWordWrap(True)
        side.addWidget(self.message_label)

        side.addStretch(1)
        note = QLabel("Expressions are predictions from a small pretrained model, "
                      "not a measure of real emotion.")
        note.setObjectName("note")
        note.setWordWrap(True)
        side.addWidget(note)

        layout.addWidget(panel)
        self.resize(1120, 640)

    # ----- camera control ---------------------------------------------------

    def start_camera(self):
        if self._worker is not None:
            return
        self._set_message("\n".join(self._pipeline.warnings), LEVEL_WARN)
        worker = CameraWorker(self._pipeline, self)
        worker.result_ready.connect(self._on_result)
        worker.failed.connect(self._on_failed)
        worker.finished.connect(self._on_worker_finished)
        self._worker = worker
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.video_label.setText("Starting camera...")
        worker.start()

    def stop_camera(self):
        if self._worker is not None:
            self._worker.request_stop()
            self.stop_button.setEnabled(False)

    def _on_result(self, image, stats, fps):
        if self._closing:
            return
        if self._worker is not None:
            self._worker.frame_consumed()
        pixmap = QPixmap.fromImage(image).scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.video_label.setPixmap(pixmap)

        self.detections_label.setText(lines_to_html(stats.summary_lines()))
        self.fps_value.setText(f"{fps:.1f}")
        self.frame_value.setText(f"{stats.process_ms:.0f} ms")
        self.hand_value.setText(f"{stats.hand_ms:.0f} ms" if stats.hands_state == "on" else "off")
        self.expression_value.setText(
            f"{stats.expression_ms:.0f} ms" if stats.expression_ms > 0 else "-"
        )

        warnings = self._pipeline.warnings
        if len(warnings) > self._warning_count:
            self._warning_count = len(warnings)
            self._set_message("\n".join(warnings), LEVEL_WARN)

    def _on_failed(self, message):
        if not self._closing:
            self._set_message(message, LEVEL_WARN)

    def _on_worker_finished(self):
        if self._closing:
            return
        worker = self._worker
        self._worker = None
        if worker is not None:
            worker.deleteLater()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.video_label.clear()
        self.video_label.setText(PLACEHOLDER_TEXT)
        self._reset_readouts()

    # ----- small helpers ----------------------------------------------------

    def _reset_readouts(self):
        self.detections_label.setText(lines_to_html([("Camera stopped", LEVEL_MUTED)]))
        for label in (self.fps_value, self.frame_value, self.hand_value, self.expression_value):
            label.setText("-")

    def _set_message(self, text, level):
        self.message_label.setText(text)
        self.message_label.setStyleSheet(f"color: {LEVEL_HTML_COLORS[level]};")
        self.message_label.setVisible(bool(text))

    def _sync_controls(self):
        pipeline = self._pipeline
        wanted = (
            (self.hands_box, pipeline.hands_enabled),
            (self.gestures_box, pipeline.gestures_enabled),
            (self.expression_box, pipeline.expression_enabled),
        )
        for box, value in wanted:
            if box.isChecked() != value:
                box.blockSignals(True)
                box.setChecked(value)
                box.blockSignals(False)
        self.hands_box.setEnabled(pipeline.hands_available)
        self.gestures_box.setEnabled(pipeline.hands_available and pipeline.hands_enabled)
        self.expression_box.setEnabled(pipeline.expression_available)

    def closeEvent(self, event):
        self._closing = True
        self._sync_timer.stop()
        if self._worker is not None:
            self._worker.request_stop()
            self._worker.wait(5000)
        self._pipeline.close()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())