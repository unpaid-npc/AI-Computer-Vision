"""Facial expression recognition.

Pipeline: MediaPipe face detector finds the face, the face is cropped to a
64x64 grayscale image, and the FER+ model (run with ONNX Runtime on the CPU)
scores 8 expression categories.

The output is a PREDICTED FACIAL EXPRESSION. It is not a measurement of what a
person actually feels.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
RunningMode = mp.tasks.vision.RunningMode

# Order defined by the FER+ model.
EXPRESSION_LABELS = (
    "neutral", "happiness", "surprise", "sadness",
    "anger", "disgust", "fear", "contempt",
)
MODEL_INPUT_SIZE = 64


@dataclass(frozen=True)
class ExpressionResult:
    """The predicted expression for one face."""

    label: str            # most likely expression category
    confidence: float     # model probability for that category, 0.0 to 1.0
    probabilities: dict   # every category -> probability
    box: tuple            # (x, y, width, height) of the analysed face crop, in pixels
    face_score: float     # confidence of the face detector


def softmax(scores):
    """Turn raw model scores into probabilities that sum to 1."""
    scores = np.asarray(scores, dtype=np.float64).ravel()
    exp = np.exp(scores - scores.max())
    return exp / exp.sum()


def expand_box(x, y, width, height, padding, frame_width, frame_height):
    """Grow a face box by `padding` (a fraction) and clamp it to the frame.

    Returns (x0, y0, x1, y1).
    """
    pad_x = int(width * padding)
    pad_y = int(height * padding)
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(frame_width, x + width + pad_x)
    y1 = min(frame_height, y + height + pad_y)
    return x0, y0, x1, y1


def preprocess_face(face_bgr):
    """Convert a BGR face crop to the model input: float32 (1, 1, 64, 64), values 0-255."""
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32).reshape(1, 1, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)


class ExpressionRecognizer:
    """Detects the largest face in a frame and predicts its expression."""

    def __init__(self, face_model_path, expression_model_path, min_face_confidence=0.5,
                 padding=0.15, num_threads=2):
        face_model_path = Path(face_model_path)
        expression_model_path = Path(expression_model_path)
        for path in (face_model_path, expression_model_path):
            if not path.exists():
                raise FileNotFoundError(
                    f"Model file not found: {path}.\n"
                    "  Download the models by running:  python scripts/download_models.py"
                )

        face_options = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=str(face_model_path)),
            running_mode=RunningMode.IMAGE,
            min_detection_confidence=min_face_confidence,
        )
        self._face_detector = FaceDetector.create_from_options(face_options)

        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = num_threads
        self._session = ort.InferenceSession(
            str(expression_model_path),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name
        self._padding = padding

    def analyze(self, frame_bgr):
        """Return an ExpressionResult for the largest face, or None if no face is found."""
        frame_height, frame_width = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        detections = self._face_detector.detect(mp_image).detections
        if not detections:
            return None

        detection = max(detections, key=lambda d: d.bounding_box.width * d.bounding_box.height)
        box = detection.bounding_box
        x0, y0, x1, y1 = expand_box(
            box.origin_x, box.origin_y, box.width, box.height,
            self._padding, frame_width, frame_height,
        )
        if x1 - x0 < 20 or y1 - y0 < 20:
            return None  # face too small to analyse reliably

        tensor = preprocess_face(frame_bgr[y0:y1, x0:x1])
        scores = self._session.run(None, {self._input_name: tensor})[0]
        probabilities = softmax(scores)
        best = int(np.argmax(probabilities))

        face_score = detection.categories[0].score if detection.categories else 0.0
        return ExpressionResult(
            label=EXPRESSION_LABELS[best],
            confidence=float(probabilities[best]),
            probabilities={label: float(p) for label, p in zip(EXPRESSION_LABELS, probabilities)},
            box=(x0, y0, x1 - x0, y1 - y0),
            face_score=float(face_score),
        )

    def close(self):
        """Release the face detector."""
        self._face_detector.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def draw_expression(frame, result):
    """Draw the face box and the predicted expression with its confidence."""
    x, y, width, height = result.box
    color = (255, 200, 0)
    cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
    text = f"Predicted: {result.label} ({result.confidence:.0%})"
    text_y = y - 10 if y > 30 else y + height + 25
    cv2.putText(frame, text, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)