"""Unit tests for the expression helpers (no camera or model files needed)."""

import numpy as np
import pytest

from src.expression_detection import (
    EXPRESSION_LABELS,
    MODEL_INPUT_SIZE,
    ExpressionResult,
    expand_box,
    preprocess_face,
    softmax,
)
from src.expression_worker import smooth_result


def make_result(label):
    probabilities = {name: 0.0 for name in EXPRESSION_LABELS}
    probabilities[label] = 1.0
    return ExpressionResult(
        label=label, confidence=1.0, probabilities=probabilities,
        box=(0, 0, 10, 10), face_score=0.9,
    )


def test_there_are_eight_expression_labels():
    assert len(EXPRESSION_LABELS) == 8


def test_softmax_sums_to_one_and_keeps_order():
    probabilities = softmax([1.0, 2.0, 3.0])
    assert probabilities.sum() == pytest.approx(1.0)
    assert probabilities[2] > probabilities[1] > probabilities[0]


def test_softmax_handles_large_scores():
    probabilities = softmax([1000.0, 1001.0])
    assert not np.isnan(probabilities).any()
    assert probabilities.sum() == pytest.approx(1.0)


def test_expand_box_adds_padding():
    assert expand_box(50, 50, 100, 100, 0.1, 400, 400) == (40, 40, 160, 160)


def test_expand_box_stays_inside_the_frame():
    assert expand_box(10, 10, 100, 100, 0.1, 115, 115) == (0, 0, 115, 115)


def test_preprocess_face_shape_and_type():
    tensor = preprocess_face(np.zeros((100, 80, 3), dtype=np.uint8))
    assert tensor.shape == (1, 1, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)
    assert tensor.dtype == np.float32


def test_preprocess_face_keeps_the_0_to_255_scale():
    tensor = preprocess_face(np.full((50, 50, 3), 255, dtype=np.uint8))
    assert tensor.max() == pytest.approx(255.0)


def test_smoothing_disabled_returns_the_current_result():
    current = make_result("surprise")
    assert smooth_result(make_result("happiness"), current, 0.0) is current


def test_smoothing_without_a_previous_result_returns_the_current_one():
    current = make_result("surprise")
    assert smooth_result(None, current, 0.5) is current


def test_smoothing_blends_probabilities():
    blended = smooth_result(make_result("happiness"), make_result("surprise"), 0.4)
    assert blended.label == "surprise"
    assert blended.confidence == pytest.approx(0.6)
    assert blended.probabilities["happiness"] == pytest.approx(0.4)