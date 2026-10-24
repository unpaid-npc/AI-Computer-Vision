"""Unit tests for the pipeline's FPS counter and status text (no camera or models needed)."""

import pytest

from src.pipeline import (
    LEVEL_INFO,
    LEVEL_MUTED,
    LEVEL_OK,
    LEVEL_WARN,
    FpsCounter,
    FrameStats,
)


def texts(stats):
    return [text for text, _ in stats.summary_lines()]


def test_fps_counter_first_tick_has_no_value():
    assert FpsCounter().tick(0.0) == 0.0


def test_fps_counter_reports_a_steady_rate():
    counter = FpsCounter(smoothing=0.9)
    fps = 0.0
    for frame in range(50):
        fps = counter.tick(frame * 0.1)
    assert fps == pytest.approx(10.0)


def test_fps_counter_ignores_zero_elapsed_time():
    counter = FpsCounter()
    counter.tick(1.0)
    assert counter.tick(1.0) == 0.0


def test_default_stats_report_everything_unavailable():
    assert texts(FrameStats()) == ["Hands: unavailable", "Expression: unavailable"]


def test_hands_off_is_reported():
    assert "Hands: OFF" in texts(FrameStats(hands_state="off"))


def test_no_hands_is_a_warning_and_no_gesture_is_muted():
    lines = FrameStats(hands_state="on", hand_count=0, gestures_enabled=True).summary_lines()
    assert lines[0] == ("Hands detected: 0", LEVEL_WARN)
    assert lines[1] == ("Gesture: none", LEVEL_MUTED)


def test_gestures_are_listed():
    stats = FrameStats(hands_state="on", hand_count=2, gestures_enabled=True,
                       gestures=("Right Fist", "Left Peace"))
    assert ("Gesture: Right Fist, Left Peace", LEVEL_OK) in stats.summary_lines()


def test_gestures_off_is_reported():
    stats = FrameStats(hands_state="on", hand_count=1, gestures_enabled=False)
    assert ("Gestures: OFF", LEVEL_MUTED) in stats.summary_lines()


def test_expression_result_and_analysis_time_are_shown():
    stats = FrameStats(expression_state="ok", expression_label="happiness",
                       expression_confidence=0.83, expression_ms=12.4)
    lines = stats.summary_lines()
    assert ("Expression: happiness (83%)", LEVEL_OK) in lines
    assert ("Analysis: 12 ms", LEVEL_INFO) in lines


def test_no_face_is_a_warning():
    stats = FrameStats(expression_state="no_face", expression_ms=9.0)
    assert ("Expression: no face detected", LEVEL_WARN) in stats.summary_lines()


def test_analysing_shows_no_analysis_time():
    stats = FrameStats(expression_state="analysing")
    assert not any(text.startswith("Analysis") for text in texts(stats))