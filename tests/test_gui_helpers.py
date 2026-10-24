"""Tests for the GUI's pure helper functions (no window or camera needed)."""

import pytest

pytest.importorskip("PySide6")

import numpy as np  # noqa: E402

from src.gui import frame_to_qimage, lines_to_html  # noqa: E402
from src.pipeline import LEVEL_OK, LEVEL_WARN  # noqa: E402


def test_frame_to_qimage_has_the_right_size_and_colour_order():
    frame = np.zeros((4, 6, 3), dtype=np.uint8)
    frame[:, :] = (255, 0, 0)  # OpenCV order is BGR, so this is blue
    image = frame_to_qimage(frame)
    assert (image.width(), image.height()) == (6, 4)
    colour = image.pixelColor(0, 0)
    assert (colour.red(), colour.green(), colour.blue()) == (0, 0, 255)


def test_lines_to_html_colours_lines_and_escapes_text():
    result = lines_to_html([("a < b", LEVEL_OK), ("second", LEVEL_WARN)])
    assert "a &lt; b" in result
    assert "#3fb950" in result and "#d29922" in result
    assert "<br>" in result