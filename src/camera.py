"""Camera helpers shared by the OpenCV window and the Qt interface."""

import cv2


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


def camera_help(index):
    """Troubleshooting hints to show when the camera cannot be opened."""
    return (
        "  - Close other apps that may be using the camera (Teams, Zoom, browser tabs).\n"
        "  - Check Windows Settings > Privacy & security > Camera, and allow desktop apps.\n"
        f"  - Try a different CAMERA_INDEX in src/config.py (currently {index})."
    )