"""Acceptance tests for the normal upload-and-predict user flow.

These tests upload real images and check that a digit is rendered. The earlier
versions posted fake bytes and only looked for the word "Prediction", which is
also part of the error page heading, so they passed even when the app failed.
"""

import re
from io import BytesIO
from pathlib import Path

from PIL import Image

SAMPLE_IMAGE = Path(__file__).parent / "test_images" / "3" / "Sign 3 (122).jpeg"
ERROR_MESSAGE = b"File cannot be processed."
PREDICTED_DIGIT = re.compile(rb'<h2 class="display-4 text-dark font-weight-bold">([0-9])</h2>')


def _jpeg_upload(size=None):
    """Return the sample hand-sign image as an in-memory JPEG, optionally resized."""
    img = Image.open(SAMPLE_IMAGE).convert("RGB")
    if size is not None:
        img = img.resize(size)
    buf = BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf


def _post_image(client, buf, name):
    """POST an image to /prediction the way the upload form does."""
    return client.post(
        "/prediction",
        data={"file": (buf, name)},
        content_type="multipart/form-data",
    )


def _assert_digit_predicted(response):
    """The result page must show a single predicted digit and no error."""
    assert response.status_code == 200
    assert ERROR_MESSAGE not in response.data
    assert PREDICTED_DIGIT.search(response.data), "no predicted digit rendered"


def test_acceptance_successful_upload(client):
    """A valid hand-sign photo is accepted and a digit 0-9 is predicted."""
    _assert_digit_predicted(_post_image(client, _jpeg_upload(), "sign.jpg"))


def test_acceptance_valid_large_image(client):
    """A large (2000x2000) but valid image is downscaled and still predicted."""
    buf = _jpeg_upload(size=(2000, 2000))
    assert buf.getbuffer().nbytes > 100_000  # really is a large upload
    _assert_digit_predicted(_post_image(client, buf, "large_image.jpg"))


def test_acceptance_valid_image_size_upload(client):
    """A non-square image of a different resolution is resized and predicted."""
    _assert_digit_predicted(_post_image(client, _jpeg_upload(size=(640, 480)), "wide.jpg"))


def test_fake_image_bytes_show_error_not_prediction(client):
    """Regression guard: non-image bytes must produce the error, never a digit."""
    response = _post_image(client, BytesIO(b"fake_image_data"), "test_image.jpg")
    assert response.status_code == 200
    assert ERROR_MESSAGE in response.data
    assert not PREDICTED_DIGIT.search(response.data)
