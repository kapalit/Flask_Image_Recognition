"""Tests for the Flask routes in app.py.

The model is mocked with monkeypatch so these tests check the web layer (routing,
request handling, templates) quickly and without depending on TensorFlow output.
"""

from io import BytesIO

import numpy as np
import pytest
from PIL import Image

ERROR_TEXT = b"File cannot be processed."


def make_image_bytes(fmt="JPEG", size=(64, 64), mode="RGB", name="digit.jpg"):
    """Return an in-memory image file suitable for a multipart upload."""
    buffer = BytesIO()
    Image.new(mode, size, color=128).save(buffer, format=fmt)
    buffer.seek(0)
    return buffer, name


@pytest.fixture
def fake_model(monkeypatch):
    """Replace the real prediction with a stub that always predicts digit 3."""
    calls = []

    def fake_predict(img):
        calls.append(img)
        return np.int64(3)

    monkeypatch.setattr("app.predict_result", fake_predict)
    return calls


# ---------- Home page ----------

def test_home_page_loads(client):
    """GET / returns 200 and renders the upload form."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"Hand Sign Digit" in response.data
    assert b'name="file"' in response.data
    assert b'action="/prediction"' in response.data


def test_unknown_route_returns_404(client):
    """An undefined URL is rejected with 404 Not Found."""
    assert client.get("/does-not-exist").status_code == 404


def test_prediction_route_rejects_get(client):
    """/prediction only accepts POST; GET returns 405 Method Not Allowed."""
    assert client.get("/prediction").status_code == 405


# ---------- Valid requests ----------

def test_valid_upload_shows_prediction(client, fake_model):
    """A valid JPEG upload renders the result page with the predicted digit."""
    response = client.post(
        "/prediction",
        data={"file": make_image_bytes()},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    assert b"Prediction" in response.data
    assert b">3</h2>" in response.data
    assert ERROR_TEXT not in response.data
    assert len(fake_model) == 1


def test_valid_upload_passes_preprocessed_batch_to_model(client, fake_model):
    """The model receives a normalised (1, 224, 224, 3) batch, not the raw upload."""
    client.post(
        "/prediction",
        data={"file": make_image_bytes(size=(500, 300))},
        content_type="multipart/form-data",
    )
    batch = fake_model[0]
    assert batch.shape == (1, 224, 224, 3)
    assert 0.0 <= batch.min() and batch.max() <= 1.0


@pytest.mark.parametrize(
    "fmt,mode,name",
    [
        ("PNG", "RGB", "digit.png"),
        ("PNG", "RGBA", "transparent.png"),
        ("PNG", "L", "grayscale.png"),
        ("BMP", "RGB", "digit.bmp"),
    ],
)
def test_other_image_formats_are_accepted(client, fake_model, fmt, mode, name):
    """PNG (incl. transparent and grayscale) and BMP uploads are predicted, not rejected."""
    response = client.post(
        "/prediction",
        data={"file": make_image_bytes(fmt=fmt, mode=mode, name=name)},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    assert b">3</h2>" in response.data
    assert len(fake_model) == 1


# ---------- Missing / invalid input ----------

def test_missing_file_field_shows_error(client, fake_model):
    """A POST without any 'file' field shows the error message instead of crashing."""
    response = client.post("/prediction", data={}, content_type="multipart/form-data")
    assert response.status_code == 200
    assert ERROR_TEXT in response.data
    assert not fake_model


def test_wrong_field_name_shows_error(client, fake_model):
    """An image sent under a different form field name is treated as missing."""
    response = client.post(
        "/prediction",
        data={"image": make_image_bytes()},
        content_type="multipart/form-data",
    )
    assert ERROR_TEXT in response.data
    assert not fake_model


def test_empty_file_shows_error(client, fake_model):
    """Submitting the form with no file chosen (empty upload) shows the error."""
    response = client.post(
        "/prediction",
        data={"file": (BytesIO(b""), "")},
        content_type="multipart/form-data",
    )
    assert ERROR_TEXT in response.data
    assert not fake_model


def test_text_file_shows_error(client, fake_model):
    """A non-image file is rejected with the error message."""
    response = client.post(
        "/prediction",
        data={"file": (BytesIO(b"this is not an image"), "notes.txt")},
        content_type="multipart/form-data",
    )
    assert ERROR_TEXT in response.data
    assert not fake_model


def test_corrupt_image_shows_error(client, fake_model):
    """A file with a JPEG header but truncated/garbage body is rejected."""
    corrupt = b"\xff\xd8\xff\xe0" + b"\x00" * 50
    response = client.post(
        "/prediction",
        data={"file": (BytesIO(corrupt), "broken.jpg")},
        content_type="multipart/form-data",
    )
    assert ERROR_TEXT in response.data
    assert not fake_model


def test_image_with_wrong_extension_is_still_predicted(client, fake_model):
    """Content, not file extension, decides validity: a PNG named .txt still works."""
    response = client.post(
        "/prediction",
        data={"file": make_image_bytes(fmt="PNG", name="digit.txt")},
        content_type="multipart/form-data",
    )
    assert b">3</h2>" in response.data


def test_model_failure_shows_error(client, monkeypatch):
    """If the model raises a ValueError (e.g. bad input shape) the user sees the error."""

    def broken_predict(_img):
        raise ValueError("bad input shape")

    monkeypatch.setattr("app.predict_result", broken_predict)
    response = client.post(
        "/prediction",
        data={"file": make_image_bytes()},
        content_type="multipart/form-data",
    )
    assert ERROR_TEXT in response.data
