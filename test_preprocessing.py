"""Unit tests for preprocess_img and predict_result in model.py.

Preprocessing tests use small in-memory images. The prediction tests use the real
model on the bundled sample images, plus a mocked model for the argmax logic.
"""

import os
from io import BytesIO

import numpy as np
import pytest
from PIL import Image, UnidentifiedImageError

import model
from model import preprocess_img, predict_result


def image_file(size=(50, 80), mode="RGB", fmt="PNG", color=255):
    """Return a file-like object containing a solid-colour image."""
    buffer = BytesIO()
    Image.new(mode, size, color=color).save(buffer, format=fmt)
    buffer.seek(0)
    return buffer


# ---------- preprocess_img: normal cases ----------

def test_preprocess_output_shape_and_dtype():
    """Any input size is resized to a single (1, 224, 224, 3) float batch."""
    result = preprocess_img(image_file(size=(640, 480)))
    assert result.shape == (1, 224, 224, 3)
    assert result.dtype == np.float32


def test_preprocess_scales_pixels_to_unit_range():
    """Pixel values are divided by 255: white becomes 1.0, black becomes 0.0."""
    assert np.allclose(preprocess_img(image_file(color=255)), 1.0)
    assert np.allclose(preprocess_img(image_file(color=0)), 0.0)


def test_preprocess_accepts_file_path():
    """preprocess_img also accepts a filesystem path (used by the sample images)."""
    result = preprocess_img("test_images/2/Sign 2 (97).jpeg")
    assert result.shape == (1, 224, 224, 3)


# ---------- preprocess_img: edge cases ----------

@pytest.mark.parametrize("size", [(1, 1), (224, 224), (3000, 10)])
def test_preprocess_handles_extreme_sizes(size):
    """Tiny, exact-size and very wide images are all resized correctly."""
    assert preprocess_img(image_file(size=size)).shape == (1, 224, 224, 3)


@pytest.mark.parametrize("mode", ["L", "RGBA", "P", "1"])
def test_preprocess_converts_non_rgb_modes(mode):
    """Grayscale, transparent, palette and 1-bit images are converted to 3 channels."""
    color = 1 if mode == "1" else 255
    result = preprocess_img(image_file(mode=mode, color=color))
    assert result.shape == (1, 224, 224, 3)


def test_preprocess_rejects_non_image_bytes():
    """Bytes that are not an image raise UnidentifiedImageError."""
    with pytest.raises(UnidentifiedImageError):
        preprocess_img(BytesIO(b"definitely not an image"))


def test_preprocess_rejects_missing_path():
    """A path that does not exist raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        preprocess_img("test_images/does_not_exist.jpeg")


# ---------- predict_result ----------

def test_predict_result_returns_argmax_of_model_output(monkeypatch):
    """predict_result returns the index of the highest score from the model."""

    class FakeModel:  # pylint: disable=too-few-public-methods
        """Minimal stand-in for the Keras model."""

        def predict(self, batch, verbose=0):  # pylint: disable=unused-argument
            """Return fixed class scores where class 7 is highest."""
            scores = np.zeros((1, 10))
            scores[0, 7] = 0.9
            return scores

    monkeypatch.setattr(model, "model", FakeModel())
    assert predict_result(np.zeros((1, 224, 224, 3))) == 7


@pytest.mark.parametrize("digit", [0, 3, 5, 9])
def test_real_model_predicts_sample_digits(digit):
    """End-to-end check with the real model on one bundled sample per digit."""
    folder = f"test_images/{digit}"
    sample = sorted(os.listdir(folder))[0]
    prediction = predict_result(preprocess_img(f"{folder}/{sample}"))
    assert isinstance(prediction, np.integer)
    assert 0 <= prediction <= 9
    assert prediction == digit
