"""Tests for CSRF protection on the upload form (SonarQube python:S4502)."""

import re
from io import BytesIO

import pytest
from PIL import Image

from app import app


@pytest.fixture
def csrf_client():
    """Yield a test client with CSRF protection enabled, as in production."""
    app.config["WTF_CSRF_ENABLED"] = True
    with app.test_client() as test_client:
        yield test_client


def _png_bytes():
    """Return a small valid PNG image as an in-memory file."""
    buf = BytesIO()
    Image.new("RGB", (32, 32), "white").save(buf, format="PNG")
    buf.seek(0)
    return buf


def test_index_form_contains_csrf_token(csrf_client):
    """The upload form must embed a hidden csrf_token field."""
    response = csrf_client.get("/")
    assert response.status_code == 200
    assert re.search(rb'name="csrf_token" type="hidden" value="[^"]+"', response.data)


def test_post_without_csrf_token_is_rejected(csrf_client):
    """A cross-site POST with no token must be refused with 400 Bad Request."""
    response = csrf_client.post(
        "/prediction",
        data={"file": (_png_bytes(), "digit.png")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert b"CSRF" in response.data


def test_post_with_valid_csrf_token_is_accepted(csrf_client):
    """A POST that carries the token issued with the form is processed normally."""
    page = csrf_client.get("/")
    token = re.search(rb'name="csrf_token" type="hidden" value="([^"]+)"', page.data).group(1)
    response = csrf_client.post(
        "/prediction",
        data={"csrf_token": token.decode(), "file": (_png_bytes(), "digit.png")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    assert b"File cannot be processed." not in response.data
