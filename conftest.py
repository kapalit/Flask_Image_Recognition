"""Shared pytest fixtures for the Flask Image Recognition tests."""

import pytest
from app import app  # This imports the Flask app for testing

@pytest.fixture
def client():
    """Yield a Flask test client for the app."""
    with app.test_client() as client:
        yield client
