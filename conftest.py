"""Shared pytest fixtures for the Flask Image Recognition tests."""

import pytest
from app import app  # This imports the Flask app for testing

@pytest.fixture
def client():
    """Yield a Flask test client for the app.

    CSRF checks are switched off for these functional tests because the test
    client posts directly without first loading the form; test_csrf.py turns
    them back on to verify the protection itself.
    """
    app.config["WTF_CSRF_ENABLED"] = False
    with app.test_client() as client:
        yield client
    app.config["WTF_CSRF_ENABLED"] = True
