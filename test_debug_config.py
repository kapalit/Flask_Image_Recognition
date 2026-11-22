"""Tests that the Flask debugger is off unless explicitly enabled (SonarQube python:S4507)."""

import pytest

from app import debug_enabled


def test_debug_disabled_by_default(monkeypatch):
    """With no FLASK_DEBUG variable the debugger must stay off."""
    monkeypatch.delenv("FLASK_DEBUG", raising=False)
    assert debug_enabled() is False


@pytest.mark.parametrize("value", ["0", "false", "", "yes", "debug"])
def test_debug_disabled_for_other_values(monkeypatch, value):
    """Anything other than an explicit "1"/"true" keeps the debugger off."""
    monkeypatch.setenv("FLASK_DEBUG", value)
    assert debug_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "TRUE", " True "])
def test_debug_enabled_when_explicitly_requested(monkeypatch, value):
    """Developers can still opt in locally with FLASK_DEBUG=1."""
    monkeypatch.setenv("FLASK_DEBUG", value)
    assert debug_enabled() is True
