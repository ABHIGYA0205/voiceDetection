"""Pytest configuration and session fixtures for VoiceShield tests."""

from __future__ import annotations

import os

# Default to mock model for unit and integration test suite runs
os.environ.setdefault("VOICESHIELD_MOCK_MODEL", "true")
