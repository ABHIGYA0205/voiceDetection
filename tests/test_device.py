"""Unit tests for the device selection and MPS/CPU fallback utility."""

from unittest.mock import patch

import pytest
import torch

from voiceshield.core.device import get_device, resolve_device_status


def test_cpu_device_explicit() -> None:
    """Verify that explicitly requesting CPU always yields a CPU device."""
    status = resolve_device_status(preferred="cpu")
    assert status.device_name == "cpu"
    assert not status.is_mps
    assert "CPU device explicitly selected" in status.explanation

    device = get_device(preferred="cpu")
    assert device.type == "cpu"


def test_mps_resolution_when_available() -> None:
    """Verify MPS device resolution when torch indicates MPS is built and available."""
    with (
        patch.object(torch.backends.mps, "is_built", return_value=True),
        patch.object(torch.backends.mps, "is_available", return_value=True),
    ):
        status = resolve_device_status(preferred="mps")
        assert status.device_name == "mps"
        assert status.is_mps
        assert status.mps_built
        assert status.mps_available

        device = get_device(preferred="mps")
        assert device.type == "mps"


def test_mps_fallback_to_cpu_when_unavailable() -> None:
    """Verify graceful fallback to CPU when MPS is requested but not available."""
    with (
        patch.object(torch.backends.mps, "is_built", return_value=False),
        patch.object(torch.backends.mps, "is_available", return_value=False),
    ):
        status = resolve_device_status(preferred="mps", fallback_to_cpu=True)
        assert status.device_name == "cpu"
        assert not status.is_mps
        assert "falling back to CPU" in status.explanation

        device = get_device(preferred="mps", fallback_to_cpu=True)
        assert device.type == "cpu"


def test_mps_strict_raises_when_unavailable() -> None:
    """Verify that disabling fallback raises a RuntimeError if MPS is missing."""
    with (
        patch.object(torch.backends.mps, "is_built", return_value=False),
        patch.object(torch.backends.mps, "is_available", return_value=False),
    ):
        with pytest.raises(
            RuntimeError, match="MPS device was requested, but MPS is not available"
        ):
            resolve_device_status(preferred="mps", fallback_to_cpu=False)


def test_environment_variable_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that VOICESHIELD_DEVICE controls default device selection."""
    monkeypatch.setenv("VOICESHIELD_DEVICE", "cpu")
    status = resolve_device_status()
    assert status.device_name == "cpu"
    assert not status.is_mps
