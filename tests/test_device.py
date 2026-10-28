"""Unit tests for the device selection and CUDA/MPS/CPU fallback utility."""

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
    assert not status.is_cuda


def test_cuda_resolution_when_available() -> None:
    """Verify CUDA device resolution and provenance diagnostics when available."""
    with (
        patch.object(torch.cuda, "is_available", return_value=True),
        patch.object(torch.cuda, "get_device_name", return_value="NVIDIA A100-SXM4-40GB"),
    ):
        status = resolve_device_status(preferred="cuda")
        assert status.device_name == "cuda"
        assert status.is_cuda
        assert not status.is_mps
        assert status.cuda_available
        assert status.cuda_device_name == "NVIDIA A100-SXM4-40GB"
        assert status.requested_device == "cuda"
        assert "CUDA device explicitly selected" in status.explanation

        device = get_device(preferred="cuda")
        assert device.type == "cuda"


def test_cuda_explicit_raises_when_unavailable() -> None:
    """Verify explicit CUDA request fails closed with RuntimeError and never falls back to CPU."""
    with patch.object(torch.cuda, "is_available", return_value=False):
        # Even with fallback_to_cpu=True, CUDA must fail closed
        with pytest.raises(
            RuntimeError, match="CUDA device was requested, but CUDA is not available"
        ):
            resolve_device_status(preferred="cuda", fallback_to_cpu=True)

        with pytest.raises(
            RuntimeError, match="CUDA device was requested, but CUDA is not available"
        ):
            get_device(preferred="cuda", fallback_to_cpu=True)


def test_cuda_environment_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify VOICESHIELD_DEVICE=cuda triggers CUDA selection or fail-closed error."""
    monkeypatch.setenv("VOICESHIELD_DEVICE", "cuda")
    with (
        patch.object(torch.cuda, "is_available", return_value=True),
        patch.object(torch.cuda, "get_device_name", return_value="Tesla T4"),
    ):
        status = resolve_device_status()
        assert status.device_name == "cuda"
        assert status.is_cuda

    with patch.object(torch.cuda, "is_available", return_value=False):
        with pytest.raises(RuntimeError, match="CUDA device was requested"):
            resolve_device_status()


def test_unsupported_device_raises_value_error() -> None:
    """Verify unsupported device identifier raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported device 'tpu'"):
        resolve_device_status(preferred="tpu")
