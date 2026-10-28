"""Hardware device selection and fallback utilities for CUDA, Apple Silicon (MPS), and CPU."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeviceStatus:
    """Diagnostic status for the selected computation device."""

    device_name: str
    is_mps: bool
    mps_built: bool
    mps_available: bool
    explanation: str
    is_cuda: bool = False
    cuda_available: bool = False
    cuda_device_name: str | None = None
    requested_device: str | None = None


def resolve_device_status(
    preferred: str | None = None,
    fallback_to_cpu: bool = True,
) -> DeviceStatus:
    """Resolve and explain the optimal torch device.

    Args:
        preferred: Explicitly requested device ('cuda', 'mps', 'cpu', or None to read environment).
        fallback_to_cpu: Whether to safely fall back to CPU if MPS is unavailable.
            Note: Explicit CUDA requests never fall back to CPU.

    Returns:
        DeviceStatus with full diagnostics.
    """
    mps_built = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_built())
    mps_available = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    cuda_available = bool(torch.cuda.is_available())
    cuda_device_name: str | None = None
    if cuda_available:
        try:
            cuda_device_name = str(torch.cuda.get_device_name(0))
        except Exception:
            cuda_device_name = "cuda"

    raw_target = preferred or os.getenv("VOICESHIELD_DEVICE") or "mps"
    target = raw_target.lower().strip()

    if target == "cuda":
        if cuda_available:
            return DeviceStatus(
                device_name="cuda",
                is_mps=False,
                mps_built=mps_built,
                mps_available=mps_available,
                is_cuda=True,
                cuda_available=True,
                cuda_device_name=cuda_device_name,
                requested_device=target,
                explanation=f"CUDA device explicitly selected ({cuda_device_name or 'cuda'}).",
            )
        raise RuntimeError("CUDA device was requested, but CUDA is not available on this system.")

    if target == "mps":
        if mps_built and mps_available:
            return DeviceStatus(
                device_name="mps",
                is_mps=True,
                mps_built=mps_built,
                mps_available=mps_available,
                is_cuda=False,
                cuda_available=cuda_available,
                cuda_device_name=cuda_device_name,
                requested_device=target,
                explanation="Apple Silicon Metal Performance Shaders (MPS) is built, available, and active.",
            )
        if fallback_to_cpu:
            reason = (
                "MPS was requested, but "
                + ("MPS is not built in this PyTorch binary. " if not mps_built else "")
                + ("MPS is not available on this hardware/OS. " if not mps_available else "")
                + "Safely falling back to CPU."
            )
            logger.warning(reason)
            return DeviceStatus(
                device_name="cpu",
                is_mps=False,
                mps_built=mps_built,
                mps_available=mps_available,
                is_cuda=False,
                cuda_available=cuda_available,
                cuda_device_name=cuda_device_name,
                requested_device=target,
                explanation=reason,
            )
        raise RuntimeError("MPS device was requested, but MPS is not available on this system.")

    if target == "cpu":
        return DeviceStatus(
            device_name="cpu",
            is_mps=False,
            mps_built=mps_built,
            mps_available=mps_available,
            is_cuda=False,
            cuda_available=cuda_available,
            cuda_device_name=cuda_device_name,
            requested_device=target,
            explanation="CPU device explicitly selected.",
        )

    raise ValueError(
        f"Unsupported device '{target}'. Supported devices are 'cpu', 'mps', and 'cuda'."
    )


def get_device(
    preferred: str | None = None,
    fallback_to_cpu: bool = True,
) -> torch.device:
    """Return the selected torch.device instance.

    Args:
        preferred: Explicitly requested device ('cuda', 'mps', or 'cpu').
        fallback_to_cpu: Whether to safely fall back to CPU if MPS is unavailable.

    Returns:
        torch.device initialized to the resolved target.
    """
    status = resolve_device_status(preferred=preferred, fallback_to_cpu=fallback_to_cpu)
    return torch.device(status.device_name)
