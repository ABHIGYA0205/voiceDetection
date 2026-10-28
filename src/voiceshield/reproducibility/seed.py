"""Deterministic random seeding utilities for Python, NumPy, PyTorch, and CUDA.

Provides the single canonical deterministic configuration across VoiceShield,
supporting strict CUDA determinism, verified runtime settings, and provenance capture.
"""

from __future__ import annotations

import logging
import os
import random
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


def get_determinism_provenance(seed: int = 42) -> dict[str, Any]:
    """Capture verified deterministic settings and CUDA/cuDNN/TF32 provenance."""
    cuda_avail = bool(torch.cuda.is_available())
    cuda_device_name = torch.cuda.get_device_name(0) if cuda_avail else None
    cuda_capability: tuple[int, int] | None = None
    if cuda_avail:
        try:
            cuda_capability = torch.cuda.get_device_capability(0)
        except Exception:
            cuda_capability = None

    deterministic_algorithms = False
    if hasattr(torch, "are_deterministic_algorithms_enabled"):
        deterministic_algorithms = bool(torch.are_deterministic_algorithms_enabled())

    cudnn_deterministic = bool(getattr(torch.backends.cudnn, "deterministic", False))
    cudnn_benchmark = bool(getattr(torch.backends.cudnn, "benchmark", False))

    cuda_allow_tf32: bool | None = None
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        cuda_allow_tf32 = bool(getattr(torch.backends.cuda.matmul, "allow_tf32", True))

    cudnn_allow_tf32: bool | None = None
    if hasattr(torch.backends, "cudnn") and hasattr(torch.backends.cudnn, "allow_tf32"):
        cudnn_allow_tf32 = bool(getattr(torch.backends.cudnn, "allow_tf32", True))

    cudnn_ver: str | None = None
    if hasattr(torch.backends, "cudnn") and torch.backends.cudnn.is_available():  # type: ignore[no-untyped-call]
        try:
            cudnn_ver = str(torch.backends.cudnn.version())  # type: ignore[no-untyped-call]
        except Exception:
            cudnn_ver = None

    return {
        "seed": seed,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms": deterministic_algorithms,
        "cudnn_deterministic": cudnn_deterministic,
        "cudnn_benchmark": cudnn_benchmark,
        "cuda_allow_tf32": cuda_allow_tf32,
        "cudnn_allow_tf32": cudnn_allow_tf32,
        "cuda_available": cuda_avail,
        "cuda_device_name": cuda_device_name,
        "cuda_capability": cuda_capability,
        "torch_version": torch.__version__,
        "cuda_version": getattr(torch.version, "cuda", None),
        "cudnn_version": cudnn_ver,
    }


def set_seed(
    seed: int = 42,
    deterministic: bool = True,
    *,
    warn_only: bool = True,
    fail_closed: bool = False,
) -> dict[str, Any]:
    """Set random seeds across Python, NumPy, and PyTorch with strict CUDA determinism.

    Preserves backward compatibility with existing callers while capturing and
    returning comprehensive determinism provenance.

    Args:
        seed: Integer seed value.
        deterministic: If True, configures deterministic algorithms and disables TF32.
        warn_only: If True and fail_closed is False, warn instead of erroring when
                   an operator lacks a deterministic kernel.
        fail_closed: If True, enforce strict deterministic algorithms without warn_only
                     and raise RuntimeError if any deterministic requirement fails.

    Returns:
        Dictionary containing verified determinism settings and provenance.
    """
    logger.info(
        "Setting global random seed to %d (deterministic=%s, fail_closed=%s)",
        seed,
        deterministic,
        fail_closed,
    )

    # 0. Validate seed range (NumPy accepted range: 0 <= seed <= 2**32 - 1)
    if not isinstance(seed, int) or isinstance(seed, bool) or not (0 <= seed <= 2**32 - 1):
        raise ValueError(
            f"Seed must be an integer between 0 and 2**32 - 1 ({2**32 - 1}), got {seed!r}"
        )

    # 1. Enforce cuBLAS workspace config before PyTorch invokes cuBLAS operations
    existing_cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if existing_cublas is not None:
        if existing_cublas not in {":4096:8", ":16:8"}:
            raise ValueError(
                f"Incompatible pre-existing CUBLAS_WORKSPACE_CONFIG={existing_cublas!r}; "
                "must be exactly ':4096:8' or ':16:8'."
            )
    else:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # 2. Python standard library and hashing
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 3. NumPy
    np.random.seed(seed)

    # 4. PyTorch CPU and CUDA
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 5. Apple Silicon MPS if available
    if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
        try:
            torch.mps.manual_seed(seed)
        except Exception as exc:
            logger.debug("MPS manual_seed failed: %s", exc)

    # 6. Strict CUDA / cuDNN and TF32 determinism settings
    if deterministic:
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            if hasattr(torch.backends.cudnn, "allow_tf32"):
                torch.backends.cudnn.allow_tf32 = False

        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = False

        if fail_closed:
            try:
                torch.use_deterministic_algorithms(True, warn_only=False)
            except Exception as exc:
                raise RuntimeError(
                    f"Strict fail-closed determinism requested, but deterministic algorithms "
                    f"could not be enforced without warnings: {exc}"
                ) from exc
        else:
            try:
                torch.use_deterministic_algorithms(True, warn_only=warn_only)
            except Exception as exc:
                logger.warning("Deterministic algorithms could not be enabled: %s", exc)
    else:
        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(False)
            except Exception:
                pass

    provenance = get_determinism_provenance(seed=seed)
    if fail_closed:
        if not provenance.get("deterministic_algorithms"):
            raise RuntimeError(
                "Strict determinism failed: PyTorch deterministic algorithms are not active."
            )
        if provenance.get("cudnn_benchmark"):
            raise RuntimeError("Strict determinism failed: cuDNN benchmark is enabled.")
        if (
            hasattr(torch.backends, "cudnn")
            and torch.backends.cudnn.is_available()  # type: ignore[no-untyped-call]
            and not provenance.get("cudnn_deterministic")
        ):
            raise RuntimeError("Strict determinism failed: cuDNN deterministic mode is not active.")
        if provenance.get("cuda_allow_tf32") is True:
            raise RuntimeError("Strict determinism failed: CUDA matmul TF32 is enabled.")
        if provenance.get("cudnn_allow_tf32") is True:
            raise RuntimeError("Strict determinism failed: cuDNN TF32 is enabled.")
        if provenance.get("cublas_workspace_config") not in {":4096:8", ":16:8"}:
            raise RuntimeError(
                f"Strict determinism failed: CUBLAS_WORKSPACE_CONFIG={provenance.get('cublas_workspace_config')!r}; "
                "must be ':4096:8' or ':16:8'."
            )

    return provenance


def set_deterministic_seed(seed: int = 42, *, fail_closed: bool = False) -> dict[str, Any]:
    """Canonical convenience function to establish deterministic execution."""
    return set_seed(
        seed=seed, deterministic=True, warn_only=not fail_closed, fail_closed=fail_closed
    )
