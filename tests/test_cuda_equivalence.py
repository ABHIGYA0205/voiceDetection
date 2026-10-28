"""CUDA device control-flow and live hardware numerical equivalence tests.

Tests device routing, fail-closed dispatch, and provenance capture under mocked conditions.
Physical CUDA-versus-CPU numerical equivalence requires real CUDA hardware.
Predeclared acceptance tolerance is atol=1e-4, rtol=1e-4; live equivalence remains
UNVERIFIED until the physical CUDA test passes on real hardware.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import torch

from scripts.reproduce_asvspoof_eval import compute_file_sha256
from voiceshield.core.device import resolve_device_status
from voiceshield.models import (
    AASISTLDetector,
    get_official_registry_entry,
    pad_for_aasist,
)
from voiceshield.reproducibility import set_deterministic_seed, set_seed

# Predeclared numerical acceptance tolerance for live physical CUDA hardware testing.
# Live hardware equivalence remains UNVERIFIED until physical CUDA tests pass.
# Exact bitwise equality across distinct processor architectures is not required due to
# varying floating-point reduction trees, cuDNN kernel implementations, and FMA instructions.
EQUIVALENCE_ATOL = 1e-4
EQUIVALENCE_RTOL = 1e-4


def generate_deterministic_synthetic_audio(
    seed: int = 42, samples: int = 64600
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Generate deterministic synthetic audio fixture for device testing."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, samples / 16000, samples, endpoint=False, dtype=np.float32)
    freq1 = 220.0 + (seed % 10) * 50.0
    freq2 = 440.0 + (seed % 7) * 70.0
    signal = 0.5 * np.sin(2 * np.pi * freq1 * t) + 0.25 * np.sin(2 * np.pi * freq2 * t)
    noise = rng.normal(0, 0.05, size=samples).astype(np.float32)
    combined = signal + noise
    peak = float(np.max(np.abs(combined)))
    if peak > 0:
        combined = combined / peak
    return np.asarray(combined, dtype=np.float32)


def test_deterministic_seeding_produces_identical_fixtures() -> None:
    """Verify that canonical deterministic seeding produces identical synthetic audio."""
    set_deterministic_seed(42)
    audio1 = generate_deterministic_synthetic_audio(seed=42)
    set_deterministic_seed(42)
    audio2 = generate_deterministic_synthetic_audio(seed=42)
    np.testing.assert_array_equal(audio1, audio2)


def test_pad_for_aasist_deterministic_framing() -> None:
    """Verify that cyclic pad/truncate produces exact deterministic framing."""
    set_seed(42)
    short_audio = generate_deterministic_synthetic_audio(seed=42, samples=32000)
    padded = pad_for_aasist(short_audio, max_len=64600)
    assert padded.shape == (64600,)

    long_audio = generate_deterministic_synthetic_audio(seed=42, samples=80000)
    truncated = pad_for_aasist(long_audio, max_len=64600)
    assert truncated.shape == (64600,)


def test_cuda_dispatch_control_flow_mocked() -> None:
    """Verify control flow for CUDA device selection and fail-closed dispatch.

    Mock tests verify control flow and dispatch logic only; they do not simulate
    or claim numerical equivalence across hardware architectures.
    """
    # 1. When CUDA is available, resolve_device_status selects cuda without fallback
    with (
        patch.object(torch.cuda, "is_available", return_value=True),
        patch.object(torch.cuda, "get_device_name", return_value="NVIDIA A100-SXM4-40GB"),
    ):
        status = resolve_device_status(preferred="cuda", fallback_to_cpu=False)
        assert status.device_name == "cuda"
        assert status.is_cuda
        assert status.cuda_available
        assert status.cuda_device_name == "NVIDIA A100-SXM4-40GB"

    # 2. When CUDA is unavailable, explicit CUDA request fails closed with RuntimeError
    with patch.object(torch.cuda, "is_available", return_value=False):
        with pytest.raises(
            RuntimeError,
            match="CUDA device was requested, but CUDA is not available on this system.",
        ):
            resolve_device_status(preferred="cuda", fallback_to_cpu=False)

        # Even with fallback_to_cpu=True, explicit CUDA request must NOT fall back to CPU
        with pytest.raises(
            RuntimeError,
            match="CUDA device was requested, but CUDA is not available on this system.",
        ):
            resolve_device_status(preferred="cuda", fallback_to_cpu=True)


def test_cuda_detector_initialization_control_flow() -> None:
    """Verify AASISTLDetector device dispatch control flow under mocked CUDA."""
    with (
        patch.object(torch.cuda, "is_available", return_value=True),
        patch.object(torch.cuda, "get_device_name", return_value="Tesla T4"),
        patch.object(torch.nn.Module, "to", return_value=None),
    ):
        detector = AASISTLDetector(
            model_id="mock/aasist-l",
            preferred_device="cuda",
            fallback_to_cpu=False,
        )
        assert detector.device.type == "cuda"


def load_equivalence_detectors(
    official_ckpt_env: str | None,
    cuda_device: str = "cuda",
) -> tuple[AASISTLDetector, AASISTLDetector]:
    """Load CPU and CUDA detectors for numerical equivalence validation.

    If official_ckpt_env is provided, strictly validates that the path exists, is readable,
    matches official AASIST-L SHA-256, and instantiates cleanly. Fails immediately without
    falling back to mock weights if the variable was supplied.
    """
    if official_ckpt_env is not None:
        ckpt_path = Path(official_ckpt_env)
        if not ckpt_path.is_file():
            pytest.fail(
                f"VOICESHIELD_OFFICIAL_CHECKPOINT was set to {official_ckpt_env!r}, "
                "but the file is missing or not a regular file."
            )
        try:
            with open(ckpt_path, "rb") as f:
                f.read(1024)
        except Exception as read_exc:
            pytest.fail(f"VOICESHIELD_OFFICIAL_CHECKPOINT at {ckpt_path} is unreadable: {read_exc}")

        reg = get_official_registry_entry("clovaai/aasist-l")
        try:
            ckpt_sha = compute_file_sha256(ckpt_path)
        except Exception as hash_exc:
            pytest.fail(f"Failed computing SHA-256 for VOICESHIELD_OFFICIAL_CHECKPOINT: {hash_exc}")

        if ckpt_sha != reg.checkpoint_sha256:
            pytest.fail(
                f"VOICESHIELD_OFFICIAL_CHECKPOINT SHA-256 integrity verification failed: "
                f"expected {reg.checkpoint_sha256}, got {ckpt_sha}"
            )

        try:
            cpu_detector = AASISTLDetector(checkpoint_path=ckpt_path, preferred_device="cpu")
            cuda_detector = AASISTLDetector(
                checkpoint_path=ckpt_path,
                preferred_device=cuda_device,
                fallback_to_cpu=False,
            )
        except Exception as load_exc:
            pytest.fail(
                f"VOICESHIELD_OFFICIAL_CHECKPOINT failed initialization or is incompatible: {load_exc}"
            )
    else:
        cpu_detector = AASISTLDetector(model_id="mock/aasist-l", preferred_device="cpu")
        cuda_detector = AASISTLDetector(
            model_id="mock/aasist-l",
            preferred_device=cuda_device,
            fallback_to_cpu=False,
        )
        cuda_detector.model.load_state_dict(cpu_detector.model.state_dict())

    return cpu_detector, cuda_detector


def test_live_cuda_numerical_equivalence_when_hardware_present() -> None:
    """Verify CPU vs CUDA numerical equivalence on physical CUDA hardware within predeclared tolerance.

    If physical CUDA hardware is not available on this runner, this test is skipped
    and live hardware numerical equivalence remains explicitly unverified.
    """
    if not torch.cuda.is_available():
        pytest.skip(
            "Physical CUDA hardware not available; live CUDA numerical equivalence remains "
            "explicitly unverified on this host."
        )

    set_deterministic_seed(42, fail_closed=True)

    official_ckpt_env = os.environ.get("VOICESHIELD_OFFICIAL_CHECKPOINT")
    cpu_detector, cuda_detector = load_equivalence_detectors(official_ckpt_env, cuda_device="cuda")

    # Test multiple deterministic fixtures across raw logits, synthetic_score, and decision consistency
    fixture_seeds = [42, 1337, 2026]
    for s in fixture_seeds:
        fixture = generate_deterministic_synthetic_audio(seed=s)

        out_cpu = cpu_detector.predict(fixture)
        out_cuda = cuda_detector.predict(fixture)

        # 1. Compare raw logits using predeclared atol and rtol
        cpu_logits = np.asarray(out_cpu.raw_logits, dtype=np.float32)
        cuda_logits = np.asarray(out_cuda.raw_logits, dtype=np.float32)
        np.testing.assert_allclose(
            cpu_logits,
            cuda_logits,
            atol=EQUIVALENCE_ATOL,
            rtol=EQUIVALENCE_RTOL,
            err_msg=(
                f"Fixture seed {s}: Live CUDA raw logits deviated from CPU "
                f"beyond predeclared tolerance (atol={EQUIVALENCE_ATOL}, rtol={EQUIVALENCE_RTOL})"
            ),
        )

        # 2. Compare synthetic_score with the same tolerance
        assert np.isclose(
            out_cpu.synthetic_score,
            out_cuda.synthetic_score,
            atol=EQUIVALENCE_ATOL,
            rtol=EQUIVALENCE_RTOL,
        ), (
            f"Fixture seed {s}: Live CUDA synthetic_score ({out_cuda.synthetic_score}) deviated from CPU "
            f"({out_cpu.synthetic_score}) beyond predeclared tolerance (atol={EQUIVALENCE_ATOL}, rtol={EQUIVALENCE_RTOL})"
        )

        # 3. Compare is_synthetic_suspected
        assert out_cpu.is_synthetic_suspected == out_cuda.is_synthetic_suspected, (
            f"Fixture seed {s}: Decision mismatch between CPU ({out_cpu.is_synthetic_suspected}) "
            f"and CUDA ({out_cuda.is_synthetic_suspected})"
        )


def test_official_checkpoint_missing_fails() -> None:
    """Verify load_equivalence_detectors fails immediately when official checkpoint file is missing."""
    with pytest.raises(pytest.fail.Exception, match="missing or not a regular file"):
        load_equivalence_detectors("/nonexistent/AASIST-L.pth")


def test_official_checkpoint_unreadable_fails(tmp_path: Path) -> None:
    """Verify load_equivalence_detectors fails immediately when official checkpoint file is unreadable."""
    dummy_file = tmp_path / "unreadable.pth"
    dummy_file.write_bytes(b"dummy")
    with patch("builtins.open", side_effect=PermissionError("Permission denied")):
        with pytest.raises(pytest.fail.Exception, match="unreadable"):
            load_equivalence_detectors(str(dummy_file))


def test_official_checkpoint_hash_invalid_fails(tmp_path: Path) -> None:
    """Verify load_equivalence_detectors fails immediately when SHA-256 does not match registry."""
    dummy_file = tmp_path / "corrupt.pth"
    dummy_file.write_bytes(b"corrupted checkpoint content")
    with pytest.raises(pytest.fail.Exception, match="integrity verification failed"):
        load_equivalence_detectors(str(dummy_file))


def test_official_checkpoint_incompatible_fails(tmp_path: Path) -> None:
    """Verify load_equivalence_detectors fails immediately when checkpoint bytes are incompatible with AASISTLDetector."""
    dummy_file = tmp_path / "incompatible.pth"
    dummy_file.write_bytes(b"invalid torch weights")
    reg = get_official_registry_entry("clovaai/aasist-l")
    with patch(
        "tests.test_cuda_equivalence.compute_file_sha256", return_value=reg.checkpoint_sha256
    ):
        with pytest.raises(pytest.fail.Exception, match="failed initialization or is incompatible"):
            load_equivalence_detectors(str(dummy_file))


def test_official_checkpoint_none_uses_mock() -> None:
    """Verify load_equivalence_detectors safely loads mock weights when official checkpoint is not specified."""
    cpu_det, cuda_det = load_equivalence_detectors(None, cuda_device="cpu")
    assert cpu_det.provenance.model_id == "mock/aasist-l"
    assert cuda_det.provenance.model_id == "mock/aasist-l"
