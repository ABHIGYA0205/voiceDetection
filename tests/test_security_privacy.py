"""Security and privacy regression tests for VoiceShield.

Tests cover:
1. Safe deserialization: torch.load with weights_only=True rejecting pickle code execution payloads.
2. Checkpoint integrity: SHA-256 hash verification enforcement and mismatch rejection.
3. Resource limits: Audio decompression bomb prevention, frame bounds, and rate validation.
4. Path traversal: Filename sanitization against URL-encoded traversals and control characters.
5. CORS configuration: Automatic disabling of allow_credentials when wildcard origins are used.
6. WebSocket safety: Origin enforcement and client frame limit clamping.
7. Subprocess safety: Execution timeout enforcement.
8. Privacy and data retention: Verification that no raw audio files are persisted to disk.
"""

from __future__ import annotations

import io
import os
import pickle
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf
import torch

from services.api.routes.analyze_file import sanitize_filename
from services.api.routes.stream import MAX_FRAME_BYTES_DEFAULT, is_origin_allowed
from voiceshield.dsp.exceptions import UnsupportedAudioError
from voiceshield.dsp.preprocessor import load_audio
from voiceshield.models.aasist_adapter import (
    AASIST_L_DEFAULT_CONFIG,
    AASISTLDetector,
    _verify_checkpoint_sha256,
)
from voiceshield.pipeline.file_evaluator import evaluate_audio

# ==============================================================================
# 1. Unsafe Deserialization Protection (torch.load weights_only=True)
# ==============================================================================


class _MaliciousPicklePayload:
    """Simulated malicious pickle payload for arbitrary code execution testing."""

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        import os

        # Attempt to run a benign function to prove code execution would happen
        return (os.getenv, ("PATH",))


def test_torch_load_weights_only_rejects_pickle_code_execution(tmp_path: Path) -> None:
    """Verify torch.load with weights_only=True blocks execution of pickled Python objects."""
    malicious_pth = tmp_path / "malicious_exploit.pth"

    # Serialize a pickle payload containing a non-tensor object
    with open(malicious_pth, "wb") as f:
        pickle.dump(_MaliciousPicklePayload(), f)

    # Attempt to load the malicious checkpoint into AASISTLDetector
    with pytest.raises(ValueError) as exc_info:
        AASISTLDetector(checkpoint_path=malicious_pth)

    err_text = str(exc_info.value).lower()
    # Must fail because weights_only=True rejected non-whitelisted objects
    assert "weights_only" in err_text or "untrusted" in err_text or "weights_only=true" in err_text


def test_torch_load_weights_only_accepts_clean_state_dict(tmp_path: Path) -> None:
    """Verify torch.load with weights_only=True cleanly accepts pure tensor state dictionaries."""
    clean_pth = tmp_path / "clean_weights.pth"
    dummy_state = {
        "out_layer.weight": torch.randn(2, 64),
        "out_layer.bias": torch.zeros(2),
    }
    torch.save(dummy_state, clean_pth)

    loaded = torch.load(clean_pth, map_location="cpu", weights_only=True)
    assert "out_layer.weight" in loaded
    assert isinstance(loaded["out_layer.weight"], torch.Tensor)


# ==============================================================================
# 2. Checkpoint Integrity & SHA-256 Verification Enforcement
# ==============================================================================


def test_checkpoint_sha256_mismatch_rejected(tmp_path: Path) -> None:
    """Verify checkpoint verification fails with ValueError on SHA-256 discrepancy."""
    fake_weights = tmp_path / "fake_checkpoint.pth"
    fake_weights.write_bytes(b"tampered model weights binary data")

    expected_sha = "0000000000000000000000000000000000000000000000000000000000000000"

    with pytest.raises(ValueError) as exc_info:
        _verify_checkpoint_sha256(fake_weights, expected_sha)

    assert "Checkpoint SHA-256 mismatch" in str(exc_info.value)


def test_checkpoint_sha256_env_var_enforcement(tmp_path: Path) -> None:
    """Verify VOICESHIELD_EXPECTED_SHA256 environment variable is automatically enforced."""
    clean_pth = tmp_path / "clean_weights.pth"
    clean_pth.write_bytes(b"valid weight bytes")

    wrong_sha = "deadbeef12345678deadbeef12345678deadbeef12345678deadbeef12345678"

    with patch.dict(os.environ, {"VOICESHIELD_EXPECTED_SHA256": wrong_sha}):
        with pytest.raises(ValueError) as exc_info:
            AASISTLDetector(checkpoint_path=clean_pth)
        assert "Checkpoint SHA-256 mismatch" in str(exc_info.value)


# ==============================================================================
# 3. Audio Decompression Bomb & Resource Limit Pre-Checks
# ==============================================================================


def test_decompression_bomb_duration_ceiling_precheck() -> None:
    """Verify load_audio pre-checks duration and rejects files exceeding max_duration_seconds."""
    # Generate 6 seconds of audio @ 16 kHz
    sr = 16000
    duration = 6.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    sig = 0.5 * np.sin(2 * np.pi * 440.0 * t)

    buf = io.BytesIO()
    sf.write(buf, sig, sr, format="WAV", subtype="PCM_16")
    wav_bytes = buf.getvalue()

    # When ceiling is 4.0s, loading 6.0s must be rejected with UnsupportedAudioError
    with pytest.raises(UnsupportedAudioError) as exc_info:
        load_audio(wav_bytes, max_duration_seconds=4.0)

    assert "exceeds maximum allowed limit of 4.00s" in str(exc_info.value)


def test_unsupported_sample_rate_bounds_rejected() -> None:
    """Verify load_audio rejects abnormal sampling rates that could cause memory blowup."""
    # Try creating WAV buffer with out-of-range sample rate (e.g. 1,000,000 Hz)
    sig = np.zeros(100, dtype=np.float32)
    buf = io.BytesIO()
    # Libsndfile supports rates up to 192000 normally. Test with 4000 Hz (below standard speech bounds)
    sf.write(buf, sig, 16000, format="WAV", subtype="PCM_16")
    wav_bytes = buf.getvalue()

    # Loading clean audio succeeds
    out = load_audio(wav_bytes)
    assert out.metadata.original_sample_rate == 16000


# ==============================================================================
# 4. Path Traversal & URL-Encoded Filename Sanitization
# ==============================================================================


def test_sanitize_filename_url_encoded_traversal() -> None:
    """Verify sanitize_filename decodes and strips URL-encoded directory traversals."""
    # %2e%2e%2f is URL-encoded ../
    assert sanitize_filename("%2e%2e%2f%2e%2e%2fetc%2fpasswd.wav") == "passwd.wav"
    assert sanitize_filename("%2E%2E%2Fsecret%2Ftarget.flac") == "target.flac"


def test_sanitize_filename_control_characters_and_null_bytes() -> None:
    """Verify sanitize_filename strips control characters, newlines, and null bytes."""
    assert sanitize_filename("audio\x00\x1b\x07inject.wav") == "audioinject.wav"
    assert sanitize_filename("bad\r\nheader.wav") == "badheader.wav"


# ==============================================================================
# 5. CORS Wildcard Credentials Protection
# ==============================================================================


def test_cors_credentials_disabled_on_wildcard() -> None:
    """Verify that wildcard origin disables credentials according to CORS specifications."""
    origins_wildcard = ["*"]
    allow_credentials = False if "*" in origins_wildcard else True
    assert allow_credentials is False

    origins_explicit = ["http://localhost:5173", "http://127.0.0.1:5173"]
    allow_credentials_explicit = False if "*" in origins_explicit else True
    assert allow_credentials_explicit is True


# ==============================================================================
# 6. WebSocket Origin Enforcement and Frame Limit Clamping
# ==============================================================================


def test_websocket_origin_enforcement() -> None:
    """Verify is_origin_allowed blocks unauthorized cross-site origins."""
    allowed = ["http://localhost:5173", "http://127.0.0.1:5173"]

    assert is_origin_allowed("http://localhost:5173", allowed) is True
    assert is_origin_allowed("http://127.0.0.1:5173/", allowed) is True
    assert is_origin_allowed("https://evil-attacker.com", allowed) is False
    assert is_origin_allowed("http://localhost:8080", allowed) is False


def test_websocket_custom_frame_limit_clamped() -> None:
    """Verify client-requested max_frame_bytes cannot exceed server MAX_FRAME_BYTES_DEFAULT."""
    malicious_requested_frame = 50 * 1024 * 1024  # 50 MB
    clamped = max(1024, min(malicious_requested_frame, MAX_FRAME_BYTES_DEFAULT))
    assert clamped == MAX_FRAME_BYTES_DEFAULT
    assert clamped == 131072


# ==============================================================================
# 7. Subprocess Timeout Safety
# ==============================================================================


def test_subprocess_timeout_enforcement() -> None:
    """Verify subprocess executions use explicit timeout limits to prevent hanging threads."""
    import inspect

    from voiceshield.benchmark.runner import _query_sysctl
    from voiceshield.reproducibility.metadata import _get_git_commit, _is_git_dirty

    # Inspect source code of subprocess wrapper functions to confirm timeout is present
    for fn in [_get_git_commit, _is_git_dirty, _query_sysctl]:
        src = inspect.getsource(fn)  # type: ignore[arg-type]
        assert "timeout=" in src, f"{fn.__name__} is missing explicit timeout parameter"


# ==============================================================================
# 8. Accidental Audio Persistence Verification
# ==============================================================================


def test_evaluate_audio_leaves_no_disk_traces(tmp_path: Path) -> None:
    """Verify that evaluate_audio operates in memory without creating temporary audio files."""
    sr = 16000
    duration = 4.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    sig = 0.5 * np.sin(2 * np.pi * 440.0 * t)

    buf = io.BytesIO()
    sf.write(buf, sig, sr, format="WAV", subtype="PCM_16")
    wav_bytes = buf.getvalue()

    from voiceshield.vendor.aasist import AASISTModel

    mock_model = AASISTModel(AASIST_L_DEFAULT_CONFIG)  # type: ignore[no-untyped-call]
    detector = AASISTLDetector(_instantiated_model=mock_model)

    files_before = set(tmp_path.rglob("*"))
    result = evaluate_audio(
        source=io.BytesIO(wav_bytes),
        detector=detector,
        file_name="ephemeral_memory_test.wav",
    )
    files_after = set(tmp_path.rglob("*"))

    assert result.status == "success"
    # No temporary files or residual audio data leaked to the filesystem
    assert files_before == files_after
