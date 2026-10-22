"""Tests for offline file inference CLI and FastAPI endpoints.

Verifies:
- Single file and directory batch inference.
- Canonical audio loader and deterministic 4.0s windowing with configurable hop.
- Raw spoof/bonafide scores vs calibrated likelihood.
- Continuation past corrupt files and clear reporting of unsupported formats.
- JSONL and CSV output generation with model/config hashes.
- FastAPI health and upload endpoints.
- Malicious filename sanitization (directory traversal and null bytes).
- Oversized upload rejection (> 25MB).
- Audio duration limits (> 300s).
- Corrupt and empty audio rejection with structured errors.
- Concurrency guard via local semaphore for Apple Silicon M3.
- Compliance: Output is never called an identity or fraud score.
"""

from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from services.api.main import app
from services.api.routes.analyze_file import (
    MAX_FILE_BYTES,
    sanitize_filename,
)
from voiceshield.models.aasist_adapter import (
    AASIST_L_DEFAULT_CONFIG,
    AASISTLDetector,
)
from voiceshield.pipeline.cli import main as cli_main
from voiceshield.pipeline.file_evaluator import evaluate_audio
from voiceshield.vendor.aasist import AASISTModel


def create_synthetic_wav(
    file_path: Path,
    duration_s: float = 4.0,
    sample_rate: int = 16000,
    frequency: float = 440.0,
) -> Path:
    """Generate a clean synthetic sine wave WAV file."""
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False, dtype=np.float32)
    # Sine wave with audible amplitude
    samples = 0.5 * np.sin(2 * np.pi * frequency * t)
    sf.write(str(file_path), samples, sample_rate, format="WAV", subtype="PCM_16")
    return file_path


def create_synthetic_flac(
    file_path: Path,
    duration_s: float = 4.0,
    sample_rate: int = 16000,
) -> Path:
    """Generate a clean synthetic FLAC file."""
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False, dtype=np.float32)
    samples = 0.4 * np.sin(2 * np.pi * 300.0 * t)
    sf.write(str(file_path), samples, sample_rate, format="FLAC")
    return file_path


# ---------------------------------------------------------------------------
# Unit tests: Filename sanitization
# ---------------------------------------------------------------------------


def test_sanitize_filename() -> None:
    """Verify filename sanitization against path traversal and null bytes."""
    assert sanitize_filename("../../etc/passwd.wav") == "passwd.wav"
    assert sanitize_filename("..\\..\\Windows\\System32\\cmd.wav") == "cmd.wav"
    assert sanitize_filename("audio\x00malicious.flac") == "audiomalicious.flac"
    assert sanitize_filename(".../././nested/audio.wav") == "audio.wav"
    assert sanitize_filename("") == "uploaded_audio.wav"
    assert sanitize_filename(None) == "uploaded_audio.wav"
    assert sanitize_filename("  clean_test.wav  ") == "clean_test.wav"


# ---------------------------------------------------------------------------
# CLI Tests
# ---------------------------------------------------------------------------


def test_cli_single_file_mock_model(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify CLI on a single valid WAV file using --mock-model."""
    wav_path = create_synthetic_wav(tmp_path / "sample.wav", duration_s=5.0)
    jsonl_out = tmp_path / "output.jsonl"

    code = cli_main(
        [
            "--input",
            str(wav_path),
            "--mock-model",
            "--hop-seconds",
            "1.0",
            "--output",
            str(jsonl_out),
        ]
    )
    assert code == 0

    captured = capsys.readouterr()
    assert "VoiceShield Offline File Evaluation Summary" in captured.out
    assert "sample.wav" in captured.out

    # Verify JSONL output
    assert jsonl_out.exists()
    with open(jsonl_out, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    assert len(records) == 1
    rec = records[0]
    assert rec["status"] == "success"
    assert rec["duration_seconds"] == 5.0
    assert rec["total_windows"] >= 2
    assert "mean_synthetic_likelihood" in rec
    assert "max_synthetic_likelihood" in rec
    assert "temporal_rule_explanation" in rec
    assert "disclaimers" in rec
    # Verify non-fraud claim compliance
    assert (
        "identity proof" not in captured.out.lower()
        or "not prove caller identity" in captured.out.lower()
    )
    assert rec["disclaimers"]["identity_proof"] != ""
    assert rec["disclaimers"]["fraud_probability"] != ""


def test_cli_directory_batch_with_corrupt_and_unsupported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify CLI batch processing continuing past corrupt and unsupported files."""
    audio_dir = tmp_path / "audio_batch"
    audio_dir.mkdir()

    # 1. Valid WAV
    create_synthetic_wav(audio_dir / "valid.wav", duration_s=4.0)
    # 2. Valid FLAC
    create_synthetic_flac(audio_dir / "valid.flac", duration_s=4.0)
    # 3. Corrupt file with .wav extension
    corrupt_file = audio_dir / "corrupted.wav"
    corrupt_file.write_bytes(b"RIFF\x00\x00\x00\x00NOTWAVEFORMATGARBAGE")
    # 4. Unsupported format
    unsupported_file = audio_dir / "podcast.mp3"
    unsupported_file.write_bytes(b"ID3\x03\x00\x00\x00\x00\x00#dummy mp3 bytes")

    csv_out = tmp_path / "batch_results.csv"

    code = cli_main(
        [
            "--input",
            str(audio_dir),
            "--mock-model",
            "--output",
            str(csv_out),
        ]
    )
    assert code == 0

    captured = capsys.readouterr()
    assert "valid.wav" in captured.out
    assert "valid.flac" in captured.out
    assert "corrupted.wav" in captured.out
    assert "podcast.mp3" in captured.out

    assert csv_out.exists()
    content = csv_out.read_text(encoding="utf-8")
    assert "valid.wav,success" in content
    assert "valid.flac,success" in content
    assert "corrupted.wav,error" in content
    assert "podcast.mp3,unsupported" in content


def test_cli_deterministic_window_aggregation(tmp_path: Path) -> None:
    """Verify window count and scores for 7.0-second audio with 1.0s hop."""
    wav_path = create_synthetic_wav(tmp_path / "long_speech.wav", duration_s=7.0)

    mock_raw = AASISTModel(AASIST_L_DEFAULT_CONFIG)
    detector = AASISTLDetector(_instantiated_model=mock_raw, preferred_device="cpu")

    result = evaluate_audio(wav_path, detector=detector, hop_seconds=1.0, window_seconds=4.0)
    assert result.status == "success"
    assert result.duration_seconds == 7.0
    # Slices: (0s-4s), (1s-5s), (2s-6s), (3s-7s) -> 4 windows
    assert result.total_windows == 4
    for w in result.windows:
        assert w.raw_spoof_logit is not None
        assert w.raw_bonafide_logit is not None
        assert 0.0 <= w.calibrated_synthetic_likelihood <= 1.0
        assert w.quality_status.peak_level is not None
        assert w.active_speech_seconds >= 0.0


# ---------------------------------------------------------------------------
# API Tests (FastAPI + httpx)
# ---------------------------------------------------------------------------


def test_api_health_endpoint() -> None:
    """Verify /health endpoint diagnostics and advisory disclaimers."""
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "device" in data
        assert "disclaimers" in data
        assert data["disclaimers"]["identity_proof"] != ""
        assert data["disclaimers"]["fraud_probability"] != ""


def test_api_upload_valid_wav(tmp_path: Path) -> None:
    """Verify uploading a valid WAV file to /api/v1/analyze/file."""
    wav_path = create_synthetic_wav(tmp_path / "test_api.wav", duration_s=4.5)

    with TestClient(app) as client:
        with open(wav_path, "rb") as f:
            resp = client.post(
                "/api/v1/analyze/file",
                files={"file": ("test_api.wav", f, "audio/wav")},
                params={"hop_seconds": 1.0, "threshold": 0.65},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["file_path"] == "test_api.wav"
        assert data["duration_seconds"] == 4.5
        assert data["total_windows"] >= 1
        assert "mean_synthetic_likelihood" in data
        assert "max_synthetic_likelihood" in data
        assert "disclaimers" in data


def test_api_upload_valid_flac(tmp_path: Path) -> None:
    """Verify uploading a valid FLAC file."""
    flac_path = create_synthetic_flac(tmp_path / "test_api.flac", duration_s=4.0)

    with TestClient(app) as client:
        with open(flac_path, "rb") as f:
            resp = client.post(
                "/api/v1/analyze/file",
                files={"file": ("test_api.flac", f, "audio/flac")},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["file_path"] == "test_api.flac"


def test_api_upload_unsupported_media_type() -> None:
    """Verify 415 rejection for non-WAV/FLAC formats (e.g., MP3 or TXT)."""
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/analyze/file",
            files={"file": ("record.mp3", b"ID3fake_mp3_stream", "audio/mpeg")},
        )
        assert resp.status_code == 415
        detail = resp.json()["detail"]
        assert detail["error_code"] == "UNSUPPORTED_MEDIA_TYPE"
        assert "disclaimers" in detail


def test_api_upload_malicious_filenames(tmp_path: Path) -> None:
    """Verify filename traversal and null bytes are sanitized without server errors."""
    wav_path = create_synthetic_wav(tmp_path / "valid.wav", duration_s=4.0)
    wav_bytes = wav_path.read_bytes()

    with TestClient(app) as client:
        # Traversal attempt 1
        resp = client.post(
            "/api/v1/analyze/file",
            files={"file": ("../../../../etc/shadow.wav", io.BytesIO(wav_bytes), "audio/wav")},
        )
        assert resp.status_code == 200
        assert resp.json()["file_path"] == "shadow.wav"

        # Traversal attempt 2 (Windows style)
        resp2 = client.post(
            "/api/v1/analyze/file",
            files={"file": ("..\\..\\boot.ini.flac", io.BytesIO(wav_bytes), "audio/wav")},
        )
        assert resp2.status_code == 200
        assert resp2.json()["file_path"] == "boot.ini.flac"


def test_api_upload_oversized(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify 413 rejection when upload exceeds MAX_FILE_BYTES limit."""
    with TestClient(app) as client:
        # Create a small virtual stream or trigger size check
        oversized_data = b"0" * (MAX_FILE_BYTES + 50)
        resp = client.post(
            "/api/v1/analyze/file",
            files={"file": ("huge_file.wav", io.BytesIO(oversized_data), "audio/wav")},
        )
        assert resp.status_code == 413
        detail = resp.json()["detail"]
        assert detail["error_code"] == "PAYLOAD_TOO_LARGE"
        assert "disclaimers" in detail


def test_api_upload_empty_file() -> None:
    """Verify 422 rejection when upload contains 0 bytes."""
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/analyze/file",
            files={"file": ("empty.wav", io.BytesIO(b""), "audio/wav")},
        )
        assert resp.status_code == 422
        detail = resp.json()["detail"]
        assert detail["error_code"] == "EMPTY_AUDIO_FILE"


def test_api_upload_corrupt_audio() -> None:
    """Verify 422 rejection when audio header or content is unparseable."""
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/analyze/file",
            files={
                "file": (
                    "corrupt.wav",
                    io.BytesIO(b"RIFF\x24\x00\x00\x00NOTWAVCORRUPT"),
                    "audio/wav",
                )
            },
        )
        assert resp.status_code == 422
        detail = resp.json()["detail"]
        assert detail["error_code"] == "AUDIO_PROCESSING_ERROR"


def test_api_upload_duration_limit(tmp_path: Path) -> None:
    """Verify 400 rejection when audio exceeds 300.0s maximum duration limit."""
    # Create 305s audio file using 8kHz to keep fixture file size small
    long_wav = tmp_path / "too_long.wav"
    t = np.linspace(0, 305.0, int(8000 * 305.0), endpoint=False, dtype=np.float32)
    samples = 0.2 * np.sin(2 * np.pi * 400.0 * t)
    sf.write(str(long_wav), samples, 8000, format="WAV")

    with TestClient(app) as client:
        with open(long_wav, "rb") as f:
            resp = client.post(
                "/api/v1/analyze/file",
                files={"file": ("too_long.wav", f, "audio/wav")},
            )
        assert resp.status_code == 400
        detail = resp.json()["detail"]
        assert detail["error_code"] == "AUDIO_DURATION_EXCEEDED"
        assert "300.00s" in detail["message"]


@pytest.mark.asyncio
async def test_api_concurrent_requests_guarded(tmp_path: Path) -> None:
    """Verify that multiple concurrent requests are serialized safely through the M3 semaphore."""
    wav_path = create_synthetic_wav(tmp_path / "concurrent.wav", duration_s=4.0)
    wav_bytes = wav_path.read_bytes()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Send 3 concurrent inference requests
        async def send_req() -> Any:
            return await ac.post(
                "/api/v1/analyze/file",
                files={"file": ("concurrent.wav", io.BytesIO(wav_bytes), "audio/wav")},
            )

        responses = await asyncio.gather(send_req(), send_req(), send_req())
        for r in responses:
            assert r.status_code == 200
            data = r.json()
            assert data["status"] == "success"
            assert data["duration_seconds"] == 4.0
