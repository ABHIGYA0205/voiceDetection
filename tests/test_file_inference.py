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
- Concurrency guard via process-wide single-worker inference executor for Apple Silicon M3.
- Compliance: Output is never called an identity or fraud score.
"""

from __future__ import annotations

import io
import json
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from services.api.main import app
from services.api.resource_admission import (
    MAX_UPLOAD_REQUEST_BYTES,
    AtomicAdmissionController,
)
from services.api.routes.analyze_file import (
    MAX_FILE_BYTES,
    sanitize_filename,
)
from voiceshield.models.aasist_adapter import (
    AASIST_L_DEFAULT_CONFIG,
    AASISTLDetector,
)
from voiceshield.models.base import BaseDetector, DetectorOutput
from voiceshield.models.provenance import ModelProvenance
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
        assert w.synthetic_score is not None
        assert 0.0 <= w.synthetic_score <= 1.0
        assert w.calibrated_synthetic_likelihood is None
        assert w.score_is_calibrated is False
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


def test_api_concurrent_requests_guarded(tmp_path: Path) -> None:
    """Verify that multiple concurrent requests are serialized safely through the single-worker executor."""
    wav_path = create_synthetic_wav(tmp_path / "concurrent.wav", duration_s=4.0)
    wav_bytes = wav_path.read_bytes()

    with TestClient(app) as client:
        # Allow up to 5 concurrent upload slots so all 3 pass admission and test worker serialization
        app.state.file_upload_admission = AtomicAdmissionController(limit=5, name="file_uploads")
        results: list[dict[str, Any]] = []
        lock = threading.Lock()

        def send_req() -> None:
            resp = client.post(
                "/api/v1/analyze/file",
                files={"file": ("concurrent.wav", io.BytesIO(wav_bytes), "audio/wav")},
            )
            assert resp.status_code == 200
            with lock:
                results.append(resp.json())

        threads = [threading.Thread(target=send_req) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 3
        for data in results:
            assert data["status"] == "success"
            assert data["duration_seconds"] == 4.0
        assert app.state.file_upload_admission.reserved_count == 0


class ControlledUploadDetector(BaseDetector):
    """Detector mock that blocks on call 1 to test physical execution concurrency."""

    provenance = ModelProvenance(
        model_id="mock/aasist-l",
        architecture="aasist_l",
        provenance_status="mock",
        checkpoint_verified=False,
        is_mock=True,
        parameter_count=85306,
    )

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.call_count = 0
        self.active_count = 0
        self.max_concurrency = 0
        self.call1_started = threading.Event()
        self.call1_release = threading.Event()

    def predict(self, audio: Any) -> DetectorOutput:
        with self.lock:
            self.call_count += 1
            call_idx = self.call_count
            self.active_count += 1
            self.max_concurrency = max(self.max_concurrency, self.active_count)

        try:
            if call_idx == 1:
                self.call1_started.set()
                self.call1_release.wait(timeout=5.0)
            n_samples = len(audio) if hasattr(audio, "__len__") else 64000
            logit_spoof = float(np.log(max(0.25, 1e-6)))
            logit_bonafide = float(np.log(max(1.0 - 0.25, 1e-6)))
            return DetectorOutput(
                raw_logits=(logit_spoof, logit_bonafide),
                bonafide_score=logit_bonafide - logit_spoof,
                synthetic_score=0.25,
                uncalibrated_synthetic_score=0.25,
                is_synthetic_suspected=False,
                input_samples=n_samples,
                device_used="cpu",
                model_id="ControlledUploadDetector",
            )
        finally:
            with self.lock:
                self.active_count -= 1

    def predict_batch(self, batch: Any) -> list[DetectorOutput]:
        return [self.predict(x) for x in batch]


class SignalingAdmissionController(AtomicAdmissionController):
    """AtomicAdmissionController with explicit condition signaling for deterministic test coordination."""

    def __init__(self, limit: int, name: str = "file_uploads") -> None:
        super().__init__(limit=limit, name=name)
        self.reservation_condition = threading.Condition(self._lock)

    def wait_for_reservations(self, count: int, timeout: float = 5.0) -> bool:
        with self.reservation_condition:
            return self.reservation_condition.wait_for(
                lambda: len(self._reserved_tokens) >= count, timeout=timeout
            )

    def try_reserve(self, token: str) -> bool:
        res = super().try_reserve(token)
        if res:
            with self.reservation_condition:
                self.reservation_condition.notify_all()
        return res

    def release(self, token: str) -> bool:
        res = super().release(token)
        with self.reservation_condition:
            self.reservation_condition.notify_all()
        return res


def test_api_upload_concurrency_admission_limit_and_worker_serialization(tmp_path: Path) -> None:
    """Test D: Concurrent accepted uploads with limit=2.

    Using one TestClient lifespan:
    - set upload admission limit to 2 via SignalingAdmissionController;
    - block the first accepted request in controlled evaluation;
    - submit additional upload requests concurrently;
    - at most 2 requests pass body admission (proven via explicit event synchronization);
    - 3rd excess request returns 429 promptly;
    - release accepted work;
    - all reservations return to zero;
    - R5 maximum physical detector concurrency strictly remains one.
    """
    wav_path = create_synthetic_wav(tmp_path / "valid.wav", duration_s=4.0)
    wav_bytes = wav_path.read_bytes()

    detector = ControlledUploadDetector()
    with TestClient(app) as client:
        orig_detector = getattr(app.state, "detector", None)
        admission = SignalingAdmissionController(limit=2, name="file_uploads")
        app.state.file_upload_admission = admission

        resp1: dict[str, Any] = {}
        resp2: dict[str, Any] = {}
        resp3: dict[str, Any] = {}

        def send_req1() -> None:
            r = client.post(
                "/api/v1/analyze/file",
                files={"file": ("req1.wav", io.BytesIO(wav_bytes), "audio/wav")},
            )
            resp1["status_code"] = r.status_code
            resp1["json"] = r.json()

        def send_req2() -> None:
            r = client.post(
                "/api/v1/analyze/file",
                files={"file": ("req2.wav", io.BytesIO(wav_bytes), "audio/wav")},
            )
            resp2["status_code"] = r.status_code
            resp2["json"] = r.json()

        def send_req3() -> None:
            r = client.post(
                "/api/v1/analyze/file",
                files={"file": ("req3.wav", io.BytesIO(wav_bytes), "audio/wav")},
            )
            resp3["status_code"] = r.status_code
            resp3["json"] = r.json()
            resp3["headers"] = dict(r.headers)

        t1 = threading.Thread(target=send_req1)
        t2 = threading.Thread(target=send_req2)
        t3 = threading.Thread(target=send_req3)

        try:
            app.state.detector = detector
            t1.start()

            # Wait until request 1 has physically entered detector call 1
            assert detector.call1_started.wait(timeout=5.0)
            assert admission.reserved_count == 1

            # Submit request 2 (which should pass admission and wait for worker queue)
            t2.start()

            # Deterministically wait until request 2 has acquired its admission reservation
            assert admission.wait_for_reservations(2, timeout=5.0)
            assert admission.reserved_count == 2

            # Submit request 3 (must fail admission immediately with 429)
            t3.start()
            t3.join(timeout=5.0)
            assert not t3.is_alive()

            assert resp3["status_code"] == 429
            detail = resp3["json"]["detail"]
            assert detail["error_code"] == "FILE_UPLOAD_CONCURRENCY_LIMIT_REACHED"
            assert resp3["headers"].get("retry-after") == "5"

            # Now release call 1
            detector.call1_release.set()
            t1.join(timeout=5.0)
            t2.join(timeout=5.0)
            assert not t1.is_alive()
            assert not t2.is_alive()

            assert resp1["status_code"] == 200
            assert resp2["status_code"] == 200

            # All reservations returned to zero
            assert admission.reserved_count == 0

            # Physical detector execution concurrency strictly remained 1
            assert detector.max_concurrency == 1
        finally:
            detector.call1_release.set()
            for t in (t1, t2, t3):
                if t.is_alive():
                    t.join(timeout=5.0)
                assert not t.is_alive()
            app.state.detector = orig_detector


def test_api_upload_total_request_ceiling_rejected_before_detector() -> None:
    """Test E: Oversized total request body rejected before detector invocation.

    - Content-Length exceeding MAX_UPLOAD_REQUEST_BYTES returns 413;
    - Detector is not invoked;
    - Reservation released back to zero.
    """
    detector = ControlledUploadDetector()
    with TestClient(app) as client:
        orig_detector = getattr(app.state, "detector", None)
        try:
            app.state.detector = detector
            overflow_len = str(MAX_UPLOAD_REQUEST_BYTES + 1024)
            resp = client.post(
                "/api/v1/analyze/file",
                headers={
                    "Content-Length": overflow_len,
                    "Content-Type": "multipart/form-data; boundary=sample_boundary",
                },
                content=b"",
            )
            assert resp.status_code == 413
            detail = resp.json()["detail"]
            assert detail["error_code"] == "PAYLOAD_TOO_LARGE"
            assert detector.call_count == 0
            assert app.state.file_upload_admission.reserved_count == 0
        finally:
            app.state.detector = orig_detector


def test_api_upload_error_cleanup_releases_reservation() -> None:
    """Test F: Request errors and processing failures release upload reservations."""
    with TestClient(app) as client:
        admission: AtomicAdmissionController = app.state.file_upload_admission
        assert admission.reserved_count == 0

        # 1. Corrupt audio (422)
        resp1 = client.post(
            "/api/v1/analyze/file",
            files={"file": ("corrupt.wav", io.BytesIO(b"RIFF\x24\x00\x00\x00NOTWAV"), "audio/wav")},
        )
        assert resp1.status_code == 422
        assert admission.reserved_count == 0

        # 2. Unsupported format (415)
        resp2 = client.post(
            "/api/v1/analyze/file",
            files={"file": ("test.txt", io.BytesIO(b"hello world"), "text/plain")},
        )
        assert resp2.status_code == 415
        assert admission.reserved_count == 0

        # 3. Empty file (422)
        resp3 = client.post(
            "/api/v1/analyze/file",
            files={"file": ("empty.wav", io.BytesIO(b""), "audio/wav")},
        )
        assert resp3.status_code == 422
        assert admission.reserved_count == 0
