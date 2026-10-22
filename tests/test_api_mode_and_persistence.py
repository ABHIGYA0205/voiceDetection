"""Tests verifying real vs mock mode enforcement and zero audio persistence."""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from services.api.main import app


def _create_wav_bytes(duration_s: float = 4.0, sample_rate: int = 16000) -> bytes:
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False, dtype=np.float32)
    samples = 0.5 * np.sin(2 * np.pi * 440.0 * t)
    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def test_mock_mode_flag_and_endpoints() -> None:
    """Verify mock mode sets is_mock_detector=True across health, ready, file, and stream."""
    with patch.dict("os.environ", {"VOICESHIELD_MOCK_MODEL": "true"}):
        with TestClient(app) as client:
            # 1. Health endpoint
            health = client.get("/health").json()
            assert health["status"] == "ok"
            assert health["model_loaded"] is True
            assert health["is_mock_detector"] is True

            # 2. Readiness endpoint
            ready = client.get("/ready").json()
            assert ready["status"] == "ready"
            assert ready["model_loaded"] is True
            assert ready["is_mock_detector"] is True

            # 3. File analysis endpoint
            wav_bytes = _create_wav_bytes(4.0)
            res = client.post(
                "/api/v1/analyze/file",
                files={"file": ("mock_test.wav", io.BytesIO(wav_bytes), "audio/wav")},
            )
            assert res.status_code == 200
            file_data = res.json()
            assert file_data["is_mock_detector"] is True
            assert file_data["metadata"]["is_mock_detector"] is True

            # 4. WebSocket endpoint
            with client.websocket_connect("/api/v1/stream") as ws:
                ws.send_text(
                    '{"type":"start","sample_rate":16000,"channels":1,"pcm_encoding":"int16"}'
                )
                start_ack = ws.receive_json()
                assert start_ack["type"] == "session_started"
                assert start_ack["is_mock_detector"] is True
                ws.send_text('{"type":"stop"}')
                stop_ack = ws.receive_json()
                assert stop_ack["type"] == "session_stopped"


def test_real_mode_refuses_ready_when_checkpoint_absent(tmp_path: Path) -> None:
    """Verify real mode (VOICESHIELD_MOCK_MODEL=false) refuses ready status when checkpoint is absent."""
    missing_checkpoint = tmp_path / "nonexistent" / "AASIST-L.pth"

    with patch.dict(
        "os.environ",
        {
            "VOICESHIELD_MOCK_MODEL": "false",
            "VOICESHIELD_CHECKPOINT_PATH": str(missing_checkpoint),
        },
    ):
        with TestClient(app) as client:
            # 1. Health endpoint stays alive (liveness), but indicates model not loaded
            health = client.get("/health").json()
            assert health["status"] == "ok"
            assert health["model_loaded"] is False
            assert health["is_mock_detector"] is False

            # 2. Readiness endpoint MUST return HTTP 503
            resp_ready = client.get("/ready")
            assert resp_ready.status_code == 503
            ready_detail = resp_ready.json()["detail"]
            assert ready_detail["status"] == "not_ready"
            assert "AASIST-L checkpoint not found" in ready_detail["reason"]
            assert "Setup Instructions" in ready_detail["reason"]

            # 3. File analysis endpoint returns 503
            wav_bytes = _create_wav_bytes(4.0)
            resp_file = client.post(
                "/api/v1/analyze/file",
                files={"file": ("test.wav", io.BytesIO(wav_bytes), "audio/wav")},
            )
            assert resp_file.status_code == 503
            assert resp_file.json()["detail"]["error_code"] == "DETECTOR_NOT_LOADED"

            # 4. WebSocket stream rejects with error event
            with client.websocket_connect("/api/v1/stream") as ws:
                ws.send_text(
                    '{"type":"start","sample_rate":16000,"channels":1,"pcm_encoding":"int16"}'
                )
                err_event = ws.receive_json()
                assert err_event["type"] == "error"
                assert err_event["error_code"] == "DETECTOR_NOT_READY"


def test_zero_audio_persistence_confirmation(tmp_path: Path) -> None:
    """Verify that neither file analysis nor streaming persists any audio files to disk."""
    workspace_root = Path.cwd()

    def count_audio_files(base_dir: Path) -> int:
        audio_extensions = {".wav", ".flac", ".mp3", ".raw", ".pcm", ".ogg"}
        count = 0
        for p in base_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in audio_extensions:
                # Exclude fixtures in tests
                if "tests" not in str(p) and "data" not in str(p):
                    count += 1
        return count

    before_count = count_audio_files(workspace_root)

    with patch.dict("os.environ", {"VOICESHIELD_MOCK_MODEL": "true"}):
        with TestClient(app) as client:
            # File analysis
            wav_bytes = _create_wav_bytes(5.0)
            resp = client.post(
                "/api/v1/analyze/file",
                files={"file": ("zero_persistence_test.wav", io.BytesIO(wav_bytes), "audio/wav")},
            )
            assert resp.status_code == 200

            # Streaming
            with client.websocket_connect("/api/v1/stream") as ws:
                ws.send_text(
                    '{"type":"start","sample_rate":16000,"channels":1,"pcm_encoding":"int16"}'
                )
                ws.receive_json()

                # Stream 2 seconds of PCM data
                pcm_data = (np.zeros(32000, dtype=np.int16)).tobytes()
                ws.send_bytes(pcm_data)

                # Send stop
                ws.send_text('{"type":"stop"}')
                ws.receive_json()

    after_count = count_audio_files(workspace_root)
    assert after_count == before_count, "Audio files were unexpectedly persisted to disk!"


def test_synthetic_fixtures_generator() -> None:
    """Verify generated non-human audio fixtures meet format specifications."""
    from scripts.demo_transport_verification import generate_synthetic_fixtures

    fixtures = generate_synthetic_fixtures()
    assert "sine_440hz_16k_mono" in fixtures
    assert "chirp_200_800hz_48k_stereo" in fixtures
    assert "silence_16k_mono" in fixtures

    for name, data in fixtures.items():
        assert len(data) > 1000
        # Verify valid WAV container
        with sf.SoundFile(io.BytesIO(data)) as sf_file:
            assert sf_file.format == "WAV"
            if "16k" in name:
                assert sf_file.samplerate == 16000
            elif "48k" in name:
                assert sf_file.samplerate == 48000
            if "stereo" in name:
                assert sf_file.channels == 2
            else:
                assert sf_file.channels == 1
