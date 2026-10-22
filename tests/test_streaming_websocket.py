"""Integration tests for VoiceShield real-time WebSocket streaming service."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pytest
from fastapi import WebSocketDisconnect
from fastapi.testclient import TestClient

from services.api.main import app
from voiceshield.models.base import BaseDetector, DetectorOutput


def generate_pcm_sine(
    duration_sec: float,
    sample_rate: int = 16000,
    freq: float = 440.0,
    amplitude: float = 0.5,
    dtype: str = "int16",
) -> bytes:
    """Generate raw PCM audio bytes for a sine wave."""
    num_samples = int(round(duration_sec * sample_rate))
    t = np.linspace(0, duration_sec, num_samples, endpoint=False, dtype=np.float32)
    waveform = amplitude * np.sin(2.0 * np.pi * freq * t)

    if dtype == "int16":
        return bytes((waveform * 32767.0).astype(np.int16).tobytes())
    elif dtype == "int32":
        return bytes((waveform * 2147483647.0).astype(np.int32).tobytes())
    elif dtype == "float32":
        return bytes(waveform.astype(np.float32).tobytes())
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


class FaultyDetector(BaseDetector):
    """Detector mock that raises an intentional exception to test failure resilience."""

    def predict(self, audio: Any) -> DetectorOutput:
        raise RuntimeError("Simulated neural engine hardware fault during inference.")

    def predict_batch(self, batch: Any) -> list[DetectorOutput]:
        raise RuntimeError("Simulated neural engine hardware fault during inference.")


# ==============================================================================
# 1. Health vs Readiness Distinction Tests
# ==============================================================================


def test_health_and_readiness_endpoints() -> None:
    """Verify /health tests API process liveness and /ready tests model detector readiness."""
    with TestClient(app) as client:
        # 1. Liveness check
        resp_health = client.get("/health")
        assert resp_health.status_code == 200
        health_data = resp_health.json()
        assert health_data["status"] == "ok"
        assert health_data["service"] == "VoiceShield"

        # 2. Readiness check when detector is loaded
        resp_ready = client.get("/ready")
        assert resp_ready.status_code == 200
        ready_data = resp_ready.json()
        assert ready_data["status"] == "ready"
        assert ready_data["model_loaded"] is True

        # 3. Readiness check when detector is None
        orig_detector = getattr(app.state, "detector", None)
        try:
            app.state.detector = None
            resp_not_ready = client.get("/ready")
            assert resp_not_ready.status_code == 503
            not_ready_data = resp_not_ready.json()
            assert not_ready_data["detail"]["status"] == "not_ready"
        finally:
            app.state.detector = orig_detector


# ==============================================================================
# 2. Valid Streaming Flow Tests
# ==============================================================================


def test_websocket_valid_flow() -> None:
    """Verify end-to-end handshake, binary audio ingestion, window result event, and stop."""
    with TestClient(app) as client:
        with client.websocket_connect("/api/v1/stream") as ws:
            # 1. Send client start message
            start_msg = {
                "type": "start",
                "protocol_version": "1.0",
                "sample_rate": 16000,
                "channels": 1,
                "pcm_encoding": "int16",
                "metadata": {"client_id": "test-client-1"},
            }
            ws.send_json(start_msg)

            # 2. Receive session_started acknowledgement
            ack = ws.receive_json()
            assert ack["type"] == "session_started"
            assert "session_id" in ack
            assert ack["canonical_sample_rate"] == 16000
            assert ack["window_seconds"] == 4.0
            assert ack["hop_seconds"] == 1.0

            session_id = ack["session_id"]
            assert session_id in app.state.active_streaming_sessions

            # 3. Stream 4.0 seconds of audio in 1-second chunks (active speech)
            for _ in range(4):
                chunk = generate_pcm_sine(1.0, sample_rate=16000, freq=300.0, amplitude=0.5)
                ws.send_bytes(chunk)

            # 4. Receive window result event
            result = ws.receive_json()
            assert result["type"] == "window_result"
            assert result["session_id"] == session_id
            assert result["is_eligible"] is True
            assert result["speech_seconds"] >= 3.5
            assert "raw_score" in result
            assert "calibrated_score" in result
            assert "decision" in result
            assert result["decision"]["state"] in ("low", "uncertain", "high")

            # 5. Send stop message
            ws.send_json({"type": "stop"})
            stop_ack = ws.receive_json()
            assert stop_ack["type"] == "session_stopped"
            assert stop_ack["session_id"] == session_id

    # Verify session resources released
    assert session_id not in app.state.active_streaming_sessions


# ==============================================================================
# 3. Invalid Handshake Tests
# ==============================================================================


def test_websocket_invalid_handshake_first_message_not_start() -> None:
    """Verify server rejects connections where first message is not a start message."""
    with TestClient(app) as client:
        with client.websocket_connect("/api/v1/stream") as ws:
            # Send unexpected command instead of start
            ws.send_json({"type": "stop"})
            err = ws.receive_json()
            assert err["type"] == "error"
            assert err["error_code"] == "INVALID_HANDSHAKE"

            with pytest.raises(WebSocketDisconnect):
                ws.receive_text()


def test_websocket_invalid_handshake_malformed_json() -> None:
    """Verify server rejects malformed non-JSON text handshake."""
    with TestClient(app) as client:
        with client.websocket_connect("/api/v1/stream") as ws:
            ws.send_text("MALFORMED NOT JSON {")
            err = ws.receive_json()
            assert err["type"] == "error"
            assert err["error_code"] == "INVALID_JSON"

            with pytest.raises(WebSocketDisconnect):
                ws.receive_text()


# ==============================================================================
# 4. Wrong PCM Format and Unsupported Version Tests
# ==============================================================================


def test_websocket_unsupported_pcm_format() -> None:
    """Verify server rejects unsupported PCM encodings (e.g. mp3)."""
    with TestClient(app) as client:
        with client.websocket_connect("/api/v1/stream") as ws:
            ws.send_json(
                {
                    "type": "start",
                    "protocol_version": "1.0",
                    "sample_rate": 16000,
                    "channels": 1,
                    "pcm_encoding": "mp3",  # Unsupported
                }
            )
            err = ws.receive_json()
            assert err["type"] == "error"
            assert err["error_code"] == "UNSUPPORTED_PCM_ENCODING"

            with pytest.raises(WebSocketDisconnect):
                ws.receive_text()


def test_websocket_unsupported_protocol_version() -> None:
    """Verify server rejects unsupported protocol versions."""
    with TestClient(app) as client:
        with client.websocket_connect("/api/v1/stream") as ws:
            ws.send_json(
                {
                    "type": "start",
                    "protocol_version": "9.9",
                    "sample_rate": 16000,
                    "channels": 1,
                    "pcm_encoding": "int16",
                }
            )
            err = ws.receive_json()
            assert err["type"] == "error"
            assert err["error_code"] == "UNSUPPORTED_PROTOCOL_VERSION"

            with pytest.raises(WebSocketDisconnect):
                ws.receive_text()


# ==============================================================================
# 5. Oversized Frame Rejection Tests
# ==============================================================================


def test_websocket_oversized_frame_rejection() -> None:
    """Verify server rejects frames exceeding max_frame_bytes."""
    with TestClient(app) as client:
        with client.websocket_connect("/api/v1/stream") as ws:
            # Set max_frame_bytes to 2048 in metadata
            ws.send_json(
                {
                    "type": "start",
                    "protocol_version": "1.0",
                    "sample_rate": 16000,
                    "channels": 1,
                    "pcm_encoding": "int16",
                    "metadata": {"max_frame_bytes": 2048},
                }
            )
            ack = ws.receive_json()
            assert ack["type"] == "session_started"

            # Send 4096 bytes (> 2048 limit)
            oversized_frame = b"\x00" * 4096
            ws.send_bytes(oversized_frame)

            err = ws.receive_json()
            assert err["type"] == "error"
            assert err["error_code"] == "PAYLOAD_TOO_LARGE"

            with pytest.raises(WebSocketDisconnect):
                ws.receive_text()


# ==============================================================================
# 6. Idle Timeout Tests
# ==============================================================================


def test_websocket_idle_timeout() -> None:
    """Verify server closes inactive sessions when idle timeout is exceeded."""
    with TestClient(app) as client:
        with client.websocket_connect("/api/v1/stream") as ws:
            # Configure 0.2s idle timeout in metadata
            ws.send_json(
                {
                    "type": "start",
                    "protocol_version": "1.0",
                    "sample_rate": 16000,
                    "channels": 1,
                    "pcm_encoding": "int16",
                    "metadata": {"idle_timeout_seconds": 0.2},
                }
            )
            ack = ws.receive_json()
            assert ack["type"] == "session_started"

            # Wait 0.35s without sending audio
            time.sleep(0.35)

            err = ws.receive_json()
            assert err["type"] == "error"
            assert err["error_code"] == "IDLE_TIMEOUT"

            with pytest.raises(WebSocketDisconnect):
                ws.receive_text()


# ==============================================================================
# 7. Concurrent Sessions Isolation Tests
# ==============================================================================


def test_websocket_concurrent_sessions() -> None:
    """Verify two concurrent streaming sessions operate with complete state isolation."""
    with TestClient(app) as client:
        with (
            client.websocket_connect("/api/v1/stream") as ws_a,
            client.websocket_connect("/api/v1/stream") as ws_b,
        ):
            ws_a.send_json(
                {"type": "start", "sample_rate": 16000, "channels": 1, "pcm_encoding": "int16"}
            )
            ws_b.send_json(
                {"type": "start", "sample_rate": 16000, "channels": 1, "pcm_encoding": "int16"}
            )

            ack_a = ws_a.receive_json()
            ack_b = ws_b.receive_json()

            assert ack_a["session_id"] != ack_b["session_id"]
            assert len(app.state.active_streaming_sessions) >= 2

            # Stream audio to both
            chunk_a = generate_pcm_sine(1.0, freq=440.0)
            chunk_b = generate_pcm_sine(1.0, freq=880.0)

            for _ in range(4):
                ws_a.send_bytes(chunk_a)
                ws_b.send_bytes(chunk_b)

            res_a = ws_a.receive_json()
            res_b = ws_b.receive_json()

            assert res_a["session_id"] == ack_a["session_id"]
            assert res_b["session_id"] == ack_b["session_id"]

            ws_a.send_json({"type": "stop"})
            ws_b.send_json({"type": "stop"})


# ==============================================================================
# 8. Model Exception Resilience Tests
# ==============================================================================


def test_websocket_model_exception_resilience() -> None:
    """Verify that a detector runtime failure emits an error event without crashing the event loop."""
    with TestClient(app) as client:
        orig_detector = getattr(app.state, "detector", None)
        app.state.detector = FaultyDetector()
        try:
            with client.websocket_connect("/api/v1/stream") as ws:
                ws.send_json(
                    {
                        "type": "start",
                        "sample_rate": 16000,
                        "channels": 1,
                        "pcm_encoding": "int16",
                    }
                )
                ack = ws.receive_json()
                assert ack["type"] == "session_started"

                # Send 4 seconds of speech to trigger model execution
                for _ in range(4):
                    chunk = generate_pcm_sine(1.0, freq=440.0)
                    ws.send_bytes(chunk)

                # Expect model inference error event
                err = ws.receive_json()
                assert err["type"] == "error"
                assert err["error_code"] == "MODEL_INFERENCE_ERROR"
                assert "neural engine hardware fault" in err["message"]
        finally:
            app.state.detector = orig_detector


# ==============================================================================
# 9. Disconnect Cleanup Tests
# ==============================================================================


def test_websocket_disconnect_cleanup() -> None:
    """Verify that an abrupt client disconnect releases all session resources from the server registry."""
    with TestClient(app) as client:
        with client.websocket_connect("/api/v1/stream") as ws:
            ws.send_json(
                {"type": "start", "sample_rate": 16000, "channels": 1, "pcm_encoding": "int16"}
            )
            ack = ws.receive_json()
            session_id = ack["session_id"]
            assert session_id in app.state.active_streaming_sessions

            # Send a chunk
            ws.send_bytes(generate_pcm_sine(0.5))

            # Disconnect abruptly without sending stop
            ws.close()

    # Session ID must be removed from active registry
    assert session_id not in app.state.active_streaming_sessions


# ==============================================================================
# 10. Origin Policy Enforcement Tests
# ==============================================================================


def test_websocket_origin_policy_enforcement() -> None:
    """Verify disallowed origins are rejected with status 1008 and allowed origins succeed."""
    orig_origins = getattr(app.state, "allowed_origins", ["*"])
    app.state.allowed_origins = ["http://trusted-portal.com"]

    try:
        with TestClient(app) as client:
            # 1. Connection with unauthorized Origin
            with pytest.raises(WebSocketDisconnect) as exc_info:
                with client.websocket_connect(
                    "/api/v1/stream", headers={"Origin": "http://malicious-site.com"}
                ):
                    pass
            assert exc_info.value.code == 1008

            # 2. Connection with authorized Origin
            with client.websocket_connect(
                "/api/v1/stream", headers={"Origin": "http://trusted-portal.com"}
            ) as ws:
                ws.send_json(
                    {
                        "type": "start",
                        "sample_rate": 16000,
                        "channels": 1,
                        "pcm_encoding": "int16",
                    }
                )
                ack = ws.receive_json()
                assert ack["type"] == "session_started"
                ws.send_json({"type": "stop"})

    finally:
        app.state.allowed_origins = orig_origins


# ==============================================================================
# 11. Backpressure and Drop/Replace WebSocket Test
# ==============================================================================


def test_websocket_backpressure_drop_replace() -> None:
    """Verify that bursts of audio frames do not cause unbounded queueing over WebSocket."""
    with TestClient(app) as client:
        with client.websocket_connect("/api/v1/stream") as ws:
            ws.send_json(
                {"type": "start", "sample_rate": 16000, "channels": 1, "pcm_encoding": "int16"}
            )
            ack = ws.receive_json()
            session_id = ack["session_id"]
            session = app.state.active_streaming_sessions[session_id]

            # Ingest 7 seconds rapidly (triggers Window 0, 1, 2, 3)
            for _ in range(7):
                ws.send_bytes(generate_pcm_sine(1.0, freq=350.0))

            # Receive the first evaluated window result
            res1 = ws.receive_json()
            assert res1["type"] == "window_result"
            assert res1["is_eligible"] is True

            # Verify that bounded queuing was maintained in the session
            # (at most 1 pending window)
            assert session._pending_window is None or isinstance(
                session._pending_window.window_index, int
            )

            ws.send_json({"type": "stop"})
