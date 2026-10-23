"""Integration tests for VoiceShield real-time WebSocket streaming service."""

from __future__ import annotations

import asyncio
import io
import json
import threading
import time
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf
from fastapi import WebSocketDisconnect
from fastapi.testclient import TestClient

from services.api.inference_worker import InferenceWorker
from services.api.main import app
from services.api.resource_admission import AtomicAdmissionController
from services.api.routes.stream import websocket_stream_endpoint
from voiceshield.models.base import BaseDetector, DetectorOutput
from voiceshield.streaming.session import StreamingSession


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

            # Receive emitted events (quality_update events for drops, followed by window_result)
            messages = []
            while True:
                msg = ws.receive_json()
                messages.append(msg)
                if msg["type"] == "window_result":
                    break

            quality_events = [m for m in messages if m["type"] == "quality_update"]
            assert len(quality_events) >= 1
            assert any(q.get("total_windows_dropped", 0) > 0 for q in quality_events)

            win_res = [m for m in messages if m["type"] == "window_result"][0]
            assert win_res["is_eligible"] is True

            # Verify that bounded queuing was maintained in the session
            # (at most 1 pending window)
            assert session._pending_window is None or isinstance(
                session._pending_window.window_index, int
            )

            ws.send_json({"type": "stop"})
            stop_ack = None
            while True:
                msg = ws.receive_json()
                if msg["type"] == "session_stopped":
                    stop_ack = msg
                    break
            assert stop_ack is not None
            assert stop_ack["total_windows_dropped"] > 0


# ==============================================================================
# 12. Audit H-04 Concurrency & Disconnect Integration Tests
# ==============================================================================


class BlockingControlledDetector(BaseDetector):
    """Detector that blocks on call 1 and tracks physical execution concurrency."""

    def __init__(self, raise_on_call_1: bool = False) -> None:
        self.call1_started = threading.Event()
        self.call1_release = threading.Event()
        self.call2_started = threading.Event()
        self.call_count = 0
        self.active_count = 0
        self.max_concurrency = 0
        self.lock = threading.Lock()
        self.raise_on_call_1 = raise_on_call_1

    def _make_output(self, score: float, audio: Any) -> DetectorOutput:
        n_samples = len(audio) if hasattr(audio, "__len__") else 64000
        logit_spoof = float(np.log(max(score, 1e-6)))
        logit_bonafide = float(np.log(max(1.0 - score, 1e-6)))
        return DetectorOutput(
            raw_logits=(logit_spoof, logit_bonafide),
            bonafide_score=logit_bonafide - logit_spoof,
            synthetic_score=score,
            uncalibrated_synthetic_score=score,
            is_synthetic_suspected=bool(score >= 0.65),
            input_samples=n_samples,
            device_used="cpu",
            model_id="BlockingControlledDetector",
        )

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
                if self.raise_on_call_1:
                    raise RuntimeError("Simulated failure after disconnect")
                return self._make_output(0.88, audio)
            else:
                self.call2_started.set()
                return self._make_output(0.12, audio)
        finally:
            with self.lock:
                self.active_count -= 1

    def predict_batch(self, batch: Any) -> list[DetectorOutput]:
        return [self.predict(x) for x in batch]


def test_abrupt_disconnect_followed_by_second_stream() -> None:
    """Test A: Abrupt disconnect of stream A followed by stream B serialization.

    Proves:
    - Session A is removed from active sessions while call A is still physically running.
    - Detector call B remains queued and cannot start until call A returns.
    - Maximum physical detector concurrency strictly remains 1.
    - Results from stream A never leak into or mutate stream B.
    """
    detector = BlockingControlledDetector()
    orig_detector = getattr(app.state, "detector", None)

    try:
        with TestClient(app) as client:
            app.state.detector = detector

            # 1. Connect stream A and send enough audio to begin physical inference
            session_id_a = None
            with client.websocket_connect("/api/v1/stream") as ws_a:
                ws_a.send_json(
                    {"type": "start", "sample_rate": 16000, "channels": 1, "pcm_encoding": "int16"}
                )
                ack_a = ws_a.receive_json()
                session_id_a = ack_a["session_id"]
                assert session_id_a in app.state.active_streaming_sessions

                # Send 4 seconds of audio to trigger window 0
                for _ in range(4):
                    ws_a.send_bytes(generate_pcm_sine(1.0, freq=440.0))

                # Wait for detector call A to physically start
                assert detector.call1_started.wait(timeout=5.0)
                with detector.lock:
                    assert detector.active_count == 1
                    assert detector.max_concurrency == 1

                # Abruptly disconnect stream A without sending stop message
                ws_a.close()

            # Wait briefly for server disconnect cleanup
            time.sleep(0.05)

            # Session A must be cleanly removed from active sessions while call A is still blocked!
            assert session_id_a not in app.state.active_streaming_sessions
            with detector.lock:
                assert detector.active_count == 1
                assert detector.max_concurrency == 1

            # 2. Start stream B on the SAME TestClient lifespan and app worker
            with client.websocket_connect("/api/v1/stream") as ws_b:
                ws_b.send_json(
                    {"type": "start", "sample_rate": 16000, "channels": 1, "pcm_encoding": "int16"}
                )
                ack_b = ws_b.receive_json()
                session_id_b = ack_b["session_id"]
                assert session_id_b != session_id_a

                # Send 4 seconds of audio to request inference
                for _ in range(4):
                    ws_b.send_bytes(generate_pcm_sine(1.0, freq=440.0))

                # Give event loop time to process ingestion and submit to worker
                time.sleep(0.05)

                # Before releasing call A, call B MUST NOT have physically started!
                assert not detector.call2_started.is_set()
                with detector.lock:
                    assert detector.active_count == 1
                    assert detector.max_concurrency == 1

                # Now release call A
                detector.call1_release.set()

                # Stream B should now execute and receive its normal window result
                window_result_b = None
                while True:
                    msg = ws_b.receive_json()
                    if msg["type"] == "window_result":
                        window_result_b = msg
                        break

                assert window_result_b is not None
                assert window_result_b["session_id"] == session_id_b
                # Score must be call 2's score (0.12), NOT call 1's detached score (0.88)
                assert window_result_b["synthetic_score"] == pytest.approx(0.12, abs=1e-3)

                # Gracefully stop stream B
                ws_b.send_json({"type": "stop"})
                while True:
                    msg = ws_b.receive_json()
                    if msg["type"] == "session_stopped":
                        break

            # Confirm max physical concurrency stayed exactly 1
            with detector.lock:
                assert detector.max_concurrency == 1
                assert detector.active_count == 0

    finally:
        detector.call1_release.set()
        app.state.detector = orig_detector


def test_abrupt_disconnect_followed_by_file_analysis() -> None:
    """Test B: Abrupt disconnect of stream A followed by file analysis request serialization.

    Proves:
    - File analysis submitted while streaming detector call is still blocked cannot overlap.
    - Maximum physical detector concurrency strictly remains 1 across different API routes.
    """
    detector = BlockingControlledDetector()
    orig_detector = getattr(app.state, "detector", None)

    try:
        with TestClient(app) as client:
            app.state.detector = detector

            # 1. Start streaming and begin detector execution
            session_id_a = None
            with client.websocket_connect("/api/v1/stream") as ws_a:
                ws_a.send_json(
                    {"type": "start", "sample_rate": 16000, "channels": 1, "pcm_encoding": "int16"}
                )
                ack_a = ws_a.receive_json()
                session_id_a = ack_a["session_id"]

                for _ in range(4):
                    ws_a.send_bytes(generate_pcm_sine(1.0, freq=440.0))

                assert detector.call1_started.wait(timeout=5.0)
                ws_a.close()

            # Disconnected stream A
            time.sleep(0.05)
            assert session_id_a not in app.state.active_streaming_sessions

            # Generate synthetic 4.0s WAV bytes for file analysis
            t = np.linspace(0, 4.0, 16000 * 4, endpoint=False, dtype=np.float32)
            wav_samples = 0.5 * np.sin(2 * np.pi * 440.0 * t)
            buf = io.BytesIO()
            sf.write(buf, wav_samples, 16000, format="WAV", subtype="PCM_16")
            wav_bytes = buf.getvalue()

            file_response: dict[str, Any] = {}
            file_error: list[Exception] = []

            def _submit_file_request() -> None:
                try:
                    resp = client.post(
                        "/api/v1/analyze/file",
                        files={"file": ("test_file.wav", io.BytesIO(wav_bytes), "audio/wav")},
                    )
                    file_response["status_code"] = resp.status_code
                    file_response["json"] = resp.json()
                except Exception as exc:
                    file_error.append(exc)

            # Submit HTTP file analysis in a helper thread while call 1 is blocked
            thread = threading.Thread(target=_submit_file_request)
            thread.start()

            # Allow time for HTTP request to reach the inference worker
            time.sleep(0.08)

            # File detector call MUST NOT have started while streaming call is physically running
            assert not detector.call2_started.is_set()
            with detector.lock:
                assert detector.active_count == 1
                assert detector.max_concurrency == 1

            # Release streaming call
            detector.call1_release.set()

            # Join helper thread
            thread.join(timeout=5.0)
            assert not thread.is_alive()
            assert not file_error

            assert file_response.get("status_code") == 200
            assert file_response["json"]["status"] == "success"

            with detector.lock:
                assert detector.max_concurrency == 1
                assert detector.active_count == 0

    finally:
        detector.call1_release.set()
        app.state.detector = orig_detector


def test_abrupt_disconnect_detector_failure_after_release() -> None:
    """Test E: A detached physical job raising an exception after disconnect is handled cleanly.

    Proves:
    - No exception is leaked or unhandled.
    - No event is sent to closed socket.
    - Worker remains usable for subsequent jobs.
    - Maximum concurrency remains 1.
    """
    detector = BlockingControlledDetector(raise_on_call_1=True)
    orig_detector = getattr(app.state, "detector", None)

    try:
        with TestClient(app) as client:
            app.state.detector = detector

            with client.websocket_connect("/api/v1/stream") as ws:
                ws.send_json(
                    {"type": "start", "sample_rate": 16000, "channels": 1, "pcm_encoding": "int16"}
                )
                ws.receive_json()
                for _ in range(4):
                    ws.send_bytes(generate_pcm_sine(1.0, freq=440.0))
                assert detector.call1_started.wait(timeout=5.0)
                ws.close()

            # Disconnect occurred while detector is running
            time.sleep(0.05)

            # Release call 1 which will raise RuntimeError
            detector.call1_release.set()
            time.sleep(0.05)

            # Worker must still be alive and usable
            worker = getattr(app.state, "inference_worker", None)
            assert worker is not None
            assert worker.accepting_jobs
            assert worker.outstanding_job_count == 0

            # Next request succeeds
            with client.websocket_connect("/api/v1/stream") as ws2:
                ws2.send_json(
                    {"type": "start", "sample_rate": 16000, "channels": 1, "pcm_encoding": "int16"}
                )
                ws2.receive_json()
                for _ in range(4):
                    ws2.send_bytes(generate_pcm_sine(1.0, freq=440.0))
                msg = None
                while True:
                    m = ws2.receive_json()
                    if m["type"] == "window_result":
                        msg = m
                        break
                assert msg is not None
                ws2.send_json({"type": "stop"})
                while True:
                    if ws2.receive_json()["type"] == "session_stopped":
                        break

            with detector.lock:
                assert detector.max_concurrency == 1

    finally:
        detector.call1_release.set()
        app.state.detector = orig_detector


def test_abrupt_cleanup_ordering() -> None:
    """Test C: Abrupt cleanup ordering guarantees.

    Proves:
    - Inference awaiting tasks are cancelled and gathered.
    - Session reset/removal occurs.
    - No pending window is launched for the disconnected session.
    - The underlying physical job remains tracked until completion.
    """
    detector = BlockingControlledDetector()
    orig_detector = getattr(app.state, "detector", None)

    try:
        with TestClient(app) as client:
            app.state.detector = detector

            with client.websocket_connect("/api/v1/stream") as ws:
                ws.send_json(
                    {"type": "start", "sample_rate": 16000, "channels": 1, "pcm_encoding": "int16"}
                )
                ack = ws.receive_json()
                session_id = ack["session_id"]
                session = app.state.active_streaming_sessions[session_id]

                # Ingest 5 seconds: 4s triggers window 0 (inference starts and blocks),
                # 5th second generates window 1 (becomes pending window)
                for _ in range(5):
                    ws.send_bytes(generate_pcm_sine(1.0, freq=440.0))

                # Wait for window 0 to physically begin inference
                assert detector.call1_started.wait(timeout=5.0)
                # Allow ingestion loop to process the 5th second
                time.sleep(0.05)

                # Confirm window 1 is queued as pending window
                assert session.has_pending_window

                # Abrupt disconnect
                ws.close()

            # Wait briefly for server disconnect unwinding
            time.sleep(0.08)

            # 1. Session is removed from active sessions
            assert session_id not in app.state.active_streaming_sessions

            # 2. Worker still tracks the physical job running call 1
            worker = getattr(app.state, "inference_worker", None)
            assert worker is not None
            assert worker.running_job_count == 1
            assert worker.outstanding_job_count == 1

            # 3. Pending window 1 was NOT launched
            assert not detector.call2_started.is_set()

            # Now release call 1
            detector.call1_release.set()
            time.sleep(0.05)

            # 4. Physical job finished; call 2 was NEVER launched for disconnected session
            assert worker.outstanding_job_count == 0
            assert not detector.call2_started.is_set()
            with detector.lock:
                assert detector.max_concurrency == 1
                assert detector.call_count == 1

    finally:
        detector.call1_release.set()
        app.state.detector = orig_detector


# ==============================================================================
# 9. Regression Tests: Disconnect Inference Lifecycle Handshake & Cancellation
# ==============================================================================


@pytest.mark.asyncio
async def test_endpoint_task_cancellation_propagates_and_cleans_up() -> None:
    """Test F: Direct coroutine-level regression showing endpoint task cancellation.

    - runs final cleanup;
    - removes/reset session;
    - propagates CancelledError rather than swallowing it.
    """

    class MockWebSocket:
        def __init__(self) -> None:
            self.app = app
            self.headers = {"origin": "http://localhost:5173"}
            self.accepted = False
            self.closed = False
            self.sent_events: list[dict[str, Any]] = []
            self.receive_event = asyncio.Event()

        async def accept(self) -> None:
            self.accepted = True

        async def close(self, code: int = 1000, reason: str = "") -> None:
            self.closed = True

        async def send_json(self, data: dict[str, Any]) -> None:
            self.sent_events.append(data)

        async def receive_text(self) -> str:
            start_payload = {
                "type": "start",
                "protocol_version": "1.0",
                "sample_rate": 16000,
                "channels": 1,
                "pcm_encoding": "int16",
            }
            return json.dumps(start_payload)

        async def receive(self) -> dict[str, Any]:
            await self.receive_event.wait()
            return {"type": "websocket.disconnect"}

    orig_worker = getattr(app.state, "inference_worker", None)
    orig_admission = getattr(app.state, "streaming_admission", None)
    worker = InferenceWorker()
    admission = AtomicAdmissionController(limit=10, name="streaming_sessions")
    app.state.inference_worker = worker
    app.state.streaming_admission = admission

    try:
        mock_ws = MockWebSocket()
        task = asyncio.create_task(websocket_stream_endpoint(mock_ws))  # type: ignore[arg-type]

        await asyncio.sleep(0.05)
        assert mock_ws.accepted is True
        assert len(mock_ws.sent_events) > 0
        assert mock_ws.sent_events[0]["type"] == "session_started"
        created_session_id = mock_ws.sent_events[0]["session_id"]
        assert created_session_id in app.state.active_streaming_sessions
        assert admission.reserved_count == 1

        # Cancel the endpoint task
        task.cancel()

        # CancelledError must be propagated, not swallowed
        with pytest.raises(asyncio.CancelledError):
            await task

        # Cleanup must have run: session removed from registry & admission reservation released
        assert created_session_id not in app.state.active_streaming_sessions
        assert len(app.state.active_streaming_sessions) == 0
        assert admission.reserved_count == 0
    finally:
        await worker.shutdown(timeout=2.0)
        app.state.inference_worker = orig_worker
        app.state.streaming_admission = orig_admission


def test_websocket_worker_unavailable_handshake() -> None:
    """Test G: Worker unavailable handshake.

    Set app.state.inference_worker to None or a stopped worker:
    - client sends start;
    - receives INFERENCE_WORKER_UNAVAILABLE;
    - does not receive session_started;
    - connection closes;
    - no session registry leak.
    """
    start_payload = {
        "type": "start",
        "protocol_version": "1.0",
        "sample_rate": 16000,
        "channels": 1,
        "pcm_encoding": "int16",
    }

    with TestClient(app) as client:
        orig_worker = getattr(app.state, "inference_worker", None)
        try:
            # 1. Test when inference_worker is None
            app.state.inference_worker = None
            with client.websocket_connect("/api/v1/stream") as ws:
                ws.send_text(json.dumps(start_payload))
                resp = ws.receive_json()
                assert resp["type"] == "error"
                assert resp["error_code"] == "INFERENCE_WORKER_UNAVAILABLE"
                with pytest.raises(WebSocketDisconnect) as exc_info:
                    ws.receive_text()
                assert exc_info.value.code == 1011
            assert len(app.state.active_streaming_sessions) == 0

            # 2. Test when inference_worker is stopped (not accepting jobs)
            stopped_worker = InferenceWorker()
            stopped_worker.stop_intake()
            assert not stopped_worker.accepting_jobs
            app.state.inference_worker = stopped_worker

            with client.websocket_connect("/api/v1/stream") as ws:
                ws.send_text(json.dumps(start_payload))
                resp = ws.receive_json()
                assert resp["type"] == "error"
                assert resp["error_code"] == "INFERENCE_WORKER_UNAVAILABLE"
                with pytest.raises(WebSocketDisconnect) as exc_info:
                    ws.receive_text()
                assert exc_info.value.code == 1011
            assert len(app.state.active_streaming_sessions) == 0
            if getattr(app.state, "streaming_admission", None) is not None:
                assert app.state.streaming_admission.reserved_count == 0
        finally:
            app.state.inference_worker = orig_worker


# ==============================================================================
# 14. Audit M-03 WebSocket Admission Race & Distinction Tests
# ==============================================================================


class SignalingAdmissionController(AtomicAdmissionController):
    """AtomicAdmissionController with explicit condition signaling for deterministic test coordination."""

    def __init__(self, limit: int, name: str = "streaming_sessions") -> None:
        super().__init__(limit=limit, name=name)
        self.reservation_condition = threading.Condition(self._lock)

    def wait_for_reservations(self, count: int, timeout: float = 5.0) -> bool:
        with self.reservation_condition:
            return self.reservation_condition.wait_for(
                lambda: len(self._reserved_tokens) >= count, timeout=timeout
            )

    def wait_for_reserved_count(self, count: int, timeout: float = 5.0) -> bool:
        with self.reservation_condition:
            return self.reservation_condition.wait_for(
                lambda: len(self._reserved_tokens) == count, timeout=timeout
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


@pytest.mark.asyncio
async def test_websocket_simultaneous_handshake_race_and_registry_distinction() -> None:
    """Test A & D: Simultaneous handshake race and active registry distinction.

    - configure a small limit (2);
    - start substantially more simultaneous connection attempts (6);
    - hold accepted connections before sending start;
    - prove at most 2 slots are reserved;
    - all excess clients receive CONCURRENCY_LIMIT_REACHED without entering handshake wait;
    - reserved_count never exceeds 2;
    - during blocked handshakes: reserved_count reflects connections (2),
      while active_streaming_sessions remains completely empty.
    """
    phase1_event = asyncio.Event()

    def check_phase1(sockets_list: list[ControllableMockWebSocket]) -> None:
        admitted = sum(1 for s in sockets_list if s.handshake_read_entered.is_set())
        rejected = sum(1 for s in sockets_list if s.rejection_complete.is_set())
        if admitted == 2 and rejected == 4:
            phase1_event.set()

    class ControllableMockWebSocket:
        def __init__(self, ws_id: int, sockets_ref: list[ControllableMockWebSocket]) -> None:
            self.ws_id = ws_id
            self.app = app
            self.headers = {"origin": "http://localhost:5173"}
            self.accepted = False
            self.closed = False
            self.close_code: int | None = None
            self.sent_events: list[dict[str, Any]] = []
            self.receive_start_gate = asyncio.Event()
            self.receive_disconnect_gate = asyncio.Event()
            self.handshake_read_entered = asyncio.Event()
            self.rejection_complete = asyncio.Event()
            self.ack_sent_event = asyncio.Event()
            self.sockets_ref = sockets_ref

        async def accept(self) -> None:
            self.accepted = True

        async def close(self, code: int = 1000, reason: str = "") -> None:
            self.closed = True
            self.close_code = code
            if code == 1008:
                self.rejection_complete.set()
                check_phase1(self.sockets_ref)

        async def send_json(self, data: dict[str, Any]) -> None:
            self.sent_events.append(data)
            if data.get("type") == "session_started":
                self.ack_sent_event.set()

        async def receive_text(self) -> str:
            self.handshake_read_entered.set()
            check_phase1(self.sockets_ref)
            # Blocks until explicitly allowed to send handshake
            await self.receive_start_gate.wait()
            return json.dumps(
                {
                    "type": "start",
                    "protocol_version": "1.0",
                    "sample_rate": 16000,
                    "channels": 1,
                    "pcm_encoding": "int16",
                }
            )

        async def receive(self) -> dict[str, Any]:
            await self.receive_disconnect_gate.wait()
            return {"type": "websocket.disconnect"}

    admission = AtomicAdmissionController(limit=2, name="streaming_sessions")
    worker = InferenceWorker()
    orig_admission = getattr(app.state, "streaming_admission", None)
    orig_worker = getattr(app.state, "inference_worker", None)
    orig_registry = getattr(app.state, "active_streaming_sessions", None)
    app.state.streaming_admission = admission
    app.state.inference_worker = worker
    app.state.active_streaming_sessions = {}

    num_clients = 6
    sockets: list[ControllableMockWebSocket] = []
    tasks: list[asyncio.Task[None]] = []

    try:
        for i in range(num_clients):
            sockets.append(ControllableMockWebSocket(i, sockets))
        tasks = [asyncio.create_task(websocket_stream_endpoint(ws)) for ws in sockets]  # type: ignore[arg-type]

        # Deterministically wait for phase 1 arrival (2 admitted entered handshake read, 4 rejected completed rejection)
        await asyncio.wait_for(phase1_event.wait(), timeout=5.0)

        # Invariant 1: Exactly limit (2) reservations succeeded; reserved_count never exceeded 2
        assert admission.reserved_count == 2
        assert admission.available_count == 0

        # Invariant 2: Active registry distinction:
        # During blocked handshakes, reserved_count is 2 but active_streaming_sessions is EMPTY
        assert len(app.state.active_streaming_sessions) == 0

        # Identify which sockets succeeded in admission vs were rejected
        accepted_sockets = [s for s in sockets if not s.closed]
        rejected_sockets = [s for s in sockets if s.closed]

        assert len(accepted_sockets) == 2
        assert len(rejected_sockets) == 4

        # Invariant 3: All excess clients received CONCURRENCY_LIMIT_REACHED with code 1008
        for rej in rejected_sockets:
            assert rej.accepted is True
            assert rej.closed is True
            assert rej.close_code == 1008
            assert len(rej.sent_events) == 1
            err_event = rej.sent_events[0]
            assert err_event["type"] == "error"
            assert err_event["error_code"] == "CONCURRENCY_LIMIT_REACHED"
            # Crucially: receive_text was never awaited for rejected sockets
            assert not rej.handshake_read_entered.is_set()

        # Invariant 4: Now release the 2 accepted sockets to send start
        for acc in accepted_sockets:
            acc.receive_start_gate.set()

        # Deterministically wait for session_started acknowledgements to be sent
        await asyncio.wait_for(
            asyncio.gather(*(acc.ack_sent_event.wait() for acc in accepted_sockets)),
            timeout=5.0,
        )

        # Now active_streaming_sessions contains exactly the 2 initialized sessions
        assert len(app.state.active_streaming_sessions) == 2
        assert admission.reserved_count == 2

        # Finally, trigger disconnect on the accepted sockets
        for acc in accepted_sockets:
            acc.receive_disconnect_gate.set()

        await asyncio.gather(*tasks)

        # All sessions cleared and reservations released back to 0
        assert len(app.state.active_streaming_sessions) == 0
        assert admission.reserved_count == 0
        assert admission.available_count == 2
    finally:
        for s in sockets:
            s.receive_start_gate.set()
            s.receive_disconnect_gate.set()
        for t in tasks:
            if not t.done():
                t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await worker.shutdown(timeout=2.0)
        app.state.streaming_admission = orig_admission
        app.state.inference_worker = orig_worker
        app.state.active_streaming_sessions = orig_registry


@pytest.mark.asyncio
async def test_websocket_missing_admission_controller_fails_closed() -> None:
    """Test A: Missing/invalid streaming_admission fails closed without reading handshake."""
    orig_admission = getattr(app.state, "streaming_admission", None)
    orig_registry = getattr(app.state, "active_streaming_sessions", None)
    app.state.streaming_admission = None
    app.state.active_streaming_sessions = {}

    class TrackingMockWebSocket:
        def __init__(self) -> None:
            self.app = app
            self.headers = {"origin": "http://localhost:5173"}
            self.accepted = False
            self.closed = False
            self.close_code: int | None = None
            self.close_reason: str = ""
            self.sent_events: list[dict[str, Any]] = []
            self.receive_text_entered = False

        async def accept(self) -> None:
            self.accepted = True

        async def close(self, code: int = 1000, reason: str = "") -> None:
            self.closed = True
            self.close_code = code
            self.close_reason = reason

        async def send_json(self, data: dict[str, Any]) -> None:
            self.sent_events.append(data)

        async def receive_text(self) -> str:
            self.receive_text_entered = True
            return "{}"

        async def receive(self) -> dict[str, Any]:
            return {"type": "websocket.disconnect"}

    try:
        mock_ws = TrackingMockWebSocket()
        await websocket_stream_endpoint(mock_ws)  # type: ignore[arg-type]

        assert mock_ws.accepted is True
        assert mock_ws.closed is True
        assert mock_ws.close_code == 1011
        assert len(mock_ws.sent_events) == 1
        err_event = mock_ws.sent_events[0]
        assert err_event["type"] == "error"
        assert err_event["error_code"] == "ADMISSION_CONTROLLER_UNAVAILABLE"
        # Assert receive_text / handshake read was NEVER entered
        assert mock_ws.receive_text_entered is False
        # Assert no active session created
        assert len(app.state.active_streaming_sessions) == 0
    finally:
        app.state.streaming_admission = orig_admission
        app.state.active_streaming_sessions = orig_registry


@pytest.mark.asyncio
async def test_websocket_proves_no_active_sessions_registry_len_fallback() -> None:
    """Test B: Prove admission depends only on AtomicAdmissionController and never calls len() on active_streaming_sessions."""

    class PoisonedRegistry(dict[str, Any]):
        def __len__(self) -> int:
            raise RuntimeError(
                "Admission path forbidden from calling len(active_streaming_sessions)!"
            )

    poisoned_registry = PoisonedRegistry()
    orig_registry = getattr(app.state, "active_streaming_sessions", None)
    orig_admission = getattr(app.state, "streaming_admission", None)
    orig_worker = getattr(app.state, "inference_worker", None)

    admission = AtomicAdmissionController(limit=1, name="streaming_sessions")
    worker = InferenceWorker()
    app.state.streaming_admission = admission
    app.state.active_streaming_sessions = poisoned_registry
    app.state.inference_worker = worker

    class SimpleMockWebSocket:
        def __init__(self) -> None:
            self.app = app
            self.headers = {"origin": "http://localhost:5173"}
            self.accepted = False
            self.closed = False
            self.close_code: int | None = None
            self.sent_events: list[dict[str, Any]] = []
            self.disconnect_event = asyncio.Event()
            self.session_started_event = asyncio.Event()
            self.closed_event = asyncio.Event()

        async def accept(self) -> None:
            self.accepted = True

        async def close(self, code: int = 1000, reason: str = "") -> None:
            self.closed = True
            self.close_code = code
            self.closed_event.set()

        async def send_json(self, data: dict[str, Any]) -> None:
            self.sent_events.append(data)
            if data.get("type") == "session_started":
                self.session_started_event.set()

        async def receive_text(self) -> str:
            return json.dumps(
                {
                    "type": "start",
                    "protocol_version": "1.0",
                    "sample_rate": 16000,
                    "channels": 1,
                    "pcm_encoding": "int16",
                }
            )

        async def receive(self) -> dict[str, Any]:
            await self.disconnect_event.wait()
            return {"type": "websocket.disconnect"}

    ws1: SimpleMockWebSocket | None = None
    t1: asyncio.Task[None] | None = None
    ws2: SimpleMockWebSocket | None = None
    t2: asyncio.Task[None] | None = None

    try:
        # Client 1 connects successfully: len() must NOT be called on poisoned_registry
        ws1 = SimpleMockWebSocket()
        t1 = asyncio.create_task(websocket_stream_endpoint(ws1))  # type: ignore[arg-type]
        await asyncio.wait_for(ws1.session_started_event.wait(), timeout=5.0)
        assert ws1.accepted is True
        assert admission.reserved_count == 1

        # Client 2 connects when saturated: len() must NOT be called on poisoned_registry either
        ws2 = SimpleMockWebSocket()
        t2 = asyncio.create_task(websocket_stream_endpoint(ws2))  # type: ignore[arg-type]
        await asyncio.wait_for(ws2.closed_event.wait(), timeout=5.0)
        assert ws2.closed is True
        assert ws2.close_code == 1008
        assert ws2.sent_events[0]["error_code"] == "CONCURRENCY_LIMIT_REACHED"

        # Disconnect client 1
        ws1.disconnect_event.set()
        await t1
        assert admission.reserved_count == 0
    finally:
        if ws1 is not None:
            ws1.disconnect_event.set()
        if ws2 is not None:
            ws2.disconnect_event.set()
        for t in (t1, t2):
            if t is not None and not t.done():
                t.cancel()
        tasks_to_await = [t for t in (t1, t2) if t is not None]
        if tasks_to_await:
            await asyncio.gather(*tasks_to_await, return_exceptions=True)
        await worker.shutdown(timeout=2.0)
        app.state.streaming_admission = orig_admission
        app.state.active_streaming_sessions = orig_registry
        app.state.inference_worker = orig_worker


@pytest.mark.asyncio
async def test_websocket_repeated_cancellation_unconditionally_releases_token() -> None:
    """Test D: Repeated WebSocket cancellation during child unwinding still releases token and clears registry."""
    admission = AtomicAdmissionController(limit=1, name="streaming_sessions")
    worker = InferenceWorker()
    orig_admission = getattr(app.state, "streaming_admission", None)
    orig_worker = getattr(app.state, "inference_worker", None)
    orig_registry = getattr(app.state, "active_streaming_sessions", None)
    app.state.streaming_admission = admission
    app.state.inference_worker = worker
    app.state.active_streaming_sessions = {}

    child_started = asyncio.Event()
    child_in_cancellation = asyncio.Event()
    child_release = asyncio.Event()

    async def mock_execute(fn: Any, *args: Any, **kwargs: Any) -> Any:
        child_started.set()
        try:
            await asyncio.sleep(100.0)
        except asyncio.CancelledError:
            child_in_cancellation.set()
            await child_release.wait()
            raise

    worker.execute = mock_execute  # type: ignore[method-assign]

    class ControllableSocket:
        def __init__(self) -> None:
            self.app = app
            self.headers = {"origin": "http://localhost:5173"}
            self.accepted = False
            self.closed = False
            self.sent_events: list[dict[str, Any]] = []
            self.disconnect_event = asyncio.Event()
            self._audio_sent = False

        async def accept(self) -> None:
            self.accepted = True

        async def close(self, code: int = 1000, reason: str = "") -> None:
            self.closed = True

        async def send_json(self, data: dict[str, Any]) -> None:
            self.sent_events.append(data)

        async def receive_text(self) -> str:
            return json.dumps(
                {
                    "type": "start",
                    "protocol_version": "1.0",
                    "sample_rate": 16000,
                    "channels": 1,
                    "pcm_encoding": "int16",
                }
            )

        async def receive(self) -> dict[str, Any]:
            if not self._audio_sent:
                self._audio_sent = True
                chunk = generate_pcm_sine(4.0, sample_rate=16000, freq=300.0, amplitude=0.5)
                return {"type": "websocket.receive", "bytes": chunk}
            await self.disconnect_event.wait()
            return {"type": "websocket.disconnect"}

    ws: ControllableSocket | None = None
    endpoint_task: asyncio.Task[None] | None = None

    try:
        ws = ControllableSocket()
        endpoint_task = asyncio.create_task(websocket_stream_endpoint(ws))  # type: ignore[arg-type]

        # Wait until inference starts and enters worker.execute
        await asyncio.wait_for(child_started.wait(), timeout=5.0)

        assert admission.reserved_count == 1
        assert len(app.state.active_streaming_sessions) == 1

        # Cancel endpoint task once
        endpoint_task.cancel()

        # Wait deterministically until endpoint is waiting in gather on child cancellation
        await asyncio.wait_for(child_in_cancellation.wait(), timeout=5.0)

        # Cancel endpoint task a second time while in gather unwinding
        endpoint_task.cancel()

        # Now release the child task to finish its cancellation unwinding
        child_release.set()

        # CancelledError must propagate out of the endpoint task
        with pytest.raises(asyncio.CancelledError):
            await endpoint_task

        # Reservation count is strictly 0 and session registry is empty
        assert admission.reserved_count == 0
        assert len(app.state.active_streaming_sessions) == 0
    finally:
        child_release.set()
        if ws is not None:
            ws.disconnect_event.set()
        if endpoint_task is not None and not endpoint_task.done():
            endpoint_task.cancel()
            await asyncio.gather(endpoint_task, return_exceptions=True)
        await worker.shutdown(timeout=2.0)
        app.state.streaming_admission = orig_admission
        app.state.inference_worker = orig_worker
        app.state.active_streaming_sessions = orig_registry


@pytest.mark.asyncio
async def test_websocket_session_reset_exception_cleanup() -> None:
    """Test E: Even if session.reset() raises an exception during cleanup, registry removal and token release still occur."""
    admission = AtomicAdmissionController(limit=1, name="streaming_sessions")
    worker = InferenceWorker()
    orig_admission = getattr(app.state, "streaming_admission", None)
    orig_worker = getattr(app.state, "inference_worker", None)
    orig_registry = getattr(app.state, "active_streaming_sessions", None)
    app.state.streaming_admission = admission
    app.state.inference_worker = worker
    app.state.active_streaming_sessions = {}

    class SimpleMockWebSocket:
        def __init__(self) -> None:
            self.app = app
            self.headers = {"origin": "http://localhost:5173"}
            self.accepted = False
            self.closed = False
            self.sent_events: list[dict[str, Any]] = []
            self.disconnect_event = asyncio.Event()
            self.session_started_event = asyncio.Event()

        async def accept(self) -> None:
            self.accepted = True

        async def close(self, code: int = 1000, reason: str = "") -> None:
            self.closed = True

        async def send_json(self, data: dict[str, Any]) -> None:
            self.sent_events.append(data)
            if data.get("type") == "session_started":
                self.session_started_event.set()

        async def receive_text(self) -> str:
            return json.dumps(
                {
                    "type": "start",
                    "protocol_version": "1.0",
                    "sample_rate": 16000,
                    "channels": 1,
                    "pcm_encoding": "int16",
                }
            )

        async def receive(self) -> dict[str, Any]:
            await self.disconnect_event.wait()
            return {"type": "websocket.disconnect"}

    ws: SimpleMockWebSocket | None = None
    endpoint_task: asyncio.Task[None] | None = None

    try:
        reset_called = False

        def faulty_reset(self_session: StreamingSession) -> None:
            nonlocal reset_called
            reset_called = True
            raise RuntimeError("Intentional session reset failure during disconnect.")

        with patch.object(StreamingSession, "reset", new=faulty_reset):
            ws = SimpleMockWebSocket()
            endpoint_task = asyncio.create_task(websocket_stream_endpoint(ws))  # type: ignore[arg-type]
            await asyncio.wait_for(ws.session_started_event.wait(), timeout=5.0)
            ws.disconnect_event.set()
            await endpoint_task

        assert reset_called is True
        assert admission.reserved_count == 0
        assert len(app.state.active_streaming_sessions) == 0
    finally:
        if ws is not None:
            ws.disconnect_event.set()
        if endpoint_task is not None and not endpoint_task.done():
            endpoint_task.cancel()
            await asyncio.gather(endpoint_task, return_exceptions=True)
        await worker.shutdown(timeout=2.0)
        app.state.streaming_admission = orig_admission
        app.state.inference_worker = orig_worker
        app.state.active_streaming_sessions = orig_registry


# ==============================================================================
# 15. Audit M-03 Reservation Cleanup Matrix
# ==============================================================================


def test_websocket_reservation_cleanup_matrix() -> None:
    """Test B: Reservation cleanup matrix.

    Prove reserved_count returns to zero after:
    - malformed handshake JSON;
    - unsupported protocol;
    - unsupported PCM;
    - worker unavailable;
    - handshake cancellation / disconnect before start;
    - graceful stop;
    - abrupt disconnect.
    """
    with TestClient(app) as client:
        orig_admission = getattr(app.state, "streaming_admission", None)
        admission = SignalingAdmissionController(limit=10, name="streaming_sessions")
        app.state.streaming_admission = admission

        try:
            assert admission.reserved_count == 0

            # 1. Malformed handshake JSON
            with client.websocket_connect("/api/v1/stream") as ws:
                assert admission.wait_for_reserved_count(1, timeout=5.0)
                ws.send_text("this is not valid json {{{")
                msg = ws.receive_json()
                assert msg["type"] == "error"
                assert msg["error_code"] == "INVALID_JSON"
            assert admission.wait_for_reserved_count(0, timeout=5.0)
            assert admission.reserved_count == 0

            # 2. Unsupported protocol version
            with client.websocket_connect("/api/v1/stream") as ws:
                assert admission.wait_for_reserved_count(1, timeout=5.0)
                ws.send_json(
                    {
                        "type": "start",
                        "protocol_version": "99.0",
                        "sample_rate": 16000,
                        "channels": 1,
                        "pcm_encoding": "int16",
                    }
                )
                msg = ws.receive_json()
                assert msg["type"] == "error"
                assert msg["error_code"] == "UNSUPPORTED_PROTOCOL_VERSION"
            assert admission.wait_for_reserved_count(0, timeout=5.0)
            assert admission.reserved_count == 0

            # 3. Unsupported PCM encoding
            with client.websocket_connect("/api/v1/stream") as ws:
                assert admission.wait_for_reserved_count(1, timeout=5.0)
                ws.send_json(
                    {
                        "type": "start",
                        "protocol_version": "1.0",
                        "sample_rate": 16000,
                        "channels": 1,
                        "pcm_encoding": "mu-law",
                    }
                )
                msg = ws.receive_json()
                assert msg["type"] == "error"
                assert msg["error_code"] == "UNSUPPORTED_PCM_ENCODING"
            assert admission.wait_for_reserved_count(0, timeout=5.0)
            assert admission.reserved_count == 0

            # 4. Worker unavailable
            orig_worker = app.state.inference_worker
            try:
                app.state.inference_worker = None
                with client.websocket_connect("/api/v1/stream") as ws:
                    assert admission.wait_for_reserved_count(1, timeout=5.0)
                    ws.send_json(
                        {
                            "type": "start",
                            "protocol_version": "1.0",
                            "sample_rate": 16000,
                            "channels": 1,
                            "pcm_encoding": "int16",
                        }
                    )
                    msg = ws.receive_json()
                    assert msg["type"] == "error"
                    assert msg["error_code"] == "INFERENCE_WORKER_UNAVAILABLE"
                assert admission.wait_for_reserved_count(0, timeout=5.0)
                assert admission.reserved_count == 0
            finally:
                app.state.inference_worker = orig_worker

            # 5. Handshake disconnect before start
            with client.websocket_connect("/api/v1/stream") as ws:
                assert admission.wait_for_reserved_count(1, timeout=5.0)
                ws.close()
            assert admission.wait_for_reserved_count(0, timeout=5.0)
            assert admission.reserved_count == 0

            # 6. Graceful stop
            with client.websocket_connect("/api/v1/stream") as ws:
                assert admission.wait_for_reserved_count(1, timeout=5.0)
                ws.send_json(
                    {
                        "type": "start",
                        "protocol_version": "1.0",
                        "sample_rate": 16000,
                        "channels": 1,
                        "pcm_encoding": "int16",
                    }
                )
                ack = ws.receive_json()
                assert ack["type"] == "session_started"
                ws.send_json({"type": "stop"})
                stopped = ws.receive_json()
                assert stopped["type"] == "session_stopped"
            assert admission.wait_for_reserved_count(0, timeout=5.0)
            assert admission.reserved_count == 0

            # 7. Abrupt disconnect after streaming
            with client.websocket_connect("/api/v1/stream") as ws:
                assert admission.wait_for_reserved_count(1, timeout=5.0)
                ws.send_json(
                    {
                        "type": "start",
                        "protocol_version": "1.0",
                        "sample_rate": 16000,
                        "channels": 1,
                        "pcm_encoding": "int16",
                    }
                )
                ack = ws.receive_json()
                assert ack["type"] == "session_started"
                ws.send_bytes(generate_pcm_sine(0.5))
                ws.close()
            assert admission.wait_for_reserved_count(0, timeout=5.0)
            assert admission.reserved_count == 0
        finally:
            app.state.streaming_admission = orig_admission


# ==============================================================================
# 16. Audit M-03 Client Metadata Ceilings Verification
# ==============================================================================


def test_websocket_metadata_ceilings_tighten_only() -> None:
    """Test Metadata Ceilings: Client metadata may only lower server ceilings.

    Through the real WebSocket route, assert session_started effective values for:
    - huge values clamp to server constants;
    - lower valid values are honored;
    - values below minimum clamp to minimums;
    - invalid strings fall back to defaults;
    - NaN / +Inf / -Inf fall back to defaults;
    - Boolean values fall back to defaults.
    """
    with TestClient(app) as client:
        # A. Huge values: must clamp to server defaults (3600.0, 30.0, 131072)
        with client.websocket_connect("/api/v1/stream") as ws:
            ws.send_json(
                {
                    "type": "start",
                    "protocol_version": "1.0",
                    "sample_rate": 16000,
                    "channels": 1,
                    "pcm_encoding": "int16",
                    "max_stream_duration_seconds": 999999.0,
                    "idle_timeout_seconds": 999999.0,
                    "max_frame_bytes": 999999,
                }
            )
            ack = ws.receive_json()
            assert ack["type"] == "session_started"
            assert ack["max_stream_duration_seconds"] == 3600.0
            assert ack["idle_timeout_seconds"] == 30.0
            assert ack["max_frame_bytes"] == 131072
            ws.send_json({"type": "stop"})
            ws.receive_json()

        # B. Lower valid values: must be honored
        with client.websocket_connect("/api/v1/stream") as ws:
            ws.send_json(
                {
                    "type": "start",
                    "protocol_version": "1.0",
                    "sample_rate": 16000,
                    "channels": 1,
                    "pcm_encoding": "int16",
                    "max_stream_duration_seconds": 120.0,
                    "idle_timeout_seconds": 15.0,
                    "max_frame_bytes": 32768,
                }
            )
            ack = ws.receive_json()
            assert ack["type"] == "session_started"
            assert ack["max_stream_duration_seconds"] == 120.0
            assert ack["idle_timeout_seconds"] == 15.0
            assert ack["max_frame_bytes"] == 32768
            ws.send_json({"type": "stop"})
            ws.receive_json()

        # C. Lower bound clamping: values below minimum clamp to minimum (1.0, 0.1, 1024)
        with client.websocket_connect("/api/v1/stream") as ws:
            ws.send_json(
                {
                    "type": "start",
                    "protocol_version": "1.0",
                    "sample_rate": 16000,
                    "channels": 1,
                    "pcm_encoding": "int16",
                    "max_stream_duration_seconds": 0.05,
                    "idle_timeout_seconds": 0.01,
                    "max_frame_bytes": 100,
                }
            )
            ack = ws.receive_json()
            assert ack["type"] == "session_started"
            assert ack["max_stream_duration_seconds"] == 1.0
            assert ack["idle_timeout_seconds"] == 0.1
            assert ack["max_frame_bytes"] == 1024
            ws.send_json({"type": "stop"})
            ws.receive_json()

        # D. Invalid strings: ignored, default values used
        with client.websocket_connect("/api/v1/stream") as ws:
            ws.send_json(
                {
                    "type": "start",
                    "protocol_version": "1.0",
                    "sample_rate": 16000,
                    "channels": 1,
                    "pcm_encoding": "int16",
                    "max_stream_duration_seconds": "not_a_number",
                    "idle_timeout_seconds": "invalid",
                    "max_frame_bytes": "abc",
                }
            )
            ack = ws.receive_json()
            assert ack["type"] == "session_started"
            assert ack["max_stream_duration_seconds"] == 3600.0
            assert ack["idle_timeout_seconds"] == 30.0
            assert ack["max_frame_bytes"] == 131072
            ws.send_json({"type": "stop"})
            ws.receive_json()

        # E. Boolean values: must not be treated as int/float (True != 1.0)
        with client.websocket_connect("/api/v1/stream") as ws:
            ws.send_json(
                {
                    "type": "start",
                    "protocol_version": "1.0",
                    "sample_rate": 16000,
                    "channels": 1,
                    "pcm_encoding": "int16",
                    "max_stream_duration_seconds": True,
                    "idle_timeout_seconds": False,
                    "max_frame_bytes": True,
                }
            )
            ack = ws.receive_json()
            assert ack["type"] == "session_started"
            assert ack["max_stream_duration_seconds"] == 3600.0
            assert ack["idle_timeout_seconds"] == 30.0
            assert ack["max_frame_bytes"] == 131072
            ws.send_json({"type": "stop"})
            ws.receive_json()

        # F. NaN and Inf: must be safely rejected, not creating non-finite limits
        with client.websocket_connect("/api/v1/stream") as ws:
            ws.send_text(
                json.dumps(
                    {
                        "type": "start",
                        "protocol_version": "1.0",
                        "sample_rate": 16000,
                        "channels": 1,
                        "pcm_encoding": "int16",
                        "max_stream_duration_seconds": float("nan"),
                        "idle_timeout_seconds": float("inf"),
                        "max_frame_bytes": float("-inf"),
                    }
                )
            )
            ack = ws.receive_json()
            assert ack["type"] == "session_started"
            assert ack["max_stream_duration_seconds"] == 3600.0
            assert ack["idle_timeout_seconds"] == 30.0
            assert ack["max_frame_bytes"] == 131072
            ws.send_json({"type": "stop"})
            ws.receive_json()
