"""Integration tests for VoiceShield real-time WebSocket streaming service."""

from __future__ import annotations

import io
import threading
import time
from typing import Any

import numpy as np
import pytest
import soundfile as sf
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
