"""FastAPI WebSocket endpoint for real-time streaming audio synthetic speech detection."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, Literal

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status
from pydantic import BaseModel, Field

from voiceshield.core.schema import Disclaimers
from voiceshield.dsp.exceptions import InvalidAudioValueError, UnsupportedAudioError
from voiceshield.models.base import BaseDetector
from voiceshield.streaming.session import StreamingSession, WindowForInference

logger = logging.getLogger("voiceshield.api.stream")
router = APIRouter(prefix="/api/v1", tags=["Streaming"])

# Default operational constraints
MAX_FRAME_BYTES_DEFAULT = 131072  # 128 KB
MAX_STREAM_DURATION_SECONDS_DEFAULT = 3600.0  # 1 hour
DEFAULT_IDLE_TIMEOUT_SECONDS = 30.0
MAX_CONCURRENT_SESSIONS_DEFAULT = 10
SUPPORTED_PCM_ENCODINGS = {"int16", "int32", "float32"}
SUPPORTED_PROTOCOL_VERSIONS = {"1.0"}


# ------------------------------------------------------------------------------
# Protocol Schemas
# ------------------------------------------------------------------------------


class ClientStartMessage(BaseModel):
    """Initial client handshake message declaring stream parameters."""

    type: Literal["start"] = "start"
    protocol_version: str = Field(default="1.0", description="Protocol version (must be 1.0).")
    sample_rate: int = Field(ge=8000, le=192000, description="Input sampling rate in Hz.")
    channels: int = Field(default=1, ge=1, le=2, description="Channel count (1=mono, 2=stereo).")
    pcm_encoding: str = Field(default="int16", description="PCM format: int16, int32, or float32.")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Optional stream metadata or timeout overrides."
    )


class ClientStopMessage(BaseModel):
    """Client request to terminate stream gracefully."""

    type: Literal["stop"] = "stop"


class ServerStartAck(BaseModel):
    """Server handshake response acknowledging accepted stream configuration."""

    type: str = "session_started"
    session_id: str
    protocol_version: str = "1.0"
    canonical_sample_rate: int = 16000
    canonical_channels: int = 1
    window_seconds: float = 4.0
    hop_seconds: float = 1.0
    min_speech_seconds: float = 0.8
    max_frame_bytes: int = MAX_FRAME_BYTES_DEFAULT
    max_stream_duration_seconds: float = MAX_STREAM_DURATION_SECONDS_DEFAULT
    idle_timeout_seconds: float = DEFAULT_IDLE_TIMEOUT_SECONDS
    is_mock_detector: bool = False
    disclaimers: Disclaimers = Field(default_factory=Disclaimers)


class ServerQualityEvent(BaseModel):
    """Real-time signal quality status updates."""

    type: str = "quality_update"
    session_id: str
    timestamp: float
    is_clipped: bool
    is_silent: bool
    has_timing_gap: bool
    clipping_count: int
    gap_count: int


class ServerWindowResult(BaseModel):
    """Acoustic anomaly evaluation and temporal policy decision for a 4-second window."""

    type: str = "window_result"
    session_id: str
    window_index: int
    window_start_sec: float
    window_end_sec: float
    speech_seconds: float
    is_eligible: bool
    synthetic_score: float | None = None
    uncalibrated_synthetic_score: float | None = None
    score_is_calibrated: bool = False
    operating_threshold: float = 0.65
    operating_threshold_source: str = "unvalidated_heuristic"
    raw_score: float | None = None
    calibrated_score: float | None = None
    decision: dict[str, Any]


class ServerStopAck(BaseModel):
    """Server confirmation of graceful session termination."""

    type: str = "session_stopped"
    session_id: str
    total_duration_seconds: float
    total_windows_evaluated: int


class ServerErrorEvent(BaseModel):
    """Structured error message sent before abnormal closure."""

    type: str = "error"
    error_code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


def is_origin_allowed(origin: str | None, allowed_origins: list[str]) -> bool:
    """Verify incoming WebSocket Origin header against configured origin policy."""
    if not allowed_origins or "*" in allowed_origins:
        return True
    if origin is None:
        # Allow requests with no Origin header (native clients, curl, test clients)
        return True
    normalized_origin = origin.strip().rstrip("/")
    for allowed in allowed_origins:
        if allowed.strip().rstrip("/") == normalized_origin:
            return True
    return False


# ------------------------------------------------------------------------------
# WebSocket Streaming Endpoint
# ------------------------------------------------------------------------------


@router.websocket("/stream")
async def websocket_stream_endpoint(websocket: WebSocket) -> None:
    """Real-time bidirectional WebSocket streaming endpoint for synthetic speech detection.

    Strict protocol:
    1. Client connects and sends JSON start message.
    2. Server validates parameters, generates session_id, and returns session_started.
    3. Client streams binary PCM audio frames.
    4. Server runs non-blocking DSP, VAD, and asynchronous model inference.
    5. Server emits real-time JSON events for quality flags, window results, and temporal decisions.
    6. Client sends JSON stop or disconnects; all session resources are released cleanly.
    """
    app = websocket.app

    # 1. Origin Policy Enforcement
    allowed_origins = getattr(app.state, "allowed_origins", ["*"])
    client_origin = websocket.headers.get("origin")
    if not is_origin_allowed(client_origin, allowed_origins):
        logger.warning("Rejected WebSocket connection from unauthorized origin: %s", client_origin)
        await websocket.close(
            code=status.WS_1008_POLICY_VIOLATION,
            reason="Origin not allowed by VoiceShield CORS policy.",
        )
        return

    # 2. Concurrency Limit Check
    active_sessions: dict[str, Any] = getattr(app.state, "active_streaming_sessions", {})
    max_concurrent = getattr(app.state, "max_concurrent_sessions", MAX_CONCURRENT_SESSIONS_DEFAULT)
    if len(active_sessions) >= max_concurrent:
        logger.warning(
            "Rejected WebSocket connection: active sessions (%d) reached limit (%d).",
            len(active_sessions),
            max_concurrent,
        )
        await websocket.accept()
        await websocket.send_json(
            ServerErrorEvent(
                error_code="CONCURRENCY_LIMIT_REACHED",
                message=(
                    f"Server concurrency limit ({max_concurrent} sessions) reached. "
                    "Please retry when existing streams disconnect."
                ),
            ).model_dump()
        )
        await websocket.close(
            code=status.WS_1008_POLICY_VIOLATION,
            reason="Concurrency limit reached.",
        )
        return

    await websocket.accept()

    session_id = str(uuid.uuid4())
    session: StreamingSession | None = None
    inference_tasks: set[asyncio.Task[None]] = set()

    # Timeouts and limits
    idle_timeout = DEFAULT_IDLE_TIMEOUT_SECONDS
    max_frame_bytes = MAX_FRAME_BYTES_DEFAULT
    max_stream_duration = MAX_STREAM_DURATION_SECONDS_DEFAULT

    try:
        # 3. Protocol Handshake: Expect JSON start message
        try:
            handshake_raw = await asyncio.wait_for(
                websocket.receive_text(),
                timeout=DEFAULT_IDLE_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            await websocket.send_json(
                ServerErrorEvent(
                    error_code="HANDSHAKE_TIMEOUT",
                    message="Timed out waiting for initial JSON start handshake message.",
                ).model_dump()
            )
            await websocket.close(
                code=status.WS_1008_POLICY_VIOLATION,
                reason="Handshake timeout.",
            )
            return
        except WebSocketDisconnect:
            return

        try:
            handshake_dict = json.loads(handshake_raw)
        except Exception:
            await websocket.send_json(
                ServerErrorEvent(
                    error_code="INVALID_JSON",
                    message="Handshake payload must be valid JSON text.",
                ).model_dump()
            )
            await websocket.close(
                code=status.WS_1003_UNSUPPORTED_DATA,
                reason="Malformed JSON handshake.",
            )
            return

        if handshake_dict.get("type") != "start":
            await websocket.send_json(
                ServerErrorEvent(
                    error_code="INVALID_HANDSHAKE",
                    message=f"First message must be type 'start', got {handshake_dict.get('type')!r}.",
                ).model_dump()
            )
            await websocket.close(
                code=status.WS_1008_POLICY_VIOLATION,
                reason="Invalid handshake message type.",
            )
            return

        try:
            start_msg = ClientStartMessage.model_validate(handshake_dict)
        except Exception as exc:
            await websocket.send_json(
                ServerErrorEvent(
                    error_code="INVALID_STREAM_CONFIG",
                    message=f"Invalid stream parameters: {exc}",
                ).model_dump()
            )
            await websocket.close(
                code=status.WS_1008_POLICY_VIOLATION,
                reason="Invalid stream parameters.",
            )
            return

        # Validate protocol version
        if start_msg.protocol_version not in SUPPORTED_PROTOCOL_VERSIONS:
            await websocket.send_json(
                ServerErrorEvent(
                    error_code="UNSUPPORTED_PROTOCOL_VERSION",
                    message=(
                        f"Protocol version '{start_msg.protocol_version}' is not supported. "
                        f"Supported versions: {sorted(SUPPORTED_PROTOCOL_VERSIONS)}."
                    ),
                ).model_dump()
            )
            await websocket.close(
                code=status.WS_1008_POLICY_VIOLATION,
                reason="Unsupported protocol version.",
            )
            return

        # Validate PCM encoding
        pcm_enc = start_msg.pcm_encoding.lower()
        if pcm_enc not in SUPPORTED_PCM_ENCODINGS:
            await websocket.send_json(
                ServerErrorEvent(
                    error_code="UNSUPPORTED_PCM_ENCODING",
                    message=(
                        f"PCM encoding '{start_msg.pcm_encoding}' is not supported. "
                        f"Supported encodings: {sorted(SUPPORTED_PCM_ENCODINGS)}."
                    ),
                ).model_dump()
            )
            await websocket.close(
                code=status.WS_1008_POLICY_VIOLATION,
                reason="Unsupported PCM encoding.",
            )
            return

        # Apply any metadata timeout / constraint overrides (clamped safely)
        if "idle_timeout_seconds" in start_msg.metadata:
            try:
                custom_timeout = float(start_msg.metadata["idle_timeout_seconds"])
                idle_timeout = max(0.1, min(custom_timeout, 300.0))
            except (ValueError, TypeError):
                pass

        if "max_frame_bytes" in start_msg.metadata:
            try:
                custom_max_frame = int(start_msg.metadata["max_frame_bytes"])
                max_frame_bytes = max(1024, min(custom_max_frame, MAX_FRAME_BYTES_DEFAULT))
            except (ValueError, TypeError):
                pass

        if "max_stream_duration_seconds" in start_msg.metadata:
            try:
                custom_duration = float(start_msg.metadata["max_stream_duration_seconds"])
                max_stream_duration = max(1.0, min(custom_duration, 86400.0))
            except (ValueError, TypeError):
                pass

        # 4. Initialize isolated StreamingSession
        session = StreamingSession(session_id=session_id)
        active_sessions[session_id] = session

        logger.info(
            "Session %s initialized: %d Hz, %d channel(s), %s encoding (idle_timeout=%.1fs)",
            session_id,
            start_msg.sample_rate,
            start_msg.channels,
            pcm_enc,
            idle_timeout,
        )

        # 5. Check model detector readiness
        detector: BaseDetector | None = getattr(app.state, "detector", None)
        if detector is None:
            load_error = getattr(
                app.state,
                "model_load_error",
                "VoiceShield detector model is not initialized or checkpoint is missing.",
            )
            logger.warning("Rejected WebSocket stream: detector not initialized: %s", load_error)
            await websocket.send_json(
                ServerErrorEvent(
                    error_code="DETECTOR_NOT_READY",
                    message=load_error,
                ).model_dump()
            )
            await websocket.close(
                code=status.WS_1011_INTERNAL_ERROR,
                reason="Detector model not initialized.",
            )
            return

        is_mock = getattr(app.state, "is_mock_detector", False)

        # 6. Send start acknowledgement
        start_ack = ServerStartAck(
            session_id=session_id,
            protocol_version="1.0",
            canonical_sample_rate=session.sample_rate,
            canonical_channels=1,
            window_seconds=session.config.windowing.window_seconds,
            hop_seconds=session.config.windowing.hop_seconds,
            min_speech_seconds=session.config.windowing.min_speech_seconds,
            max_frame_bytes=max_frame_bytes,
            max_stream_duration_seconds=max_stream_duration,
            idle_timeout_seconds=idle_timeout,
            is_mock_detector=is_mock,
        )
        await websocket.send_json(start_ack.model_dump())

        semaphore: asyncio.Semaphore | None = getattr(app.state, "inference_semaphore", None)

        # 6. Audio Ingestion Loop
        stream_start_time = asyncio.get_event_loop().time()
        sample_rate = start_msg.sample_rate
        channels = start_msg.channels
        total_windows_evaluated = 0

        async def _dispatch_inference(win: WindowForInference) -> None:
            nonlocal total_windows_evaluated
            try:
                if detector is not None:
                    # Model forward pass safely offloaded from the event loop
                    if semaphore is not None:
                        async with semaphore:
                            output = await asyncio.to_thread(detector.predict, win.samples)
                    else:
                        output = await asyncio.to_thread(detector.predict, win.samples)

                    synth_score = float(output.synthetic_score)
                else:
                    # Model fallback simulation
                    synth_score = 0.5

                assert session is not None
                decision = session.complete_inference(
                    synthetic_score=synth_score,
                    window=win,
                    calibrated_score=None,
                    score_is_calibrated=False,
                )
                total_windows_evaluated += 1

                result_event = ServerWindowResult(
                    session_id=session_id,
                    window_index=win.window_index,
                    window_start_sec=win.start_time,
                    window_end_sec=win.end_time,
                    speech_seconds=win.speech_seconds,
                    is_eligible=True,
                    synthetic_score=synth_score,
                    uncalibrated_synthetic_score=synth_score,
                    score_is_calibrated=False,
                    operating_threshold=0.65,
                    operating_threshold_source="unvalidated_heuristic",
                    raw_score=synth_score,
                    calibrated_score=None,
                    decision=decision.model_dump(),
                )
                await websocket.send_json(result_event.model_dump())

            except Exception as exc:
                logger.error("Inference execution failed in session %s: %s", session_id, exc)
                if session is not None:
                    session._in_flight_inference = False
                try:
                    await websocket.send_json(
                        ServerErrorEvent(
                            error_code="MODEL_INFERENCE_ERROR",
                            message=f"Model evaluation failed: {exc}",
                        ).model_dump()
                    )
                except Exception:
                    pass

            finally:
                # If a fresher pending window was queued while inference was running, pop and process it
                if session is not None and session.has_pending_window:
                    next_win = session.pop_pending_window()
                    if next_win is not None:
                        task = asyncio.create_task(_dispatch_inference(next_win))
                        inference_tasks.add(task)
                        task.add_done_callback(inference_tasks.discard)

        while True:
            try:
                message = await asyncio.wait_for(
                    websocket.receive(),
                    timeout=idle_timeout,
                )
            except TimeoutError:
                logger.info(
                    "Session %s idle timeout exceeded (%.1fs of inactivity).",
                    session_id,
                    idle_timeout,
                )
                await websocket.send_json(
                    ServerErrorEvent(
                        error_code="IDLE_TIMEOUT",
                        message=f"Session closed after {idle_timeout:.1f}s of inactivity.",
                    ).model_dump()
                )
                await websocket.close(
                    code=status.WS_1000_NORMAL_CLOSURE,
                    reason="Idle timeout exceeded.",
                )
                break

            msg_type = message.get("type")
            if msg_type == "websocket.disconnect":
                break

            # Handle text messages (e.g., stop request)
            if "text" in message:
                text_content = message["text"]
                try:
                    cmd_data = json.loads(text_content)
                except Exception:
                    await websocket.send_json(
                        ServerErrorEvent(
                            error_code="UNEXPECTED_MESSAGE",
                            message="Unexpected non-JSON text message during audio streaming.",
                        ).model_dump()
                    )
                    continue

                if cmd_data.get("type") == "stop":
                    logger.info("Session %s received stop command from client.", session_id)
                    elapsed = asyncio.get_event_loop().time() - stream_start_time
                    await websocket.send_json(
                        ServerStopAck(
                            session_id=session_id,
                            total_duration_seconds=float(elapsed),
                            total_windows_evaluated=total_windows_evaluated,
                        ).model_dump()
                    )
                    await websocket.close(
                        code=status.WS_1000_NORMAL_CLOSURE,
                        reason="Client requested stop.",
                    )
                    break
                else:
                    await websocket.send_json(
                        ServerErrorEvent(
                            error_code="UNKNOWN_COMMAND",
                            message=f"Unknown text command: {cmd_data.get('type')!r}.",
                        ).model_dump()
                    )
                    continue

            # Handle binary audio frame
            if "bytes" in message:
                pcm_bytes: bytes = message["bytes"]

                # 1. Enforce maximum frame size
                if len(pcm_bytes) > max_frame_bytes:
                    logger.warning(
                        "Session %s sent frame of %d bytes, exceeding maximum %d bytes.",
                        session_id,
                        len(pcm_bytes),
                        max_frame_bytes,
                    )
                    await websocket.send_json(
                        ServerErrorEvent(
                            error_code="PAYLOAD_TOO_LARGE",
                            message=(
                                f"Audio frame size ({len(pcm_bytes)} bytes) exceeds "
                                f"maximum permitted limit ({max_frame_bytes} bytes)."
                            ),
                        ).model_dump()
                    )
                    await websocket.close(
                        code=status.WS_1009_MESSAGE_TOO_BIG,
                        reason="Frame size exceeds maximum permitted limit.",
                    )
                    break

                # 2. Enforce maximum stream duration
                elapsed_stream = asyncio.get_event_loop().time() - stream_start_time
                if elapsed_stream > max_stream_duration:
                    logger.info(
                        "Session %s reached maximum stream duration (%.1fs).",
                        session_id,
                        max_stream_duration,
                    )
                    await websocket.send_json(
                        ServerErrorEvent(
                            error_code="STREAM_DURATION_EXCEEDED",
                            message=f"Maximum stream duration ({max_stream_duration}s) exceeded.",
                        ).model_dump()
                    )
                    await websocket.close(
                        code=status.WS_1008_POLICY_VIOLATION,
                        reason="Stream duration limit reached.",
                    )
                    break

                # 3. Synchronous, fast DSP ingestion into ring buffer
                try:
                    ingest_res = session.ingest_pcm(
                        pcm_data=pcm_bytes,
                        sample_rate=sample_rate,
                        channels=channels,
                        timestamp=elapsed_stream,
                        dtype=pcm_enc,
                    )
                except (InvalidAudioValueError, UnsupportedAudioError) as exc:
                    logger.warning("Session %s corrupt audio frame: %s", session_id, exc)
                    await websocket.send_json(
                        ServerErrorEvent(
                            error_code="AUDIO_DECODE_ERROR",
                            message=f"Invalid audio frame data: {exc}",
                        ).model_dump()
                    )
                    continue

                # 4. Emit quality updates if clipping or timing gap detected
                if (
                    ingest_res.quality_flags.is_clipped
                    or ingest_res.quality_flags.has_timing_gap
                    or ingest_res.quality_flags.is_silent
                ):
                    quality_event = ServerQualityEvent(
                        session_id=session_id,
                        timestamp=elapsed_stream,
                        is_clipped=ingest_res.quality_flags.is_clipped,
                        is_silent=ingest_res.quality_flags.is_silent,
                        has_timing_gap=ingest_res.quality_flags.has_timing_gap,
                        clipping_count=ingest_res.quality_flags.clipping_count,
                        gap_count=ingest_res.quality_flags.gap_count,
                    )
                    await websocket.send_json(quality_event.model_dump())

                # 5. Handle immediate abstention decision (insufficient speech)
                if ingest_res.immediate_decision is not None:
                    total_windows_evaluated += 1
                    imm = ingest_res.immediate_decision
                    abstained_event = ServerWindowResult(
                        session_id=session_id,
                        window_index=imm.window_index,
                        window_start_sec=imm.window_start_sec,
                        window_end_sec=imm.window_end_sec,
                        speech_seconds=imm.speech_seconds,
                        is_eligible=False,
                        synthetic_score=None,
                        uncalibrated_synthetic_score=None,
                        score_is_calibrated=False,
                        operating_threshold=0.65,
                        operating_threshold_source="unvalidated_heuristic",
                        raw_score=None,
                        calibrated_score=None,
                        decision=imm.model_dump(),
                    )
                    await websocket.send_json(abstained_event.model_dump())

                # 6. Dispatch eligible window for inference if worker is idle
                if session.has_pending_window and not session.in_flight_inference:
                    win = session.pop_pending_window()
                    if win is not None:
                        task = asyncio.create_task(_dispatch_inference(win))
                        inference_tasks.add(task)
                        task.add_done_callback(inference_tasks.discard)

    except WebSocketDisconnect:
        logger.info("Session %s client disconnected.", session_id)
    except Exception as exc:
        logger.exception("Unexpected error in streaming session %s: %s", session_id, exc)
    finally:
        # Cancel any pending inference background tasks
        for task in inference_tasks:
            if not task.done():
                task.cancel()

        # Cleanly release session resources and remove from registry
        if session is not None:
            session.reset()
        active_sessions.pop(session_id, None)

        logger.info("Session %s resources released successfully.", session_id)
