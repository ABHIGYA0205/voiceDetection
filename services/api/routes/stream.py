"""FastAPI WebSocket endpoint for real-time streaming audio synthetic speech detection."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import uuid
from typing import Any, Literal

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status
from pydantic import BaseModel, Field

from services.api.inference_worker import InferenceWorker, InferenceWorkerClosedError
from services.api.resource_admission import (
    AtomicAdmissionController,
)
from voiceshield.config.schema import VoiceShieldConfig
from voiceshield.core.schema import Disclaimers
from voiceshield.dsp.exceptions import AudioPreprocessingError
from voiceshield.models.base import BaseDetector
from voiceshield.streaming.policy import ReasonCode
from voiceshield.streaming.session import StreamingSession, WindowForInference

logger = logging.getLogger("voiceshield.api.stream")
router = APIRouter(prefix="/api/v1", tags=["Streaming"])

# Default operational constraints
MAX_FRAME_BYTES_DEFAULT = 131072  # 128 KB
MAX_STREAM_DURATION_SECONDS_DEFAULT = 3600.0  # 1 hour
DEFAULT_IDLE_TIMEOUT_SECONDS = 30.0
SUPPORTED_PCM_ENCODINGS = {"int16", "int32", "float32"}
SUPPORTED_PROTOCOL_VERSIONS = {"1.0"}


def _parse_clamped_numeric(
    val: Any,
    min_val: float,
    max_ceiling: float,
    as_int: bool = False,
) -> float | int | None:
    """Parse and clamp numeric metadata, ensuring it only tightens server ceilings.

    Rejects bool, non-numeric, NaN, and infinite values.
    Returns clamped value between [min_val, max_ceiling], or None if invalid.
    """
    if isinstance(val, bool):
        return None
    try:
        num = float(val)
    except (ValueError, TypeError):
        return None

    if not math.isfinite(num):
        return None

    clamped = max(min_val, min(num, max_ceiling))
    return int(clamped) if as_int else clamped


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
    config_hash: str | None = None
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
    total_windows_generated: int = 0
    total_windows_evaluated: int = 0
    total_windows_dropped: int = 0


class ServerContinuityResetEvent(BaseModel):
    """Event emitted on timing discontinuity when continuity segment is reset."""

    type: str = "continuity_reset"
    session_id: str
    reason: str = ReasonCode.INSUFFICIENT_CONTIGUOUS_AUDIO.value
    timestamp_delta: float
    gap_count: int
    continuity_segment_id: int
    timestamp: float


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
    total_windows_generated: int = 0
    total_windows_dropped: int = 0
    status: str = "success"
    final_reason: str | None = None


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

    # 2. Atomic Session Admission Reservation
    session_id = str(uuid.uuid4())
    token = session_id

    streaming_admission: AtomicAdmissionController | None = getattr(
        app.state, "streaming_admission", None
    )

    if streaming_admission is None or not isinstance(
        streaming_admission, AtomicAdmissionController
    ):
        logger.error(
            "Streaming admission controller is not configured or unavailable on app.state."
        )
        await websocket.accept()
        await websocket.send_json(
            ServerErrorEvent(
                error_code="ADMISSION_CONTROLLER_UNAVAILABLE",
                message="Streaming admission infrastructure is unavailable. Please retry later.",
            ).model_dump()
        )
        await websocket.close(
            code=status.WS_1011_INTERNAL_ERROR,
            reason="Admission controller unavailable.",
        )
        return

    slot_reserved = streaming_admission.try_reserve(token)
    if not slot_reserved:
        logger.warning(
            "Rejected WebSocket connection: streaming admission limit (%d) reached.",
            streaming_admission.limit,
        )
        await websocket.accept()
        await websocket.send_json(
            ServerErrorEvent(
                error_code="CONCURRENCY_LIMIT_REACHED",
                message=(
                    f"Server concurrency limit ({streaming_admission.limit} sessions) reached. "
                    "Please retry when existing streams disconnect."
                ),
            ).model_dump()
        )
        await websocket.close(
            code=status.WS_1008_POLICY_VIOLATION,
            reason="Concurrency limit reached.",
        )
        return

    active_sessions: dict[str, Any] | None = getattr(app.state, "active_streaming_sessions", None)
    session: StreamingSession | None = None
    inference_tasks: set[asyncio.Task[None]] = set()
    is_closing = False

    # Timeouts and limits
    idle_timeout = DEFAULT_IDLE_TIMEOUT_SECONDS
    max_frame_bytes = MAX_FRAME_BYTES_DEFAULT
    max_stream_duration = MAX_STREAM_DURATION_SECONDS_DEFAULT

    try:
        await websocket.accept()
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

        # Apply any metadata timeout / constraint overrides (clamped safely; may only tighten server ceilings)
        merged_meta: dict[str, Any] = dict(start_msg.metadata)
        for meta_k in ("idle_timeout_seconds", "max_frame_bytes", "max_stream_duration_seconds"):
            if meta_k in handshake_dict and meta_k not in merged_meta:
                merged_meta[meta_k] = handshake_dict[meta_k]

        if "idle_timeout_seconds" in merged_meta:
            custom_timeout = _parse_clamped_numeric(
                merged_meta["idle_timeout_seconds"],
                min_val=0.1,
                max_ceiling=DEFAULT_IDLE_TIMEOUT_SECONDS,
            )
            if custom_timeout is not None:
                idle_timeout = float(custom_timeout)

        if "max_frame_bytes" in merged_meta:
            custom_max_frame = _parse_clamped_numeric(
                merged_meta["max_frame_bytes"],
                min_val=1024,
                max_ceiling=float(MAX_FRAME_BYTES_DEFAULT),
                as_int=True,
            )
            if custom_max_frame is not None:
                max_frame_bytes = int(custom_max_frame)

        if "max_stream_duration_seconds" in merged_meta:
            custom_duration = _parse_clamped_numeric(
                merged_meta["max_stream_duration_seconds"],
                min_val=1.0,
                max_ceiling=MAX_STREAM_DURATION_SECONDS_DEFAULT,
            )
            if custom_duration is not None:
                max_stream_duration = float(custom_duration)

        # 4. Check model detector readiness
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

        # Check inference worker readiness before session acknowledgement
        worker: InferenceWorker | None = getattr(app.state, "inference_worker", None)
        if worker is None or not worker.accepting_jobs:
            logger.warning(
                "Rejected WebSocket stream: inference worker unavailable or not accepting jobs."
            )
            await websocket.send_json(
                ServerErrorEvent(
                    error_code="INFERENCE_WORKER_UNAVAILABLE",
                    message="VoiceShield inference worker is not initialized or not accepting jobs.",
                ).model_dump()
            )
            await websocket.close(
                code=status.WS_1011_INTERNAL_ERROR,
                reason="Inference worker unavailable.",
            )
            return

        # Check effective config readiness and integrity
        effective_config: VoiceShieldConfig | None = getattr(app.state, "effective_config", None)
        effective_config_hash: str | None = getattr(app.state, "effective_config_hash", None)
        if (
            effective_config is None
            or not isinstance(effective_config, VoiceShieldConfig)
            or effective_config_hash is None
            or effective_config.compute_hash() != effective_config_hash
        ):
            logger.error(
                "Rejected WebSocket stream: effective config or hash unavailable/mismatched on app.state."
            )
            await websocket.send_json(
                ServerErrorEvent(
                    error_code="CONFIG_UNAVAILABLE",
                    message="VoiceShield configuration is unavailable or not initialized.",
                ).model_dump()
            )
            await websocket.close(
                code=status.WS_1011_INTERNAL_ERROR,
                reason="Server configuration unavailable.",
            )
            return

        # 5. Initialize isolated StreamingSession with effective config
        session = StreamingSession(config=effective_config, session_id=session_id)
        if active_sessions is not None:
            active_sessions[session_id] = session

        logger.info(
            "Session %s initialized: %d Hz, %d channel(s), %s encoding (idle_timeout=%.1fs)",
            session_id,
            start_msg.sample_rate,
            start_msg.channels,
            pcm_enc,
            idle_timeout,
        )

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
            config_hash=effective_config_hash,
        )
        await websocket.send_json(start_ack.model_dump())

        # 6. Audio Ingestion Loop
        stream_start_time = asyncio.get_event_loop().time()
        sample_rate = start_msg.sample_rate
        channels = start_msg.channels
        total_windows_evaluated = 0
        total_eligible_evaluated = 0
        had_model_inference_error = False
        audio_timeline_seconds = 0.0

        async def _dispatch_inference(win: WindowForInference) -> None:
            nonlocal total_windows_evaluated, total_eligible_evaluated, had_model_inference_error
            try:
                if worker is None:
                    raise RuntimeError("VoiceShield inference worker is not initialized.")
                if not worker.accepting_jobs:
                    raise InferenceWorkerClosedError("VoiceShield inference worker is closed.")
                if detector is None:
                    raise RuntimeError("VoiceShield detector is not initialized.")

                output = await worker.execute(detector.predict, win.samples)
                synth_score = float(output.synthetic_score)

                if is_closing:
                    logger.info("Discarding inference result for closing session %s", session_id)
                    return

                assert session is not None
                decision = session.complete_inference(
                    synthetic_score=synth_score,
                    window=win,
                    calibrated_score=None,
                    score_is_calibrated=False,
                )
                if decision is None:
                    logger.info(
                        "Suppressed stale pre-gap inference result for session %s (win segment %d != session segment %d)",
                        session_id,
                        win.continuity_segment_id,
                        session.continuity_segment_id,
                    )
                    return

                total_windows_evaluated += 1
                total_eligible_evaluated += 1

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
                    operating_threshold=session.config.temporal_policy.threshold_high,
                    operating_threshold_source="unvalidated_heuristic",
                    raw_score=synth_score,
                    calibrated_score=None,
                    decision=decision.model_dump(),
                )
                if not is_closing:
                    await websocket.send_json(result_event.model_dump())

            except asyncio.CancelledError:
                # Disconnect cancellation is lifecycle cancellation, not a neural model failure.
                logger.info(
                    "Inference coroutine cancelled for session %s (window %d)",
                    session_id,
                    win.window_index,
                )
                if session is not None:
                    session._in_flight_inference = False
                    session._in_flight_window = None
                raise

            except Exception as exc:
                had_model_inference_error = True
                logger.error("Inference execution failed in session %s: %s", session_id, exc)
                if session is not None:
                    session._in_flight_inference = False
                    session._in_flight_window = None
                if not is_closing:
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
                # If a fresher pending window was queued while inference was running, pop and process it,
                # unless the session is closing/disconnected.
                if (
                    not is_closing
                    and session is not None
                    and session.has_pending_window
                    and not session.in_flight_inference
                ):
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
                    assert session is not None
                    new_eligible, new_abstentions = session.finalize()

                    # 1. Send immediate abstention events for any window completed during flush
                    for imm in new_abstentions:
                        total_windows_evaluated += 1
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
                            operating_threshold=session.config.temporal_policy.threshold_high,
                            operating_threshold_source="unvalidated_heuristic",
                            raw_score=None,
                            calibrated_score=None,
                            decision=imm.model_dump(),
                        )
                        await websocket.send_json(abstained_event.model_dump())

                    # 2. If the worker is idle and an eligible window is pending (e.g. from flush), dispatch it
                    if session.has_pending_window and not session.in_flight_inference:
                        pending_win = session.pop_pending_window()
                        if pending_win is not None:
                            task = asyncio.create_task(_dispatch_inference(pending_win))
                            inference_tasks.add(task)
                            task.add_done_callback(inference_tasks.discard)

                    # 3. Wait for all in-flight and dynamically launched pending tasks to finish
                    while inference_tasks:
                        await asyncio.gather(*list(inference_tasks), return_exceptions=True)

                    elapsed = asyncio.get_event_loop().time() - stream_start_time

                    if had_model_inference_error:
                        final_status = "error"
                        final_reason = "model_inference_error"
                    elif session.total_windows_generated == 0:
                        final_status = "insufficient_evidence"
                        final_reason = ReasonCode.INSUFFICIENT_CONTIGUOUS_AUDIO.value
                    elif total_eligible_evaluated > 0:
                        final_status = "success"
                        final_reason = None
                    else:
                        final_status = "insufficient_evidence"
                        final_reason = ReasonCode.INSUFFICIENT_SPEECH_DURATION.value

                    await websocket.send_json(
                        ServerStopAck(
                            session_id=session_id,
                            total_duration_seconds=float(elapsed),
                            total_windows_evaluated=total_windows_evaluated,
                            total_windows_generated=session.total_windows_generated,
                            total_windows_dropped=session.total_windows_dropped,
                            status=final_status,
                            final_reason=final_reason,
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
                gap_thresh = (
                    session.config.temporal_policy.timing_gap_threshold_seconds if session else 0.25
                )
                if elapsed_stream > audio_timeline_seconds + gap_thresh:
                    frame_timestamp = elapsed_stream
                else:
                    frame_timestamp = audio_timeline_seconds

                try:
                    ingest_res = session.ingest_pcm(
                        pcm_data=pcm_bytes,
                        sample_rate=sample_rate,
                        channels=channels,
                        timestamp=frame_timestamp,
                        dtype=pcm_enc,
                    )
                except AudioPreprocessingError as exc:
                    logger.warning("Session %s corrupt audio frame: %s", session_id, exc)
                    await websocket.send_json(
                        ServerErrorEvent(
                            error_code="AUDIO_DECODE_ERROR",
                            message=f"Invalid audio frame data: {exc}",
                        ).model_dump()
                    )
                    continue

                frame_duration = ingest_res.samples_ingested / sample_rate
                audio_timeline_seconds = frame_timestamp + frame_duration

                # 4. Emit continuity reset event if discontinuity occurred
                if ingest_res.continuity_reset:
                    reset_event = ServerContinuityResetEvent(
                        session_id=session_id,
                        reason=ReasonCode.INSUFFICIENT_CONTIGUOUS_AUDIO.value,
                        timestamp_delta=ingest_res.timestamp_delta,
                        gap_count=ingest_res.quality_flags.gap_count,
                        continuity_segment_id=ingest_res.continuity_segment_id,
                        timestamp=elapsed_stream,
                    )
                    await websocket.send_json(reset_event.model_dump())

                # 5. Emit quality/backpressure updates if clipping, timing gap, silence, or drops detected
                if (
                    ingest_res.quality_flags.is_clipped
                    or ingest_res.quality_flags.has_timing_gap
                    or ingest_res.quality_flags.is_silent
                    or ingest_res.windows_dropped_this_ingest > 0
                ):
                    quality_event = ServerQualityEvent(
                        session_id=session_id,
                        timestamp=elapsed_stream,
                        is_clipped=ingest_res.quality_flags.is_clipped,
                        is_silent=ingest_res.quality_flags.is_silent,
                        has_timing_gap=ingest_res.quality_flags.has_timing_gap,
                        clipping_count=ingest_res.quality_flags.clipping_count,
                        gap_count=ingest_res.quality_flags.gap_count,
                        total_windows_generated=session.total_windows_generated,
                        total_windows_evaluated=total_windows_evaluated,
                        total_windows_dropped=session.total_windows_dropped,
                    )
                    await websocket.send_json(quality_event.model_dump())

                # 6. Handle immediate abstention decisions (insufficient speech)
                for imm in ingest_res.immediate_decisions:
                    total_windows_evaluated += 1
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
                        operating_threshold=session.config.temporal_policy.threshold_high,
                        operating_threshold_source="unvalidated_heuristic",
                        raw_score=None,
                        calibrated_score=None,
                        decision=imm.model_dump(),
                    )
                    await websocket.send_json(abstained_event.model_dump())

                # 7. Dispatch eligible window for inference if worker is idle
                if session.has_pending_window and not session.in_flight_inference:
                    pending_win = session.pop_pending_window()
                    if pending_win is not None:
                        task = asyncio.create_task(_dispatch_inference(pending_win))
                        inference_tasks.add(task)
                        task.add_done_callback(inference_tasks.discard)

    except WebSocketDisconnect:
        logger.info("Session %s client disconnected.", session_id)
    except asyncio.CancelledError:
        logger.info("Session %s task cancelled by server/lifespan.", session_id)
        raise
    except Exception as exc:
        logger.exception("Unexpected error in streaming session %s: %s", session_id, exc)
    finally:
        is_closing = True
        try:
            # Cancel any pending inference background tasks and await their unwinding
            tasks_snapshot = list(inference_tasks)
            for task in tasks_snapshot:
                if not task.done():
                    task.cancel()
            if tasks_snapshot:
                await asyncio.gather(*tasks_snapshot, return_exceptions=True)
        finally:
            try:
                # Cleanly release session resources even if gather was interrupted
                if session is not None:
                    session.reset()
            except Exception as reset_exc:
                logger.warning("Error resetting session %s: %s", session_id, reset_exc)
            finally:
                try:
                    # Remove from registry even if session.reset() raised
                    if active_sessions is not None:
                        active_sessions.pop(session_id, None)
                except Exception as reg_exc:
                    logger.warning(
                        "Error removing session %s from registry: %s", session_id, reg_exc
                    )
                finally:
                    # Unconditionally release atomic admission reservation exactly once
                    if slot_reserved:
                        slot_reserved = False
                        if streaming_admission is not None:
                            streaming_admission.release(token)

        logger.info("Session %s resources released successfully.", session_id)
