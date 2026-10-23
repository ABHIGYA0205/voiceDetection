"""FastAPI service entrypoint for VoiceShield."""

from __future__ import annotations

import logging
import math
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware

from services.api.inference_worker import InferenceWorker
from services.api.resource_admission import (
    MAX_CONCURRENT_FILE_UPLOADS_DEFAULT,
    MAX_CONCURRENT_SESSIONS_DEFAULT,
    AtomicAdmissionController,
    FileUploadAdmissionMiddleware,
)
from services.api.routes.analyze_file import router as analyze_router
from services.api.routes.stream import router as stream_router
from voiceshield import __version__
from voiceshield.config.loader import load_config
from voiceshield.config.schema import VoiceShieldConfig
from voiceshield.core.device import resolve_device_status
from voiceshield.core.schema import Disclaimers
from voiceshield.models.aasist_adapter import (
    AASIST_L_DEFAULT_CONFIG,
    AASISTLDetector,
    CheckpointNotFoundError,
)
from voiceshield.vendor.aasist import AASISTModel

logger = logging.getLogger("voiceshield.api")


def _parse_positive_int_env(key: str, default: int) -> int:
    """Parse positive integer environment variable safely with fallback on invalid/negative/zero input."""
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        val = int(raw)
        if val > 0:
            return val
        logger.warning(
            "Invalid %s value %r (must be a positive integer); using fallback %d",
            key,
            raw,
            default,
        )
        return default
    except (ValueError, TypeError):
        logger.warning(
            "Invalid %s value %r (must be an integer); using fallback %d",
            key,
            raw,
            default,
        )
        return default


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load canonical configuration, initialize concurrency workers, and preload detector at startup."""
    max_concurrent_sessions = _parse_positive_int_env(
        "VOICESHIELD_MAX_CONCURRENT_SESSIONS", MAX_CONCURRENT_SESSIONS_DEFAULT
    )
    max_concurrent_file_uploads = _parse_positive_int_env(
        "VOICESHIELD_MAX_CONCURRENT_FILE_UPLOADS", MAX_CONCURRENT_FILE_UPLOADS_DEFAULT
    )

    app.state.max_concurrent_sessions = max_concurrent_sessions
    app.state.max_concurrent_file_uploads = max_concurrent_file_uploads

    app.state.streaming_admission = AtomicAdmissionController(
        limit=max_concurrent_sessions, name="streaming_sessions"
    )
    app.state.file_upload_admission = AtomicAdmissionController(
        limit=max_concurrent_file_uploads, name="file_uploads"
    )
    app.state.inference_worker = InferenceWorker()
    app.state.active_streaming_sessions = {}

    # 1. Load canonical or custom configuration path
    config_path_target = getattr(app.state, "_initial_config_path", None) or os.getenv(
        "VOICESHIELD_CONFIG_PATH"
    )
    base_config = load_config(config_path_target)

    # 2. Reconcile explicit environment overrides
    env_checkpoint = os.getenv("VOICESHIELD_CHECKPOINT_PATH")
    env_device = os.getenv("VOICESHIELD_DEVICE")

    config_dict = base_config.model_dump()
    if env_checkpoint is not None and env_checkpoint.strip():
        config_dict["model"]["checkpoint_path"] = env_checkpoint.strip()
    if env_device is not None and env_device.strip():
        config_dict["model"]["device_preferred"] = env_device.strip().lower()

    effective_config = VoiceShieldConfig.model_validate(config_dict)
    effective_config_hash = effective_config.compute_hash()

    app.state.effective_config = effective_config
    app.state.effective_config_hash = effective_config_hash

    # 3. Detector initialization using effective configuration
    checkpoint_path = Path(effective_config.model.checkpoint_path)
    preferred_device = effective_config.model.device_preferred
    fallback_to_cpu = effective_config.model.fallback_to_cpu
    operating_threshold = effective_config.temporal_policy.threshold_high

    force_mock = os.getenv("VOICESHIELD_MOCK_MODEL", "false").lower() in ("true", "1", "yes")

    app.state.is_mock_detector = False
    app.state.model_load_error = None
    app.state.detector = None

    if force_mock:
        logger.info(
            "Initializing VoiceShield with mock AASIST-L architecture weights "
            "(VOICESHIELD_MOCK_MODEL=true; mock mode active for tests/UI demo only)..."
        )
        mock_raw = AASISTModel(AASIST_L_DEFAULT_CONFIG)
        app.state.detector = AASISTLDetector(
            _instantiated_model=mock_raw,
            preferred_device=preferred_device,
            fallback_to_cpu=fallback_to_cpu,
            threshold=operating_threshold,
        )
        app.state.is_mock_detector = True
    else:
        if not checkpoint_path.is_file():
            instructions = (
                f"AASIST-L checkpoint not found at: {checkpoint_path}.\n"
                "Real mode refuses to mark model as ready without valid checkpoint weights.\n"
                "Setup Instructions:\n"
                "  1. Download official AASIST-L checkpoint:\n"
                "     mkdir -p models/weights\n"
                f"     curl -L -o {checkpoint_path} "
                "https://huggingface.co/SpeechAntiSpoofingBenchmarks/AASIST-L/resolve/main/AASIST-L.pth\n"
                "  2. Or start in mock-detector mode for testing/UI demos:\n"
                "     export VOICESHIELD_MOCK_MODEL=true"
            )
            logger.warning(instructions)
            app.state.model_load_error = instructions
            app.state.detector = None
            app.state.is_mock_detector = False
        else:
            logger.info("Loading AASIST-L checkpoint from: %s", checkpoint_path)
            try:
                app.state.detector = AASISTLDetector(
                    checkpoint_path=checkpoint_path,
                    preferred_device=preferred_device,
                    fallback_to_cpu=fallback_to_cpu,
                    threshold=operating_threshold,
                )
                app.state.is_mock_detector = False
            except CheckpointNotFoundError as exc:
                logger.error("Checkpoint error: %s", exc)
                app.state.model_load_error = str(exc)
                app.state.detector = None
                app.state.is_mock_detector = False
            except Exception as exc:
                logger.error("Unexpected error loading checkpoint: %s", exc)
                app.state.model_load_error = f"Error loading checkpoint {checkpoint_path}: {exc}"
                app.state.detector = None
                app.state.is_mock_detector = False

    try:
        yield
    finally:
        logger.info("VoiceShield API shutting down. Closing admission intake...")
        streaming_admission: AtomicAdmissionController | None = getattr(
            app.state, "streaming_admission", None
        )
        if streaming_admission is not None:
            streaming_admission.close_intake()
            leaked_streams = streaming_admission.reserved_count
            if leaked_streams > 0:
                logger.warning(
                    "Streaming admission closed with %d unreleased reservation(s).",
                    leaked_streams,
                )

        upload_admission: AtomicAdmissionController | None = getattr(
            app.state, "file_upload_admission", None
        )
        if upload_admission is not None:
            upload_admission.close_intake()
            leaked_uploads = upload_admission.reserved_count
            if leaked_uploads > 0:
                logger.warning(
                    "File upload admission closed with %d unreleased reservation(s).",
                    leaked_uploads,
                )

        logger.info("Draining inference worker...")
        worker: InferenceWorker | None = getattr(app.state, "inference_worker", None)
        if worker is not None:
            raw_timeout = os.getenv("VOICESHIELD_INFERENCE_SHUTDOWN_TIMEOUT", "5.0")
            fallback_timeout = 5.0
            shutdown_timeout = fallback_timeout
            try:
                parsed = float(raw_timeout)
                if math.isfinite(parsed) and parsed >= 0.0:
                    shutdown_timeout = parsed
                else:
                    logger.warning(
                        "Invalid VOICESHIELD_INFERENCE_SHUTDOWN_TIMEOUT value %r (must be finite and >= 0); "
                        "using fallback %.1fs",
                        raw_timeout,
                        fallback_timeout,
                    )
            except (ValueError, TypeError):
                logger.warning(
                    "Invalid VOICESHIELD_INFERENCE_SHUTDOWN_TIMEOUT value %r; using fallback %.1fs",
                    raw_timeout,
                    fallback_timeout,
                )
            drained = await worker.shutdown(timeout=shutdown_timeout)
            if not drained:
                logger.error(
                    "Inference worker shutdown timeout (%.2fs) expired with jobs unresolved.",
                    shutdown_timeout,
                )

        logger.info("Releasing active streaming sessions...")
        for _sid, sess in list(app.state.active_streaming_sessions.items()):
            try:
                sess.reset()
            except Exception:
                pass
        app.state.active_streaming_sessions.clear()


def create_app(config_path: str | Path | None = None) -> FastAPI:
    """Application factory for VoiceShield API with optional initial configuration path selector."""
    application = FastAPI(
        title="VoiceShield Advisory API",
        version=__version__,
        description=(
            "Advisory synthetic speech detection service. "
            "Outputs do NOT prove caller identity and do NOT constitute fraud probabilities."
        ),
        lifespan=lifespan,
    )

    if config_path is not None:
        application.state._initial_config_path = config_path

    # CORS and Origin policy for local development and deployment
    allowed_origins_raw = os.getenv(
        "VOICESHIELD_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    allowed_origins = [origin.strip() for origin in allowed_origins_raw if origin.strip()]
    application.state.allowed_origins = allowed_origins

    # Disable credentials if wildcard origin is used to prevent CORS specification violations and security risks
    allow_credentials = False if "*" in allowed_origins else True

    application.add_middleware(FileUploadAdmissionMiddleware)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.include_router(analyze_router)
    application.include_router(stream_router)

    @application.get("/health", tags=["System"])
    async def health_check() -> dict[str, Any]:
        """Return system liveness status, device acceleration diagnostics, and advisory disclaimers."""
        device_status = resolve_device_status()
        disclaimers = Disclaimers()

        return {
            "status": "ok",
            "service": "VoiceShield",
            "version": __version__,
            "model_loaded": getattr(application.state, "detector", None) is not None,
            "is_mock_detector": getattr(application.state, "is_mock_detector", False),
            "config_hash": getattr(application.state, "effective_config_hash", None),
            "device": {
                "name": device_status.device_name,
                "is_mps": device_status.is_mps,
                "mps_available": device_status.mps_available,
                "explanation": device_status.explanation,
            },
            "disclaimers": disclaimers.model_dump(),
        }

    @application.get("/ready", tags=["System"])
    async def readiness_check(request: Request) -> dict[str, Any]:
        """Return model readiness status distinguishing API liveness from detector readiness."""
        detector = getattr(request.app.state, "detector", None)
        config_hash = getattr(request.app.state, "effective_config_hash", None)
        if detector is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "status": "not_ready",
                    "service": "VoiceShield",
                    "reason": (
                        getattr(request.app.state, "model_load_error", None)
                        or "Detector model is not initialized or checkpoint is missing."
                    ),
                    "is_mock_detector": False,
                    "config_hash": config_hash,
                },
            )

        device_status = resolve_device_status()
        return {
            "status": "ready",
            "service": "VoiceShield",
            "version": __version__,
            "model_loaded": True,
            "is_mock_detector": getattr(request.app.state, "is_mock_detector", False),
            "device": device_status.device_name,
            "config_hash": config_hash,
        }

    return application


app = create_app()
