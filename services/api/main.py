"""FastAPI service entrypoint for VoiceShield."""

from __future__ import annotations

import logging
import math
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from services.api.inference_worker import InferenceWorker
from services.api.model_integrity import (
    ModelIntegrityError,
    ensure_model_ready_for_http,
    validate_runtime_model_provenance,
)
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
from voiceshield.config.overrides import apply_config_overrides
from voiceshield.config.schema import VoiceShieldConfig
from voiceshield.core.device import resolve_device_status
from voiceshield.core.schema import Disclaimers
from voiceshield.models import (
    AASIST_L_DEFAULT_CONFIG,
    AASISTLDetector,
    CheckpointError,
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
    # 1. Load canonical or custom configuration path and reconcile explicit environment overrides FIRST
    config_path_target = getattr(app.state, "_initial_config_path", None) or os.getenv(
        "VOICESHIELD_CONFIG_PATH"
    )
    base_config = load_config(config_path_target)

    env_model_id = os.getenv("VOICESHIELD_MODEL_ID")
    env_checkpoint = os.getenv("VOICESHIELD_CHECKPOINT_PATH")
    env_device = os.getenv("VOICESHIELD_DEVICE")

    effective_config = apply_config_overrides(
        base_config=base_config,
        model_id=env_model_id,
        checkpoint_path=env_checkpoint,
        device_preferred=env_device,
    )
    effective_config_hash = effective_config.compute_hash()

    app.state.effective_config = effective_config
    app.state.effective_config_hash = effective_config_hash

    # 2. Concurrency workers and admission controllers are initialized ONLY after config is validated
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

    # 3. Detector initialization using effective configuration
    checkpoint_path = Path(effective_config.model.checkpoint_path)
    model_id = effective_config.model.model_id
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
            model_id="mock/aasist-l",
            preferred_device=preferred_device,
            fallback_to_cpu=fallback_to_cpu,
            threshold=operating_threshold,
        )
        app.state.is_mock_detector = True
    else:
        logger.info(
            "Initializing AASIST-L detector (model_id=%s) from: %s",
            model_id,
            checkpoint_path,
        )
        try:
            app.state.detector = AASISTLDetector(
                checkpoint_path=checkpoint_path,
                model_id=model_id,
                preferred_device=preferred_device,
                fallback_to_cpu=fallback_to_cpu,
                threshold=operating_threshold,
            )
            app.state.is_mock_detector = False
        except CheckpointError as exc:
            logger.error("Checkpoint error initializing detector: %s", exc)
            app.state.model_load_error = str(exc)
            app.state.detector = None
            app.state.is_mock_detector = False
        except Exception as exc:
            logger.error("Unexpected error loading detector: %s", exc)
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
        detector = getattr(application.state, "detector", None)
        effective_cfg = getattr(application.state, "effective_config", None)
        actual_device = (
            detector.device.type if detector is not None and hasattr(detector, "device") else None
        )
        configured_device = (
            effective_cfg.model.device_preferred
            if isinstance(effective_cfg, VoiceShieldConfig)
            else None
        )
        device_status = resolve_device_status()
        disclaimers = Disclaimers()

        active_device_name = (
            actual_device if actual_device is not None else device_status.device_name
        )

        model_provenance = None
        model_loaded = False
        try:
            validated_prov = validate_runtime_model_provenance(application.state)
            model_provenance = validated_prov.model_dump()
            model_loaded = True
        except ModelIntegrityError:
            model_provenance = None
            model_loaded = False

        return {
            "status": "ok",
            "service": "VoiceShield",
            "version": __version__,
            "model_loaded": model_loaded,
            "model_ready": model_loaded,
            "model_integrity_valid": model_loaded,
            "is_mock_detector": getattr(application.state, "is_mock_detector", False),
            "config_hash": getattr(application.state, "effective_config_hash", None),
            "model_provenance": model_provenance,
            "device": {
                "name": active_device_name,
                "is_mps": active_device_name == "mps",
                "mps_available": device_status.mps_available,
                "configured_preference": configured_device,
                "actual_detector_device": actual_device,
                "explanation": device_status.explanation,
            },
            "disclaimers": disclaimers.model_dump(),
        }

    @application.get("/ready", tags=["System"])
    async def readiness_check(request: Request) -> dict[str, Any]:
        """Return model readiness status distinguishing API liveness from detector readiness."""
        prov = ensure_model_ready_for_http(request.app.state)
        detector = request.app.state.detector
        actual_device = detector.device.type if hasattr(detector, "device") else "cpu"
        stored_hash = getattr(request.app.state, "effective_config_hash", None)

        return {
            "status": "ready",
            "service": "VoiceShield",
            "version": __version__,
            "model_loaded": True,
            "is_mock_detector": getattr(request.app.state, "is_mock_detector", False),
            "device": actual_device,
            "config_hash": stored_hash,
            "model_id": prov.model_id,
            "provenance_status": prov.provenance_status,
            "checkpoint_verified": prov.checkpoint_verified,
            "checkpoint_sha256": prov.checkpoint_sha256,
            "upstream_commit": prov.upstream_commit,
            "parameter_count": prov.parameter_count,
            "model_provenance": prov.model_dump(),
        }

    return application


app = create_app()
