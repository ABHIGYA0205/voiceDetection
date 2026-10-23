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
from services.api.routes.analyze_file import router as analyze_router
from services.api.routes.stream import router as stream_router
from voiceshield import __version__
from voiceshield.core.device import resolve_device_status
from voiceshield.core.schema import Disclaimers
from voiceshield.models.aasist_adapter import (
    AASIST_L_DEFAULT_CONFIG,
    AASISTLDetector,
    CheckpointNotFoundError,
)
from voiceshield.vendor.aasist import AASISTModel

logger = logging.getLogger("voiceshield.api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Preload model detector once at startup and initialize concurrency worker for Apple Silicon M3."""
    app.state.inference_worker = InferenceWorker()
    app.state.active_streaming_sessions = {}
    app.state.max_concurrent_sessions = int(os.getenv("VOICESHIELD_MAX_CONCURRENT_SESSIONS", "10"))

    checkpoint_env = os.getenv("VOICESHIELD_CHECKPOINT_PATH", "models/weights/AASIST-L.pth")
    force_mock = os.getenv("VOICESHIELD_MOCK_MODEL", "false").lower() in ("true", "1", "yes")
    preferred_device = os.getenv("VOICESHIELD_DEVICE")
    checkpoint_path = Path(checkpoint_env)

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
        logger.info("VoiceShield API shutting down. Draining inference worker...")
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


app = FastAPI(
    title="VoiceShield Advisory API",
    version=__version__,
    description=(
        "Advisory synthetic speech detection service. "
        "Outputs do NOT prove caller identity and do NOT constitute fraud probabilities."
    ),
    lifespan=lifespan,
)

# CORS and Origin policy for local development and deployment
allowed_origins_raw = os.getenv(
    "VOICESHIELD_CORS_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173",
).split(",")
allowed_origins = [origin.strip() for origin in allowed_origins_raw if origin.strip()]
app.state.allowed_origins = allowed_origins

# Disable credentials if wildcard origin is used to prevent CORS specification violations and security risks
allow_credentials = False if "*" in allowed_origins else True

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze_router)
app.include_router(stream_router)


@app.get("/health", tags=["System"])
async def health_check() -> dict[str, Any]:
    """Return system liveness status, device acceleration diagnostics, and advisory disclaimers."""
    device_status = resolve_device_status()
    disclaimers = Disclaimers()

    return {
        "status": "ok",
        "service": "VoiceShield",
        "version": __version__,
        "model_loaded": getattr(app.state, "detector", None) is not None,
        "is_mock_detector": getattr(app.state, "is_mock_detector", False),
        "device": {
            "name": device_status.device_name,
            "is_mps": device_status.is_mps,
            "mps_available": device_status.mps_available,
            "explanation": device_status.explanation,
        },
        "disclaimers": disclaimers.model_dump(),
    }


@app.get("/ready", tags=["System"])
async def readiness_check(request: Request) -> dict[str, Any]:
    """Return model readiness status distinguishing API liveness from detector readiness."""
    detector = getattr(request.app.state, "detector", None)
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
    }
