"""API router for offline file-based audio analysis."""

from __future__ import annotations

import io
import re
import urllib.parse
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile, status

from services.api.inference_worker import InferenceWorker, InferenceWorkerClosedError
from services.api.model_integrity import (
    ModelIntegrityError,
    validate_runtime_model_provenance,
)
from services.api.resource_admission import (
    MAX_FILE_BYTES as MAX_FILE_BYTES,
)
from services.api.resource_admission import (
    ApiErrorDetail as ApiErrorDetail,
)
from voiceshield.config.loader import ConfigError
from voiceshield.config.overrides import apply_config_overrides
from voiceshield.pipeline.file_evaluator import FileInferenceResult, evaluate_audio

router = APIRouter(prefix="/api/v1", tags=["Analysis"])

# Constraints
MAX_AUDIO_SECONDS = 300.0  # 5 minutes
SUPPORTED_EXTENSIONS = {".wav", ".flac"}


def sanitize_filename(filename: str | None) -> str:
    """Sanitize user-provided filename to prevent directory traversal and null byte injections."""
    if not filename:
        return "uploaded_audio.wav"

    # URL-decode to prevent double-encoding traversals (e.g. %2e%2e%2f)
    cleaned = urllib.parse.unquote(filename)

    # Remove null bytes and control characters
    cleaned = cleaned.replace("\x00", "")
    cleaned = re.sub(r"[\x01-\x1f\x7f-\x9f]", "", cleaned)

    # Normalize Windows backslashes to forward slashes
    cleaned = cleaned.replace("\\", "/")

    # Strip directory paths
    base_name = Path(cleaned).name
    # Strip any remaining path traversal tokens
    base_name = re.sub(r"\.\.+", "", base_name)
    base_name = base_name.strip(" /\\.")

    return base_name if base_name else "uploaded_audio.wav"


@router.post(
    "/analyze/file",
    response_model=FileInferenceResult,
    responses={
        400: {"model": ApiErrorDetail, "description": "Invalid duration or corrupt payload"},
        413: {"model": ApiErrorDetail, "description": "File exceeds 25 MB limit"},
        415: {"model": ApiErrorDetail, "description": "Unsupported media format (WAV/FLAC only)"},
        422: {"model": ApiErrorDetail, "description": "Unprocessable audio data"},
    },
    summary="Analyze uploaded audio file for synthetic speech indicators",
)
async def analyze_audio_file(
    request: Request,
    file: Annotated[
        UploadFile,
        File(description="Audio file in WAV or FLAC format (max 25MB, max 300s)"),
    ],
    hop_seconds: Annotated[
        float | None,
        Query(
            ge=0.25,
            le=4.0,
            description="Sliding window step size in seconds. If omitted, uses server configuration.",
        ),
    ] = None,
    threshold: Annotated[
        float | None,
        Query(
            ge=0.01,
            le=0.99,
            description="Synthetic speech likelihood decision threshold. If omitted, uses server configuration.",
        ),
    ] = None,
) -> FileInferenceResult:
    """Analyze an uploaded WAV or FLAC file without permanent storage.

    - Processes audio strictly in memory with no retention.
    - Evaluates 4.0-second sliding windows with deterministic feature extraction.
    - Concurrency-protected via process-wide single-worker inference executor suitable for Apple Silicon M3 unified memory.
    - Advisory output only: does NOT determine caller identity or fraud likelihood.
    """
    # 0. Validate the complete runtime integrity contract before any request work.
    try:
        validate_runtime_model_provenance(request.app.state)
    except ModelIntegrityError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=ApiErrorDetail(
                error_code=exc.error_code,
                message=exc.message,
            ).model_dump(),
        ) from exc
    effective_cfg = request.app.state.effective_config

    # Validate and apply request overrides using centralized helper
    try:
        request_config = apply_config_overrides(
            base_config=effective_cfg,
            hop_seconds=hop_seconds,
            threshold=threshold,
        )
    except ConfigError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=ApiErrorDetail(
                error_code="INVALID_CONFIG_OVERRIDE",
                message=str(exc),
                details={"hop_seconds": hop_seconds, "threshold": threshold},
            ).model_dump(),
        ) from exc

    raw_filename = file.filename or "unknown.wav"
    clean_name = sanitize_filename(raw_filename)
    extension = Path(clean_name).suffix.lower()

    # 1. Type validation (WAV/FLAC only)
    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=ApiErrorDetail(
                error_code="UNSUPPORTED_MEDIA_TYPE",
                message=f"Unsupported file format '{extension}'. VoiceShield accepts WAV and FLAC files.",
                details={
                    "filename": clean_name,
                    "supported_extensions": list(SUPPORTED_EXTENSIONS),
                },
            ).model_dump(),
        )

    # 2. Size limit enforcement (read up to MAX_FILE_BYTES + 1 to detect overflow)
    # Ephemeral in-memory evaluation buffer; no permanent persistence
    content = await file.read(MAX_FILE_BYTES + 1)
    if len(content) > MAX_FILE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=ApiErrorDetail(
                error_code="PAYLOAD_TOO_LARGE",
                message=f"Uploaded file exceeds the maximum allowed size of {MAX_FILE_BYTES / (1024 * 1024):.0f} MB.",
                details={"max_bytes": MAX_FILE_BYTES, "received_bytes": len(content)},
            ).model_dump(),
        )

    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=ApiErrorDetail(
                error_code="EMPTY_AUDIO_FILE",
                message="Uploaded file is empty (0 bytes).",
                details={"filename": clean_name},
            ).model_dump(),
        )

    # 3. Use the detector validated by the shared runtime integrity contract.
    detector = request.app.state.detector

    # 4. Concurrency-guarded execution via process-wide single-worker executor
    worker: InferenceWorker | None = getattr(request.app.state, "inference_worker", None)
    if worker is None or not worker.accepting_jobs:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=ApiErrorDetail(
                error_code="INFERENCE_WORKER_UNAVAILABLE",
                message="VoiceShield inference worker is unavailable or shutting down.",
            ).model_dump(),
        )

    try:
        result = await worker.execute(
            evaluate_audio,
            source=io.BytesIO(content),
            detector=detector,
            config=request_config,
            file_name=clean_name,
            max_duration_seconds=MAX_AUDIO_SECONDS,
        )
        assert isinstance(result, FileInferenceResult)
    except InferenceWorkerClosedError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=ApiErrorDetail(
                error_code="INFERENCE_WORKER_UNAVAILABLE",
                message=f"VoiceShield inference worker is closed: {exc}",
            ).model_dump(),
        ) from exc

    # 5. Result validation and error mapping
    if result.status == "error":
        err_msg = result.error_message or "Unknown evaluation error"
        if "exceeds maximum allowed limit" in err_msg:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ApiErrorDetail(
                    error_code="AUDIO_DURATION_EXCEEDED",
                    message=err_msg,
                    details={"max_duration_seconds": MAX_AUDIO_SECONDS},
                ).model_dump(),
            )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=ApiErrorDetail(
                error_code="AUDIO_PROCESSING_ERROR",
                message=err_msg,
                details={"filename": clean_name},
            ).model_dump(),
        )

    return result
