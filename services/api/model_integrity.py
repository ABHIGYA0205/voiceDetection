"""Central model provenance and runtime integrity validation for VoiceShield API."""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException, status

from voiceshield.config.schema import VoiceShieldConfig
from voiceshield.models.provenance import ModelProvenance


class ModelIntegrityError(Exception):
    """Raised when runtime model state or provenance violates integrity invariants."""

    def __init__(self, message: str, error_code: str = "DETECTOR_NOT_READY") -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code


def validate_runtime_model_provenance(app_state: Any) -> ModelProvenance:
    """Validate that app.state.detector has a valid, untampered ModelProvenance consistent with configured runtime state.

    Shared runtime validator validating all of:
    - detector exists;
    - provenance is exactly a valid ModelProvenance instance;
    - effective config exists and is a VoiceShieldConfig;
    - stored config hash exists, is a nonempty string, and equals a safely recomputed hash;
    - config-hash computation failures become ModelIntegrityError;
    - explicit runtime mock state exactly matches mock provenance;
    - non-mock official config exactly matches official provenance;
    - non-mock custom config exactly matches custom provenance.

    Raises:
        ModelIntegrityError: If detector is absent, provenance is missing or untyped,
            config or config hash is invalid, or provenance contradicts runtime state.
    """
    # 1. Effective configuration exists and is a VoiceShieldConfig
    effective_cfg = getattr(app_state, "effective_config", None)
    if effective_cfg is None or not isinstance(effective_cfg, VoiceShieldConfig):
        raise ModelIntegrityError(
            "Effective configuration unavailable or integrity check failed.",
            error_code="CONFIG_UNAVAILABLE",
        )

    # 2. Stored config hash exists, is a nonempty string, and equals a safely recomputed hash
    stored_hash = getattr(app_state, "effective_config_hash", None)
    if not isinstance(stored_hash, str) or not stored_hash:
        raise ModelIntegrityError(
            "Effective configuration unavailable or integrity check failed.",
            error_code="CONFIG_UNAVAILABLE",
        )

    try:
        recomputed_hash = effective_cfg.compute_hash()
    except Exception as exc:
        raise ModelIntegrityError(
            "Effective configuration unavailable or integrity check failed.",
            error_code="CONFIG_UNAVAILABLE",
        ) from exc

    if recomputed_hash != stored_hash:
        raise ModelIntegrityError(
            "Effective configuration unavailable or integrity check failed.",
            error_code="CONFIG_UNAVAILABLE",
        )

    # 3. Detector exists
    detector = getattr(app_state, "detector", None)
    if detector is None:
        load_error = (
            getattr(app_state, "model_load_error", None)
            or "Detector model is not initialized or checkpoint is missing."
        )
        raise ModelIntegrityError(load_error, error_code="DETECTOR_NOT_LOADED")

    # 4. Provenance is exactly a valid ModelProvenance instance
    prov = getattr(detector, "provenance", None)
    if prov is None or not isinstance(prov, ModelProvenance):
        raise ModelIntegrityError(
            "Detector provenance is missing, untyped, or invalid.",
            error_code="DETECTOR_NOT_LOADED",
        )

    # 5. Runtime mock state vs non-mock state and provenance cross-check
    raw_mock_state = getattr(app_state, "is_mock_detector", None)
    if not isinstance(raw_mock_state, bool):
        raise ModelIntegrityError(
            "Runtime mock state is missing or not a strict boolean.",
            error_code="DETECTOR_NOT_LOADED",
        )
    is_mock_state = raw_mock_state

    if is_mock_state:
        if prov.model_id != "mock/aasist-l" or prov.provenance_status != "mock" or not prov.is_mock:
            raise ModelIntegrityError(
                "Mock runtime state requires mock/aasist-l provenance with mock status and is_mock=True.",
                error_code="DETECTOR_NOT_LOADED",
            )
    else:
        if prov.is_mock or prov.provenance_status == "mock" or prov.model_id == "mock/aasist-l":
            raise ModelIntegrityError(
                "Non-mock runtime state cannot accept mock model provenance.",
                error_code="DETECTOR_NOT_LOADED",
            )
        expected_model_id = effective_cfg.model.model_id
        if effective_cfg.model.architecture != prov.architecture:
            raise ModelIntegrityError(
                f"Configured architecture '{effective_cfg.model.architecture}' conflicts with detector provenance '{prov.architecture}'.",
                error_code="DETECTOR_NOT_LOADED",
            )
        if prov.model_id != expected_model_id:
            raise ModelIntegrityError(
                f"Configured model_id '{expected_model_id}' conflicts with detector provenance '{prov.model_id}'.",
                error_code="DETECTOR_NOT_LOADED",
            )
        if expected_model_id == "clovaai/aasist-l":
            if prov.provenance_status != "official_verified" or not prov.checkpoint_verified:
                raise ModelIntegrityError(
                    "Official clovaai/aasist-l mode requires official_verified provenance with verified checkpoint.",
                    error_code="DETECTOR_NOT_LOADED",
                )
        elif expected_model_id == "custom/aasist-l":
            if prov.provenance_status != "custom_unverified" or prov.checkpoint_verified:
                raise ModelIntegrityError(
                    "Custom mode requires custom_unverified provenance with checkpoint_verified=False.",
                    error_code="DETECTOR_NOT_LOADED",
                )

    return prov


def ensure_model_ready_for_http(app_state: Any) -> ModelProvenance:
    """Validate runtime model provenance, mapping ModelIntegrityError to HTTPException(503) for HTTP endpoints."""
    stored_hash = getattr(app_state, "effective_config_hash", None)
    config_hash_val = stored_hash if isinstance(stored_hash, str) and stored_hash else None
    is_mock_state = bool(getattr(app_state, "is_mock_detector", False))

    try:
        return validate_runtime_model_provenance(app_state)
    except ModelIntegrityError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "not_ready",
                "service": "VoiceShield",
                "reason": exc.message,
                "is_mock_detector": is_mock_state,
                "config_hash": config_hash_val,
            },
        ) from exc
