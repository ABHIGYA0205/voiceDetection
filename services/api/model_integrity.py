"""Central model provenance and runtime integrity validation for VoiceShield API."""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException, status

from voiceshield.config.schema import VoiceShieldConfig
from voiceshield.models.provenance import ModelProvenance


class ModelIntegrityError(Exception):
    """Raised when runtime model state or provenance violates integrity invariants."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


def validate_runtime_model_provenance(app_state: Any) -> ModelProvenance:
    """Validate that app.state.detector has a valid, untampered ModelProvenance consistent with configured runtime state.

    Raises:
        ModelIntegrityError: If detector is absent, provenance is missing or untyped, or provenance contradicts runtime state.
    """
    detector = getattr(app_state, "detector", None)
    if detector is None:
        load_error = (
            getattr(app_state, "model_load_error", None)
            or "Detector model is not initialized or checkpoint is missing."
        )
        raise ModelIntegrityError(load_error)

    prov = getattr(detector, "provenance", None)
    if prov is None or not isinstance(prov, ModelProvenance):
        raise ModelIntegrityError("Detector provenance is missing, untyped, or invalid.")

    is_mock_state = bool(getattr(app_state, "is_mock_detector", False))
    effective_cfg: VoiceShieldConfig | None = getattr(app_state, "effective_config", None)

    if is_mock_state:
        if prov.model_id != "mock/aasist-l" or prov.provenance_status != "mock" or not prov.is_mock:
            raise ModelIntegrityError(
                "Mock runtime state requires mock/aasist-l provenance with mock status and is_mock=True."
            )
    else:
        if prov.is_mock or prov.provenance_status == "mock" or prov.model_id == "mock/aasist-l":
            raise ModelIntegrityError("Non-mock runtime state cannot accept mock model provenance.")
        if effective_cfg is not None:
            expected_model_id = effective_cfg.model.model_id
            if prov.model_id != expected_model_id:
                raise ModelIntegrityError(
                    f"Configured model_id '{expected_model_id}' conflicts with detector provenance '{prov.model_id}'."
                )
            if expected_model_id == "clovaai/aasist-l":
                if prov.provenance_status != "official_verified" or not prov.checkpoint_verified:
                    raise ModelIntegrityError(
                        "Official clovaai/aasist-l mode requires official_verified provenance with verified checkpoint."
                    )
            elif expected_model_id == "custom/aasist-l":
                if prov.provenance_status != "custom_unverified" or prov.checkpoint_verified:
                    raise ModelIntegrityError(
                        "Custom mode requires custom_unverified provenance with checkpoint_verified=False."
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
