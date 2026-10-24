"""Focused regressions for final provenance validation gaps."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from fastapi.testclient import TestClient

from services.api.main import create_app
from voiceshield.models import OFFICIAL_AASIST_L, ModelProvenance


def test_official_repository_must_match_registry() -> None:
    """Official provenance rejects an upstream repository different from the registry."""
    with pytest.raises(ValueError, match="upstream_repository"):
        ModelProvenance(
            model_id="clovaai/aasist-l",
            architecture="aasist_l",
            provenance_status="official_verified",
            checkpoint_verified=True,
            checkpoint_sha256=OFFICIAL_AASIST_L.checkpoint_sha256,
            expected_checkpoint_sha256=OFFICIAL_AASIST_L.checkpoint_sha256,
            upstream_repository="https://attacker.invalid/not-upstream",
            upstream_commit=OFFICIAL_AASIST_L.upstream_commit,
            parameter_count=OFFICIAL_AASIST_L.parameter_count,
            checkpoint_size_bytes=OFFICIAL_AASIST_L.checkpoint_size_bytes,
            source_license_spdx=OFFICIAL_AASIST_L.source_license_spdx,
            checkpoint_rights_note=OFFICIAL_AASIST_L.checkpoint_rights_note,
        )


@pytest.mark.parametrize("checkpoint_size_bytes", [None, 0, -1, True, 1.5, "426428"])
def test_custom_checkpoint_size_must_be_positive_integer(
    checkpoint_size_bytes: Any,
) -> None:
    """Custom provenance requires a measured, strictly positive integer checkpoint size."""
    with pytest.raises(ValueError, match="checkpoint_size_bytes"):
        ModelProvenance(
            model_id="custom/aasist-l",
            architecture="aasist_l",
            provenance_status="custom_unverified",
            checkpoint_verified=False,
            checkpoint_sha256="0" * 64,
            parameter_count=1,
            checkpoint_size_bytes=checkpoint_size_bytes,
        )


def test_custom_checkpoint_size_accepts_positive_integer() -> None:
    """Custom provenance accepts a measured positive integer checkpoint size."""
    provenance = ModelProvenance(
        model_id="custom/aasist-l",
        architecture="aasist_l",
        provenance_status="custom_unverified",
        checkpoint_verified=False,
        checkpoint_sha256="0" * 64,
        parameter_count=1,
        checkpoint_size_bytes=426428,
    )
    assert provenance.checkpoint_size_bytes == 426428


@pytest.mark.parametrize(
    "state_case", ["missing_config", "untyped_config", "bad_hash", "invalid_provenance"]
)
def test_health_remains_live_for_invalid_runtime_state(state_case: str) -> None:
    """Health remains 200 while reporting invalid model readiness and null provenance."""
    app = create_app()
    with TestClient(app) as client:
        original = {
            "effective_config": app.state.effective_config,
            "effective_config_hash": app.state.effective_config_hash,
            "detector": app.state.detector,
        }
        try:
            if state_case == "missing_config":
                app.state.effective_config = None
            elif state_case == "untyped_config":
                app.state.effective_config = "not-a-config"
            elif state_case == "bad_hash":
                app.state.effective_config_hash = "corrupt"
            else:
                app.state.detector = SimpleNamespace(
                    device=torch.device("cpu"),
                    provenance={"model_id": "mock/aasist-l"},
                )

            response = client.get("/health")
            body = response.json()
            assert response.status_code == 200
            assert body["model_loaded"] is False
            assert body["model_ready"] is False
            assert body["model_integrity_valid"] is False
            assert body["model_provenance"] is None
            if state_case in {"missing_config", "untyped_config"}:
                assert body["device"]["configured_preference"] is None
        finally:
            for key, value in original.items():
                setattr(app.state, key, value)
