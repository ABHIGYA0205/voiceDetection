"""Contract tests for equivalent readiness routes."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from httpx import Response

from services.api.main import app


def _readiness_pair(client: TestClient) -> tuple[Response, Response]:
    return client.get("/ready"), client.get("/api/v1/ready")


def test_readiness_aliases_match_in_explicit_mock_mode() -> None:
    with patch.dict("os.environ", {"VOICESHIELD_MOCK_MODEL": "true"}):
        with TestClient(app) as client:
            ready, alias = _readiness_pair(client)

    assert ready.status_code == alias.status_code == 200
    ready_body = ready.json()
    alias_body = alias.json()
    for key in ("model_id", "provenance_status", "checkpoint_verified", "config_hash"):
        assert ready_body[key] == alias_body[key]
    assert ready_body["model_id"] == "mock/aasist-l"
    assert ready_body["provenance_status"] == "mock"
    assert ready_body["checkpoint_verified"] is False
    assert ready_body["config_hash"]


def test_readiness_aliases_match_when_official_checkpoint_is_missing(tmp_path: Path) -> None:
    missing = tmp_path / "missing" / "AASIST-L.pth"
    with patch.dict(
        "os.environ",
        {
            "VOICESHIELD_MOCK_MODEL": "false",
            "VOICESHIELD_CHECKPOINT_PATH": str(missing),
        },
    ):
        with TestClient(app) as client:
            ready, alias = _readiness_pair(client)

    assert ready.status_code == alias.status_code == 503
    assert ready.json()["detail"] == alias.json()["detail"]
    assert ready.json()["detail"]["status"] == "not_ready"


def test_readiness_aliases_fail_closed_for_corrupt_config_hash() -> None:
    with patch.dict("os.environ", {"VOICESHIELD_MOCK_MODEL": "true"}):
        with TestClient(app) as client:
            original_hash = app.state.effective_config_hash
            app.state.effective_config_hash = "corrupt"
            try:
                ready, alias = _readiness_pair(client)
            finally:
                app.state.effective_config_hash = original_hash

    assert ready.status_code == alias.status_code == 503
    assert ready.json()["detail"] == alias.json()["detail"]
    assert ready.json()["detail"]["status"] == "not_ready"
