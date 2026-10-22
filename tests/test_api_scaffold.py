"""Unit tests for the FastAPI scaffolding and health check endpoint."""

from fastapi.testclient import TestClient

from services.api.main import app

client = TestClient(app)


def test_health_endpoint() -> None:
    """Verify that /health responds with status ok and device diagnostics."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "VoiceShield"
    assert "device" in data
    assert "name" in data["device"]
    assert "disclaimers" in data
    assert "advisory_action" in data["disclaimers"]
