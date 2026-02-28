"""Smoke tests for the FastAPI application."""

from fastapi.testclient import TestClient

from procurement_copilot.api.main import app

client = TestClient(app)


def test_health_returns_200() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
