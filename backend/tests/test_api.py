"""Tests for FastAPI backend."""

import pytest
from fastapi.testclient import TestClient

try:
    from src.app import app
except ImportError:
    from backend.src.app import app

client = TestClient(app)


def test_root_dashboard():
    response = client.get("/")
    assert response.status_code == 200
    assert "SLSL Avatar Engine" in response.text


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_translate_api():
    payload = {"text": "I am going to the store"}
    response = client.post("/api/v1/translate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["glosses"] == ["I", "GO", "STORE"]
    assert len(data["animation_sequence"]) == 3
    assert data["animation_sequence"][0]["gloss"] == "I"
    assert data["animation_sequence"][1]["gloss"] == "GO"
    assert data["animation_sequence"][2]["gloss"] == "STORE"


def test_vocabulary_endpoint():
    response = client.get("/api/v1/vocabulary")
    assert response.status_code == 200
    data = response.json()
    assert data["total_glosses"] > 0
    assert "GO" in data["glosses"]
    assert "STORE" in data["glosses"]


def test_sample_sentences_endpoint():
    response = client.get("/api/v1/sample-sentences")
    assert response.status_code == 200
    data = response.json()
    assert len(data["samples"]) > 0


def test_static_animations_endpoint():
    # Verify static file serving of approved BLUE.json
    response = client.get("/animations/slsl/BLUE.json")
    assert response.status_code == 200
    data = response.json()
    assert data["gloss"] == "BLUE"
    assert data["fps"] == 30
    assert len(data["frames"]) > 0


def test_approved_animations_endpoint():
    response = client.get("/api/v1/animations/approved")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] >= 10
    assert "BLUE" in data["approved_glosses"]
    assert "HELLO" in data["approved_glosses"]


def test_manifest_endpoint():
    response = client.get("/api/v1/animations/manifest")
    assert response.status_code == 200
    data = response.json()
    assert data["total_signs"] == 383


def test_translate_with_pose_metadata():
    payload = {"text": "hello blue today"}
    response = client.post("/api/v1/translate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["glosses"] == ["TODAY", "HELLO", "BLUE"]
    for cue in data["animation_sequence"]:
        assert cue["has_pose_animation"] is True
        assert cue["sign_status"] == "approved"
        assert cue["animation_url"] is not None
        assert cue["animation_url"].startswith("/animations/slsl/")
