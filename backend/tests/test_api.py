"""Tests for FastAPI backend."""

import pytest
from fastapi.testclient import TestClient
from backend.src.app import app

client = TestClient(app)


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
