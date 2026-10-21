"""
Tests for Voice Typing API endpoint and visualizer integration.
"""

import os
import io
import re
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

try:
    from src.app import app
except ImportError:
    from backend.src.app import app

client = TestClient(app)


def test_voice_visualizer_dom_elements():
    """Verify visualizer HTML contains all voice controls and no emojis."""
    response = client.get("/")
    assert response.status_code == 200
    html = response.text

    # Assert voice controls exist in the served HTML
    assert 'id="btnVoiceInput"' in html
    assert 'id="voiceLangSelect"' in html
    assert 'id="chkAutoTranslate"' in html
    assert 'id="voiceStatus"' in html

    # Assert 0 emojis in HTML
    emoji_pattern = re.compile(
        r'[\U00010000-\U0010ffff]|[\u2600-\u26ff]|[\u2700-\u27bf]|[\u2300-\u23ff]|[\u2b50-\u2b55]'
    )
    emojis = emoji_pattern.findall(html)
    assert len(emojis) == 0, f"Found emojis in HTML: {set(emojis)}"


def test_voice_transcribe_empty_file():
    """Empty audio payload should return 400 Bad Request."""
    empty_file = io.BytesIO(b"")
    response = client.post(
        "/api/v1/voice/transcribe",
        files={"file": ("test.webm", empty_file, "audio/webm")},
        data={"language": "en"}
    )
    assert response.status_code == 400
    assert "empty or corrupted" in response.json()["detail"]


def test_voice_transcribe_missing_key():
    """When GEMINI_API_KEY is not set, should return 503."""
    dummy_audio = io.BytesIO(b"RIFF" + b"\x00" * 200)
    with patch.dict(os.environ, {"GEMINI_API_KEY": ""}, clear=False):
        response = client.post(
            "/api/v1/voice/transcribe",
            files={"file": ("test.webm", dummy_audio, "audio/webm")},
            data={"language": "en"}
        )
        assert response.status_code == 503
        assert "Gemini API key is not configured" in response.json()["detail"]


def test_voice_transcribe_english_mocked():
    """Valid English audio upload with mock Gemini returns clean transcript."""
    dummy_audio = io.BytesIO(b"RIFF" + b"\x00" * 300)

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "The doctor came to the hospital"
    mock_client.models.generate_content.return_value = mock_response

    with patch.dict(os.environ, {"GEMINI_API_KEY": "fake_test_key"}), \
         patch("google.genai.Client", return_value=mock_client):
        response = client.post(
            "/api/v1/voice/transcribe",
            files={"file": ("test.webm", dummy_audio, "audio/webm")},
            data={"language": "en"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["transcript"] == "The doctor came to the hospital"
        assert data["detected_language"] == "en"
        assert data["audio_size_bytes"] == 304
        assert data["mime_type"] == "audio/webm"


def test_voice_transcribe_sinhala_mocked():
    """Valid Sinhala audio upload with mock Gemini returns Sinhala script transcript."""
    dummy_audio = io.BytesIO(b"RIFF" + b"\x00" * 300)

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "මම පොතක් කියවනවා"
    mock_client.models.generate_content.return_value = mock_response

    with patch.dict(os.environ, {"GEMINI_API_KEY": "fake_test_key"}), \
         patch("google.genai.Client", return_value=mock_client):
        response = client.post(
            "/api/v1/voice/transcribe",
            files={"file": ("test.webm", dummy_audio, "audio/webm")},
            data={"language": "si"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["transcript"] == "මම පොතක් කියවනවා"
        assert data["detected_language"] == "si"
