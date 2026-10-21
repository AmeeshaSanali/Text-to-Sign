"""Unit tests for Gemini-powered sign language translator."""

import pytest
from unittest.mock import MagicMock, patch

try:
    from src.text_to_gloss.gemini_translator import GeminiGlossTranslator, GeminiTranslationResponse, GeminiGlossToken
except ImportError:
    from backend.src.text_to_gloss.gemini_translator import GeminiGlossTranslator, GeminiTranslationResponse, GeminiGlossToken


def test_gemini_translator_initialization():
    translator = GeminiGlossTranslator(api_key="test_key_123")
    assert translator.api_key == "test_key_123"
    assert len(translator.allowed_glosses) > 0
    assert "HELLO" in translator.allowed_glosses
    assert "MOTHER" in translator.allowed_glosses


def test_gemini_system_instruction_contains_vocabulary():
    translator = GeminiGlossTranslator()
    sys_inst = translator.build_system_instruction()
    assert "VOCABULARY RESTRICTION" in sys_inst
    assert "HELLO" in sys_inst
    assert "DOCTOR" in sys_inst
    assert "OUT-OF-VOCABULARY" in sys_inst


def test_gemini_is_available_check():
    translator = GeminiGlossTranslator(api_key=None)
    with patch.dict("os.environ", {}, clear=True):
        assert translator.is_available() is False

    with patch.dict("os.environ", {"GEMINI_API_KEY": "AIzaSyTestKey"}):
        # When google-genai is installed and key is set, is_available is True
        assert translator.is_available() is True


def test_gemini_mock_translation():
    translator = GeminiGlossTranslator(api_key="AIzaSyMockKey")

    mock_response_json = {
        "detected_language": "si",
        "original_text": "අද මගේ අම්මා කනවා",
        "glosses": ["TODAY", "MY", "MOTHER", "EAT"],
        "tokens": [
            {
                "original_word": "අද",
                "gloss": "TODAY",
                "match_type": "exact",
                "confidence": 1.0,
                "facial_expression": "neutral",
                "note": "Time-fronted"
            },
            {
                "original_word": "මගේ",
                "gloss": "MY",
                "match_type": "exact",
                "confidence": 1.0,
                "facial_expression": "neutral"
            },
            {
                "original_word": "අම්මා",
                "gloss": "MOTHER",
                "match_type": "exact",
                "confidence": 1.0,
                "facial_expression": "neutral"
            },
            {
                "original_word": "කනවා",
                "gloss": "EAT",
                "match_type": "exact",
                "confidence": 1.0,
                "facial_expression": "neutral"
            }
        ],
        "reordered": True,
        "grammatical_explanation": "Time-fronted SOV order"
    }

    mock_client = MagicMock()
    mock_response = MagicMock()
    import json
    mock_response.text = json.dumps(mock_response_json)
    mock_client.models.generate_content.return_value = mock_response

    with patch("google.genai.Client", return_value=mock_client):
        result = translator.translate("අද මගේ අම්මා කනවා")
        assert result.detected_language == "si"
        assert result.glosses == ["TODAY", "MY", "MOTHER", "EAT"]
        assert len(result.tokens) == 4
        assert result.tokens[0].gloss == "TODAY"
        assert result.tokens[3].gloss == "EAT"
