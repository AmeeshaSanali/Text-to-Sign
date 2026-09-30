import pytest

try:
    from src.text_to_gloss.translator import DictionaryGlossTranslator
except ImportError:
    from backend.src.text_to_gloss.translator import DictionaryGlossTranslator


@pytest.fixture
def translator():
    return DictionaryGlossTranslator(enable_reordering=True, fingerspell_unknown=True)


def test_quickstart_canonical_sentence(translator):
    """Deliverable: running translate('I am going to the store') returns ['I', 'GO', 'STORE']."""
    result = translator.translate("I am going to the store")
    assert result.glosses == ["I", "GO", "STORE"]


def test_drops_articles_and_copulas(translator):
    result = translator.translate("The student is at a school")
    assert result.glosses == ["STUDENT", "SCHOOL"]


def test_phrase_mapping(translator):
    result = translator.translate("Thank you")
    assert result.glosses == ["THANK-YOU"]

    result2 = translator.translate("Hello how are you")
    assert result2.glosses == ["HELLO", "HOW", "YOU"]


def test_time_fronting_grammar(translator):
    result = translator.translate("I went to school yesterday")
    assert result.glosses == ["YESTERDAY", "I", "GO", "SCHOOL"]
    assert result.reordered is True


def test_wh_question_movement(translator):
    # Dynamic wh-question reordering: Wh-word shifts to end
    result = translator.translate("Where are you going?")
    assert result.glosses == ["YOU", "GO", "WHERE"]
    assert result.reordered is True

    # Idiomatic phrase mapping
    phrase_result = translator.translate("What is your name?")
    assert phrase_result.glosses == ["YOUR", "NAME", "WHAT"]


def test_fingerspelling_unknown_word(translator):
    result = translator.translate("I live in Kandy")
    assert "I" in result.glosses
    assert "#KANDY" in result.glosses or "KANDY" in result.unmapped_words
