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


def test_sinhala_greeting(translator):
    result = translator.translate("ආයුබෝවන්")
    assert result.detected_language == "si"
    assert result.glosses == ["HELLO"]


def test_sinhala_morphological_stemming_and_time_fronting(translator):
    # ඊයේ (time marker) -> fronted, මගේ (genitive) -> MY, අම්මා -> MOTHER, කෑවා (irregular past) -> EAT
    result = translator.translate("මගේ අම්මා ඊයේ කෑවා")
    assert result.detected_language == "si"
    assert result.glosses == ["YESTERDAY", "MY", "MOTHER", "EAT"]
    assert result.reordered is True


def test_cross_lingual_gloss_convergence(translator):
    # English and Sinhala inputs converge to identical SLSL sign sequence
    en_res = translator.translate("Today my mother eats")
    si_res = translator.translate("අද මගේ අම්මා කනවා")

    assert en_res.glosses == ["TODAY", "MY", "MOTHER", "EAT"]
    assert si_res.glosses == ["TODAY", "MY", "MOTHER", "EAT"]
    assert en_res.glosses == si_res.glosses


def test_compound_sign_decomposition(translator):
    # Hospital is decomposed into DOCTOR + PLACE (Tier 3)
    en_res = translator.translate("hospital")
    assert en_res.glosses == ["DOCTOR", "PLACE"]
    assert en_res.tokens[0].match_type == "compound"

    si_res = translator.translate("රෝහල")
    assert si_res.glosses == ["DOCTOR", "PLACE"]
    assert si_res.tokens[0].match_type == "compound"


def test_semantic_subsumption_fallback(translator):
    # Physician -> DOCTOR (Tier 2 Semantic Approximation)
    result = translator.translate("physician")
    assert result.glosses == ["DOCTOR"]
    assert result.tokens[0].match_type == "semantic_approximation"
    assert result.tokens[0].confidence >= 0.85

    # Automobile / Van -> VEHICLE
    auto_res = translator.translate("automobile")
    assert auto_res.glosses == ["VEHICLE"]
    assert auto_res.tokens[0].match_type == "semantic_approximation"


def test_sinhala_dactylology(translator):
    # Unknown proper name in Sinhala is fingerspelled with Sinhala character tokens
    result = translator.translate("නිපුන්")
    assert result.detected_language == "si"
    assert len(result.fingerspelled_words) > 0
    assert result.tokens[0].is_fingerspelled is True
    assert result.tokens[0].gloss.startswith("#")

