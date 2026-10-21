import pytest
import numpy as np

try:
    from src.text_to_gloss.neural_grounder import (
        NeuralCrossLingualGrounder,
        NeuralToken,
        NeuralGroundingResult,
    )
except ImportError:
    from backend.src.text_to_gloss.neural_grounder import (
        NeuralCrossLingualGrounder,
        NeuralToken,
        NeuralGroundingResult,
    )



def test_extract_semantic_units_english():
    grounder = NeuralCrossLingualGrounder()
    sentence = "The doctor is going to the hospital today"
    units = grounder.extract_semantic_units(sentence, lang="en")
    assert "is" not in units
    assert "the" not in [u.lower() for u in units]
    assert "doctor" in units
    assert "hospital" in units
    assert "today" in units


def test_extract_semantic_units_sinhala():
    grounder = NeuralCrossLingualGrounder()
    sentence = "මම අද පොතක් කියවනවා"
    units = grounder.extract_semantic_units(sentence, lang="si")
    assert "මම" in units
    assert "අද" in units
    assert "පොතක්" in units


def test_grammatical_reordering_time_fronting():
    grounder = NeuralCrossLingualGrounder(enable_reordering=True)
    tokens = [
        NeuralToken(original_word="I", gloss="I", similarity=0.9),
        NeuralToken(original_word="eat", gloss="EAT", similarity=0.85),
        NeuralToken(original_word="today", gloss="TODAY", similarity=0.92),
    ]
    reordered, did_reorder = grounder._apply_grammatical_reordering(tokens, is_question=False)
    assert did_reorder is True
    assert [t.gloss for t in reordered] == ["TODAY", "I", "EAT"]


def test_grammatical_reordering_question_final():
    grounder = NeuralCrossLingualGrounder(enable_reordering=True)
    tokens = [
        NeuralToken(original_word="where", gloss="WHERE", similarity=0.9),
        NeuralToken(original_word="you", gloss="YOU", similarity=0.85),
        NeuralToken(original_word="go", gloss="GO", similarity=0.88),
    ]
    reordered, did_reorder = grounder._apply_grammatical_reordering(tokens, is_question=True)
    assert did_reorder is True
    assert [t.gloss for t in reordered] == ["YOU", "GO", "WHERE"]


def test_ground_empty_string():
    grounder = NeuralCrossLingualGrounder()
    res = grounder.ground("")
    assert res.glosses == []
    assert res.tokens == []
    assert res.mean_confidence == 0.0


def test_ground_result_structure_with_mock():
    # Test end-to-end grounding with pre-populated query cache to avoid API calls in CI
    grounder = NeuralCrossLingualGrounder()
    embedder = grounder.embedder

    # Ensure BOOK and DOCTOR prototypes exist
    book_idx = embedder.gloss_to_index["BOOK"]
    doc_idx = embedder.gloss_to_index["DOCTOR"]
    book_vec = embedder.embeddings_matrix[book_idx]
    doc_vec = embedder.embeddings_matrix[doc_idx]

    embedder._query_cache["book"] = book_vec
    embedder._query_cache["doctor"] = doc_vec

    res = grounder.ground("doctor book", language="en")
    assert isinstance(res, NeuralGroundingResult)
    assert len(res.tokens) == 2
    assert res.glosses == ["DOCTOR", "BOOK"]
    assert res.tokens[0].match_type == "exact"
    assert res.tokens[0].similarity > 0.95
    assert len(res.tokens[0].alternatives) > 0
