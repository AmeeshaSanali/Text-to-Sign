import pytest
import numpy as np
from pathlib import Path

try:
    from src.text_to_gloss.dense_embedder import DenseSignEmbedder
except ImportError:
    from backend.src.text_to_gloss.dense_embedder import DenseSignEmbedder



def test_embedder_cache_and_dimensions():
    embedder = DenseSignEmbedder()
    assert embedder.is_indexed() is True
    assert embedder.embeddings_matrix is not None
    assert embedder.embeddings_matrix.shape == (383, 3072)
    assert embedder.embeddings_matrix.dtype == np.float32
    assert len(embedder.catalog) == 383


def test_embedder_matrix_l2_normalization():
    embedder = DenseSignEmbedder()
    matrix = embedder.embeddings_matrix
    norms = np.linalg.norm(matrix, axis=1)
    # Every sign vector must be unit-normalized: ||v|| = 1.0 within epsilon
    np.testing.assert_allclose(norms, 1.0, atol=1e-4)


def test_catalog_structure():
    embedder = DenseSignEmbedder()
    for item in embedder.catalog:
        assert "gloss" in item
        assert "category" in item
        assert "label" in item
        assert isinstance(item["gloss"], str)
        assert len(item["gloss"]) > 0


def test_compute_similarities_exact_match():
    embedder = DenseSignEmbedder()
    # Pick the prototype for 'DOCTOR'
    doctor_idx = embedder.gloss_to_index.get("DOCTOR")
    assert doctor_idx is not None
    doctor_vec = embedder.embeddings_matrix[doctor_idx]

    sims = embedder.compute_similarities(doctor_vec)
    assert len(sims) == 383
    # Exact dot product with itself must be 1.0
    assert pytest.approx(sims[doctor_idx], abs=1e-4) == 1.0
    # DOCTOR must be the top argmax
    assert np.argmax(sims) == doctor_idx


def test_get_top_k_retrieval():
    embedder = DenseSignEmbedder()
    book_idx = embedder.gloss_to_index.get("BOOK")
    assert book_idx is not None
    book_vec = embedder.embeddings_matrix[book_idx]

    top_5 = embedder.get_top_k(book_vec, k=5)
    assert len(top_5) == 5
    assert top_5[0]["gloss"] == "BOOK"
    assert pytest.approx(top_5[0]["similarity"], abs=1e-3) == 1.0
    assert top_5[0]["rank"] == 1
    # Check descending order of similarities
    for i in range(len(top_5) - 1):
        assert top_5[i]["similarity"] >= top_5[i + 1]["similarity"]


def test_query_cache():
    embedder = DenseSignEmbedder()
    dummy_vec = np.ones(3072, dtype=np.float32) / np.float32(np.sqrt(3072))
    embedder._query_cache["test_token"] = dummy_vec

    result = embedder.embed_text("test_token")
    np.testing.assert_allclose(result, dummy_vec, atol=1e-5)
