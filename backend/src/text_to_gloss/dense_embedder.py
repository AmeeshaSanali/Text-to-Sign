"""
Dense Cross-Lingual Semantic Embedder & Prototype Indexer
Part of the Neural Semantic Grounding Architecture

Provides:
- 3072-dimensional continuous latent space via gemini-embedding-001
- Pre-indexing and persistent caching of all 383 canonical sign classes in SSL400 (data/sign_embeddings_383.npy)
- Sub-millisecond vector matrix cosine similarity search via NumPy
- Graceful offline fallback using cached embeddings
"""

import os
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np

# Load .env if present
try:
    from dotenv import load_dotenv
    root_dir = Path(__file__).resolve().parent.parent.parent.parent
    for env_path in [root_dir / ".env", Path(__file__).resolve().parent.parent / ".env"]:
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
except ImportError:
    pass

from .sinhala_processor import SINHALA_TO_GLOSS
from .semantic_fallback import SEMANTIC_TAXONOMY, COMPOUND_DECOMPOSITIONS


class DenseSignEmbedder:
    """Manages continuous dense vector representations for the 383 canonical SSL400 signs."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-embedding-001",
        cache_dir: Optional[Path] = None,
    ):
        self.api_key = api_key
        self.model_name = model_name

        if cache_dir is None:
            self.cache_dir = Path(__file__).resolve().parent.parent.parent.parent / "data"
        else:
            self.cache_dir = Path(cache_dir)

        self.matrix_path = self.cache_dir / "sign_embeddings_383.npy"
        self.catalog_path = self.cache_dir / "sign_catalog_383.json"
        self.manifest_path = self.cache_dir / "manifest.csv"

        self.embeddings_matrix: Optional[np.ndarray] = None  # shape (383, 3072)
        self.catalog: List[Dict[str, Any]] = []             # length 383
        self.gloss_to_index: Dict[str, int] = {}
        self._query_cache: Dict[str, np.ndarray] = {}

        # Load from cache if available
        self._load_cached_index()

    def _load_cached_index(self) -> bool:
        """Loads pre-computed matrix and catalog from disk if present."""
        if self.matrix_path.exists() and self.catalog_path.exists():
            try:
                self.embeddings_matrix = np.load(self.matrix_path)
                with open(self.catalog_path, "r", encoding="utf-8") as f:
                    self.catalog = json.load(f)

                # Normalize rows to unit length for exact cosine dot product
                norms = np.linalg.norm(self.embeddings_matrix, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                self.embeddings_matrix = self.embeddings_matrix / norms

                self.gloss_to_index = {item["gloss"]: idx for idx, item in enumerate(self.catalog)}
                return True
            except Exception as e:
                print(f"Warning: Failed loading cached embedding index: {e}")
        return False

    def is_indexed(self) -> bool:
        """Returns True if the 383-sign prototype matrix is loaded in memory."""
        return self.embeddings_matrix is not None and len(self.catalog) > 0

    def _get_api_client(self):
        """Initializes Google GenAI Client with active API key."""
        key = self.api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY is not configured.")
        from google import genai
        return genai.Client(api_key=key)

    def generate_prototypes_from_manifest(self) -> List[Dict[str, Any]]:
        """
        Synthesizes rich bilingual semantic prototypes for each canonical sign in manifest.csv.
        Combines English gloss, category, morphological synonyms, and Sinhala lexical terms.
        """
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        # Map glosses to known Sinhala terms from rules & taxonomy
        gloss_to_sinhala: Dict[str, List[str]] = {}
        for si_w, g in SINHALA_TO_GLOSS.items():
            gloss_to_sinhala.setdefault(g, []).append(si_w)
        for term, data in SEMANTIC_TAXONOMY.items():
            g = data[0]
            gloss_to_sinhala.setdefault(g, []).append(term)

        # Extended contextual definitions for key SLSL signs
        domain_contexts = {
            "AYUBOWAN": "traditional Sri Lankan greeting, wishing long life, hello, respectful salutation",
            "HELLO": "greeting, hi, welcome, cordial social meeting",
            "THANK_YOU": "gratitude, appreciation, thank you very much, polite thanks",
            "DOCTOR": "physician, medic, healthcare practitioner, hospital clinic doctor, medical specialist",
            "HOSPITAL": "hospital, clinic, medical center, healthcare facility, infirmary",
            "BOOK": "book, reading book, literature, publication, textbook, study material",
            "STUDY": "study, reading, learn, revise, examine, student education, schooling",
            "WRITE": "write, writing, pencil, pen, notation, document text",
            "MOVIE": "movie, film, cinema, teledrama, drama, television show, video motion picture",
            "HOUSE": "house, home, residence, household, living place",
            "SCHOOL": "school, academy, classroom, education building, student study institute",
            "FOOD": "food, meal, edible cuisine, cooked food, dish",
            "EAT": "eat, consume food, dine, tasting, eating breakfast lunch dinner",
            "DRINK": "drink, beverage, consume liquid, water, tea, milk",
            "TEA": "tea, Ceylon tea, hot brewed beverage, tea cup",
            "MILK": "milk, dairy beverage, fresh cow milk",
            "WATER": "water, drinking liquid, hydration, aqua",
            "TREE": "tree, plant, flora, nature tree, timber branch",
            "FLOWER": "flower, bloom, blossom, floral petal",
            "CAR": "car, automobile, motor vehicle, passenger sedan",
            "BUS": "bus, public transport bus, omnibus, coach",
            "TRAIN": "train, railway, locomotive, railway carriage",
            "POLICE": "police, police officer, law enforcement, security cop",
            "TODAY": "today, present day, current date, now day",
            "YESTERDAY": "yesterday, previous day, past date",
            "TOMORROW": "tomorrow, next day, upcoming date",
        }

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            manifest_rows = list(reader)

        prototypes = []
        seen_glosses = set()

        for row in manifest_rows:
            gloss = row.get("gloss", "").strip().upper()
            if not gloss or gloss in seen_glosses:
                continue
            seen_glosses.add(gloss)

            category = row.get("category", "General").strip()
            label = row.get("label_in_dataset", gloss).strip()

            si_terms = gloss_to_sinhala.get(gloss, [])
            si_str = ", ".join(list(dict.fromkeys(si_terms))[:8]) if si_terms else ""

            extra_ctx = domain_contexts.get(gloss, "")
            readable_gloss = gloss.replace("_", " ").lower()

            # Construct dense bilingual prototype text
            text_parts = [
                f"Sri Lankan Sign Language concept: {gloss}.",
                f"Category: {category}.",
                f"English words: {label}, {readable_gloss}.",
            ]
            if extra_ctx:
                text_parts.append(f"Context: {extra_ctx}.")
            if si_str:
                text_parts.append(f"Sinhala terms: {si_str}.")

            prototype_text = " ".join(text_parts)

            prototypes.append({
                "gloss": gloss,
                "label": label,
                "category": category,
                "take": row.get("take", f"{label}_001"),
                "sinhala_terms": si_terms,
                "prototype_text": prototype_text,
            })

        # Ensure deterministically sorted by gloss
        prototypes.sort(key=lambda x: x["gloss"])
        return prototypes

    def build_index(self, force_rebuild: bool = False, batch_size: int = 50) -> bool:
        """
        Builds the 383-sign prototype embedding matrix using gemini-embedding-001
        and saves it to data/sign_embeddings_383.npy and data/sign_catalog_383.json.
        Resumes from partial checkpoint if interrupted by API rate limits.
        """
        if not force_rebuild and self.is_indexed():
            return True

        prototypes = self.generate_prototypes_from_manifest()
        N = len(prototypes)
        print(f"Total canonical sign classes to index: {N}")

        partial_matrix_path = self.cache_dir / "sign_embeddings_partial.npy"
        embeddings_list = []
        start_idx = 0

        # Resume from checkpoint if available
        if partial_matrix_path.exists() and not force_rebuild:
            try:
                partial_arr = np.load(partial_matrix_path)
                embeddings_list = [row for row in partial_arr]
                start_idx = len(embeddings_list)
                print(f"Resuming embedding index from checkpoint at index {start_idx}/{N}...")
            except Exception:
                embeddings_list = []
                start_idx = 0

        client = self._get_api_client()

        for i in range(start_idx, N, batch_size):
            batch = prototypes[i:i + batch_size]
            batch_texts = [item["prototype_text"] for item in batch]

            success = False
            for attempt in range(5):
                try:
                    res = client.models.embed_content(
                        model=self.model_name,
                        contents=batch_texts,
                    )
                    for emb in res.embeddings:
                        embeddings_list.append(emb.values)

                    # Save checkpoint
                    np.save(partial_matrix_path, np.array(embeddings_list, dtype=np.float32))
                    success = True
                    break
                except Exception as e:
                    err_str = str(e)
                    is_rate_limit = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str
                    wait_sec = 42 if is_rate_limit else (attempt + 1) * 3
                    print(f"Batch at index {i} attempt {attempt + 1} hit rate limit / error. Pausing {wait_sec}s for quota window...")
                    time.sleep(wait_sec)

            if not success:
                raise RuntimeError(f"Failed to embed batch starting at index {i} after multiple retries.")

            print(f"  Indexed {len(embeddings_list)} / {N} sign classes...")

        # Convert to NumPy float32 matrix
        matrix = np.array(embeddings_list, dtype=np.float32)

        # Normalize each row to unit L2 norm: ||v|| = 1
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        matrix = matrix / norms

        # Save final complete index to disk
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.matrix_path, matrix)
        with open(self.catalog_path, "w", encoding="utf-8") as f:
            json.dump(prototypes, f, ensure_ascii=False, indent=2)

        # Remove partial checkpoint
        if partial_matrix_path.exists():
            partial_matrix_path.unlink()

        print(f"Successfully saved dense embedding matrix to: {self.matrix_path.resolve()}")
        print(f"Successfully saved sign catalog to: {self.catalog_path.resolve()}")

        self.embeddings_matrix = matrix
        self.catalog = prototypes
        self.gloss_to_index = {item["gloss"]: idx for idx, item in enumerate(self.catalog)}
        return True

    def embed_text(self, text_or_texts: Union[str, List[str]]) -> np.ndarray:
        """
        Embeds a single query string or list of strings into normalized 3072-dimensional vectors.
        Utilizes an in-memory query cache to eliminate redundant API calls.
        Returns: np.ndarray of shape (3072,) or (M, 3072).
        """
        is_single = isinstance(text_or_texts, str)
        contents = [text_or_texts] if is_single else text_or_texts

        uncached_indices = []
        uncached_texts = []
        results = [None] * len(contents)

        for idx, txt in enumerate(contents):
            if txt in self._query_cache:
                results[idx] = self._query_cache[txt]
            else:
                uncached_indices.append(idx)
                uncached_texts.append(txt)

        if uncached_texts:
            client = self._get_api_client()
            res = client.models.embed_content(
                model=self.model_name,
                contents=uncached_texts,
            )

            vectors = np.array([e.values for e in res.embeddings], dtype=np.float32)
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vectors = vectors / norms

            for u_idx, text_key, vec in zip(uncached_indices, uncached_texts, vectors):
                self._query_cache[text_key] = vec
                results[u_idx] = vec

        matrix_out = np.array(results, dtype=np.float32)
        return matrix_out[0] if is_single else matrix_out

    def compute_similarities(self, query_vec: np.ndarray) -> np.ndarray:
        """
        Computes cosine similarities against all 383 canonical sign vectors in sub-millisecond time.
        s = V * q (where both V and q are unit normalized).
        Returns: 1D array of shape (383,) with values in [-1.0, 1.0].
        """
        if not self.is_indexed():
            raise RuntimeError("Embedder index is not loaded. Call build_index() first.")

        if query_vec.ndim == 1:
            return np.dot(self.embeddings_matrix, query_vec)
        else:
            return np.dot(query_vec, self.embeddings_matrix.T)

    def get_top_k(
        self,
        query: Union[str, np.ndarray],
        k: int = 5,
        min_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Finds the top-k nearest semantic sign classes in the continuous hypersphere.
        """
        if isinstance(query, str):
            query_vec = self.embed_text(query)
        else:
            query_vec = query

        sims = self.compute_similarities(query_vec)
        top_indices = np.argsort(sims)[::-1][:k]

        results = []
        for rank, idx in enumerate(top_indices):
            score = float(sims[idx])
            if score < min_threshold:
                continue
            cat_item = self.catalog[idx]
            results.append({
                "rank": rank + 1,
                "gloss": cat_item["gloss"],
                "category": cat_item["category"],
                "similarity": round(score, 4),
                "label": cat_item["label"],
                "take": cat_item.get("take"),
            })

        return results
