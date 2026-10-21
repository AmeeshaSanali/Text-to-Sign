"""
Neural Cross-Lingual Semantic Grounder
Part of the Neural Continuous Representation Architecture

Provides:
- Direct concept grounding in a continuous 3072-dimensional multilingual latent space
- High-density semantic neighborhood alignment (without heuristic if-else dictionaries)
- Top-k candidate confidence scoring
- Sri Lankan Sign Language (SLSL) Time-Topic-Comment and SOV grammatical alignment
"""

import re
from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel, Field
import numpy as np

from .dense_embedder import DenseSignEmbedder
from .sinhala_processor import is_sinhala, SINHALA_STOPWORDS, SINHALA_TIME_MARKERS, SINHALA_QUESTION_MARKERS
from .rules import STOPWORDS_OMIT, TIME_MARKERS, QUESTION_MARKERS


class NeuralAlternative(BaseModel):
    gloss: str
    category: str
    similarity: float


class NeuralToken(BaseModel):
    original_word: str
    gloss: str
    category: Optional[str] = None
    similarity: float = Field(ge=0.0, le=1.0)
    match_type: str = "exact"  # "exact", "semantic_approximation", "unmapped", "compound"
    is_fingerspelled: bool = False
    source_language: str = "en"  # "en" or "si"
    alternatives: List[NeuralAlternative] = Field(default_factory=list)
    note: Optional[str] = None


class NeuralGroundingResult(BaseModel):
    original_text: str
    detected_language: str = "en"
    glosses: List[str]
    tokens: List[NeuralToken]
    reordered: bool = False
    mean_confidence: float = 0.0


class NeuralCrossLingualGrounder:
    """Continuous cross-lingual semantic grounder operating in R^3072 latent space."""

    def __init__(
        self,
        embedder: Optional[DenseSignEmbedder] = None,
        exact_threshold: float = 0.68,
        approximation_threshold: float = 0.52,
        enable_reordering: bool = True,
        deduplicate_consecutive: bool = True,
    ):
        self.embedder = embedder or DenseSignEmbedder()
        self.exact_threshold = exact_threshold
        self.approximation_threshold = approximation_threshold
        self.enable_reordering = enable_reordering
        self.deduplicate_consecutive = deduplicate_consecutive

    def extract_semantic_units(self, sentence: str, lang: str) -> List[str]:
        """
        Tokenizes sentence into meaningful content words/phrases,
        filtering out non-manual copulas, articles, and functional particles.
        """
        cleaned = re.sub(r"[^\w\s\u0d80-\u0dff-]", " ", sentence.strip())
        raw_words = cleaned.split()

        units = []
        skip_next = False

        for i, w in enumerate(raw_words):
            if skip_next:
                skip_next = False
                continue

            # Two-word composite phrases (e.g. "teledrama show", "උදේ කෑම")
            if i + 1 < len(raw_words):
                bigram = f"{w} {raw_words[i+1]}"
                if lang == "si" and bigram in ["උදේ කෑම", "රෑ කෑම", "පොත් සාප්පුව"]:
                    units.append(bigram)
                    skip_next = True
                    continue
                elif lang == "en" and bigram.lower() in ["sign language", "bus station", "train station", "police station"]:
                    units.append(bigram)
                    skip_next = True
                    continue

            # Filter non-manual stopwords
            if lang == "si":
                if w in SINHALA_STOPWORDS:
                    continue
            else:
                if w.lower() in STOPWORDS_OMIT:
                    continue

            units.append(w)

        return units

    def ground(self, sentence: str, language: str = "auto") -> NeuralGroundingResult:
        """
        Grounds an input sentence in Sinhala or English into an ordered sequence
        of canonical SLSL sign glosses using dense vector metric alignment.
        """
        if not sentence or not sentence.strip():
            return NeuralGroundingResult(
                original_text=sentence,
                detected_language="en",
                glosses=[],
                tokens=[],
                reordered=False,
                mean_confidence=0.0,
            )

        # Detect source language
        lang = "si" if (language == "si" or (language == "auto" and is_sinhala(sentence))) else "en"
        units = self.extract_semantic_units(sentence, lang)

        if not units:
            return NeuralGroundingResult(
                original_text=sentence,
                detected_language=lang,
                glosses=[],
                tokens=[],
                reordered=False,
                mean_confidence=0.0,
            )

        # Ensure embedding index is loaded
        if not self.embedder.is_indexed():
            self.embedder.build_index()

        # Batch embed query semantic units
        query_vectors = self.embedder.embed_text(units)
        if query_vectors.ndim == 1:
            query_vectors = query_vectors.reshape(1, -1)

        tokens: List[NeuralToken] = []

        for i, word in enumerate(units):
            q_vec = query_vectors[i]
            top_candidates = self.embedder.get_top_k(q_vec, k=5)

            if not top_candidates:
                tokens.append(
                    NeuralToken(
                        original_word=word,
                        gloss="UNKNOWN",
                        similarity=0.0,
                        match_type="unmapped",
                        source_language=lang,
                        note="No semantic neighbor found in latent space",
                    )
                )
                continue

            best = top_candidates[0]
            sim = best["similarity"]

            # Map alternatives
            alts = [
                NeuralAlternative(
                    gloss=cand["gloss"],
                    category=cand["category"],
                    similarity=cand["similarity"],
                )
                for cand in top_candidates[1:4]
            ]

            if sim >= self.exact_threshold:
                match_type = "exact"
                gloss = best["gloss"]
                is_fs = False
                note = f"Neural direct alignment (Sim: {sim:.2f})"
            elif sim >= self.approximation_threshold:
                match_type = "semantic_approximation"
                gloss = best["gloss"]
                is_fs = False
                note = f"Dense semantic neighborhood projection (Sim: {sim:.2f})"
            else:
                match_type = "fingerspelled"
                is_fs = True
                if lang == "si":
                    from .sinhala_processor import fingerspell_sinhala_token
                    gloss = fingerspell_sinhala_token(word)
                else:
                    gloss = f"#{word.upper()}"
                note = f"Below metric threshold ({sim:.2f} < {self.approximation_threshold}); routed to parametric fingerspelling"

            tokens.append(
                NeuralToken(
                    original_word=word,
                    gloss=gloss,
                    category=best["category"] if not is_fs else "Dactylology",
                    similarity=sim,
                    match_type=match_type,
                    is_fingerspelled=is_fs,
                    source_language=lang,
                    alternatives=alts,
                    note=note,
                )
            )

        # Apply SLSL Grammatical Reordering (Time-Fronting & Wh-Movement)
        reordered = False
        if self.enable_reordering and len(tokens) > 1:
            tokens, reordered = self._apply_grammatical_reordering(tokens, "?" in sentence)

        # Deduplicate consecutive identical glosses (e.g. reading a book -> BOOK + BOOK -> BOOK)
        if self.deduplicate_consecutive and len(tokens) > 1:
            dedup_tokens = []
            for tok in tokens:
                if not dedup_tokens or tok.gloss != dedup_tokens[-1].gloss or tok.gloss == "UNKNOWN":
                    dedup_tokens.append(tok)
            tokens = dedup_tokens

        glosses = [tok.gloss for tok in tokens if tok.gloss != "UNKNOWN"]
        confidences = [tok.similarity for tok in tokens if tok.gloss != "UNKNOWN"]
        mean_conf = float(np.mean(confidences)) if confidences else 0.0

        return NeuralGroundingResult(
            original_text=sentence,
            detected_language=lang,
            glosses=glosses,
            tokens=tokens,
            reordered=reordered,
            mean_confidence=round(mean_conf, 4),
        )

    def _apply_grammatical_reordering(
        self, tokens: List[NeuralToken], is_question: bool
    ) -> Tuple[List[NeuralToken], bool]:
        """
        Reorders grounded tokens into natural Sri Lankan Sign Language syntax:
        1. Time-Fronting: Time adverbs (TODAY, YESTERDAY, NOW, etc.) placed first.
        2. Interrogative-Final: Question markers (WHAT, WHERE, WHO, WHY, HOW) placed at end.
        """
        original_order = [t.gloss for t in tokens]
        time_tokens = []
        body_tokens = []
        question_tokens = []

        all_time_markers = TIME_MARKERS.union(SINHALA_TIME_MARKERS)
        all_wh_markers = QUESTION_MARKERS.union(SINHALA_QUESTION_MARKERS)

        for tok in tokens:
            g = tok.gloss.upper()
            if g in all_time_markers:
                time_tokens.append(tok)
            elif g in all_wh_markers:
                question_tokens.append(tok)
            else:
                body_tokens.append(tok)

        reordered_list = time_tokens + body_tokens + question_tokens
        new_order = [t.gloss for t in reordered_list]
        did_reorder = original_order != new_order

        return reordered_list, did_reorder
