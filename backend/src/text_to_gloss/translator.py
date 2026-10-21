"""Dictionary-based Text-to-Gloss translator engine with 4-Tier Hierarchical Bilingual OOV Architecture."""

import re
from typing import Dict, List, Optional, Set
from pydantic import BaseModel, Field

from .rules import (
    STOPWORDS_OMIT,
    PHRASE_MAPPINGS,
    WORD_TO_GLOSS,
    TIME_MARKERS,
    QUESTION_MARKERS,
)
from .sinhala_processor import (
    is_sinhala,
    stem_sinhala_word,
    fingerspell_sinhala_token,
    SINHALA_STOPWORDS,
    SINHALA_TO_GLOSS,
)
from .semantic_fallback import SemanticFallbackResolver, ResolutionResult


class GlossToken(BaseModel):
    """Represents an individual gloss token mapped from the original text."""
    original: str
    gloss: str
    is_fingerspelled: bool = False
    confidence: float = 1.0
    match_type: str = "exact"  # "exact", "semantic_approximation", "compound", "fingerspelled"
    source_language: str = "en"  # "en" or "si"
    note: Optional[str] = None


class TranslationResult(BaseModel):
    """Result of text-to-gloss translation."""
    original_text: str
    detected_language: str = "en"
    glosses: List[str]
    tokens: List[GlossToken]
    unmapped_words: List[str] = Field(default_factory=list)
    fingerspelled_words: List[str] = Field(default_factory=list)
    reordered: bool = False


class DictionaryGlossTranslator:
    """Bilingual (English & Sinhala) gloss translator with 4-Tier Hierarchical OOV Resolution.

    4-Tier Architecture:
    - Tier 1: Canonical Lexicon Lookup & Morphological Lemmatization (Exact Root Match)
    - Tier 2: Semantic Subsumption / Near-Synonym & Hypernym Fallback (Cosine/Confidence >= 0.70)
    - Tier 3: Compositional / Compound Sign Decomposition (e.g. hospital -> DOCTOR + PLACE)
    - Tier 4: Bilingual Dactylology / Fingerspelling (English A-Z / Sinhala අත්අකුරු හෝඩිය)
    """

    def __init__(
        self,
        custom_dictionary: Optional[Dict[str, str]] = None,
        custom_stopwords: Optional[Set[str]] = None,
        enable_reordering: bool = True,
        fingerspell_unknown: bool = False,
    ):
        self.word_to_gloss = dict(WORD_TO_GLOSS)
        if custom_dictionary:
            self.word_to_gloss.update(
                {k.lower(): v.upper() for k, v in custom_dictionary.items()}
            )

        self.stopwords = set(STOPWORDS_OMIT)
        if custom_stopwords:
            self.stopwords.update({s.lower() for s in custom_stopwords})

        self.enable_reordering = enable_reordering
        self.fingerspell_unknown = fingerspell_unknown
        self.fallback_resolver = SemanticFallbackResolver(confidence_threshold=0.70)

    def clean_text(self, text: str) -> str:
        """Normalize punctuation and whitespace while preserving meaning."""
        text = text.strip()
        # Expand common English contractions for simpler parsing
        contractions = {
            r"\bi'm\b": "i am",
            r"\bcan't\b": "cannot",
            r"\bwon't\b": "will not",
            r"\bdon't\b": "dont",
            r"\bdoesn't\b": "does not",
            r"\bdidn't\b": "did not",
            r"\bit's\b": "it is",
            r"\bwhat's\b": "what is",
            r"\bwhere's\b": "where is",
            r"\bthat's\b": "that is",
        }
        for pattern, replacement in contractions.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        return text

    def translate(self, sentence: str, language: str = "auto") -> TranslationResult:
        """
        Translate an English or Sinhala sentence into a list of Sign Language gloss tokens
        using the 4-Tier Hierarchical OOV Resolution Architecture.
        """
        if not sentence or not sentence.strip():
            return TranslationResult(
                original_text=sentence,
                detected_language="en",
                glosses=[],
                tokens=[],
                unmapped_words=[],
                fingerspelled_words=[],
                reordered=False,
            )

        # Detect language if set to auto
        detected_lang = "si" if (language == "si" or (language == "auto" and is_sinhala(sentence))) else "en"

        if detected_lang == "si":
            return self._translate_sinhala(sentence)
        else:
            return self._translate_english(sentence)

    def _translate_sinhala(self, sentence: str) -> TranslationResult:
        """Sinhala pipeline with morphological stemming, compound decompounding, and SOV ordering."""
        is_question = ("?" in sentence) or ("ද" in sentence.split())
        # Clean punctuation except Sinhala characters and spaces
        cleaned = re.sub(r"[^\w\s\u0d80-\u0dff-]", " ", sentence)
        raw_words = cleaned.split()

        tokens: List[GlossToken] = []
        unmapped: List[str] = []
        fingerspelled: List[str] = []

        # Process phrases and words
        idx = 0
        while idx < len(raw_words):
            word = raw_words[idx]

            # Check 2-word phrase matches first (e.g., 'බොහොම ස්තූතියි', 'උදේ කෑම')
            if idx + 1 < len(raw_words):
                two_word = f"{word} {raw_words[idx + 1]}"
                if two_word in SINHALA_TO_GLOSS:
                    tokens.append(
                        GlossToken(
                            original=two_word,
                            gloss=SINHALA_TO_GLOSS[two_word],
                            is_fingerspelled=False,
                            confidence=1.0,
                            match_type="exact",
                            source_language="si",
                        )
                    )
                    idx += 2
                    continue

                fallback_res = self.fallback_resolver.resolve(two_word)
                if fallback_res:
                    for g in fallback_res.glosses:
                        tokens.append(
                            GlossToken(
                                original=two_word,
                                gloss=g,
                                is_fingerspelled=False,
                                confidence=fallback_res.confidence,
                                match_type=fallback_res.match_type,
                                source_language="si",
                                note=fallback_res.note,
                            )
                        )
                    idx += 2
                    continue

            # Drop stopwords
            if word in SINHALA_STOPWORDS:
                idx += 1
                continue

            # Tier 1a: Direct Exact Match
            if word in SINHALA_TO_GLOSS:
                tokens.append(
                    GlossToken(
                        original=word,
                        gloss=SINHALA_TO_GLOSS[word],
                        is_fingerspelled=False,
                        confidence=1.0,
                        match_type="exact",
                        source_language="si",
                    )
                )
                idx += 1
                continue

            # Tier 1b: Morphological Stemming
            stemmed = stem_sinhala_word(word)
            if stemmed in SINHALA_TO_GLOSS:
                tokens.append(
                    GlossToken(
                        original=word,
                        gloss=SINHALA_TO_GLOSS[stemmed],
                        is_fingerspelled=False,
                        confidence=0.98,
                        match_type="exact",
                        source_language="si",
                        note=f"Stemmed: '{word}' -> '{stemmed}'",
                    )
                )
                idx += 1
                continue

            # Tier 3 & Tier 2: Compound Decomposition & Semantic Fallback (via original or stem)
            fallback_res = self.fallback_resolver.resolve(word) or self.fallback_resolver.resolve(stemmed)
            if fallback_res:
                for g in fallback_res.glosses:
                    tokens.append(
                        GlossToken(
                            original=word,
                            gloss=g,
                            is_fingerspelled=False,
                            confidence=fallback_res.confidence,
                            match_type=fallback_res.match_type,
                            source_language="si",
                            note=fallback_res.note,
                        )
                    )
                idx += 1
                continue

            # Tier 4: Dactylology / Fingerspelling fallback
            unmapped.append(word)
            if self.fingerspell_unknown:
                fs_token = fingerspell_sinhala_token(word)
                fingerspelled.append(word)
                tokens.append(
                    GlossToken(
                        original=word,
                        gloss=fs_token,
                        is_fingerspelled=True,
                        confidence=0.6,
                        match_type="fingerspelled",
                        source_language="si",
                        note="Sinhala dactylology",
                    )
                )
            else:
                tokens.append(
                    GlossToken(
                        original=word,
                        gloss="UNKNOWN",
                        is_fingerspelled=False,
                        confidence=0.0,
                        match_type="unmapped",
                        source_language="si",
                        note=f"Unmapped word: '{word}' (no SLSL gesture in dataset)",
                    )
                )

            idx += 1

        # Apply sign language grammatical reordering (Time-first, Wh-movement)
        reordered = False
        if self.enable_reordering and len(tokens) > 1:
            tokens, reordered = self._apply_grammatical_reordering(tokens, is_question)

        glosses = [tok.gloss for tok in tokens]

        return TranslationResult(
            original_text=sentence,
            detected_language="si",
            glosses=glosses,
            tokens=tokens,
            unmapped_words=unmapped,
            fingerspelled_words=fingerspelled,
            reordered=reordered,
        )

    def _translate_english(self, sentence: str) -> TranslationResult:
        """English pipeline with phrase matching, stopwords, semantic fallback, and sign grammar."""
        cleaned = self.clean_text(sentence)
        is_question = "?" in sentence
        text_lower = re.sub(r"[^\w\s-]", " ", cleaned.lower())

        tokens: List[GlossToken] = []
        unmapped: List[str] = []
        fingerspelled: List[str] = []

        # 1. Match multi-word phrases first
        phrase_keys = sorted(PHRASE_MAPPINGS.keys(), key=len, reverse=True)
        text_for_phrases = f" {text_lower} "

        phrase_replacements: Dict[str, List[str]] = {}
        placeholder_idx = 0

        for phrase in phrase_keys:
            padded_phrase = f" {phrase} "
            if padded_phrase in text_for_phrases:
                placeholder = f"__PHRASE_{placeholder_idx}__"
                phrase_replacements[placeholder] = PHRASE_MAPPINGS[phrase]
                text_for_phrases = text_for_phrases.replace(padded_phrase, f" {placeholder} ", 1)
                placeholder_idx += 1

        processed_tokens = text_for_phrases.strip().split()

        # 2. Process words and placeholders
        for word in processed_tokens:
            if word in phrase_replacements:
                for gloss in phrase_replacements[word]:
                    tokens.append(
                        GlossToken(
                            original="[phrase]",
                            gloss=gloss,
                            is_fingerspelled=False,
                            confidence=1.0,
                            match_type="exact",
                            source_language="en",
                        )
                    )
                continue

            # Drop English stopwords / copulas / articles
            if word in self.stopwords:
                continue

            # Tier 1: Canonical Lexicon Lookup
            if word in self.word_to_gloss:
                tokens.append(
                    GlossToken(
                        original=word,
                        gloss=self.word_to_gloss[word],
                        is_fingerspelled=False,
                        confidence=1.0,
                        match_type="exact",
                        source_language="en",
                    )
                )
                continue

            # Tier 3 & Tier 2: Compound Decomposition & Semantic Fallback
            fallback_res = self.fallback_resolver.resolve(word)
            if fallback_res:
                for g in fallback_res.glosses:
                    tokens.append(
                        GlossToken(
                            original=word,
                            gloss=g,
                            is_fingerspelled=False,
                            confidence=fallback_res.confidence,
                            match_type=fallback_res.match_type,
                            source_language="en",
                            note=fallback_res.note,
                        )
                    )
                continue

            # Tier 4: Fingerspelling fallback
            unmapped.append(word)
            if self.fingerspell_unknown:
                fs_token = f"#{word.upper()}"
                fingerspelled.append(word)
                tokens.append(
                    GlossToken(
                        original=word,
                        gloss=fs_token,
                        is_fingerspelled=True,
                        confidence=0.7,
                        match_type="fingerspelled",
                        source_language="en",
                    )
                )
            else:
                tokens.append(
                    GlossToken(
                        original=word,
                        gloss=word.upper(),
                        is_fingerspelled=False,
                        confidence=0.5,
                        match_type="unmapped",
                        source_language="en",
                    )
                )

        # 3. Apply sign language grammatical reordering
        reordered = False
        if self.enable_reordering and len(tokens) > 1:
            tokens, reordered = self._apply_grammatical_reordering(tokens, is_question)

        glosses = [tok.gloss for tok in tokens]

        return TranslationResult(
            original_text=sentence,
            detected_language="en",
            glosses=glosses,
            tokens=tokens,
            unmapped_words=unmapped,
            fingerspelled_words=fingerspelled,
            reordered=reordered,
        )

    def _apply_grammatical_reordering(
        self, tokens: List[GlossToken], is_question: bool
    ) -> tuple[List[GlossToken], bool]:
        """Apply sign language grammatical ordering:
        1. Time-Topic-Comment: Bring time words (YESTERDAY, TODAY, etc.) to the start.
        2. Wh-Movement: In Wh-questions, move the Wh-word (WHAT, WHERE, WHO, etc.) to the end.
        """
        reordered = False
        result = list(tokens)

        # 1. Time fronting: move any time marker found to the beginning
        time_tokens = []
        non_time_tokens = []
        has_time_shift = False

        for idx, tok in enumerate(result):
            if tok.gloss in TIME_MARKERS:
                time_tokens.append(tok)
                if idx != len(time_tokens) - 1:
                    has_time_shift = True
            else:
                non_time_tokens.append(tok)

        if has_time_shift and time_tokens:
            result = time_tokens + non_time_tokens
            reordered = True

        # 2. Wh-word to end if interrogative
        if is_question and len(result) > 1:
            if result[0].gloss in QUESTION_MARKERS:
                wh_tok = result.pop(0)
                result.append(wh_tok)
                reordered = True

        return result, reordered
