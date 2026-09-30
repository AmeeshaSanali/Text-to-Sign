"""Dictionary-based Text-to-Gloss translator engine."""

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


class GlossToken(BaseModel):
    """Represents an individual gloss token mapped from the original text."""
    original: str
    gloss: str
    is_fingerspelled: bool = False
    confidence: float = 1.0


class TranslationResult(BaseModel):
    """Result of text-to-gloss translation."""
    original_text: str
    glosses: List[str]
    tokens: List[GlossToken]
    unmapped_words: List[str] = Field(default_factory=list)
    fingerspelled_words: List[str] = Field(default_factory=list)
    reordered: bool = False


class DictionaryGlossTranslator:
    """Dictionary-based gloss translator with grammatical reordering and fallback.

    Fulfills the Day 1 Quick Start deliverable:
    - Drops articles and copulas (e.g., 'am', 'the', 'to').
    - Maps inflected words to base glosses (e.g., 'going' -> 'GO').
    - Implements sign language grammatical ordering (Time-first, Wh-question end).
    - Supports fingerspelling fallback for out-of-vocabulary words.
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

    def clean_text(self, text: str) -> str:
        """Normalize punctuation and whitespace while preserving meaning."""
        text = text.strip()
        # Expand common contractions for simpler parsing
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

    def translate(self, sentence: str) -> TranslationResult:
        """Translate an English sentence into a list of Sign Language gloss tokens."""
        if not sentence or not sentence.strip():
            return TranslationResult(
                original_text=sentence,
                glosses=[],
                tokens=[],
                unmapped_words=[],
                fingerspelled_words=[],
                reordered=False,
            )

        cleaned = self.clean_text(sentence)
        is_question = "?" in sentence
        # Lowercase and strip punctuation for tokenization
        text_lower = re.sub(r"[^\w\s-]", " ", cleaned.lower())

        tokens: List[GlossToken] = []
        unmapped: List[str] = []
        fingerspelled: List[str] = []

        # 1. Match multi-word phrases first (e.g., 'thank you', 'how are you')
        matched_ranges: List[tuple[int, int]] = []
        words = text_lower.split()
        remaining_words: List[str] = []

        # Check for phrase matches in order of phrase length (longest first)
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
                        )
                    )
                continue

            # Drop stopwords / copulas / articles
            if word in self.stopwords:
                continue

            # Map to dictionary gloss
            if word in self.word_to_gloss:
                tokens.append(
                    GlossToken(
                        original=word,
                        gloss=self.word_to_gloss[word],
                        is_fingerspelled=False,
                        confidence=1.0,
                    )
                )
            else:
                unmapped.append(word)
                if self.fingerspell_unknown:
                    # Treat unknown words as fingerspelled: e.g. #JOHN or letters
                    fs_token = f"#{word.upper()}"
                    fingerspelled.append(word)
                    tokens.append(
                        GlossToken(
                            original=word,
                            gloss=fs_token,
                            is_fingerspelled=True,
                            confidence=0.7,
                        )
                    )
                else:
                    # Default: uppercase the word directly
                    tokens.append(
                        GlossToken(
                            original=word,
                            gloss=word.upper(),
                            is_fingerspelled=False,
                            confidence=0.5,
                        )
                    )

        # 3. Apply grammatical reordering rules (if enabled)
        reordered = False
        if self.enable_reordering and len(tokens) > 1:
            reordered_tokens, did_reorder = self._apply_grammatical_reordering(
                tokens, is_question
            )
            tokens = reordered_tokens
            reordered = did_reorder

        glosses = [tok.gloss for tok in tokens]

        return TranslationResult(
            original_text=sentence,
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
            # If the first token is a question word, move it to the end
            if result[0].gloss in QUESTION_MARKERS:
                wh_tok = result.pop(0)
                result.append(wh_tok)
                reordered = True

        return result, reordered
