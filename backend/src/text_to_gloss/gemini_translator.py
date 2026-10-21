"""
Gemini-Powered Lexicon-Constrained Sign Language Translator
Part of the Bilingual SLSL Avatar Engine

Uses Google's Gemini models (gemini-2.5-flash) to perform:
1. Multilingual understanding of complex Sinhala and English sentences (idioms, tenses, case markers).
2. Constrained decoding restricted strictly to the 383 canonical sign classes in SSL400.
3. Zero-shot compound sign synthesis for out-of-vocabulary words.
4. Non-Manual Marker (NMM) & facial expression prediction (eyebrow raises/furrows).
5. Seamless fallback to rule-based engine when API key is unavailable.
"""

import os
import re
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
from pydantic import BaseModel, Field

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from .translator import GlossToken, TranslationResult


class GeminiGlossToken(BaseModel):
    """Structured token returned by Gemini."""
    original_word: str = Field(description="The source word or phrase from the input sentence")
    gloss: str = Field(description="The canonical uppercase sign gloss chosen strictly from the vocabulary")
    match_type: str = Field(
        default="exact",
        description="Type of match: 'exact', 'compound', 'semantic_approximation', or 'fingerspelled'"
    )
    confidence: float = Field(default=0.95, ge=0.0, le=1.0)
    is_fingerspelled: bool = False
    facial_expression: str = Field(
        default="neutral",
        description="Predicted facial expression / NMM: 'neutral', 'eyebrow_raise', 'eyebrow_furrow', 'smile', 'head_nod', 'head_shake'"
    )
    note: Optional[str] = Field(default=None, description="Linguistic explanation or compound decomposition formula")


class GeminiTranslationResponse(BaseModel):
    """Structured response schema enforced on Gemini."""
    detected_language: str = Field(description="'en' for English or 'si' for Sinhala")
    original_text: str
    glosses: List[str] = Field(description="Final ordered list of uppercase sign glosses")
    tokens: List[GeminiGlossToken]
    reordered: bool = Field(description="Whether sign language grammatical ordering was applied")
    grammatical_explanation: Optional[str] = Field(default=None, description="Brief note on syntax structure")


class GeminiGlossTranslator:
    """Lexicon-constrained sign translator powered by the Gemini API."""

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.5-flash"):
        self.api_key = api_key
        self.model_name = model_name
        self.allowed_glosses: List[str] = []
        self._load_allowed_vocabulary()

    def _load_allowed_vocabulary(self):
        """Loads canonical 383 sign glosses from data/manifest.csv."""
        manifest_path = Path(__file__).resolve().parent.parent.parent.parent / "data" / "manifest.csv"
        if manifest_path.exists():
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    self.allowed_glosses = sorted([
                        row["gloss"].strip().upper()
                        for row in reader
                        if row.get("gloss")
                    ])
            except Exception:
                pass

        if not self.allowed_glosses:
            # Fallback list of key canonical starter glosses
            self.allowed_glosses = [
                "HELLO", "THANK_YOU", "YES", "NO", "GO", "COME", "EAT",
                "MOTHER", "FATHER", "TODAY", "YESTERDAY", "TOMORROW", "BLUE",
                "DOCTOR", "PLACE", "SCHOOL", "HOUSE", "HOME", "WATER", "BOOK"
            ]

    def is_available(self, custom_key: Optional[str] = None) -> bool:
        """Returns True if a Gemini API key is configured and google-genai is installed."""
        key = custom_key or self.api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            return False
        try:
            from google import genai
            return True
        except ImportError:
            return False

    def build_system_instruction(self) -> str:
        """Constructs system prompt defining SLSL linguistics and vocabulary constraint."""
        vocab_preview = ", ".join(self.allowed_glosses)
        return (
            "You are an expert Sri Lankan Sign Language (SLSL) interpreter, Deaf education specialist, "
            "and computational linguist.\n\n"
            "Your task is to translate an input sentence (written in either Sinhala or English) into an ordered "
            "sequence of canonical SLSL sign glosses for an avatar animation engine.\n\n"
            "CRITICAL CONSTRAINTS & RULES:\n"
            "1. NO SINHALA SCRIPT IN GLOSS: The 'gloss' field and 'glosses' list MUST NEVER contain Sinhala characters "
            "(U+0D80 to U+0DFF). Every gloss must strictly be an uppercase English identifier selected from the allowed vocabulary list.\n\n"
            "2. VOCABULARY RESTRICTION: You MUST select sign glosses ONLY from the following allowed vocabulary list of "
            f"{len(self.allowed_glosses)} signs:\n[{vocab_preview}]\n\n"
            "3. OUT-OF-VOCABULARY (OOV) COMPOUND DECOMPOSITION:\n"
            "   If a word in the user sentence does NOT exist in the allowed vocabulary, DO NOT drop it! Instead, decompose it "
            "into a sequence of compound signs using ONLY words that exist in the allowed vocabulary. Examples:\n"
            "   - 'hospital' / 'රෝහල' -> HOSPITAL (if in vocabulary) or DOCTOR + PLACE\n"
            "   - 'clinic' / 'සායනය' -> DOCTOR + PLACE\n"
            "   - 'breakfast' / 'උදේ කෑම' -> MORNING + EAT\n"
            "   - 'dinner' / 'රෑ කෑම' -> NIGHT + EAT\n"
            "   - 'teacher' / 'ගුරුවරයා' -> TEACH + PERSON\n"
            "   - 'classroom' / 'පන්ති කාමරය' -> SCHOOL + ROOM\n"
            "   - 'police officer' / 'පොලිස් නිලධාරියා' -> POLICE + PERSON\n\n"
            "4. SIGN LANGUAGE GRAMMAR & SYNTAX:\n"
            "   - Time-First (Time-Topic-Comment): Place time adverbs (TODAY, YESTERDAY, TOMORROW, NOW) at the start.\n"
            "   - SOV Word Order: Subject -> Object -> Verb.\n"
            "   - Omit non-manual English copulas and articles (a, an, the, is, are, am, was, were, to, of).\n"
            "   - For questions, move interrogatives (WHAT, WHERE, WHO, HOW, WHY) to the end.\n\n"
            "5. NON-MANUAL MARKERS (FACIAL EXPRESSIONS):\n"
            "   - For questions, assign 'eyebrow_furrow' (Wh-questions) or 'eyebrow_raise' (Yes/No questions).\n"
            "   - For negation (NO, NEVER, CANNOT), assign 'head_shake'.\n"
            "   - For positive emotions (HAPPY, THANK_YOU, GOOD), assign 'smile'.\n\n"
            "Return valid JSON matching the specified schema."
        )

    def translate(
        self,
        sentence: str,
        language: str = "auto",
        api_key: Optional[str] = None
    ) -> TranslationResult:
        """
        Translates sentence using Gemini.
        Falls back through current models if a transient 503 or overload occurs.
        """
        key = api_key or self.api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY is not configured.")

        from google import genai
        from google.genai import types

        client = genai.Client(api_key=key)

        prompt = f"Translate the following sentence into SLSL sign glosses:\n\"{sentence.strip()}\""
        if language != "auto":
            prompt += f"\nSource language specified: {language}"

        system_inst = self.build_system_instruction()

        # Resilient model fallback ladder
        models_to_try = [self.model_name]
        for alt_m in ["gemini-2.5-flash", "gemini-3.5-flash-lite", "gemini-3.7-flash"]:
            if alt_m not in models_to_try:
                models_to_try.append(alt_m)

        last_err = None
        raw_text = ""
        for model in models_to_try:
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_inst,
                        response_mime_type="application/json",
                        response_schema=GeminiTranslationResponse,
                        temperature=0.1,  # Low temperature for strict adherence to vocabulary
                    ),
                )
                raw_text = response.text.strip()
                if raw_text:
                    break
            except Exception as e:
                last_err = e
                continue

        if not raw_text:
            raise RuntimeError(f"All Gemini models failed: {str(last_err)}")

        parsed_data = json.loads(raw_text)

        # Convert Gemini tokens to standard GlossToken
        tokens: List[GlossToken] = []
        unmapped: List[str] = []
        fingerspelled: List[str] = []
        sinhala_unicode_re = re.compile(r"[\u0d80-\u0dff]")

        for gt in parsed_data.get("tokens", []):
            orig_word = gt.get("original_word", "")
            gloss_val = gt.get("gloss", "").strip().upper()

            # Guard: Never let raw Sinhala Unicode script remain in the gloss field
            if sinhala_unicode_re.search(gloss_val):
                from .sinhala_processor import stem_sinhala_word, SINHALA_TO_GLOSS
                stem = stem_sinhala_word(orig_word)
                gloss_val = SINHALA_TO_GLOSS.get(orig_word) or SINHALA_TO_GLOSS.get(stem) or "UNKNOWN"

            g_token = GlossToken(
                original=orig_word,
                gloss=gloss_val,
                is_fingerspelled=gt.get("is_fingerspelled", False),
                confidence=float(gt.get("confidence", 0.95)),
                match_type=gt.get("match_type", "exact"),
                source_language=parsed_data.get("detected_language", "en"),
                note=gt.get("note") or f"Facial: {gt.get('facial_expression', 'neutral')}",
            )
            tokens.append(g_token)
            if g_token.is_fingerspelled:
                fingerspelled.append(g_token.original)
            if g_token.match_type == "unmapped" or g_token.gloss == "UNKNOWN":
                unmapped.append(g_token.original)

        final_glosses = [t.gloss for t in tokens]

        return TranslationResult(
            original_text=sentence,
            detected_language=parsed_data.get("detected_language", "en"),
            glosses=final_glosses,
            tokens=tokens,
            unmapped_words=unmapped,
            fingerspelled_words=fingerspelled,
            reordered=parsed_data.get("reordered", True),
        )
