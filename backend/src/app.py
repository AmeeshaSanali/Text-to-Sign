"""FastAPI backend service for Text-to-Gloss translation and avatar animation dispatch."""

from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .text_to_gloss.translator import (
    DictionaryGlossTranslator,
    TranslationResult,
    GlossToken,
)
from .text_to_gloss.rules import WORD_TO_GLOSS

app = FastAPI(
    title="SLSL Avatar Engine API",
    description="Backend API for text-to-sign translation and avatar animation sequencing.",
    version="0.1.0",
)

# Enable CORS for Unity Editor, WebGL builds, and browser testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global translator instance
translator = DictionaryGlossTranslator(enable_reordering=True, fingerspell_unknown=True)


class TranslateRequest(BaseModel):
    text: str = Field(
        ...,
        json_schema_extra={"example": "I am going to the store"},
        description="Input English sentence",
    )
    fingerspell_unknown: Optional[bool] = Field(
        default=True,
        description="Whether out-of-vocabulary words should be fingerspelled"
    )
    enable_reordering: Optional[bool] = Field(
        default=True,
        description="Whether to apply Sign Language grammar reordering rules"
    )


class AnimationCue(BaseModel):
    order: int
    gloss: str
    animation_clip: str
    is_fingerspelled: bool
    estimated_duration_sec: float = 1.0


class TranslateResponse(BaseModel):
    original_text: str
    glosses: List[str]
    tokens: List[GlossToken]
    unmapped_words: List[str]
    fingerspelled_words: List[str]
    reordered: bool
    animation_sequence: List[AnimationCue]
    method: str = "rule_based_dictionary"


@app.get("/health")
def health_check():
    """Health check endpoint to verify backend status."""
    return {"status": "ok", "service": "slsl-avatar-backend", "version": "0.1.0"}


@app.post("/api/v1/translate", response_model=TranslateResponse)
def translate_text(req: TranslateRequest):
    """Translate an English text sentence into an ordered sign-gloss animation sequence."""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    # Instantiate custom translator if parameters differ from default
    if req.enable_reordering != translator.enable_reordering or req.fingerspell_unknown != translator.fingerspell_unknown:
        engine = DictionaryGlossTranslator(
            enable_reordering=req.enable_reordering,
            fingerspell_unknown=req.fingerspell_unknown,
        )
    else:
        engine = translator

    result: TranslationResult = engine.translate(req.text)

    # Build animation sequence cues for Unity consumption
    animation_sequence: List[AnimationCue] = []
    for idx, tok in enumerate(result.tokens):
        # Clip name convention: matches uppercase gloss or FS prefix
        clip_name = tok.gloss
        duration = 0.6 if tok.is_fingerspelled else 1.2
        animation_sequence.append(
            AnimationCue(
                order=idx,
                gloss=tok.gloss,
                animation_clip=clip_name,
                is_fingerspelled=tok.is_fingerspelled,
                estimated_duration_sec=duration,
            )
        )

    return TranslateResponse(
        original_text=result.original_text,
        glosses=result.glosses,
        tokens=result.tokens,
        unmapped_words=result.unmapped_words,
        fingerspelled_words=result.fingerspelled_words,
        reordered=result.reordered,
        animation_sequence=animation_sequence,
        method="rule_based_dictionary",
    )


@app.get("/api/v1/vocabulary")
def get_vocabulary():
    """List all supported glosses and base words currently in the dictionary."""
    gloss_set = sorted(set(WORD_TO_GLOSS.values()))
    return {
        "total_glosses": len(gloss_set),
        "glosses": gloss_set,
        "sample_mappings": {k: WORD_TO_GLOSS[k] for k in list(WORD_TO_GLOSS.keys())[:20]},
    }


@app.get("/api/v1/sample-sentences")
def get_sample_sentences():
    """Return pre-tested sentences that can be used directly for testing."""
    return {
        "samples": [
            {"english": "I am going to the store", "expected_glosses": ["I", "GO", "STORE"]},
            {"english": "Hello, how are you?", "expected_glosses": ["HELLO", "HOW", "YOU"]},
            {"english": "I went to school yesterday", "expected_glosses": ["YESTERDAY", "I", "GO", "SCHOOL"]},
            {"english": "What is your name?", "expected_glosses": ["YOUR", "NAME", "WHAT"]},
            {"english": "Thank you my friend", "expected_glosses": ["THANK-YOU", "MY", "FRIEND"]},
            {"english": "Please help me", "expected_glosses": ["PLEASE", "HELP", "I"]},
        ]
    }
