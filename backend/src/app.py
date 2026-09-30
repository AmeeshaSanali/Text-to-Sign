"""FastAPI backend service for Text-to-Gloss translation and avatar animation dispatch."""

from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
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


@app.get("/", response_class=HTMLResponse)
def root_dashboard():
    """Interactive web dashboard and quick translation tester."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>SLSL Avatar Engine — Quick Tester</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 30px; display: flex; justify-content: center; }
    .card { background: #1e293b; border-radius: 12px; padding: 28px; max-width: 680px; width: 100%; box-shadow: 0 10px 25px rgba(0,0,0,0.5); }
    h1 { margin-top: 0; font-size: 22px; display: flex; align-items: center; gap: 10px; }
    .status { background: #166534; color: #86efac; padding: 4px 10px; border-radius: 20px; font-size: 13px; font-weight: normal; }
    p { color: #94a3b8; font-size: 14px; line-height: 1.5; }
    input[type="text"] { width: 100%; padding: 12px 14px; border-radius: 8px; border: 1px solid #334155; background: #0f172a; color: #fff; font-size: 15px; box-sizing: border-box; margin-bottom: 12px; }
    button { background: #3b82f6; color: white; border: none; padding: 10px 18px; border-radius: 6px; font-size: 14px; font-weight: 600; cursor: pointer; transition: background 0.2s; }
    button:hover { background: #2563eb; }
    .badge { display: inline-block; background: #3b82f6; color: white; font-weight: bold; padding: 6px 12px; border-radius: 6px; margin: 4px; font-family: monospace; font-size: 15px; }
    .badge-fs { background: #d97706; }
    #result-box { margin-top: 20px; padding: 16px; border-radius: 8px; background: #0f172a; border: 1px solid #334155; display: none; }
    .links { margin-top: 24px; padding-top: 18px; border-top: 1px solid #334155; font-size: 13px; display: flex; gap: 16px; }
    .links a { color: #60a5fa; text-decoration: none; }
    .links a:hover { text-decoration: underline; }
    .samples { margin-top: 12px; font-size: 13px; color: #64748b; }
    .sample-btn { background: none; border: 1px solid #334155; color: #94a3b8; padding: 4px 8px; border-radius: 4px; font-size: 12px; cursor: pointer; margin: 2px; }
    .sample-btn:hover { background: #334155; color: #fff; }
  </style>
</head>
<body>
  <div class="card">
    <h1>SLSL Avatar Engine <span class="status">🟢 Backend Online</span></h1>
    <p>Test the <b>Day 1 Text-to-Gloss engine</b> directly in your browser or connect it to Unity.</p>

    <input type="text" id="sentence" value="I am going to the store" placeholder="Type an English sentence...">
    <div>
      <button onclick="translateText()">Translate to Sign Glosses</button>
    </div>

    <div class="samples">
      Try samples:
      <button class="sample-btn" onclick="setSample('I am going to the store')">"I am going to the store"</button>
      <button class="sample-btn" onclick="setSample('Hello, how are you?')">"Hello, how are you?"</button>
      <button class="sample-btn" onclick="setSample('I went to school yesterday')">"I went to school yesterday"</button>
      <button class="sample-btn" onclick="setSample('What is your name?')">"What is your name?"</button>
    </div>

    <div id="result-box">
      <div style="font-size: 13px; color: #94a3b8; margin-bottom: 6px;">Output Gloss Sequence:</div>
      <div id="glosses-container"></div>
      <div id="meta-container" style="margin-top: 10px; font-size: 12px; color: #64748b;"></div>
    </div>

    <div class="links">
      <a href="/docs" target="_blank">📖 Swagger API Docs</a>
      <a href="/health" target="_blank">🩺 Health Check</a>
      <a href="/api/v1/vocabulary" target="_blank">📚 Vocabulary List</a>
      <a href="/api/v1/sample-sentences" target="_blank">🧪 Sample Sentences</a>
    </div>
  </div>

  <script>
    async function translateText() {
      const text = document.getElementById('sentence').value;
      if (!text.trim()) return;
      const res = await fetch('/api/v1/translate', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({text: text, enable_reordering: true, fingerspell_unknown: true})
      });
      const data = await res.json();
      const container = document.getElementById('glosses-container');
      const meta = document.getElementById('meta-container');
      container.innerHTML = '';

      data.glosses.forEach(g => {
        const span = document.createElement('span');
        span.className = g.startsWith('#') ? 'badge badge-fs' : 'badge';
        span.textContent = g;
        container.appendChild(span);
      });

      meta.innerHTML = `Original: "${data.original_text}" | Total Glosses: ${data.glosses.length} ${data.reordered ? ' | <i>(Grammar reordered)</i>' : ''}`;
      document.getElementById('result-box').style.display = 'block';
    }
    function setSample(str) {
      document.getElementById('sentence').value = str;
      translateText();
    }
    translateText();
  </script>
</body>
</html>"""


class TranslateRequest(BaseModel):
    text: str = Field(
        ...,
        json_schema_extra={"example": "I am going to the store"},
        description="Input English sentence",
    )
    fingerspell_unknown: Optional[bool] = Field(
        default=True,
        description="Whether out-of-vocabulary words should be fingerspelled",
    )
    enable_reordering: Optional[bool] = Field(
        default=True,
        description="Whether to apply Sign Language grammar reordering rules",
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
