import os
import ast
import csv
from pathlib import Path
from typing import List, Optional, Any, Dict
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Automatically load .env from project root or backend directory
root_dir = Path(__file__).resolve().parent.parent.parent
for env_candidate in [root_dir / ".env", Path(__file__).resolve().parent.parent / ".env"]:
    if env_candidate.exists():
        load_dotenv(dotenv_path=env_candidate)

from .text_to_gloss.translator import (
    DictionaryGlossTranslator,
    TranslationResult,
    GlossToken,
)
from .text_to_gloss.gemini_translator import GeminiGlossTranslator
from .text_to_gloss.rules import WORD_TO_GLOSS
from .animation.library import get_animation_library
from .animation.fingerspelling_generator import get_fingerspelling_generator

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

# Mount static animations directory for runtime JSON loading
animations_dir = Path(__file__).resolve().parent.parent.parent / "data" / "animations"
if animations_dir.exists():
    app.mount("/animations", StaticFiles(directory=str(animations_dir)), name="animations")

# Global translator instances
translator = DictionaryGlossTranslator(enable_reordering=True, fingerspell_unknown=True)
gemini_translator = GeminiGlossTranslator()
from .text_to_gloss.neural_grounder import NeuralCrossLingualGrounder
neural_grounder = NeuralCrossLingualGrounder()
fs_generator = get_fingerspelling_generator()


visualizer_html_path = Path(__file__).resolve().parent / "static" / "visualizer.html"


@app.get("/", response_class=HTMLResponse)
def root_dashboard():
    """Serves the primary SLSL 2D Pose Coordinate Avatar Visualizer."""
    if visualizer_html_path.exists():
        return FileResponse(str(visualizer_html_path))
    return HTMLResponse("<h2>Visualizer not found</h2>", status_code=404)


class TranslateRequest(BaseModel):
    text: str = Field(
        ...,
        json_schema_extra={"example": "I am going to the store"},
        description="Input English or Sinhala sentence",
    )
    language: Optional[str] = Field(
        default="auto",
        description="Source language ('auto', 'en', or 'si')",
    )
    engine: Optional[str] = Field(
        default="hybrid",
        description="Translation engine: 'gemini', 'rules', or 'hybrid'",
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
    animation_url: Optional[str] = None
    has_pose_animation: bool = False
    sign_status: str = "no_sign"  # "approved", "clip_fallback", "no_sign"
    match_type: str = "exact"     # "exact", "semantic_approximation", "compound", "fingerspelled"
    confidence: float = 1.0
    note: Optional[str] = None


class TranslateResponse(BaseModel):
    original_text: str
    detected_language: str = "en"
    glosses: List[str]
    tokens: List[GlossToken]
    unmapped_words: List[str]
    fingerspelled_words: List[str]
    reordered: bool
    animation_sequence: List[AnimationCue]
    method: str = "hierarchical_bilingual_oov"


@app.get("/health")
def health_check():
    """Health check endpoint to verify backend status."""
    return {"status": "ok", "service": "slsl-avatar-backend", "version": "0.1.0"}


@app.get("/api/v1/engine-status")
def get_engine_status():
    """Returns availability of translation engines (Gemini, Neural Grounder, and local 4-tier rules)."""
    if not os.getenv("GEMINI_API_KEY"):
        for env_path in [root_dir / ".env", Path(__file__).resolve().parent.parent / ".env"]:
            if env_path.exists():
                load_dotenv(dotenv_path=env_path)
    gemini_key_present = bool(os.getenv("GEMINI_API_KEY"))
    gemini_ready = gemini_translator.is_available()
    neural_ready = neural_grounder.embedder.is_indexed()
    return {
        "gemini_available": gemini_ready,
        "neural_available": neural_ready,
        "gemini_key_configured": gemini_key_present,
        "model": "gemini-2.5-flash",
        "embedding_model": "gemini-embedding-001 (R^3072)",
        "default_engine": "neural" if neural_ready else ("gemini" if gemini_ready else "rules"),
        "vocabulary_count": len(gemini_translator.allowed_glosses),
    }


@app.post("/api/v1/translate", response_model=TranslateResponse)
def translate_text(req: TranslateRequest):
    """Translate an English or Sinhala text sentence into an ordered sign-gloss animation sequence."""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    if not os.getenv("GEMINI_API_KEY"):
        for env_path in [root_dir / ".env", Path(__file__).resolve().parent.parent / ".env"]:
            if env_path.exists():
                load_dotenv(dotenv_path=env_path)

    # Determine whether to use Gemini or local rule engine
    if not os.getenv("GEMINI_API_KEY"):
        for env_path in [root_dir / ".env", Path(__file__).resolve().parent.parent / ".env"]:
            if env_path.exists():
                load_dotenv(dotenv_path=env_path)

    use_gemini = (req.engine in ["gemini", "hybrid"]) and gemini_translator.is_available()
    method_used = "hierarchical_bilingual_oov"

    if use_gemini:
        try:
            result: TranslationResult = gemini_translator.translate(
                req.text, language=req.language or "auto"
            )
            method_used = "gemini_constrained_llm"
        except Exception as e:
            # Fallback to local 4-tier engine
            if req.enable_reordering != translator.enable_reordering or req.fingerspell_unknown != translator.fingerspell_unknown:
                engine = DictionaryGlossTranslator(
                    enable_reordering=req.enable_reordering,
                    fingerspell_unknown=req.fingerspell_unknown,
                )
            else:
                engine = translator
            result: TranslationResult = engine.translate(req.text, language=req.language or "auto")
            method_used = f"hybrid_rules_fallback ({str(e)})"
    else:
        if req.enable_reordering != translator.enable_reordering or req.fingerspell_unknown != translator.fingerspell_unknown:
            engine = DictionaryGlossTranslator(
                enable_reordering=req.enable_reordering,
                fingerspell_unknown=req.fingerspell_unknown,
            )
        else:
            engine = translator
        result: TranslationResult = engine.translate(req.text, language=req.language or "auto")

    anim_lib = get_animation_library()

    # Build animation sequence cues for playback consumption
    animation_sequence: List[AnimationCue] = []
    for idx, tok in enumerate(result.tokens):
        clip_name = tok.gloss
        duration = 0.6 if tok.is_fingerspelled else 1.2
        anim_info = anim_lib.get_animation_info(tok.gloss)

        if anim_info:
            anim_url = anim_info["url"]
            has_pose = True
            status = "approved"
        elif tok.gloss in ["SALUTE", "SHOOTING_ARROW"]:
            anim_url = None
            has_pose = False
            status = "clip_fallback"
        else:
            anim_url = None
            has_pose = False
            status = "no_sign"

        animation_sequence.append(
            AnimationCue(
                order=idx,
                gloss=tok.gloss,
                animation_clip=clip_name,
                is_fingerspelled=tok.is_fingerspelled,
                estimated_duration_sec=duration,
                animation_url=anim_url,
                has_pose_animation=has_pose,
                sign_status=status,
                match_type=tok.match_type,
                confidence=tok.confidence,
                note=tok.note,
            )
        )

    return TranslateResponse(
        original_text=result.original_text,
        detected_language=result.detected_language,
        glosses=result.glosses,
        tokens=result.tokens,
        unmapped_words=result.unmapped_words,
        fingerspelled_words=result.fingerspelled_words,
        reordered=result.reordered,
        animation_sequence=animation_sequence,
        method=method_used,
    )


@app.get("/api/v1/animations/approved")
def list_approved_animations():
    """List all currently approved SLSL pose animations available for avatar playback."""
    anim_lib = get_animation_library()
    approved = anim_lib.list_approved_glosses()
    return {
        "count": len(approved),
        "approved_glosses": approved,
        "items": [anim_lib.get_animation_info(g) for g in approved],
    }


@app.get("/api/v1/animations/manifest")
def get_manifest():
    """Get full manifest information for all 383 dataset signs."""
    anim_lib = get_animation_library()
    return {
        "total_signs": len(anim_lib._manifest),
        "signs": anim_lib.list_all_manifest_entries(),
    }


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


# =====================================================================
# 2D Pose Coordinate Animation & Dataset Endpoints
# =====================================================================

_project_root = Path(__file__).resolve().parent.parent.parent
_env_dataset = os.getenv("DATASET_CSV_ROOT")
if _env_dataset and Path(_env_dataset).exists():
    dataset_csv_root = Path(_env_dataset)
elif (_project_root / "Dataset - MP - CSV").exists():
    dataset_csv_root = _project_root / "Dataset - MP - CSV"
elif (_project_root / "Dataset" / "Dataset - MP - CSV").exists():
    dataset_csv_root = _project_root / "Dataset" / "Dataset - MP - CSV"
else:
    dataset_csv_root = _project_root / "Dataset - MP - CSV"
visualizer_html_path = Path(__file__).resolve().parent / "static" / "visualizer.html"


@app.get("/visualizer", response_class=HTMLResponse)
def get_visualizer_page():
    """Serve interactive 2D pose coordinate animation player."""
    if visualizer_html_path.exists():
        return FileResponse(str(visualizer_html_path))
    raise HTTPException(status_code=404, detail="Visualizer HTML not found.")


@app.get("/api/v1/dataset/categories")
def list_dataset_categories():
    """List all categories available in the MediaPipe CSV dataset."""
    if not dataset_csv_root.exists():
        raise HTTPException(status_code=404, detail="Dataset CSV directory not found.")
    cats = sorted([
        d.name for d in dataset_csv_root.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])
    return {"categories": cats, "total": len(cats)}


@app.get("/api/v1/dataset/signs")
def list_dataset_signs(category: str = Query(..., description="Category name, e.g. Greetings, Verbs")):
    """List all sign words in a given dataset category."""
    cat_dir = dataset_csv_root / category
    if not cat_dir.exists() or not cat_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Category '{category}' not found.")
    signs = sorted([
        d.name for d in cat_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])
    return {"category": category, "signs": signs, "total": len(signs)}


@app.get("/api/v1/dataset/takes")
def list_dataset_takes(
    category: str = Query(..., description="Category name"),
    sign: str = Query(..., description="Sign name"),
):
    """List all CSV take recordings for a given category and sign."""
    sign_dir = dataset_csv_root / category / sign
    if not sign_dir.exists() or not sign_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Sign '{sign}' in '{category}' not found.")
    csvs = sorted(sign_dir.glob("*.csv"))
    takes = [c.stem for c in csvs]
    return {"category": category, "sign": sign, "takes": takes, "total": len(takes)}


@app.get("/api/v1/dataset/landmarks")
def get_dataset_landmarks(
    category: str = Query(..., description="Category name"),
    sign: str = Query(..., description="Sign name"),
    take: str = Query(..., description="Take filename or stem"),
):
    """Parse and return 33 standard MediaPipe Pose landmark coordinates per frame for 2D animation."""
    csv_file = dataset_csv_root / category / sign / (take if take.endswith(".csv") else f"{take}.csv")
    if not csv_file.exists():
        raise HTTPException(status_code=404, detail=f"Take CSV '{csv_file.name}' not found.")

    try:
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)

        frames = []
        for row in rows:
            if len(row) < 33:
                continue
            frame_pts = [ast.literal_eval(c) for c in row[:33]]
            frames.append(frame_pts)

        return {
            "category": category,
            "sign": sign,
            "take": csv_file.stem,
            "fps": 10.0,
            "frame_count": len(frames),
            "frames": frames,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error parsing landmark CSV: {str(e)}")


class KinematicsRequest(BaseModel):
    category: Optional[str] = None
    sign: Optional[str] = None
    take: Optional[str] = None
    frames: Optional[List[List[Any]]] = None
    fps: Optional[float] = 10.0


@app.post("/api/v1/dataset/kinematics")
def compute_dataset_kinematics(req: KinematicsRequest):
    """
    Computes time-series landmark coordinate trajectories and velocities
    for Head (Cyan), Body (Green), and Hands (Pink) for any take or animation sequence.
    """
    frames = req.frames
    if not frames:
        if not (req.category and req.sign and req.take):
            raise HTTPException(status_code=400, detail="Either 'frames' or ('category', 'sign', 'take') must be provided.")
        csv_file = dataset_csv_root / req.category / req.sign / (req.take if req.take.endswith(".csv") else f"{req.take}.csv")
        if not csv_file.exists():
            raise HTTPException(status_code=404, detail=f"Take CSV '{csv_file.name}' not found.")
        try:
            with open(csv_file, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                rows = list(reader)
            frames = []
            for r in rows:
                if len(r) >= 33:
                    frames.append([ast.literal_eval(c) for c in r[:33]])
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading CSV: {str(e)}")

    if not frames:
        return {"frame_count": 0, "timestamps": [], "head": {}, "body": {}, "hands": {}, "metrics": {}}

    fps = req.fps or 10.0
    dt = 1.0 / fps
    T = len(frames)
    timestamps = [round(i * dt, 4) for i in range(T)]

    # Extract 3D coordinates per landmark across time
    # Index mapping:
    # 0: nose, 11: left_shoulder, 12: right_shoulder, 15: left_wrist, 16: right_wrist
    def get_pt(f_idx, lm_idx):
        pt = frames[f_idx][lm_idx]
        return [round(float(pt[0]), 4), round(float(pt[1]), 4), round(float(pt[2]), 4)]

    def get_trajectory(lm_idx):
        return [get_pt(f, lm_idx) for f in range(T)]

    def get_velocity(lm_idx):
        traj = get_trajectory(lm_idx)
        vels = [0.0] * T
        for i in range(1, T):
            dx = traj[i][0] - traj[i-1][0]
            dy = traj[i][1] - traj[i-1][1]
            dz = traj[i][2] - traj[i-1][2]
            dist = (dx*dx + dy*dy + dz*dz) ** 0.5
            vels[i] = round(dist / dt, 4)
        if T > 1:
            vels[0] = vels[1]
        return vels

    nose_traj = get_trajectory(0)
    l_sh_traj = get_trajectory(11)
    r_sh_traj = get_trajectory(12)
    l_wr_traj = get_trajectory(15)
    r_wr_traj = get_trajectory(16)

    nose_vel = get_velocity(0)
    r_sh_vel = get_velocity(12)
    l_wr_vel = get_velocity(15)
    r_wr_vel = get_velocity(16)

    peak_hand_v = max(max(l_wr_vel), max(r_wr_vel)) if T > 0 else 0.0

    return {
        "frame_count": T,
        "fps": fps,
        "duration_sec": round(timestamps[-1] if timestamps else 0.0, 3),
        "timestamps": timestamps,
        "head": {
            "color": "#00E5FF",
            "nose": nose_traj,
            "velocity": nose_vel,
        },
        "body": {
            "color": "#00E676",
            "left_shoulder": l_sh_traj,
            "right_shoulder": r_sh_traj,
            "velocity": r_sh_vel,
        },
        "hands": {
            "color": "#FF3366",
            "left_wrist": l_wr_traj,
            "right_wrist": r_wr_traj,
            "left_wrist_velocity": l_wr_vel,
            "right_wrist_velocity": r_wr_vel,
        },
        "metrics": {
            "peak_hand_velocity": peak_hand_v,
            "mean_hand_velocity": round(sum(r_wr_vel) / max(len(r_wr_vel), 1), 4),
        }
    }


class Translate2DRequest(BaseModel):
    text: str = Field(..., description="English or Sinhala sentence")
    language: Optional[str] = Field(default="auto", description="Source language ('auto', 'en', 'si')")
    engine: Optional[str] = Field(default="neural", description="Engine: 'neural', 'gemini', 'rules', or 'hybrid'")


@app.post("/api/v1/translate-and-resolve-2d")
def translate_and_resolve_2d(req: Translate2DRequest):
    """
    Translates an English or Sinhala sentence into an ordered sign sequence
    and resolves each gloss to its 2D MediaPipe Pose landmark frames from the dataset.
    Supports continuous neural latent grounding, LLM translation, or rule-based baselines.
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    if not os.getenv("GEMINI_API_KEY"):
        for env_path in [root_dir / ".env", Path(__file__).resolve().parent.parent / ".env"]:
            if env_path.exists():
                load_dotenv(dotenv_path=env_path)

    engine_choice = (req.engine or "neural").lower()
    method_used = "neural_dense_grounding_r3072"

    if engine_choice == "neural":
        try:
            result = neural_grounder.ground(req.text, language=req.language or "auto")
            method_used = "neural_dense_grounding_r3072"
        except Exception as e:
            result = translator.translate(req.text, language=req.language or "auto")
            method_used = f"rules_fallback ({str(e)})"
    elif engine_choice == "gemini":
        if gemini_translator.is_available():
            try:
                result = gemini_translator.translate(req.text, language=req.language or "auto")
                method_used = "gemini_constrained_llm"
            except Exception as e:
                result = translator.translate(req.text, language=req.language or "auto")
                method_used = f"rules_fallback ({str(e)})"
        else:
            result = translator.translate(req.text, language=req.language or "auto")
            method_used = "rules_fallback_no_gemini_key"
    elif engine_choice == "rules":
        result = translator.translate(req.text, language=req.language or "auto")
        method_used = "hierarchical_bilingual_rules"
    else:  # hybrid
        try:
            result = neural_grounder.ground(req.text, language=req.language or "auto")
            method_used = "hybrid_neural_dense_r3072"
        except Exception:
            result = translator.translate(req.text, language=req.language or "auto")
            method_used = "hybrid_rules_fallback"

    anim_lib = get_animation_library()

    sequence_items = []
    for tok in result.tokens:
        clean_gloss = tok.gloss.strip().upper().replace("-", "_")
        manifest_entry = anim_lib._manifest.get(clean_gloss)

        status = "no_sign"
        frames = []
        take = None
        category = None
        sign_label = None

        if manifest_entry:
            category = manifest_entry["category"]
            sign_label = manifest_entry["label_in_dataset"]
            take = manifest_entry["take"]
            csv_path = dataset_csv_root / category / sign_label / f"{take}.csv"

            if csv_path.exists():
                try:
                    with open(csv_path, "r", encoding="utf-8") as f:
                        reader = csv.reader(f)
                        for row in reader:
                            if len(row) >= 33:
                                frames.append([ast.literal_eval(c) for c in row[:33]])
                    status = "available" if frames else "empty"
                except Exception:
                    status = "error"

        orig_w = getattr(tok, "original_word", getattr(tok, "original", ""))
        conf = getattr(tok, "similarity", getattr(tok, "confidence", 1.0))
        is_fs = getattr(tok, "is_fingerspelled", False) or tok.gloss.startswith("#")

        # Fallback to Parametric Dactylology (Fingerspelling) if no dataset take exists
        if status != "available" and (is_fs or tok.match_type == "fingerspelled" or tok.gloss.startswith("#")):
            try:
                fs_frames = fs_generator.generate_token_frames(tok.gloss, orig_w)
                if fs_frames:
                    frames = fs_frames
                    status = "available"
                    category = "Dactylology"
                    sign_label = f"Fingerspell_{tok.gloss.replace('#', '')}"
                    take = "Parametric_Keyframe_Spline"
            except Exception:
                pass

        raw_alts = getattr(tok, "alternatives", [])
        alts = [a.model_dump() if hasattr(a, "model_dump") else a for a in raw_alts]

        sequence_items.append({
            "gloss": tok.gloss,
            "original_word": orig_w,
            "match_type": tok.match_type,
            "confidence": round(float(conf), 4),
            "is_fingerspelled": getattr(tok, "is_fingerspelled", False),
            "alternatives": alts,
            "note": tok.note,
            "category": category,
            "sign_label": sign_label,
            "take": take,
            "status": status,
            "fps": 10.0,
            "frame_count": len(frames),
            "frames": frames,
        })

    return {
        "original_text": result.original_text,
        "detected_language": result.detected_language,
        "glosses": result.glosses,
        "reordered": result.reordered,
        "method": method_used,
        "mean_confidence": round(getattr(result, "mean_confidence", 1.0), 4),
        "sequence": sequence_items,
    }


class NeuralGroundRequest(BaseModel):
    text: str = Field(..., description="English or Sinhala sentence")
    language: Optional[str] = Field(default="auto", description="Source language ('auto', 'en', 'si')")


@app.post("/api/v1/neural-ground")
def api_neural_ground(req: NeuralGroundRequest):
    """
    Direct endpoint for Neural Cross-Lingual Semantic Grounding in continuous R^3072 latent space.
    Returns grounded SLSL sign glosses with cosine similarities, semantic neighborhoods, and alternatives.
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    return neural_grounder.ground(req.text, language=req.language or "auto")


@app.get("/api/v1/evaluation/benchmark")
def get_benchmark_ablation(live_llm: bool = Query(default=False, description="Whether to invoke live LLM calls")):
    """
    Returns the Academic Benchmark comparative ablation study across all 4 architectures
    with LaTeX table output ready for thesis and research publication.
    """
    from .text_to_gloss.benchmark_evaluator import AcademicBenchmarkEvaluator
    evaluator = AcademicBenchmarkEvaluator()
    study = evaluator.run_full_ablation_study(live_llm=live_llm)
    return study


@app.post("/api/v1/voice/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    language: Optional[str] = Form("auto")
):
    """
    Transcribes spoken audio into Sinhala or English text using Gemini 2.5 Flash.
    Acts as a server-side fallback for browsers without native Web Speech API support.
    """
    if not file:
        raise HTTPException(status_code=400, detail="Audio file is required.")

    audio_bytes = await file.read()
    if not audio_bytes or len(audio_bytes) < 10:
        raise HTTPException(status_code=400, detail="Audio file is empty or corrupted.")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="Gemini API key is not configured. Please set GEMINI_API_KEY."
        )

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)

        mime_type = file.content_type or "audio/webm"
        if ";" in mime_type:
            mime_type = mime_type.split(";")[0].strip()

        lang_instruction = ""
        if language == "si":
            lang_instruction = "The audio is spoken in Sinhala. Transcribe it into Sinhala script (සිංහල)."
        elif language == "en":
            lang_instruction = "The audio is spoken in English. Transcribe it into English text."
        else:
            lang_instruction = "Detect whether the audio is spoken in Sinhala or English and transcribe verbatim into that language."

        prompt = (
            f"{lang_instruction}\n"
            "CRITICAL: Output ONLY the exact transcribed words with no commentary, introductory text, markdown code blocks, or quotation marks."
        )

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
                prompt
            ],
            config=types.GenerateContentConfig(
                temperature=0.1,
            )
        )

        transcript = (response.text or "").strip()
        transcript = transcript.strip('`"\'').strip()

        detected_lang = language if language in ("si", "en") else (
            "si" if any("\u0D80" <= c <= "\u0DFF" for c in transcript) else "en"
        )

        return {
            "transcript": transcript,
            "detected_language": detected_lang,
            "audio_size_bytes": len(audio_bytes),
            "mime_type": mime_type
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Voice transcription failed: {str(e)}")




