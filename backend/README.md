# SLSL Avatar Engine — Backend Service

Python backend service for the Text-to-Sign Avatar system.

## Features
- **Day 1 Quick Start Engine**: Rule-based & dictionary text-to-gloss translation.
- **Linguistic Rules**:
  - Drops articles (`a`, `an`, `the`) and copulas (`am`, `is`, `are`, `was`, `were`).
  - Inflection normalization (`going`, `went` → `GO`).
  - Common multi-word phrase mappings (`thank you` → `THANK-YOU`).
  - Grammatical reordering (Time-first, Wh-question end).
  - Fingerspelling fallback for out-of-vocabulary words.
- **FastAPI Endpoints**:
  - `POST /api/v1/translate`: Translate English sentences into ordered gloss sequences and animation cues.
  - `GET /api/v1/vocabulary`: View supported gloss words.
  - `GET /api/v1/sample-sentences`: Test sample sentences.
  - `GET /health`: Health status.
- **Interactive CLI**: Fast terminal testing.

---

## Quick Start Instructions

### 1. Activate Environment
```bash
cd backend
source .venv/bin/activate
```

### 2. Run Interactive CLI
```bash
python -m backend.src.cli
```
Or pass a sentence directly:
```bash
python -m backend.src.cli "I am going to the store"
```
Output:
```
[Original]: I am going to the store
[Glosses] : ['I', 'GO', 'STORE']
```

### 3. Run FastAPI Backend Server
```bash
python -m uvicorn backend.src.app:app --reload --port 8000
```
- Open Swagger API Documentation in browser: **http://127.0.0.1:8000/docs**
- Test health check: **http://127.0.0.1:8000/health**

### 4. Run Unit Tests
```bash
PYTHONPATH=. pytest backend/tests/ -v
```
