# SLSL Text-to-Sign Avatar — Project Plan

**Project**: Final-year capstone — text-to-sign-language avatar system, prototyped on ASL, adapted to Sri Lankan Sign Language (SLSL).

**Architecture (target end state)**:
```
Text input → Text-to-Gloss model (NLP/ML) → Gloss-to-Pose model (ML) → Unity avatar (Blender-rigged humanoid)
```
Hybrid approach: keyframed/dictionary animations for core vocabulary, ML-predicted joint coordinates for generalization to related/unseen words, fingerspelling fallback for anything else.

---

## PART 1 — Quick Start (3-Day Skeleton)

**Goal**: prove the pipeline connects end-to-end with the simplest possible version of each stage. No ML yet. No real rigging yet. Just moving parts wired together.

Mental model — three boxes:
```
[Type a sentence] → [Turn into sign-gloss words] → [Avatar plays matching animations]
```

### Day 1 — Dumb "brain" (dictionary-based gloss translation)
- Write a Python script with a hardcoded `word_to_gloss` dictionary (drop articles/copulas like "am", "the", "to"; keep content words).
- Function `translate(sentence) -> list[str]` returns a gloss sequence.
- No ML, no corpus yet — this is a legitimate baseline, not a placeholder to be ashamed of.
- **Deliverable**: running `translate("I am going to the store")` prints `['I', 'GO', 'STORE']`.

### Day 2 — Dumb avatar (pre-made animations, manual trigger)
- Get a free pre-rigged humanoid from Mixamo (don't rig your own yet).
- Download 2-3 pre-made animation clips from Mixamo.
- Import into Unity, write a minimal C# script that plays one animation clip on a button press via `animator.Play("ClipName")`.
- **Deliverable**: button press → avatar plays an animation.

### Day 3 — Connect the two
- Hardcode a gloss sequence (e.g. `["I", "GO", "STORE"]`) into a Unity C# script (don't build the API yet — that's later).
- Loop through the sequence, playing one animation clip per gloss word with a short delay between each (`animator.Play(word)` + `WaitForSeconds`).
- **Deliverable**: a hardcoded "sentence" plays as a sequence of avatar animations, one after another.

**By end of Day 3**: working skeleton — sentence concept → gloss sequence → sequential avatar animation. Everything below (real ML models, real corpora, real rigging) slots into this skeleton later. This is a legitimate, presentable milestone for 3 days of work.

---

## PART 2 — Full Project Plan

### Phase 0: Setup & Foundations (Weeks 1–2)
- Install/confirm toolchain: Unity (LTS), Blender, Python 3.10+, PyTorch, HuggingFace `transformers`, MediaPipe, FastAPI.
- Get supervisor sign-off on scope: ASL prototype → SLSL adaptation, in writing in the proposal.
- Download and inspect ASLG-PC12 corpus (English↔ASL-gloss parallel pairs) firsthand — check actual formatting, not just docs.
- Download and inspect SSL400 (Sri Lankan Sign Language video dataset) — confirm it's isolated-sign-recognition data (video → label), not a translation corpus.
- Start outreach to a Deaf school/association for SLSL data/validation access **now** — longest lead time in the whole project.

### Phase 1: Text → Gloss Model (Weeks 3–6)
- Build a rule-based baseline: spaCy POS tagging + reordering rules (drop articles/copulas, reorder toward topic-comment structure). This is your fallback and comparison point.
- Fine-tune a small seq2seq transformer (T5-small) on ASLG-PC12 for text→gloss.
- Evaluate both with BLEU score on a held-out test set.
- **Checkpoint**: typed English sentence → gloss sequence, from both methods, with a documented comparison.

### Phase 2: Avatar & Retargeting Pipeline (Weeks 5–9, overlaps Phase 1)
- Rig a humanoid in Blender, export to Unity as Humanoid rig type.
- Hand-animate ~10-15 test signs to validate the Blender→Unity pipeline end-to-end.
- Build a JSON gloss-to-animation dictionary lookup in Unity (C# controller).
- Implement crossfade blending between signs (`Animator.CrossFade`) so transitions aren't robotic.
- **Checkpoint**: a hardcoded gloss sequence plays as a smooth signed animation in Unity — no ML involved yet.

### Phase 3: Vocabulary Build-Out (Weeks 8–14, overlaps Phase 2)
- Scale from the 10-15 test signs to a real target vocabulary (~150-300 signs).
- Use SSL400 videos (for SLSL-relevant signs) or ASL reference videos (prototype phase) as keyframing reference.
- Track progress in a simple spreadsheet: word, gloss, animated (Y/N), source reference.
- This is pure production time — budget generously, it's the most underestimated phase.

### Phase 4: Gloss → Pose ML Model (Weeks 12–18)
- Extract pose coordinates from vocabulary reference videos using MediaPipe Holistic.
- Build/train a gloss→pose sequence generation model (Progressive Transformer-style: transformer encoder over gloss tokens, autoregressive decoder outputting continuous joint-coordinate frames, regression loss (MSE/L1), counter-value trick for sequence termination).
- Start with a small subset of vocabulary as proof of concept before scaling.
- Evaluate with MPJPE (Mean Per Joint Position Error) and DTW-based sequence similarity.
- **Checkpoint**: model produces plausible coordinate sequences for at least some signs, evaluated quantitatively.

### Phase 5: Full Integration (Weeks 17–20)
- Build FastAPI backend serving both models (text→gloss, gloss→pose), loaded once at startup, exposing a `/translate` endpoint.
- Build Unity UI (ideally with UI Toolkit): text input → gloss display → avatar animation, calling the backend via `UnityWebRequest`.
- Package the Python backend with PyInstaller; launch it as a background subprocess from Unity on app startup, kill on close.
- Wire hybrid logic: known vocabulary → dictionary animation; unknown/similar words → ML-predicted coordinates; fully unknown → fingerspelling fallback.
- Build final retargeting layer: map predicted MediaPipe-style joint coordinates onto the Unity rig's bone hierarchy.

### Phase 6: SLSL Adaptation (Weeks 19–24, if timeline allows)
- Swap vocabulary/dictionary to SLSL signs using data/contacts from Deaf-community outreach.
- Attempt small-scale fine-tuning or transfer for SLSL gloss ordering; if a full corpus isn't feasible, fall back to SLSL grammar rules for the rule-based path.
- Even a small, honestly-scoped SLSL demo (20-30 words) is a legitimate result — do not feel pressure to match the ASL prototype's scale.

### Phase 7: Evaluation & Writing (Weeks 22–28)
- User testing if possible, ideally with Deaf/signing participants — flag ethics approval needs to supervisor early (can have its own lead time).
- Write up quantitative results (BLEU, MPJPE, DTW) and qualitative limitations honestly.
- Final report/thesis writing — start well before the last 2 weeks.

---

## Critical Path Risks (flag to supervisor early)
1. **SLSL data/contact access** — longest lead time in the project, start immediately.
2. **Ethics approval** for any human evaluation involving Deaf participants — check university process now.
3. **Vocabulary animation is a time sink** — manual production work disguised as a "small" task; protect time for it explicitly in scheduling.

---

## Known Limitations to State Honestly (not hide) in the Thesis
- Dictionary/gloss-swap translation is closer to keyword/Signed Exact English style signing than fully linguistically valid sign language grammar — name this limitation explicitly.
- ML-based generalization to unseen vocabulary is soft interpolation between phonologically/semantically similar trained signs, not true novel-sign invention — this is a known limitation even in well-resourced sign language research (e.g. ASL, DGS), and will be weaker still given SLSL's limited training data.
- SLSL currently has no large public annotated corpus; Phase 6's scope should be set realistically based on whatever data volume is actually obtained.
