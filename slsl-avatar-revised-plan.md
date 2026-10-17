# SLSL Text-to-Sign Avatar — Project Plan

**Project**: Final-year capstone — text-to-sign-language avatar system for Sri Lankan Sign Language (SLSL), built on the SSL400 dataset.

**Architecture (target end state)**:
```
Text input → Text-to-Gloss (rule-based, FastAPI) → gloss lookup → sign animation data (JSON) → Unity WebGL avatar (Mixamo Y Bot)
```
Signs are data files of per-frame bone rotations, generated from SSL400 videos with MediaPipe and retargeted onto the avatar. Adding a sign needs no Unity rebuild.

**Scope decisions**
- Text to sign only. Sign-to-text is out of scope.
- SLSL directly, using SSL400. The ASL prototype is dropped.
- Vocabulary is limited to English words that exist in SSL400 (383 signs). Anything else is shown as "no sign available".
- Text-to-gloss stays rule-based. Machine learning is an optional stretch (Part 3), not part of the core.
- The SSL400 license is not yet confirmed; record it in the README before anything is shown or shared.

---

## PART 1 — Quick Start (3-Day Skeleton) — DONE

The skeleton is built: a Python dictionary-based gloss translator, a Mixamo Y Bot in Unity playing the Salute and Shooting Arrow clips, a hardcoded gloss sequence played one animation after another, and a working Unity WebGL build. Keep it as the fallback "clip mode" while the real signs are added.

---

## PART 2 — Full Project Plan

### SSL400 facts (verified)
- 383 sign classes in 16 categories, 4,236 videos (about 11 takes per sign).
- Three parallel trees with identical layout `Dataset/<Variant>/<Category>/<Sign_Word>/<Sign_Word>_<NNN>.<ext>`:
  - `Dataset - Original/`: raw `.mp4` / `.MOV`
  - `Dataset - MP - CSV/`: MediaPipe landmarks
  - `Dataset - MP - VID/`: overlay videos
- Labels are English words (plus a few such as `Ayubowan`). Some names need cleaning (`Cant`, `Dont`, `1. one`, `Good morning`).
- **The provided CSVs (checked on `Blue_001.csv`)**: no header row; one row per frame (37 frames); 132 columns = 33 pose landmarks x 4 values, each cell a string like `"[x, y, z, visibility]"` that needs `ast.literal_eval`. They are **pose-only, with no hand or finger landmarks**, and x/y are normalized image coordinates with noisy relative z. Use them only for quick checks and for choosing good takes; **re-extract from the `Original` videos** (with hands and world landmarks) for the final animations.

```python
import csv, ast, numpy as np
def load_pose_csv(path):
    rows = list(csv.reader(open(path)))
    return np.array([[ast.literal_eval(c) for c in r] for r in rows])  # (frames, 33, 4)
```

### Data flow

At request time:
```
user types text
   |  (finished word: space / Enter / punctuation)
   v
web app / Unity UI --> POST /translate --> text_to_gloss (rules) --> gloss tokens
   |                      animation/library.py: gloss in manifest and approved?
   |                              yes -> /animations/slsl/<GLOSS>.json
   |                              no  -> flagged "no sign"
   v
Unity: queue of glosses --> PoseJsonLoader (fetch JSON) --> SignPlayer (apply bone rotations, blend between signs)
```

Offline, once per sign:
```
SSL400 Original video --> extract.py (pose + hands, world landmarks) --> clean.py --> retarget.py --> export_unity.py --> data/animations/slsl/<GLOSS>.json
```

### Folder changes (added to the existing Ameesha workspace; see `project-structure.md`)
- `pose-pipeline/src/`: add `audit_ssl400.py`, `build_manifest.py`, `batch.py` (alongside `extract.py`, `clean.py`, `retarget.py`, `export_unity.py`).
- Keep the SSL400 dataset **outside git** (for example `data/ssl400/`, gitignored) and pass its path to scripts as an argument. Never commit the videos.
- `data/animations/` uses only `slsl/`.
- `data/manifest.csv` header:
  ```
  gloss,label_in_dataset,category,language,source_video,take,signer_id,mirrored,license,status,notes
  ```
  `status` values: `todo`, `extracted`, `cleaned`, `retargeted`, `approved`. Only `approved` rows are served to the avatar.

### Two parallel tracks
- **Track A**: typed word to queued sign, first with the two test clips.
- **Track B**: real signs from SSL400.

They only meet at Phase 5, so neither blocks the other.

---

### Phase A: Typed-word playback (Track A, 1-2 days)
Goal: each finished word typed plays its sign in order, even while earlier signs are still playing.
- Detect finished words in `SignAvatarUI.cs` (`onValueChanged`: last character is a space or punctuation, or Enter).
- Send the full text so far to `/translate` (reordering rules need the whole sentence). Track how many glosses were already queued and enqueue only the new ones. Define what happens when the user edits earlier words (for example: clear the queue and restart).
- In `SignAvatarController.cs`: a `Queue<string>` and one coroutine that plays each gloss, waits for its clip length, then plays the next; new words can arrive during playback.
- Unknown gloss: skip it and show it greyed out in the UI. Do not throw.
- For testing, map two or three test words to the Salute and Shooting Arrow clips. Keep this clip mode as a fallback after pose mode is added.
- **Checkpoint**: typing `hello go blue` plays three signs in order; typing a fourth word while the third is playing queues it.

### Phase 0: Setup and Data Audit (Track B, 1-2 days)
- Confirm toolchain: Unity LTS, Python 3.10+, MediaPipe, FastAPI.
- Run `audit_ssl400.py <dataset_root>` and save the report to `docs/ssl400-audit.md`:
  - counts per category and per sign, takes per sign, mismatches between the three trees;
  - per video: fps, resolution, duration, frame count (and whether it matches the CSV row count);
  - CSV sanity: all 33 landmarks present, share of frames with low visibility on the wrists;
  - names needing cleanup.
- Manual checks, written into the README: the **license text**, number of signers, whether both hands are fully visible, whether some signs are left-handed.
- Start outreach now to a Deaf school or association (or another fluent SLSL signer) for sign review. This has the longest lead time of anything in the project.

### Phase 1: Vocabulary and Manifest (1 day)
- `build_manifest.py` walks the `Original` tree and writes `data/manifest.csv` with a cleaned gloss per sign (uppercase, spaces to underscores, `1. one` to `ONE`, `Cant` to `CANT`; keep `label_in_dataset` untouched).
- Choose a **starter set of 10 words** (suggested: HELLO, THANK_YOU, YES, NO, GO, COME, EAT, MOTHER, TODAY, BLUE; replace any that look poor in the audit).
- For each starter word pick **one best take**: use the overlay video in `MP - VID` to find a take where both hands are visible throughout, and where the signer and dominant hand are consistent with the other starter signs. Record the take and the reason in `notes`.

### Phase 2: Extraction with Hands (2 days)
`extract.py --video <path> --out <path>`:
- Run MediaPipe Holistic (or Pose plus Hand Landmarkers) on the Original video.
- Save pose **world** landmarks (33 x xyz + visibility), left hand and right hand (21 x xyz each), image-space copies, and a per-frame `hand_detected` mask. Missing hands are NaN, not zero.
- Write a debug overlay video.
- Compare against the provided CSV for the same take as a sanity check (wrist movement should agree in shape).
- **Checkpoint**: for the 10 starter takes, both hands are tracked in most frames; report the share of frames with a lost hand.

### Phase 3: Cleaning (1-2 days)
`clean.py`: trim idle frames before and after the sign; interpolate gaps up to about 5 frames (flag longer ones); smooth (Savitzky-Golay or One Euro filter); normalize to a mid-hip root and shoulder-width scale; resample to 30 fps; mirror if the take is flagged `mirrored`. Plot before and after for each starter sign in `notebooks/inspect_landmarks.ipynb`.

### Phase 4: Retargeting to Y Bot (4-6 days; the hardest phase)
Direction-based retargeting, built in this order with a visual check after each step:
1. Arms only (shoulder, elbow, wrist) for one sign, played in the Unity Editor.
2. Wrist and palm orientation from wrist, index MCP and pinky MCP.
3. Finger curl for the four fingers, clamped to valid ranges.
4. Thumb (last and hardest).

Method: take Y Bot's T-pose bone directions from the Humanoid avatar, compute the rotation from each rest direction to the target direction per frame, apply down the hierarchy as local rotations, and export quaternions per Humanoid bone.

Output format (`data/animations/slsl/<GLOSS>.json`):
```json
{ "gloss": "BLUE", "language": "slsl", "fps": 30,
  "frames": [ { "bones": { "RightUpperArm": [x,y,z,w], "RightHand": [x,y,z,w] } } ] }
```
- **Checkpoint**: one starter sign (for example BLUE) is recognizable against its reference video, judged by someone who reads SLSL. "Recognizable" is the bar, not perfect.

### Phase 5: Unity Pose Playback and Integration (2-3 days)
- `PoseJsonLoader.cs`: fetch the JSON by gloss from the backend (`UnityWebRequest`) and cache it.
- `SignPlayer.cs`: apply bone rotations in `LateUpdate()` with `Animator.SetBoneLocalRotation`; slerp from the last frame of one sign to the first of the next over about 0.2 s; return to an idle pose when the queue is empty.
- Backend: `animation/library.py` resolves a gloss to its JSON through the manifest (approved rows only); mount `data/animations/` at `/animations`; extend `/translate` additively (existing `gloss` and `animations` fields stay as they are) and update `docs/api-contract.md`.
- Test in the **WebGL build**, not only in the Editor. Load JSON at runtime; do not bake it into the build.
- **Checkpoint**: typing a sentence made only of starter words plays real SLSL signs in the browser build, with smooth transitions.

### Phase 6: Scale Up (ongoing, in batches)
- `batch.py` runs Phases 2-4 over every manifest row marked `todo` and logs failures (lost hands, odd durations, retarget errors).
- Grow in batches of 20-30 signs, prioritizing words useful for the demo scenarios. Do not aim for all 383.
- Only `approved` signs are served. Approval means a person reviewed the avatar against the reference.

### Phase 7: Validation (spread through the project, formal at the end)
- A fluent SLSL signer reviews sign quality and flags wrong signs. Raise **ethics approval** with your supervisor now if you will test with Deaf participants (it has its own lead time).
- Report simple, honest numbers: share of vocabulary approved, share of typed test sentences fully covered by the vocabulary, reviewer ratings per sign.

### Phase 8: Writing
Start well before the last two weeks. State limitations plainly (see below).

---

## Timeline (about 3 weeks to a convincing demo; adjust to your deadline)

| Days | Track A | Track B |
|------|---------|---------|
| 1-2 | Phase A: typed-word queue | Phase 0 audit, Phase 1 manifest |
| 3-4 | Polish; UI states for unknown words | Phase 2 extraction, Phase 3 cleaning |
| 5-10 | | Phase 4 retargeting (arms, wrist, fingers, thumb) |
| 11-13 | Phase 5 integration, WebGL test | Phase 5 integration |
| 14+ | | Phase 6 scale-up in batches; Phase 7 review |

---

## PART 3 — Optional Stretch (only after the core works)
- Fingerspelling fallback, if SSL400 or another source has alphabet signs.
- Sinhala input: a Sinhala-word-to-gloss dictionary.
- A learned gloss-to-pose model trained on the retargeted data (about 11 takes x 383 signs is a start, but expect weak generalization to unseen words). Ask your supervisor whether the project needs a machine-learning contribution; if it does, decide early because it changes the work.

---

## Critical Path Risks (flag to supervisor early)
1. **Retargeting quality** is the biggest unknown. Timebox each sub-step; if fingers stay poor, ship accurate arms and wrists with simple handshapes and say so.
2. **SSL400 license** unconfirmed: resolve before demos, publishing, or sharing derived files.
3. **Sign review access**: no fluent SLSL reviewer means no way to confirm signs are correct. Start outreach now.
4. **Ethics approval** for any human evaluation involving Deaf participants: check the university process now.
5. **Signer or handedness inconsistency across takes** makes signs look inconsistent; check in Phase 0.
6. **Some dataset entries may be phrases or compound signs** (Day, Month, Addition); do not assume every folder is one simple word.
7. **Typed words outside the vocabulary** will fail often; the UI must make this clear, never silent.
8. **WebGL surprises**: test early and often.

---

## Known Limitations to State Honestly (not hide) in the Thesis
- Vocabulary is limited to SSL400's 383 signs; typed words outside it have no sign.
- Words are signed one by one from a rule-based gloss; this is closer to keyword signing than fully linguistically valid SLSL grammar, and facial grammar is not modeled.
- Single-camera depth makes handshapes approximate; signs are judged "recognizable", not exact.
- SSL400 contains isolated signs, not sentences, so there is no SLSL sentence-to-gloss training data.

---

## Tasks for Claude Code (in order)
1. Apply the folder changes above to the existing Ameesha workspace; delete nothing existing.
2. Phase A: implement the typed-word queue in `SignAvatarUI.cs` and `SignAvatarController.cs`, keeping the current clip mode working.
3. Write `audit_ssl400.py` and run it when given the dataset path; save the report to `docs/ssl400-audit.md`.
4. Write `build_manifest.py` and generate `data/manifest.csv`.
5. Write `extract.py` (pose world landmarks plus hands, NaN for missing, overlay video) and run it on the 10 starter takes.
6. Write `clean.py` and the inspection notebook.
7. Write `retarget.py` and `export_unity.py`, arms first; produce one JSON and play it in the Editor.
8. Write `PoseJsonLoader.cs` and `SignPlayer.cs`; test in the Editor, then in the WebGL build.
9. Add backend `animation/library.py`, the `/animations` static mount, additive `/translate` fields and tests, and update `docs/api-contract.md`.
10. Extend retargeting to wrist, fingers, thumb; then `batch.py` and scale-up.

**Definition of done for the core project**: a user types an English sentence in the web app; each finished word whose sign is approved is performed by the avatar in order, with smooth transitions; unknown words are shown clearly as having no sign.
