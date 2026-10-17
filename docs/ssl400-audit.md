# SSL400 Dataset Audit Report

**Audit Date**: 2026-10-17  
**Dataset Location**: `/Users/nipun/Desktop/Projects/Ameesha/Dataset`  
**Scope**: Complete structural, video, and MediaPipe CSV audit for Sri Lankan Sign Language (SLSL) Text-to-Sign Avatar Project.

---

## 1. Executive Summary

- **Total Sign Classes (Vocabulary)**: **383** across **16** categories.
- **Total Videos**: **4,236** recordings (averaging **11.1** takes per sign).
- **Tree Alignment**: **100% synchronized** — all 4,236 takes in `Dataset - Original` have exact 1:1 counterparts in `Dataset - MP - CSV` and `Dataset - MP - VID`.
- **Video Standardization**: **100% uniform** across all 4,236 videos: **512x512 resolution** at **10.0 FPS** (mean duration: **3.19s**).
- **CSV Landmarks Status**: Each row has **exactly 33 landmarks** (standard MediaPipe Pose). **NO hand or finger landmarks exist in the CSVs**.
- **Key Takeaway for Pipeline**: Provided CSVs are pose-only and downsampled/smoothed; **full re-extraction with MediaPipe Holistic (Pose + Hands) from `Dataset - Original` is mandatory for hand shape retargeting**.

---

## 2. Dataset Structure & Tree Verification

| Subdirectory | Format | File Count | Verified | Purpose |
| :--- | :--- | :---: | :---: | :--- |
| `Dataset - Original` | `.mp4` (2,623), `.MOV` (1,613) | 4,236 | Yes | Source video recordings of signs |
| `Dataset - MP - CSV` | `.csv` | 4,236 | Yes | Pre-extracted 33 pose landmark coordinates |
| `Dataset - MP - VID` | `.mp4` (2,623), `.MOV` (1,613) | 4,236 | Yes | Overlay videos with MediaPipe pose skeleton |

### Tree Discrepancies

> **Zero discrepancies detected**. All 383 sign classes and 4,236 takes match identically across all three parallel folder structures.

## 3. Categories & Class Distribution

| Category | Signs Count | Total Takes | Avg Takes / Sign | Min Takes | Max Takes |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Adjectives** | 41 | 443 | 10.8 | 1 | 67 |
| **Adverb** | 14 | 85 | 6.1 | 1 | 27 |
| **Colors** | 14 | 242 | 17.3 | 1 | 48 |
| **Conjunctions** | 1 | 4 | 4.0 | 4 | 4 |
| **Days** | 22 | 247 | 11.2 | 1 | 36 |
| **Determiner** | 1 | 3 | 3.0 | 3 | 3 |
| **Greetings** | 5 | 245 | 49.0 | 2 | 116 |
| **Interjection** | 1 | 6 | 6.0 | 6 | 6 |
| **Months** | 14 | 115 | 8.2 | 1 | 28 |
| **Nouns** | 87 | 978 | 11.2 | 1 | 62 |
| **Numbers** | 26 | 179 | 6.9 | 2 | 23 |
| **People** | 32 | 369 | 11.5 | 1 | 53 |
| **Places** | 13 | 144 | 11.1 | 1 | 61 |
| **Preposition** | 12 | 81 | 6.8 | 4 | 9 |
| **Vehicles** | 10 | 83 | 8.3 | 1 | 30 |
| **Verbs** | 90 | 1012 | 11.2 | 1 | 46 |
| **Total / Overall** | **383** | **4236** | **11.1** | **1** | **116** |

## 4. Video Properties (Dataset - Original)

### 4.1 Resolution & Framing
- `512x512` (1:1 square aspect ratio): **4,236 videos** (100.0%)

### 4.2 Frame Rate (FPS)
- `10.0 FPS`: **4,236 videos** (100.0%)

### 4.3 Video Duration & Frame Counts
- **Min Duration**: 2.90s (29 frames)
- **Max Duration**: 3.30s (33 frames)
- **Average Duration**: 3.19s (31.9 frames)

## 5. Landmark Analysis (Dataset - MP - CSV)

### 5.1 Landmark Column Integrity
- Total frames parsed: **280,418**
- Rows with `33` columns: **280,418** (100.0%)

> [!IMPORTANT]
> Every CSV row contains exactly **33 columns**, each containing `[x, y, z, visibility]` representing the standard MediaPipe Pose model (nose to ankles). **No hand landmarks (21 per hand) are present**. Re-extraction using MediaPipe Holistic or Pose + Hands is necessary.

### 5.2 Wrist Tracking & Visibility
Wrist landmarks (Left Wrist: index 15, Right Wrist: index 16) are critical for upper-body sign retargeting:

- Frames with Left Wrist visibility < 0.5: **86,217** (30.75%)
- Frames with Right Wrist visibility < 0.5: **51,322** (18.30%)
- Frames with Both Wrists visibility < 0.5: **46,966** (16.75%)

### 5.3 Frame Count Comparison (Video vs. CSV Rows)
- **Exact frame count matches**: 58 (1.37%)
- **CSV row count range**: Min=10, Max=259, Avg=66.2
- **Explanation**: Many CSVs have more rows (e.g. 70-120) than the 10 FPS videos (approx 32 frames), indicating the CSVs were generated from higher-framerate (e.g. 25-30 FPS) original captures prior to downsampling or during an asynchronous capture session.

## 6. Nomenclature & Gloss Normalization

Found **49** sign names requiring normalization into standard uppercase identifiers:

| Original Dataset Label | Category | Normalized Gloss Identifier |
| :--- | :--- | :--- |
| `1. one` | Numbers | `ONE` |
| `10. ten` | Numbers | `TEN` |
| `11. eleven` | Numbers | `ELEVEN` |
| `12. twelve` | Numbers | `TWELVE` |
| `13. thirteen` | Numbers | `THIRTEEN` |
| `14. fourteen` | Numbers | `FOURTEEN` |
| `15. fifteen` | Numbers | `FIFTEEN` |
| `16. sixteen` | Numbers | `SIXTEEN` |
| `17. seventeen` | Numbers | `SEVENTEEN` |
| `18. eighteen` | Numbers | `EIGHTEEN` |
| `19. nineteen` | Numbers | `NINETEEN` |
| `2. two` | Numbers | `TWO` |
| `20. twenty` | Numbers | `TWENTY` |
| `3. three` | Numbers | `THREE` |
| `4. four` | Numbers | `FOUR` |
| `5. five` | Numbers | `FIVE` |
| `6. six` | Numbers | `SIX` |
| `7. seven` | Numbers | `SEVEN` |
| `8. eight` | Numbers | `EIGHT` |
| `9. nine` | Numbers | `NINE` |
| `Brother in law` | People | `BROTHER_IN_LAW` |
| `Bus station` | Places | `BUS_STATION` |
| `Ceiling fan` | Nouns | `CEILING_FAN` |
| `Cell phone` | Nouns | `CELL_PHONE` |
| `Day after tomorrow` | Days | `DAY_AFTER_TOMORROW` |
| `Dont know` | Adverb | `DONT_KNOW` |
| `Elder bro` | People | `ELDER_BRO` |
| `Elder sister` | People | `ELDER_SISTER` |
| `Get up` | Verbs | `GET_UP` |
| `Good evening` | Days | `GOOD_EVENING` |
| `Good morning` | Days | `GOOD_MORNING` |
| `Good night` | Days | `GOOD_NIGHT` |
| `Grand father` | People | `GRAND_FATHER` |
| `Grand mother` | People | `GRAND_MOTHER` |
| `Grand son` | People | `GRAND_SON` |
| `How are you` | Greetings | `HOW_ARE_YOU` |
| `How many` | Nouns | `HOW_MANY` |
| `How much` | Nouns | `HOW_MUCH` |
| `I know` | Verbs | `I_KNOW` |
| `Next door` | Places | `NEXT_DOOR` |
| `Not good` | Adjectives | `NOT_GOOD` |
| `Not like (dislike)` | Adverb | `DISLIKE` |
| `Police station` | Places | `POLICE_STATION` |
| `Sign language` | Nouns | `SIGN_LANGUAGE` |
| `Sister in law` | People | `SISTER_IN_LAW` |
| `Thank you` | Greetings | `THANK_YOU` |
| `Train station` | Places | `TRAIN_STATION` |
| `Younger bro` | People | `YOUNGER_BRO` |
| `Younger sister` | People | `YOUNGER_SISTER` |

## 7. Starter Set of 10 Signs (Phase 1 Selection)

Audit of the 10 recommended starter signs for Phase 1 & 2 proof-of-concept:

| Gloss | Category / Sign | Takes | Best Take Candidate | Video Frames | CSV Frames | Avg Wrist Visibility | Notes |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :--- |
| **`HELLO`** | Greetings/Hello | 102 | `Hello_049` | 32 | 61 | 0.981 | High visibility take |
| **`THANK_YOU`** | Greetings/Thank you | 116 | `Thank you_058` | 32 | 34 | 0.993 | High visibility take |
| **`YES`** | Interjection/Yes | 6 | `Yes_002` | 30 | 23 | 0.921 | High visibility take |
| **`NO`** | Determiner/No | 3 | `No_003` | 32 | 45 | 0.583 | High visibility take |
| **`GO`** | Verbs/Go | 34 | `Go_034` | 32 | 45 | 0.967 | High visibility take |
| **`COME`** | Verbs/Come | 34 | `Come_030` | 29 | 11 | 0.989 | High visibility take |
| **`EAT`** | Verbs/Eat | 46 | `Eat_021` | 32 | 118 | 0.934 | High visibility take |
| **`MOTHER`** | People/Mother | 28 | `Mother_027` | 32 | 56 | 0.998 | High visibility take |
| **`TODAY`** | Days/Today | 36 | `Today_036` | 32 | 53 | 0.996 | High visibility take |
| **`BLUE`** | Colors/Blue | 5 | `Blue_001` | 32 | 37 | 0.996 | High visibility take |

### Top 3 Take Candidates per Starter Sign

- **`HELLO`**: `Hello_049` (vis: 0.98, dur: 3.2s), `Hello_046` (vis: 0.97, dur: 3.2s), `Hello_047` (vis: 0.97, dur: 3.2s)
- **`THANK_YOU`**: `Thank you_058` (vis: 0.99, dur: 3.2s), `Thank you_055` (vis: 0.99, dur: 3.2s), `Thank you_056` (vis: 0.99, dur: 3.2s)
- **`YES`**: `Yes_002` (vis: 0.92, dur: 3.0s), `Yes_006` (vis: 0.91, dur: 3.2s), `Yes_004` (vis: 0.89, dur: 3.2s)
- **`NO`**: `No_003` (vis: 0.58, dur: 3.2s), `No_001` (vis: 0.48, dur: 3.2s), `No_002` (vis: 0.30, dur: 3.2s)
- **`GO`**: `Go_034` (vis: 0.97, dur: 3.2s), `Go_029` (vis: 0.95, dur: 3.2s), `Go_033` (vis: 0.93, dur: 3.1s)
- **`COME`**: `Come_030` (vis: 0.99, dur: 2.9s), `Come_028` (vis: 0.98, dur: 3.3s), `Come_034` (vis: 0.98, dur: 3.3s)
- **`EAT`**: `Eat_021` (vis: 0.93, dur: 3.2s), `Eat_003` (vis: 0.92, dur: 3.2s), `Eat_022` (vis: 0.89, dur: 3.2s)
- **`MOTHER`**: `Mother_027` (vis: 1.00, dur: 3.2s), `Mother_021` (vis: 1.00, dur: 3.2s), `Mother_023` (vis: 0.99, dur: 3.2s)
- **`TODAY`**: `Today_036` (vis: 1.00, dur: 3.2s), `Today_035` (vis: 0.99, dur: 3.2s), `Today_025` (vis: 0.98, dur: 3.2s)
- **`BLUE`**: `Blue_001` (vis: 1.00, dur: 3.2s), `Blue_002` (vis: 0.99, dur: 3.2s), `Blue_004` (vis: 0.99, dur: 3.2s)

## 8. Actionable Recommendations for Pipeline Implementation

1. **Proceed with 10 Starter Signs**: All 10 candidate words (`HELLO`, `THANK_YOU`, `YES`, `NO`, `GO`, `COME`, `EAT`, `MOTHER`, `TODAY`, `BLUE`) exist and have strong take options.
2. **Implement `build_manifest.py` (Phase 1)**: Walk `Dataset - Original` to generate `data/manifest.csv` with standard schema and default status `todo`, flagging the 10 starter takes.
3. **Implement `extract.py` with MediaPipe Holistic (Phase 2)**: Re-extract from the original 512x512 videos to record pose world coordinates and 21 landmark hands per frame.
4. **Signer & Handedness Review**: Check whether signers are consistently right-handed or left-handed in the starter takes using `Dataset - MP - VID` before retargeting.
