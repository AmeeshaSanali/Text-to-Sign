# SLSL Avatar Engine — API Contract & Integration Specification

**Version**: 0.2.0  
**Backend Framework**: FastAPI (Python 3.10+)  
**Consumers**: Unity WebGL Client, Unity Editor, Web Dashboard

---

## 1. Overview & Data Flow

```
User Input (English)
       │
       ▼
[POST /api/v1/translate] ──► Grammar Reordering & Dictionary Translation
       │
       ▼
TranslateResponse:
  - glosses: ["TODAY", "HELLO", "BLUE"]
  - animation_sequence:
      - order: 0, gloss: "TODAY", sign_status: "approved", animation_url: "/animations/slsl/TODAY.json"
      - order: 1, gloss: "HELLO", sign_status: "approved", animation_url: "/animations/slsl/HELLO.json"
      - order: 2, gloss: "BLUE",  sign_status: "approved", animation_url: "/animations/slsl/BLUE.json"
       │
       ▼
Unity Client (PoseJsonLoader.cs & SignPlayer.cs):
  - Fetches GET /animations/slsl/<GLOSS>.json
  - Deserializes bone rotations per frame
  - Plays sign on Mixamo Y Bot with crossfade slerp
```

---

## 2. API Endpoints

### 2.1 POST `/api/v1/translate`
Translates English text into an ordered sign-gloss sequence with corresponding animation metadata.

#### Request Body
```json
{
  "text": "Hello, how are you today?",
  "enable_reordering": true,
  "fingerspell_unknown": true
}
```

| Field | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `text` | string | *required* | Input sentence in English. |
| `enable_reordering` | bool | `true` | Apply SLSL grammar rules (time-fronting, wh-question end-shift). |
| `fingerspell_unknown` | bool | `true` | Prefix unrecognized words with `#` for fingerspelling fallback. |

#### Response (`200 OK`)
```json
{
  "original_text": "Hello, how are you today?",
  "glosses": ["TODAY", "HELLO", "HOW", "YOU"],
  "tokens": [
    {"original_word": "today", "gloss": "TODAY", "is_fingerspelled": false, "rule_applied": "time_fronting"},
    {"original_word": "hello", "gloss": "HELLO", "is_fingerspelled": false, "rule_applied": "dictionary_match"},
    {"original_word": "how", "gloss": "HOW", "is_fingerspelled": false, "rule_applied": "phrase_mapping"},
    {"original_word": "you", "gloss": "YOU", "is_fingerspelled": false, "rule_applied": "phrase_mapping"}
  ],
  "unmapped_words": [],
  "fingerspelled_words": [],
  "reordered": true,
  "animation_sequence": [
    {
      "order": 0,
      "gloss": "TODAY",
      "animation_clip": "TODAY",
      "is_fingerspelled": false,
      "estimated_duration_sec": 1.2,
      "animation_url": "/animations/slsl/TODAY.json",
      "has_pose_animation": true,
      "sign_status": "approved"
    },
    {
      "order": 1,
      "gloss": "HELLO",
      "animation_clip": "HELLO",
      "is_fingerspelled": false,
      "estimated_duration_sec": 1.2,
      "animation_url": "/animations/slsl/HELLO.json",
      "has_pose_animation": true,
      "sign_status": "approved"
    }
  ],
  "method": "rule_based_dictionary"
}
```

#### Sign Status Values
- `"approved"`: The sign has a verified, retargeted 3D bone animation available via `animation_url`.
- `"clip_fallback"`: The sign maps to an embedded test animation clip (e.g. `SALUTE`, `SHOOTING_ARROW`).
- `"no_sign"`: The sign does not yet have an approved animation. In the Unity UI, this is flagged as "no sign available".

---

### 2.2 GET `/animations/slsl/{GLOSS}.json`
Static file endpoint serving the frame-by-frame local bone rotations for the requested sign.

#### Response Format (`200 OK`)
```json
{
  "gloss": "BLUE",
  "language": "slsl",
  "fps": 30,
  "frameCount": 94,
  "durationSec": 3.1,
  "frames": [
    {
      "bones": {
        "RightUpperArm": [0.0, -0.49601, -0.47012, 0.73004],
        "RightLowerArm": [0.0, -0.67678, 0.20233, 0.70784],
        "LeftUpperArm": [0.0, 0.14337, 0.61407, 0.77612],
        "LeftLowerArm": [0.0, 0.03082, 0.51549, 0.85634],
        "RightHand": [0.0, -0.12329, 0.34884, 0.92904],
        "RightIndexIntermediate": [0.0, 0.0, 0.31344, 0.94961],
        "RightIndexDistal": [0.0, 0.0, 0.57358, 0.81915]
      }
    }
  ]
}
```

Bones conform strictly to Unity's `HumanBodyBones` enum names.

---

### 2.3 GET `/api/v1/animations/approved`
Lists all sign glosses that currently have approved 3D animations ready for playback.

#### Response (`200 OK`)
```json
{
  "count": 10,
  "approved_glosses": [
    "BLUE",
    "COME",
    "EAT",
    "GO",
    "HELLO",
    "MOTHER",
    "NO",
    "THANK_YOU",
    "TODAY",
    "YES"
  ],
  "items": [
    {
      "gloss": "BLUE",
      "status": "approved",
      "category": "Colors",
      "url": "/animations/slsl/BLUE.json",
      "file_path": "/Users/nipun/Desktop/Projects/Ameesha/data/animations/slsl/BLUE.json",
      "source_take": "Blue_001"
    }
  ]
}
```

---

### 2.4 GET `/api/v1/animations/manifest`
Returns the status of all 383 dataset signs in `data/manifest.csv`.

#### Response (`200 OK`)
```json
{
  "total_signs": 383,
  "signs": [
    {
      "gloss": "BLUE",
      "label_in_dataset": "Blue",
      "category": "Colors",
      "language": "slsl",
      "source_video": "Dataset - Original/Colors/Blue/Blue_001.mp4",
      "take": "Blue_001",
      "signer_id": "unknown",
      "mirrored": false,
      "license": "unconfirmed",
      "status": "approved",
      "notes": "starter_set; high visibility take (1.00 wrist vis)"
    }
  ]
}
```

---

## 3. Unity Client Integration Contract

1. **Fetching**: `PoseJsonLoader.cs` issues a GET request to `http://127.0.0.1:8000/animations/slsl/{GLOSS}.json` via `UnityWebRequest`.
2. **Parsing**: JSON text is parsed into `AnimationClipData` with `List<Dictionary<HumanBodyBones, Quaternion>>`.
3. **Playback**: `SignPlayer.cs` overrides bone rotations in `LateUpdate()` with `boneTransform.localRotation = quaternion`.
4. **Blending**: Between consecutive signs, `SignPlayer.cs` crossfades poses over `0.2` seconds using `Quaternion.Slerp`.
