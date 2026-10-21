"""Tests for FastAPI backend."""

import pytest
from fastapi.testclient import TestClient

try:
    from src.app import app
except ImportError:
    from backend.src.app import app

client = TestClient(app)


def test_root_dashboard():
    response = client.get("/")
    assert response.status_code == 200
    assert "SLSL Avatar Engine" in response.text


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_translate_api():
    payload = {"text": "I am going to the store", "engine": "rules"}
    response = client.post("/api/v1/translate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["glosses"] == ["I", "GO", "STORE"]
    assert len(data["animation_sequence"]) == 3
    assert data["animation_sequence"][0]["gloss"] == "I"
    assert data["animation_sequence"][1]["gloss"] == "GO"
    assert data["animation_sequence"][2]["gloss"] == "STORE"


def test_translate_sinhala_api():
    payload = {"text": "ආයුබෝවන්, මගේ අම්මා කනවා", "engine": "rules"}
    response = client.post("/api/v1/translate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["detected_language"] == "si"
    assert "HELLO" in data["glosses"]
    assert "MOTHER" in data["glosses"]
    assert "EAT" in data["glosses"]
    assert len(data["animation_sequence"]) >= 3


def test_translate_compound_oov_api():
    payload = {"text": "hospital", "engine": "rules"}
    response = client.post("/api/v1/translate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["glosses"] == ["DOCTOR", "PLACE"]
    assert data["animation_sequence"][0]["match_type"] == "compound"


def test_translate_and_resolve_2d_endpoint():
    payload = {"text": "ආයුබෝවන්, මගේ අම්මා කනවා", "engine": "rules"}
    response = client.post("/api/v1/translate-and-resolve-2d", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["detected_language"] == "si"
    assert ("HELLO" in data["glosses"] or "AYUBOWAN" in data["glosses"])
    assert "MOTHER" in data["glosses"]
    assert len(data["sequence"]) >= 3
    # Check that HELLO and MOTHER have frames loaded
    hello_item = next(s for s in data["sequence"] if s["gloss"] in ["HELLO", "AYUBOWAN"])
    assert hello_item["status"] == "available"
    assert hello_item["frame_count"] > 0
    assert len(hello_item["frames"][0]) == 33




def test_engine_status_endpoint():
    response = client.get("/api/v1/engine-status")
    assert response.status_code == 200
    data = response.json()
    assert "gemini_available" in data
    assert "default_engine" in data
    assert data["model"] == "gemini-2.5-flash"


def test_vocabulary_endpoint():
    response = client.get("/api/v1/vocabulary")
    assert response.status_code == 200
    data = response.json()
    assert data["total_glosses"] > 0
    assert "GO" in data["glosses"]
    assert "STORE" in data["glosses"]


def test_sample_sentences_endpoint():
    response = client.get("/api/v1/sample-sentences")
    assert response.status_code == 200
    data = response.json()
    assert len(data["samples"]) > 0


def test_static_animations_endpoint():
    # Verify static file serving of approved BLUE.json
    response = client.get("/animations/slsl/BLUE.json")
    assert response.status_code == 200
    data = response.json()
    assert data["gloss"] == "BLUE"
    assert data["fps"] == 30
    assert len(data["frames"]) > 0


def test_approved_animations_endpoint():
    response = client.get("/api/v1/animations/approved")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] >= 10
    assert "BLUE" in data["approved_glosses"]
    assert "HELLO" in data["approved_glosses"]


def test_manifest_endpoint():
    response = client.get("/api/v1/animations/manifest")
    assert response.status_code == 200
    data = response.json()
    assert data["total_signs"] == 383


def test_translate_with_pose_metadata():
    payload = {"text": "hello blue today", "engine": "rules"}
    response = client.post("/api/v1/translate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["glosses"] == ["TODAY", "HELLO", "BLUE"]
    for cue in data["animation_sequence"]:
        assert cue["has_pose_animation"] is True
        assert cue["sign_status"] == "approved"
        assert cue["animation_url"] is not None
        assert cue["animation_url"].startswith("/animations/slsl/")


def test_visualizer_html_endpoint():
    response = client.get("/visualizer")
    assert response.status_code == 200
    assert "SLSL 2D Pose Coordinate Animation Visualizer" in response.text
    assert "poseCanvas" in response.text


def test_dataset_categories_endpoint():
    response = client.get("/api/v1/dataset/categories")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] >= 16
    assert "Greetings" in data["categories"]
    assert "Colors" in data["categories"]


def test_dataset_signs_endpoint():
    response = client.get("/api/v1/dataset/signs?category=Greetings")
    assert response.status_code == 200
    data = response.json()
    assert data["category"] == "Greetings"
    assert data["total"] > 0
    assert "Hello" in data["signs"]


def test_dataset_takes_endpoint():
    response = client.get("/api/v1/dataset/takes?category=Greetings&sign=Hello")
    assert response.status_code == 200
    data = response.json()
    assert data["sign"] == "Hello"
    assert data["total"] > 0
    assert any("Hello_049" in t for t in data["takes"])


def test_dataset_landmarks_endpoint():
    response = client.get("/api/v1/dataset/landmarks?category=Greetings&sign=Hello&take=Hello_049")
    assert response.status_code == 200
    data = response.json()
    assert data["sign"] == "Hello"
    assert data["frame_count"] > 0
    # Check 33 MediaPipe pose landmarks per frame
    first_frame = data["frames"][0]
    assert len(first_frame) == 33
    # Check [x, y, z, vis] format
    assert len(first_frame[0]) == 4
    # Check coordinate values are within valid normalized space
    assert 0.0 <= first_frame[0][0] <= 1.0  # x
    assert 0.0 <= first_frame[0][1] <= 1.0  # y


def test_dataset_kinematics_endpoint():
    payload = {
        "category": "Greetings",
        "sign": "Hello",
        "take": "Hello_049",
        "fps": 10.0,
    }
    response = client.post("/api/v1/dataset/kinematics", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["frame_count"] > 0
    assert data["fps"] == 10.0
    assert len(data["timestamps"]) == data["frame_count"]

    # Check 3 anatomical color groups
    assert "head" in data and "nose" in data["head"]
    assert "body" in data and "left_shoulder" in data["body"]
    assert "hands" in data and "right_wrist" in data["hands"]

    # Check color assignments
    assert data["head"]["color"] == "#00E5FF"
    assert data["body"]["color"] == "#00E676"
    assert data["hands"]["color"] == "#FF3366"

    # Check metrics
    assert "metrics" in data
    assert "peak_hand_velocity" in data["metrics"]
    assert data["metrics"]["peak_hand_velocity"] >= 0.0

    # Test passing custom raw frames
    dummy_frame = [[0.5, 0.5, 0.0, 1.0]] * 33
    raw_payload = {"frames": [dummy_frame, dummy_frame], "fps": 10.0}
    raw_res = client.post("/api/v1/dataset/kinematics", json=raw_payload)
    assert raw_res.status_code == 200
    raw_data = raw_res.json()
    assert raw_data["frame_count"] == 2
    assert len(raw_data["timestamps"]) == 2


def test_neural_ground_api_endpoint():
    # Pre-populate query cache to ensure determinism and zero API cost
    try:
        from src.app import neural_grounder
    except ImportError:
        from backend.src.app import neural_grounder

    embedder = neural_grounder.embedder
    doc_idx = embedder.gloss_to_index["DOCTOR"]
    hosp_idx = embedder.gloss_to_index["HOSPITAL"]
    embedder._query_cache["doctor"] = embedder.embeddings_matrix[doc_idx]
    embedder._query_cache["hospital"] = embedder.embeddings_matrix[hosp_idx]

    response = client.post("/api/v1/neural-ground", json={"text": "doctor hospital", "language": "en"})
    assert response.status_code == 200
    data = response.json()
    assert data["glosses"] == ["DOCTOR", "HOSPITAL"]
    assert len(data["tokens"]) == 2
    assert data["tokens"][0]["match_type"] == "exact"
    assert "alternatives" in data["tokens"][0]


def test_translate_and_resolve_2d_neural_engine():
    try:
        from src.app import neural_grounder
    except ImportError:
        from backend.src.app import neural_grounder

    embedder = neural_grounder.embedder
    hello_idx = embedder.gloss_to_index["HELLO"]
    embedder._query_cache["hello"] = embedder.embeddings_matrix[hello_idx]

    payload = {
        "text": "hello",
        "language": "en",
        "engine": "neural"
    }
    response = client.post("/api/v1/translate-and-resolve-2d", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["method"] == "neural_dense_grounding_r3072"
    assert "HELLO" in data["glosses"]
    assert len(data["sequence"]) > 0
    first = data["sequence"][0]
    assert first["gloss"] == "HELLO"
    assert first["status"] == "available"
    assert first["frame_count"] > 0
    assert len(first["frames"]) > 0


def test_benchmark_ablation_endpoint():
    response = client.get("/api/v1/evaluation/benchmark?live_llm=false")
    assert response.status_code == 200
    data = response.json()
    assert data["corpus_size"] == 14
    assert len(data["results"]) == 4
    model_names = [r["model_name"] for r in data["results"]]
    assert any("Model A" in m for m in model_names)
    assert any("Model B" in m for m in model_names)
    assert any("Model C" in m for m in model_names)
    assert any("Model D" in m for m in model_names)
    assert "latex_table" in data
    assert "\\begin{table}" in data["latex_table"]


def test_translate_and_resolve_2d_fingerspelling_fallback():
    # Test unmapped proper noun routing to parametric fingerspelling
    payload = {
        "text": "Nipun",
        "language": "en",
        "engine": "rules"  # using rules with fingerspell_unknown=True
    }
    response = client.post("/api/v1/translate-and-resolve-2d", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["sequence"]) > 0
    item = data["sequence"][0]
    assert item["is_fingerspelled"] is True
    assert item["status"] == "available"
    assert item["category"] == "Dactylology"
    assert item["take"] == "Parametric_Keyframe_Spline"
    assert item["frame_count"] > 0
    assert len(item["frames"]) == item["frame_count"]
    # Check 33 landmarks
    assert len(item["frames"][0]) == 33




