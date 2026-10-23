import io

import pytest

from reelretain.app import create_app
from reelretain.features import FeatureExtractionError, TARGET_COLUMNS


class ReadyPredictor:
    available = True

    def info(self):
        return {"available": True, "status": "ready"}

    def predict(self, _features):
        return {
            target: {
                "estimate": 5.0,
                "raw_estimate": 5.0,
                "unit": "%",
                "constrained": False,
                "interpretation": "constrained experimental estimate, not a measured fact",
            }
            for target in TARGET_COLUMNS
        }


@pytest.fixture
def app(tmp_path):
    return create_app({
        "TESTING": True,
        "ARTIFACT_DIR": tmp_path / "missing-artifacts",
        "UPLOAD_DIR": tmp_path / "uploads",
        "MAX_CONTENT_LENGTH": 1024,
        "AUDIO_TIMEOUT_SECONDS": 1,
        "VIDEO_TIMEOUT_SECONDS": 1,
    })


def test_health_and_model_info_report_missing_models(app):
    client = app.test_client()
    health = client.get("/api/health")
    info = client.get("/api/model-info")
    assert health.status_code == 200
    assert health.json["model_status"] == "unavailable"
    assert info.status_code == 200
    assert info.json["model"]["available"] is False
    assert "synthetic" in info.json["model"]["disclaimer"].lower()
    assert info.json["upload"]["max_bytes"] == 1024
    assert ".mp4" in info.json["upload"]["extensions"]


def test_homepage_and_frontend_assets_are_served_by_flask(app):
    client = app.test_client()

    homepage = client.get("/")
    stylesheet = client.get("/static/css/app.css")
    application = client.get("/static/js/app.js")
    api_module = client.get("/static/js/api.js")

    assert homepage.status_code == 200
    assert b"Know your reel" in homepage.data
    assert b"synthetic" in homepage.data.lower()
    assert b"Audience Attraction Lab" in homepage.data
    assert b"Experimental estimates \xe2\x80\x94 synthetic training data" in homepage.data
    assert b"id=\"topic\"" in homepage.data
    assert b"id=\"recommendation-list\"" in homepage.data
    assert b'type="module"' in homepage.data
    assert stylesheet.status_code == 200
    assert b"#080b12" in stylesheet.data
    assert application.status_code == 200
    assert "javascript" in application.mimetype
    assert b"analyzeVideo" in application.data
    assert b"Copy hook" in application.data
    assert b"Copy caption" in application.data
    assert b"navigator.clipboard.writeText" in application.data
    assert b"recommendations.items.map" in application.data
    assert api_module.status_code == 200


def test_analyze_rejects_missing_and_unsupported_uploads(app):
    client = app.test_client()
    missing = client.post("/api/analyze")
    unsupported = client.post(
        "/api/analyze",
        data={"video": (io.BytesIO(b"x"), "photo.png", "image/png")},
        content_type="multipart/form-data",
    )
    assert missing.status_code == 400
    assert missing.json["error"]["code"] == "missing_video"
    assert unsupported.status_code == 400
    assert unsupported.json["error"]["code"] == "unsupported_file"


def test_analyze_rejects_mismatched_content_type(app):
    response = app.test_client().post(
        "/api/analyze",
        data={"video": (io.BytesIO(b"x"), "clip.mp4", "image/png")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert response.json["error"]["code"] == "invalid_content_type"


def test_analyze_rejects_empty_upload_and_cleans_temp_file(app):
    app.extensions["predictor"] = ReadyPredictor()
    response = app.test_client().post(
        "/api/analyze",
        data={"video": (io.BytesIO(b""), "empty.mp4", "video/mp4")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 422
    assert response.json["error"]["code"] == "video_processing_failed"
    assert list((app.config["UPLOAD_DIR"]).glob("*")) == []


def test_analyze_returns_recovery_message_if_models_missing(app):
    response = app.test_client().post(
        "/api/analyze",
        data={"video": (io.BytesIO(b"video"), "clip.mp4", "video/mp4")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 503
    assert response.json["error"]["code"] == "model_unavailable"


def test_successful_analysis_cleans_temporary_upload(app, monkeypatch):
    app.extensions["predictor"] = ReadyPredictor()
    seen_path = {}

    def fake_extract(path, caption, _audio_timeout, _video_timeout):
        seen_path["path"] = path
        assert path.is_file()
        assert caption == "Hello #test"
        return {"video_duration_sec": 8.0}, ["Audio unavailable"]

    monkeypatch.setattr("reelretain.app.extract_video_features", fake_extract)
    response = app.test_client().post(
        "/api/analyze",
        data={
            "video": (io.BytesIO(b"video-bytes"), "my clip.mp4", "video/mp4"),
            "caption": "Hello #test",
            "topic": "urban gardening",
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    assert response.json["ok"] is True
    assert set(response.json["estimates"]) == set(TARGET_COLUMNS)
    assert response.json["recommendations"]["mode"] == "personalized"
    assert len([item for item in response.json["recommendations"]["items"] if item["category"] == "Opening hook"]) == 3
    assert "urban gardening" in response.json["recommendations"]["items"][0]["example"]
    assert not any(item["category"] == "Audio" for item in response.json["recommendations"]["items"])
    assert response.json["warnings"] == ["Audio unavailable"]
    assert not seen_path["path"].exists()


def test_successful_analysis_generates_general_recommendations_without_caption_or_audio(app, monkeypatch):
    app.extensions["predictor"] = ReadyPredictor()
    monkeypatch.setattr(
        "reelretain.app.extract_video_features",
        lambda *_args: ({"video_duration_sec": 10.0, "scene_changes": 3, "motion_intensity": 0.2,
                         "audio_energy": None, "silence_duration_sec": None}, []),
    )
    response = app.test_client().post(
        "/api/analyze",
        data={"video": (io.BytesIO(b"video-bytes"), "my clip.mp4", "video/mp4")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    assert response.json["recommendations"]["mode"] == "general"
    assert len(response.json["recommendations"]["items"]) >= 7
    assert not any(item["category"] == "Audio" for item in response.json["recommendations"]["items"])


def test_recommendation_error_is_logged_and_falls_back_without_failing_analysis(app, monkeypatch, caplog):
    app.extensions["predictor"] = ReadyPredictor()
    monkeypatch.setattr(
        "reelretain.app.extract_video_features",
        lambda *_args: ({"video_duration_sec": 10.0}, []),
    )

    def fail(*_args):
        raise RuntimeError("recommendation engine unavailable")

    monkeypatch.setattr("reelretain.app.generate_recommendations", fail)
    response = app.test_client().post(
        "/api/analyze",
        data={"video": (io.BytesIO(b"video-bytes"), "my clip.mp4", "video/mp4")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    assert response.json["recommendations"]["mode"] == "general"
    assert response.json["recommendations"]["items"]
    assert any("recommendation generation failed" in record.message.lower() for record in caplog.records)
    assert any("general advice is shown" in warning for warning in response.json["warnings"])


@pytest.mark.parametrize(
    ("field", "value"),
    [("topic", "x" * 121), ("caption", "x" * 2201)],
)
def test_analyze_rejects_oversized_recommendation_context(app, field, value):
    app.config["MAX_CONTENT_LENGTH"] = 4096
    response = app.test_client().post(
        "/api/analyze",
        data={"video": (io.BytesIO(b"video-bytes"), "clip.mp4", "video/mp4"), field: value},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert response.json["error"]["code"] == "invalid_context"


def test_analyze_returns_clear_processing_error(app, monkeypatch):
    app.extensions["predictor"] = ReadyPredictor()

    def fail(*_args):
        raise FeatureExtractionError("The video could not be decoded.")

    monkeypatch.setattr("reelretain.app.extract_video_features", fail)
    response = app.test_client().post(
        "/api/analyze",
        data={"video": (io.BytesIO(b"corrupt"), "bad.mp4", "video/mp4")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 422
    assert response.json["error"]["code"] == "video_processing_failed"


def test_oversized_upload_returns_json_413(app):
    response = app.test_client().post(
        "/api/analyze",
        data={"video": (io.BytesIO(b"x" * 2000), "large.mp4", "video/mp4")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 413
    assert response.json["error"]["code"] == "upload_too_large"
