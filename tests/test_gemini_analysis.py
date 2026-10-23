import json
from types import SimpleNamespace

import pytest

from reelretain.gemini_analysis import (
    GeminiAnalysisError,
    GeminiVideoAnalyzer,
    validate_analysis,
)


def valid_payload():
    return {
        "main_topic_and_niche": "Home gardening",
        "description": "A short planting demonstration.",
        "opening_hook": "A quick question about herbs.",
        "hook_clarity": "It names the topic but could preview the result.",
        "main_message": "Plant herbs in a container.",
        "content_structure": "Intro, planting steps, result.",
        "useful_moments": [
            {"timestamp_seconds": 3.5, "observation": "The container is shown.", "suggested_action": "Label the soil."},
            {"timestamp_seconds": 20, "observation": "Out of range.", "suggested_action": "Review."},
        ],
        "visual_presentation": "Close-up planting shots.",
        "audio_observations": None,
        "alternative_hooks": ["Hook one", "Hook two", "Hook three"],
        "editing_changes": [
            {
                "timestamp_seconds": 4,
                "what_to_change": "Show the herb varieties before planting.",
                "operation": "rearrange",
                "example": "Place the variety labels on screen first.",
                "viewer_rationale": "This clarifies the demonstration.",
                "priority": "Medium",
            }
        ],
        "caption_alternatives": [{"caption": "Grow herbs in a small space.", "cta": "Which herb would you plant?"}],
        "on_screen_text": ["Three steps to plant herbs"],
        "subtitle_recommendations": [],
        "storyboard": ["Show the pot", "Plant the herbs", "Show the result"],
        "ending_and_payoff": "Show the completed pot and recap the steps.",
        "potential_audiences": [
            {
                "age_band": "Not age-specific",
                "relevance": "Relevant to people interested in gardening.",
                "need_or_interest": "Want to grow herbs in limited space.",
                "accessible_presentation": "Show each step clearly with common materials.",
                "reasoning": "The content is a practical gardening tutorial.",
            }
        ],
        "uncertainties": ["The plant species are not fully clear."],
        "unavailable_observations": ["Actual viewer response"],
    }


class FakeFiles:
    def __init__(self, processing=False, delete_error=False):
        self.remote = SimpleNamespace(
            name="files/temporary-video",
            uri="gs://temporary-video",
            mime_type="video/mp4",
            state="PROCESSING" if processing else "ACTIVE",
        )
        self.processing = processing
        self.deleted = []
        self.delete_error = delete_error

    def upload(self, **kwargs):
        self.upload_args = kwargs
        return self.remote

    def get(self, **kwargs):
        self.get_args = kwargs
        self.remote.state = "ACTIVE"
        return self.remote

    def delete(self, **kwargs):
        self.deleted.append(kwargs["name"])
        if self.delete_error:
            raise RuntimeError("delete failed")


class FakeInteractions:
    def __init__(self, output):
        self.output = output
        self.args = None

    def create(self, **kwargs):
        self.args = kwargs
        if isinstance(self.output, Exception):
            raise self.output
        return SimpleNamespace(output_text=self.output)


class FakeClient:
    def __init__(self, output, processing=False, delete_error=False):
        self.files = FakeFiles(processing, delete_error)
        self.interactions = FakeInteractions(output)
        self.closed = False

    def close(self):
        self.closed = True


def test_missing_key_does_not_construct_or_call_gemini(tmp_path):
    called = []
    analyzer = GeminiVideoAnalyzer("", client_factory=lambda **_: called.append(True))

    with pytest.raises(GeminiAnalysisError, match="not configured"):
        analyzer.analyze(tmp_path / "never-used.mp4", {}, "")
    assert called == []


def test_structured_video_analysis_uses_files_api_and_deletes_remote_file(tmp_path):
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"video")
    client = FakeClient(json.dumps(valid_payload()), processing=True)
    analyzer = GeminiVideoAnalyzer(
        "test-key", client_factory=lambda **_: client, poll_interval_seconds=0.001
    )

    result = analyzer.analyze(path, {"video_duration_sec": 10}, "herb caption", "gardening")

    assert result["main_topic_and_niche"] == "Home gardening"
    assert result["useful_moments"][0]["timestamp_seconds"] == 3.5
    assert result["useful_moments"][1]["timestamp_seconds"] is None
    assert result["editing_changes"][0]["evidence_source"] == "gemini_content_analysis"
    assert result["potential_audiences"][0]["evidence_source"] == "gemini_content_analysis"
    assert client.files.upload_args["file"] == str(path)
    assert client.files.deleted == ["files/temporary-video"]
    assert client.closed is True
    assert client.interactions.args["model"] == "gemini-3.8-flash"
    assert client.interactions.args["response_format"]["mime_type"] == "application/json"
    assert client.interactions.args["input"][0]["type"] == "video"


def test_provider_failure_returns_domain_error_and_still_deletes_remote_upload(tmp_path):
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"video")
    client = FakeClient(RuntimeError("429 rate limit"))
    analyzer = GeminiVideoAnalyzer("test-key", client_factory=lambda **_: client)

    with pytest.raises(GeminiAnalysisError, match="failed"):
        analyzer.analyze(path, {"video_duration_sec": 10})

    assert client.files.deleted == ["files/temporary-video"]


def test_remote_delete_failure_is_logged_without_discarding_valid_analysis(tmp_path, caplog):
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"video")
    client = FakeClient(json.dumps(valid_payload()), delete_error=True)
    analyzer = GeminiVideoAnalyzer("test-key", client_factory=lambda **_: client)

    assert analyzer.analyze(path, {"video_duration_sec": 10})["main_topic_and_niche"] == "Home gardening"
    assert "Could not delete temporary Gemini file" in caplog.text


def test_file_processing_timeout_deletes_remote_upload(tmp_path, monkeypatch):
    import reelretain.gemini_analysis as module

    path = tmp_path / "clip.mp4"
    path.write_bytes(b"video")
    client = FakeClient(json.dumps(valid_payload()), processing=True)
    analyzer = GeminiVideoAnalyzer("test-key", timeout_seconds=1, client_factory=lambda **_: client)
    clock_values = iter([0.0, 2.0])
    monkeypatch.setattr(module.time, "monotonic", lambda: next(clock_values))
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)

    with pytest.raises(GeminiAnalysisError, match="timed out"):
        analyzer.analyze(path, {"video_duration_sec": 10})

    assert client.files.deleted == ["files/temporary-video"]


def test_invalid_structured_output_is_rejected():
    payload = valid_payload()
    payload["alternative_hooks"] = ["only one"]

    with pytest.raises(GeminiAnalysisError, match="exactly three"):
        validate_analysis(payload, 10)


def test_timestamp_values_are_finite_nonnegative_and_within_duration():
    payload = valid_payload()
    payload["useful_moments"].extend([
        {"timestamp_seconds": -1, "observation": "negative", "suggested_action": "ignore"},
        {"timestamp_seconds": float("nan"), "observation": "nan", "suggested_action": "ignore"},
    ])
    normalized = validate_analysis(payload, 10)

    assert [moment["timestamp_seconds"] for moment in normalized["useful_moments"]] == [
        3.5, None, None, None,
    ]
