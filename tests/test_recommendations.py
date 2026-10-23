from reelretain.recommendations import generate_recommendations


def test_recommendations_use_topic_and_measured_visual_features():
    features = {
        "video_duration_sec": 20,
        "scene_changes": 1,
        "motion_intensity": 0.12,
        "audio_energy": 0.08,
        "silence_duration_sec": 3,
    }
    cooking = generate_recommendations(features, topic="sourdough baking")
    running = generate_recommendations(features, topic="beginner running")

    assert cooking["mode"] == "personalized"
    assert running["mode"] == "personalized"
    cooking_examples = [item["example"] for item in cooking["items"] if item["example"]]
    running_examples = [item["example"] for item in running["items"] if item["example"]]
    assert cooking_examples != running_examples
    assert any("sourdough baking" in example for example in cooking_examples)
    assert any("beginner running" in example for example in running_examples)
    assert any("1 sampled scene-change candidates" in item["what"] for item in cooking["items"])
    assert any(item["category"] == "Audio" for item in cooking["items"])
    assert all(
        item["priority"] in {"High", "Medium", "Low"}
        and item["what"] and item["why"] and item["action"]
        for item in cooking["items"]
    )


def test_recommendations_fall_back_to_caption_context_without_audio():
    response = generate_recommendations(
        {"video_duration_sec": 12, "scene_changes": 4, "motion_intensity": 0.3,
         "audio_energy": None, "silence_duration_sec": None},
        caption="A simple guide to balcony herbs",
    )

    assert response["mode"] == "personalized"
    assert any("balcony herbs" in item["what"] for item in response["items"])
    assert not any(item["category"] == "Audio" for item in response["items"])


def test_recommendations_handle_missing_and_invalid_features_as_general_advice():
    response = generate_recommendations(
        {"video_duration_sec": float("nan"), "scene_changes": "unknown", "audio_energy": None},
        caption="",
        topic="",
    )

    assert response["mode"] == "general"
    assert response["items"]
    assert "General advice" in response["context"]
    assert not any(item["category"] == "Audio" for item in response["items"])


def test_fallback_preserves_audio_guidance_when_audio_was_measured():
    from reelretain.recommendations import fallback_recommendations

    response = fallback_recommendations({"audio_energy": 0.15})

    assert any(item["category"] == "Audio" for item in response["items"])
