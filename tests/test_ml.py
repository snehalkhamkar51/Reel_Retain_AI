import json

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression

from ml import train_model
from ml.train_model import load_dataset, train_models
from reelretain.features import TARGET_COLUMNS
from reelretain.predict import Predictor


def make_dataset(rows=24):
    return pd.DataFrame({
        "record_id": np.arange(rows),
        "is_synthetic": True,
        "video_duration_sec": np.linspace(8, 35, rows),
        "scene_changes": np.arange(rows) % 8,
        "motion_intensity": np.linspace(0.1, 0.9, rows),
        "silence_duration_sec": np.linspace(0.1, 2, rows),
        "caption_word_count": np.arange(rows) + 10,
        "hashtag_count": np.arange(rows) % 5,
        "audio_energy": np.linspace(0.1, 0.9, rows),
        "speech_rate_wpm": np.linspace(80, 180, rows),
        "hook_time_sec": 1.0,
        "caption_sync_score": 0.5,
        "face_presence": 1,
        "topic_niche": "food",
        "video_format": "voiceover",
        "posting_hour_utc": 10,
        "posting_dayofweek_utc": 3,
        "watched_percentage_synthetic": np.linspace(10, 90, rows),
        "average_watch_time_sec_synthetic": np.linspace(2, 20, rows),
        "early_dropoff_rate_pct_synthetic": np.linspace(80, 10, rows),
    })


def test_load_dataset_excludes_identifiers_and_targets(tmp_path):
    path = tmp_path / "data.csv"
    make_dataset().to_csv(path, index=False)

    frame, features = load_dataset(path)

    assert features == [
        "video_duration_sec", "scene_changes", "motion_intensity",
        "silence_duration_sec", "caption_word_count", "hashtag_count", "audio_energy",
    ]
    assert "record_id" not in features
    assert "is_synthetic" not in features
    assert "speech_rate_wpm" not in features
    assert set(features).isdisjoint(TARGET_COLUMNS)
    assert len(frame) == 24


def test_dataset_requires_real_file_and_rejects_unknown_features(tmp_path):
    with pytest.raises(FileNotFoundError, match="authentic"):
        load_dataset(tmp_path / "missing.csv")
    bad = make_dataset()
    bad["mystery_feature"] = 1
    path = tmp_path / "bad.csv"
    bad.to_csv(path, index=False)
    with pytest.raises(ValueError, match="unsupported input columns"):
        load_dataset(path)


def test_trains_all_targets_and_predicts_constrained_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(train_model, "candidate_models", lambda: {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=5, random_state=42, min_samples_leaf=2
        ),
        "gradient_boosting": GradientBoostingRegressor(
            random_state=42, n_estimators=5
        ),
    })
    dataset = tmp_path / "training.csv"
    make_dataset().to_csv(dataset, index=False)
    artifact_dir = tmp_path / "artifacts"

    report = train_models(dataset, artifact_dir)
    predictor = Predictor(artifact_dir)
    estimates = predictor.predict({
        "video_duration_sec": 10,
        "scene_changes": 2,
        "motion_intensity": 0.4,
        "silence_duration_sec": 1.0,
        "caption_word_count": 15,
        "hashtag_count": 2,
        "audio_energy": 0.4,
    })

    assert set(report["targets"]) == set(TARGET_COLUMNS)
    assert predictor.available
    assert set(estimates) == set(TARGET_COLUMNS)
    for target, estimate in estimates.items():
        assert estimate["unit"] in {"%", "seconds"}
        assert 0 <= estimate["estimate"] <= (10 if target == TARGET_COLUMNS[1] else 100)
        assert estimate["interpretation"] == "constrained experimental estimate, not a measured fact"
    saved_report = json.loads((artifact_dir / "evaluation_report.json").read_text(encoding="utf-8"))
    assert "synthetic" in saved_report["data_disclaimer"].lower()
    assert set(saved_report["dataset"]["feature_columns"]).isdisjoint(
        {*TARGET_COLUMNS, "record_id", "is_synthetic"}
    )
    assert saved_report["dataset"]["row_count"] == 24
    assert (artifact_dir / f"{TARGET_COLUMNS[0]}.joblib").is_file()


def test_synthetic_target_values_outside_display_range_are_preserved(tmp_path):
    frame = make_dataset()
    frame.loc[0, TARGET_COLUMNS[0]] = 115
    path = tmp_path / "bad-target.csv"
    frame.to_csv(path, index=False)
    loaded, _ = load_dataset(path)
    assert loaded.loc[0, TARGET_COLUMNS[0]] == 115
