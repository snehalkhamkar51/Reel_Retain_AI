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
        "video_duration_sec": np.linspace(8, 35, rows),
        "orientation": np.where(np.arange(rows) % 2, "portrait", "landscape"),
        "watched_percentage_synthetic": np.linspace(10, 90, rows),
        "average_watch_time_sec_synthetic": np.linspace(2, 20, rows),
        "early_dropoff_rate_pct_synthetic": np.linspace(80, 10, rows),
    })


def test_load_dataset_excludes_identifiers_and_targets(tmp_path):
    path = tmp_path / "data.csv"
    make_dataset().to_csv(path, index=False)

    frame, features = load_dataset(path)

    assert features == ["video_duration_sec", "orientation"]
    assert "record_id" not in features
    assert set(features).isdisjoint(TARGET_COLUMNS)
    assert len(frame) == 24


def test_dataset_requires_real_file_and_rejects_unknown_features(tmp_path):
    with pytest.raises(FileNotFoundError, match="authentic"):
        load_dataset(tmp_path / "missing.csv")
    bad = make_dataset().rename(columns={"orientation": "mystery_feature"})
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
        "orientation": "portrait",
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
    assert (artifact_dir / f"{TARGET_COLUMNS[0]}.joblib").is_file()


def test_invalid_target_values_are_rejected(tmp_path):
    frame = make_dataset()
    frame.loc[0, TARGET_COLUMNS[0]] = 101
    path = tmp_path / "bad-target.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="between 0 and 100"):
        load_dataset(path)
