from pathlib import Path
import pickle

import joblib
import numpy as np
import pandas as pd

from reelretain.features import FEATURE_COLUMNS, TARGET_COLUMNS


DISPLAY_UNITS = {
    "watched_percentage_synthetic": "%",
    "average_watch_time_sec_synthetic": "seconds",
    "early_dropoff_rate_pct_synthetic": "%",
}

TARGET_LABELS = {
    "watched_percentage_synthetic": "Estimated watched percentage",
    "average_watch_time_sec_synthetic": "Estimated average watch time",
    "early_dropoff_rate_pct_synthetic": "Estimated early drop-off rate",
}

DISCLAIMER = (
    "ReelRetain AI currently uses models trained on synthetic data. Its retention estimates have not "
    "been validated against real audience analytics and may differ from actual results. Use them as "
    "experimental guidance, not as guaranteed outcomes."
)


class ModelUnavailableError(RuntimeError):
    """Required trained model artifacts are missing or invalid."""


class Predictor:
    def __init__(self, artifact_dir: str | Path):
        self.artifact_dir = Path(artifact_dir)
        self.models = {}
        self.load_error = None
        self._load()

    def _load(self) -> None:
        loaded = {}
        try:
            for target in TARGET_COLUMNS:
                path = self.artifact_dir / f"{target}.joblib"
                if not path.is_file():
                    raise FileNotFoundError(path.name)
                bundle = joblib.load(path)
                if (
                    not isinstance(bundle, dict)
                    or bundle.get("target") != target
                    or not hasattr(bundle.get("pipeline"), "predict")
                    or not isinstance(bundle.get("feature_columns"), list)
                    or not bundle["feature_columns"]
                    or any(column not in FEATURE_COLUMNS for column in bundle["feature_columns"])
                    or not isinstance(bundle.get("selected_model"), str)
                    or not isinstance(bundle.get("dataset_rows"), int)
                    or not isinstance(bundle.get("metrics"), dict)
                    or "selected_model_test_metrics" not in bundle["metrics"]
                ):
                    raise ValueError(f"Invalid model artifact for {target}")
                loaded[target] = bundle
        except (OSError, ValueError, EOFError, KeyError, ImportError, pickle.UnpicklingError) as exc:
            self.load_error = str(exc)
            self.models = {}
            return
        self.models = loaded

    @property
    def available(self) -> bool:
        return set(self.models) == set(TARGET_COLUMNS)

    def info(self) -> dict:
        return {
            "available": self.available,
            "status": "ready" if self.available else "unavailable",
            "targets": list(TARGET_COLUMNS),
            "supported_features": list(FEATURE_COLUMNS),
            "selected_models": {
                target: self.models[target]["selected_model"] for target in self.models
            },
            "training_dataset_rows": {
                target: self.models[target]["dataset_rows"] for target in self.models
            } if self.available else {},
            "synthetic_test_metrics": {
                target: self.models[target]["metrics"]["selected_model_test_metrics"]
                for target in self.models
            } if self.available else {},
            "disclaimer": DISCLAIMER,
            "recovery": None if self.available else (
                "Train the models with the supplied synthetic dataset using "
                "`python training/train_model.py`."
            ),
        }

    def predict(self, features: dict) -> dict:
        if not self.available:
            raise ModelUnavailableError("Trained model artifacts are unavailable.")
        estimates = {}
        for target, bundle in self.models.items():
            columns = bundle["feature_columns"]
            missing = [column for column in columns if column not in features]
            if missing:
                raise ValueError(f"Required extracted features are unavailable: {', '.join(missing)}")
            row = pd.DataFrame([{key: features[key] for key in columns}])
            prediction = float(bundle["pipeline"].predict(row)[0])
            if not np.isfinite(prediction):
                raise ValueError(f"The model produced a non-finite estimate for {target}.")
            low, high = (0.0, 100.0)
            if target == "average_watch_time_sec_synthetic":
                high = float(features["video_duration_sec"])
            constrained = min(max(prediction, low), high)
            estimates[target] = {
                "label": TARGET_LABELS[target],
                "unit": DISPLAY_UNITS[target],
                "raw_estimate": prediction,
                "estimate": constrained,
                "constrained": prediction != constrained,
                "interpretation": "constrained experimental estimate, not a measured fact",
                "selected_model": bundle["selected_model"],
            }
        return estimates
