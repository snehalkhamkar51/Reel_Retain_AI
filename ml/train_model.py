import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from ml.evaluate import regression_metrics
from ml.preprocessing import make_preprocessor
from reelretain.features import EXCLUDED_DATASET_COLUMNS, FEATURE_COLUMNS, TARGET_COLUMNS


RANDOM_SEED = 42
TARGET_DEFINITIONS = {
    "watched_percentage_synthetic": "Synthetic percentage of a video's duration watched.",
    "average_watch_time_sec_synthetic": "Synthetic average watch time in seconds.",
    "early_dropoff_rate_pct_synthetic": "Synthetic percentage of viewers dropping off early.",
}
EXCLUDED_COLUMN_REASONS = {
    "record_id": "Identifier; not a predictive input.",
    "is_synthetic": "Synthetic-data provenance flag; constant in this dataset and not a video feature.",
    "speech_rate_wpm": "Not extracted from uploaded videos because speech transcription is not performed.",
    "hook_time_sec": "Not reliably measured by the current video extractor.",
    "caption_sync_score": "Not measured without time-aligned captions.",
    "face_presence": "Face detection is not implemented in the video extractor.",
    "topic_niche": "Semantic content category is not inferred from the upload.",
    "video_format": "Dataset labels content styles (for example, voiceover); the extractor does not classify them.",
    "posting_hour_utc": "Posting schedule metadata is not available from a video upload.",
    "posting_dayofweek_utc": "Posting schedule metadata is not available from a video upload.",
}
LIMITATIONS = [
    "All target values and input features are synthetic; metrics do not validate real audience behavior.",
    "Synthetic feature distributions and semantics may differ from features extracted from real uploads.",
    "The held-out test set is synthetic and cannot establish real-world predictive validity.",
    "Display estimates are clipped to physical presentation ranges; clipping does not make them measured facts.",
]


def load_dataset(path: str | Path) -> tuple[pd.DataFrame, list[str]]:
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}. Obtain the authentic "
            "reelretain_synthetic_training_dataset_5000.csv from its owner; "
            "this project does not generate replacement data."
        )
    frame = pd.read_csv(dataset_path)
    if frame.empty:
        raise ValueError("The training dataset is empty.")
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate dataset columns are not allowed: {duplicates}")
    missing_targets = [target for target in TARGET_COLUMNS if target not in frame.columns]
    if missing_targets:
        raise ValueError(f"Dataset is missing required synthetic target columns: {missing_targets}")
    unknown = [
        column for column in frame.columns
        if column not in FEATURE_COLUMNS
        and column not in TARGET_COLUMNS
        and column not in EXCLUDED_DATASET_COLUMNS
    ]
    if unknown:
        raise ValueError(f"Dataset contains unsupported input columns: {unknown}")
    feature_columns = [column for column in FEATURE_COLUMNS if column in frame.columns]
    if not feature_columns:
        raise ValueError(f"Dataset must contain at least one supported feature: {list(FEATURE_COLUMNS)}")
    missing_features = [column for column in FEATURE_COLUMNS if column not in feature_columns]
    if missing_features:
        raise ValueError(f"Dataset is missing measurable video features: {missing_features}")
    if "is_synthetic" in frame and not frame["is_synthetic"].eq(True).all():
        raise ValueError("The dataset is expected to contain only rows marked is_synthetic=True.")
    for column in feature_columns:
        original = frame[column]
        numeric = pd.to_numeric(original, errors="coerce")
        invalid = original.notna() & numeric.isna()
        if invalid.any() or np.isinf(numeric.dropna().to_numpy(dtype=float)).any():
            raise ValueError(f"Feature {column} must contain only finite numbers or missing values.")
        frame[column] = numeric
    for target in TARGET_COLUMNS:
        frame[target] = pd.to_numeric(frame[target], errors="coerce")
        values = frame[target].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Target {target} contains missing or non-finite values.")
    if len(frame) < 12:
        raise ValueError("At least 12 rows are required for train, validation, and test splits.")
    return frame, feature_columns


def candidate_models():
    return {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=200, random_state=RANDOM_SEED, n_jobs=1, min_samples_leaf=2
        ),
        "gradient_boosting": GradientBoostingRegressor(random_state=RANDOM_SEED),
    }


def train_models(dataset_path: str | Path, artifact_dir: str | Path) -> dict:
    frame, feature_columns = load_dataset(dataset_path)
    indices = np.arange(len(frame))
    train_validation_idx, test_idx = train_test_split(
        indices, test_size=0.15, random_state=RANDOM_SEED
    )
    train_idx, validation_idx = train_test_split(
        train_validation_idx, test_size=(0.15 / 0.85), random_state=RANDOM_SEED
    )
    features = frame[feature_columns]
    output_dir = Path(artifact_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "title": "ReelRetain AI model evaluation",
        "data_disclaimer": (
            "Evaluation uses synthetic inputs and synthetic targets only; it does not "
            "represent validation on genuine Instagram or other audience analytics."
        ),
        "dataset": {
            "filename": Path(dataset_path).name,
            "row_count": int(len(frame)),
            "feature_columns": feature_columns,
            "target_columns": list(TARGET_COLUMNS),
            "excluded_columns": {
                **{
                    column: EXCLUDED_COLUMN_REASONS[column]
                    for column in EXCLUDED_DATASET_COLUMNS
                    if column in frame.columns
                },
                **{target: "Prediction target; excluded from all model inputs." for target in TARGET_COLUMNS},
            },
            "column_dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        },
        "random_seed": RANDOM_SEED,
        "split": {"training_rows": len(train_idx), "validation_rows": len(validation_idx), "test_rows": len(test_idx)},
        "selection_procedure": (
            "For each target, select the candidate with the lowest validation MAE. "
            "Refit that pipeline on training plus validation rows, then report final "
            "metrics on the untouched test split."
        ),
        "targets": {},
        "limitations": LIMITATIONS,
        "domain_mismatch": (
            "The dataset features are synthetic and may not share distributions or semantics "
            "with measured uploads. In particular, scene_changes and motion_intensity use "
            "a sampled-frame approximation, and audio_energy uses decoded-audio RMS. "
            "Unavailable audio values at inference are imputed using training medians."
        ),
        "reproducible_training": (
            "python training/train_model.py --dataset data/reelretain_synthetic_training_dataset_5000.csv "
                "--model-dir models"
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    models = candidate_models()
    for target in TARGET_COLUMNS:
        candidate_results = {}
        for name, estimator in models.items():
            pipeline = Pipeline([
                ("preprocessing", make_preprocessor(features.iloc[train_idx])),
                ("model", estimator),
            ])
            pipeline.fit(features.iloc[train_idx], frame[target].iloc[train_idx])
            validation_predictions = pipeline.predict(features.iloc[validation_idx])
            candidate_results[name] = regression_metrics(
                frame[target].iloc[validation_idx], validation_predictions
            )
        selected_name = min(candidate_results, key=lambda name: candidate_results[name]["mae"])
        selected_pipeline = Pipeline([
            ("preprocessing", make_preprocessor(features.iloc[train_validation_idx])),
            ("model", candidate_models()[selected_name]),
        ])
        selected_pipeline.fit(
            features.iloc[train_validation_idx], frame[target].iloc[train_validation_idx]
        )
        test_predictions = selected_pipeline.predict(features.iloc[test_idx])
        target_report = {
            "definition": TARGET_DEFINITIONS[target],
            "selected_model": selected_name,
            "selection_metric": "validation_mae",
            "validation_metrics_by_candidate": candidate_results,
            "selected_model_test_metrics": regression_metrics(
                frame[target].iloc[test_idx], test_predictions
            ),
            "test_metrics_are_synthetic_data_only": True,
        }
        report["targets"][target] = target_report
        joblib.dump(
            {
                "target": target,
                "pipeline": selected_pipeline,
                "feature_columns": feature_columns,
                "dataset_rows": len(frame),
                "selected_model": selected_name,
                "metrics": target_report,
                "data_disclaimer": report["data_disclaimer"],
            },
            output_dir / f"{target}.joblib",
        )
    report_path = output_dir / "evaluation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Train ReelRetain AI synthetic-data models.")
    parser.add_argument("--dataset", default="data/reelretain_synthetic_training_dataset_5000.csv", help="Path to the synthetic training CSV.")
    parser.add_argument("--artifact-dir", default="models", help="Where model files and report are saved.")
    args = parser.parse_args(argv)
    try:
        report = train_models(args.dataset, args.artifact_dir)
    except (FileNotFoundError, ValueError, OSError, pd.errors.ParserError) as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        return 1
    for target, results in report["targets"].items():
        print(f"{target}: selected={results['selected_model']}; "
              f"synthetic test metrics={results['selected_model_test_metrics']}")
    print(f"Evaluation report: {Path(args.artifact_dir) / 'evaluation_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
