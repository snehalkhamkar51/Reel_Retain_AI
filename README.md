# ReelRetain AI

ReelRetain AI is a local prototype for extracting measurable features from short videos and estimating watched percentage, average watch time, and early drop-off rate.

> **Synthetic-data limitation:** The expected training dataset contains synthetic input features and synthetic target values. Model metrics and predictions are experimental and are not evidence of real-audience retention performance. Video-derived features may also differ in distribution and meaning from the synthetic training features. Do not use these estimates as measured facts or for publication decisions.

## Architecture

- `reelretain/features.py` extracts video metadata, sampled-frame scene changes and motion, optional FFmpeg audio energy/silence, and caption counts. Speech rate, hook timing, and caption synchronization are left unavailable unless a reliable measurement is implemented.
- `training/train_model.py` trains from the supplied local dataset path; `ml/` validates the CSV, fits a separate scikit-learn pipeline for each synthetic target, compares Linear Regression, Random Forest, and Gradient Boosting, and saves selected pipelines and a report.
- `reelretain/predict.py` loads trained artifacts and returns constrained experimental estimates.
- `reelretain/app.py` provides the Flask health, model-info, and video-analysis endpoints.

The pipeline uses only recognized feature columns; target columns and `record_id` are never model inputs. Numeric and categorical features are imputed and transformed within a `ColumnTransformer` fitted on training data. Candidate selection uses validation MAE, with a separate held-out test set reserved for the selected model's final metrics.

## Prerequisites and setup

- Python 3.10 or newer
- FFmpeg on `PATH` for audio-energy and silence extraction (video analysis still runs without it and reports audio features as unavailable)
- The supplied `data/reelretain_synthetic_training_dataset_5000.csv` synthetic dataset.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Training validates the target and supported feature schema and exits with a clear error if the dataset is missing or incompatible.

## Train and evaluate

```powershell
python training\train_model.py
```

The default dataset is `data/reelretain_synthetic_training_dataset_5000.csv`. Model files and `evaluation_report.json` are written to `models/`; the trained artifacts and report are committed so the API can run immediately after checkout. The synthetic CSV remains local and is not committed. To specify other paths:

```powershell
python training\train_model.py --dataset C:\path\to\reelretain_synthetic_training_dataset_5000.csv --artifact-dir C:\path\to\models
```

Each target's report lists validation MAE/RMSE/R² for all candidate models and held-out test MAE/RMSE/R² for the model selected by validation MAE. The default deterministic split assigns 70% to training, 15% to validation, and 15% to the untouched test set. Metrics describe synthetic data only. Synthetic labels may exceed physical display ranges (the supplied watched-percentage column includes values over 100); raw model predictions and training metrics are not rewritten, while API display estimates are constrained and marked as such. Use artifacts only from a trusted source: joblib/pickle files can execute code when loaded.

## Run the app

```powershell
python run.py
```

The server listens on `http://127.0.0.1:5000` by default. Environment configuration (all optional; copy `.env.example` to `.env` to load local settings):

| Variable | Default | Purpose |
|---|---:|---|
| `REELRETAIN_HOST` | `127.0.0.1` | Bind address |
| `REELRETAIN_PORT` | `5000` | Port |
| `REELRETAIN_MAX_CONTENT_LENGTH` | `104857600` | Maximum upload/request bytes |
| `REELRETAIN_ARTIFACT_DIR` | `models` | Trained model directory |
| `REELRETAIN_UPLOAD_DIR` | `uploads` | Temporary upload directory (files are removed after processing) |
| `REELRETAIN_AUDIO_TIMEOUT_SECONDS` | `15` | FFmpeg audio extraction timeout |
| `REELRETAIN_VIDEO_TIMEOUT_SECONDS` | `30` | OpenCV video decoding timeout |

Do not expose the development server directly to the public internet. Run behind a production WSGI server and add deployment-specific request/time limits for hosted use.

## API

### `GET /api/health`

Returns application and model readiness. A missing model is reported as `model_status: "unavailable"`; it does not prevent the health endpoint or server from starting.

### `GET /api/model-info`

Returns supported feature names, target definitions, synthetic-data disclosure, and model availability.

### `POST /api/analyze`

Send `multipart/form-data` with a `video` file and optional UTF-8 `caption` field:

```powershell
curl.exe -X POST http://127.0.0.1:5000/api/analyze `
  -F "video=@.\sample.mp4;type=video/mp4" `
  -F "caption=An example caption #reels"
```

Successful response fields include `features`, `estimates` (units, raw prediction, display-constrained estimate, and whether it was constrained), `warnings`, `request_id`, and `disclaimer`. Watched percentage and early drop-off are constrained to 0–100%; average watch time is constrained to 0–video duration. The response clearly labels them as synthetic-model estimates, not measurements. Without trained artifacts, the endpoint returns HTTP 503 with recovery instructions. Validation and processing errors use a consistent JSON error shape and request ID.

## Dataset schema and feature contract

Required target columns:

- `watched_percentage_synthetic`
- `average_watch_time_sec_synthetic`
- `early_dropoff_rate_pct_synthetic`

Optional identifier: `record_id` (always excluded). The supplied 5,000-row CSV has 20 columns: `record_id`, duration, scene changes, motion, speech rate, silence duration, caption/hashtag counts, hook time, caption synchronization, audio energy, face presence, niche, video format, posting hour/day, the three synthetic targets, and `is_synthetic`. Actual pandas dtypes and selected training features are recorded in `models/evaluation_report.json`. The model inputs use the seven dataset columns that the current extractor measures: `video_duration_sec`, `scene_changes`, `motion_intensity`, `silence_duration_sec`, `caption_word_count`, `hashtag_count`, and `audio_energy`. Target columns, `record_id`, and `is_synthetic` are never model inputs. Other source columns are explicitly excluded because they are metadata or are not measured by the current extractor. Unknown columns are rejected rather than silently passed to a model.

Unavailable audio values are represented as missing, surfaced in warnings, and median-imputed by the fitted training pipeline. Audio analysis requires FFmpeg. Speech rate is not guessed without transcription, and face presence, hook timing, topic niche, video style, posting time, and caption synchronization are not inferred. The selected dataset's synthetic distributions and feature semantics may not match real extraction. This feature contract does not eliminate the synthetic-to-real domain mismatch.

Video frames are resized to 64×64 grayscale and sampled at approximately one frame per second. A sampled normalized mean absolute grayscale difference of at least `0.30` counts as a scene-change candidate; motion intensity is the mean of those normalized differences. FFmpeg converts audio to mono 16 kHz float samples; energy is RMS, and silence duration is the total duration of 20 ms windows below `0.01` RMS (−40 dBFS). Audio and video decoding have configurable timeouts.

## Tests

```powershell
python -m pytest
```

Tests use small in-memory fixtures and generated media where supported; they do not call external services or create replacement training data.

## Troubleshooting and known limitations

- **Dataset missing:** Restore the supplied CSV at `data/reelretain_synthetic_training_dataset_5000.csv` or pass its path with `--dataset`. No replacement data is generated.
- **Model unavailable:** Run `python training\train_model.py` to train and save all three target pipelines, or set `REELRETAIN_ARTIFACT_DIR` to a trusted directory containing them.
- **Audio unavailable:** Install FFmpeg and add it to `PATH`; the API warns and leaves audio-derived values missing.
- **Codec or corrupt video:** Install codec support required by OpenCV/FFmpeg and retry with a valid supported video.
- **Real-world validity:** No real audience-retention labels or calibration data are supplied. Phase 2 must collect consented real video/retention pairs, define consistent feature semantics, retrain and evaluate on leakage-free real-world splits, and independently validate calibration and fairness before production use.

Training artifacts and the evaluation report are under `models/`; rerun training with the supplied CSV to regenerate them. Model performance and predictions remain experimental, not validated real-world retention estimates.
