# ReelRetain AI

ReelRetain AI is a local prototype for extracting measurable features from short videos and estimating watched percentage, average watch time, and early drop-off rate.

> **Synthetic-data limitation:** The expected training dataset contains synthetic input features and synthetic target values. Model metrics and predictions are experimental and are not evidence of real-audience retention performance. Video-derived features may also differ in distribution and meaning from the synthetic training features. Do not use these estimates as measured facts or for publication decisions.

## Architecture

- `reelretain/features.py` extracts video metadata, sampled-frame scene changes and motion, optional FFmpeg audio energy/silence, and caption counts. Speech rate, hook timing, and caption synchronization are left unavailable unless a reliable measurement is implemented.
- `training/train_model.py` trains from the supplied local dataset path; `ml/` validates the CSV, fits a separate scikit-learn pipeline for each synthetic target, compares Linear Regression, Random Forest, and Gradient Boosting, and saves selected pipelines and a report.
- `reelretain/predict.py` loads trained artifacts and returns constrained experimental estimates.
- `reelretain/recommendations.py` builds topic-, caption-, and measured-feature-based edit suggestions without an external AI service.
- `reelretain/gemini_analysis.py` optionally sends a consented uploaded video to Google's supported Gen AI Python SDK for structured content understanding, validates the output and timestamps, and requests deletion of its temporary remote file.
- `reelretain/references.py` validates user-supplied YouTube, Instagram Reel, and TikTok links without fetching or scraping them.
- `reelretain/app.py` provides the Flask health, model-info, and video-analysis endpoints, including recommendation results and a logged fallback if generation fails.

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

The landing page, video analyzer, results dashboard, static assets, and JSON API are served by Flask at `http://127.0.0.1:5000` by default. No second frontend or CORS configuration is required. Gemini content understanding is optional and may incur Google API charges; all local extraction, synthetic model estimates, and deterministic suggestions remain available without a key. To enable Gemini, create a Google Gemini API key, copy `.env.example` to `.env`, set `REELRETAIN_GEMINI_API_KEY` in that ignored local file, and restart the app. The UI requires explicit consent for each uploaded video before sending it to Google. ReelRetain requests deletion of its temporary remote Gemini file after processing, but Google's processing and retention are subject to Google's policies. Never commit `.env` or paste the key into source code. Environment configuration:

| Variable | Default | Purpose |
|---|---:|---|
| `REELRETAIN_HOST` | `127.0.0.1` | Bind address |
| `REELRETAIN_PORT` | `5000` | Port |
| `REELRETAIN_MAX_CONTENT_LENGTH` | `104857600` | Maximum upload/request bytes |
| `REELRETAIN_ARTIFACT_DIR` | `models` | Trained model directory |
| `REELRETAIN_UPLOAD_DIR` | `uploads` | Temporary upload directory (files are removed after processing) |
| `REELRETAIN_AUDIO_TIMEOUT_SECONDS` | `15` | FFmpeg audio extraction timeout |
| `REELRETAIN_VIDEO_TIMEOUT_SECONDS` | `30` | OpenCV video decoding timeout |
| `REELRETAIN_GEMINI_API_KEY` | empty | Optional Google Gemini API key; required for external content understanding |
| `REELRETAIN_GEMINI_MODEL` | `gemini-3.8-flash` | Optional Gemini model used for video analysis |
| `REELRETAIN_GEMINI_TIMEOUT_SECONDS` | `120` | Gemini API and file-processing timeout |
| `REELRETAIN_GEMINI_POLL_INTERVAL_SECONDS` | `2` | Remote video processing poll interval |

Do not expose the development server directly to the public internet. Run behind a production WSGI server and add deployment-specific request/time limits for hosted use.

## API

### `GET /api/health`

Returns application and model readiness. A missing model is reported as `model_status: "unavailable"`; it does not prevent the health endpoint or server from starting.

### `GET /api/model-info`

Returns supported feature names, target definitions, synthetic-data disclosure, and model availability.

### `POST /api/analyze`

Send `multipart/form-data` with a `video` file, optional UTF-8 `caption` and `topic`, optional `reference_urls` (one allowlisted public URL per line, at most five), and `gemini_consent=true` only when the uploader has explicitly opted in. The topic can be a subject or niche and is limited to 120 characters; captions are limited to 2,200 characters. Omitting the consent field never sends the video to Google.

```powershell
curl.exe -X POST http://127.0.0.1:5000/api/analyze `
  -F "video=@.\sample.mp4;type=video/mp4" `
  -F "caption=An example caption #reels" `
  -F "topic=beginner home gardening"
```

To explicitly request external Gemini processing, add `-F "gemini_consent=true"` after informing the video owner. For user-supplied references, add `-F "reference_urls=https://youtu.be/VIDEO_ID"`; these URLs are not visited by ReelRetain.

Successful response fields include `features`, `estimates` (units, raw prediction, display-constrained estimate, and whether it was constrained), `recommendations` (`mode`, context, and prioritized suggestion items), `warnings`, `request_id`, and `disclaimer`. Suggestions use the supplied topic/caption and measured visual features; audio-specific guidance appears only when audio energy was extracted. Without a topic or caption, the response marks its advice as general. These are feature-based suggestions, not predictions of audience response. Watched percentage and early drop-off are constrained to 0–100%; average watch time is constrained to 0–video duration. The response clearly labels estimates as synthetic-model outputs, not measurements. Without trained artifacts, the endpoint returns HTTP 503 with recovery instructions. Validation and processing errors use a consistent JSON error shape and request ID.

The results page includes an Audience Attraction Lab with three opening hooks, visual/pacing guidance, a caption and call to action, overlay text, an ending suggestion, and audio guidance when audio was analyzed. Hooks and caption examples have copy buttons. The concise “Experimental estimates — synthetic training data” notice remains visible next to the estimates and can be expanded for the full limitation.

When Gemini is explicitly enabled, the report also includes a video description, topic, opening and structure observations, timestamped moments only when valid and within the measured duration, editing suggestions, caption/CTA options, a mini storyboard, audio notes when provided, and potential audience hypotheses. These are model-generated suggestions, not independently verified facts or actual audience demographics. Invalid/out-of-range timestamps are removed. Gemini does not supply or alter numeric retention estimates; those still come exclusively from the existing synthetic-data ML models. If Gemini is unavailable or fails, local predictions and deterministic suggestions continue to work.

The reference section accepts up to five HTTPS public-video URLs from YouTube, Instagram Reels, or TikTok. It does not call platform APIs or retrieve titles, creators, dates, views, likes, comments, or retention measurements. Each link is labeled user-provided and unverified; no metric comparison is claimed. Platform API access and reliable public retention data are not available in this prototype.

The integration uses the official [`google-genai` Python SDK](https://github.com/googleapis/python-genai), the [Gemini video understanding](https://ai.google.dev/gemini-api/docs/video-understanding) Files API upload and the [Interactions API structured-output](https://ai.google.dev/gemini-api/docs/structured-output) request. The selected default model is `gemini-3.8-flash`; it can be changed using `REELRETAIN_GEMINI_MODEL`. The report and UI distinguish locally measured properties, Gemini-generated analysis, experimental ML estimates, unverified user-supplied references, and audience hypotheses/general advice. A future real-retention model would require creator-authorized videos paired with actual retention analytics, documented consent and provenance, and leakage-resistant evaluation on representative real-world splits; that training/validation path is not implemented or claimed here.

## Dataset schema and feature contract

Required target columns:

- `watched_percentage_synthetic`
- `average_watch_time_sec_synthetic`
- `early_dropoff_rate_pct_synthetic`

Optional identifier: `record_id` (always excluded). The supplied 5,000-row CSV has 20 columns: `record_id`, duration, scene changes, motion, speech rate, silence duration, caption/hashtag counts, hook time, caption synchronization, audio energy, face presence, niche, video format, posting hour/day, the three synthetic targets, and `is_synthetic`. Actual pandas dtypes and selected training features are recorded in `models/evaluation_report.json`. The model inputs use the seven dataset columns that the current extractor measures: `video_duration_sec`, `scene_changes`, `motion_intensity`, `silence_duration_sec`, `caption_word_count`, `hashtag_count`, and `audio_energy`. Target columns, `record_id`, and `is_synthetic` are never model inputs. Other source columns are explicitly excluded because they are metadata or are not measured by the current extractor. Unknown columns are rejected rather than silently passed to a model.

Unavailable audio values are represented as missing, surfaced in warnings, and median-imputed by the fitted training pipeline. Audio analysis requires FFmpeg. Speech rate is not guessed without transcription, and face presence, hook timing, topic niche, video style, posting time, and caption synchronization are not inferred. The selected dataset's synthetic distributions and feature semantics may not match real extraction. This feature contract does not eliminate the synthetic-to-real domain mismatch.

Video frames are resized to 64×64 grayscale and sampled at approximately one frame per second. A sampled normalized mean absolute grayscale difference of at least `0.30` counts as a scene-change candidate; motion intensity is the mean of those normalized differences. FFmpeg converts audio to mono 16 kHz float samples; energy is RMS, and silence duration is the total duration of 20 ms windows below `0.01` RMS (−40 dBFS). Audio and video decoding have configurable timeouts.

The UI's review prompts are simple heuristics, not model conclusions: a measured silence duration at or above 20% of the video suggests reviewing those spans; fewer than two or more than twenty detected scene changes per minute suggests checking pacing; and zero supplied caption words suggests adding the caption for counts. These thresholds do not prove why a viewer might leave.

## Tests

```powershell
python -m pytest
```

Tests use small in-memory fixtures and generated media where supported; they do not call external services or create replacement training data. The Flask test client also checks that the landing page and frontend assets are served. The browser interface requires no frontend install step.

## Troubleshooting and known limitations

- **Dataset missing:** Restore the supplied CSV at `data/reelretain_synthetic_training_dataset_5000.csv` or pass its path with `--dataset`. No replacement data is generated.
- **Model unavailable:** Run `python training\train_model.py` to train and save all three target pipelines, or set `REELRETAIN_ARTIFACT_DIR` to a trusted directory containing them.
- **Audio unavailable:** Install FFmpeg and add it to `PATH`; the API warns and leaves audio-derived values missing.
- **Codec or corrupt video:** Install codec support required by OpenCV/FFmpeg and retry with a valid supported video.
- **Real-world validity:** No real audience-retention labels or calibration data are supplied. Phase 2 must collect consented real video/retention pairs, define consistent feature semantics, retrain and evaluate on leakage-free real-world splits, and independently validate calibration and fairness before production use.

Training artifacts and the evaluation report are under `models/`; rerun training with the supplied CSV to regenerate them. Model performance and predictions remain experimental, not validated real-world retention estimates.
