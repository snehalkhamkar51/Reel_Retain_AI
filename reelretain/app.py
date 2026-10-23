import logging
import mimetypes
import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from reelretain.config import get_config
from reelretain.features import (
    FeatureExtractionError,
    SUPPORTED_EXTENSIONS,
    extract_video_features,
)
from reelretain.gemini_analysis import (
    DEFAULT_GEMINI_MODEL,
    GeminiAnalysisError,
    GeminiVideoAnalyzer,
)
from reelretain.predict import DISCLAIMER, ModelUnavailableError, Predictor
from reelretain.recommendations import fallback_recommendations, generate_recommendations
from reelretain.references import InvalidReferenceError, parse_user_references

mimetypes.add_type("text/javascript", ".js", strict=True)


def create_app(config_overrides: dict | None = None) -> Flask:
    app = Flask(__name__, static_folder="static", template_folder="templates")
    app.config.update(get_config())
    if config_overrides:
        app.config.update(config_overrides)
    app.config["MAX_CONTENT_LENGTH"] = int(app.config["MAX_CONTENT_LENGTH"])
    app.config["GEMINI_MODEL"] = str(
        app.config.get("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
    ).strip()
    app.extensions["predictor"] = Predictor(app.config["ARTIFACT_DIR"])
    app.extensions["gemini_analyzer"] = GeminiVideoAnalyzer(
        app.config["GEMINI_API_KEY"],
        app.config["GEMINI_MODEL"],
        int(app.config["GEMINI_TIMEOUT_SECONDS"]),
        int(app.config["GEMINI_POLL_INTERVAL_SECONDS"]),
    )

    def error_response(status: int, code: str, message: str, request_id: str):
        return jsonify({
            "ok": False,
            "error": {"code": code, "message": message},
            "request_id": request_id,
            "disclaimer": DISCLAIMER,
        }), status

    @app.errorhandler(RequestEntityTooLarge)
    def handle_too_large(_error):
        return error_response(
            413, "upload_too_large", "The upload exceeds the configured size limit.",
            getattr(request, "request_id", uuid.uuid4().hex),
        )

    @app.before_request
    def assign_request_id():
        request.request_id = uuid.uuid4().hex

    @app.get("/")
    def home():
        return render_template("index.html")

    @app.get("/api/health")
    def health():
        predictor = app.extensions["predictor"]
        return jsonify({
            "ok": True,
            "status": "healthy",
            "model_status": "ready" if predictor.available else "unavailable",
            "gemini_available": app.extensions["gemini_analyzer"].available,
            "disclaimer": DISCLAIMER,
            "request_id": request.request_id,
        })

    @app.get("/api/model-info")
    def model_info():
        predictor = app.extensions["predictor"]
        return jsonify({
            "ok": True,
            "model": predictor.info(),
            "upload": {
                "max_bytes": app.config["MAX_CONTENT_LENGTH"],
                "extensions": sorted(SUPPORTED_EXTENSIONS),
            },
            "content_analysis": {
                "provider": "Google Gemini",
                "available": app.extensions["gemini_analyzer"].available,
                "model": app.config["GEMINI_MODEL"] or DEFAULT_GEMINI_MODEL,
                "requires_explicit_consent": True,
            },
            "reference_retrieval": {
                "mode": "user-provided URLs only",
                "automatically_retrieved": False,
                "verified_metrics_available": False,
            },
            "request_id": request.request_id,
        })

    @app.post("/api/analyze")
    def analyze():
        upload = request.files.get("video")
        if upload is None or not upload.filename:
            return error_response(400, "missing_video", "A video file is required.", request.request_id)
        safe_name = secure_filename(upload.filename)
        suffix = Path(safe_name).suffix.lower()
        if not safe_name or suffix not in SUPPORTED_EXTENSIONS:
            return error_response(400, "unsupported_file", "The video extension is not supported.", request.request_id)
        content_type = (upload.mimetype or "").lower()
        if content_type and content_type != "application/octet-stream" and not content_type.startswith("video/"):
            return error_response(400, "invalid_content_type", "The uploaded content type must be a video.", request.request_id)
        if request.content_length is not None and request.content_length > app.config["MAX_CONTENT_LENGTH"]:
            return error_response(413, "upload_too_large", "The upload exceeds the configured size limit.", request.request_id)
        caption = request.form.get("caption", "")
        topic = request.form.get("topic", "")
        raw_reference_urls = request.form.get("reference_urls", "")
        if len(caption) > 2200 or len(topic) > 120:
            return error_response(
                400, "invalid_context",
                "Caption must be 2,200 characters or fewer and topic must be 120 characters or fewer.",
                request.request_id,
            )
        if len(raw_reference_urls) > 12000:
            return error_response(400, "invalid_references", "Reference URL input is too long.", request.request_id)
        try:
            references = parse_user_references(raw_reference_urls)
        except InvalidReferenceError as exc:
            return error_response(400, "invalid_references", str(exc), request.request_id)
        gemini_requested = request.form.get("gemini_consent", "").lower() in {"true", "yes", "1"}
        predictor = app.extensions["predictor"]
        if not predictor.available:
            return error_response(
                503, "model_unavailable",
                "Trained models are unavailable. Train them with the authentic dataset before analyzing videos.",
                request.request_id,
            )

        temp_path = None
        gemini_result = {
            "status": "not_requested",
            "model": app.config["GEMINI_MODEL"],
            "analysis": None,
            "message": "Gemini video analysis was not requested.",
        }
        try:
            with _temporary_upload(upload, suffix, app.config["UPLOAD_DIR"]) as saved_path:
                temp_path = saved_path
                features, warnings = extract_video_features(
                    temp_path, caption,
                    int(app.config["AUDIO_TIMEOUT_SECONDS"]),
                    int(app.config["VIDEO_TIMEOUT_SECONDS"]),
                )
                analyzer = app.extensions["gemini_analyzer"]
                if gemini_requested and not analyzer.available:
                    gemini_result.update({
                        "status": "unavailable",
                        "message": "Gemini is not configured; local analysis and recommendations are available.",
                    })
                elif gemini_requested:
                    try:
                        analysis = analyzer.analyze(saved_path, features, caption, topic)
                        gemini_result.update({
                            "status": "complete",
                            "analysis": analysis,
                            "message": "Gemini content analysis completed.",
                        })
                    except GeminiAnalysisError as exc:
                        logging.warning(
                            "Gemini analysis failed for request %s (%s)",
                            request.request_id,
                            type(exc).__name__,
                        )
                        gemini_result.update({
                            "status": "error",
                            "message": "Gemini content analysis is unavailable; local analysis remains available.",
                        })
                        warnings.append("Gemini content analysis could not be completed; local recommendations are shown.")
            estimates = predictor.predict(features)
            try:
                recommendations = generate_recommendations(features, caption, topic)
            except Exception:
                logging.exception("Recommendation generation failed for request %s", request.request_id)
                recommendations = fallback_recommendations(features)
                warnings.append("Personalized recommendations could not be prepared; general advice is shown.")
            return jsonify({
                "ok": True,
                "request_id": request.request_id,
                "features": features,
                "estimates": estimates,
                "recommendations": recommendations,
                "content_analysis": gemini_result,
                "audience_hypotheses": (
                    gemini_result["analysis"]["potential_audiences"]
                    if gemini_result["analysis"] else []
                ),
                "references": references,
                "methodology": {
                    "measured_properties": "Extracted locally from the uploaded video and optional supplied caption.",
                    "ai_content_analysis": (
                        "Generated by Google Gemini only after explicit consent; timestamps are checked against video duration."
                    ),
                    "experimental_predictions": "Produced by the existing ML pipelines trained and evaluated on synthetic data.",
                    "verified_external_metrics": "None. Reference URLs are user-provided and are not fetched or verified.",
                    "audience_hypotheses": "Potential audience hypotheses from Gemini are not actual viewer demographics.",
                },
                "warnings": warnings,
                "disclaimer": DISCLAIMER,
            })
        except FeatureExtractionError as exc:
            return error_response(422, "video_processing_failed", str(exc), request.request_id)
        except ValueError as exc:
            return error_response(422, "feature_contract_mismatch", str(exc), request.request_id)
        except ModelUnavailableError:
            return error_response(503, "model_unavailable", "Trained models are unavailable.", request.request_id)
        except OSError:
            logging.exception("Video processing failed for request %s", request.request_id)
            return error_response(500, "processing_failed", "The video could not be processed.", request.request_id)
        finally:
            if temp_path is not None:
                try:
                    Path(temp_path).unlink(missing_ok=True)
                except OSError:
                    logging.exception("Could not remove temporary upload for request %s", request.request_id)

    return app


def _temporary_upload(upload, suffix: str, upload_dir: str | Path):
    import tempfile
    from contextlib import contextmanager

    @contextmanager
    def saved():
        directory = Path(upload_dir)
        directory.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            prefix="reelretain-", suffix=suffix, dir=directory, delete=False
        )
        path = Path(handle.name)
        handle.close()
        try:
            upload.save(path)
            if path.stat().st_size == 0:
                raise FeatureExtractionError("The uploaded video is empty.")
            yield path
        finally:
            path.unlink(missing_ok=True)
    return saved()
