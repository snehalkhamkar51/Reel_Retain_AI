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
from reelretain.predict import DISCLAIMER, ModelUnavailableError, Predictor

mimetypes.add_type("text/javascript", ".js", strict=True)


def create_app(config_overrides: dict | None = None) -> Flask:
    app = Flask(__name__, static_folder="static", template_folder="templates")
    app.config.update(get_config())
    if config_overrides:
        app.config.update(config_overrides)
    app.config["MAX_CONTENT_LENGTH"] = int(app.config["MAX_CONTENT_LENGTH"])
    app.extensions["predictor"] = Predictor(app.config["ARTIFACT_DIR"])

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
        predictor = app.extensions["predictor"]
        if not predictor.available:
            return error_response(
                503, "model_unavailable",
                "Trained models are unavailable. Train them with the authentic dataset before analyzing videos.",
                request.request_id,
            )

        temp_path = None
        try:
            with _temporary_upload(upload, suffix, app.config["UPLOAD_DIR"]) as saved_path:
                temp_path = saved_path
                features, warnings = extract_video_features(
                    temp_path, request.form.get("caption", ""),
                    int(app.config["AUDIO_TIMEOUT_SECONDS"]),
                    int(app.config["VIDEO_TIMEOUT_SECONDS"]),
                )
            estimates = predictor.predict(features)
            return jsonify({
                "ok": True,
                "request_id": request.request_id,
                "features": features,
                "estimates": estimates,
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
