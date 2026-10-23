import math
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np


FEATURE_COLUMNS = (
    "video_duration_sec",
    "frame_rate_fps",
    "width_px",
    "height_px",
    "scene_change_count",
    "motion_intensity",
    "audio_energy_rms",
    "silence_duration_sec",
    "speech_rate_wpm",
    "caption_word_count",
    "hashtag_count",
    "hook_timing_sec",
    "caption_sync_score",
    "orientation",
)

TARGET_COLUMNS = (
    "watched_percentage_synthetic",
    "average_watch_time_sec_synthetic",
    "early_dropoff_rate_pct_synthetic",
)

SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".m4v", ".webm", ".avi"}


class FeatureExtractionError(ValueError):
    """A video could not be safely decoded or measured."""


def _audio_features(path: Path, timeout_seconds: int) -> tuple[float | None, float | None, list[str]]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return None, None, ["Audio features unavailable: FFmpeg was not found on PATH."]
    command = [
        ffmpeg, "-nostdin", "-v", "error", "-i", str(path),
        "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000",
        "-f", "f32le", "pipe:1",
    ]
    try:
        result = subprocess.run(
            command, capture_output=True, timeout=timeout_seconds, check=False
        )
    except subprocess.TimeoutExpired:
        return None, None, ["Audio features unavailable: FFmpeg exceeded its time limit."]
    except OSError:
        return None, None, ["Audio features unavailable: FFmpeg could not be started."]
    if result.returncode != 0:
        return None, None, ["Audio features unavailable: the video has no decodable audio track."]
    samples = np.frombuffer(result.stdout, dtype="<f4")
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        return None, None, ["Audio features unavailable: the decoded audio track is empty."]
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    # Silence is the fraction of 20 ms windows below -40 dBFS.
    window_size = 320
    windows = samples[: samples.size - samples.size % window_size].reshape(-1, window_size)
    if windows.size == 0:
        return rms, None, ["Silence duration unavailable: audio was shorter than one analysis window."]
    window_rms = np.sqrt(np.mean(np.square(windows, dtype=np.float64), axis=1))
    silence_duration = float(np.count_nonzero(window_rms < 0.01) * 0.02)
    return rms, silence_duration, []


def extract_video_features(
    path: str | Path,
    caption: str = "",
    audio_timeout_seconds: int = 15,
    video_timeout_seconds: int = 30,
) -> tuple[dict, list[str]]:
    video_path = Path(path)
    if not video_path.is_file() or video_path.stat().st_size == 0:
        raise FeatureExtractionError("The uploaded video is empty or unavailable.")
    if video_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise FeatureExtractionError("The uploaded file extension is not supported.")

    try:
        capture = cv2.VideoCapture(str(video_path))
    except cv2.error as exc:
        raise FeatureExtractionError("The video could not be decoded; check its codec and file integrity.") from exc
    if not capture.isOpened():
        capture.release()
        raise FeatureExtractionError("The video could not be decoded; check its codec and file integrity.")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not math.isfinite(fps) or fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
            raise FeatureExtractionError("The video metadata is invalid or incomplete.")
        duration = frame_count / fps
        sample_stride = max(1, int(round(fps)))
        sampled_frames = []
        frame_index = 0
        started_at = time.monotonic()
        while len(sampled_frames) < 600:
            if time.monotonic() - started_at > video_timeout_seconds:
                raise FeatureExtractionError("Video decoding exceeded its configured time limit.")
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % sample_stride == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                sampled_frames.append(cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA))
            frame_index += 1
        if not sampled_frames:
            raise FeatureExtractionError("No video frames could be decoded.")
    except cv2.error as exc:
        raise FeatureExtractionError("The video could not be decoded; check its codec and file integrity.") from exc
    finally:
        capture.release()

    differences = [
        float(np.mean(cv2.absdiff(previous, current)) / 255.0)
        for previous, current in zip(sampled_frames, sampled_frames[1:])
    ]
    # Frames are sampled at approximately 1 fps; a normalized mean absolute
    # grayscale change of 0.30 or greater counts as a scene-change candidate.
    scene_changes = sum(change >= 0.30 for change in differences)
    motion = float(np.mean(differences)) if differences else 0.0
    audio_energy, silence_duration, warnings = _audio_features(video_path, audio_timeout_seconds)
    words = caption.split()
    features = {
        "video_duration_sec": float(duration),
        "frame_rate_fps": fps,
        "width_px": width,
        "height_px": height,
        "scene_change_count": int(scene_changes),
        "motion_intensity": motion,
        "audio_energy_rms": audio_energy,
        "silence_duration_sec": silence_duration,
        "speech_rate_wpm": None,
        "caption_word_count": len(words),
        "hashtag_count": sum(token.startswith("#") and len(token) > 1 for token in words),
        "hook_timing_sec": None,
        "caption_sync_score": None,
        "orientation": "landscape" if width > height else "portrait" if height > width else "square",
    }
    unavailable = [
        key for key, value in features.items()
        if value is None and key in {"audio_energy_rms", "silence_duration_sec", "speech_rate_wpm", "hook_timing_sec", "caption_sync_score"}
    ]
    if "speech_rate_wpm" in unavailable:
        warnings.append("Speech rate unavailable: no speech recognition or transcription is performed.")
    if "hook_timing_sec" in unavailable:
        warnings.append("Hook timing unavailable: it is not inferred from visual content.")
    if "caption_sync_score" in unavailable:
        warnings.append("Caption synchronization unavailable: no time-aligned captions were provided.")
    return features, warnings
