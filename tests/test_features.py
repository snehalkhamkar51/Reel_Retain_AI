import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from reelretain.features import FeatureExtractionError, _audio_features, extract_video_features


class FakeCapture:
    def __init__(self, _path):
        self.index = 0
        self.frames = [
            np.zeros((32, 32, 3), dtype=np.uint8),
            np.zeros((32, 32, 3), dtype=np.uint8),
            np.full((32, 32, 3), 255, dtype=np.uint8),
        ]

    def isOpened(self):
        return True

    def get(self, prop):
        return {
            cv2.CAP_PROP_FPS: 1.0,
            cv2.CAP_PROP_FRAME_COUNT: 3.0,
            cv2.CAP_PROP_FRAME_WIDTH: 32.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 32.0,
        }[prop]

    def read(self):
        if self.index >= len(self.frames):
            return False, None
        frame = self.frames[self.index]
        self.index += 1
        return True, frame

    def release(self):
        pass


def test_extracts_metadata_scene_motion_and_caption(tmp_path, monkeypatch):
    path = tmp_path / "sample.mp4"
    path.write_bytes(b"not-decoded-by-mock")
    monkeypatch.setattr("reelretain.features.cv2.VideoCapture", FakeCapture)
    monkeypatch.setattr("reelretain.features.shutil.which", lambda _name: None)

    features, warnings = extract_video_features(path, "Hello #reels #AI")

    assert features["video_duration_sec"] == 3.0
    assert features["frame_rate_fps"] == 1.0
    assert features["scene_changes"] == 1
    assert features["motion_intensity"] > 0
    assert features["audio_energy"] is None
    assert features["caption_word_count"] == 3
    assert features["hashtag_count"] == 2
    assert any("FFmpeg was not found" in warning for warning in warnings)


def test_rejects_empty_corrupted_and_unsupported_video(tmp_path, monkeypatch):
    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    with pytest.raises(FeatureExtractionError, match="empty"):
        extract_video_features(empty)

    unsupported = tmp_path / "image.png"
    unsupported.write_bytes(b"data")
    with pytest.raises(FeatureExtractionError, match="extension"):
        extract_video_features(unsupported)

    corrupted = tmp_path / "corrupt.mp4"
    corrupted.write_bytes(b"broken")

    class ClosedCapture:
        def __init__(self, _path):
            pass

        def isOpened(self):
            return False

        def release(self):
            pass

    monkeypatch.setattr("reelretain.features.cv2.VideoCapture", ClosedCapture)
    with pytest.raises(FeatureExtractionError, match="could not be decoded"):
        extract_video_features(corrupted)


def test_no_audio_track_is_reported_as_unavailable(tmp_path, monkeypatch):
    path = tmp_path / "silent.mp4"
    path.write_bytes(b"video")
    monkeypatch.setattr("reelretain.features.shutil.which", lambda _name: "ffmpeg")
    monkeypatch.setattr(
        "reelretain.features.subprocess.run",
        lambda *_args, **_kwargs: type("Result", (), {"returncode": 1, "stdout": b""})(),
    )

    energy, silence, warnings = _audio_features(path, timeout_seconds=1)

    assert energy is None
    assert silence is None
    assert any("no decodable audio track" in warning for warning in warnings)


def test_video_decode_timeout_is_reported(tmp_path, monkeypatch):
    path = tmp_path / "slow.mp4"
    path.write_bytes(b"video")
    monkeypatch.setattr("reelretain.features.cv2.VideoCapture", FakeCapture)
    monkeypatch.setattr("reelretain.features.time.monotonic", iter([0.0, 2.0]).__next__)

    with pytest.raises(FeatureExtractionError, match="time limit"):
        extract_video_features(path, video_timeout_seconds=1)
