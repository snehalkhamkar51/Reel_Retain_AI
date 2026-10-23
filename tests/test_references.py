import pytest

from reelretain.references import InvalidReferenceError, parse_user_references


@pytest.mark.parametrize(
    ("url", "platform"),
    [
        ("https://www.youtube.com/watch?v=video123", "YouTube"),
        ("https://youtu.be/video123", "YouTube"),
        ("https://www.instagram.com/reel/abc123/", "Instagram"),
        ("https://www.tiktok.com/@creator/video/123456", "TikTok"),
    ],
)
def test_supported_user_reference_urls_have_unverified_provenance(url, platform):
    reference = parse_user_references(url)[0]

    assert reference["platform"] == platform
    assert reference["verified"] is False
    assert reference["evidence_source"] == "user_provided_url"
    assert reference["public_metrics"] is None
    assert reference["metrics_collected_at"] is None
    assert reference["similarities"] is None


def test_empty_reference_input_returns_explicit_empty_result():
    assert parse_user_references(" \r\n ") == []


@pytest.mark.parametrize(
    "url",
    [
        "http://www.youtube.com/watch?v=x",
        "https://example.com/video/1",
        "https://youtube.com.evil.example/watch?v=x",
        "https://user@youtube.com/watch?v=x",
        "https://www.instagram.com/p/",
        "https://www.tiktok.com/@creator/not-a-video/1",
        "https://www.youtube.com/not-a-video/path",
        "https://youtube.com:abc/watch?v=x",
    ],
)
def test_untrusted_or_malformed_reference_urls_are_rejected(url):
    with pytest.raises(InvalidReferenceError):
        parse_user_references(url)


def test_reference_count_is_bounded():
    urls = "\n".join(f"https://youtu.be/video{i}" for i in range(6))

    with pytest.raises(InvalidReferenceError, match="five"):
        parse_user_references(urls)
