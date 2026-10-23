from urllib.parse import parse_qs, urlsplit


_ALLOWED_HOSTS = {
    "youtu.be": "YouTube",
    "youtube.com": "YouTube",
    "www.youtube.com": "YouTube",
    "m.youtube.com": "YouTube",
    "instagram.com": "Instagram",
    "www.instagram.com": "Instagram",
    "tiktok.com": "TikTok",
    "www.tiktok.com": "TikTok",
}


class InvalidReferenceError(ValueError):
    """A supplied reference is not an accepted public video URL."""


def parse_user_references(value: str) -> list[dict]:
    if not isinstance(value, str):
        raise InvalidReferenceError("Reference URLs must be supplied as text.")
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    if len(lines) > 5:
        raise InvalidReferenceError("Enter no more than five public reference URLs.")

    references = []
    for url in lines:
        platform = _platform_for_url(url)
        references.append({
            "platform": platform,
            "url": url,
            "title": None,
            "creator": None,
            "published_date": None,
            "duration_seconds": None,
            "public_metrics": None,
            "metrics_collected_at": None,
            "similarities": None,
            "differences": None,
            "techniques_to_test": None,
            "verified": False,
            "evidence_source": "user_provided_url",
            "notice": (
                "URL supplied by the user; the application did not fetch or verify the video, "
                "metadata, metrics, or similarity."
            ),
        })
    return references


def _platform_for_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise InvalidReferenceError("A reference URL is malformed.") from exc
    host = (parsed.hostname or "").lower().rstrip(".")
    platform = _ALLOWED_HOSTS.get(host)
    if (
        parsed.scheme != "https"
        or not platform
        or parsed.username is not None
        or parsed.password is not None
        or port not in (None, 443)
        or not _looks_like_video_url(platform, host, parsed.path, parsed.query)
    ):
        raise InvalidReferenceError(
            "Use an HTTPS YouTube, Instagram Reel, or TikTok video URL."
        )
    return platform


def _looks_like_video_url(platform: str, host: str, path: str, query: str) -> bool:
    segments = [segment for segment in path.split("/") if segment]
    if platform == "YouTube":
        if host == "youtu.be":
            return len(segments) == 1
        if segments and segments[0] == "watch":
            return bool(parse_qs(query).get("v", [""])[0])
        return len(segments) == 2 and segments[0] in {"shorts", "embed", "live"}
    if platform == "Instagram":
        return len(segments) >= 2 and segments[0] in {"reel", "reels", "p"}
    return len(segments) >= 3 and segments[0].startswith("@") and segments[1] == "video"
