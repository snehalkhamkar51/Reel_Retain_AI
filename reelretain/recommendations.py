import math
import re


def generate_recommendations(features: dict, caption: str = "", topic: str = "") -> dict:
    focus = _clean_context(topic) or _caption_focus(caption) or "your video"
    personalized = bool(_clean_context(topic) or _clean_context(caption))
    items = []

    hooks = (
        f"{focus}: start with the one detail viewers need to know.",
        f"A quick {focus} breakdown—here's the key step.",
        f"Working on {focus}? Try this before your next step.",
    )
    for index, hook in enumerate(hooks, start=1):
        items.append(_item(
            "Opening hook",
            "High" if index == 1 else "Medium",
            f"Open with a clear, topic-led promise for {focus}.",
            "A specific opening helps viewers understand the subject before deciding whether to keep watching.",
            f"Put this line in the first spoken sentence or opening frame, then deliver on it immediately.",
            hook,
            hook,
        ))

    duration = _number(features.get("video_duration_sec"))
    scene_changes = _number(features.get("scene_changes"))
    motion = _number(features.get("motion_intensity"))
    scene_rate = scene_changes * 60 / duration if scene_changes is not None and duration and duration > 0 else None
    pacing_detail = []
    if scene_changes is not None:
        pacing_detail.append(f"{scene_changes:g} sampled scene-change candidates")
    if scene_rate is not None:
        pacing_detail.append(f"about {scene_rate:.1f} per minute")
    if motion is not None:
        pacing_detail.append(f"motion intensity {motion:.3f}")
    measured = ", ".join(pacing_detail) if pacing_detail else "limited visual measurements"
    if scene_rate is not None and scene_rate < 2:
        pacing_action = "Review the sampled intervals and consider adding a purposeful cut, crop, or close-up where it reinforces the point."
    elif scene_rate is not None and scene_rate > 20:
        pacing_action = "Review the fast-changing sampled intervals and hold the most important shot long enough for the point to land."
    else:
        pacing_action = "Keep each visual change tied to a point; review the edit once without audio to check that the story remains easy to follow."
    items.append(_item(
        "Pacing and visuals",
        "Medium",
        f"Use the measured visual signals as an edit-review cue: {measured}.",
        "Scene-change candidates and motion describe sampled frames, not whether the pace is good or how viewers respond.",
        pacing_action,
        f"For {focus}, pair each key step with a relevant shot or on-screen detail.",
    ))

    supplied_caption = _clean_context(caption)
    caption_base = supplied_caption[:150].rstrip(" .,;:!?") if supplied_caption else f"A quick note about {focus}"
    caption_example = f"{caption_base}. Try it, then tell me which part helped you most."
    items.append(_item(
        "Caption and CTA",
        "Medium",
        "Make the caption state the value of the reel, then invite one easy response.",
        "A clear takeaway and a focused call to action give interested viewers a useful next step without promising a particular result.",
        "Keep the main idea in the first sentence and end with one relevant question or action.",
        caption_example,
        caption_example,
    ))

    overlay = f"{focus}: the key step"
    items.append(_item(
        "On-screen text",
        "Low",
        f"Add a short label that orients viewers to {focus}.",
        "Concise text can make the subject clear when a viewer is watching with sound off.",
        "Show one readable phrase near the opening, keep it clear of interface controls, and remove it when it is no longer useful.",
        overlay,
    ))

    audio_energy = _number(features.get("audio_energy"))
    silence = _number(features.get("silence_duration_sec"))
    if audio_energy is not None:
        if duration is not None and duration > 0 and silence is not None:
            audio_detail = f"measured RMS energy {audio_energy:.3f} and {silence:.1f} seconds of thresholded silence"
            audio_action = (
                f"Listen through the measured quiet spans and trim only pauses that are unintended; "
                f"check that speech and music remain comfortably balanced."
            )
        else:
            audio_detail = f"measured RMS energy {audio_energy:.3f}"
            audio_action = "Listen on headphones and phone speakers; adjust speech and background audio by ear rather than treating RMS as a quality score."
        items.append(_item(
            "Audio",
            "Medium",
            f"Review the audio using the available measurement: {audio_detail}.",
            "Energy and thresholded silence do not measure clarity, loudness balance, music rights, or audience preference.",
            audio_action,
            f"For {focus}, let the key spoken instruction remain easy to hear.",
        ))

    items.append(_item(
        "Ending",
        "Low",
        f"Close {focus} with a takeaway that completes the opening promise.",
        "A concise payoff makes the reel feel complete and gives viewers something practical to remember.",
        "Use the final shot to summarize one useful point, show the finished result, or name a next step.",
        f"“The key takeaway for {focus}: [one useful, specific point].”",
    ))
    return {
        "mode": "personalized" if personalized else "general",
        "context": (
            "Personalized with your supplied topic or caption and the available measured features."
            if personalized
            else "General advice: no topic or caption was supplied; visual guidance uses the available measured features."
        ),
        "items": items,
    }


def fallback_recommendations(features: dict | None = None) -> dict:
    items = [
        _item("Opening hook", "High", "Start with the main subject.", "A clear opening sets context.", "Name the topic in the first sentence.", "“Here’s the key thing to know.”", "“Here’s the key thing to know.”"),
        _item("Opening hook", "Medium", "Lead with a useful question.", "A focused question helps viewers recognize the subject.", "Ask one question the reel answers.", "“Want to understand this better?”", "“Want to understand this better?”"),
        _item("Opening hook", "Medium", "Preview the takeaway.", "A clear preview makes the purpose easy to follow.", "State the practical result without overstating it.", "“In this reel, I’ll show one useful step.”", "“In this reel, I’ll show one useful step.”"),
        _item("Pacing and visuals", "Medium", "Review the visual sequence.", "Frame sampling cannot determine whether pacing feels right.", "Check that each cut supports the point being made."),
        _item("Caption and CTA", "Medium", "Pair a concise takeaway with one invitation.", "A focused next step is easier to act on.", "Add one relevant question or action.", "“What would you try first? Share your take.”", "“What would you try first? Share your take.”"),
        _item("On-screen text", "Low", "Add a short subject label.", "Text can orient viewers watching without sound.", "Use one readable phrase and keep it clear of interface controls.", "“The key step”"),
        _item("Ending", "Low", "Finish with a useful takeaway.", "A clear close completes the idea.", "Summarize the main point or show the result."),
    ]
    audio_energy = _number(features.get("audio_energy")) if isinstance(features, dict) else None
    if audio_energy is not None:
        items.append(_item(
            "Audio",
            "Medium",
            f"Review the audio using the measured RMS energy of {audio_energy:.3f}.",
            "Energy does not measure clarity, loudness balance, or audience preference.",
            "Listen on headphones and phone speakers; adjust speech and background audio by ear.",
        ))
    return {
        "mode": "general",
        "context": "General advice is shown because personalized recommendations could not be prepared.",
        "items": items,
    }


def _item(category, priority, what, why, action, example=None, copy_text=None):
    return {
        "category": category,
        "priority": priority,
        "what": what,
        "why": why,
        "action": action,
        "example": example,
        "copy_text": copy_text,
    }


def _number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _clean_context(value):
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", value).strip(" \t\r\n.,;:!?")[:80]


def _caption_focus(caption):
    clean = _clean_context(caption)
    return re.split(r"[.!?\n]", clean, maxsplit=1)[0][:60].strip() if clean else ""
