"""What each short-video platform needs, in one place.

Every clip is rendered once as a caption-free master; a platform version
burns captions placed for THAT platform's overlays onto it (see
repurpose.platform_clip) and comes with packaging written for it. All three
are 9:16, 1080x1920, H.264/AAC, at most 60 s, so one master serves them.

Safe areas are the parts of the screen each app covers with its own buttons,
caption and title, in pixels of a 1080x1920 frame. They are approximate
(the apps change their layouts, and differ between phones); the "universal"
layout, used for the default clip, stays clear of all three at once.
Re-check against the apps' current layouts when they change.
"""
from __future__ import annotations

PLATFORMS = {
    "tiktok": {
        "label": "TikTok",
        # Top: Following/For You tabs. Right: like/comment/share rail. Bottom: username, caption, sound.
        "safe": {"top": 250, "bottom": 430, "left": 70, "right": 160},
        "caption_size": 60,
        "max_seconds": 60,
        "title_chars": 0,           # TikTok has no separate title
        "caption_chars": 150,       # what shows before "more" (the hard limit is far higher)
        "hashtags": (3, 5),
        "notes": "Caption and hashtags in one text box; the first line is what people see.",
    },
    "shorts": {
        "label": "YouTube Shorts",
        # Top: search/menu. Right: like/dislike/comment rail. Bottom: title, channel, subscribe.
        "safe": {"top": 220, "bottom": 360, "left": 70, "right": 140},
        "caption_size": 62,
        "max_seconds": 60,
        "title_chars": 100,         # YouTube's title limit
        "caption_chars": 300,
        "hashtags": (2, 3),         # YouTube shows the first three above the title
        "notes": "A title (up to 100 characters) plus a description; the first hashtags show above the title.",
    },
    "reels": {
        "label": "Instagram Reels",
        # Top: Reels header and camera. Right: like/comment/share rail. Bottom: account, caption, audio.
        "safe": {"top": 240, "bottom": 420, "left": 70, "right": 150},
        "caption_size": 60,
        "max_seconds": 60,
        "title_chars": 0,
        "caption_chars": 125,       # what shows before "more"
        "hashtags": (3, 5),
        "notes": "One caption; a few specific hashtags work better than many.",
    },
}

UNIVERSAL = {
    "safe": {k: max(p["safe"][k] for p in PLATFORMS.values()) for k in ("top", "bottom", "left", "right")},
    "caption_size": None,           # the caption style's own size
}


def layout(name: str | None) -> dict:
    """Caption layout for a platform, or the universal one."""
    if name in PLATFORMS:
        p = PLATFORMS[name]
        return {"safe": dict(p["safe"]), "caption_size": p["caption_size"], "platform": name}
    return {"safe": dict(UNIVERSAL["safe"]), "caption_size": None, "platform": "universal"}
