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

Specs checked Nov 2026 (third-party guides; the apps' own help pages don't publish all of them and the guides
partly disagree): all three take 1080x1920 9:16 H.264/AAC MP4 at 30 fps. Length caps: Shorts 3 min (since Oct
2024), Reels 90 s to 3 min depending on the account, TikTok 10 min or more. Viyo's clips are at most 60 s, so
they fit every app. Audio: YouTube recommends 48 kHz; guides for Reels split between 44.1 and 48 kHz.

Hook timings are an editorial choice (TikTok's feed is swiped fastest, so its hook text is shortest), not a
measured platform rule. Pacing is deliberately the same everywhere: there's no reliable evidence that one app
needs a different cut of the same moment, so the platforms differ in layout, hook, encoding and packaging.
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
        # Hook text on screen for the opening seconds, and where (pixels below the top safe edge).
        "hook_seconds": 2.2, "hook_offset": 0,
        # Upload-friendly encoding: 30 fps, a 2-second keyframe interval, a capped bitrate the app won't need to
        # squeeze hard, 44.1 kHz AAC.
        "export": {"fps": 30, "maxrate": "8M", "bufsize": "16M", "keyint": 60, "audio_rate": 44100},
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
        "hook_seconds": 2.6, "hook_offset": 0,
        # YouTube's recommended upload settings: 48 kHz audio, closed GOPs about half the frame rate.
        "export": {"fps": 30, "maxrate": "10M", "bufsize": "20M", "keyint": 15, "audio_rate": 48000},
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
        "hook_seconds": 2.4, "hook_offset": 30,
        "export": {"fps": 30, "maxrate": "8M", "bufsize": "16M", "keyint": 60, "audio_rate": 44100},
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
        return {"safe": dict(p["safe"]), "caption_size": p["caption_size"], "platform": name,
                "hook_seconds": p["hook_seconds"], "hook_offset": p["hook_offset"]}
    return {"safe": dict(UNIVERSAL["safe"]), "caption_size": None, "platform": "universal"}
