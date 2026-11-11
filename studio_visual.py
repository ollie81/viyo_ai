"""What the assistant can actually SEE for one message.

When the creator asks about how things look ("why does Maya look different",
"check the outfits", "is the room the same"), the chat attaches a small set of
real images from this production: the scene pictures that matter, the scene
before for comparison, the characters' reference pictures, one frame from a
scene video when there is no picture, and place/object pictures. Every image
is resized (longest side MAX_SIDE, JPEG) and the set is capped (MAX_IMAGES),
so a message never sends whole videos or huge files.

Only paths from the creator's own project rows are read (the caller has
already checked ownership). Anything that can't be loaded is listed as
unavailable, so the assistant never claims to have seen it.
"""
from __future__ import annotations

import io
import os
import re
import subprocess
import tempfile
from collections import OrderedDict
from typing import Optional

MAX_IMAGES = int(os.environ.get("STUDIO_CHAT_MAX_IMAGES", "8"))
MAX_SIDE = 768
MAX_VIDEO_BYTES = 40 * 1024 * 1024
TOKENS_PER_IMAGE = 560          # a 768px image is about two tiles for Gemini; a safe estimate

_VISUAL = re.compile(r"\b(look(s|ing)?( different| the same| like)?|appearance|face|hair|hairstyle|outfit|wear(s|ing)?|"
                     r"dress|clothes|costume|shirt|jacket|coat|picture|image|photo|frame|visual(ly)?|see|shot|"
                     r"background|room|place|location|prop|bag|letter|phone|consistent|consistency|match(es)?|"
                     r"same person|changed|compare|comparison|different|differ|inspect|check the (pictures|images|scenes))\b",
                     re.I)
_ALL = re.compile(r"\b(all|every|entire|whole) (the )?(scenes?|episode|film|movie|production|pictures)\b", re.I)

_frames: "OrderedDict[str, bytes]" = OrderedDict()     # video frame cache (path -> jpeg), small


def wants_visual(message: str) -> bool:
    return bool(_VISUAL.search(message or ""))


def shrink(data: bytes) -> Optional[bytes]:
    """JPEG, longest side MAX_SIDE. None if it isn't a readable image."""
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(data))
        img = img.convert("RGB")
        img.thumbnail((MAX_SIDE, MAX_SIDE))
        out = io.BytesIO()
        img.save(out, "JPEG", quality=80)
        return out.getvalue()
    except Exception:
        return None


def video_frame(files, path: str) -> Optional[bytes]:
    """One frame from the middle of a scene video (cached)."""
    if path in _frames:
        _frames.move_to_end(path)
        return _frames[path]
    try:
        data = files.get(path)
        if not data or len(data) > MAX_VIDEO_BYTES:
            return None
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = os.path.join(tmp, "v.mp4"), os.path.join(tmp, "f.jpg")
            with open(src, "wb") as f:
                f.write(data)
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", "1", "-i", src, "-frames:v", "1", dst],
                           check=True, timeout=30, capture_output=True)
            with open(dst, "rb") as f:
                frame = shrink(f.read())
    except Exception:
        return None
    if frame:
        _frames[path] = frame
        while len(_frames) > 32:
            _frames.popitem(last=False)
    return frame


def _scene_numbers_for(names: list[str], scenes: list[dict]) -> list[int]:
    out = []
    for i, s in enumerate(scenes, start=1):
        text = " ".join([s.get("visual") or "", " ".join(d.get("character") or "" for d in s.get("dialogue") or []),
                         " ".join(c.get("name", "") for c in (s.get("continuity") or {}).get("characters") or [])]).lower()
        if any(n.split(" ")[0].lower() in text for n in names):
            out.append(i)
    return out


def gather(files, scenes: list[dict], cast: list[dict], elements: list[dict], focus: dict,
           message: str, where: str = "", owner: Optional[str] = None) -> tuple[list[tuple[str, bytes]], list[str], list[str]]:
    """(images [(label, jpeg)], attached labels, unavailable notes) for one message.
    Labels name the production/episode (`where`) and the scene number."""
    tag = f"{where} · " if where else ""
    import studio_identity as ident
    wanted: list[tuple[str, str, str]] = []      # (label, kind, path or scene id)
    n = len(scenes)
    nums = [x for x in focus.get("scenes") or [] if 1 <= x <= n]
    names = [c for c in focus.get("characters") or []]
    if not nums and names:
        hits = _scene_numbers_for(names, scenes)
        nums = sorted({hits[0], hits[-1]}) if hits else []
    partial = ""
    if _ALL.search(message or "") or (not nums and not names):
        with_pics = [i for i, s in enumerate(scenes, start=1) if s.get("image_path") or s.get("video_path")]
        room = max(1, MAX_IMAGES - min(2, len(cast)))
        if len(with_pics) > room:
            step = len(with_pics) / room
            nums = [with_pics[int(k * step)] for k in range(room)]
            partial = (f"Only {len(nums)} of {len(with_pics)} scene pictures are attached (a sample). This is NOT a "
                       "check of every scene: for that, propose check_scenes.")
        else:
            nums = with_pics
    elif len(nums) == 1:
        nums = [x for x in (nums[0] - 1, nums[0]) if 1 <= x <= n]     # one scene: the scene before, to compare
    for c in cast:
        if c["name"] in names or (not names and len(cast) <= 2):
            for label, path in ident.references_for(c, "front", None, 1):
                wanted.append((f"{c['name']}'s approved reference picture", "image", path))
    for k in nums:
        s = scenes[k - 1]
        if s.get("image_path"):
            wanted.append((f"{tag}scene {k} picture ({s.get('heading') or ''})", "image", s["image_path"]))
        elif s.get("video_path"):
            wanted.append((f"{tag}scene {k} video frame ({s.get('heading') or ''})", "video", s["video_path"]))
        else:
            wanted.append((f"{tag}scene {k}", "missing", ""))
    images, attached, unavailable = [], [], []
    for label, kind, path in wanted:
        if kind == "missing":
            unavailable.append(f"{label}: no picture or video made yet")
            continue
        if owner and not str(path).startswith(f"{owner}/"):
            # Studio files are stored under their owner's id; anything else is never read.
            unavailable.append(f"{label}: not accessible")
            continue
        if len(images) >= MAX_IMAGES:
            unavailable.append(f"{label}: not attached (limit of {MAX_IMAGES} images per message)")
            continue
        data = None
        try:
            data = video_frame(files, path) if kind == "video" else shrink(files.get(path))
        except Exception:
            data = None
        if data:
            images.append((label, data))
            attached.append(label)
        else:
            unavailable.append(f"{label}: couldn't be loaded")
    if partial:
        unavailable.insert(0, partial)
    return images, attached, unavailable


def manifest(attached: list[str], unavailable: list[str], visual: bool) -> str:
    """What the prompt says about images, so the assistant never claims to have seen one it didn't get."""
    if not visual:
        return ("VISUAL MATERIAL: none attached to this message. You have NOT seen the pictures; rely on the PICTURE "
                "CHECK results and say so if asked how something looks.")
    lines = ["VISUAL MATERIAL (these images are attached to this message and you can see them):"]
    lines += [f"- Image {i}: {label}" for i, label in enumerate(attached, start=1)] or ["- (none could be attached)"]
    if unavailable:
        lines.append("NOT seen (don't describe these as if you saw them):")
        lines += [f"- {u}" for u in unavailable]
    lines.append("Describe only what the attached images show. Compare faces, hair, clothes, place, objects and "
                 "positions between them and with the PROJECT facts. Say 'it looks like' when unsure.")
    return "\n".join(lines)
