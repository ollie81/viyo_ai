"""
Viyo Creator Studio API: creators make AI dramas, episodes and films,
scene by scene, and pay with Studio credits.

  * Every route checks the signed-in (non-guest) account owns what it
    touches; someone else's project answers 404.
  * Every paid action is priced on the server (studio_pricing), shown to
    the creator first (/estimate), then held and queued in one atomic
    step (studio_create_job) with an idempotency key, so a double tap
    never charges twice. The worker keeps the real cost and returns the
    rest; failures return everything.
  * Files live in the private creator-studio bucket; the app only gets
    one-hour signed links.
  * Studio credits are separate from Watch coins (see creator_studio_schema.sql).
"""
from __future__ import annotations

import os
import re
import threading
import time
import uuid
from collections import defaultdict, deque
from typing import Literal, Optional

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field

import studio_ai
import studio_casting
import studio_identity as ident
import studio_lipsync
import studio_sfx
import studio_pricing as pricing
from studio_store import Files, Store, SupabaseFiles, SupabaseStore

router = APIRouter(prefix="/api/v1/studio", tags=["creator_studio"])

SIGN_SECONDS = 3600
JOB_KINDS = ("script", "breakdown", "portrait", "image", "voice", "video", "render", "outfits", "cover", "elements",
             "element")
LANE = {"render": "render"}
CREDIT_PACKS = [  # credits are worth $0.01 each to Viyo after payment fees
    {"id": "starter", "credits": 500, "label": "500 credits"},
    {"id": "creator", "credits": 1500, "label": "1,500 credits"},
    {"id": "studio", "credits": 4000, "label": "4,000 credits"},
    {"id": "pro", "credits": 10000, "label": "10,000 credits"},
]
# Lemon Squeezy: 5% + $0.50, +1.5% outside the US (their fees page). Packs
# are priced for the international case so every buyer's credits are covered.
LS_PCT, LS_FIXED = 0.065, 0.50

_store: Optional[Store] = None
_files: Optional[Files] = None
_job_log: dict[str, deque] = defaultdict(deque)


def get_store() -> Store:
    global _store
    if _store is None:
        _store = SupabaseStore(_admin())
    return _store


def get_files() -> Files:
    global _files
    if _files is None:
        _files = SupabaseFiles(_admin())
    return _files


def _admin():
    url, key = os.environ.get("SUPABASE_URL", ""), os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    if not (url and key):
        raise HTTPException(status_code=503, detail="Studio is not configured.")
    from supabase import create_client
    return create_client(url, key)


async def creator(authorization: str = Header(None)) -> str:
    from main import get_current_user_id_no_guest
    uid = await get_current_user_id_no_guest(authorization)
    if os.environ.get("STUDIO_CREATORS_OPEN", "true").lower() == "false":
        allowed = {u.strip() for u in os.environ.get("STUDIO_ALLOWED_USERS", "").split(",") if u.strip()}
        if uid not in allowed:
            raise HTTPException(status_code=403, detail="Viyo Studio is invite-only right now.")
    return uid


def _require_admin(x_admin_key: str = Header(None)) -> None:
    key = os.environ.get("ADMIN_API_KEY", "")
    if not key or x_admin_key != key:
        raise HTTPException(status_code=403, detail="Invalid admin key.")


# ---------------------------------------------------------------- helpers

def _uuid(value: str) -> str:
    try:
        return str(uuid.UUID(str(value)))
    except ValueError:
        raise HTTPException(status_code=404, detail="Not found.")


def _owned(table: str, row_id: str, user_id: str) -> dict:
    row = get_store().one(table, {"id": _uuid(row_id)})
    if not row or row.get("owner_id") != user_id:
        raise HTTPException(status_code=404, detail="Not found.")
    return row


def _project(project_id: str, user_id: str) -> dict:
    p = _owned("studio_projects", project_id, user_id)
    if p.get("status") == "archived":
        raise HTTPException(status_code=404, detail="Not found.")
    return p


def _balance(user_id: str) -> dict:
    b = get_store().rpc("studio_credit_balance", {"p_user_id": user_id}) or {}
    return {k: int(b.get(k) or 0) for k in ("available", "purchased", "promotional", "held")}


def _now_iso() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _sign(paths: list[Optional[str]]) -> dict[str, str]:
    paths = sorted({p for p in paths if p})
    if not paths:
        return {}
    try:
        return get_files().sign(paths, SIGN_SECONDS)
    except Exception as e:
        print(f"[WARN] studio signing failed: {e}")
        return {}


def _touch(project_id: str) -> None:
    import datetime
    get_store().update("studio_projects", {"id": project_id},
                       {"updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat()})


def _rate_limit(user_id: str, limit: int = 40, window: int = 60) -> None:
    now, q = time.time(), _job_log[user_id]
    while q and now - q[0] > window:
        q.popleft()
    if len(q) >= limit:
        raise HTTPException(status_code=429, detail="Too many requests at once. Wait a moment and try again.")
    q.append(now)


# ---------------------------------------------------------------- models

class ProjectIn(BaseModel):
    title: str = Field(..., min_length=1, max_length=120)
    logline: str = Field("", max_length=2000)
    kind: Literal["short", "episode", "series", "film"] = "episode"
    aspect_ratio: Literal["9:16", "16:9"] = "9:16"
    genre: str = Field("", max_length=60)
    visual_style: str = Field("", max_length=400)
    language: str = Field("English", max_length=40)
    target_seconds: int = Field(90, ge=15, le=600)
    captions: bool = True
    accept_terms: bool = False
    # 'video': every scene is a moving clip where the characters talk (the
    # default); 'pictures': still images with saved voices (cheaper).
    production_style: Literal["video", "pictures"] = "video"


class ProjectPatch(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=120)
    logline: Optional[str] = Field(None, max_length=2000)
    genre: Optional[str] = Field(None, max_length=60)
    visual_style: Optional[str] = Field(None, max_length=400)
    language: Optional[str] = Field(None, max_length=40)
    target_seconds: Optional[int] = Field(None, ge=15, le=600)
    captions: Optional[bool] = None
    script: Optional[str] = Field(None, max_length=60000)
    aspect_ratio: Optional[Literal["9:16", "16:9"]] = None
    music_volume: Optional[float] = Field(None, ge=0, le=1)
    image_quality: Optional[Literal["standard", "high"]] = None
    production_style: Optional[Literal["video", "pictures"]] = None


class CharacterIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=60)
    description: str = Field("", max_length=1200)
    voice: str = ""
    voice_style: str = Field("", max_length=80)


class CharacterPatch(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=60)
    description: Optional[str] = Field(None, max_length=1200)
    voice: Optional[str] = None
    voice_style: Optional[str] = Field(None, max_length=80)
    # Changing the voice of a character whose lines are already recorded
    # must be confirmed (409 otherwise).
    confirm_voice_change: bool = False


class Line(BaseModel):
    character: str = Field("", max_length=60)
    text: str = Field(..., max_length=600)
    emotion: str = Field("", max_length=40)   # the feeling of this line, e.g. "sad"
    delivery: str = Field("", max_length=80)  # optional performance note, e.g. "soft, hesitant"


class SceneIn(BaseModel):
    heading: str = Field("", max_length=200)
    visual: str = Field("", max_length=2000)
    dialogue: list[Line] = Field(default_factory=list, max_length=12)
    duration_seconds: float = Field(5, ge=2, le=20)
    mode: Optional[Literal["still", "video"]] = None   # default: the production's style
    position: Optional[int] = None
    sfx: list[str] = Field(default_factory=list, max_length=4)
    speech: Optional[Literal["video_model", "saved_voice", "lipsync"]] = None


class ScenePatch(BaseModel):
    heading: Optional[str] = Field(None, max_length=200)
    visual: Optional[str] = Field(None, max_length=2000)
    dialogue: Optional[list[Line]] = Field(None, max_length=12)
    duration_seconds: Optional[float] = Field(None, ge=2, le=20)
    mode: Optional[Literal["still", "video"]] = None
    sfx: Optional[list[str]] = Field(None, max_length=4)
    speech: Optional[Literal["video_model", "saved_voice", "lipsync"]] = None
    # Which outfit each character wears here ({character id: outfit id};
    # empty = keep the outfit from the scene before), and the camera framing.
    outfits: Optional[dict[str, str]] = None
    # One-scene-only changes on top of the chosen outfit, per character and
    # part: {character id: {"clothing"|"shoes"|"accessories"|"makeup": "..."}}.
    outfit_overrides: Optional[dict[str, dict[str, str]]] = None
    camera: Optional[Literal["auto", "close_up", "front", "side", "back", "full_body", "wide"]] = None
    # The scene's place ("" = work it out from the heading) and objects
    # (props_auto = work them out from the text).
    location_id: Optional[str] = None
    props: Optional[list[str]] = Field(None, max_length=20)
    props_auto: bool = False


def _clean_sfx(items: list[str]) -> list[str]:
    bad = [x for x in items if x not in studio_sfx.EFFECTS]
    if bad:
        raise HTTPException(status_code=400, detail=f"Unknown sound effect: {bad[0]}")
    return list(dict.fromkeys(items))


class OrderIn(BaseModel):
    scene_ids: list[str] = Field(..., max_length=200)


class JobRequest(BaseModel):
    kind: Literal["script", "breakdown", "portrait", "image", "voice", "video", "render", "lipsync", "outfits", "cover",
                  "elements", "element"]
    target_ids: list[str] = Field(default_factory=list, max_length=100)
    request_id: str = Field(..., min_length=8, max_length=80)
    # Portrait jobs only: which reference-sheet views to make, or one outfit's picture.
    views: list[str] = Field(default_factory=list, max_length=7)
    outfit_id: Optional[str] = None
    # Cover jobs only: the creator's idea for the poster (optional).
    cover_note: Optional[str] = Field(None, max_length=300)
    # A new scene list for a project that already has scenes must be
    # confirmed: unchanged scenes keep their work, changed ones are set
    # aside (never deleted).
    confirm_replace: bool = False


class EstimateRequest(BaseModel):
    kind: Literal["script", "breakdown", "portrait", "image", "voice", "video", "render", "lipsync", "outfits", "cover",
                  "elements", "element"]
    target_ids: list[str] = Field(default_factory=list, max_length=100)
    views: list[str] = Field(default_factory=list, max_length=7)
    outfit_id: Optional[str] = None
    # Cover jobs only: the creator's idea for the poster (optional).
    cover_note: Optional[str] = Field(None, max_length=300)


class IdentityPatch(BaseModel):
    # {"basic": {...}, "face": {...}, "body": {...}, "default_outfit": "..."}; empty value clears a field.
    identity: dict


class OutfitIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=60)
    description: str = Field("", max_length=800)
    # {"clothing", "shoes", "accessories", "makeup"}: each changes on its own.
    parts: dict[str, str] = Field(default_factory=dict)
    is_temporary: bool = False


class OutfitPatch(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=60)
    description: Optional[str] = Field(None, max_length=800)
    parts: Optional[dict[str, str]] = None
    is_temporary: Optional[bool] = None


class OutfitAccept(BaseModel):
    use_in_suggested_scenes: bool = True


class OutfitImport(BaseModel):
    outfit_id: str


class AssistantGenerateIn(BaseModel):
    request_id: str = Field(..., min_length=8, max_length=80)
    # Must match the latest server estimate. This makes price confirmation
    # explicit before the existing credit-hold/job flow is entered.
    confirm_credits: int = Field(..., ge=0)


class AssistantApproveIn(BaseModel):
    confirm_story: bool = False


class GrantIn(BaseModel):
    user: str = Field(..., description="user id, email or @username")
    credits: int = Field(..., gt=0, le=1_000_000)
    kind: Literal["grant", "purchase"] = "grant"
    note: str = Field("", max_length=200)
    expires_days: Optional[int] = Field(None, ge=1, le=3650)
    amount_usd: float = Field(0, ge=0)
    fee_usd: float = Field(0, ge=0)
    reference: str = Field("", max_length=120)


# ---------------------------------------------------------------- serialisers

def _character_out(c: dict, urls: dict, outfits: Optional[list] = None, library: Optional[list] = None) -> dict:
    return {"id": c["id"], "name": c["name"], "description": c["description"], "voice": c["voice"],
            "voice_provider": c.get("voice_provider") or "google", "voice_style": c.get("voice_style") or "",
            "voice_language": c.get("voice_language"), "voice_locked": bool(c.get("voice_locked_at")),
            "voice_changed_at": c.get("voice_changed_at"), "identity_id": c.get("identity_id") or c["id"],
            "age_category": c.get("age_category"), "age_years": c.get("age_years"),
            "age_confidence": c.get("age_confidence"), "age_label": studio_casting.AGE_LABEL.get(c.get("age_category") or ""),
            "gender_presentation": c.get("gender_presentation"), "voice_category": c.get("voice_category"),
            "voice_description": c.get("voice_description") or "", "voice_status": c.get("voice_status") or
            ("cast" if c.get("voice") else "needs_voice"),
            "portrait_url": urls.get(c.get("reference_path") or ""), "position": c.get("position", 0),
            **_identity_out(c, urls, outfits, library)}


def _identity_out(c: dict, urls: dict, outfits: Optional[list] = None, library: Optional[list] = None) -> dict:
    views = c.get("reference_views") or {}
    status = ident.sheet_status(views)
    return {
        "identity": c.get("identity") or {}, "identity_locked": bool(c.get("identity_locked_at")),
        "guessed_fields": ident.guessed_fields(c.get("identity")),
        "identity_locked_at": c.get("identity_locked_at"), "identity_version": int(c.get("identity_version") or 0),
        "reference_views": [{"view": v, "label": ident.VIEW_LABEL[v], "status": status[v],
                             "url": urls.get((views.get(v) or {}).get("path") or "")} for v in ident.VIEWS],
        "lock_problem": ident.ready_to_lock(views),
        "outfits": [_outfit_out(o, urls) for o in _sorted_outfits(outfits) if (o.get("status") or "saved") == "saved"],
        "outfit_suggestions": [_outfit_out(o, urls) for o in _sorted_outfits(outfits) if o.get("status") == "suggested"],
        "outfit_library": [_outfit_out(o, urls) for o in library or []],
    }


def _sorted_outfits(outfits) -> list:
    return sorted(outfits or [], key=lambda o: (not o.get("is_default"), o.get("position") or 0, o.get("created_at") or ""))


def _outfit_out(o: dict, urls: dict) -> dict:
    sig = o.get("image_signature")
    return {"id": o["id"], "character_id": o["character_id"], "name": o["name"], "description": o.get("description") or "",
            "parts": ident.clean_parts(o.get("parts")), "is_default": bool(o.get("is_default")),
            "is_temporary": bool(o.get("is_temporary")), "status": o.get("status") or "saved",
            "suggested_scenes": o.get("suggested_scenes") or [], "reason": o.get("reason") or "",
            "image_url": urls.get(o.get("reference_path") or ""), "has_image": bool(o.get("reference_path")),
            "image_outdated": bool(o.get("reference_path") and sig and sig != ident.outfit_signature(o)),
            "project_id": o.get("project_id")}


def _outfit_library(c: dict, own: list) -> list:
    """Saved outfits of the same person in the other episodes of the series,
    not already in this one: reused as they are, picture included."""
    try:
        group = [g for g in _identity_group(c) if g["project_id"] != c["project_id"]]
    except Exception:
        return []
    have = {o["name"].strip().lower() for o in own}
    out, seen = [], set()
    for g in group:
        for o in _outfits(g["project_id"]):
            key = o["name"].strip().lower()
            if (o["character_id"] == g["id"] and (o.get("status") or "saved") == "saved" and not o.get("is_temporary")
                    and key not in have and key not in seen):
                seen.add(key)
                out.append(o)
    return out


def _identity_paths(c: dict) -> list:
    return [c.get("reference_path")] + [(e or {}).get("path") for e in (c.get("reference_views") or {}).values()]


def _outfits(project_id: str) -> list:
    """Outfit rows (none before creator_studio_identity.sql is run)."""
    st = get_store()
    try:
        return st.select("studio_outfits", {"project_id": project_id})
    except Exception:
        if hasattr(st, "conn"):
            st.conn.rollback()
        return []


# Columns added by creator_studio_casting.sql and creator_studio_voices.sql.
# Writes fall back step by step when a file hasn't been run yet, so the
# rest of the record still saves.
CASTING_COLUMNS = ("age_category", "age_years", "age_confidence", "gender_presentation", "personality",
                   "voice_category", "voice_description", "voice_status", "voice_assignment_version")
VOICE_COLUMNS = ("voice_style", "voice_language", "voice_changed_at", "voice_provider", "voice_locked_at", "identity_id")


def _write_with_fallback(write, fields: dict):
    """Tries the full write, then without casting columns, then without the
    stage-1 voice columns too."""
    st = get_store()
    attempts = [dict(fields), {k: v for k, v in fields.items() if k not in CASTING_COLUMNS},
                {k: v for k, v in fields.items() if k not in CASTING_COLUMNS + VOICE_COLUMNS}]
    last = None
    for f in attempts:
        try:
            return write(f)
        except Exception as e:
            last = e
            if hasattr(st, "conn"):
                st.conn.rollback()
    raise last


def _catalog() -> list:
    from studio_worker import load_catalog
    return load_catalog(get_store())


def _available_voice(voice: str) -> bool:
    return any(v.voice_id == voice and v.status == "available" for v in _catalog())


def _identity_group(c: dict) -> list[dict]:
    """Every copy of this person across the series (same owner)."""
    root = c.get("identity_id") or c["id"]
    st = get_store()
    rows = {r["id"]: r for r in st.select("studio_characters", {"identity_id": root}) if r["owner_id"] == c["owner_id"]}
    first = st.one("studio_characters", {"id": root})
    if first and first["owner_id"] == c["owner_id"]:
        rows[first["id"]] = first
    rows[c["id"]] = c
    return list(rows.values())


def _outdated_voices(s: dict, cast: Optional[dict]) -> list[str]:
    """Names of characters whose voice changed after this scene was recorded."""
    if not cast or not s.get("audio_path"):
        return []
    return [cast[cid]["name"] for cid, v in (s.get("audio_voices") or {}).items()
            if cid in cast and (cast[cid].get("voice") or "") != v]


def _consistency(s: dict, cast: Optional[dict], resolved: Optional[dict]) -> list[str]:
    """Characters whose identity or outfit changed after this scene's picture
    or video was made (redo the scene to match). Scenes made before a
    character had a reference sheet are flagged too."""
    if not cast or not (s.get("image_path") or s.get("video_path")) or "made_with" not in s:
        return []
    from studio_worker import characters_in_scene
    made = s.get("made_with") or {}
    out = []
    for c in characters_in_scene(s, list(cast.values())):
        if not ident.sheet_status(c.get("reference_views") or {}).get(ident.ANCHOR) in ("draft", "approved"):
            continue
        m = made.get(c["id"])
        if not m or not m.get("anchor"):
            out.append(f"Made before {c['name']}'s reference sheet. Redo to match {c['name']}'s identity.")
        elif int(m.get("identity_version") or 0) != int(c.get("identity_version") or 0):
            out.append(f"{c['name']}'s identity changed after this was made. Redo to match.")
        elif resolved is not None and m.get("outfit_id") != ((resolved.get(c["id"]) or {}).get("id")):
            out.append(f"{c['name']}'s outfit for this scene changed after it was made. Redo to match.")
    return out


def _scene_world(s: dict, elements: Optional[list]) -> dict:
    import studio_elements
    if not elements:
        return {"location_used": None, "props_used": []}
    loc = studio_elements.scene_location(s, elements)
    return {"location_used": {"id": loc["id"], "name": loc["name"]} if loc else None,
            "props_used": [{"id": e["id"], "name": e["name"]} for e in studio_elements.scene_props(s, elements)]}


def _elements(project_id: str) -> list:
    """Places and objects (none before creator_studio_elements.sql is run)."""
    st = get_store()
    try:
        return sorted(st.select("studio_elements", {"project_id": project_id}),
                      key=lambda e: (e["kind"] != "location", e.get("position") or 0))
    except Exception:
        if hasattr(st, "conn"):
            st.conn.rollback()
        return []


def _element_out(e: dict, urls: dict) -> dict:
    return {"id": e["id"], "kind": e["kind"], "name": e["name"], "description": e.get("description") or "",
            "details": e.get("details") or {}, "status": e.get("status") or "draft",
            "image_url": urls.get(e.get("reference_path") or ""), "has_image": bool(e.get("reference_path"))}


def _scene_out(s: dict, urls: dict, cast: Optional[dict] = None, project: Optional[dict] = None,
               resolved: Optional[dict] = None, elements: Optional[list] = None) -> dict:
    return {"id": s["id"], "position": s["position"], "heading": s["heading"], "visual": s["visual"],
            "dialogue": s.get("dialogue") or [], "duration_seconds": float(s["duration_seconds"]), "mode": s["mode"],
            "image_url": urls.get(s.get("image_path") or ""), "video_url": urls.get(s.get("video_path") or ""),
            "audio_url": urls.get(s.get("audio_path") or ""),
            "audio_seconds": float(s["audio_seconds"]) if s.get("audio_seconds") else None,
            "has_image": bool(s.get("image_path")), "has_video": bool(s.get("video_path")),
            "has_audio": bool(s.get("audio_path")), "sfx": s.get("sfx") or [],
            "voice_outdated": _outdated_voices(s, cast),
            "speech": _speech(s, project), "speech_setting": s.get("speech"),
            "lipsync_url": urls.get(s.get("lipsync_path") or ""), "has_lipsync": bool(s.get("lipsync_path")),
            "outfits": s.get("outfits") or {}, "outfit_overrides": s.get("outfit_overrides") or {},
            "camera": s.get("camera") or "auto",
            "camera_used": ident.scene_camera(s),
            "made_with_references": (s.get("made_with") or {}).get("_refs") or [],
            "location_id": s.get("location_id"), "props": s.get("props"),
            **_scene_world(s, elements),
            "consistency": _consistency(s, cast, resolved)}


def _speech(s: dict, project: Optional[dict]) -> str:
    from studio_worker import scene_speech
    return scene_speech(s, project or {})


def _job_out(j: dict) -> dict:
    return {"id": j["id"], "kind": j["kind"], "status": j["status"], "scene_id": j.get("scene_id"),
            "character_id": j.get("character_id"), "estimated_credits": int(j.get("estimated_credits") or 0),
            "charged_credits": int(j.get("charged_credits") or 0), "error": j.get("error_message"),
            "created_at": j.get("created_at"), "finished_at": j.get("finished_at")}


def _render_out(r: dict) -> dict:
    return {"id": r["id"], "version": r["version"], "aspect_ratio": r["aspect_ratio"], "width": r.get("width"),
            "height": r.get("height"), "duration_seconds": float(r.get("duration_seconds") or 0),
            "bytes": r.get("bytes"), "created_at": r.get("created_at"),
            "post_id": r.get("post_id"), "published_at": r.get("published_at")}


def _project_out(p: dict) -> dict:
    keys = ("id", "title", "logline", "kind", "aspect_ratio", "genre", "visual_style", "language", "target_seconds",
            "captions", "script", "created_at", "updated_at", "music_name", "series_id")
    out = {k: p.get(k) for k in keys}
    out.update(has_music=bool(p.get("music_path")), music_volume=float(p.get("music_volume") or 0.25),
               image_quality=p.get("image_quality") or "standard",
               production_style=p.get("production_style") or "pictures",
               story_bible=p.get("story_bible") or None)
    return out


# ---------------------------------------------------------------- account

@router.get("/me")
async def me(user_id: str = Depends(creator)):
    return {
        "balance": _balance(user_id),
        "voices": [{"name": n, "gender": g, "style": s} for n, g, s in studio_ai.VOICES],
        "usd_per_credit": pricing.USD_PER_CREDIT,
        "packs": [{**p, "price_usd": pricing.pack_price(p["credits"], LS_PCT, LS_FIXED)} for p in CREDIT_PACKS],
        "checkout_available": False,
        "sound_effects": studio_sfx.labels(),
        "lipsync_available": studio_lipsync.enabled(),
        "price_checked": pricing.PRICE_CHECKED,
    }


VOICE_SAMPLE_TEXT = "Hi. This is how I sound. I can be the voice of one of the characters in your story."


def _voice_samples() -> dict[str, dict]:
    try:
        return {r["voice"]: r for r in get_store().select("studio_voice_samples", {})}
    except Exception:
        return {}   # creator_studio_voices.sql not run yet


@router.get("/voices")
async def voices(user_id: str = Depends(creator)):
    """The voices a character can have, with a short sample to listen to
    when an admin has made them."""
    samples = _voice_samples()
    urls = _sign([s["path"] for s in samples.values()])
    return {"provider": "google", "voices": [
        {"name": v.voice_id, "gender": v.gender, "style": v.descriptor, "ages": v.age_categories,
         "sample_url": urls.get((samples.get(v.voice_id) or {}).get("path") or "")}
        for v in _catalog() if v.status == "available"]}


# ---------------------------------------------------------------- admin voice gallery

class CatalogPatch(BaseModel):
    age_categories: Optional[list[str]] = Field(None, max_length=8)
    languages: Optional[list[str]] = Field(None, max_length=200)
    status: Optional[Literal["available", "unavailable"]] = None
    gender: Optional[Literal["female", "male", "unspecified"]] = None
    notes: Optional[str] = Field(None, max_length=1000)
    checked_by: str = Field("admin", max_length=60)


class CatalogIn(CatalogPatch):
    provider: Literal["google"] = "google"
    voice_id: str = Field(..., min_length=1, max_length=80)
    descriptor: str = Field("", max_length=40)


def _clean_catalog(fields: dict) -> dict:
    if "age_categories" in fields:
        bad = [a for a in fields["age_categories"] if a not in studio_casting.AGE_CATEGORIES]
        if bad:
            raise HTTPException(status_code=400, detail=f"Unknown age group: {', '.join(bad)}.")
        fields["age_categories"] = sorted(set(fields["age_categories"]), key=studio_casting.AGE_CATEGORIES.index)
    if "languages" in fields:
        langs = []
        for lang in fields["languages"]:
            if lang == "all":
                langs.append("all")
                continue
            ok, name = studio_casting.language_supported("google", lang)
            if not ok:
                raise HTTPException(status_code=400, detail=f"The voice model doesn't support {lang}.")
            langs.append(name)
        fields["languages"] = langs or ["all"]
    return fields


@router.get("/admin/voices", dependencies=[Depends(_require_admin)])
async def admin_voices():
    """Internal voice gallery: every catalog voice, what has been verified
    about it, its sample and how many characters use it."""
    st = get_store()
    try:
        rows = st.select("studio_voice_catalog", {})
    except Exception:
        raise HTTPException(status_code=503, detail="Run creator_studio_casting.sql first.")
    samples = _voice_samples()
    urls = _sign([s["path"] for s in samples.values()])
    uses: dict = {}
    for c in st.select("studio_characters", {}):
        if c.get("voice"):
            uses[c["voice"]] = uses.get(c["voice"], 0) + 1
    waiting = [c for c in st.select("studio_characters", {}) if not c.get("voice") and c.get("voice_status") == "needs_voice"]
    return {"age_groups": studio_casting.AGE_CATEGORIES, "voices": [
        {**{k: r.get(k) for k in ("provider", "voice_id", "gender", "descriptor", "age_categories", "languages",
                                   "status", "verified_by", "verified_at", "notes")},
         "sample_url": urls.get((samples.get(r["voice_id"]) or {}).get("path") or ""),
         "characters": uses.get(r["voice_id"], 0)}
        for r in sorted(rows, key=lambda r: (r["provider"], r["voice_id"]))],
        "characters_waiting": [{"name": c["name"], "age_category": c.get("age_category"),
                                "language": c.get("voice_language")} for c in waiting[:100]],
        "waiting_count": len(waiting)}


@router.patch("/admin/voices/{provider}/{voice_id}", dependencies=[Depends(_require_admin)])
async def admin_update_voice(provider: str, voice_id: str, body: CatalogPatch):
    """Record what an admin verified by listening, or take a voice out of
    casting. Characters already using it keep it (never silently changed)."""
    st = get_store()
    row = st.one("studio_voice_catalog", {"provider": provider, "voice_id": voice_id})
    if not row:
        raise HTTPException(status_code=404, detail="No such voice.")
    fields = _clean_catalog(body.model_dump(exclude_none=True, exclude={"checked_by"}))
    if "age_categories" in fields:
        fields.update(verified_by=body.checked_by, verified_at=_now_iso())
    fields["updated_at"] = _now_iso()
    row = st.update("studio_voice_catalog", {"provider": provider, "voice_id": voice_id}, fields)[0]
    in_use = sum(1 for c in st.select("studio_characters", {"voice": voice_id}))
    return {"voice": row, "characters_using": in_use,
            "note": ("Characters already using this voice keep it." if fields.get("status") == "unavailable" and in_use
                     else None)}


@router.post("/admin/voices", dependencies=[Depends(_require_admin)])
async def admin_add_voice(body: CatalogIn):
    """Add another voice id from the provider (for example from Google's
    extended voice library) after listening to it."""
    fields = _clean_catalog(body.model_dump(exclude_none=True, exclude={"checked_by"}))
    fields.setdefault("age_categories", [])
    fields.setdefault("languages", ["all"])
    if fields["age_categories"]:
        fields.update(verified_by=body.checked_by, verified_at=_now_iso())
    st = get_store()
    if st.one("studio_voice_catalog", {"provider": body.provider, "voice_id": body.voice_id}):
        raise HTTPException(status_code=409, detail="That voice is already in the catalog.")
    return {"voice": st.insert("studio_voice_catalog", [fields])[0]}


class SamplesIn(BaseModel):
    confirm_spend: bool = False
    voices: list[str] = Field(default_factory=list, max_length=40)


@router.post("/admin/voices/samples", dependencies=[Depends(_require_admin)])
async def admin_voice_samples(body: SamplesIn):
    """Makes the missing voice samples with the real voice model. Costs real
    money (about $0.0007 per voice), so it only runs with confirm_spend."""
    have = _voice_samples()
    known = [v.voice_id for v in _catalog() if v.provider == "google"]
    todo = [v for v in (body.voices or known) if v in known and v not in have]
    est = pricing.voice_cost(len(todo) * 6.0)
    if not body.confirm_spend:
        return {"would_make": todo, "estimated_cost_usd": round(est.provider_cost_usd, 4), "made": []}
    ai, files, st, made, failed = studio_ai.GoogleStudioAI(), get_files(), get_store(), [], []
    for v in todo:
        try:
            wav, secs, _cost = ai.voice(VOICE_SAMPLE_TEXT, v, "")
            path = f"voice-samples/{v.lower()}-{uuid.uuid4().hex[:8]}.wav"
            files.put(path, wav, "audio/wav")
            st.insert("studio_voice_samples", [{"voice": v, "path": path, "seconds": round(secs, 2)}])
            made.append(v)
        except Exception as e:
            failed.append({"voice": v, "error": str(e)[:200]})
    return {"made": made, "failed": failed}


# ---------------------------------------------------------------- live provider test (admin)

class LiveCheckIn(BaseModel):
    confirm_spend: bool = False
    cap_usd: float = Field(5.0, gt=0, le=5.0)


_live_lock = threading.Lock()
_live_running: dict = {}


@router.post("/admin/live-check", dependencies=[Depends(_require_admin)])
async def admin_live_check(body: LiveCheckIn):
    """Runs the first real test of every paid provider (Gemini text, image,
    voice, Omni video, sync.so) on one tiny production, with a hard cap.
    No creator data or credits are touched. Returns a results link."""
    import studio_livecheck
    if not body.confirm_spend:
        return {"would_spend_up_to_usd": body.cap_usd, "expected_usd": 2.35,
                "note": "Send confirm_spend to run it. Costs real money on Viyo's Google and sync.so accounts."}
    with _live_lock:
        busy = [c for c in _live_running.values() if c.report.get("status") == "running"]
        if busy:
            raise HTTPException(status_code=409, detail="A live test is already running.")
        check = studio_livecheck.start_in_background(get_files(), body.cap_usd)
        _live_running[check.id] = check
    return {"id": check.id, "results_path": f"/api/v1/studio/live-check/{check.id}"}


@router.get("/live-check/{check_id}")
async def live_check_report(check_id: str):
    """Results of a live test. The id is long and random; there is no
    personal data in it, only test files and costs."""
    import studio_livecheck
    if not re.fullmatch(r"[A-Za-z0-9_-]{20,40}", check_id or ""):
        raise HTTPException(status_code=404, detail="Not found.")
    check = _live_running.get(check_id)
    report = dict(check.report) if check else studio_livecheck.latest_report(get_files(), check_id)
    if not report:
        raise HTTPException(status_code=404, detail="Not found.")
    names = sorted({f for s in report.get("steps", []) for f in (s.get("files") or [])})
    paths = [f"livecheck/{check_id}/{n}" for n in names]
    try:
        signed = get_files().sign(paths, 7 * 24 * 3600) if paths else {}
    except Exception:
        signed = {}
    report["file_urls"] = {n: signed.get(f"livecheck/{check_id}/{n}") for n in names}
    report["spent_usd"] = round(check.budget.spent, 4) if check else report.get("spent_usd")
    return report


@router.get("/credits")
async def credits(user_id: str = Depends(creator)):
    rows = get_store().select("studio_credit_ledger", {"user_id": user_id}, order="created_at", desc=True, limit=60)
    return {"balance": _balance(user_id),
            "history": [{"kind": r["kind"], "credits": int(r["credits"]), "note": r.get("note"),
                         "job_id": r.get("job_id"), "created_at": r["created_at"]} for r in rows]}


# ---------------------------------------------------------------- projects

@router.get("/projects")
async def list_projects(user_id: str = Depends(creator)):
    st = get_store()
    projects = [p for p in st.select("studio_projects", {"owner_id": user_id}, order="updated_at", desc=True, limit=100)
                if p.get("status") != "archived"]
    ids = [p["id"] for p in projects]
    scenes = st.select("studio_scenes", {"project_id": ids}) if ids else []
    renders = st.select("studio_renders", {"project_id": ids}) if ids else []
    first_image: dict[str, tuple] = {}
    counts: dict[str, list] = defaultdict(lambda: [0, 0])
    for s in scenes:
        c = counts[s["project_id"]]
        c[0] += 1
        if s.get("image_path"):
            c[1] += 1
            cur = first_image.get(s["project_id"])
            if cur is None or s["position"] < cur[0]:
                first_image[s["project_id"]] = (s["position"], s["image_path"])
    render_count = defaultdict(int)
    for r in renders:
        render_count[r["project_id"]] += 1
    urls = _sign([v[1] for v in first_image.values()] + [p.get("cover_path") for p in projects])
    return {"balance": _balance(user_id), "projects": [{
        **{k: p[k] for k in ("id", "title", "logline", "kind", "aspect_ratio", "genre", "updated_at")},
        "scene_count": counts[p["id"]][0], "scenes_with_images": counts[p["id"]][1],
        "render_count": render_count[p["id"]],
        "cover_url": urls.get(p.get("cover_path") or ""),
        "thumbnail_url": urls.get(p.get("cover_path") or "") or urls.get(first_image.get(p["id"], (0, ""))[1])}
        for p in projects]}


@router.post("/projects")
async def create_project(body: ProjectIn, user_id: str = Depends(creator)):
    if not body.accept_terms:
        raise HTTPException(status_code=400, detail="Please confirm the characters and story are yours to use.")
    _rate_limit(user_id)
    row = body.model_dump(exclude={"accept_terms"})
    row.update(owner_id=user_id, title=row["title"].strip())
    st = get_store()
    try:
        created = st.insert("studio_projects", [row])[0]
    except Exception:
        # creator_studio_video_drama.sql not run yet: make a pictures production.
        if hasattr(st, "conn"):
            st.conn.rollback()
        row.pop("production_style", None)
        created = st.insert("studio_projects", [row])[0]
    return _project_out(created)


@router.post("/projects/{project_id}/next-episode")
async def next_episode(project_id: str, user_id: str = Depends(creator)):
    """A new episode in the same series: same settings and the same cast,
    with each character keeping their voice, delivery note and portrait."""
    src = _project(project_id, user_id)
    _rate_limit(user_id)
    st = get_store()
    series = src.get("series_id") or src["id"]
    if not src.get("series_id"):
        st.update("studio_projects", {"id": src["id"]}, {"series_id": series})
    episodes = st.select("studio_projects", {"series_id": series, "owner_id": user_id})
    n = max(2, len(episodes) + 1)   # `episodes` already includes this one
    base = re.sub(r"\s*[·-]\s*Episode \d+$", "", src["title"]).strip() or src["title"]
    row = {k: src.get(k) for k in ("kind", "aspect_ratio", "genre", "visual_style", "language", "target_seconds",
                                   "captions")}
    if src.get("production_style"):
        row["production_style"] = src["production_style"]
    row.update(owner_id=user_id, series_id=series, title=f"{base} · Episode {n}"[:120], logline="")
    if src.get("image_quality"):
        row["image_quality"] = src["image_quality"]
    new = st.insert("studio_projects", [row])[0]
    cast = sorted(st.select("studio_characters", {"project_id": src["id"]}), key=lambda c: c.get("position") or 0)
    copies = []
    for c in cast:
        copy = {"project_id": new["id"], "owner_id": user_id, "name": c["name"], "description": c.get("description") or "",
                "voice": c.get("voice") or "", "reference_path": c.get("reference_path"),
                "rights_confirmed": bool(c.get("rights_confirmed")), "position": c.get("position") or 0}
        from studio_worker import SERIES_KEYS
        for k in SERIES_KEYS:
            if k in c and k not in copy:
                copy[k] = c.get(k)
        if "identity_id" in c:
            copy["identity_id"] = c.get("identity_id") or c["id"]
        # The approved look comes along: identity, reference sheet, lock.
        for k in ("identity", "reference_views", "identity_locked_at", "identity_version"):
            if k in c:
                copy[k] = c.get(k)
        copies.append(copy)
    if copies:
        made = {r["name"]: r["id"] for r in st.insert("studio_characters", copies)}
        old = {c["id"]: c["name"] for c in cast}
        outfits = [{"project_id": new["id"], "owner_id": user_id, "character_id": made[old[o["character_id"]]],
                    "name": o["name"], "description": o.get("description") or "", "reference_path": o.get("reference_path"),
                    "is_default": bool(o.get("is_default")), "position": o.get("position") or 0,
                    "parts": o.get("parts") or {}, "image_signature": o.get("image_signature")}
                   for o in _outfits(src["id"]) if old.get(o["character_id"]) in made
                   and (o.get("status") or "saved") == "saved" and not o.get("is_temporary")]
        if outfits:
            st.insert("studio_outfits", outfits)
    for e in _elements(src["id"]):
        st.insert("studio_elements", [{k: e.get(k) for k in ("kind", "name", "description", "details", "reference_path",
                                                             "status", "position")} |
                                      {"project_id": new["id"], "owner_id": user_id}])
    return _project_out(new)


@router.get("/projects/{project_id}")
async def get_project(project_id: str, user_id: str = Depends(creator)):
    p = _project(project_id, user_id)
    st = get_store()
    chars = sorted(st.select("studio_characters", {"project_id": p["id"]}), key=lambda c: (c.get("position") or 0, c["name"]))
    scenes = sorted(st.select("studio_scenes", {"project_id": p["id"]}), key=lambda s: s["position"])
    renders = st.select("studio_renders", {"project_id": p["id"]}, order="version", desc=True, limit=20)
    jobs = st.select("studio_jobs", {"project_id": p["id"]}, order="created_at", desc=True, limit=60)
    outfits = _outfits(p["id"])
    urls = _sign([x for c in chars for x in _identity_paths(c)] + [o.get("reference_path") for o in outfits] +
                 [s.get(k) for s in scenes for k in ("image_path", "video_path", "audio_path")])
    from studio_worker import characters_in_scene
    resolved = ident.resolve_outfits(scenes, outfits, chars, characters_in_scene)
    elements = _elements(p["id"])
    eurls = _sign([e.get("reference_path") for e in elements])
    libraries = {c["id"]: _outfit_library(c, [o for o in outfits if o["character_id"] == c["id"]])
                 for c in chars} if p.get("series_id") else {}
    urls.update(_sign([o.get("reference_path") for lib in libraries.values() for o in lib]))
    project_out = _project_out(p)
    covers = _covers(p)
    curls = _sign([c["path"] for c in covers])
    project_out["cover_url"] = curls.get(p.get("cover_path") or "") or (
        _sign([p.get("cover_path")]).get(p.get("cover_path") or "") if p.get("cover_path") else None)
    project_out["covers"] = [{"id": c["id"], "url": curls.get(c["path"]), "is_main": c["path"] == p.get("cover_path"),
                              "created_at": c.get("created_at")} for c in covers]
    return {"project": project_out,
            "characters": [_character_out(c, urls, [o for o in outfits if o["character_id"] == c["id"]],
                                          libraries.get(c["id"])) for c in chars],
            "scenes": [_scene_out(s, urls, {c["id"]: c for c in chars}, p, resolved.get(s["id"], {}), elements)
                       for s in scenes],
            "elements": [_element_out(e, eurls) for e in elements],
            "renders": [_render_out(r) for r in renders],
            "jobs": [_job_out(j) for j in jobs], "balance": _balance(user_id),
            "production": _production_out(p)}


def _covers(p: dict) -> list:
    """Every cover made for this production, newest first (all are kept)."""
    rows = get_store().select("studio_assets", {"project_id": p["id"], "kind": "image"})
    return sorted([r for r in rows if not r.get("scene_id") and r["owner_id"] == p["owner_id"]],
                  key=lambda r: r.get("created_at") or "", reverse=True)


class CoverPick(BaseModel):
    cover_id: str


@router.post("/projects/{project_id}/cover")
async def choose_cover(project_id: str, body: CoverPick, user_id: str = Depends(creator)):
    """Make one of this production's covers the main one."""
    p = _project(project_id, user_id)
    cover = next((c for c in _covers(p) if c["id"] == _uuid(body.cover_id)), None)
    if not cover:
        raise HTTPException(status_code=404, detail="Not found.")
    get_store().update("studio_projects", {"id": p["id"]}, {"cover_path": cover["path"], "updated_at": _now_iso()})
    return {"ok": True, "cover_id": cover["id"]}


@router.get("/projects/{project_id}/assistant")
async def creator_assistant_state(project_id: str, user_id: str = Depends(creator)):
    """Creator-facing assistant context for one owned production.

    The assistant is a project workspace, not a shared admin chat. It returns
    only this creator's story, script and script-job status.
    """
    p = _project(project_id, user_id)
    st = get_store()
    jobs = st.select("studio_jobs", {"project_id": p["id"]}, order="created_at", desc=True, limit=60)
    script_jobs = [j for j in jobs if j.get("kind") == "script"]
    bible = p.get("story_bible") or {}
    approval = bible.get("_assistant_approval") or {}
    estimate = _quote("script", p, [])[0]
    return {
        "project": _project_out(p),
        "story_status": "approved" if approval.get("approved") else ("draft" if bible else "not_started"),
        "story_approved_at": approval.get("approved_at"),
        "script_job": _job_out(script_jobs[0]) if script_jobs else None,
        "script_estimate": estimate,
        "balance": _balance(user_id),
    }


@router.post("/projects/{project_id}/assistant/generate")
async def creator_assistant_generate(project_id: str, body: AssistantGenerateIn,
                                     user_id: str = Depends(creator)):
    """Start the existing priced story/script workflow from the assistant.

    The caller must first display /estimate and echo its credit total. The
    existing idempotent job/hold/worker path remains the only way to run AI.
    """
    p = _project(project_id, user_id)
    estimate = _quote("script", p, [])[0]
    if estimate["blocked"]:
        raise HTTPException(status_code=400, detail=estimate["blocked"])
    if body.confirm_credits != estimate["credits"]:
        raise HTTPException(status_code=409, detail={
            "code": "price_changed", "message": "The estimate changed. Review the new price before continuing.",
            "estimate": estimate, "balance": _balance(user_id)})
    result = await start_jobs(project_id, JobRequest(kind="script", request_id=body.request_id), user_id)
    return {**result, "estimate": estimate}


@router.post("/projects/{project_id}/assistant/approve")
async def creator_assistant_approve(project_id: str, body: AssistantApproveIn,
                                    user_id: str = Depends(creator)):
    """Record the creator's approval of the generated story bible.

    Approval does not generate or replace any paid assets.
    """
    p = _project(project_id, user_id)
    if not body.confirm_story:
        raise HTTPException(status_code=400, detail="Confirm that you approve this story before continuing.")
    bible = p.get("story_bible") or {}
    if not p.get("script") or not bible:
        raise HTTPException(status_code=409, detail="Generate the story and screenplay before approving it.")
    import copy
    updated = copy.deepcopy(bible)
    updated["_assistant_approval"] = {"approved": True, "approved_at": _now_iso()}
    try:
        p = get_store().update("studio_projects", {"id": p["id"]},
                               {"story_bible": updated, "updated_at": _now_iso()})[0]
    except Exception as e:
        if hasattr(get_store(), "conn"):
            get_store().conn.rollback()
        print(f"[WARN] Creator Studio story approval could not be saved: {str(e)[:160]}")
        raise HTTPException(status_code=503, detail="Story approval storage is not ready. Run creator_studio_story.sql.")
    return {"ok": True, "story_status": "approved", "story_approved_at": updated["_assistant_approval"]["approved_at"],
            "project": _project_out(p)}


@router.patch("/projects/{project_id}")
async def update_project(project_id: str, body: ProjectPatch, user_id: str = Depends(creator)):
    p = _project(project_id, user_id)
    fields = body.model_dump(exclude_none=True)
    if fields:
        import copy
        import datetime
        # Approval applies to the exact idea and screenplay the creator reviewed.
        # Any edit to either invalidates it so stale approval cannot authorize
        # production of a changed story.
        if ("script" in fields or "logline" in fields) and p.get("story_bible"):
            bible = copy.deepcopy(p["story_bible"])
            bible.pop("_assistant_approval", None)
            fields["story_bible"] = bible
        fields["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        p = get_store().update("studio_projects", {"id": p["id"]}, fields)[0]
    return _project_out(p)


@router.post("/projects/{project_id}/archive")
async def archive_project(project_id: str, user_id: str = Depends(creator)):
    """Hides the project. Its files and renders are kept."""
    p = _project(project_id, user_id)
    get_store().update("studio_projects", {"id": p["id"]}, {"status": "archived"})
    return {"ok": True}


# ---------------------------------------------------------------- characters

@router.post("/projects/{project_id}/characters")
async def add_character(project_id: str, body: CharacterIn, user_id: str = Depends(creator)):
    p = _project(project_id, user_id)
    st = get_store()
    from studio_worker import profile_fields
    cast_rows = st.select("studio_characters", {"project_id": p["id"]})
    profile = studio_casting.build_profile(body.name.strip(), None, body.description, p.get("language") or "English")
    fields = {"project_id": p["id"], "owner_id": user_id, "name": body.name.strip(),
              "description": body.description, "position": 999}
    extra = {**profile_fields(profile), "voice_style": body.voice_style.strip(), "voice_language": p.get("language")}
    if body.voice and _available_voice(body.voice):
        fields["voice"] = body.voice
        extra.update(voice_status="manual", voice_assignment_version=1,
                     voice_category=studio_casting.voice_category(profile))
    else:
        # Cast automatically, the same way the scene list does.
        seed = studio_casting.identity_seed(user_id, p.get("series_id") or p["id"], body.name)
        d = studio_casting.cast(profile, _catalog(), {c.get("voice") for c in cast_rows if c.get("voice")}, seed,
                                body.description)
        fields["voice"] = d.voice
        extra.update(voice_status=d.status, voice_category=d.voice_category, voice_description=d.voice_description,
                     voice_language=d.language, voice_assignment_version=1 if d.voice else 0)
        if not extra["voice_style"]:
            extra["voice_style"] = d.voice_style
    row = _write_with_fallback(lambda f: st.insert("studio_characters", [f])[0], {**fields, **extra})
    _touch(p["id"])
    return _character_out(row, {})


@router.patch("/characters/{character_id}")
async def update_character(character_id: str, body: CharacterPatch, user_id: str = Depends(creator)):
    c = _owned("studio_characters", character_id, user_id)
    st = get_store()
    before = {"voice": c.get("voice") or "", "voice_style": c.get("voice_style") or ""}
    fields = body.model_dump(exclude_none=True, exclude={"confirm_voice_change"})
    if "voice" in fields and not _available_voice(fields["voice"]):
        raise HTTPException(status_code=400, detail="Unknown voice.")
    if "voice_style" in fields:
        fields["voice_style"] = fields["voice_style"].strip()
    changes_voice = ("voice" in fields and fields["voice"] != (c.get("voice") or "")) or \
                    ("voice_style" in fields and fields["voice_style"] != (c.get("voice_style") or ""))
    for k in [k for k in ("voice", "voice_style") if k in fields and not changes_voice]:
        fields.pop(k)
    recorded: list[dict] = []
    group = _identity_group(c) if changes_voice else [c]
    if changes_voice:
        ids = {g["id"] for g in group}
        for pid in {g["project_id"] for g in group}:
            recorded += [s for s in st.select("studio_scenes", {"project_id": pid})
                         if s.get("audio_path") and ids & set((s.get("audio_voices") or {}).keys())]
        if recorded and not body.confirm_voice_change:
            n = len(recorded)
            raise HTTPException(status_code=409, detail={
                "code": "confirm_voice_change", "scenes": n, "scene_ids": [s["id"] for s in recorded],
                "message": (f"{c['name']} already speaks in {n} recorded scene{'s' if n != 1 else ''}. Those recordings "
                            f"keep the old voice until you record them again. New and re-recorded lines will use "
                            f"the new voice, and so will episodes you make from this one.")})
        fields["voice_changed_at"] = _now_iso()
        fields["voice_status"] = "manual"
        fields["voice_assignment_version"] = int(c.get("voice_assignment_version") or 0) + 1
    if fields:
        try:
            c = st.update("studio_characters", {"id": c["id"]}, fields)[0]
        except Exception:
            if hasattr(st, "conn"):
                st.conn.rollback()
            cid = c["id"]
            c = _write_with_fallback(lambda f: st.update("studio_characters", {"id": cid}, f)[0] if f
                                     else st.one("studio_characters", {"id": cid}), fields)
    if changes_voice:
        # The voice belongs to the person, not the episode: every copy of
        # this character in the series changes with it.
        same = {k: c.get(k) for k in ("voice", "voice_style", "voice_changed_at", "voice_status",
                                      "voice_assignment_version") if k in c}
        for g in group:
            if g["id"] != c["id"]:
                try:
                    st.update("studio_characters", {"id": g["id"]}, same)
                except Exception as e:
                    print(f"[WARN] studio: series voice sync failed for {g['id']}: {str(e)[:120]}")
    outfits = [o for o in _outfits(c["project_id"]) if o["character_id"] == c["id"]]
    out = _character_out(c, _sign(_identity_paths(c) + [o.get("reference_path") for o in outfits]), outfits)
    if changes_voice:
        hist = {"character_id": c["id"], "project_id": c["project_id"], "owner_id": user_id,
                "old_voice": before["voice"], "new_voice": c.get("voice") or "",
                "old_style": before["voice_style"], "new_style": c.get("voice_style") or "",
                "scenes_affected": len(recorded), "reason": "creator"}
        for attempt in (hist, {k: v for k, v in hist.items() if k != "reason"}):   # 'reason' needs the casting SQL
            try:
                st.insert("studio_voice_changes", [attempt])
                break
            except Exception as e:
                if hasattr(st, "conn"):
                    st.conn.rollback()
                print(f"[WARN] studio voice history not saved for {c['id']}: {str(e)[:160]}")
        out["rerecord_scene_ids"] = [s["id"] for s in recorded if s["project_id"] == c["project_id"]]
        out["other_episodes_affected"] = len({s["project_id"] for s in recorded if s["project_id"] != c["project_id"]})
        cat = c.get("age_category")
        voice = next((v for v in _catalog() if v.voice_id == c.get("voice")), None)
        if cat in studio_casting.STRICT and voice is not None and cat not in voice.age_categories:
            out["age_warning"] = (f"{c['name']} is a {studio_casting.AGE_LABEL[cat]}, and {voice.voice_id} hasn't been "
                                  "checked as suitable for that age.")
    return out


# ------------------------------------------------- character consistency

def _save_identity_fields(c: dict, fields: dict) -> dict:
    st = get_store()
    try:
        return st.update("studio_characters", {"id": c["id"]}, fields)[0]
    except Exception:
        if hasattr(st, "conn"):
            st.conn.rollback()
        raise HTTPException(status_code=503, detail="Character identities aren't ready yet. Run creator_studio_identity.sql.")


def _identity_response(c: dict) -> dict:
    outfits = [o for o in _outfits(c["project_id"]) if o["character_id"] == c["id"]]
    library = _outfit_library(c, outfits)
    return _character_out(c, _sign(_identity_paths(c) + [o.get("reference_path") for o in outfits + library]),
                          outfits, library)


def _unlocked(c: dict) -> None:
    if c.get("identity_locked_at"):
        raise HTTPException(status_code=409, detail={"code": "identity_locked",
                                                     "message": f"{c['name']}'s identity is locked. Unlock it to change it."})


@router.patch("/characters/{character_id}/identity")
async def update_identity(character_id: str, body: IdentityPatch, user_id: str = Depends(creator)):
    """The creator edits the permanent identity (only while it is unlocked)."""
    c = _owned("studio_characters", character_id, user_id)
    _unlocked(c)
    identity = ident.merge_identity(c.get("identity") or {}, body.identity)
    c = _save_identity_fields(c, {"identity": identity, "identity_version": int(c.get("identity_version") or 0) + 1})
    _touch(c["project_id"])
    return _identity_response(c)


@router.post("/characters/{character_id}/views/{view}/approve")
async def approve_view(character_id: str, view: str, user_id: str = Depends(creator)):
    c = _owned("studio_characters", character_id, user_id)
    if view not in ident.VIEWS:
        raise HTTPException(status_code=404, detail="Unknown view.")
    _unlocked(c)
    views = dict(c.get("reference_views") or {})
    if not (views.get(view) or {}).get("path"):
        raise HTTPException(status_code=409, detail="Make this view first.")
    views[view] = dict(views[view], status="approved")
    return _identity_response(_save_identity_fields(c, {"reference_views": views}))


@router.post("/characters/{character_id}/identity/lock")
async def lock_identity(character_id: str, user_id: str = Depends(creator)):
    """Approve every finished view and lock the identity: from now on every
    scene uses exactly this person, and nothing changes it until unlocked."""
    c = _owned("studio_characters", character_id, user_id)
    views = dict(c.get("reference_views") or {})
    problem = ident.ready_to_lock(views)
    if problem:
        raise HTTPException(status_code=409, detail={"code": "not_ready", "message": problem})
    for v, e in views.items():
        if (e or {}).get("path"):
            views[v] = dict(e, status="approved")
    c = _save_identity_fields(c, {"reference_views": views, "identity_locked_at": _now_iso()})
    return _identity_response(c)


@router.post("/characters/{character_id}/identity/unlock")
async def unlock_identity(character_id: str, user_id: str = Depends(creator)):
    c = _owned("studio_characters", character_id, user_id)
    return _identity_response(_save_identity_fields(c, {"identity_locked_at": None}))


@router.post("/characters/{character_id}/outfits")
async def add_outfit(character_id: str, body: OutfitIn, user_id: str = Depends(creator)):
    c = _owned("studio_characters", character_id, user_id)
    st = get_store()
    if not body.name.strip():
        raise HTTPException(status_code=400, detail="Give the outfit a name.")
    # No limit on outfits: a character can change clothes as often as the story needs.
    existing = [o for o in _outfits(c["project_id"]) if o["character_id"] == c["id"]
                and (o.get("status") or "saved") == "saved"]
    try:
        st.insert("studio_outfits", [{"project_id": c["project_id"], "owner_id": user_id, "character_id": c["id"],
                                      "name": body.name.strip(), "description": body.description.strip(),
                                      "parts": ident.clean_parts(body.parts), "is_temporary": body.is_temporary,
                                      "is_default": not existing, "position": len(existing)}])
    except Exception:
        if hasattr(st, "conn"):
            st.conn.rollback()
        raise HTTPException(status_code=503, detail="Outfits aren't ready yet. Run creator_studio_identity.sql.")
    _touch(c["project_id"])
    return _identity_response(c)


def _owned_outfit(outfit_id: str, user_id: str) -> dict:
    st = get_store()
    try:
        o = st.one("studio_outfits", {"id": _uuid(outfit_id)})
    except HTTPException:
        raise
    except Exception:
        if hasattr(st, "conn"):
            st.conn.rollback()
        o = None
    if not o or o["owner_id"] != user_id:
        raise HTTPException(status_code=404, detail="Not found.")
    return o


@router.patch("/outfits/{outfit_id}")
async def update_outfit(outfit_id: str, body: OutfitPatch, user_id: str = Depends(creator)):
    o = _owned_outfit(outfit_id, user_id)
    raw = body.model_dump(exclude_none=True)
    fields = {k: v.strip() for k, v in raw.items() if isinstance(v, str)}
    if "parts" in raw:
        fields["parts"] = ident.clean_parts(raw["parts"])   # the whole set; an empty part is cleared
    if "is_temporary" in raw:
        fields["is_temporary"] = raw["is_temporary"]
    if "name" in fields and not fields["name"]:
        raise HTTPException(status_code=400, detail="Give the outfit a name.")
    if fields:
        get_store().update("studio_outfits", {"id": o["id"]}, {**fields, "updated_at": _now_iso()})
    return _identity_response(get_store().one("studio_characters", {"id": o["character_id"]}))


@router.delete("/outfits/{outfit_id}")
async def delete_outfit(outfit_id: str, user_id: str = Depends(creator)):
    o = _owned_outfit(outfit_id, user_id)
    if o.get("is_default"):
        raise HTTPException(status_code=409, detail="The main outfit can't be removed. Edit it instead.")
    st = get_store()
    st.delete("studio_outfits", {"id": o["id"]})
    # Scenes that chose it fall back to the outfit from the scene before.
    for s in st.select("studio_scenes", {"project_id": o["project_id"]}):
        choice = dict(s.get("outfits") or {})
        if choice.get(o["character_id"]) == o["id"]:
            choice.pop(o["character_id"])
            st.update("studio_scenes", {"id": s["id"]}, {"outfits": choice})
    return _identity_response(st.one("studio_characters", {"id": o["character_id"]}))


@router.post("/outfits/{outfit_id}/accept")
async def accept_outfit(outfit_id: str, body: OutfitAccept, user_id: str = Depends(creator)):
    """The creator keeps an AI-suggested outfit, and (if they want) uses it in
    the scenes it was suggested for. Nothing is applied without this."""
    o = _owned_outfit(outfit_id, user_id)
    st = get_store()
    if o.get("status") == "suggested":
        st.update("studio_outfits", {"id": o["id"]}, {"status": "saved", "updated_at": _now_iso()})
        if body.use_in_suggested_scenes:
            wanted = set(o.get("suggested_scenes") or [])
            for s in st.select("studio_scenes", {"project_id": o["project_id"]}):
                if s["id"] in wanted:
                    st.update("studio_scenes", {"id": s["id"]}, {"outfits": {**(s.get("outfits") or {}), o["character_id"]: o["id"]}})
        _touch(o["project_id"])
    return _identity_response(st.one("studio_characters", {"id": o["character_id"]}))


@router.post("/characters/{character_id}/outfits/import")
async def import_outfit(character_id: str, body: OutfitImport, user_id: str = Depends(creator)):
    """Reuse a saved outfit from another episode of the series: copied with
    its picture, nothing is generated again."""
    c = _owned("studio_characters", character_id, user_id)
    src = _owned_outfit(body.outfit_id, user_id)
    if src["character_id"] not in {g["id"] for g in _identity_group(c)}:
        raise HTTPException(status_code=400, detail="That outfit belongs to another character.")
    st = get_store()
    existing = [o for o in _outfits(c["project_id"]) if o["character_id"] == c["id"]]
    st.insert("studio_outfits", [{"project_id": c["project_id"], "owner_id": user_id, "character_id": c["id"],
                                  "name": src["name"], "description": src.get("description") or "",
                                  "parts": src.get("parts") or {}, "reference_path": src.get("reference_path"),
                                  "image_signature": src.get("image_signature"), "is_default": not existing,
                                  "position": len(existing)}])
    _touch(c["project_id"])
    return _identity_response(c)


def _clean_outfit_overrides(scene: dict, overrides: dict) -> dict:
    cast = {c["id"] for c in get_store().select("studio_characters", {"project_id": scene["project_id"]})}
    clean = {}
    for cid, parts in (overrides or {}).items():
        if cid not in cast:
            raise HTTPException(status_code=400, detail="That character isn't in this production.")
        bad = [k for k in parts if k not in ident.OUTFIT_PARTS]
        if bad:
            raise HTTPException(status_code=400, detail=f"Unknown outfit part: {bad[0]}.")
        p = ident.clean_parts(parts)
        if p:
            clean[cid] = p
    return clean


def _clean_outfit_choice(scene: dict, choice: dict) -> dict:
    outfits = {o["id"]: o for o in _outfits(scene["project_id"]) if (o.get("status") or "saved") == "saved"}
    clean = {}
    for cid, oid in (choice or {}).items():
        if not oid:
            continue
        o = outfits.get(oid)
        if not o or o["character_id"] != cid:
            raise HTTPException(status_code=400, detail="That outfit doesn't belong to that character.")
        clean[cid] = oid
    return clean


@router.delete("/characters/{character_id}")
async def delete_character(character_id: str, user_id: str = Depends(creator)):
    c = _owned("studio_characters", character_id, user_id)
    get_store().delete("studio_characters", {"id": c["id"]})
    return {"ok": True}


# ---------------------------------------------------------------- scenes

@router.post("/projects/{project_id}/scenes")
async def add_scene(project_id: str, body: SceneIn, user_id: str = Depends(creator)):
    p = _project(project_id, user_id)
    st = get_store()
    scenes = sorted(st.select("studio_scenes", {"project_id": p["id"]}), key=lambda s: s["position"])
    if len(scenes) >= 80:
        raise HTTPException(status_code=400, detail="A production can have up to 80 scenes.")
    pos = len(scenes) if body.position is None else max(0, min(body.position, len(scenes)))
    for s in scenes[pos:]:
        st.update("studio_scenes", {"id": s["id"]}, {"position": s["position"] + 1})
    row = st.insert("studio_scenes", [{"project_id": p["id"], "owner_id": user_id, "position": pos,
                                       "heading": body.heading, "visual": body.visual,
                                       "dialogue": [l.model_dump() for l in body.dialogue],
                                       "duration_seconds": body.duration_seconds,
                                       "mode": body.mode or ("video" if p.get("production_style") == "video" else "still"),
                                       "sfx": _clean_sfx(body.sfx), **({"speech": body.speech} if body.speech else {})}])[0]
    _touch(p["id"])
    return _scene_out(row, {}, None, p)


@router.patch("/scenes/{scene_id}")
async def update_scene(scene_id: str, body: ScenePatch, user_id: str = Depends(creator)):
    s = _owned("studio_scenes", scene_id, user_id)
    fields = body.model_dump(exclude_none=True)
    if fields.get("speech") == "lipsync" and not studio_lipsync.enabled():
        raise HTTPException(status_code=400, detail="Lip sync isn't switched on yet.")
    if "sfx" in fields:
        fields["sfx"] = _clean_sfx(fields["sfx"])
    if "outfits" in fields:
        fields["outfits"] = _clean_outfit_choice(s, fields["outfits"])
    if "outfit_overrides" in fields:
        fields["outfit_overrides"] = _clean_outfit_overrides(s, fields["outfit_overrides"])
    fields.pop("props_auto", None)
    if body.location_id is not None or body.props is not None or body.props_auto:
        own = {e["id"]: e for e in _elements(s["project_id"])}
        if body.location_id is not None:
            if body.location_id and (own.get(body.location_id) or {}).get("kind") != "location":
                raise HTTPException(status_code=400, detail="That place isn't in this production.")
            fields["location_id"] = body.location_id or None
        if body.props_auto:
            fields["props"] = None
        elif body.props is not None:
            if any((own.get(x) or {}).get("kind") != "prop" for x in body.props):
                raise HTTPException(status_code=400, detail="That object isn't in this production.")
            fields["props"] = list(dict.fromkeys(body.props))
    if "dialogue" in fields:
        fields["dialogue"] = [l.model_dump() for l in body.dialogue]
        # New lines need a new voice track; the old file stays in the asset history.
        def key(lines):
            return [((l.get("character") or ""), (l.get("text") or ""), (l.get("emotion") or ""), (l.get("delivery") or ""))
                    for l in lines or []]
        if key(fields["dialogue"]) != key(s.get("dialogue")):
            fields.update(audio_path=None, audio_seconds=None)
            if "lipsync_path" in s:
                fields["lipsync_path"] = None
    if fields:
        s = get_store().update("studio_scenes", {"id": s["id"]}, fields)[0]
        _touch(s["project_id"])
    return _scene_out(s, _sign([s.get("image_path"), s.get("video_path"), s.get("audio_path")]))


@router.delete("/scenes/{scene_id}")
async def delete_scene(scene_id: str, user_id: str = Depends(creator)):
    s = _owned("studio_scenes", scene_id, user_id)
    st = get_store()
    st.delete("studio_scenes", {"id": s["id"]})
    for other in st.select("studio_scenes", {"project_id": s["project_id"]}):
        if other["position"] > s["position"]:
            st.update("studio_scenes", {"id": other["id"]}, {"position": other["position"] - 1})
    _touch(s["project_id"])
    return {"ok": True}


@router.post("/projects/{project_id}/scenes/order")
async def reorder_scenes(project_id: str, body: OrderIn, user_id: str = Depends(creator)):
    p = _project(project_id, user_id)
    st = get_store()
    scenes = {s["id"]: s for s in st.select("studio_scenes", {"project_id": p["id"]})}
    ids = [_uuid(i) for i in body.scene_ids]
    if set(ids) != set(scenes) or len(ids) != len(scenes):
        raise HTTPException(status_code=400, detail="Send every scene of this production exactly once.")
    for pos, sid in enumerate(ids):
        if scenes[sid]["position"] != pos:
            st.update("studio_scenes", {"id": sid}, {"position": pos})
    _touch(p["id"])
    return {"ok": True}


# ---------------------------------------------------------------- pricing + jobs

def _targets(kind: str, p: dict, target_ids: list[str]) -> list[dict]:
    st = get_store()
    if kind in ("script", "breakdown", "render", "outfits", "cover", "elements"):
        return [p]
    table = {"portrait": "studio_characters", "element": "studio_elements"}.get(kind, "studio_scenes")
    rows = st.select(table, {"project_id": p["id"]})
    if target_ids:
        wanted = {_uuid(t) for t in target_ids}
        rows = [r for r in rows if r["id"] in wanted]
        if len(rows) != len(wanted):
            raise HTTPException(status_code=404, detail="Some of those items don't exist.")
    return sorted(rows, key=lambda r: r.get("position") or 0)


def _price(kind: str, p: dict, target: dict, cast: list[dict],
           opts: Optional[dict] = None) -> tuple[pricing.Cost, Optional[str]]:
    """(cost, reason it can't be made yet)."""
    if kind == "script":
        return pricing.estimate_script(p["target_seconds"], len(p["title"]) + len(p["logline"])), \
            (None if p["logline"].strip() else "Describe your idea first.")
    if kind == "elements":
        n = len(get_store().select("studio_scenes", {"project_id": p["id"]}))
        return pricing.text_cost(1500 + min(len(p.get("script") or ""), 14000) // 3, pricing.THINKING_BUDGET + 400 + n * 60), \
            (None if (p.get("script") or "").strip() or n else "Write the script first.")
    if kind == "element":
        size = "2K" if p.get("image_quality") == "high" else "1K"
        return pricing.image_cost(1, 400, size), (None if (target.get("name") or "").strip() else "Name it first.")
    if kind == "cover":
        from studio_worker import COVER_REFS
        size = "2K" if p.get("image_quality") == "high" else "1K"
        return pricing.combine([pricing.text_cost(1800, pricing.THINKING_BUDGET + 400),
                                pricing.image_cost(1, 900 + pricing.SHEET_REF_TOKENS * COVER_REFS, size)]), \
            (None if (p.get("title") or "").strip() else "Give the production a title first.")
    if kind == "outfits":
        n = len(get_store().select("studio_characters", {"project_id": p["id"]}))
        return pricing.estimate_outfit_suggestions(len(p.get("script") or ""), n), \
            (None if n and (p.get("script") or "").strip() else "Write the script and create the cast first.")
    if kind == "breakdown":
        return pricing.estimate_breakdown(len(p["script"]), p["target_seconds"]), \
            (None if p["script"].strip() else "Write or paste a script first.")
    size = "2K" if p.get("image_quality") == "high" else "1K"
    if kind == "portrait" and (opts or {}).get("outfit_id"):
        views = target.get("reference_views") or {}
        if target.get("id") != (opts or {}).get("outfit_character"):
            return pricing.image_cost(1, 400, size), "That outfit belongs to another character."
        return pricing.image_cost(1, 400 + pricing.SHEET_REF_TOKENS, size), \
            (None if (views.get(ident.ANCHOR) or {}).get("path") else "Make the front view first.")
    if kind == "portrait" and (opts or {}).get("views"):
        order = ident.view_order(opts["views"])
        if not order:
            return pricing.image_cost(1, 400, size), "Choose which views to make."
        cost = pricing.estimate_sheet(len(order), size, not ident.has_identity(target.get("identity")))
        if target.get("identity_locked_at"):
            return cost, f"{target['name']}'s identity is locked. Unlock it first."
        has_front = ((target.get("reference_views") or {}).get(ident.ANCHOR) or {}).get("path")
        if ident.ANCHOR not in order and not has_front:
            return cost, "Make the front view first: every other view is made from it."
        return cost, None
    if kind == "portrait":
        return pricing.image_cost(1, 400, size), (None if target["description"].strip() else "Describe this character first.")
    if kind == "image":
        from studio_worker import characters_in_scene
        refs = sum(1 for c in characters_in_scene(target, cast) if c.get("reference_path"))
        return pricing.image_cost(1, 400 + 1300 * min(refs, 4), size), \
            (None if (target["visual"] or target["heading"]).strip() else "Describe what this scene shows first.")
    if kind == "voice":
        from studio_worker import scene_speech
        text = " ".join(d.get("text", "") for d in target.get("dialogue") or [])
        if text.strip() and scene_speech(target, p) == "video_model":
            return pricing.estimate_voice(text), "The characters speak this scene's lines in the video."
        return pricing.estimate_voice(text), (_voice_problem(target, cast) if text.strip() else "This scene has no lines.")
    if kind == "video":
        from studio_worker import speech_clip_seconds, talks_in_video
        if talks_in_video(dict(target, mode="video"), p):
            secs = speech_clip_seconds(target)
            if secs is None:
                return pricing.estimate_video(15), ("This scene has too much dialogue for one video clip "
                                                    "(15 seconds at most). Split it into two scenes.")
            return pricing.estimate_video(secs), None
        return pricing.estimate_video(target.get("duration_seconds") or 8.0), None
    if kind == "lipsync":
        from studio_worker import needs_lipsync, speech_seconds
        lines = [d for d in target.get("dialogue") or [] if (d.get("text") or "").strip()]
        secs = float(target.get("audio_seconds") or 0) + 0.5 if target.get("audio_seconds") else speech_seconds(lines)
        cost = studio_lipsync.estimate_lipsync(max(secs, float(target.get("duration_seconds") or 0)))
        if not studio_lipsync.enabled():
            return cost, "Lip sync isn't switched on yet."
        if not needs_lipsync(target, p):
            return cost, "This scene doesn't use lip sync."
        if not target.get("audio_path"):
            return cost, "Record this scene's voices first."
        if not (target.get("image_path") or (target.get("mode") == "video" and target.get("video_path"))):
            return cost, "Make this scene's video or picture first."
        return cost, None
    if kind == "render":
        scenes = get_store().select("studio_scenes", {"project_id": p["id"]})
        secs = sum(max(float(s["duration_seconds"]), float(s.get("audio_seconds") or 0) + 0.5) for s in scenes)
        missing = next((s["position"] + 1 for s in sorted(scenes, key=lambda s: s["position"])
                        if not (s.get("image_path") or (s["mode"] == "video" and s.get("video_path")))), None)
        reason = "Add scenes first." if not scenes else (f"Scene {missing} has no picture yet." if missing else None)
        from studio_worker import talks_in_video
        unspoken = next((s["position"] + 1 for s in sorted(scenes, key=lambda s: s["position"])
                         if talks_in_video(s, p) and not s.get("video_path")), None)
        if reason is None and unspoken:
            reason = f"Scene {unspoken}'s talking video isn't made yet."
        from studio_worker import needs_lipsync
        unsynced = next((s["position"] + 1 for s in sorted(scenes, key=lambda s: s["position"])
                         if needs_lipsync(s, p) and not s.get("lipsync_path")), None)
        if reason is None and unsynced:
            reason = f"Scene {unsynced}'s lip sync isn't made yet."
        return pricing.render_cost(secs), reason
    raise HTTPException(status_code=400, detail="Unknown job type.")


def _voice_problem(scene: dict, cast: list[dict]) -> Optional[str]:
    """Same checks the worker makes, before any credits are held."""
    from studio_worker import is_narrator, speaker_name
    by_name = {c["name"].strip().lower(): c for c in cast}
    for d in scene.get("dialogue") or []:
        if not (d.get("text") or "").strip():
            continue
        speaker = speaker_name(d.get("character") or "")
        if not speaker or is_narrator(speaker):
            continue
        ch = by_name.get(speaker.lower())
        if ch is None:
            return f"\"{speaker}\" speaks in this scene but isn't in your cast."
        if ch.get("voice"):
            if not _available_voice(ch["voice"]):
                return f"{ch['name']}'s voice isn't available. Choose a new voice for {ch['name']}."
            continue
        # No voice yet: would casting find one now? (Nothing is saved here.)
        from studio_worker import profile_from_row
        proj = get_store().one("studio_projects", {"id": ch["project_id"]}) or {}
        prof = profile_from_row(ch, proj)
        d = studio_casting.cast(prof, _catalog(), set(), "check", ch.get("description") or "")
        if d.status == "unsupported_language":
            return f"{ch['name']}'s lines can't be voiced in {prof.language} yet: the voice model doesn't support it."
        if not d.voice:
            who = studio_casting.AGE_LABEL.get(prof.age_category, "")
            return (f"{ch['name']} is a {who}, and Viyo doesn't have a suitable {who} voice yet. "
                    "Viyo won't use an adult voice instead.")
    return None


def _portrait_opts(views: list[str], outfit_id: Optional[str], user_id: str) -> dict:
    if outfit_id:
        o = _owned_outfit(outfit_id, user_id)
        return {"outfit_id": o["id"], "outfit_character": o["character_id"]}
    bad = [v for v in views if v not in ident.VIEWS]
    if bad:
        raise HTTPException(status_code=400, detail=f"Unknown view: {bad[0]}.")
    return {"views": ident.view_order(views)} if views else {}


def _quote(kind: str, p: dict, target_ids: list[str], opts: Optional[dict] = None) -> list[dict]:
    cast = get_store().select("studio_characters", {"project_id": p["id"]}) if kind in ("image", "voice") else []
    items = []
    for t in _targets(kind, p, target_ids):
        cost, reason = _price(kind, p, t, cast, opts)
        items.append({"target_id": None if kind in ("script", "breakdown", "render", "outfits", "cover", "elements") else t["id"],
                      "credits": pricing.hold_credits(kind, cost), "cost_usd": round(cost.total_usd, 5),
                      "blocked": reason})
    return items


@router.post("/projects/{project_id}/estimate")
async def estimate(project_id: str, body: EstimateRequest, user_id: str = Depends(creator)):
    p = _project(project_id, user_id)
    opts = _portrait_opts(body.views, body.outfit_id, user_id) if body.kind == "portrait" else None
    items = _quote(body.kind, p, body.target_ids, opts)
    ready = [i for i in items if not i["blocked"]]
    return {"kind": body.kind, "items": items, "total_credits": sum(i["credits"] for i in ready),
            "ready_count": len(ready), "balance": _balance(user_id),
            "note": "Text steps hold a little extra and return what isn't used." if body.kind in ("script", "breakdown", "outfits") else None}


@router.post("/projects/{project_id}/jobs")
async def start_jobs(project_id: str, body: JobRequest, user_id: str = Depends(creator)):
    p = _project(project_id, user_id)
    st = get_store()
    # Assistant-generated story plans must be explicitly approved by their
    # owner before they can be converted into cast and scenes. Enforce this
    # here, not only in the client, so direct API calls cannot bypass review.
    if body.kind == "breakdown":
        bible = p.get("story_bible") or {}
        if isinstance(bible, dict) and bible and not (bible.get("_assistant_approval") or {}).get("approved"):
            raise HTTPException(status_code=409, detail={
                "code": "story_approval_required",
                "message": "Review and approve the story before creating cast and scenes."
            })
    _rate_limit(user_id)
    if body.kind == "breakdown" and not body.confirm_replace:
        scenes = st.select("studio_scenes", {"project_id": p["id"]})
        if scenes:
            pics = sum(1 for s in scenes if s.get("image_path") or s.get("video_path"))
            voiced = sum(1 for s in scenes if s.get("audio_path"))
            raise HTTPException(status_code=409, detail={
                "code": "confirm_replace", "scenes": len(scenes), "with_pictures": pics, "with_voice": voiced,
                "message": (f"This production already has {len(scenes)} scene{'s' if len(scenes) != 1 else ''}"
                            f" ({pics} with pictures, {voiced} with voice). A new scene list keeps every scene that "
                            "hasn't changed, with its pictures and voice. Scenes that change are replaced, and the "
                            "old versions are saved, not deleted.")})
    created, problems = [], []
    opts = _portrait_opts(body.views, body.outfit_id, user_id) if body.kind == "portrait" else None
    job_input = {k: v for k, v in (opts or {}).items() if k in ("views", "outfit_id")}
    if body.kind == "cover" and (body.cover_note or "").strip():
        job_input = {"note": body.cover_note.strip()}
    for item in _quote(body.kind, p, body.target_ids, opts):
        if item["blocked"]:
            problems.append({"target_id": item["target_id"], "error": item["blocked"]})
            continue
        key = f"{user_id}:{body.request_id}:{body.kind}:{item['target_id'] or p['id']}"
        r = st.rpc("studio_create_job", {
            "p_owner": user_id, "p_project": p["id"], "p_kind": body.kind, "p_lane": LANE.get(body.kind, "generation"),
            "p_key": key, "p_credits": item["credits"],
            "p_input": {"element_id": item["target_id"]} if body.kind == "element" else job_input,
            "p_scene": item["target_id"] if body.kind in ("image", "voice", "video", "lipsync") else None,
            "p_character": item["target_id"] if body.kind == "portrait" else None, "p_max_open": 30})
        if r and r.get("ok"):
            created.append(r["job_id"])
        else:
            err = (r or {}).get("error", "failed")
            problems.append({"target_id": item["target_id"], "error": {
                "insufficient_credits": "Not enough credits.", "too_many_jobs": "Too many jobs running. Wait for some to finish.",
                "project_not_found": "Not found."}.get(err, err)})
            if err in ("insufficient_credits", "too_many_jobs"):
                break
    if not created and problems:
        status = 402 if any(x["error"] == "Not enough credits." for x in problems) else 400
        raise HTTPException(status_code=status, detail={"message": problems[0]["error"], "problems": problems,
                                                        "balance": _balance(user_id)})
    return {"job_ids": created, "problems": problems, "balance": _balance(user_id)}


@router.get("/projects/{project_id}/jobs")
async def project_jobs(project_id: str, user_id: str = Depends(creator)):
    p = _project(project_id, user_id)
    jobs = get_store().select("studio_jobs", {"project_id": p["id"]}, order="created_at", desc=True, limit=60)
    return {"jobs": [_job_out(j) for j in jobs], "balance": _balance(user_id)}


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str, user_id: str = Depends(creator)):
    r = get_store().rpc("studio_cancel_job", {"p_owner": user_id, "p_job_id": _uuid(job_id)}) or {}
    if not r.get("ok"):
        if r.get("error") == "already_started":
            raise HTTPException(status_code=409, detail="This has already started, so it can't be stopped.")
        raise HTTPException(status_code=404, detail="Not found.")
    return {"ok": True, "balance": _balance(user_id)}


# ---------------------------------------------------------------- music

MUSIC_MAX_BYTES = 25 * 1024 * 1024
MUSIC_MAX_SECONDS = 15 * 60


@router.post("/projects/{project_id}/music")
async def upload_music(project_id: str, file: UploadFile = File(...), rights_confirmed: bool = Form(False),
                       user_id: str = Depends(creator)):
    """The creator's own background track. Stored privately, re-encoded to
    AAC (which also drops file metadata), used under the whole film."""
    p = _project(project_id, user_id)
    if not rights_confirmed:
        raise HTTPException(status_code=400, detail="Confirm you have the rights to use this music.")
    _rate_limit(user_id)
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "upload")
        size = 0
        with open(src, "wb") as f:
            while chunk := await file.read(1 << 20):
                size += len(chunk)
                if size > MUSIC_MAX_BYTES:
                    raise HTTPException(status_code=413, detail="Music files can be up to 25 MB.")
                f.write(chunk)
        try:
            import studio_render
            info = studio_render.probe(src)
        except Exception:
            raise HTTPException(status_code=400, detail="That doesn't look like an audio file. Try MP3, M4A or WAV.")
        if info["duration"] <= 1 or info["duration"] > MUSIC_MAX_SECONDS:
            raise HTTPException(status_code=400, detail="Music must be between 1 second and 15 minutes long.")
        out = os.path.join(tmp, "music.m4a")
        r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src, "-vn", "-map_metadata", "-1", "-ac", "2",
                            "-ar", "48000", "-c:a", "aac", "-b:a", "192k", out], capture_output=True)
        if r.returncode != 0:
            raise HTTPException(status_code=400, detail="That doesn't look like an audio file. Try MP3, M4A or WAV.")
        from studio_store import new_path
        path = new_path(user_id, p["id"], "music", "m4a")
        get_files().put_file(path, out, "audio/mp4")
        get_store().insert("studio_assets", [{"owner_id": user_id, "project_id": p["id"], "kind": "music", "path": path,
                                              "bytes": os.path.getsize(out), "duration_seconds": round(info["duration"], 2)}])
    name = os.path.basename(file.filename or "music")[:80]
    get_store().update("studio_projects", {"id": p["id"]}, {"music_path": path, "music_name": name})
    _touch(p["id"])
    return {"ok": True, "music_name": name, "duration_seconds": round(info["duration"], 1)}


@router.delete("/projects/{project_id}/music")
async def remove_music(project_id: str, user_id: str = Depends(creator)):
    """Stops using the track. The file stays in the project's history."""
    p = _project(project_id, user_id)
    get_store().update("studio_projects", {"id": p["id"]}, {"music_path": None, "music_name": None})
    _touch(p["id"])
    return {"ok": True}


# ---------------------------------------------------------------- renders

@router.post("/renders/{render_id}/download")
async def download_render(render_id: str, user_id: str = Depends(creator)):
    r = _owned("studio_renders", render_id, user_id)
    p = get_store().one("studio_projects", {"id": r["project_id"]}) or {}
    slug = re.sub(r"[^a-z0-9]+", "-", (p.get("title") or "viyo-film").lower()).strip("-")[:50] or "viyo-film"
    name = f"{slug}-v{r['version']}.mp4"
    if r.get("bunny_video_id"):
        import bunny_stream
        try:
            info = bunny_info(r["bunny_video_id"])
        except bunny_stream.BunnyError:
            raise HTTPException(status_code=502, detail="Video storage didn't answer. Try again in a minute.")
        if bunny_stream.is_failed(info):
            raise HTTPException(status_code=409, detail="This render failed to process. Render it again.")
        if not bunny_stream.is_finished(info):
            raise HTTPException(status_code=409, detail="Your film is still being prepared. Try again in a minute or two.")
        return {"url": bunny_stream.source_download_url(r["bunny_video_id"], info), "file_name": name,
                "expires_at": None, "width": r.get("width"), "height": r.get("height")}
    url = get_files().sign([r["path"]], SIGN_SECONDS, download_name=name).get(r["path"])
    if not url:
        raise HTTPException(status_code=502, detail="Couldn't prepare the download. Try again.")
    return {"url": url, "file_name": name, "expires_at": int(time.time()) + SIGN_SECONDS,
            "width": r.get("width"), "height": r.get("height")}


# ---------------------------------------------------------------- places and objects

class ElementIn(BaseModel):
    kind: Literal["location", "prop"]
    name: str = Field(..., min_length=1, max_length=80)
    description: str = Field("", max_length=800)
    details: dict[str, str] = Field(default_factory=dict)


class ElementPatch(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=80)
    description: Optional[str] = Field(None, max_length=800)
    details: Optional[dict[str, str]] = None
    status: Optional[Literal["draft", "approved"]] = None


def _clean_details(d: dict) -> dict:
    return {k: str(v).strip()[:120] for k, v in (d or {}).items()
            if k in ("time_of_day", "mood", "owner") and str(v).strip()}


def _owned_element(element_id: str, user_id: str) -> dict:
    st = get_store()
    try:
        e = st.one("studio_elements", {"id": _uuid(element_id)})
    except HTTPException:
        raise
    except Exception:
        if hasattr(st, "conn"):
            st.conn.rollback()
        e = None
    if not e or e["owner_id"] != user_id:
        raise HTTPException(status_code=404, detail="Not found.")
    return e


@router.post("/projects/{project_id}/elements")
async def add_element(project_id: str, body: ElementIn, user_id: str = Depends(creator)):
    p = _project(project_id, user_id)
    if not body.name.strip():
        raise HTTPException(status_code=400, detail="Give it a name.")
    st = get_store()
    try:
        e = st.insert("studio_elements", [{"project_id": p["id"], "owner_id": user_id, "kind": body.kind,
                                           "name": body.name.strip(), "description": body.description.strip(),
                                           "details": _clean_details(body.details), "position": len(_elements(p["id"]))}])[0]
    except Exception:
        if hasattr(st, "conn"):
            st.conn.rollback()
        raise HTTPException(status_code=503, detail="Places and objects aren't ready yet. Run creator_studio_elements.sql.")
    _touch(p["id"])
    return _element_out(e, {})


@router.patch("/elements/{element_id}")
async def update_element(element_id: str, body: ElementPatch, user_id: str = Depends(creator)):
    e = _owned_element(element_id, user_id)
    raw = body.model_dump(exclude_none=True)
    fields = {k: v.strip() for k, v in raw.items() if isinstance(v, str) and k != "status"}
    if "name" in fields and not fields["name"]:
        raise HTTPException(status_code=400, detail="Give it a name.")
    if "details" in raw:
        fields["details"] = _clean_details(raw["details"])
    if "status" in raw:
        if raw["status"] == "approved" and not e.get("reference_path"):
            raise HTTPException(status_code=409, detail="Make its picture first.")
        fields["status"] = raw["status"]
    if fields:
        e = get_store().update("studio_elements", {"id": e["id"]}, {**fields, "updated_at": _now_iso()})[0]
    return _element_out(e, _sign([e.get("reference_path")]))


@router.delete("/elements/{element_id}")
async def delete_element(element_id: str, user_id: str = Depends(creator)):
    e = _owned_element(element_id, user_id)
    st = get_store()
    for s in st.select("studio_scenes", {"project_id": e["project_id"]}):
        fields = {}
        if s.get("location_id") == e["id"]:
            fields["location_id"] = None
        if isinstance(s.get("props"), list) and e["id"] in s["props"]:
            fields["props"] = [x for x in s["props"] if x != e["id"]]
        if fields:
            st.update("studio_scenes", {"id": s["id"]}, fields)
    st.delete("studio_elements", {"id": e["id"]})
    return {"ok": True}


# ---------------------------------------------------------------- one-click production

class ProductionQuoteIn(BaseModel):
    # Make a reference sheet for characters that have none (recommended),
    # instead of a single portrait.
    sheets: bool = True


class ProductionIn(ProductionQuoteIn):
    request_id: str = Field(..., min_length=8, max_length=80)
    # The total the creator saw; refused if the price changed since.
    confirm_credits: int = Field(..., ge=0)


def _production_quote(p: dict, body: ProductionQuoteIn) -> dict:
    import studio_production
    q = studio_production.plan(p, {"sheets": body.sheets})
    q["balance"] = _balance(p["owner_id"])
    q["note"] = ("Credits for the whole production are reserved when you start. Each step keeps only what it really "
                 "costs; failed steps are refunded and tried again once; what's left comes back at the end.")
    return q


def _latest_run(p: dict) -> Optional[dict]:
    try:
        runs = get_store().select("studio_production_runs", {"project_id": p["id"]}, order="created_at", desc=True, limit=1)
    except Exception:
        if hasattr(get_store(), "conn"):
            get_store().conn.rollback()
        return None
    return runs[0] if runs else None


def _production_out(p: dict) -> Optional[dict]:
    import studio_production
    run = _latest_run(p)
    try:
        return studio_production.progress(get_store(), run) if run else None
    except Exception as e:
        print(f"[WARN] studio production progress: {str(e)[:160]}")
        return None


def _owned_run(run_id: str, user_id: str) -> dict:
    try:
        run = get_store().one("studio_production_runs", {"id": _uuid(run_id)})
    except HTTPException:
        raise
    except Exception:
        if hasattr(get_store(), "conn"):
            get_store().conn.rollback()
        run = None
    if not run or run["owner_id"] != user_id:
        raise HTTPException(status_code=404, detail="Not found.")
    return run


@router.post("/projects/{project_id}/production/quote")
async def production_quote(project_id: str, body: ProductionQuoteIn, user_id: str = Depends(creator)):
    """Everything this production still needs to become a finished film, with
    the total price. Nothing is charged."""
    return _production_quote(_project(project_id, user_id), body)


@router.post("/projects/{project_id}/production")
async def start_production(project_id: str, body: ProductionIn, user_id: str = Depends(creator)):
    """Reserves the total and makes everything, in order, in the background."""
    import studio_production
    p = _project(project_id, user_id)
    _rate_limit(user_id)
    q = _production_quote(p, body)
    if q["blocked"]:
        raise HTTPException(status_code=409, detail={"code": "blocked", "message": q["blocked"][0], "blocked": q["blocked"]})
    if not q["stages"]:
        raise HTTPException(status_code=409, detail={"code": "nothing_to_do", "message": "Everything is already made."})
    if q["total_credits"] != body.confirm_credits:
        raise HTTPException(status_code=409, detail={"code": "price_changed", "message": "The price changed. Check it again.",
                                                     "total_credits": q["total_credits"]})
    plan_ = {"stages": [{"stage": s["stage"], "label": s["label"], "credits": s["credits"],
                         "items": [{"target_id": i["target_id"], "label": i["label"], "credits": i["credits"]}
                                   for i in s["items"]]} for s in q["stages"]]}
    try:
        r = get_store().rpc("studio_run_reserve", {
            "p_owner": user_id, "p_project": p["id"], "p_credits": q["total_credits"],
            "p_key": f"{user_id}:{body.request_id}:production", "p_options": {"sheets": body.sheets, "round": 0},
            "p_plan": plan_})
    except Exception as e:
        if hasattr(get_store(), "conn"):
            get_store().conn.rollback()
        print(f"[ERROR] studio production reserve: {str(e)[:200]}")
        raise HTTPException(status_code=503, detail="One-click production isn't ready yet. Run creator_studio_production.sql.")
    if not r or not r.get("ok"):
        err = (r or {}).get("error")
        if err == "insufficient_credits":
            raise HTTPException(status_code=402, detail={"message": "Not enough credits.", "balance": _balance(user_id),
                                                         "needed": q["total_credits"]})
        if err == "already_running":
            raise HTTPException(status_code=409, detail={"code": "already_running",
                                                         "message": "A production is already running for this project."})
        raise HTTPException(status_code=400, detail=err or "failed")
    run = get_store().one("studio_production_runs", {"id": r["run_id"]})
    return {"production": studio_production.progress(get_store(), run), "balance": _balance(user_id)}


@router.get("/projects/{project_id}/production")
async def production_status(project_id: str, user_id: str = Depends(creator)):
    import studio_production
    run = _latest_run(_project(project_id, user_id))
    return {"production": studio_production.progress(get_store(), run) if run else None}


@router.post("/production/{run_id}/resume")
async def resume_production(run_id: str, user_id: str = Depends(creator)):
    """After fixing what it asked for (or adding credits): carry on. Each step
    that failed gets one more try; nothing already made is redone."""
    import studio_production
    run = _owned_run(run_id, user_id)
    if run["status"] != "needs_attention":
        raise HTTPException(status_code=409, detail="This production isn't waiting for you.")
    opts = dict(run.get("options") or {})
    opts["round"] = int(opts.get("round") or 0) + 1
    get_store().update("studio_production_runs", {"id": run["id"]},
                       {"status": "running", "problem": None, "options": opts, "updated_at": _now_iso()})
    return {"production": studio_production.progress(get_store(), get_store().one("studio_production_runs", {"id": run["id"]}))}


@router.post("/production/{run_id}/cancel")
async def cancel_production(run_id: str, user_id: str = Depends(creator)):
    """Stops the run: steps not started yet are canceled and refunded, and the
    rest of the reservation comes back. Anything already made is kept."""
    import studio_production
    run = _owned_run(run_id, user_id)
    if run["status"] in ("done", "canceled"):
        raise HTTPException(status_code=409, detail="This production has already finished.")
    st = get_store()
    for j in studio_production._run_jobs(st, run):
        if j["status"] == "queued":
            st.rpc("studio_cancel_job", {"p_owner": user_id, "p_job_id": j["id"]})
    st.rpc("studio_run_finish", {"p_run": run["id"], "p_status": "canceled", "p_problem": None})
    return {"production": studio_production.progress(st, st.one("studio_production_runs", {"id": run["id"]})),
            "balance": _balance(user_id)}


# ---------------------------------------------------------------- publish

_publisher = None


def get_publisher():
    global _publisher
    if _publisher is None:
        import studio_publish
        _publisher = studio_publish.Publisher()
    return _publisher


class PublishIn(BaseModel):
    # 0 = free to watch. Otherwise the coin price per episode (the usual
    # first-episodes-free rule of Viyo Dramas applies to paid series).
    coin_price: int = Field(0, ge=0, le=500)
    caption: str = Field("", max_length=500)
    confirm: bool = False


def _viyo_series_for(p: dict) -> Optional[str]:
    """This production's Viyo drama: its own, or one an earlier episode of the
    same Studio series was published into."""
    if p.get("viyo_series_id"):
        return p["viyo_series_id"]
    if p.get("series_id"):
        for e in get_store().select("studio_projects", {"series_id": p["series_id"], "owner_id": p["owner_id"]}):
            if e.get("viyo_series_id"):
                return e["viyo_series_id"]
    return None


@router.post("/renders/{render_id}/publish")
async def publish_render(render_id: str, body: PublishIn, user_id: str = Depends(creator)):
    """Posts a finished film to Viyo under the creator's account: an episode
    of their Viyo drama (made on first publish), live at once, on the
    drama's page and in the feed. The main cover is the drama's poster."""
    import studio_publish
    r = _owned("studio_renders", render_id, user_id)
    if "post_id" not in r:
        raise HTTPException(status_code=503, detail="Publishing isn't ready yet. Run creator_studio_publish.sql.")
    if r.get("post_id"):
        raise HTTPException(status_code=409, detail="This version is already published.")
    if not body.confirm:
        raise HTTPException(status_code=400, detail="Confirm that you want to publish this to Viyo.")
    p = _project(r["project_id"], user_id)
    pub = get_publisher()
    files = get_files()
    # The video: already in Bunny, or uploaded there now.
    video_id = r.get("bunny_video_id")
    if not video_id:
        if not r.get("path"):
            raise HTTPException(status_code=409, detail="This render has no video file. Render it again.")
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            local = os.path.join(tmp, "film.mp4")
            with open(local, "wb") as f:
                f.write(files.get(r["path"]))
            try:
                video_id = pub.bunny_upload(local, p["title"] or "Viyo")
            except Exception as e:
                print(f"[ERROR] studio publish {render_id}: video upload failed: {str(e)[:200]}")
                raise HTTPException(status_code=502, detail="The video couldn't be uploaded. Try again in a minute.")
    # The main cover becomes a public poster.
    cover_url = None
    if p.get("cover_path"):
        try:
            cover_url = pub.public_image(files.get(p["cover_path"]), "image/png", user_id)
        except Exception as e:
            print(f"[WARN] studio publish {render_id}: cover not published: {str(e)[:160]}")
    series_id = _viyo_series_for(p)
    series = pub.series(series_id) if series_id else None
    if series and series.get("user_id") != user_id:
        series = None
    if series is None:
        series = pub.create_series({
            "user_id": user_id, "title": (p["title"] or "Untitled")[:120], "description": (p.get("logline") or "")[:2000],
            "cover_image_url": cover_url, "genre": p.get("genre") or "Drama",
            "coin_price_per_episode": body.coin_price or 30,
            "content_type": studio_publish.content_type_for(p.get("kind")),
            "orientation": studio_publish.orientation_for(p.get("aspect_ratio")),
        })
    elif cover_url:
        pub.update_series(series["id"], {"cover_image_url": cover_url})
    if body.coin_price and series.get("coin_price_per_episode") != body.coin_price:
        pub.update_series(series["id"], {"coin_price_per_episode": body.coin_price})
    episode = pub.last_episode(series["id"]) + 1
    row = {"user_id": user_id, "post_type": "video",
           "caption": body.caption.strip() or f"{series['title']} - Episode {episode}",
           "media_url": pub.bunny_playback(video_id), "thumbnail_url": cover_url,
           "duration_seconds": int(float(r.get("duration_seconds") or 0)) or None, "is_private": False,
           "series_id": series["id"], "episode_number": episode,
           "video_provider": "bunny", "bunny_video_id": video_id, "video_status": "processing"}
    if body.coin_price == 0:
        row["is_free"] = True
    post = pub.insert_post(row)
    st = get_store()
    st.update("studio_renders", {"id": r["id"]}, {"post_id": post["id"], "published_at": _now_iso()})
    st.update("studio_projects", {"id": p["id"]}, {"viyo_series_id": series["id"], "updated_at": _now_iso()})
    pub.after_post(post, user_id, series["id"], episode)
    return {"post_id": post["id"], "series_id": series["id"], "episode_number": episode, "cover_url": cover_url,
            "free": body.coin_price == 0}


def bunny_info(video_id: str) -> dict:
    import bunny_stream
    return bunny_stream.get_video_info(video_id)


# ---------------------------------------------------------------- admin

@router.post("/admin/credits/grant", dependencies=[Depends(_require_admin)])
async def admin_grant(body: GrantIn):
    """Adds credits by hand: a promotional grant, or a purchase paid outside
    the app (mobile money, PayPal) until card checkout is switched on."""
    st = get_store()
    target = body.user.strip()
    uid = None
    try:
        uid = str(uuid.UUID(target))
    except ValueError:
        name = target.lstrip("@")
        rows = st.select("profiles", {"username": name}, limit=1)
        uid = rows[0]["id"] if rows else None
    if not uid:
        raise HTTPException(status_code=404, detail="No account found for that user.")
    import datetime
    expires = (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=body.expires_days)).isoformat() \
        if body.expires_days and body.kind == "grant" else None
    key = f"admin:{body.kind}:{body.reference or uuid.uuid4().hex}"
    r = st.rpc("studio_add_credits", {
        "p_user_id": uid, "p_credits": body.credits, "p_source": body.kind, "p_key": key,
        "p_note": body.note or ("Studio credits" if body.kind == "purchase" else "Promotional credits"),
        "p_expires_at": expires, "p_provider": "manual" if body.kind == "purchase" else None,
        "p_order_id": body.reference or None, "p_gross_usd": body.amount_usd, "p_fee_usd": body.fee_usd})
    if not r or not r.get("ok"):
        raise HTTPException(status_code=400, detail=(r or {}).get("error", "Could not add credits."))
    return {"ok": True, "user_id": uid, "duplicate": bool(r.get("duplicate")), "balance": _balance(uid)}
