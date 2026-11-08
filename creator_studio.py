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
import studio_lipsync
import studio_sfx
import studio_pricing as pricing
from studio_store import Files, Store, SupabaseFiles, SupabaseStore

router = APIRouter(prefix="/api/v1/studio", tags=["creator_studio"])

SIGN_SECONDS = 3600
JOB_KINDS = ("script", "breakdown", "portrait", "image", "voice", "video", "render")
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


def _clean_sfx(items: list[str]) -> list[str]:
    bad = [x for x in items if x not in studio_sfx.EFFECTS]
    if bad:
        raise HTTPException(status_code=400, detail=f"Unknown sound effect: {bad[0]}")
    return list(dict.fromkeys(items))


class OrderIn(BaseModel):
    scene_ids: list[str] = Field(..., max_length=200)


class JobRequest(BaseModel):
    kind: Literal["script", "breakdown", "portrait", "image", "voice", "video", "render", "lipsync"]
    target_ids: list[str] = Field(default_factory=list, max_length=100)
    request_id: str = Field(..., min_length=8, max_length=80)
    # A new scene list for a project that already has scenes must be
    # confirmed: unchanged scenes keep their work, changed ones are set
    # aside (never deleted).
    confirm_replace: bool = False


class EstimateRequest(BaseModel):
    kind: Literal["script", "breakdown", "portrait", "image", "voice", "video", "render", "lipsync"]
    target_ids: list[str] = Field(default_factory=list, max_length=100)


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

def _character_out(c: dict, urls: dict) -> dict:
    return {"id": c["id"], "name": c["name"], "description": c["description"], "voice": c["voice"],
            "voice_provider": c.get("voice_provider") or "google", "voice_style": c.get("voice_style") or "",
            "voice_language": c.get("voice_language"), "voice_locked": bool(c.get("voice_locked_at")),
            "voice_changed_at": c.get("voice_changed_at"), "identity_id": c.get("identity_id") or c["id"],
            "age_category": c.get("age_category"), "age_years": c.get("age_years"),
            "age_confidence": c.get("age_confidence"), "age_label": studio_casting.AGE_LABEL.get(c.get("age_category") or ""),
            "gender_presentation": c.get("gender_presentation"), "voice_category": c.get("voice_category"),
            "voice_description": c.get("voice_description") or "", "voice_status": c.get("voice_status") or
            ("cast" if c.get("voice") else "needs_voice"),
            "portrait_url": urls.get(c.get("reference_path") or ""), "position": c.get("position", 0)}


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


def _scene_out(s: dict, urls: dict, cast: Optional[dict] = None, project: Optional[dict] = None) -> dict:
    return {"id": s["id"], "position": s["position"], "heading": s["heading"], "visual": s["visual"],
            "dialogue": s.get("dialogue") or [], "duration_seconds": float(s["duration_seconds"]), "mode": s["mode"],
            "image_url": urls.get(s.get("image_path") or ""), "video_url": urls.get(s.get("video_path") or ""),
            "audio_url": urls.get(s.get("audio_path") or ""),
            "audio_seconds": float(s["audio_seconds"]) if s.get("audio_seconds") else None,
            "has_image": bool(s.get("image_path")), "has_video": bool(s.get("video_path")),
            "has_audio": bool(s.get("audio_path")), "sfx": s.get("sfx") or [],
            "voice_outdated": _outdated_voices(s, cast),
            "speech": _speech(s, project), "speech_setting": s.get("speech"),
            "lipsync_url": urls.get(s.get("lipsync_path") or ""), "has_lipsync": bool(s.get("lipsync_path"))}


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
            "bytes": r.get("bytes"), "created_at": r.get("created_at")}


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
    urls = _sign([v[1] for v in first_image.values()])
    return {"balance": _balance(user_id), "projects": [{
        **{k: p[k] for k in ("id", "title", "logline", "kind", "aspect_ratio", "genre", "updated_at")},
        "scene_count": counts[p["id"]][0], "scenes_with_images": counts[p["id"]][1],
        "render_count": render_count[p["id"]],
        "thumbnail_url": urls.get(first_image.get(p["id"], (0, ""))[1])} for p in projects]}


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
        copies.append(copy)
    if copies:
        st.insert("studio_characters", copies)
    return _project_out(new)


@router.get("/projects/{project_id}")
async def get_project(project_id: str, user_id: str = Depends(creator)):
    p = _project(project_id, user_id)
    st = get_store()
    chars = sorted(st.select("studio_characters", {"project_id": p["id"]}), key=lambda c: (c.get("position") or 0, c["name"]))
    scenes = sorted(st.select("studio_scenes", {"project_id": p["id"]}), key=lambda s: s["position"])
    renders = st.select("studio_renders", {"project_id": p["id"]}, order="version", desc=True, limit=20)
    jobs = st.select("studio_jobs", {"project_id": p["id"]}, order="created_at", desc=True, limit=60)
    urls = _sign([c.get("reference_path") for c in chars] +
                 [s.get(k) for s in scenes for k in ("image_path", "video_path", "audio_path")])
    return {"project": _project_out(p), "characters": [_character_out(c, urls) for c in chars],
            "scenes": [_scene_out(s, urls, {c["id"]: c for c in chars}, p) for s in scenes],
            "renders": [_render_out(r) for r in renders],
            "jobs": [_job_out(j) for j in jobs], "balance": _balance(user_id)}


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
    out = _character_out(c, _sign([c.get("reference_path")]))
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
    if kind in ("script", "breakdown", "render"):
        return [p]
    table = "studio_characters" if kind == "portrait" else "studio_scenes"
    rows = st.select(table, {"project_id": p["id"]})
    if target_ids:
        wanted = {_uuid(t) for t in target_ids}
        rows = [r for r in rows if r["id"] in wanted]
        if len(rows) != len(wanted):
            raise HTTPException(status_code=404, detail="Some of those items don't exist.")
    return sorted(rows, key=lambda r: r.get("position") or 0)


def _price(kind: str, p: dict, target: dict, cast: list[dict]) -> tuple[pricing.Cost, Optional[str]]:
    """(cost, reason it can't be made yet)."""
    if kind == "script":
        return pricing.estimate_script(p["target_seconds"], len(p["title"]) + len(p["logline"])), \
            (None if p["logline"].strip() else "Describe your idea first.")
    if kind == "breakdown":
        return pricing.estimate_breakdown(len(p["script"]), p["target_seconds"]), \
            (None if p["script"].strip() else "Write or paste a script first.")
    size = "2K" if p.get("image_quality") == "high" else "1K"
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


def _quote(kind: str, p: dict, target_ids: list[str]) -> list[dict]:
    cast = get_store().select("studio_characters", {"project_id": p["id"]}) if kind in ("image", "voice") else []
    items = []
    for t in _targets(kind, p, target_ids):
        cost, reason = _price(kind, p, t, cast)
        items.append({"target_id": None if kind in ("script", "breakdown", "render") else t["id"],
                      "credits": pricing.hold_credits(kind, cost), "cost_usd": round(cost.total_usd, 5),
                      "blocked": reason})
    return items


@router.post("/projects/{project_id}/estimate")
async def estimate(project_id: str, body: EstimateRequest, user_id: str = Depends(creator)):
    p = _project(project_id, user_id)
    items = _quote(body.kind, p, body.target_ids)
    ready = [i for i in items if not i["blocked"]]
    return {"kind": body.kind, "items": items, "total_credits": sum(i["credits"] for i in ready),
            "ready_count": len(ready), "balance": _balance(user_id),
            "note": "Text steps hold a little extra and return what isn't used." if body.kind in ("script", "breakdown") else None}


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
    for item in _quote(body.kind, p, body.target_ids):
        if item["blocked"]:
            problems.append({"target_id": item["target_id"], "error": item["blocked"]})
            continue
        key = f"{user_id}:{body.request_id}:{body.kind}:{item['target_id'] or p['id']}"
        r = st.rpc("studio_create_job", {
            "p_owner": user_id, "p_project": p["id"], "p_kind": body.kind, "p_lane": LANE.get(body.kind, "generation"),
            "p_key": key, "p_credits": item["credits"], "p_input": {},
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
