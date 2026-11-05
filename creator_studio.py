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
import time
import uuid
from collections import defaultdict, deque
from typing import Literal, Optional

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field

import studio_ai
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


class CharacterIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=60)
    description: str = Field("", max_length=1200)
    voice: str = ""


class CharacterPatch(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=60)
    description: Optional[str] = Field(None, max_length=1200)
    voice: Optional[str] = None


class Line(BaseModel):
    character: str = Field("", max_length=60)
    text: str = Field(..., max_length=600)


class SceneIn(BaseModel):
    heading: str = Field("", max_length=200)
    visual: str = Field("", max_length=2000)
    dialogue: list[Line] = Field(default_factory=list, max_length=12)
    duration_seconds: float = Field(5, ge=2, le=20)
    mode: Literal["still", "video"] = "still"
    position: Optional[int] = None
    sfx: list[str] = Field(default_factory=list, max_length=4)


class ScenePatch(BaseModel):
    heading: Optional[str] = Field(None, max_length=200)
    visual: Optional[str] = Field(None, max_length=2000)
    dialogue: Optional[list[Line]] = Field(None, max_length=12)
    duration_seconds: Optional[float] = Field(None, ge=2, le=20)
    mode: Optional[Literal["still", "video"]] = None
    sfx: Optional[list[str]] = Field(None, max_length=4)


def _clean_sfx(items: list[str]) -> list[str]:
    bad = [x for x in items if x not in studio_sfx.EFFECTS]
    if bad:
        raise HTTPException(status_code=400, detail=f"Unknown sound effect: {bad[0]}")
    return list(dict.fromkeys(items))


class OrderIn(BaseModel):
    scene_ids: list[str] = Field(..., max_length=200)


class JobRequest(BaseModel):
    kind: Literal["script", "breakdown", "portrait", "image", "voice", "video", "render"]
    target_ids: list[str] = Field(default_factory=list, max_length=100)
    request_id: str = Field(..., min_length=8, max_length=80)


class EstimateRequest(BaseModel):
    kind: Literal["script", "breakdown", "portrait", "image", "voice", "video", "render"]
    target_ids: list[str] = Field(default_factory=list, max_length=100)


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
            "portrait_url": urls.get(c.get("reference_path") or ""), "position": c.get("position", 0)}


def _scene_out(s: dict, urls: dict) -> dict:
    return {"id": s["id"], "position": s["position"], "heading": s["heading"], "visual": s["visual"],
            "dialogue": s.get("dialogue") or [], "duration_seconds": float(s["duration_seconds"]), "mode": s["mode"],
            "image_url": urls.get(s.get("image_path") or ""), "video_url": urls.get(s.get("video_path") or ""),
            "audio_url": urls.get(s.get("audio_path") or ""),
            "audio_seconds": float(s["audio_seconds"]) if s.get("audio_seconds") else None,
            "has_image": bool(s.get("image_path")), "has_video": bool(s.get("video_path")),
            "has_audio": bool(s.get("audio_path")), "sfx": s.get("sfx") or []}


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
            "captions", "script", "created_at", "updated_at", "music_name")
    out = {k: p.get(k) for k in keys}
    out.update(has_music=bool(p.get("music_path")), music_volume=float(p.get("music_volume") or 0.25),
               image_quality=p.get("image_quality") or "standard")
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
        "price_checked": pricing.PRICE_CHECKED,
    }


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
    created = get_store().insert("studio_projects", [row])[0]
    return _project_out(created)


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
            "scenes": [_scene_out(s, urls) for s in scenes], "renders": [_render_out(r) for r in renders],
            "jobs": [_job_out(j) for j in jobs], "balance": _balance(user_id)}


@router.patch("/projects/{project_id}")
async def update_project(project_id: str, body: ProjectPatch, user_id: str = Depends(creator)):
    p = _project(project_id, user_id)
    fields = body.model_dump(exclude_none=True)
    if fields:
        import datetime
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
    voice = body.voice if body.voice in studio_ai.VOICE_NAMES else ""
    row = get_store().insert("studio_characters", [{"project_id": p["id"], "owner_id": user_id, "name": body.name.strip(),
                                                    "description": body.description, "voice": voice,
                                                    "position": 999}])[0]
    _touch(p["id"])
    return _character_out(row, {})


@router.patch("/characters/{character_id}")
async def update_character(character_id: str, body: CharacterPatch, user_id: str = Depends(creator)):
    c = _owned("studio_characters", character_id, user_id)
    fields = body.model_dump(exclude_none=True)
    if "voice" in fields and fields["voice"] not in studio_ai.VOICE_NAMES:
        raise HTTPException(status_code=400, detail="Unknown voice.")
    if fields:
        c = get_store().update("studio_characters", {"id": c["id"]}, fields)[0]
    return _character_out(c, _sign([c.get("reference_path")]))


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
                                       "duration_seconds": body.duration_seconds, "mode": body.mode,
                                       "sfx": _clean_sfx(body.sfx)}])[0]
    _touch(p["id"])
    return _scene_out(row, {})


@router.patch("/scenes/{scene_id}")
async def update_scene(scene_id: str, body: ScenePatch, user_id: str = Depends(creator)):
    s = _owned("studio_scenes", scene_id, user_id)
    fields = body.model_dump(exclude_none=True)
    if "sfx" in fields:
        fields["sfx"] = _clean_sfx(fields["sfx"])
    if "dialogue" in fields:
        fields["dialogue"] = [l.model_dump() for l in body.dialogue]
        # New lines need a new voice track; the old file stays in the asset history.
        if fields["dialogue"] != s.get("dialogue"):
            fields.update(audio_path=None, audio_seconds=None)
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
        text = " ".join(d.get("text", "") for d in target.get("dialogue") or [])
        return pricing.estimate_voice(text), (None if text.strip() else "This scene has no lines.")
    if kind == "video":
        return pricing.estimate_video(8.0), None
    if kind == "render":
        scenes = get_store().select("studio_scenes", {"project_id": p["id"]})
        secs = sum(max(float(s["duration_seconds"]), float(s.get("audio_seconds") or 0) + 0.5) for s in scenes)
        missing = next((s["position"] + 1 for s in sorted(scenes, key=lambda s: s["position"])
                        if not (s.get("image_path") or (s["mode"] == "video" and s.get("video_path")))), None)
        reason = "Add scenes first." if not scenes else (f"Scene {missing} has no picture yet." if missing else None)
        return pricing.render_cost(secs), reason
    raise HTTPException(status_code=400, detail="Unknown job type.")


def _quote(kind: str, p: dict, target_ids: list[str]) -> list[dict]:
    cast = get_store().select("studio_characters", {"project_id": p["id"]}) if kind == "image" else []
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
    _rate_limit(user_id)
    st = get_store()
    created, problems = [], []
    for item in _quote(body.kind, p, body.target_ids):
        if item["blocked"]:
            problems.append({"target_id": item["target_id"], "error": item["blocked"]})
            continue
        key = f"{user_id}:{body.request_id}:{body.kind}:{item['target_id'] or p['id']}"
        r = st.rpc("studio_create_job", {
            "p_owner": user_id, "p_project": p["id"], "p_kind": body.kind, "p_lane": LANE.get(body.kind, "generation"),
            "p_key": key, "p_credits": item["credits"], "p_input": {},
            "p_scene": item["target_id"] if body.kind in ("image", "voice", "video") else None,
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
