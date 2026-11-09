"""One-click production: everything a production still needs, in order,
under one confirmed price.

Stages, each waiting for the one before it:
  sheets  -> each character's reference sheet (or a portrait, if sheets are off)
  images  -> a picture for every scene that has none (the first frame of its video)
  voices  -> saved-voice recordings for scenes whose lines aren't spoken in the video
  videos  -> a clip for every video scene
  lipsync -> lip sync for scenes set to it (only when lip sync is switched on)
  render  -> the finished film

The plan is always worked out from the production as it is now, so anything
already made (by hand or by an earlier run) is never made or paid for twice,
and a run can be resumed at any point. Each step is an ordinary Studio job
(the same queue, prices, holds, refunds and retry rules as a job started by
hand); its credits come out of the run's reservation. A step that fails is
refunded and tried again once; if it fails again the run pauses and tells
the creator which scenes need them. The film is only assembled once every
scene is ready.
"""
from __future__ import annotations

import json
from typing import Optional

import studio_identity as ident
import studio_pricing as pricing

STAGES = ("sheets", "images", "voices", "videos", "lipsync", "render")
STAGE_KIND = {"sheets": "portrait", "images": "image", "voices": "voice", "videos": "video",
              "lipsync": "lipsync", "render": "render"}
STAGE_LABEL = {"sheets": "Character sheets", "images": "Scene pictures", "voices": "Voices", "videos": "Video clips",
               "lipsync": "Lip sync", "render": "Final film"}
ATTEMPTS = 2          # each step: the first try and one automatic retry
# Reasons a step can't run *yet* that an earlier stage of the run takes care of.
LATER = ("has no picture yet", "isn't made yet", "Record this scene's voices first")
LANE = {"render": "render"}


def _cs():
    import creator_studio
    return creator_studio


# ------------------------------------------------------------------ plan

def plan(p: dict, options: Optional[dict] = None) -> dict:
    """What is still needed, stage by stage, with prices; and what blocks it."""
    from studio_worker import needs_lipsync, scene_speech
    cs = _cs()
    st = cs.get_store()
    options = {"sheets": True, **(options or {})}
    cast = sorted(st.select("studio_characters", {"project_id": p["id"]}), key=lambda c: c.get("position") or 0)
    scenes = sorted(st.select("studio_scenes", {"project_id": p["id"]}), key=lambda s: s["position"])
    stages: dict[str, list] = {s: [] for s in STAGES}
    blocked: list[str] = []

    def add(stage, target, label, opts=None):
        cost, reason = cs._price(STAGE_KIND[stage], p, target, cast, opts)
        if reason and any(m in reason for m in LATER):
            reason = None          # an earlier stage of this run makes it
        if reason:
            blocked.append(f"{label}: {reason}")
            return
        stages[stage].append({"target_id": target["id"], "label": label, "credits": pricing.hold_credits(STAGE_KIND[stage], cost),
                              "input": opts or {}})

    for c in cast:
        has_front = ((c.get("reference_views") or {}).get(ident.ANCHOR) or {}).get("path")
        if options["sheets"] and not has_front and not c.get("identity_locked_at"):
            add("sheets", c, f"{c['name']}'s reference sheet", {"views": list(ident.VIEWS)})
        elif not options["sheets"] and not c.get("reference_path") and (c.get("description") or "").strip():
            add("sheets", c, f"{c['name']}'s portrait")
    for i, s in enumerate(scenes, start=1):
        if not s.get("image_path"):
            add("images", s, f"Scene {i} picture")
        lines = [d for d in s.get("dialogue") or [] if (d.get("text") or "").strip()]
        if lines and not s.get("audio_path") and scene_speech(s, p) != "video_model":
            add("voices", s, f"Scene {i} voices")
        if s.get("mode") == "video" and not s.get("video_path"):
            add("videos", s, f"Scene {i} video")
        if needs_lipsync(s, p) and not s.get("lipsync_path"):
            try:
                import studio_lipsync
                if studio_lipsync.enabled():
                    add("lipsync", s, f"Scene {i} lip sync")
            except Exception:
                pass
    if not scenes:
        blocked.append("Break the script into scenes first.")
    if options.get("render", True) and scenes:
        add("render", p, "The finished film")
    out = [{"stage": s, "label": STAGE_LABEL[s], "items": stages[s], "credits": sum(i["credits"] for i in stages[s])}
           for s in STAGES if stages[s]]
    return {"stages": out, "total_credits": sum(x["credits"] for x in out), "blocked": blocked,
            "scenes": len(scenes), "characters": len(cast), "options": options}


# ------------------------------------------------------------------ advance

def _round(run: dict) -> int:
    opts = run.get("options") or {}
    if isinstance(opts, str):
        opts = json.loads(opts)
    return int(opts.get("round") or 0)


def _run_jobs(st, run: dict) -> list:
    prefix = f"run:{run['id']}:"
    return [j for j in st.select("studio_jobs", {"project_id": run["project_id"]})
            if (j.get("idempotency_key") or "").startswith(prefix)]


def _stage_of(job: dict) -> str:
    return job["idempotency_key"].split(":")[2]


def advance(st, run: dict) -> bool:
    """Moves one run forward. Returns True if it started any step."""
    if run["status"] != "running":
        return False
    p = st.one("studio_projects", {"id": run["project_id"]})
    if not p:
        st.rpc("studio_run_finish", {"p_run": run["id"], "p_status": "canceled", "p_problem": "The production was deleted."})
        return False
    options = run.get("options") or {}
    if isinstance(options, str):
        options = json.loads(options)
    rnd = _round(run)
    jobs = _run_jobs(st, run)
    now = plan(p, options)
    needed = {x["stage"]: x["items"] for x in now["stages"]}
    for stage in STAGES:
        mine = [j for j in jobs if _stage_of(j) == stage]
        if any(j["status"] in ("queued", "running") for j in mine):
            return False                                  # wait for this stage
        if stage == "render":
            done = [j for j in mine if j["status"] == "succeeded"]
            if done:
                render = next(iter(st.select("studio_renders", {"job_id": done[-1]["id"]})), None)
                st.update("studio_production_runs", {"id": run["id"]},
                          {"render_id": (render or {}).get("id")})
                st.rpc("studio_run_finish", {"p_run": run["id"], "p_status": "done", "p_problem": None})
                return False
        items = needed.get(stage) or []
        if not items:
            continue
        stuck, started = [], False
        for item in items:
            tries = [j for j in mine if j["idempotency_key"].split(":")[3] == item["target_id"]
                     and j["idempotency_key"].split(":")[4] == str(rnd)]
            if len(tries) >= ATTEMPTS:
                last = sorted(tries, key=lambda j: j.get("created_at") or "")[-1]
                stuck.append(f"{item['label']}: {last.get('error_message') or 'it failed twice'}")
                continue
            key = f"run:{run['id']}:{stage}:{item['target_id']}:{rnd}:{len(tries)}"
            kind = STAGE_KIND[stage]
            r = st.rpc("studio_run_create_job", {
                "p_run": run["id"], "p_kind": kind, "p_lane": LANE.get(stage, "generation"), "p_key": key,
                "p_credits": item["credits"], "p_input": item["input"],
                "p_scene": item["target_id"] if kind in ("image", "voice", "video", "lipsync") else None,
                "p_character": item["target_id"] if kind == "portrait" else None,
                # The first try of a step draws from the reservation; retries
                # (in this round or after a resume) hold from the balance,
                # where the failed try's refund went.
                "p_from_reserve": not any(j["idempotency_key"].split(":")[3] == item["target_id"] for j in mine)})
            if r and r.get("ok"):
                started = True
            else:
                err = (r or {}).get("error")
                msg = {"insufficient_credits": "Not enough credits left to continue. Add credits, then resume."}.get(
                    err, f"Couldn't start {item['label'].lower()} ({err}).")
                st.rpc("studio_run_finish", {"p_run": run["id"], "p_status": "needs_attention", "p_problem": msg})
                return started
        if started:
            return True
        if stuck:
            st.rpc("studio_run_finish", {"p_run": run["id"], "p_status": "needs_attention",
                                         "p_problem": "These need you: " + " · ".join(stuck[:6])})
            return False
    # Everything is made but no render was asked for.
    st.rpc("studio_run_finish", {"p_run": run["id"], "p_status": "done", "p_problem": None})
    return False


def advance_all(st) -> bool:
    """Called by the generation worker when it has nothing else to do."""
    try:
        runs = st.select("studio_production_runs", {"status": "running"})
    except Exception:
        if hasattr(st, "conn"):
            st.conn.rollback()
        return False          # creator_studio_production.sql not run yet
    started = False
    for run in runs:
        try:
            started = advance(st, run) or started
        except Exception as e:
            if hasattr(st, "conn"):
                st.conn.rollback()
            print(f"[WARN] studio production {run['id']}: {str(e)[:200]}")
    return started


# ------------------------------------------------------------------ progress

def progress(st, run: dict) -> dict:
    plan_ = run.get("plan") or {}
    if isinstance(plan_, str):
        plan_ = json.loads(plan_)
    jobs = _run_jobs(st, run)
    p = st.one("studio_projects", {"id": run["project_id"]}) or {}
    try:
        now = {x["stage"]: len(x["items"]) for x in plan(p, run.get("options") or {})["stages"]} if p else {}
    except Exception:
        now = {}
    stages = []
    for x in plan_.get("stages") or []:
        mine = [j for j in jobs if _stage_of(j) == x["stage"]]
        total = len(x["items"])
        remaining = min(total, now.get(x["stage"], 0)) if x["stage"] != "render" else (
            0 if any(j["status"] == "succeeded" for j in mine) else 1)
        stages.append({"stage": x["stage"], "label": x["label"], "total": total, "done": total - remaining,
                       "working": sum(1 for j in mine if j["status"] in ("queued", "running")),
                       "failed": sum(1 for j in mine if j["status"] == "failed")})
    total = sum(s["total"] for s in stages) or 1
    done = sum(s["done"] for s in stages)
    held = 0
    if run.get("hold_id"):
        h = st.one("studio_credit_holds", {"id": run["hold_id"]})
        if h and h.get("status") == "open":
            held = int(h["credits_held"]) - int(h.get("credits_captured") or 0)
    return {"id": run["id"], "status": run["status"], "problem": run.get("problem"),
            "quoted_credits": int(run.get("quoted_credits") or 0),
            "spent_credits": sum(int(j.get("charged_credits") or 0) for j in jobs),
            "reserved_left": held, "percent": round(100 * done / total) if run["status"] != "done" else 100,
            "stages": stages, "render_id": run.get("render_id"), "created_at": run.get("created_at"),
            "finished_at": run.get("finished_at")}
