"""One-click production: everything a production still needs, in order,
under one confirmed price.

Stages, each waiting for the one before it:
  places  -> a reference picture for each place and object (studio_elements)
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

STAGES = ("places", "sheets", "direct", "images", "voices", "videos", "lipsync", "render")
STAGE_KIND = {"places": "element", "sheets": "portrait", "direct": "shotplan", "images": "image", "voices": "voice",
              "videos": "video", "lipsync": "lipsync", "render": "render"}
STAGE_LABEL = {"places": "Places and objects", "sheets": "Character sheets", "direct": "Directing (camera shots)",
               "images": "Pictures", "voices": "Voices", "videos": "Video clips", "lipsync": "Lip sync", "render": "Final film"}
ATTEMPTS = 2          # each step: the first try and one automatic retry
# Reasons a step can't run *yet* that an earlier stage of the run takes care of.
LATER = ("has no picture yet", "isn't made yet", "Record this scene's voices first", "shot's picture first")
LANE = {"render": "render"}
VIDEO_STAGES = ("videos", "lipsync")      # paid video: waits for the storyboard review when it's on
STORYBOARD_NOTE = ("Your storyboard is ready. Check the characters, framing, actions, camera directions and scene "
                   "details of every picture. Fix anything that's wrong, then approve it to start the video clips.")
WAITING = ("running", "needs_attention", "needs_approval", "storyboard_review")


def _cs():
    import creator_studio
    return creator_studio


# ------------------------------------------------------------------ plan

def plan(p: dict, options: Optional[dict] = None) -> dict:
    """What is still needed, stage by stage, with prices; and what blocks it."""
    from studio_worker import needs_lipsync, scene_speech
    cs = _cs()
    st = cs.get_store()
    options = {"sheets": True, "direct": True, **(options or {})}
    cast = sorted(st.select("studio_characters", {"project_id": p["id"]}), key=lambda c: c.get("position") or 0)
    scenes = sorted(st.select("studio_scenes", {"project_id": p["id"]}), key=lambda s: s["position"])
    stages: dict[str, list] = {s: [] for s in STAGES}
    blocked: list[str] = []

    def add(stage, target, label, opts=None, kind=None):
        kind = kind or STAGE_KIND[stage]
        cost, reason = cs._price(kind, p, target, cast, opts)
        if reason and any(m in reason for m in LATER):
            reason = None          # an earlier stage of this run makes it
        if reason:
            blocked.append(f"{label}: {reason}")
            return
        stages[stage].append({"target_id": target["id"], "label": label, "credits": pricing.hold_credits(kind, cost),
                              "input": opts or {}, "kind": kind, **({"scene_id": target["scene_id"]}
                                                                      if kind in ("shot_image", "shot_video") else {})})

    try:
        elements = st.select("studio_elements", {"project_id": p["id"]})
    except Exception:
        if hasattr(st, "conn"):
            st.conn.rollback()
        elements = []
    for e in sorted(elements, key=lambda e: (e["kind"] != "location", e.get("position") or 0)):
        if not e.get("reference_path"):
            add("places", e, f"{e['name']} picture", {"element_id": e["id"]})
    for c in cast:
        has_front = ((c.get("reference_views") or {}).get(ident.ANCHOR) or {}).get("path")
        if options["sheets"] and not has_front and not c.get("identity_locked_at"):
            add("sheets", c, f"{c['name']}'s reference sheet", {"views": list(ident.VIEWS)})
        elif not options["sheets"] and not c.get("reference_path") and (c.get("description") or "").strip():
            add("sheets", c, f"{c['name']}'s portrait")
    import studio_shots
    pending: list = []
    size = "2K" if p.get("image_quality") == "high" else "1K"
    one_check = pricing.estimate_inspection(4) if p.get("auto_inspect") is not False else None

    def _reserve(scene, i, k):
        chars = len((scene.get("continuity") or {}).get("characters") or []) or 1
        img = pricing.image_cost(1, 400 + 1300 * min(chars + 1, 4), size)
        pic = pricing.combine([img, one_check, img, one_check] if one_check else [img])   # incl. one correction
        stages["images"].append({"target_id": f"reserve-{scene['id']}", "label": f"Scene {i}: up to {k} shot pictures",
                                 "credits": k * pricing.hold_credits("shot_image", pic), "reserve_only": True,
                                 "kind": "shot_image", "input": {}})
        if scene.get("mode") == "video":
            secs = max(3, min(15, int(float(scene.get("duration_seconds") or 6) / k + 0.999) + 2))
            clip = pricing.estimate_video(secs)
            if any((d.get("text") or "").strip() for d in scene.get("dialogue") or []) and \
                    scene_speech(scene, p) == "video_model" and p.get("auto_inspect") is not False:
                # Talking shots: their voice check and room for one automatic voice fix (only used if needed).
                clip = pricing.combine([clip, pricing.estimate_listen(secs, video=True),
                                        cs._voice_reserve(p, scene, secs, True)])
            stages["videos"].append({"target_id": f"reserve-{scene['id']}", "label": f"Scene {i}: up to {k} shot clips",
                                     "credits": k * clip.credits, "reserve_only": True,
                                     "kind": "shot_video", "input": {}})
    shots_by: dict = {}
    for sh in cs._project_shots(p["id"]):
        shots_by.setdefault(sh["scene_id"], []).append(sh)
    for i, s in enumerate(scenes, start=1):
        shots = sorted(shots_by.get(s["id"], []), key=lambda r: r["position"])
        if studio_shots.uses_shots(s, shots) and studio_shots.plan_outdated(s, shots):
            blocked.append(studio_shots.OUTDATED.format(n=i))
            continue
        if studio_shots.uses_shots(s, shots):
            # A scene with an approved shot plan is made shot by shot (and its
            # voices once, if saved voices speak it); never a single clip it won't use.
            for sh in shots:
                n = sh["position"] + 1
                if not sh.get("image_path") and not sh.get("approved_at"):
                    add("images", sh, f"Scene {i}, shot {n} picture", {"shot_id": sh["id"]}, "shot_image")
                if s.get("mode") == "video" and not sh.get("video_path") and not sh.get("approved_at"):
                    add("videos", sh, f"Scene {i}, shot {n} clip", {"shot_id": sh["id"]}, "shot_video")
            lines = [d for d in s.get("dialogue") or [] if (d.get("text") or "").strip()]
            if lines and not s.get("audio_path") and scene_speech(s, p) != "video_model":
                add("voices", s, f"Scene {i} voices")
            continue
        # A scene that already has its picture or clip is reused as it is, never directed again.
        finished = s.get("image_path") or s.get("video_path")
        approved_single = s.get("image_approved_path") and s.get("image_approved_path") == s.get("image_path")
        if options.get("direct") and not finished and not approved_single and \
                not (s.get("made_with") or {}).get("_direct_failed"):
            # The AI director takes the scene: it plans the shots (unless a plan already exists, which is
            # used as it is), the run approves the plan, then every shot is made. What the shots will cost
            # is reserved now (an upper bound) and what isn't used is returned at the end.
            if shots:
                pending.append(s["id"])
            else:
                add("direct", s, f"Scene {i} directing", {"auto": True})
            _reserve(s, i, len(shots) or studio_shots.estimate_shots(s))
            lines = [d for d in s.get("dialogue") or [] if (d.get("text") or "").strip()]
            if lines and not s.get("audio_path") and scene_speech(s, p) != "video_model":
                add("voices", s, f"Scene {i} voices")
            continue
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
    # Coverage: a conversation or a long moment filmed as ONE still or clip looks like a slideshow. Say which
    # scenes would be single shots, so the creator can have them directed as several shots first.
    single = [(i, s) for i, s in enumerate(scenes, start=1)
              if not studio_shots.uses_shots(s, shots_by.get(s["id"], []))
              and (len([d for d in s.get("dialogue") or [] if (d.get("text") or "").strip()]) >= 2
                   or float(s.get("duration_seconds") or 0) >= 7)]
    if options.get("direct"):
        single = [(i, s) for i, s in single if (s.get("made_with") or {}).get("_direct_failed")]
    advice = [f"{len(single)} scene{'s' if len(single) != 1 else ''} ({', '.join(str(i) for i, _ in single[:12])}"
              f"{'...' if len(single) > 12 else ''}) would be filmed as one shot each. Plan their camera shots first "
              "(wide, over-the-shoulder, close-up, reaction) for real coverage."] if single else []
    # A distinct cast: characters who look alike are pointed out before paying (studio_dna; advice, never a block).
    try:
        import studio_dna
        for pr in studio_dna.cast_check(cast)["pairs"]:
            if pr["severity"] in ("high", "medium"):
                advice.append(f"{pr['problem']} {pr['fix']}")
    except Exception:
        pass
    directed = sum(1 for x in stages["direct"]) + len(pending)
    return {"stages": out, "total_credits": sum(x["credits"] for x in out), "blocked": blocked, "advice": advice,
            "single_shot_scenes": [s["id"] for _, s in single], "pending_plans": pending, "directed_scenes": directed,
            "note_direct": ("The AI director plans every scene's camera shots before anything is made. Shot "
                            "pictures and clips are reserved at most; what isn't used is returned at the end.")
            if directed else "",
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


def approved(run: dict) -> int:
    return int(run.get("approved_credits") if run.get("approved_credits") is not None else run.get("quoted_credits") or 0)


def committed(st, run: dict, jobs: Optional[list] = None) -> int:
    """What the run has spent or tied up: finished steps at their charge, waiting or running steps at their hold."""
    jobs = _run_jobs(st, run) if jobs is None else jobs
    return sum(int(j.get("charged_credits") or 0) if j["status"] == "succeeded"
               else int(j.get("estimated_credits") or 0) if j["status"] in ("queued", "running") else 0 for j in jobs)


def _confirmed_targets(run: dict) -> set:
    plan_ = run.get("plan") or {}
    if isinstance(plan_, str):
        plan_ = json.loads(plan_)
    return {i.get("target_id") for x in plan_.get("stages") or [] for i in x.get("items") or []}


def _still_needed(now: dict, jobs: list, from_stage: str) -> list:
    """Items the run still has to start, from this stage on (not the ones already running)."""
    open_targets = {j["idempotency_key"].split(":")[3] for j in jobs if j["status"] in ("queued", "running")}
    order = STAGES.index(from_stage)
    out = []
    for x in now["stages"]:
        if STAGES.index(x["stage"]) < order:
            continue
        for i in x["items"]:
            if i["target_id"] in open_targets:
                continue
            if i.get("reserve_only") and any(y["target_id"] != i["target_id"] and y.get("scene_id") == i["target_id"][8:]
                                             for z in now["stages"] for y in z["items"]):
                continue
            out.append({**i, "stage": x["stage"]})
    return out


def _ask_for_budget(st, run: dict, jobs: list, now: dict, stage: str) -> bool:
    """Pauses the run and asks for more if what's left to make no longer fits the approved budget.
    Returns True if it paused."""
    left = _still_needed(now, jobs, stage)
    need = committed(st, run, jobs) + sum(int(i["credits"]) for i in left)
    extra = need - approved(run)
    if extra <= 0:
        return False
    confirmed = _confirmed_targets(run)
    new = [i["label"] for i in left if i["target_id"] not in confirmed and not i.get("reserve_only")]
    why = (f"New steps that weren't in the approved plan: {', '.join(new[:6])}{'...' if len(new) > 6 else ''}"
           if new else "Steps that failed and need another try, or prices that changed since you approved")
    reason = (f"This production needs {extra} more credits than you approved to finish. {why}. Approve the "
              f"increase (new total {approved(run) + extra} credits) to continue, or cancel and keep what's made.")
    st.rpc("studio_run_request_budget", {"p_run": run["id"], "p_extra": extra, "p_reason": reason})
    return True


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
    options = {"direct": False, **options}      # runs started before automatic directing keep their plan
    rnd = _round(run)
    jobs = _run_jobs(st, run)
    now = plan(p, options)
    if options.get("direct") and _approve_directed(st, p, jobs, now):
        p = st.one("studio_projects", {"id": run["project_id"]}) or p
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
        items = [i for i in needed.get(stage) or [] if not i.get("reserve_only")]
        if not items:
            continue
        if stage in VIDEO_STAGES and options.get("storyboard_review") and not run.get("storyboard_approved_at"):
            # Nothing paid for video starts before the creator has approved the storyboard.
            st.rpc("studio_run_finish", {"p_run": run["id"], "p_status": "storyboard_review", "p_problem": STORYBOARD_NOTE})
            return False
        if run.get("approved_credits") is not None and _ask_for_budget(st, run, jobs, now, stage):
            return False
        stuck, started = [], False
        for item in items:
            tries = [j for j in mine if j["idempotency_key"].split(":")[3] == item["target_id"]
                     and j["idempotency_key"].split(":")[4] == str(rnd)]
            if len(tries) >= ATTEMPTS:
                last = sorted(tries, key=lambda j: j.get("created_at") or "")[-1]
                stuck.append(f"{item['label']}: {last.get('error_message') or 'it failed twice'}")
                continue
            key = f"run:{run['id']}:{stage}:{item['target_id']}:{rnd}:{len(tries)}"
            kind = item.get("kind") or STAGE_KIND[stage]
            r = st.rpc("studio_run_create_job", {
                "p_run": run["id"], "p_kind": kind, "p_lane": LANE.get(stage, "generation"), "p_key": key,
                "p_credits": item["credits"], "p_input": item["input"],
                "p_scene": item["target_id"] if kind in ("image", "voice", "video", "lipsync", "shotplan")
                else item.get("scene_id") if kind in ("shot_image", "shot_video") else None,
                "p_character": item["target_id"] if kind == "portrait" else None,
                # The first try of a step draws from the reservation; retries
                # (in this round or after a resume) hold from the balance,
                # where the failed try's refund went.
                "p_from_reserve": not any(j["idempotency_key"].split(":")[3] == item["target_id"] for j in mine)})
            if r and r.get("ok"):
                started = True
            elif (r or {}).get("error") == "budget_exceeded":
                # The database refused to go past the approved budget (another step got there first): ask.
                if not _ask_for_budget(st, st.one("studio_production_runs", {"id": run["id"]}) or run,
                                       _run_jobs(st, run), now, stage):
                    st.rpc("studio_run_request_budget", {"p_run": run["id"], "p_extra": int(r.get("needed") or 1),
                                                         "p_reason": f"{item['label']} needs {r.get('needed')} more "
                                                                     "credits than are left in your approved budget."})
                return started
            else:
                err = (r or {}).get("error")
                msg = {"insufficient_credits": "Not enough credits left to continue. Add credits, then resume."}.get(
                    err, f"Couldn't start {item['label'].lower()} ({err}).")
                st.rpc("studio_run_finish", {"p_run": run["id"], "p_status": "needs_attention", "p_problem": msg})
                return started
        if started:
            return True
        if stuck and stage == "direct":
            # Directing failed twice: that scene is made as one clip instead (never left unmade).
            for item in items:
                tries = [j for j in mine if j["idempotency_key"].split(":")[3] == item["target_id"]]
                if len(tries) >= ATTEMPTS and all(j["status"] == "failed" for j in tries):
                    sc = st.one("studio_scenes", {"id": item["target_id"]})
                    if sc:
                        st.update("studio_scenes", {"id": sc["id"]},
                                  {"made_with": {**(sc.get("made_with") or {}), "_direct_failed": True}})
            return False
        if stuck:
            st.rpc("studio_run_finish", {"p_run": run["id"], "p_status": "needs_attention",
                                         "p_problem": "These need you: " + " · ".join(stuck[:6])})
            return False
    # Everything is made but no render was asked for.
    st.rpc("studio_run_finish", {"p_run": run["id"], "p_status": "done", "p_problem": None})
    return False


def _approve_directed(st, p: dict, jobs: list, now: dict) -> bool:
    """The run approves the plans the director made for it (and plans that already existed), once no
    directing is still running. Never touches a scene the creator approved or made already."""
    if any(_stage_of(j) == "direct" and j["status"] in ("queued", "running") for j in jobs):
        return False
    directed = {j.get("scene_id") for j in jobs if _stage_of(j) == "direct" and j["status"] == "succeeded"}
    todo = (directed | set(now.get("pending_plans") or [])) - {None}
    changed = False
    for sid in todo:
        sc = st.one("studio_scenes", {"id": sid})
        if not sc or sc.get("shot_plan_approved") or not st.select("studio_shots", {"scene_id": sid}):
            continue
        st.update("studio_scenes", {"id": sid}, {"shot_plan_approved": True, "made_with": {
            **(sc.get("made_with") or {}), "_direction": {**((sc.get("made_with") or {}).get("_direction") or {}),
                                                         "auto_approved": True}}})
        changed = True
    return changed


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
        ro = run.get("options") or {}
        ro = json.loads(ro) if isinstance(ro, str) else ro
        now = {x["stage"]: len(x["items"]) for x in plan(p, {"direct": False, **ro})["stages"]} if p else {}
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
    spent = sum(int(j.get("charged_credits") or 0) for j in jobs if j["status"] == "succeeded")
    refunded = sum(int(j.get("estimated_credits") or 0) - (int(j.get("charged_credits") or 0) if j["status"] == "succeeded" else 0)
                   for j in jobs if j["status"] in ("succeeded", "failed", "canceled"))
    budget = approved(run)
    nxt = next((s for s in stages if s["done"] < s["total"]), None)
    opts = run.get("options") or {}
    opts = json.loads(opts) if isinstance(opts, str) else opts
    return {"id": run["id"], "status": run["status"], "problem": run.get("problem"),
            "quoted_credits": int(run.get("quoted_credits") or 0),
            "approved_credits": budget, "needs_credits": int(run.get("needs_credits") or 0) or None,
            "spent_credits": spent, "refunded_credits": max(refunded, 0),
            "committed_credits": committed(st, run, jobs), "remaining_budget": max(budget - committed(st, run, jobs), 0),
            "reserved_left": held, "percent": round(100 * done / total) if run["status"] != "done" else 100,
            "stages": stages, "completed_stages": [s["label"] for s in stages if s["done"] >= s["total"]],
            "next_step": (nxt or {}).get("label") if run["status"] not in ("done", "canceled") else None,
            "storyboard_review": bool(opts.get("storyboard_review")),
            "storyboard_approved_at": run.get("storyboard_approved_at"),
            "render_id": run.get("render_id"), "created_at": run.get("created_at"),
            "finished_at": run.get("finished_at")}


def history(st, run: dict) -> dict:
    """Where every credit of this production went: the reservation and any approved increases, each step's hold,
    charge and refund, and what came back at the end."""
    plan_ = run.get("plan") or {}
    plan_ = json.loads(plan_) if isinstance(plan_, str) else plan_
    labels = {i["target_id"]: i["label"] for x in plan_.get("stages") or [] for i in x.get("items") or []}
    try:
        events = st.select("studio_run_budget_events", {"run_id": run["id"]}, order="id")
    except Exception:
        if hasattr(st, "conn"):
            st.conn.rollback()
        events = []
    steps = []
    for j in sorted(_run_jobs(st, run), key=lambda j: j.get("created_at") or ""):
        parts = j["idempotency_key"].split(":")
        held = int(j.get("estimated_credits") or 0)
        charged = int(j.get("charged_credits") or 0) if j["status"] == "succeeded" else 0
        steps.append({"stage": STAGE_LABEL.get(parts[2], parts[2]), "label": labels.get(parts[3]) or STAGE_LABEL.get(parts[2]),
                      "try": int(parts[5]) + 1 if len(parts) > 5 else 1, "status": j["status"], "held": held,
                      "charged": charged, "refunded": held - charged if j["status"] in ("succeeded", "failed", "canceled") else 0,
                      "error": j.get("error_message") if j["status"] == "failed" else None,
                      "at": j.get("finished_at") or j.get("created_at")})
    return {"events": [{"kind": e["kind"], "credits": int(e["credits"] or 0), "note": e.get("note") or "",
                        "at": e.get("created_at")} for e in events],
            "steps": steps,
            "totals": {"approved": approved(run), "charged": sum(s["charged"] for s in steps),
                       "refunded": sum(s["refunded"] for s in steps),
                       "returned": sum(int(e["credits"] or 0) for e in events if e["kind"] == "returned")}}
