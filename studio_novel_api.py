"""Story Intelligence stage 2 API: start a story from an idea, the interview, the novel outline and its approval.

Every AI step goes through the Studio's normal paid-call path (creator_studio._paid_call: a hold for the most it
can cost, settled to what was really used, nothing charged if it fails). Approving an outline is free and seeds
the story canon (studio_canon) through studio_novel.seed_canon.
"""
from __future__ import annotations

import datetime
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

import creator_studio as cs
import studio_novel as nv
import studio_pricing as pricing

router = APIRouter(prefix="/api/v1/studio", tags=["studio_novel"])


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


class StoryIn(BaseModel):
    project_id: str
    idea: str = Field(..., min_length=10, max_length=4000)
    length: Literal["short", "standard", "long"] = "standard"
    episodes: int = Field(0, ge=0, le=200)


class InterviewIn(BaseModel):
    request_id: str = Field(..., min_length=8, max_length=80)
    answer: Optional[str] = Field(None, max_length=2000)
    skip: bool = False            # "VIYO decides" for the waiting question
    finish: bool = False          # stop asking: plan the outline with what we have


class OutlineIn(BaseModel):
    request_id: str = Field(..., min_length=8, max_length=80)
    note: str = Field("", max_length=1500)


class OutlineEditIn(BaseModel):
    outline: dict


class ApproveIn(BaseModel):
    version: int


def _story(story_id: str, user_id: str) -> dict:
    return cs._owned("studio_stories", story_id, user_id)


def _save(story: dict, fields: dict) -> dict:
    return cs.get_store().update("studio_stories", {"id": story["id"]}, {**fields, "updated_at": _now()})[0]


def _world_text(story: dict) -> str:
    """The production's story world, if it has one (studio_world), for the interview and the outline."""
    import studio_world
    from studio_worker import StudioWorker
    if not story.get("project_id"):
        return ""
    p = cs.get_store().one("studio_projects", {"id": story["project_id"]})
    if not p:
        return ""
    return studio_world.block(StudioWorker(cs.get_store(), cs.get_files(), None, "generation").story_world(p), "story")


def _out(story: dict) -> dict:
    probs = story.get("outline_problems") or []
    t = nv.target(story)
    return {"id": story["id"], "project_id": story.get("project_id"), "title": story.get("title") or "",
            "idea": story.get("idea") or "", "length": story.get("length"), "chapters": t["chapters"],
            "episodes": story.get("episodes") or 0, "stage": story.get("stage"),
            "interview": {"turns": nv.turns(story), "known": nv.known(story), "missing": nv.missing(story),
                          "unverified": (story.get("interview") or {}).get("unverified") or [],
                          "question": (nv.pending(story) or {}).get("question"), "ready": nv.ready(story),
                          "asked": len(nv.turns(story)), "max_questions": nv.MAX_QUESTIONS},
            "outline": story.get("outline") or {}, "problems": probs, "blocking": nv.blocking(probs),
            "outline_version": story.get("outline_version") or 0,
            "approved_version": story.get("outline_approved_version"), "approved_at": story.get("outline_approved_at"),
            "prices": {"question": pricing.estimate_interview_turn().credits,
                       "outline": pricing.estimate_outline(t["chapters"]).credits}}


@router.post("/stories")
async def start_story(body: StoryIn, user_id: str = Depends(cs.creator)):
    """Starts a novel from the creator's idea, for one of their productions (one story per production)."""
    import studio_ai
    p = cs._project(body.project_id, user_id)
    existing = [s for s in cs.get_store().select("studio_stories", {"project_id": p["id"]}) if s.get("status") == "draft"]
    if existing:
        return {"story": _out(existing[0]), "existing": True}
    if not studio_ai.moderate_text(body.idea):
        raise HTTPException(status_code=422, detail="This idea can't be made on Viyo. Please change it.")
    row = cs.get_store().insert("studio_stories", [{"owner_id": user_id, "project_id": p["id"],
                                                    "series_id": p.get("series_id"), "title": (p.get("title") or "")[:200],
                                                    "idea": body.idea.strip(), "length": body.length,
                                                    "episodes": body.episodes, "stage": "interview",
                                                    "interview": {"turns": [], "known": {}}}])[0]
    return {"story": _out(row), "existing": False}


@router.get("/stories/{story_id}")
async def get_story(story_id: str, user_id: str = Depends(cs.creator)):
    return {"story": _out(_story(story_id, user_id))}


@router.get("/projects/{project_id}/story")
async def project_story(project_id: str, user_id: str = Depends(cs.creator)):
    p = cs._project(project_id, user_id)
    rows = [s for s in cs.get_store().select("studio_stories", {"project_id": p["id"]}) if s.get("status") == "draft"]
    return {"story": _out(rows[0]) if rows else None}


@router.post("/stories/{story_id}/interview")
async def interview(story_id: str, body: InterviewIn, user_id: str = Depends(cs.creator)):
    """Answers the waiting question (or skips it, or finishes the interview), then asks the next one: a paid
    step only when a new question is actually needed."""
    import studio_ai
    story = _story(story_id, user_id)
    if story.get("stage") != "interview":
        raise HTTPException(status_code=409, detail="The interview is finished; the outline is next.")
    if body.answer is not None or body.skip:
        if body.answer and not studio_ai.moderate_text(body.answer):
            raise HTTPException(status_code=422, detail="This answer can't be used on Viyo. Please change it.")
        try:
            iv = nv.record_answer(story, body.answer, skip=body.skip)
        except ValueError as e:
            raise HTTPException(status_code=409, detail=str(e))
        story = _save(story, {"interview": iv})                 # kept even if the next question fails
    if body.finish:
        story = _save(story, {"interview": {**(story.get("interview") or {}), "done": True}})
    if nv.pending(story):
        raise HTTPException(status_code=409, detail="Answer the question first, or let VIYO decide.")
    if nv.ready(story):
        if not (story.get("interview") or {}).get("done"):
            story = _save(story, {"interview": {**(story.get("interview") or {}), "done": True}})
        return {"story": _out(story), "balance": cs._balance(user_id)}
    cs._rate_limit(user_id)
    prompt = nv.interview_prompt(story, _world_text(story))
    raw = cs._paid_call(user_id, pricing.estimate_interview_turn().credits,
                        lambda: cs.get_ai().novel_json(prompt, nv.TURN_SCHEMA, "interview question"), "the next question")
    iv, question = nv.apply_turn(story, raw)
    if not question["topic"]:
        iv["done"] = True
    else:
        iv = nv.add_question(iv, question)
    story = _save(story, {"interview": iv})
    return {"story": _out(story), "balance": cs._balance(user_id)}


@router.post("/stories/{story_id}/outline")
async def make_outline(story_id: str, body: OutlineIn, user_id: str = Depends(cs.creator)):
    """Plans the whole novel (a new version each time), checked and sent back once if the check finds problems."""
    import studio_ai
    import studio_story
    story = _story(story_id, user_id)
    if story.get("stage") == "outline_approved":
        raise HTTPException(status_code=409, detail="This outline is approved and its characters are in the story "
                                                    "memory. Start a new story to plan a different one.")
    if story.get("stage") == "interview" and not nv.ready(story):
        raise HTTPException(status_code=409, detail="Answer a few more questions first, or finish the interview.")
    if body.note and not studio_ai.moderate_text(body.note):
        raise HTTPException(status_code=422, detail="This note can't be used on Viyo. Please change it.")
    cs._rate_limit(user_id)
    craft = studio_story.genre_craft((nv.known(story).get("genre_tone") or {}).get("value") or "", story.get("idea") or "")
    base = nv.outline_prompt(story, _world_text(story), craft, body.note)
    ai = cs.get_ai()

    def plan():
        raw, cost = ai.novel_json(base, nv.OUTLINE_SCHEMA, "novel outline")
        o = nv.clean_outline(raw)
        probs = nv.outline_problems(o, story)
        if probs:
            raw2, cost2 = ai.novel_json(nv.fix_prompt(base, o, probs), nv.OUTLINE_SCHEMA, "novel outline")
            cost = pricing.combine([cost, cost2])
            o2 = nv.clean_outline(raw2)
            p2 = nv.outline_problems(o2, story)
            if o2.get("chapters") and len(p2) <= len(probs):
                o, probs = o2, p2
        summary = " ".join([o.get("premise") or ""] + [c.get("summary") or "" for c in o.get("chapters") or []])
        if not studio_ai.moderate_text(summary[:20000]):
            raise HTTPException(status_code=422, detail="The outline came out with content Viyo can't make. "
                                                        "Change the idea or add a note, then try again.")
        return (o, probs), cost

    outline, problems = cs._paid_call(user_id, pricing.estimate_outline(nv.target(story)["chapters"]).credits,
                                      plan, "the novel outline")
    story = _save(story, {"outline": outline, "outline_problems": problems, "stage": "outline",
                          "outline_version": int(story.get("outline_version") or 0) + 1,
                          "outline_approved_version": None, "outline_approved_at": None,
                          "title": story.get("title") or outline.get("title") or ""})
    return {"story": _out(story), "balance": cs._balance(user_id)}


@router.put("/stories/{story_id}/outline")
async def edit_outline(story_id: str, body: OutlineEditIn, user_id: str = Depends(cs.creator)):
    """The creator's own edits to the outline (free): checked again, saved as a new version."""
    import studio_ai
    story = _story(story_id, user_id)
    if story.get("stage") != "outline":
        raise HTTPException(status_code=409, detail="There's no outline to edit (or it's already approved).")
    o = nv.clean_outline(body.outline)
    if not studio_ai.moderate_text(" ".join([o.get("premise") or ""] + [c.get("summary") or "" for c in o["chapters"]])[:20000]):
        raise HTTPException(status_code=422, detail="This outline can't be made on Viyo. Please change it.")
    story = _save(story, {"outline": o, "outline_problems": nv.outline_problems(o, story),
                          "outline_version": int(story.get("outline_version") or 0) + 1})
    return {"story": _out(story)}


@router.post("/stories/{story_id}/outline/approve")
async def approve_outline(story_id: str, body: ApproveIn, user_id: str = Depends(cs.creator)):
    """The creator approves this exact outline version. Its characters, places, objects, secrets, relationships
    and planned threads go into the story memory as approved canon, each with the outline text as evidence."""
    from studio_canon import CanonRepo
    story = _story(story_id, user_id)
    if story.get("stage") != "outline":
        raise HTTPException(status_code=409, detail="There's no outline waiting for approval.")
    if body.version != int(story.get("outline_version") or 0):
        raise HTTPException(status_code=409, detail="The outline changed since you opened it. Look at the new "
                                                    "version, then approve it.")
    stop = nv.blocking(story.get("outline_problems") or [])
    if stop:
        raise HTTPException(status_code=409, detail={"message": "Fix these first.", "problems": stop})
    seeded = nv.seed_canon(CanonRepo(cs.get_store(), story["id"], user_id), story.get("outline") or {}, body.version)
    story = _save(story, {"stage": "outline_approved", "outline_approved_version": body.version,
                          "outline_approved_at": _now()})
    return {"story": _out(story), "memory": seeded}


# ====================================================================== stage 3: chapters

class WriteChapterIn(BaseModel):
    request_id: str = Field(..., min_length=8, max_length=80)
    note: str = Field("", max_length=1500)


class ApproveChapterIn(BaseModel):
    version: int
    with_suggestions: bool = False      # also approve this chapter's memory suggestions (conflicts are skipped)


class MemoryDecisionIn(BaseModel):
    approve: dict = Field(default_factory=dict)     # {"facts": [ids], "events": [ids], "threads": [ids]}
    reject: dict = Field(default_factory=dict)
    reason: str = Field("", max_length=400)


def _chapters(story: dict) -> dict:
    return {c["number"]: c for c in cs.get_store().select("studio_story_chapters", {"story_id": story["id"]})}


def _chapter_jobs(story: dict) -> list:
    if not story.get("project_id"):
        return []
    jobs = cs.get_store().select("studio_jobs", {"project_id": story["project_id"], "kind": "chapter"},
                                 order="created_at", desc=True, limit=20)
    return [j for j in jobs if (j.get("input") or {}).get("story_id") == story["id"]]


def _next_chapter(story: dict, chapters: dict) -> Optional[int]:
    """The chapter that can be written now: the first one, or the one after the last approved chapter."""
    total = len((story.get("outline") or {}).get("chapters") or [])
    if story.get("stage") != "outline_approved" or not total:
        return None
    done = [k for k, c in chapters.items() if c.get("status") == "approved"]
    n = max(done) + 1 if done else 1
    return n if n <= total else None


def _proposals(story: dict, n: int) -> dict:
    """This chapter's memory suggestions waiting for the creator, in plain sentences, with their evidence."""
    from studio_canon import CanonRepo, describe
    canon = CanonRepo(cs.get_store(), story["id"], story["owner_id"]).load()
    facts = [{"id": f["id"], "statement": describe(f, canon.names), "quote": f.get("source_quote") or "",
              "conflicts": [c["reason"] for c in canon.conflicts_for(f)]}
             for f in canon.facts if f.get("status") == "proposed" and f.get("source_chapter") == n]
    events = [{"id": e["id"], "statement": e["summary"], "quote": e.get("source_quote") or ""}
              for e in canon.events if e.get("status") == "proposed" and int(e["chapter"]) == n]
    threads = [{"id": t["id"], "statement": f"{t['kind']}: {t['description']}", "quote": t.get("planted_quote") or ""}
               for t in canon.threads if t.get("status") == "proposed" and int(t["planted_chapter"]) == n]
    return {"facts": facts, "events": events, "threads": threads}


def _chapter_summary(c: dict) -> dict:
    ex = c.get("extraction") or {}
    return {"number": c["number"], "title": c.get("title") or "", "status": c.get("status"), "words": c.get("words") or 0,
            "version": c.get("version") or 1, "summary": c.get("summary") or "",
            "issues": len(ex.get("issues") or []), "proposed": ex.get("proposed") or {}}


@router.get("/stories/{story_id}/chapters")
async def list_chapters(story_id: str, user_id: str = Depends(cs.creator)):
    story = _story(story_id, user_id)
    chapters = _chapters(story)
    plan = (story.get("outline") or {}).get("chapters") or []
    jobs = _chapter_jobs(story)
    writing = next(({"chapter": (j.get("input") or {}).get("chapter"), "status": j["status"], "job_id": j["id"],
                     "error": j.get("error_message")} for j in jobs), None)
    return {"chapters": [_chapter_summary(chapters[k]) if k in chapters else
                         {"number": k, "title": c.get("title") or "", "status": "planned"}
                         for k, c in ((int(c.get("number") or 0), c) for c in plan)],
            "next": _next_chapter(story, chapters), "writing": writing,
            "price": pricing.chapter_hold(nv.target(story)["words"]), "balance": cs._balance(user_id)}


@router.get("/stories/{story_id}/chapters/{number}")
async def get_chapter(story_id: str, number: int, user_id: str = Depends(cs.creator)):
    story = _story(story_id, user_id)
    c = _chapters(story).get(number)
    if not c:
        raise HTTPException(status_code=404, detail="This chapter hasn't been written yet.")
    return {"chapter": {**_chapter_summary(c), "text": c.get("text") or "", "extraction": c.get("extraction") or {}},
            "suggestions": _proposals(story, number)}


@router.post("/stories/{story_id}/chapters/{number}/write")
async def write_chapter(story_id: str, number: int, body: WriteChapterIn, user_id: str = Depends(cs.creator)):
    """Queues one chapter (an ordinary Studio job: credits held, the unused part returned, everything returned if
    it fails). Chapters are written in order, each after the one before it is approved."""
    import studio_ai
    story = _story(story_id, user_id)
    if story.get("stage") != "outline_approved":
        raise HTTPException(status_code=409, detail="Approve the novel plan first.")
    if not story.get("project_id"):
        raise HTTPException(status_code=409, detail="This story isn't linked to a production.")
    chapters = _chapters(story)
    nxt = _next_chapter(story, chapters)
    if nxt is None or number != nxt:
        raise HTTPException(status_code=409, detail=(f"Chapter {nxt} is next: chapters are written in order, each "
                                                     "after the one before it is approved.") if nxt else
                            "Every chapter is written and approved.")
    if any(j["status"] in ("queued", "running") for j in _chapter_jobs(story)):
        raise HTTPException(status_code=409, detail="A chapter is already being written. Wait for it to finish.")
    if body.note and not studio_ai.moderate_text(body.note):
        raise HTTPException(status_code=422, detail="This note can't be used on Viyo. Please change it.")
    cs._rate_limit(user_id)
    credits = pricing.chapter_hold(nv.target(story)["words"])
    r = cs.get_store().rpc("studio_create_job", {
        "p_owner": user_id, "p_project": story["project_id"], "p_kind": "chapter", "p_lane": "generation",
        "p_key": f"{user_id}:{body.request_id}:chapter:{story['id']}:{number}", "p_credits": credits,
        "p_input": {"story_id": story["id"], "chapter": number, "note": body.note.strip()}})
    if not r or not r.get("ok"):
        err = (r or {}).get("error", "failed")
        if err == "insufficient_credits":
            raise HTTPException(status_code=402, detail={"message": "Not enough credits.", "balance": cs._balance(user_id)})
        raise HTTPException(status_code=400, detail={"too_many_jobs": "Too many jobs running. Wait for some to finish."}
                            .get(err, err))
    return {"job_id": r["job_id"], "credits_held": credits, "balance": cs._balance(user_id)}


@router.post("/stories/{story_id}/chapters/{number}/approve")
async def approve_chapter(story_id: str, number: int, body: ApproveChapterIn, user_id: str = Depends(cs.creator)):
    """The creator approves this exact chapter version. With with_suggestions, its memory suggestions are approved
    too (one by one: anything that contradicts a creator-locked fact is skipped and listed), and the threads it
    pays off are marked resolved."""
    from studio_canon import CanonConflict, CanonRepo
    story = _story(story_id, user_id)
    c = _chapters(story).get(number)
    if not c:
        raise HTTPException(status_code=404, detail="This chapter hasn't been written yet.")
    if c.get("status") == "approved":
        raise HTTPException(status_code=409, detail="This chapter is already approved.")
    if body.version != int(c.get("version") or 1):
        raise HTTPException(status_code=409, detail="A newer version of this chapter exists. Read it, then approve it.")
    approved, skipped = 0, []
    if body.with_suggestions:
        repo = CanonRepo(cs.get_store(), story["id"], user_id)
        s = _proposals(story, number)
        for f in s["facts"]:
            try:
                repo.approve(f["id"])
                approved += 1
            except CanonConflict as e:
                skipped.append({"statement": f["statement"], "reason": e.conflicts[0]["reason"] if e.conflicts else str(e)})
        for e in s["events"]:
            repo.approve_event(e["id"])
            approved += 1
        for t in s["threads"]:
            repo.approve_thread(t["id"])
            approved += 1
        for r in (c.get("extraction") or {}).get("resolved") or []:
            try:
                repo.resolve_thread(r["thread_id"], number, r.get("quote") or "")
            except Exception:
                pass
    row = cs.get_store().update("studio_story_chapters", {"id": c["id"]},
                                {"status": "approved", "approved_at": _now(), "updated_at": _now()})[0]
    return {"chapter": _chapter_summary(row), "approved_suggestions": approved, "skipped": skipped}


@router.post("/stories/{story_id}/memory/decide")
async def decide_memory(story_id: str, body: MemoryDecisionIn, user_id: str = Depends(cs.creator)):
    """The creator approves or rejects individual memory suggestions (facts, events, threads)."""
    from studio_canon import CanonConflict, CanonError, CanonRepo
    story = _story(story_id, user_id)
    repo = CanonRepo(cs.get_store(), story["id"], user_id)
    done, refused = [], []
    for kind, fn in (("facts", repo.approve), ("events", repo.approve_event), ("threads", repo.approve_thread)):
        for rid in body.approve.get(kind) or []:
            try:
                fn(cs._uuid(rid))
                done.append(rid)
            except CanonConflict as e:
                refused.append({"id": rid, "reason": e.conflicts[0]["reason"] if e.conflicts else str(e)})
            except CanonError as e:
                refused.append({"id": rid, "reason": str(e)})
    reason = body.reason.strip() or "rejected by the creator"
    for rid in body.reject.get("facts") or []:
        try:
            repo.retire(cs._uuid(rid), reason)
            done.append(rid)
        except CanonError as e:
            refused.append({"id": rid, "reason": str(e)})
    for kind, table in (("events", "studio_story_events"), ("threads", "studio_story_threads")):
        for rid in body.reject.get(kind) or []:
            row = cs.get_store().one(table, {"id": cs._uuid(rid), "story_id": story["id"]})
            if row and row.get("status") == "proposed":
                cs.get_store().update(table, {"id": row["id"]}, {"status": "retired"})
                done.append(rid)
            else:
                refused.append({"id": rid, "reason": "Only a suggestion can be rejected."})
    return {"done": done, "refused": refused}
