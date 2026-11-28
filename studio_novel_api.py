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
    accept_problems: bool = False       # approve even though continuity problems are undecided or need a fix


class ProblemDecisionIn(BaseModel):
    version: int                                              # the chapter version the creator is looking at
    action: Literal["fix_chapter", "chapter_right", "intended"]
    reason: str = Field("", max_length=400)                   # why it's intentional (VIYO's reason if left empty)
    until_end: bool = False                                   # "intended" for the rest of the story, not only here
    unlock: bool = False                                      # "chapter_right" may retire a creator-locked fact


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


def _problems(c: dict) -> list:
    """A chapter's continuity problems (chapters written before stage 5 get them rebuilt, without VIYO's judgement)."""
    import studio_canon_check as cc
    ex = c.get("extraction") or {}
    return ex["problems"] if "problems" in ex else cc.problems_from(ex)


def _chapter_summary(c: dict) -> dict:
    import studio_canon_check as cc
    ex = c.get("extraction") or {}
    return {"number": c["number"], "title": c.get("title") or "", "status": c.get("status"), "words": c.get("words") or 0,
            "version": c.get("version") or 1, "summary": c.get("summary") or "",
            "issues": len(ex.get("issues") or []), "problems": cc.summary(_problems(c)),
            "proposed": ex.get("proposed") or {},
            "review": (c.get("review") or {}).get("status"),
            "revised": ((c.get("review") or {}).get("revision") or {}).get("kept")}


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
            "next": _next_chapter(story, chapters), "writing": writing, "stage": story.get("stage"),
            "price": pricing.chapter_hold(nv.target(story)["words"]), "balance": cs._balance(user_id)}


@router.get("/stories/{story_id}/chapters/{number}")
async def get_chapter(story_id: str, number: int, user_id: str = Depends(cs.creator)):
    story = _story(story_id, user_id)
    c = _chapters(story).get(number)
    if not c:
        raise HTTPException(status_code=404, detail="This chapter hasn't been written yet.")
    import studio_canon_check as cc
    probs = _problems(c)
    return {"chapter": {**_chapter_summary(c), "text": c.get("text") or "", "extraction": c.get("extraction") or {},
                        "review_report": c.get("review") or {}},
            "suggestions": _proposals(story, number),
            "problems": [{**p, "actions": cc.allowed_actions(p)} for p in probs],
            "fix_note": cc.fix_note(probs)}


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
    import studio_canon_check as cc
    still_open = cc.open_problems(_problems(c))
    if still_open and not body.accept_problems:
        raise HTTPException(status_code=409, detail={
            "message": f"{len(still_open)} continuity problem(s) need a decision first (or a new version of the "
                       "chapter). You can also approve it anyway.", "open_problems": len(still_open), "code": "open_problems"})
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
    fields = {"status": "approved", "approved_at": _now(), "updated_at": _now()}
    if still_open:                      # approved anyway: the problems stay on record, marked as accepted
        fields["extraction"] = {**(c.get("extraction") or {}), "problems": _problems(c),
                                "approved_with_open_problems": len(still_open)}
    row = cs.get_store().update("studio_story_chapters", {"id": c["id"]}, fields)[0]
    return {"chapter": _chapter_summary(row), "approved_suggestions": approved, "skipped": skipped,
            "approved_with_open_problems": len(still_open)}


@router.post("/stories/{story_id}/chapters/{number}/problems/{problem_id}/decide")
async def decide_problem(story_id: str, number: int, problem_id: str, body: ProblemDecisionIn,
                         user_id: str = Depends(cs.creator)):
    """The creator settles one continuity problem (studio_canon_check): the chapter needs fixing, the memory was
    wrong, or it's deliberate. Free; VIYO's judgement is shown but never decides."""
    import studio_ai
    import studio_canon_check as cc
    from studio_canon import CanonError, CanonRepo
    story = _story(story_id, user_id)
    c = _chapters(story).get(number)
    if not c:
        raise HTTPException(status_code=404, detail="This chapter hasn't been written yet.")
    if c.get("status") == "approved":
        raise HTTPException(status_code=409, detail="This chapter is already approved.")
    if body.version != int(c.get("version") or 1):
        raise HTTPException(status_code=409, detail="A newer version of this chapter exists. Read it first.")
    probs = _problems(c)
    p = cc.find(probs, problem_id)
    if not p:
        raise HTTPException(status_code=404, detail="This problem isn't in this version of the chapter.")
    if body.action not in cc.allowed_actions(p):
        raise HTTPException(status_code=422, detail={
            "intended": "Two facts that disagree can't both be true, so this one can't be marked as intended.",
            "chapter_right": "There's no memory fact to replace for this problem."}.get(body.action, "Not allowed."))
    repo = CanonRepo(cs.get_store(), story["id"], user_id)
    resolution = {"action": body.action, "at": _now()}
    try:
        if body.action == "fix_chapter":
            if p.get("proposed_fact_id"):                      # the chapter's wrong fact never reaches the memory
                f = cs.get_store().one("studio_story_facts", {"id": p["proposed_fact_id"], "story_id": story["id"]})
                if f and f.get("status") == "proposed":
                    repo.retire(f["id"], f"chapter {number}: continuity fix requested")
        elif body.action == "chapter_right":
            f = cs.get_store().one("studio_story_facts", {"id": p["memory_fact_id"], "story_id": story["id"]})
            if not f:
                raise HTTPException(status_code=404, detail="The memory fact no longer exists.")
            if f.get("status") == "creator_locked" and not body.unlock:
                raise HTTPException(status_code=409, detail={
                    "message": "You locked this fact. Confirm that you want to unlock and replace it.", "locked": True,
                    "code": "confirm_unlock"})
            if f.get("status") != "retired":
                repo.retire(f["id"], f"replaced by chapter {number}", force_locked=body.unlock)
            resolution["retired_fact"] = f["id"]
        else:                                                  # intended
            reason = body.reason.strip() or ((p.get("judgement") or {}).get("why") if
                                             (p.get("judgement") or {}).get("verdict") == "intended" else "")
            if not reason:
                raise HTTPException(status_code=422, detail="Say in a few words why it's intentional.")
            if body.reason and not studio_ai.moderate_text(body.reason):
                raise HTTPException(status_code=422, detail="This reason can't be used on Viyo. Please change it.")
            x = repo.add_exception(p["rule"], reason, subject_id=p.get("subject_id"), from_chapter=number,
                                   until_chapter=None if body.until_end else number)
            resolution.update(exception=x["id"], reason=reason[:400], until_end=body.until_end)
    except CanonError as e:
        raise HTTPException(status_code=409, detail=str(e))
    probs = [{**q, "resolution": resolution} if q["id"] == p["id"] else q for q in probs]
    cs.get_store().update("studio_story_chapters", {"id": c["id"]},
                          {"extraction": {**(c.get("extraction") or {}), "problems": probs}, "updated_at": _now()})
    return {"problem": {**p, "resolution": resolution}, "summary": cc.summary(probs), "fix_note": cc.fix_note(probs)}


@router.get("/stories/{story_id}/problems")
async def story_problems(story_id: str, user_id: str = Depends(cs.creator)):
    """Everything inconsistent in the story right now: problems inside the memory itself (two approved facts that
    can't both be true, a cause after its effect, an overdue thread), and each chapter's open continuity problems."""
    import studio_canon_check as cc
    from studio_canon import CanonRepo
    story = _story(story_id, user_id)
    chapters = _chapters(story)
    current = max([k for k, c in chapters.items() if c.get("status") == "approved"], default=0) + 1
    memory = CanonRepo(cs.get_store(), story["id"], user_id).load().audit(current_chapter=current)
    per_chapter = [{"number": k, "status": c.get("status"), **cc.summary(_problems(c))}
                   for k, c in sorted(chapters.items()) if _problems(c)]
    return {"memory": memory, "chapters": per_chapter}


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


# ====================================================================== stage 6: the novel and its Story Bible

class ApproveNovelIn(BaseModel):
    chapters: int = Field(..., ge=1)           # how many approved chapters the creator is looking at
    accept_problems: bool = False              # approve even though the memory still has inconsistencies


def _blocking_memory_problems(canon) -> list:
    """Inconsistencies inside the memory that would go into the Story Bible (overdue threads are only warned)."""
    return [m for m in canon.audit() if m.get("rule") in ("conflicting_facts", "object_two_places", "cause_after_effect")]


def _seed_production(story: dict, canon, user_id: str) -> dict:
    """Links the novel's characters, places and objects to the production's cast and elements: an existing one with
    the same name is linked (never changed), a missing one is created (characters are cast automatically)."""
    import studio_story_bible as sb
    st = cs.get_store()
    p = cs._project(story["project_id"], user_id)
    out = {"characters_created": 0, "characters_linked": 0, "elements_created": 0, "elements_linked": 0, "notes": []}
    cast = {c["name"].lower(): c for c in st.select("studio_characters", {"project_id": p["id"]})}
    outline = story.get("outline") or {}
    for e in canon.entities:
        if e.get("status") not in ("approved", "creator_locked") or e["kind"] != "character":
            continue
        if e.get("character_id") and any(c["id"] == e["character_id"] for c in cast.values()):
            continue
        row = cast.get(e["name"].lower()) or cast.get(e["name"][:60].lower())
        if row:
            out["characters_linked"] += 1
        else:
            row = cs.create_character(p, user_id, e["name"][:60], sb.character_description(sb.character_profile(canon, e, outline)))
            cast[row["name"].lower()] = row
            out["characters_created"] += 1
        st.update("studio_story_entities", {"id": e["id"]}, {"character_id": row["id"]})
    try:
        have = {(x["kind"], x["name"].lower()): x for x in st.select("studio_elements", {"project_id": p["id"]})}
        for el in sb.element_rows(canon, outline):
            if el["element_id"] and any(x["id"] == el["element_id"] for x in have.values()):
                continue
            row = have.get((el["kind"], el["name"].lower()))
            if row:
                out["elements_linked"] += 1
            else:
                row = st.insert("studio_elements", [{"project_id": p["id"], "owner_id": user_id, "kind": el["kind"],
                                                     "name": el["name"], "description": el["description"],
                                                     "details": {}, "position": len(have)}])[0]
                have[(el["kind"], el["name"].lower())] = row
                out["elements_created"] += 1
            st.update("studio_story_entities", {"id": el["entity_id"]}, {"element_id": row["id"]})
    except Exception as e:                       # places and objects need creator_studio_elements.sql
        if hasattr(st, "conn"):
            st.conn.rollback()
        out["notes"].append(f"Places and objects weren't added to the production: {type(e).__name__}.")
    cs._touch(p["id"])
    return out


def _compile_to_production(story: dict, user_id: str) -> dict:
    import studio_story_bible as sb
    from studio_canon import CanonRepo
    canon = CanonRepo(cs.get_store(), story["id"], user_id).load()
    compiled = sb.compile_bible(story, _chapters(story), canon)
    p = cs._project(story["project_id"], user_id)
    cs.get_store().update("studio_projects", {"id": p["id"]},
                          {"story_bible": sb.merge_into(p.get("story_bible"), compiled), "updated_at": _now()})
    seeded = _seed_production(story, CanonRepo(cs.get_store(), story["id"], user_id).load(), user_id)
    return {"characters": len(compiled["saga"]["characters"]), "secrets": len(compiled["saga"]["secrets"]),
            "mysteries": len(compiled["saga"]["mysteries"]), "open_threads": len(compiled["saga"]["open_threads"]),
            "knowledge": len(compiled["knowledge"]), "chapters": len(compiled["novel"]["chapters"]), **seeded}


@router.post("/stories/{story_id}/approve")
async def approve_novel(story_id: str, body: ApproveNovelIn, user_id: str = Depends(cs.creator)):
    """The creator approves the whole novel (every chapter approved first). Free: its memory becomes the
    production's Story Bible and its characters, places and objects join the production."""
    from studio_canon import CanonRepo
    story = _story(story_id, user_id)
    if story.get("stage") == "novel_approved":
        raise HTTPException(status_code=409, detail="This novel is already approved.")
    if story.get("stage") != "outline_approved" or not story.get("project_id"):
        raise HTTPException(status_code=409, detail="Approve the novel plan and write its chapters first.")
    chapters = _chapters(story)
    planned = [int(c.get("number") or 0) for c in (story.get("outline") or {}).get("chapters") or []]
    missing = [n for n in planned if (chapters.get(n) or {}).get("status") != "approved"]
    if missing:
        raise HTTPException(status_code=409, detail={
            "message": f"{len(missing)} chapter(s) aren't approved yet (first: chapter {missing[0]}).", "missing": missing[:50]})
    if body.chapters != len(planned):
        raise HTTPException(status_code=409, detail="The novel changed since you opened it. Look again, then approve.")
    problems = _blocking_memory_problems(CanonRepo(cs.get_store(), story["id"], user_id).load())
    if problems and not body.accept_problems:
        raise HTTPException(status_code=409, detail={
            "message": f"The story memory has {len(problems)} inconsistency(ies). Fix them in the Story Bible, or "
                       "approve anyway.", "problems": problems[:20], "code": "memory_problems"})
    result = _compile_to_production(story, user_id)
    story = _save(story, {"stage": "novel_approved", "novel_approved_at": _now()})
    return {"story": _out(story), "bible": result, "approved_with_memory_problems": len(problems)}


@router.post("/stories/{story_id}/bible/sync")
async def sync_bible(story_id: str, user_id: str = Depends(cs.creator)):
    """Compiles the memory into the production again (after the creator corrected or locked facts). Free."""
    story = _story(story_id, user_id)
    if story.get("stage") != "novel_approved":
        raise HTTPException(status_code=409, detail="Approve the novel first.")
    return {"bible": _compile_to_production(story, user_id)}


@router.get("/stories/{story_id}/bible")
async def story_bible(story_id: str, user_id: str = Depends(cs.creator)):
    """The Story Bible browser: characters, secrets and who knows them, places, objects, threads and the timeline,
    from the approved memory; plus whether the production's bible has been compiled."""
    import studio_story_bible as sb
    from studio_canon import CanonRepo
    story = _story(story_id, user_id)
    canon = CanonRepo(cs.get_store(), story["id"], user_id).load()
    p = cs.get_store().one("studio_projects", {"id": story["project_id"]}) if story.get("project_id") else None
    saga = ((p or {}).get("story_bible") or {}).get("saga") or {}
    return {**sb.browse(canon, story), "approved": story.get("stage") == "novel_approved",
            "approved_at": story.get("novel_approved_at"),
            "compiled_at": saga.get("built_at") if (saga.get("source") or {}).get("story_id") == story["id"] else None,
            "memory_problems": _blocking_memory_problems(canon)}


# ====================================================================== stage 7: the episode planner

class EpisodePlanIn(BaseModel):
    request_id: str = Field(..., min_length=8, max_length=80)
    episodes: int = Field(0, ge=0, le=200)          # 0 = the number chosen when the story started
    seconds: int = Field(0, ge=0, le=600)           # 0 = the production's episode length (or 90)
    note: str = Field("", max_length=1500)
    # "feature": one continuous movie of `minutes`, made as parts of at most 10 minutes (studio_feature)
    format: Literal["episodes", "feature"] = "episodes"
    minutes: int = Field(0, ge=0, le=90)


class EpisodePlanEditIn(BaseModel):
    episodes: list[dict] = Field(..., max_length=200)


class ApprovePlanIn(BaseModel):
    version: int


def _plan_out(story: dict) -> dict:
    plan = story.get("episode_plan") or []
    created = []
    if story.get("episodes_created_at"):
        rows = cs.get_store().select("studio_projects", {"story_id": story["id"], "owner_id": story["owner_id"]})
        created = sorted(({"number": r.get("episode_number"), "project_id": r["id"], "title": r.get("title") or ""}
                          for r in rows if not ((r.get("story_bible") or {}).get("feature") or {}).get("master")),
                         key=lambda x: x["number"] or 0)
    chapters = len((story.get("outline") or {}).get("chapters") or [])
    n = int(story.get("episodes") or 0) or len(plan) or max(1, chapters)
    return {"plan": plan, "problems": story.get("episode_plan_problems") or [],
            "version": story.get("episode_plan_version") or 0, "approved_version": story.get("episode_plan_approved_version"),
            "seconds": story.get("episode_seconds") or 90, "episodes_wanted": int(story.get("episodes") or 0),
            "created": created, "created_at": story.get("episodes_created_at"),
            "format": story.get("plan_format") or "episodes", "feature_minutes": story.get("feature_minutes"),
            "price": pricing.estimate_episode_plan(chapters, n).credits, "stage": story.get("stage"),
            "chapters": len((story.get("outline") or {}).get("chapters") or [])}


def _approved_chapters(story: dict) -> list:
    return [c for _, c in sorted(_chapters(story).items()) if c.get("status") == "approved"]


@router.get("/stories/{story_id}/episodes")
async def episode_plan(story_id: str, user_id: str = Depends(cs.creator)):
    return _plan_out(_story(story_id, user_id))


@router.post("/stories/{story_id}/episodes/plan")
async def make_episode_plan(story_id: str, body: EpisodePlanIn, user_id: str = Depends(cs.creator)):
    """Plans the episodes of an approved novel (a paid step, a new version each time), checked and sent back once
    if the check finds problems."""
    import studio_adapt as ad
    import studio_ai
    story = _story(story_id, user_id)
    if story.get("stage") != "novel_approved":
        raise HTTPException(status_code=409, detail="Approve the novel first.")
    if story.get("episodes_created_at"):
        raise HTTPException(status_code=409, detail="The episodes have already been created from the approved plan.")
    import studio_feature
    feature = body.format == "feature"
    if feature:
        if not studio_feature.MIN_MINUTES <= body.minutes <= studio_feature.MAX_MINUTES:
            raise HTTPException(status_code=422, detail=f"A feature film is {studio_feature.MIN_MINUTES} to "
                                                        f"{studio_feature.MAX_MINUTES} minutes long.")
        n, part_seconds = studio_feature.parts_for(body.minutes)
    else:
        n = body.episodes or int(story.get("episodes") or 0)
    if n < 1:
        raise HTTPException(status_code=422, detail="Choose how many episodes the novel should become.")
    if body.note and not studio_ai.moderate_text(body.note):
        raise HTTPException(status_code=422, detail="This note can't be used on Viyo. Please change it.")
    p = cs._project(story["project_id"], user_id)
    seconds = body.seconds or int(story.get("episode_seconds") or 0) or int(p.get("target_seconds") or 90)
    seconds = part_seconds if feature else min(600, max(30, seconds))
    chapters = _approved_chapters(story)
    cs._rate_limit(user_id)
    base = ad.plan_prompt(story, chapters, n, seconds, body.note, feature=feature)
    ai = cs.get_ai()

    def plan():
        raw, cost = ai.novel_json(base, ad.PLAN_SCHEMA, "episode plan")
        pl = ad.clean_plan(raw)
        probs = ad.plan_problems(pl, len(chapters), n)
        if probs:
            raw2, cost2 = ai.novel_json(ad.fix_prompt(base, pl, probs), ad.PLAN_SCHEMA, "episode plan")
            cost = pricing.combine([cost, cost2])
            pl2 = ad.clean_plan(raw2)
            p2 = ad.plan_problems(pl2, len(chapters), n)
            if pl2 and len(p2) <= len(probs):
                pl, probs = pl2, p2
        text = " ".join(e["summary"] + " " + " ".join(e["beats"]) for e in pl)
        if not studio_ai.moderate_text(text[:20000]):
            raise HTTPException(status_code=422, detail="The episode plan came out with content Viyo can't make. "
                                                        "Add a note and try again.")
        return (pl, probs), cost

    pl, probs = cs._paid_call(user_id, pricing.estimate_episode_plan(len(chapters), n).credits, plan, "the episode plan")
    story = _save(story, {"episode_plan": pl, "episode_plan_problems": probs, "episodes": n, "episode_seconds": seconds,
                          "plan_format": body.format, "feature_minutes": body.minutes if feature else None,
                          "episode_plan_version": int(story.get("episode_plan_version") or 0) + 1,
                          "episode_plan_approved_version": None})
    return {**_plan_out(story), "balance": cs._balance(user_id)}


@router.put("/stories/{story_id}/episodes/plan")
async def edit_episode_plan(story_id: str, body: EpisodePlanEditIn, user_id: str = Depends(cs.creator)):
    """The creator's own changes to the plan (free): checked again, saved as a new version."""
    import studio_adapt as ad
    import studio_ai
    story = _story(story_id, user_id)
    if story.get("stage") != "novel_approved" or story.get("episodes_created_at") or not story.get("episode_plan"):
        raise HTTPException(status_code=409, detail="There's no episode plan to edit (or the episodes already exist).")
    pl = ad.clean_plan({"episodes": body.episodes})
    if not studio_ai.moderate_text(" ".join(e["summary"] + " " + " ".join(e["beats"]) for e in pl)[:20000]):
        raise HTTPException(status_code=422, detail="This plan can't be made on Viyo. Please change it.")
    probs = ad.plan_problems(pl, len(_approved_chapters(story)), len(pl))
    story = _save(story, {"episode_plan": pl, "episode_plan_problems": probs, "episodes": len(pl) or story.get("episodes"),
                          "episode_plan_version": int(story.get("episode_plan_version") or 0) + 1})
    return _plan_out(story)


@router.post("/stories/{story_id}/episodes/plan/approve")
async def approve_episode_plan(story_id: str, body: ApprovePlanIn, user_id: str = Depends(cs.creator)):
    """The creator approves this exact plan: one episode production per planned episode is created (free), in a
    new series, with the novel's production's settings, cast and places, and the novel's Story Bible."""
    import uuid as _uuid

    import studio_adapt as ad
    from studio_canon import CanonRepo
    story = _story(story_id, user_id)
    if story.get("episodes_created_at"):
        raise HTTPException(status_code=409, detail="The episodes have already been created.")
    if story.get("stage") != "novel_approved" or not story.get("episode_plan"):
        raise HTTPException(status_code=409, detail="There's no episode plan to approve.")
    if body.version != int(story.get("episode_plan_version") or 0):
        raise HTTPException(status_code=409, detail="The plan changed since you opened it. Look again, then approve.")
    if story.get("episode_plan_problems"):
        raise HTTPException(status_code=409, detail={"message": "Fix the plan's problems first.",
                                                     "problems": story["episode_plan_problems"]})
    st = cs.get_store()
    src = cs._project(story["project_id"], user_id)
    plan = story["episode_plan"]
    series = str(_uuid.uuid4())
    bible = dict(src.get("story_bible") or {})
    bible.pop("_assistant_approval", None)
    if bible.get("saga"):
        bible["saga"] = ad.place_reveals(bible["saga"], plan)
    keep = {k: bible[k] for k in ("world", "saga", "knowledge", "novel", "voices", "voice_refs") if k in bible}
    title = (story.get("title") or src.get("title") or "Story")[:80]
    repo = CanonRepo(st, story["id"], user_id)
    made = []
    feature = story.get("plan_format") == "feature"
    master = None
    if feature:
        # The finished movie: a production of its own whose render joins the parts (studio_feature).
        row = {k: src.get(k) for k in ("aspect_ratio", "genre", "visual_style", "language", "captions")}
        row.update(owner_id=user_id, series_id=series, kind="film", title=title[:120],
                   target_seconds=sum(int(story.get("episode_seconds") or 600) for _ in plan),
                   logline=(src.get("logline") or "")[:1000], story_id=story["id"],
                   story_bible={**keep, "feature": {"id": series, "master": True, "of": len(plan), "title": title}})
        master = st.insert("studio_projects", [row])[0]
    for e in plan:
        row = {k: src.get(k) for k in ("aspect_ratio", "genre", "visual_style", "language", "captions")}
        for k in ("production_style", "image_quality"):
            if src.get(k):
                row[k] = src[k]
        row.update(owner_id=user_id, series_id=series, kind="film" if feature else "episode",
                   target_seconds=int(story.get("episode_seconds") or 90),
                   title=(f"{title} · Part {e['number']}: {e['title']}" if feature else
                          f"{title} · Episode {e['number']}: {e['title']}")[:120], logline=e["summary"][:1000],
                   story_bible={**keep, "feature": {"id": series, "part": e["number"], "of": len(plan),
                                                    "master_id": master["id"], "title": title}} if feature else keep,
                   story_id=story["id"], episode_number=e["number"])
        new = st.insert("studio_projects", [row])[0]
        cs.copy_cast_and_elements(src, new, user_id)
        repo.set_episode(e["number"], e["chapter_from"], e["chapter_to"], project_id=new["id"])
        made.append({"number": e["number"], "project_id": new["id"], "title": new["title"]})
    story = _save(story, {"episodes_created_at": _now(), "episode_series_id": series,
                          "episode_plan_approved_version": body.version})
    return {**_plan_out(story), "created": made, **({"feature": {"id": series, "master_id": master["id"],
                                                                 "title": master["title"], "parts": len(made)}}
                                                    if master else {})}
