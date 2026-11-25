"""Story Intelligence stage 5: continuity problems judged by VIYO (advice only) and settled by the creator. Real
Postgres, the real job queue and credit path, a fake AI (no paid calls)."""
import uuid

import studio_canon_check as cc
from studio_canon import CanonRepo
from test_creator_studio import H, balance, env, grant, new_project, run, studio_schema  # noqa: F401
from test_story_chapters import FILLER, approve, approved_story, chapters_schema, nv, write  # noqa: F401

DEATH = ("Mama Grace died in her sleep that night, the radio still murmuring beside her. " + FILLER +
         " At dawn the street was silent.")
READ_DEATH = {"summary": "Mama Grace dies.", "claims": [], "facts": [
    {"kind": "status", "subject": "Mama Grace", "value": "dead", "statement": "Mama Grace is dead.",
     "quote": "Mama Grace died in her sleep that night"}]}
GHOST = ("Mama Grace stood at the foot of the bed, pale as smoke, and Amani knew she was seeing a ghost. " + FILLER +
         " Then the room was empty again.")
READ_GHOST = {"summary": "Amani sees Mama Grace's ghost.", "facts": [], "claims": [
    {"type": "appears", "subject": "Mama Grace", "mode": "present", "quote": "Mama Grace stood at the foot of the bed"}]}


def dead_then_ghost(client, ai, gen, u, judge=None):
    """Chapter 1: Mama Grace dies (approved). Chapter 2: she appears again (flagged)."""
    sid = approved_story(client, ai, u)
    ai.novel_results = {"chapter": DEATH, "chapter memory": READ_DEATH}
    write(client, u, sid, 1)
    run(gen)
    assert approve(client, u, sid, 1).status_code == 200
    ai.novel_results = {"chapter": GHOST, "chapter memory": READ_GHOST}
    if judge is not None:
        ai.novel_results["continuity judgement"] = judge
    write(client, u, sid, 2)
    run(gen)
    return sid


def chapter(client, u, sid, n):
    return client.get(f"/api/v1/studio/stories/{sid}/chapters/{n}", headers=H(u)).json()


def decide(client, u, sid, n, pid, action, version=1, **kw):
    return client.post(f"/api/v1/studio/stories/{sid}/chapters/{n}/problems/{pid}/decide", headers=H(u),
                       json={"version": version, "action": action, **kw})


def judged(verdict, quote="", fix="", why="because"):
    return lambda prompt: {"judgements": [{"id": line.split()[1].rstrip(":"), "verdict": verdict, "intent": "ghost",
                                           "why": why, "quote": quote, "fix": fix}
                                          for line in prompt.splitlines() if line.startswith("PROBLEM ")]}


def test_viyo_judges_each_problem_with_a_quote_and_only_when_there_are_problems(nv):
    client, db, store, ai, gen, u = nv
    sid = dead_then_ghost(client, ai, gen, u, judge=judged("intended", quote="knew she was seeing a ghost"))
    calls = [c for c in ai.calls if c == ("novel_json", "continuity judgement")]
    assert len(calls) == 1                                    # chapter 1 had no problems: no judgement call there
    prompt = [p for w, p in ai.novel_prompts if w == "continuity judgement"][-1]
    assert "Mama Grace is dead" in prompt and "Mama Grace stood at the foot of the bed" in prompt
    got = chapter(client, u, sid, 2)
    [p] = got["problems"]
    assert p["rule"] == "dead_character_active" and p["source"] == "claim"
    assert p["judgement"]["verdict"] == "intended" and p["judgement"]["quote"] == "knew she was seeing a ghost"
    assert p["memory_statement"] == "Mama Grace is dead." and p["memory_quote"]
    assert set(p["actions"]) == {"fix_chapter", "chapter_right", "intended"}
    assert got["chapter"]["problems"]["open"] == 1 and got["chapter"]["problems"]["intended"] == 1


def test_intended_without_evidence_becomes_unclear_and_a_failed_judgement_still_shows_the_problem(nv):
    client, db, store, ai, gen, u = nv
    sid = dead_then_ghost(client, ai, gen, u, judge=judged("intended", quote="words that are not in the chapter"))
    p = chapter(client, u, sid, 2)["problems"][0]
    assert p["judgement"]["verdict"] == "unclear" and p["judgement"]["quote"] == "" and "couldn't show" in p["judgement"]["note"]

    sid2 = approved_story(client, ai, u)
    ai.novel_results = {"chapter": DEATH, "chapter memory": READ_DEATH}
    write(client, u, sid2, 1)
    run(gen)
    approve(client, u, sid2, 1)

    def broken(prompt):
        raise RuntimeError("provider down")
    ai.novel_results = {"chapter": GHOST, "chapter memory": READ_GHOST, "continuity judgement": broken}
    write(client, u, sid2, 2)
    run(gen)
    got = chapter(client, u, sid2, 2)
    assert got["chapter"]["status"] == "draft"                # the chapter is kept
    assert len(got["problems"]) == 1 and "judgement" not in got["problems"][0]


def test_a_chapter_with_open_problems_isnt_approved_unless_the_creator_insists(nv):
    client, db, store, ai, gen, u = nv
    sid = dead_then_ghost(client, ai, gen, u, judge=judged("error", fix="Remove Mama Grace from the scene."))
    r = approve(client, u, sid, 2)
    assert r.status_code == 409 and r.json()["detail"]["open_problems"] == 1
    r = client.post(f"/api/v1/studio/stories/{sid}/chapters/2/approve", headers=H(u),
                    json={"version": 1, "with_suggestions": True, "accept_problems": True})
    assert r.status_code == 200 and r.json()["approved_with_open_problems"] == 1
    ex = store.one("studio_story_chapters", {"story_id": sid, "number": 2})["extraction"]
    assert ex["approved_with_open_problems"] == 1 and len(ex["problems"]) == 1      # kept on record


def test_fix_chapter_blocks_approval_and_writes_the_note_for_the_next_version(nv):
    client, db, store, ai, gen, u = nv
    sid = dead_then_ghost(client, ai, gen, u, judge=judged("error", fix="Remove Mama Grace from the scene."))
    pid = chapter(client, u, sid, 2)["problems"][0]["id"]
    r = decide(client, u, sid, 2, pid, "fix_chapter")
    assert r.status_code == 200
    assert "Mama Grace is dead" in r.json()["fix_note"] and "Remove Mama Grace from the scene." in r.json()["fix_note"]
    assert approve(client, u, sid, 2).status_code == 409                       # still needs a new version
    # The new version is written with the note and starts with its own (here: no) problems.
    note = chapter(client, u, sid, 2)["fix_note"]
    ai.novel_results = {"chapter": "Amani sat alone where Mama Grace used to sit. " + FILLER,
                        "chapter memory": {"summary": "Amani grieves.", "facts": [], "claims": []}}
    assert write(client, u, sid, 2, note=note).status_code == 200
    run(gen)
    prompt = [p for w, p in ai.novel_prompts if w == "chapter"][-1]
    assert "Remove Mama Grace from the scene." in prompt
    got = chapter(client, u, sid, 2)
    assert got["chapter"]["version"] == 2 and got["problems"] == []
    assert approve(client, u, sid, 2, version=2).status_code == 200


def test_intended_records_an_exception_so_it_isnt_reported_again_there(nv):
    client, db, store, ai, gen, u = nv
    sid = dead_then_ghost(client, ai, gen, u, judge=judged("intended", quote="knew she was seeing a ghost",
                                                            why="She appears as a ghost, clearly marked."))
    pid = chapter(client, u, sid, 2)["problems"][0]["id"]
    r = decide(client, u, sid, 2, pid, "intended", until_end=True)            # VIYO's reason is used
    assert r.status_code == 200 and r.json()["summary"]["open"] == 0
    [x] = store.select("studio_story_exceptions", {"story_id": sid})
    assert x["rule"] == "dead_character_active" and x["from_chapter"] == 2 and x["until_chapter"] is None
    assert x["reason"] == "She appears as a ghost, clearly marked."
    assert approve(client, u, sid, 2).status_code == 200
    # Chapter 3: she appears again; the exception covers it, so it's reported as excepted, not as a problem.
    ai.novel_results = {"chapter": "Mama Grace stood at the foot of the bed once more. " + FILLER,
                        "chapter memory": {"summary": "x", "facts": [], "claims": [
                            {"type": "appears", "subject": "Mama Grace", "mode": "present",
                             "quote": "Mama Grace stood at the foot of the bed once more"}]}}
    write(client, u, sid, 3)
    run(gen)
    got = chapter(client, u, sid, 3)
    assert got["problems"] == [] and len(got["chapter"]["extraction"]["excepted"]) == 1


def test_intended_needs_a_reason_when_viyo_didnt_give_one(nv):
    client, db, store, ai, gen, u = nv
    sid = dead_then_ghost(client, ai, gen, u, judge=judged("error"))
    pid = chapter(client, u, sid, 2)["problems"][0]["id"]
    r = decide(client, u, sid, 2, pid, "intended")
    assert r.status_code == 422 and "why" in r.json()["detail"]
    r = decide(client, u, sid, 2, pid, "intended", reason="Her ghost visits Amani.")
    assert r.status_code == 200
    x = store.select("studio_story_exceptions", {"story_id": sid})[0]
    assert x["until_chapter"] == 2 and x["reason"] == "Her ghost visits Amani."


def test_chapter_right_retires_the_memory_fact_and_a_locked_one_needs_confirmation(nv):
    client, db, store, ai, gen, u = nv
    sid = approved_story(client, ai, u)
    repo = CanonRepo(store, sid, str(u))
    canon = repo.load()
    locked = repo.creator_fact({"kind": "trait", "subject_id": canon.eid("Amani Uwase"), "attribute": "age",
                                "value": "24"}, lock=True)
    ai.novel_results = {"chapter": "Amani, only nineteen years old, ran the shop alone. " + FILLER,
                        "chapter memory": {"summary": "x", "claims": [], "facts": [
                            {"kind": "trait", "subject": "Amani Uwase", "attribute": "age", "value": "19",
                             "statement": "Amani is 19.", "quote": "Amani, only nineteen years old"}]},
                        "continuity judgement": judged("error")}
    write(client, u, sid, 1)
    run(gen)
    [p] = chapter(client, u, sid, 1)["problems"]
    assert p["source"] == "fact" and p["memory_locked"] is True
    assert "intended" not in p["actions"]
    assert decide(client, u, sid, 1, p["id"], "intended", reason="x").status_code == 422
    r = decide(client, u, sid, 1, p["id"], "chapter_right")
    assert r.status_code == 409 and r.json()["detail"]["locked"] is True
    assert store.one("studio_story_facts", {"id": locked["id"]})["status"] == "creator_locked"
    r = decide(client, u, sid, 1, p["id"], "chapter_right", unlock=True)
    assert r.status_code == 200
    assert store.one("studio_story_facts", {"id": locked["id"]})["status"] == "retired"
    # Now the chapter's own fact is approved with the chapter: Amani is 19.
    r = approve(client, u, sid, 1)
    assert r.status_code == 200 and r.json()["skipped"] == []
    canon = repo.load()
    assert canon.state_at((3, 0)).traits_of(canon.eid("Amani Uwase"))["age"] == "19"


def test_fix_chapter_on_a_fact_problem_keeps_the_wrong_fact_out_of_memory(nv):
    client, db, store, ai, gen, u = nv
    sid = approved_story(client, ai, u)
    repo = CanonRepo(store, sid, str(u))
    canon = repo.load()
    # An approved (not locked) fact for the same moment: a later change would just be a change over time.
    repo.creator_fact({"kind": "trait", "subject_id": canon.eid("Amani Uwase"), "attribute": "age", "value": "24",
                       "valid_from_chapter": 1})
    ai.novel_results = {"chapter": "Amani, only nineteen years old, ran the shop alone. " + FILLER,
                        "chapter memory": {"summary": "x", "claims": [], "facts": [
                            {"kind": "trait", "subject": "Amani Uwase", "attribute": "age", "value": "19",
                             "statement": "Amani is 19.", "quote": "Amani, only nineteen years old"}]}}
    write(client, u, sid, 1)
    run(gen)
    p = chapter(client, u, sid, 1)["problems"][0]
    assert decide(client, u, sid, 1, p["id"], "fix_chapter").status_code == 200
    assert store.one("studio_story_facts", {"id": p["proposed_fact_id"]})["status"] == "retired"
    assert chapter(client, u, sid, 1)["suggestions"]["facts"] == []


def test_decisions_are_for_one_version_only_the_owner_decides_and_an_approved_chapter_is_closed(nv):
    client, db, store, ai, gen, u = nv
    sid = dead_then_ghost(client, ai, gen, u, judge=judged("error"))
    pid = chapter(client, u, sid, 2)["problems"][0]["id"]
    assert decide(client, u, sid, 2, pid, "fix_chapter", version=7).status_code == 409
    assert decide(client, u, sid, 2, "nope", "fix_chapter").status_code == 404
    from conftest import make_user
    other = make_user(db)
    assert decide(client, other, sid, 2, pid, "fix_chapter").status_code == 404
    assert client.get(f"/api/v1/studio/stories/{sid}/problems", headers=H(other)).status_code == 404
    decide(client, u, sid, 2, pid, "intended", reason="ghost")
    approve(client, u, sid, 2)
    assert decide(client, u, sid, 2, pid, "fix_chapter").status_code == 409


def test_the_story_problem_overview(nv):
    client, db, store, ai, gen, u = nv
    sid = dead_then_ghost(client, ai, gen, u, judge=judged("error"))
    repo = CanonRepo(store, sid, str(u))
    canon = repo.load()
    eric = canon.eid("Eric Habimana")
    repo.creator_fact({"kind": "trait", "subject_id": eric, "attribute": "age", "value": "30"})
    repo.creator_fact({"kind": "trait", "subject_id": eric, "attribute": "age", "value": "31"})
    r = client.get(f"/api/v1/studio/stories/{sid}/problems", headers=H(u)).json()
    assert any(m["rule"] == "conflicting_facts" for m in r["memory"])
    assert r["chapters"] == [{"number": 2, "status": "draft", "total": 1, "open": 1, "errors": 1, "intended": 0,
                              "unclear": 0, "decided": 0}]


def test_older_chapters_get_their_problems_rebuilt_and_the_judgement_is_in_the_hold(nv):
    client, db, store, ai, gen, u = nv
    sid = dead_then_ghost(client, ai, gen, u, judge=judged("error"))
    row = store.one("studio_story_chapters", {"story_id": sid, "number": 2})
    ex = dict(row["extraction"])
    ex.pop("problems")                                         # as if written before stage 5
    store.update("studio_story_chapters", {"id": row["id"]}, {"extraction": ex})
    [p] = chapter(client, u, sid, 2)["problems"]
    assert p["rule"] == "dead_character_active" and "judgement" not in p
    assert decide(client, u, sid, 2, p["id"], "fix_chapter").status_code == 200
    import studio_pricing as pricing
    assert pricing.chapter_hold(3000) >= pricing.credits_for(pricing.estimate_chapter(3000).total_usd)


def test_pure_helpers():
    ex = {"issues": [{"rule": "knowledge_leak", "subject": "Eric", "subject_id": "e1", "quote": "Eric knew",
                      "reason": "too early", "evidence": {"id": "f1", "statement": "Eric learns it in 5",
                                                          "source_quote": "he learned", "status": "approved"}}],
          "conflicts": [{"fact": "p1", "statement": "Amani is 19", "with": [
              {"fact": {"id": "f2", "statement": "Amani is 24", "source_quote": "", "status": "creator_locked"},
               "blocking": True, "reason": "creator-locked"}]}]}
    probs = cc.problems_from(ex)
    assert [p["source"] for p in probs] == ["claim", "fact"]
    assert probs[0]["id"] == cc.problems_from(ex)[0]["id"]                     # stable ids
    assert cc.allowed_actions(probs[1]) == ["fix_chapter", "chapter_right"]
    assert cc.open_problems(probs) == probs
    decided = [{**probs[0], "resolution": {"action": "intended"}}, {**probs[1], "resolution": {"action": "fix_chapter"}}]
    assert cc.open_problems(decided) == [decided[1]]
    assert cc.carry_decisions(probs, decided)[0]["resolution"] == {"action": "intended"}
    assert "Amani is 19" in cc.fix_note(decided) and cc.fix_note(probs) == ""
    raw = {"judgements": [{"id": probs[0]["id"], "verdict": "maybe", "why": "?"},
                          {"id": probs[1]["id"], "verdict": "error", "why": "wrong age", "fix": "Make her 24."}]}
    out = cc.apply_judgements(probs, raw, "Eric knew the truth already.")
    assert out[0]["judgement"]["verdict"] == "unclear" and out[1]["judgement"]["fix"] == "Make her 24."
    assert cc.context("a b c " * 400 + "the needle here " + "x " * 400, "the needle here").count("the needle here") == 1
