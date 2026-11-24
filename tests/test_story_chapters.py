"""Story Intelligence stage 3: chapters written in order from the story memory, read back into proposals, checked for
continuity, approved by the creator. Real Postgres, the real job queue and credit path, a fake AI (no paid calls)."""
import uuid

import pytest

import creator_studio as cs
import studio_ai
import studio_chapters as ch
import studio_novel_api
from conftest import ROOT, connect, make_user
from studio_canon import CanonRepo
from test_creator_studio import H, balance, env, grant, new_project, run, studio_schema  # noqa: F401
from test_story_interview import outline as make_outline

FILLER = " ".join(["The afternoon light moved slowly across the floor of the narrow shop."] * 140)   # ~1,600 words


@pytest.fixture(scope="module")
def chapters_schema(studio_schema):
    conn = connect()
    for f in ("creator_studio_canon.sql", "creator_studio_novel.sql", "creator_studio_chapters.sql",
              "creator_studio_chapters.sql", "creator_studio_review.sql"):                              # twice: safe to run again
        conn.execute((ROOT / f).read_text())
    conn.close()


@pytest.fixture
def nv(chapters_schema, env, monkeypatch):
    client, db, store, files, ai, gen, ren = env
    monkeypatch.setattr(cs, "_ai", ai)
    client.app.include_router(studio_novel_api.router)
    u = make_user(db)
    grant(client, u, 5000)
    return client, db, store, ai, gen, u


def approved_story(client, ai, u):
    """A short novel (12 chapters) whose outline the creator approved: Amani, Eric, Mama Grace, the letter."""
    pid = new_project(client, u, title="The Dress")
    s = client.post("/api/v1/studio/stories", headers=H(u), json={
        "project_id": pid, "idea": "A seamstress finds a letter sewn into a wedding dress.", "length": "short"}).json()["story"]
    ai.novel_results = {"interview question": {"understood": [], "question": {"topic": "goal", "text": "?"}}}
    client.post(f"/api/v1/studio/stories/{s['id']}/interview", headers=H(u), json={"request_id": uuid.uuid4().hex})
    client.post(f"/api/v1/studio/stories/{s['id']}/interview", headers=H(u),
                json={"request_id": uuid.uuid4().hex, "skip": True, "finish": True})
    ai.novel_results = {"novel outline": make_outline(kept_from_creator=[])}
    s = client.post(f"/api/v1/studio/stories/{s['id']}/outline", headers=H(u), json={"request_id": uuid.uuid4().hex}).json()["story"]
    r = client.post(f"/api/v1/studio/stories/{s['id']}/outline/approve", headers=H(u), json={"version": s["outline_version"]})
    assert r.status_code == 200, r.text
    return s["id"]


def write(client, u, sid, n, note=""):
    return client.post(f"/api/v1/studio/stories/{sid}/chapters/{n}/write", headers=H(u),
                       json={"request_id": uuid.uuid4().hex, "note": note})


def approve(client, u, sid, n, version=1, with_suggestions=True):
    return client.post(f"/api/v1/studio/stories/{sid}/chapters/{n}/approve", headers=H(u),
                       json={"version": version, "with_suggestions": with_suggestions})


CH1 = ("Amani found the letter on a Tuesday, folded small inside the hem of the ivory dress. " + FILLER +
       " Mama Grace took the letter from her hands and locked it in the tin box under the counter. "
       "Eric Habimana stood in the doorway, turning his gold watch, and said nothing at all.")
READ1 = {"summary": "Amani finds a letter; Mama Grace hides it; Eric watches.",
         "entities": [],
         "facts": [{"kind": "location", "subject": "the letter", "subject_kind": "object", "object": "the tailor shop",
                    "statement": "The letter is locked in the tin box in the tailor shop.",
                    "quote": "locked it in the tin box under the counter"},
                   {"kind": "trait", "subject": "Eric Habimana", "attribute": "habit", "value": "turns his gold watch",
                    "statement": "Eric turns his gold watch when he is nervous.", "quote": "turning his gold watch"},
                   {"kind": "trait", "subject": "Amani Uwase", "attribute": "eyes", "value": "green",
                    "statement": "Amani has green eyes.", "quote": "her green eyes flashed"}],      # not in the text
         "events": [{"summary": "Amani finds the letter in the dress hem.", "participants": ["Amani Uwase"],
                     "quote": "Amani found the letter on a Tuesday"}],
         "threads": [{"kind": "mystery", "description": "Why does Eric say nothing?", "entities": ["Eric Habimana"],
                      "quote": "and said nothing at all"}],
         "claims": [{"type": "appears", "subject": "Amani Uwase", "mode": "present", "quote": "Amani found the letter on a Tuesday"},
                    {"type": "acts", "subject": "Mama Grace", "mode": "present", "quote": "Mama Grace took the letter from her hands"}]}


def test_a_chapter_is_written_read_back_and_approved_into_memory(nv):
    client, db, store, ai, gen, u = nv
    sid = approved_story(client, ai, u)
    ai.novel_results = {"chapter": CH1, "chapter memory": READ1}
    before = balance(client, u)["available"]
    r = write(client, u, sid, 1)
    assert r.status_code == 200, r.text
    held = r.json()["credits_held"]
    run(gen)
    lst = client.get(f"/api/v1/studio/stories/{sid}/chapters", headers=H(u)).json()
    first = lst["chapters"][0]
    assert first["status"] == "draft" and first["words"] > 1250 and lst["next"] == 1
    assert [c["status"] for c in lst["chapters"][1:3]] == ["planned", "planned"]
    spent = before - balance(client, u)["available"]
    assert 0 < spent <= held                                     # charged what was used, never more than held

    got = client.get(f"/api/v1/studio/stories/{sid}/chapters/1", headers=H(u)).json()
    assert got["chapter"]["text"].startswith("Amani found the letter")
    ex = got["chapter"]["extraction"]
    assert ex["proposed"]["facts"] == 2 and any("isn't in the passage" in r["reason"] for r in ex["rejected"])
    assert ex["issues"] == []
    statements = [f["statement"] for f in got["suggestions"]["facts"]]
    assert "The letter is locked in the tin box in the tailor shop." in statements
    assert all(f["quote"] for f in got["suggestions"]["facts"])
    # The prompt was built from the plan and the memory, not invented.
    prompt = [p for w, p in ai.novel_prompts if w == "chapter"][-1]
    assert "THIS CHAPTER — 1. Chapter 1" in prompt and "STORY CANON at chapter 1" in prompt
    assert "burn scar on her left hand" in prompt                 # Amani's appearance from the approved outline

    # Not canon until approved.
    canon = CanonRepo(store, sid, str(u)).load()
    assert canon.state_at((2, 0)).location_of(canon.eid("the letter")) is None
    r = approve(client, u, sid, 1)
    assert r.status_code == 200 and r.json()["approved_suggestions"] == 4
    canon = CanonRepo(store, sid, str(u)).load()
    assert canon.state_at((2, 0)).location_of(canon.eid("the letter")) == canon.eid("the tailor shop")
    assert client.get(f"/api/v1/studio/stories/{sid}/chapters", headers=H(u)).json()["next"] == 2


def test_the_next_chapter_is_written_from_approved_memory_and_the_real_ending(nv):
    client, db, store, ai, gen, u = nv
    sid = approved_story(client, ai, u)
    ai.novel_results = {"chapter": CH1, "chapter memory": READ1}
    write(client, u, sid, 1)
    run(gen)
    # Chapter 2 can't be written before chapter 1 is approved.
    r = write(client, u, sid, 2)
    assert r.status_code == 409 and "Chapter 1 is next" in r.json()["detail"]
    approve(client, u, sid, 1)
    ai.novel_results = {"chapter": "Eric came back the next morning. " + FILLER,
                        "chapter memory": {"summary": "Eric returns.", "facts": [], "claims": []}}
    assert write(client, u, sid, 2).status_code == 200
    run(gen)
    prompt = [p for w, p in ai.novel_prompts if w == "chapter"][-1]
    assert "HOW CHAPTER 1 ACTUALLY ENDED" in prompt and "and said nothing at all" in prompt
    assert "Eric turns his gold watch when he is nervous." in prompt        # approved in chapter 1
    assert "green eyes" not in prompt                                       # rejected: never reached the memory


def test_where_a_fact_happens_comes_from_its_quote_not_the_ai(nv):
    client, db, store, ai, gen, u = nv
    sid = approved_story(client, ai, u)
    ai.novel_results = {"chapter": CH1, "chapter memory": READ1}
    write(client, u, sid, 1)
    run(gen)
    canon = CanonRepo(store, sid, str(u)).load()
    loc = next(f for f in canon.facts if f["kind"] == "location" and f["source_chapter"] == 1)
    ev = next(e for e in canon.events if e["chapter"] == 1)
    assert loc["valid_from_seq"] > 900 and ev["seq"] < 50          # the letter is hidden at the end, found at the start
    assert loc["source_ref"] == "chapter 1 v1"


def test_continuity_problems_are_reported_before_approval(nv):
    client, db, store, ai, gen, u = nv
    sid = approved_story(client, ai, u)
    death = ("Mama Grace died in her sleep that night, the radio still murmuring beside her. " + FILLER +
             " At dawn the street was silent.")
    ai.novel_results = {"chapter": death, "chapter memory": {
        "summary": "Mama Grace dies.", "claims": [],
        "facts": [{"kind": "status", "subject": "Mama Grace", "value": "dead", "statement": "Mama Grace is dead.",
                   "quote": "Mama Grace died in her sleep that night"}]}}
    write(client, u, sid, 1)
    run(gen)
    approve(client, u, sid, 1)
    ch2 = ("Mama Grace laughed at the stove and stirred the beans. " + FILLER +
           " Amani remembered how Mama Grace used to sing when she was small.")
    ai.novel_results = {"chapter": ch2, "chapter memory": {"summary": "x", "facts": [], "claims": [
        {"type": "speaks", "subject": "Mama Grace", "mode": "present", "quote": "Mama Grace laughed at the stove"},
        {"type": "speaks", "subject": "Mama Grace", "mode": "flashback", "as_of_chapter": 0,
         "quote": "Mama Grace used to sing when she was small"}]}}
    write(client, u, sid, 2)
    run(gen)
    ex = client.get(f"/api/v1/studio/stories/{sid}/chapters/2", headers=H(u)).json()["chapter"]["extraction"]
    assert [i["rule"] for i in ex["issues"]] == ["dead_character_active"]       # the flashback isn't reported
    assert ex["issues"][0]["quote"] == "Mama Grace laughed at the stove"
    assert client.get(f"/api/v1/studio/stories/{sid}/chapters", headers=H(u)).json()["chapters"][1]["issues"] == 1


def test_a_suggestion_that_contradicts_a_locked_fact_is_skipped_on_approval(nv):
    client, db, store, ai, gen, u = nv
    sid = approved_story(client, ai, u)
    repo = CanonRepo(store, sid, str(u))
    canon = repo.load()
    repo.creator_fact({"kind": "trait", "subject_id": canon.eid("Amani Uwase"), "attribute": "age", "value": "24"}, lock=True)
    text = "Amani, only nineteen years old, ran the shop alone. " + FILLER
    ai.novel_results = {"chapter": text, "chapter memory": {"summary": "x", "claims": [], "facts": [
        {"kind": "trait", "subject": "Amani Uwase", "attribute": "age", "value": "19", "statement": "Amani is 19.",
         "quote": "Amani, only nineteen years old"}]}}
    write(client, u, sid, 1)
    run(gen)
    r = approve(client, u, sid, 1)
    assert r.status_code == 200
    assert r.json()["approved_suggestions"] == 0 and r.json()["skipped"][0]["reason"] == "creator-locked"
    canon = repo.load()
    assert canon.state_at((5, 0)).traits_of(canon.eid("Amani Uwase"))["age"] == "24"


def test_rewriting_a_draft_replaces_its_suggestions_and_old_versions_cant_be_approved(nv):
    client, db, store, ai, gen, u = nv
    sid = approved_story(client, ai, u)
    ai.novel_results = {"chapter": CH1, "chapter memory": READ1}
    write(client, u, sid, 1)
    run(gen)
    ai.novel_results = {"chapter": "Amani found the letter on a Tuesday. " + FILLER,
                        "chapter memory": {"summary": "v2", "claims": [], "facts": [], "events": [
                            {"summary": "Amani finds the letter.", "quote": "Amani found the letter on a Tuesday"}]}}
    assert write(client, u, sid, 1, note="Shorter opening").status_code == 200
    run(gen)
    got = client.get(f"/api/v1/studio/stories/{sid}/chapters/1", headers=H(u)).json()
    assert got["chapter"]["version"] == 2 and got["suggestions"]["facts"] == []
    assert len(got["suggestions"]["events"]) == 1 and len(got["suggestions"]["threads"]) == 0
    assert "Shorter opening" in [p for w, p in ai.novel_prompts if w == "chapter"][-1]
    assert approve(client, u, sid, 1, version=1).status_code == 409
    assert approve(client, u, sid, 1, version=2).status_code == 200


def test_a_cut_off_chapter_or_a_failed_read_back_costs_nothing(nv):
    client, db, store, ai, gen, u = nv
    sid = approved_story(client, ai, u)
    before = balance(client, u)["available"]
    ai.novel_results = {"chapter": "Too short.", "chapter memory": READ1}
    write(client, u, sid, 1)
    run(gen)
    lst = client.get(f"/api/v1/studio/stories/{sid}/chapters", headers=H(u)).json()
    assert lst["writing"]["status"] == "failed" and "shorter than planned" in lst["writing"]["error"]
    assert lst["chapters"][0]["status"] == "planned" and balance(client, u)["available"] == before
    ai.novel_results = {"chapter": CH1}
    ai.fail = {"novel_json": studio_ai.ProviderError("down")}
    write(client, u, sid, 1)
    run(gen)
    lst = client.get(f"/api/v1/studio/stories/{sid}/chapters", headers=H(u)).json()
    assert "couldn't be read back" in lst["writing"]["error"] and lst["chapters"][0]["status"] == "planned"
    assert balance(client, u)["available"] == before


def test_one_chapter_at_a_time_and_no_credits_no_chapter(nv):
    client, db, store, ai, gen, u = nv
    sid = approved_story(client, ai, u)
    assert write(client, u, sid, 1).status_code == 200
    r = write(client, u, sid, 1)
    assert r.status_code == 409 and "already being written" in r.json()["detail"]
    poor = make_user(db)
    grant(client, poor, 200)
    sid2 = approved_story(client, ai, poor)
    db.execute("update public.studio_credit_lots set credits_remaining = 0 where user_id = %s", (poor,))
    assert write(client, poor, sid2, 1).status_code == 402


def test_paid_off_threads_are_resolved_when_the_chapter_is_approved(nv):
    client, db, store, ai, gen, u = nv
    sid = approved_story(client, ai, u)
    ai.novel_results = {"chapter": CH1, "chapter memory": READ1}
    write(client, u, sid, 1)
    run(gen)
    approve(client, u, sid, 1)

    def read2(prompt):
        ref = next(line.split(":")[0] for line in prompt.splitlines() if "Why does Eric say nothing?" in line)
        return {"summary": "Eric explains.", "facts": [], "claims": [],
                "resolved_threads": [{"ref": ref, "quote": "Eric finally told her why he had been silent"}]}
    ai.novel_results = {"chapter": "Eric finally told her why he had been silent. " + FILLER, "chapter memory": read2}
    write(client, u, sid, 2)
    run(gen)
    approve(client, u, sid, 2)
    t = next(t for t in CanonRepo(store, sid, str(u)).load().threads if t["description"] == "Why does Eric say nothing?")
    assert t["state"] == "resolved" and t["resolved_chapter"] == 2


def test_the_creator_decides_suggestions_one_by_one(nv):
    client, db, store, ai, gen, u = nv
    sid = approved_story(client, ai, u)
    ai.novel_results = {"chapter": CH1, "chapter memory": READ1}
    write(client, u, sid, 1)
    run(gen)
    s = client.get(f"/api/v1/studio/stories/{sid}/chapters/1", headers=H(u)).json()["suggestions"]
    keep, drop = s["facts"][0]["id"], s["facts"][1]["id"]
    r = client.post(f"/api/v1/studio/stories/{sid}/memory/decide", headers=H(u),
                    json={"approve": {"facts": [keep]}, "reject": {"facts": [drop], "threads": [s["threads"][0]["id"]]}})
    assert r.status_code == 200 and set(r.json()["done"]) == {keep, drop, s["threads"][0]["id"]}
    left = client.get(f"/api/v1/studio/stories/{sid}/chapters/1", headers=H(u)).json()["suggestions"]
    assert left["facts"] == [] and left["threads"] == [] and len(left["events"]) == 1
    statuses = {f["id"]: f["status"] for f in CanonRepo(store, sid, str(u)).load().facts}
    assert statuses[keep] == "approved" and statuses[drop] == "retired"


def test_only_the_owner_writes_reads_or_approves_chapters(nv):
    client, db, store, ai, gen, u = nv
    sid = approved_story(client, ai, u)
    other = make_user(db)
    grant(client, other, 1000)
    assert write(client, other, sid, 1).status_code == 404
    assert client.get(f"/api/v1/studio/stories/{sid}/chapters", headers=H(other)).status_code == 404
    assert approve(client, other, sid, 1).status_code == 404


def test_pure_helpers():
    text = "# Chapter 3: The Hem\nShe sewed until dawn.\n```"
    assert ch.clean_text(text, "The Hem") == "She sewed until dawn."
    body = "Alpha beta gamma delta. " * 50 + "the last words are here"
    assert ch.seq_of("Alpha beta gamma delta", body) == 0
    assert ch.seq_of("the last words are here", body) > 950
    assert ch.seq_of("words that are not there", body) is None
    assert ch.tail("a b c d e", 2) == "d e"
