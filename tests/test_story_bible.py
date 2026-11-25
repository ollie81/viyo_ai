"""Story Intelligence stage 6: approving the whole novel compiles its memory into the production's Story Bible (no AI)
and links its characters, places and objects to the production. Real Postgres, a fake AI (no paid calls)."""
import pytest

import creator_studio as cs
import studio_story_bible as sb
from conftest import ROOT, connect, make_user
from studio_canon import CanonRepo
from test_creator_studio import H, balance, env, grant, new_project, run, studio_schema  # noqa: F401
from test_story_chapters import FILLER, approve, approved_story, write
from test_story_chapters import chapters_schema as _chapters_schema  # noqa: F401
import studio_novel_api


@pytest.fixture(scope="module")
def bible_schema(_chapters_schema):
    conn = connect()
    for f in ("creator_studio_story_bible.sql", "creator_studio_story_bible.sql"):     # twice: safe to run again
        conn.execute((ROOT / f).read_text())
    conn.execute((ROOT / "creator_studio_elements.sql").read_text())
    conn.close()


@pytest.fixture
def nv(bible_schema, env, monkeypatch):
    client, db, store, files, ai, gen, ren = env
    monkeypatch.setattr(cs, "_ai", ai)
    client.app.include_router(studio_novel_api.router)
    u = make_user(db)
    grant(client, u, 20000)
    return client, db, store, ai, gen, u


def chapter_text(n):
    if n == 1:
        return ("Amani found the letter folded inside the hem. " + FILLER +
                " Mama Grace locked the letter in the tin box under the counter.")
    if n == 9:
        return ("Mama Grace finally told Amani that Eric's father wrote the letter. " + FILLER + " Amani wept.")
    return f"The morning of part {n} began quietly in the shop. " + FILLER


def chapter_read(n):
    if n == 1:
        return {"summary": "Amani finds the letter; Mama Grace hides it.", "claims": [],
                "facts": [{"kind": "location", "subject": "the letter", "subject_kind": "object",
                           "object": "the tailor shop", "statement": "The letter is in the tailor shop.",
                           "quote": "locked the letter in the tin box"}],
                "events": [{"summary": "Amani finds the letter.", "participants": ["Amani Uwase"],
                            "quote": "Amani found the letter folded inside the hem"}]}
    if n == 9:
        return {"summary": "The truth comes out.", "claims": [],
                "facts": [{"kind": "knows", "subject": "Amani Uwase", "object": "Who wrote the letter",
                           "statement": "Amani knows who wrote the letter.",
                           "quote": "told Amani that Eric's father wrote the letter"}],
                "events": [{"summary": "Mama Grace confesses who wrote the letter.", "participants": ["Mama Grace"],
                            "quote": "Mama Grace finally told Amani"}]}
    return {"summary": f"Things happen in chapter {n}.", "facts": [], "claims": []}


def written_novel(client, ai, gen, u, chapters=12):
    sid = approved_story(client, ai, u)
    for n in range(1, chapters + 1):
        ai.novel_results = {"chapter": chapter_text(n), "chapter memory": chapter_read(n)}
        assert write(client, u, sid, n).status_code == 200
        run(gen)
        assert approve(client, u, sid, n).status_code == 200, n
    return sid


def approve_novel(client, u, sid, chapters=12, **kw):
    return client.post(f"/api/v1/studio/stories/{sid}/approve", headers=H(u), json={"chapters": chapters, **kw})


def test_the_novel_is_approved_only_when_every_chapter_is(nv):
    client, db, store, ai, gen, u = nv
    sid = written_novel(client, ai, gen, u, chapters=3)
    r = approve_novel(client, u, sid)
    assert r.status_code == 409 and r.json()["detail"]["missing"][0] == 4


def test_approving_compiles_the_story_bible_and_joins_the_cast_for_free(nv):
    client, db, store, ai, gen, u = nv
    sid = written_novel(client, ai, gen, u)
    story = store.one("studio_stories", {"id": sid})
    before = balance(client, u)["available"]
    calls = len(ai.calls)
    assert approve_novel(client, u, sid, chapters=11).status_code == 409          # stale view
    r = approve_novel(client, u, sid)
    assert r.status_code == 200, r.text
    assert balance(client, u)["available"] == before and len(ai.calls) == calls   # free: no AI call
    b = r.json()["bible"]
    assert b["characters"] == 3 and b["characters_created"] == 3 and b["chapters"] == 12
    assert r.json()["story"]["stage"] == "novel_approved"

    p = store.one("studio_projects", {"id": story["project_id"]})
    saga = p["story_bible"]["saga"]
    assert saga["approved"] is True and saga["source"] == {"kind": "novel", "story_id": sid, "chapters": 12}
    amani = next(c for c in saga["characters"] if c["name"] == "Amani Uwase")
    assert amani["wants"] == "clear her mother's name" and amani["look"].startswith("tall, a burn scar")
    secret = saga["secrets"][0]
    assert secret["secret"] == "Eric's father wrote it" and "Amani Uwase" in secret["known_by"]
    assert secret["reveal_chapter"] == 9
    assert {"character": "Amani Uwase", "knows": "Who wrote the letter", "since_chapter": 9} in p["story_bible"]["knowledge"]
    assert [c["number"] for c in p["story_bible"]["novel"]["chapters"]] == list(range(1, 13))
    assert saga["turning_points"][0] == {"chapter": 1, "episode": 0, "what": "Amani finds the letter."}
    assert "She wins the case" in saga["ending"]

    cast = store.select("studio_characters", {"project_id": story["project_id"]})
    assert sorted(c["name"] for c in cast) == ["Amani Uwase", "Eric Habimana", "Mama Grace"]
    amani_row = next(c for c in cast if c["name"] == "Amani Uwase")
    assert "burn scar" in amani_row["description"] and amani_row["voice"]                 # cast automatically
    canon = CanonRepo(store, sid, str(u)).load()
    assert canon.entity("Amani Uwase")["character_id"] == amani_row["id"]
    els = store.select("studio_elements", {"project_id": story["project_id"]})
    assert {(e["kind"], e["name"]) for e in els} == {("location", "the tailor shop"), ("prop", "the letter")}
    assert "proof" in next(e for e in els if e["name"] == "the letter")["description"]
    assert approve_novel(client, u, sid).status_code == 409                                # once


def test_existing_cast_and_elements_are_linked_never_duplicated_or_changed(nv):
    client, db, store, ai, gen, u = nv
    sid = written_novel(client, ai, gen, u)
    pid = store.one("studio_stories", {"id": sid})["project_id"]
    mine = client.post(f"/api/v1/studio/projects/{pid}/characters", headers=H(u),
                       json={"name": "Mama Grace", "description": "my own description"}).json()
    store.insert("studio_elements", [{"project_id": pid, "owner_id": str(u), "kind": "location",
                                      "name": "The Tailor Shop", "description": "mine"}])
    r = approve_novel(client, u, sid)
    b = r.json()["bible"]
    assert b["characters_linked"] == 1 and b["characters_created"] == 2
    assert b["elements_linked"] == 1 and b["elements_created"] == 1
    grace = [c for c in store.select("studio_characters", {"project_id": pid}) if c["name"] == "Mama Grace"]
    assert len(grace) == 1 and grace[0]["description"] == "my own description" and grace[0]["id"] == mine["id"]
    # Syncing again adds nothing new.
    r = client.post(f"/api/v1/studio/stories/{sid}/bible/sync", headers=H(u))
    assert r.status_code == 200
    b = r.json()["bible"]
    assert b["characters_created"] == 0 and b["characters_linked"] == 0 and b["elements_created"] == 0
    assert len(store.select("studio_characters", {"project_id": pid})) == 3


def test_memory_inconsistencies_block_approval_unless_the_creator_insists(nv):
    client, db, store, ai, gen, u = nv
    sid = written_novel(client, ai, gen, u)
    repo = CanonRepo(store, sid, str(u))
    eric = repo.load().eid("Eric Habimana")
    repo.creator_fact({"kind": "trait", "subject_id": eric, "attribute": "height", "value": "tall"})
    repo.creator_fact({"kind": "trait", "subject_id": eric, "attribute": "height", "value": "short"})
    r = approve_novel(client, u, sid)
    assert r.status_code == 409 and r.json()["detail"]["code"] == "memory_problems"
    bible = client.get(f"/api/v1/studio/stories/{sid}/bible", headers=H(u)).json()
    assert bible["memory_problems"] and bible["approved"] is False and bible["compiled_at"] is None
    r = approve_novel(client, u, sid, accept_problems=True)
    assert r.status_code == 200 and r.json()["approved_with_memory_problems"] == 1


def test_the_bible_browser_and_what_never_reaches_the_bible(nv):
    client, db, store, ai, gen, u = nv
    sid = written_novel(client, ai, gen, u)
    repo = CanonRepo(store, sid, str(u))
    canon = repo.load()
    # An unreviewed suggestion and a locked correction.
    repo.propose_fact({"kind": "trait", "subject_id": canon.eid("Eric Habimana"), "attribute": "eyes", "value": "grey",
                       "statement": "Eric has grey eyes.", "source_quote": "x", "valid_from_chapter": 3})
    repo.creator_fact({"kind": "trait", "subject_id": canon.eid("Amani Uwase"), "attribute": "age", "value": "25",
                       "valid_from_chapter": 6}, lock=True)
    approve_novel(client, u, sid)
    b = client.get(f"/api/v1/studio/stories/{sid}/bible", headers=H(u)).json()
    amani = next(c for c in b["characters"] if c["name"] == "Amani Uwase")
    assert amani["age"] == "25" and amani["has_locked_facts"] is True and amani["cast_id"]
    assert amani["status"] == "alive"
    assert any(r["with"] == "Mama Grace" for r in amani["relationships"])
    eric = next(c for c in b["characters"] if c["name"] == "Eric Habimana")
    assert "eyes" not in str(eric)                                                   # proposed: not in the bible
    assert b["secrets"][0]["known_by"] == [{"who": "Mama Grace", "since_chapter": 0},      # from the outline
                                           {"who": "Amani Uwase", "since_chapter": 9}]     # learned in chapter 9
    assert b["objects"][0]["name"] == "the letter" and b["objects"][0]["where"] == "the tailor shop"
    assert b["timeline"][0]["chapter"] == 1 and b["approved"] and b["compiled_at"]
    saga = store.one("studio_projects", {"id": store.one("studio_stories", {"id": sid})["project_id"]})["story_bible"]["saga"]
    assert next(c for c in saga["characters"] if c["name"] == "Amani Uwase")["age"] == "25"


def test_only_the_owner_and_only_after_approval(nv):
    client, db, store, ai, gen, u = nv
    sid = written_novel(client, ai, gen, u, chapters=1)
    other = make_user(db)
    assert approve_novel(client, other, sid).status_code == 404
    assert client.get(f"/api/v1/studio/stories/{sid}/bible", headers=H(other)).status_code == 404
    assert client.post(f"/api/v1/studio/stories/{sid}/bible/sync", headers=H(u)).status_code == 409


def test_merge_keeps_the_rest_of_the_production_bible():
    old = {"world": {"era": "today"}, "voices": {"amani": {"voice": "Kore"}}, "_assistant_approval": {"v": 1},
           "saga": {"premise": "old"}}
    merged = sb.merge_into(old, {"saga": {"premise": "new"}, "knowledge": [], "novel": {"chapters": []}})
    assert merged["world"] == {"era": "today"} and merged["voices"] == {"amani": {"voice": "Kore"}}
    assert merged["saga"] == {"premise": "new"} and "_assistant_approval" not in merged
    assert sb.character_description({"look": "tall", "age": "24", "role": "hero", "voice": "soft"}) == \
        "tall. Age: 24. Role: hero. Voice: soft"
