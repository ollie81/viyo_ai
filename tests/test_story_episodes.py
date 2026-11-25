"""Story Intelligence stage 7: the episode planner. An approved novel is split into episodes (one paid plan call,
checked by code), the creator approves one version, and episode productions are created with the novel's cast,
places and Story Bible; each episode's writer gets its plan, the novel's own words and the memory at that episode.
Real Postgres, a fake AI (no paid calls)."""
import uuid

import pytest

import creator_studio as cs
import studio_adapt as ad
import studio_novel_api
import studio_story
from conftest import ROOT, connect, make_user
from test_creator_studio import H, balance, env, grant, new_project, run, studio_schema  # noqa: F401
from test_story_bible import approve_novel, written_novel
from test_story_bible import bible_schema as _bible_schema  # noqa: F401
from test_story_chapters import chapters_schema as _chapters_schema  # noqa: F401


@pytest.fixture(scope="module")
def episodes_schema(_bible_schema):
    conn = connect()
    for _ in range(2):                                                       # safe to run again
        conn.execute((ROOT / "creator_studio_episode_plan.sql").read_text())
    conn.close()


@pytest.fixture
def nv(episodes_schema, env, monkeypatch):
    client, db, store, files, ai, gen, ren = env
    monkeypatch.setattr(cs, "_ai", ai)
    client.app.include_router(studio_novel_api.router)
    u = make_user(db)
    grant(client, u, 20000)
    return client, db, store, ai, gen, u, files


def good_plan(n=5, chapters=12):
    """n episodes covering 1..chapters in order (sharing chapters when n > chapters)."""
    eps, per = [], chapters / n
    for i in range(n):
        a = int(i * per) + 1
        b = max(a, int((i + 1) * per))
        eps.append({"number": i + 1, "title": f"Part {i + 1}", "chapter_from": min(a, chapters),
                    "chapter_to": min(b, chapters), "summary": f"Episode {i + 1} happens.",
                    "beats": ["a beat", "another beat"], "hook": "A shock.", "turn": "A turn.",
                    "cliffhanger": "A door opens." if i < n - 1 else "She wins and loses."})
    eps[-1]["chapter_to"] = chapters
    return {"episodes": eps}


def plan(client, u, sid, episodes=5, **kw):
    return client.post(f"/api/v1/studio/stories/{sid}/episodes/plan", headers=H(u),
                       json={"request_id": uuid.uuid4().hex, "episodes": episodes, **kw})


def approved_novel(client, ai, gen, u):
    sid = written_novel(client, ai, gen, u)
    assert approve_novel(client, u, sid).status_code == 200
    return sid


def test_episodes_are_planned_only_for_an_approved_novel_and_need_a_number(nv):
    client, db, store, ai, gen, u, files = nv
    sid = written_novel(client, ai, gen, u, chapters=2)
    assert plan(client, u, sid).status_code == 409
    sid = approved_novel(client, ai, gen, u)
    r = plan(client, u, sid, episodes=0)
    assert r.status_code == 422 and "how many episodes" in r.json()["detail"]


def test_a_plan_is_made_checked_and_charged_once(nv):
    client, db, store, ai, gen, u, files = nv
    sid = approved_novel(client, ai, gen, u)
    ai.novel_results = {"episode plan": good_plan(5)}
    before = balance(client, u)["available"]
    hold = client.get(f"/api/v1/studio/stories/{sid}/episodes", headers=H(u)).json()["price"]
    r = plan(client, u, sid, episodes=5, seconds=120)
    assert r.status_code == 200, r.text
    d = r.json()
    assert len(d["plan"]) == 5 and d["problems"] == [] and d["version"] == 1 and d["seconds"] == 120
    spent = before - balance(client, u)["available"]
    assert 0 < spent and len([c for c in ai.calls if c == ("novel_json", "episode plan")]) == 1
    assert hold > 0
    prompt = [p for w, p in ai.novel_prompts if w == "episode plan"][-1]
    assert "EXACTLY 5" in prompt and "Amani finds the letter; Mama Grace hides it." in prompt


def test_a_bad_plan_is_sent_back_once_and_a_still_bad_plan_cant_be_approved(nv):
    client, db, store, ai, gen, u, files = nv
    sid = approved_novel(client, ai, gen, u)
    gap = good_plan(5)
    gap["episodes"][2]["chapter_from"] = 9                                    # skips chapters
    ai.novel_results = {"episode plan": [gap, good_plan(5)]}
    d = plan(client, u, sid).json()
    assert d["problems"] == [] and len([c for c in ai.calls if c == ("novel_json", "episode plan")]) == 2
    bad = good_plan(4)                                                        # wrong count, twice
    ai.novel_results = {"episode plan": [bad, bad]}
    d = plan(client, u, sid).json()
    assert any("exactly 5" in p for p in d["problems"]) and d["version"] == 2
    r = client.post(f"/api/v1/studio/stories/{sid}/episodes/plan/approve", headers=H(u), json={"version": 2})
    assert r.status_code == 409 and r.json()["detail"]["problems"]


def test_the_creator_edits_the_plan_and_it_is_checked_again(nv):
    client, db, store, ai, gen, u, files = nv
    sid = approved_novel(client, ai, gen, u)
    ai.novel_results = {"episode plan": good_plan(5)}
    d = plan(client, u, sid).json()
    eps = d["plan"]
    eps[0]["title"] = "The Hem"
    eps[1]["hook"] = ""
    r = client.put(f"/api/v1/studio/stories/{sid}/episodes/plan", headers=H(u), json={"episodes": eps})
    d = r.json()
    assert d["version"] == 2 and d["plan"][0]["title"] == "The Hem"
    assert d["problems"] == ["Episode 2 has no opening hook."]


def test_approving_creates_the_episode_productions_with_cast_places_and_bible(nv):
    client, db, store, ai, gen, u, files = nv
    sid = approved_novel(client, ai, gen, u)
    story = store.one("studio_stories", {"id": sid})
    ai.novel_results = {"episode plan": good_plan(5)}
    d = plan(client, u, sid, seconds=90).json()
    before = balance(client, u)["available"]
    assert client.post(f"/api/v1/studio/stories/{sid}/episodes/plan/approve", headers=H(u),
                       json={"version": 7}).status_code == 409                # stale
    r = client.post(f"/api/v1/studio/stories/{sid}/episodes/plan/approve", headers=H(u), json={"version": d["version"]})
    assert r.status_code == 200, r.text
    assert balance(client, u)["available"] == before                        # free
    made = r.json()["created"]
    assert [m["number"] for m in made] == [1, 2, 3, 4, 5] and made[0]["title"].startswith("The Dress · Episode 1: Part 1")

    eps = [store.one("studio_projects", {"id": m["project_id"]}) for m in made]
    assert len({e["series_id"] for e in eps}) == 1 and eps[0]["series_id"] != story["project_id"]
    assert all(e["story_id"] == sid and e["target_seconds"] == 90 and e["kind"] == "episode" for e in eps)
    assert [e["episode_number"] for e in eps] == [1, 2, 3, 4, 5]
    cast = store.select("studio_characters", {"project_id": eps[2]["id"]})
    assert sorted(c["name"] for c in cast) == ["Amani Uwase", "Eric Habimana", "Mama Grace"]
    assert {e["name"] for e in store.select("studio_elements", {"project_id": eps[2]["id"]})} == {"the tailor shop", "the letter"}
    saga = eps[0]["story_bible"]["saga"]
    # The secret comes out in chapter 9: placed in the episode that covers chapter 9.
    nine = ad.episode_for_chapter(d["plan"], 9)
    assert saga["secrets"][0]["reveal_episode"] == nine and nine > 1
    assert saga["length"] == {"mode": "episodes", "count": 5} and len(saga["episodes"]) == 5
    rows = sorted(store.select("studio_story_episodes", {"story_id": sid}), key=lambda r: r["number"])
    assert [(r["chapter_from"], r["chapter_to"]) for r in rows] == [(e["chapter_from"], e["chapter_to"]) for e in d["plan"]]
    # Once only; the plan can't be remade after.
    assert client.post(f"/api/v1/studio/stories/{sid}/episodes/plan/approve", headers=H(u),
                       json={"version": d["version"]}).status_code == 409
    assert plan(client, u, sid).status_code == 409
    listed = client.get(f"/api/v1/studio/stories/{sid}/episodes", headers=H(u)).json()["created"]
    assert [x["project_id"] for x in listed] == [m["project_id"] for m in made]


def test_each_episode_is_written_from_its_plan_the_novels_words_and_the_memory_at_that_episode(nv):
    client, db, store, ai, gen, u, files = nv
    sid = approved_novel(client, ai, gen, u)
    ai.novel_results = {"episode plan": good_plan(5)}
    d = plan(client, u, sid).json()
    made = client.post(f"/api/v1/studio/stories/{sid}/episodes/plan/approve", headers=H(u),
                       json={"version": d["version"]}).json()["created"]
    from studio_worker import StudioWorker
    w = StudioWorker(store, files, ai, "generation")
    nine = ad.episode_for_chapter(d["plan"], 9)
    ep = w.novel_episode(store.one("studio_projects", {"id": made[nine - 1]["project_id"]}))
    assert ep["number"] == nine and ep["of"] == 5 and ep["last"] is (nine == 5)
    assert any("Eric's father wrote the letter" in p["text"] for p in ep["passages"])        # chapter 9's own words
    assert "STORY CANON at chapter" in ep["memory"]
    first = w.novel_episode(store.one("studio_projects", {"id": made[0]["project_id"]}))
    assert "Amani found the letter folded inside the hem" in first["passages"][0]["text"]
    # At the first episode only Mama Grace knows who wrote the letter (Amani learns it in chapter 9).
    assert "Known by: Mama Grace." in first["memory"] and "burn scar" in first["memory"]
    prompt = studio_story.plan_prompt({"title": "x", "kind": "episode", "target_seconds": 90, "novel_episode": ep}, [])
    assert "THIS EPISODE ADAPTS THE APPROVED NOVEL" in prompt and "THE NOVEL'S OWN WORDS" in prompt
    assert w.novel_episode({"id": "x", "owner_id": str(u)}) is None                          # not a novel episode


def test_only_the_owner_plans_or_approves(nv):
    client, db, store, ai, gen, u, files = nv
    sid = approved_novel(client, ai, gen, u)
    other = make_user(db)
    grant(client, other, 1000)
    assert plan(client, other, sid).status_code == 404
    assert client.get(f"/api/v1/studio/stories/{sid}/episodes", headers=H(other)).status_code == 404
    assert client.post(f"/api/v1/studio/stories/{sid}/episodes/plan/approve", headers=H(other),
                       json={"version": 0}).status_code == 404


def test_pure_planner_helpers():
    p = ad.clean_plan(good_plan(5))
    assert ad.plan_problems(p, 12, 5) == []
    gap = [dict(e) for e in p]
    gap[2]["chapter_from"] = gap[2]["chapter_to"] = gap[1]["chapter_to"] + 2
    assert any("no gaps" in x for x in ad.plan_problems(gap, 12, 5))
    short = [dict(e) for e in p]
    short[-1]["chapter_to"] = 10
    assert any("reach chapter 12" in x for x in ad.plan_problems(short, 12, 5))
    many = ad.clean_plan(good_plan(30))                                      # more episodes than chapters
    assert ad.plan_problems(many, 12, 30) == []
    texts = {c: "\n".join(f"Paragraph {i} of chapter {c}." for i in range(10)) for c in range(1, 13)}
    sharing = [e["number"] for e in many if e["chapter_from"] <= 1 <= e["chapter_to"]]
    parts = [ad.source_passages(many, n, texts)[0]["text"] for n in sharing]
    assert "".join(parts).count("Paragraph 0 of chapter 1.") == 1                           # split, in order, no repeats
    assert parts[0].startswith("Paragraph 0") and len(parts) > 1
    long = {1: " ".join(["word"] * 9000)}
    one = [{"number": 1, "chapter_from": 1, "chapter_to": 1}]
    assert len(ad.source_passages(one, 1, long, max_words=500)[0]["text"].split()) <= 502
    saga = {"secrets": [{"secret": "s", "reveal_chapter": 9}], "turning_points": [{"chapter": 2, "what": "x"}],
            "mysteries": []}
    placed = ad.place_reveals(saga, p)
    assert placed["secrets"][0]["reveal_episode"] == ad.episode_for_chapter(p, 9)
    assert placed["turning_points"][0]["episode"] == ad.episode_for_chapter(p, 2)
    assert ad.block({}) == "" and "Cliffhanger" in ad.block({**p[0], "of": 5})
