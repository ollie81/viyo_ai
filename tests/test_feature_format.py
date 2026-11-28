"""Feature films: one continuous movie made as parts of at most 10 minutes, joined at the end.

What must pass before features are called supported (the owner's rule): long-story MEMORY, RENDERING of the parts
without breaks, RECOVERY from a failed part, and the final JOIN. Pure checks first, then the pipeline with a fake AI,
local Postgres and real ffmpeg: a novel planned as a feature (30 and 60 minutes), master + parts sharing cast, look
and memory, parts made and rendered, a failed part refunded and remade, and the movie joined into one film with no
fade to black at the part breaks.

NOT covered here (and not claimed): a real 60-minute movie on the production server (disk, memory, time)."""
import os
import re
import subprocess
import uuid

import pytest

import creator_studio as cs
import studio_adapt as ad
import studio_feature as feat
import studio_novel_api
import studio_render as render
from conftest import ROOT, connect, make_user
from test_creator_studio import H, balance, env, grant, run, start, studio_schema  # noqa: F401
from test_story_bible import approve_novel, written_novel
from test_story_bible import bible_schema as _bible_schema  # noqa: F401
from test_story_chapters import chapters_schema as _chapters_schema  # noqa: F401
from test_story_episodes import good_plan
from test_video_drama import split


# ------------------------------------------------------------------ pure

@pytest.mark.parametrize("minutes,expected", [(30, (3, 600)), (60, (6, 600)), (45, (5, 540)), (15, (2, 450)),
                                              (90, (9, 600)), (5, (2, 450)), (200, (9, 600))])
def test_a_feature_is_split_into_parts_of_at_most_ten_minutes(minutes, expected):
    n, secs = feat.parts_for(minutes)
    assert (n, secs) == expected and secs <= feat.MAX_PART_SECONDS


def test_parts_fade_only_at_the_movies_start_and_end():
    part = lambda k: {"story_bible": {"feature": {"id": "f", "part": k, "of": 3}}}       # noqa: E731
    assert feat.fades(part(1), True, True) == (True, False)
    assert feat.fades(part(2), True, True) == (False, False)
    assert feat.fades(part(3), True, True) == (False, True)
    assert feat.fades({}, True, True) == (True, True), "a normal production is unchanged"


def test_the_plan_and_the_writer_are_told_it_is_one_continuous_movie():
    story = {"title": "The Dress", "outline": {"premise": "x", "chapters": []}}
    p = ad.plan_prompt(story, [{"number": 1, "title": "a", "summary": "b"}], 3, 600, feature=True)
    assert "ONE feature film of about 30 minutes" in p and feat.PLAN_NOTE in p and "cliffhanger that makes" not in p
    assert "short-drama episodes" in ad.plan_prompt(story, [{"number": 1}], 3, 90), "episodes are unchanged"
    ep = {"number": 2, "of": 3, "title": "The shop", "chapter_from": 4, "chapter_to": 8, "summary": "s",
          "beats": ["a"], "hook": "Amani opens the letter", "cliffhanger": "She walks out", "feature": True,
          "memory": "", "passages": [{"chapter": 4, "text": "words"}]}
    b = ad.block(ep, "STORY CANON at chapter 4")
    assert "PART 2 OF 3 OF ONE CONTINUOUS FEATURE FILM" in b and "begins exactly where part 1 ended" in b
    assert "No recap" in b and "STORY CANON at chapter 4" in b and "THE NOVEL'S OWN WORDS" in b
    assert "Cliffhanger:" not in b


def test_parts_join_end_to_end_without_re_encoding_or_black_between_them(tmp_path):
    from PIL import Image
    parts = []
    for k, colour in enumerate([(200, 60, 60), (60, 200, 60), (60, 60, 200)]):
        img = str(tmp_path / f"p{k}.png")
        Image.new("RGB", (540, 960), colour).save(img)
        out = str(tmp_path / f"part{k}.mp4")
        # The same render as any part: the first part fades in, the last fades out, none at the breaks.
        render.render_film([render.SceneInput(duration=4, image=img, fade_in=0.6 if k == 0 else 0.0,
                                              fade_out=0.8 if k == 2 else 0.0)], "9:16", out, str(tmp_path), captions=False)
        parts.append(out)
    joined = str(tmp_path / "movie.mp4")
    feat.join(parts, joined, str(tmp_path))
    info = render.probe(joined)
    assert abs(info["duration"] - 12.0) < 0.3 and (info["width"], info["height"]) == (1080, 1920)
    log = subprocess.run(["ffmpeg", "-i", joined, "-vf", "blackdetect=d=0.3:pix_th=0.05", "-an", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    blacks = [float(x) for x in re.findall(r"black_start:([\d.]+)", log)]
    assert all(b < 1.0 or b > 11.0 for b in blacks), f"no black at the part breaks (4 s, 8 s): {blacks}"


# ------------------------------------------------------------------ the pipeline

@pytest.fixture(scope="module")
def feature_schema(_bible_schema):
    conn = connect()
    for f in ("creator_studio_episode_plan.sql", "creator_studio_feature.sql", "creator_studio_feature.sql"):
        conn.execute((ROOT / f).read_text())
    conn.close()


@pytest.fixture
def nv(feature_schema, env, monkeypatch):
    client, db, store, files, ai, gen, ren = env
    monkeypatch.setattr(cs, "_ai", ai)
    client.app.include_router(studio_novel_api.router)
    ai.varied_pictures = True
    u = make_user(db)
    grant(client, u, 400000)
    return client, db, store, files, ai, gen, ren, u


def feature(client, ai, gen, u, minutes, parts):
    sid = written_novel(client, ai, gen, u)
    assert approve_novel(client, u, sid).status_code == 200
    ai.novel_results = {"episode plan": good_plan(parts)}
    r = client.post(f"/api/v1/studio/stories/{sid}/episodes/plan", headers=H(u),
                    json={"request_id": uuid.uuid4().hex, "format": "feature", "minutes": minutes})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["format"] == "feature" and d["feature_minutes"] == minutes and d["seconds"] <= 600
    a = client.post(f"/api/v1/studio/stories/{sid}/episodes/plan/approve", headers=H(u), json={"version": d["version"]})
    assert a.status_code == 200, a.text
    return sid, a.json()


def test_a_30_minute_feature_is_planned_as_one_movie_in_three_parts(nv):
    client, db, store, files, ai, gen, ren, u = nv
    bad = client.post(f"/api/v1/studio/stories/{uuid.uuid4()}/episodes/plan", headers=H(u),
                      json={"request_id": uuid.uuid4().hex, "format": "feature", "minutes": 120})
    assert bad.status_code == 422
    sid, out = feature(client, ai, gen, u, 30, 3)
    f = out["feature"]
    assert f["parts"] == 3 and [m["number"] for m in out["created"]] == [1, 2, 3]
    master = store.one("studio_projects", {"id": f["master_id"]})
    parts = [store.one("studio_projects", {"id": m["project_id"]}) for m in out["created"]]
    assert master["kind"] == "film" and feat.is_master(master) and master["target_seconds"] == 1800
    assert all(p["kind"] == "film" and p["series_id"] == master["series_id"] and feat.is_part(p) for p in parts)
    assert [p["title"].split(" · ")[1].split(":")[0] for p in parts] == ["Part 1", "Part 2", "Part 3"]
    assert client.get(f"/api/v1/studio/stories/{sid}/episodes", headers=H(u)).json()["created"] == out["created"], \
        "the master isn't listed as a part"
    casts = [{c["name"]: c.get("identity_id") for c in store.select("studio_characters", {"project_id": p["id"]})}
             for p in parts]
    assert casts[0] and all(c == casts[0] for c in casts), "one cast, the same identities, in every part"


def test_memory_follows_the_movie_across_six_parts_of_a_60_minute_feature(nv):
    client, db, store, files, ai, gen, ren, u = nv
    sid, out = feature(client, ai, gen, u, 60, 6)
    from studio_worker import StudioWorker
    w = StudioWorker(store, files, ai, "generation")
    plan = store.one("studio_stories", {"id": sid})["episode_plan"]
    points = []
    for m in out["created"]:
        ep = w.novel_episode(store.one("studio_projects", {"id": m["project_id"]}))
        span = next(e for e in plan if e["number"] == m["number"])
        assert ep["feature"] is True and ep["of"] == 6 and ep["last"] is (m["number"] == 6)
        happened = [int(x) for x in re.findall(r"Happened \(chapter (\d+)\)", ep["memory"])]
        assert all(c <= span["chapter_to"] for c in happened), "no part's writer sees later chapters"
        points.append(int(re.search(r"STORY CANON at chapter (\d+)", ep["memory"]).group(1)))
        assert "OF ONE CONTINUOUS FEATURE FILM" in ad.block(ep, ep["memory"])
    assert points == sorted(points) and points[0] < points[-1], "the memory moves forward with the movie"


SCENES = [{"heading": "INT. SHOP - DAY", "visual": "Amani sews at the window", "duration_seconds": 4,
           "dialogue": [{"character": "Amani Uwase", "text": "Not today."}]},
          {"heading": "INT. SHOP - DAY", "visual": "Eric comes in with the letter", "duration_seconds": 4,
           "dialogue": [{"character": "Eric Habimana", "text": "Read this."}]}]


def make_part(client, ai, gen, ren, store, u, pid):
    cast = store.select("studio_characters", {"project_id": pid})
    split(client, ai, gen, u, pid, [dict(s) for s in SCENES],
          characters=[{"name": c["name"], "description": c.get("description") or ""} for c in cast])
    q = client.post(f"/api/v1/studio/projects/{pid}/production/quote", headers=H(u), json={"direct": False}).json()
    r = client.post(f"/api/v1/studio/projects/{pid}/production", headers=H(u),
                    json={"request_id": uuid.uuid4().hex, "confirm_credits": q["total_credits"], "direct": False,
                          "storyboard_review": False})
    assert r.status_code == 200, r.text
    for _ in range(30):
        run(gen)
        run(ren)
        s = client.get(f"/api/v1/studio/projects/{pid}/production", headers=H(u)).json()["production"]
        if s["status"] in ("done", "needs_attention", "needs_approval", "canceled"):
            break
    return s


def test_parts_render_recover_from_a_failure_and_join_into_one_movie(nv, monkeypatch):
    client, db, store, files, ai, gen, ren, u = nv
    sid, out = feature(client, ai, gen, u, 30, 3)
    master_id, feature_id = out["feature"]["master_id"], out["feature"]["id"]
    parts = [m["project_id"] for m in out["created"]]
    # Nothing rendered yet: the join is refused before any credit is held, and says which parts are missing.
    est = client.post(f"/api/v1/studio/projects/{master_id}/estimate", headers=H(u), json={"kind": "render"}).json()
    assert est["blocked"] and "Render parts 1, 2, 3 first" in str(est["items"][0]["blocked"])
    ov = client.get(f"/api/v1/studio/features/{feature_id}", headers=H(u)).json()
    assert ov["ready_to_join"] is False and ov["missing"] == [1, 2, 3] and ov["credits_to_finish"] > 0
    # Part 2's first render fails: refunded, retried by the run, and the part still finishes.
    real = render.render_film
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 2:
            raise render.RenderError("ffmpeg crashed")
        return real(*a, **k)
    monkeypatch.setattr(render, "render_film", flaky)
    for pid in parts:
        assert make_part(client, ai, gen, ren, store, u, pid)["status"] == "done", pid
    failed = db.execute("select count(*), coalesce(sum(charged_credits), 0) from public.studio_jobs where project_id = %s "
                        "and kind = 'render' and status = 'failed'", (parts[1],)).fetchone()
    assert failed[0] == 1 and int(failed[1]) == 0, "the failed part render was refunded"
    ov = client.get(f"/api/v1/studio/features/{feature_id}", headers=H(u)).json()
    assert ov["ready_to_join"] is True and all(p["rendered"] for p in ov["parts"])
    # The join: one film, every part in order, no new music, checked like any film.
    before = balance(client, u)["available"]
    assert start(client, u, master_id, "render").status_code == 200
    run(ren)
    job = next(j for j in client.get(f"/api/v1/studio/projects/{master_id}/jobs", headers=H(u)).json()["jobs"]
               if j["kind"] == "render")
    assert job["status"] == "succeeded", job
    total = sum(p["seconds"] for p in ov["parts"])
    film = client.get(f"/api/v1/studio/features/{feature_id}", headers=H(u)).json()["film"]
    assert film and abs(film["duration_seconds"] - total) < 0.5, (film, total)
    spent = before - balance(client, u)["available"]
    assert 0 < spent < 20, "the join costs only its compute (no music, no AI)"
    usage = db.execute("select u.provider_cost_usd from public.studio_usage u join public.studio_jobs j on j.id = u.job_id "
                       "where j.project_id = %s and j.kind = 'render'", (master_id,)).fetchone()
    assert float(usage[0]) == 0
    # The parts' films fade only at the movie's start and end.
    for k, pid in enumerate(parts):
        scenes = sorted(store.select("studio_scenes", {"project_id": pid}), key=lambda s: s["position"])
        from studio_worker import StudioWorker
        p = store.one("studio_projects", {"id": pid})
        first = StudioWorker._film_edit(scenes[0], scenes, False, p)
        last = StudioWorker._film_edit(scenes[-1], scenes, False, p)
        assert (first["fade_in"] > 0) is (k == 0) and (last["fade_out"] > 0) is (k == 2), k
