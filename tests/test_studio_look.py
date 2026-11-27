"""The look bible, film-wide shot variety and dissolves.

Pure checks (presets, the world's palette, the creator's edits, the prompt block, repetition, the fake director's
variation, the film report, a real ffmpeg dissolve), then the pipeline with a fake AI: every picture and clip gets
the identical look, it's frozen on the first picture, a half-made film never gets a new look half way, the creator's
edit is used from then on, a series shares one look, character sheets stay neutral, a rewrite keeps the look,
repeated shot patterns are re-planned, and a time jump dissolves. Then a look-off vs look-on comparison."""
import json
import os
import subprocess
import uuid

import pytest

import film_qc
import studio_film_report as fr
import studio_look as look
import studio_render as render
import studio_shots as ss
from conftest import make_user
from test_creator_studio import H, balance, env, grant, run, start, studio_schema  # noqa: F401
from test_video_drama import DAN, MAYA, project, split

pytestmark = pytest.mark.skipif(bool(os.environ.get("STUDIO_PREMIGRATION")), reason="needs the Studio schema")


# ------------------------------------------------------------------ the look itself

def test_each_genre_gets_its_own_look_and_the_world_and_creator_style_win():
    th = look.default_look({"genre": "Thriller"})
    ro = look.default_look({"genre": "Romance"})
    ac = look.default_look({"genre": "Action"})
    assert th["genre"] == "thriller" and ro["genre"] == "romance" and ac["genre"] == "action"
    assert th["light"] != ro["light"] and th["palette"] != ro["palette"] and "teal-and-orange" in ac["contrast_and_grade"]
    assert look.default_look({"genre": ""})["genre"] == "drama"
    w = look.default_look({"genre": "Thriller", "visual_style": "Kigali at night, wet streets"},
                          {"visual": {"palette": "red earth, banana-leaf green, indigo", "textures": "woven sisal"}})
    assert w["palette"] == ["red earth", "banana-leaf green", "indigo"], "the story world's colours win"
    assert "woven sisal" in w["texture"] and w["style"] == "Kigali at night, wet streets"
    assert set(look.FIELDS) <= set(w)


def test_the_creator_edits_only_what_they_change():
    base = look.default_look({"genre": "Drama"})
    out = look.clean({"light": "hard noon sun", "palette": "ochre; black; white", "nonsense": "x"}, base)
    assert out["light"] == "hard noon sun" and out["palette"] == ["ochre", "black", "white"]
    assert out["lens"] == base["lens"] and "nonsense" not in out
    assert out["edited"] == ["light", "palette"] and out["source"] == "creator"


def test_the_prompt_block_is_identical_text_with_every_part():
    lk = look.default_look({"genre": "Mystery", "visual_style": "noir"})
    b1, b2 = look.block(lk), look.block(dict(lk))
    assert b1 == b2 and b1.startswith("LOOK OF THE FILM")
    for label in ("Creator's style: noir", "Lens:", "Depth of field:", "Camera:", "Light:", "Colour palette:",
                  "Contrast and grade:", "Texture:", "Framing:", "Avoid:"):
        assert label in b1, label
    assert "time of day, weather and light still apply" in b1


def test_the_kill_switch(monkeypatch):
    monkeypatch.setattr(look, "ENABLED", False)
    assert look.block(look.default_look({})) == "" and look.recent_block([("wide_two_shot",)]) == ""
    assert look.repetition([{"shot_type": "medium"}], [("medium",)] * 3) == []


# ------------------------------------------------------------------ film-wide variety

P = ("wide_two_shot", "over_the_shoulder", "reaction")


def test_repeated_patterns_and_openings_are_sent_back():
    same = [{"shot_type": t} for t in P]
    assert look.repetition(same, [P, P])[0]["kind"] == "repetition"
    opening = [{"shot_type": "wide_two_shot"}, {"shot_type": "close_up"}]
    assert "also open on wide two shot" in look.repetition(opening, [("wide_two_shot", "medium")] * 3)[0]["problem"]
    varied = [{"shot_type": "close_up"}, {"shot_type": "wide_two_shot"}, {"shot_type": "reaction"}]
    assert look.repetition(varied, [P, P, P]) == []
    rb = look.recent_block([P, ("medium", "close_up")])
    assert "wide two shot -> over the shoulder -> reaction" in rb and "match cut" in rb
    assert [p["kind"] for p in ss.plan_quality({"dialogue": []}, same, [P, P])].count("repetition") == 1


def test_the_fake_director_varies_its_opening_when_told():
    raw = {"shots": [{"type": "wide_two_shot"}, {"type": "over_the_shoulder"}, {"type": "reaction"}]}
    prompt = "x" + look.recent_block([P, P])
    assert [s["type"] for s in ss.vary(raw, prompt)["shots"]][:2] == ["over_the_shoulder", "wide_two_shot"]
    beats = {"shots": [{"type": "establishing_wide", "beat": "setup"}, {"type": "medium", "beat": "escalation"}]}
    assert [x["beat"] for x in ss.vary(beats, "x" + look.recent_block([("establishing_wide",)] * 2))["shots"]] == \
        ["setup", "escalation"], "only the framing changes: the beats stay in order"
    assert ss.vary(raw, "no recent block") == raw


def test_the_film_report_flags_scenes_filmed_identically():
    scenes = [{"id": f"s{i}", "position": i, "duration_seconds": 6, "mode": "still", "image_path": f"i{i}",
               "dialogue": []} for i in range(1, 6)]
    shots = {f"s{i}": [{"id": f"s{i}-{k}", "shot_type": t, "duration_seconds": 2, "characters": [], "dialogue": [],
                        "image_path": f"s{i}-{k}.png"} for k, t in enumerate(P)] for i in range(1, 5)}
    rep = fr.build({"target_seconds": 30}, scenes, shots, [], {s["id"]: [] for s in scenes})
    hit = [p for p in rep["problems"] if "exactly the same shots in the same order" in p["problem"]]
    assert hit and hit[0]["problem"].startswith("Scenes 1 to 4")


# ------------------------------------------------------------------ dissolves

def test_a_dissolve_overlaps_two_scenes_and_the_final_check_expects_it(tmp_path):
    from PIL import Image
    imgs = []
    for i, c in enumerate([(200, 40, 40), (40, 40, 200)]):
        p = str(tmp_path / f"i{i}.png")
        Image.new("RGB", (540, 960), c).save(p)
        imgs.append(p)
    scenes = [render.SceneInput(duration=3, image=imgs[0]), render.SceneInput(duration=3, image=imgs[1], dissolve_in=0.8)]
    out = str(tmp_path / "film.mp4")
    info = render.render_film(scenes, "9:16", out, str(tmp_path), captions=False)
    assert abs(info["duration"] - 5.2) < 0.15, info["duration"]
    mid = subprocess.run(["ffmpeg", "-v", "error", "-ss", "2.6", "-i", out, "-frames:v", "1", "-vf", "scale=1:1",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
    assert mid[0] > 60 and mid[2] > 60, f"mid-dissolve frame mixes both scenes: {list(mid)}"
    plan = film_qc.plan_from_inputs(scenes)
    assert sum(p["duration"] for p in plan) == pytest.approx(5.2)
    assert film_qc.inspect_film(out, plan, "9:16")["passed"]


# ------------------------------------------------------------------ the pipeline (fake AI, local Postgres)

def sc(heading, visual, lines=(), transition="cut", secs=6):
    return {"heading": heading, "visual": visual, "duration_seconds": secs,
            "dialogue": [{"character": c, "text": t} for c, t in lines],
            "continuity": {"transition": transition,
                           "characters": [{"name": "Maya", "in_frame": True}, {"name": "Daniel", "in_frame": True}]}}


TALKS = [sc("INT. FLAT - NIGHT", "Maya waits at the table", [("Maya", "You're late."), ("Daniel", "I know.")], "new_place"),
         sc("INT. FLAT - NIGHT", "Daniel sits across from her", [("Daniel", "They found me."), ("Maya", "Who?")]),
         sc("INT. FLAT - NIGHT", "They lean closer over the table", [("Maya", "What do we do?"), ("Daniel", "We run.")]),
         sc("INT. FLAT - NIGHT", "Maya stands and paces", [("Maya", "Tonight?"), ("Daniel", "Now.")]),
         sc("EXT. BUS STATION - DAWN", "Maya and Daniel wait with one bag", [("Daniel", "Ready?"), ("Maya", "No.")],
            "time_jump")]


def produce(client, gen, ren, u, pid, rounds=40, **body):
    q = client.post(f"/api/v1/studio/projects/{pid}/production/quote", headers=H(u), json=body).json()
    r = client.post(f"/api/v1/studio/projects/{pid}/production", headers=H(u),
                    json={"request_id": uuid.uuid4().hex, "confirm_credits": q["total_credits"], "storyboard_review": False,
                          **body})
    assert r.status_code == 200, r.text
    for _ in range(rounds):
        run(gen)
        run(ren)
        s = client.get(f"/api/v1/studio/projects/{pid}/production", headers=H(u)).json()["production"]
        if s["status"] in ("done", "needs_attention", "needs_approval", "canceled"):
            break
    return q, s


def film(env, genre="Thriller", style="video", scenes=TALKS, **over):
    client, db, store, files, ai, gen, ren = env
    ai.varied_pictures = True
    u = make_user(db)
    grant(client, u, 300000)
    pid = project(client, u, production_style=style, genre=genre, **over)["id"]
    split(client, ai, gen, u, pid, [dict(s) for s in scenes], characters=(MAYA, DAN))
    return client, db, store, files, ai, gen, ren, u, pid


def picture_prompts(ai, start=0):
    return [x["prompt"] for x in getattr(ai, "image_log", [])[start:] if "SHOT CONTINUITY" in x["prompt"] or "SCENE:" in x["prompt"]]


def test_every_picture_gets_the_same_look_frozen_on_the_first_picture(env):
    client, db, store, files, ai, gen, ren, u, pid = film(env)
    assert client.get(f"/api/v1/studio/projects/{pid}/look", headers=H(u)).json()["saved"] is False
    before = balance(client, u)["available"]
    q, s = produce(client, gen, ren, u, pid)
    assert s["status"] == "done"
    prompts = picture_prompts(ai)
    blocks = {p.split("LOOK OF THE FILM", 1)[1].split("\nSHOT CONTINUITY", 1)[0].split("\nSCENE CONTEXT", 1)[0]
              for p in prompts if "LOOK OF THE FILM" in p}
    assert prompts and all("LOOK OF THE FILM" in p for p in prompts), "every scene and shot picture gets the look"
    assert len(blocks) == 1, "and it's the identical look for every one"
    assert "teal-and-amber" in next(iter(blocks)), "the thriller look"
    saved = client.get(f"/api/v1/studio/projects/{pid}/look", headers=H(u)).json()
    assert saved["saved"] is True and saved["applies"] is True and saved["look"]["genre"] == "thriller"
    sheets = [x["prompt"] for x in ai.image_log if "SHOT CONTINUITY" not in x["prompt"] and "SCENE:" not in x["prompt"]]
    assert sheets and not any("LOOK OF THE FILM" in p for p in sheets), "character sheets stay neutral references"
    assert balance(client, u)["available"] == before - s["spent_credits"], "the look itself costs nothing"


def test_the_creators_edit_is_used_for_the_pictures_made_after(env):
    client, db, store, files, ai, gen, ren, u, pid = film(env, style="pictures")
    r = client.put(f"/api/v1/studio/projects/{pid}/look", headers=H(u),
                   json={"light": "hard noon sun, no shadows hidden", "palette": ["ochre", "black"]})
    assert r.status_code == 200 and r.json()["look"]["edited"] == ["light", "palette"]
    n = len(getattr(ai, "image_log", []))
    produce(client, gen, ren, u, pid)
    prompts = picture_prompts(ai, n)
    assert prompts and all("hard noon sun, no shadows hidden" in p and "ochre, black" in p for p in prompts)
    reset = client.put(f"/api/v1/studio/projects/{pid}/look", headers=H(u), json={"reset": True}).json()
    assert reset["look"]["source"] == "default" and reset["look"]["light"] != "hard noon sun, no shadows hidden"
    assert client.put(f"/api/v1/studio/projects/{uuid.uuid4()}/look", headers=H(u), json={}).status_code == 404


def test_a_half_made_film_never_gets_a_new_look_half_way(env, monkeypatch):
    client, db, store, files, ai, gen, ren, u, pid = film(env, style="pictures")
    monkeypatch.setattr(look, "ENABLED", False)          # pictures made before the look bible existed
    first = client.get(f"/api/v1/studio/projects/{pid}", headers=H(u)).json()["scenes"][0]["id"]
    start(client, u, pid, "image", [first])
    run(gen)
    monkeypatch.setattr(look, "ENABLED", True)
    st = client.get(f"/api/v1/studio/projects/{pid}/look", headers=H(u)).json()
    assert st["saved"] is False and st["applies"] is False and "started before the look bible" in st["note"]
    n = len(getattr(ai, "image_log", []))
    second = client.get(f"/api/v1/studio/projects/{pid}", headers=H(u)).json()["scenes"][1]["id"]
    start(client, u, pid, "image", [second])
    run(gen)
    assert not any("LOOK OF THE FILM" in p for p in picture_prompts(ai, n)), "no new look half way"
    client.put(f"/api/v1/studio/projects/{pid}/look", headers=H(u), json={"light": "candlelight"})
    third = client.get(f"/api/v1/studio/projects/{pid}", headers=H(u)).json()["scenes"][2]["id"]
    n = len(getattr(ai, "image_log", []))
    start(client, u, pid, "image", [third])
    run(gen)
    assert any("candlelight" in p for p in picture_prompts(ai, n)), "unless the creator saves one"


def test_a_rewrite_keeps_the_look_and_a_series_shares_one(env):
    client, db, store, files, ai, gen, ren, u, pid = film(env, style="pictures")
    client.put(f"/api/v1/studio/projects/{pid}/look", headers=H(u), json={"light": "neon rain"})
    client.patch(f"/api/v1/studio/projects/{pid}", headers=H(u), json={"script": ""})
    assert start(client, u, pid, "script").status_code == 200
    run(gen)
    bible = db.execute("select story_bible from public.studio_projects where id = %s", (pid,)).fetchone()[0]
    assert (bible or {}).get("look", {}).get("light") == "neon rain", "a new draft keeps the film's look"
    from studio_worker import StudioWorker
    sid = db.execute("select series_id from public.studio_projects where id = %s", (pid,)).fetchone()[0]
    if sid is None:
        sid = str(uuid.uuid4())
        db.execute("update public.studio_projects set series_id = %s where id = %s", (sid, pid))
    other = project(client, u, production_style="pictures", genre="Romance")["id"]
    db.execute("update public.studio_projects set series_id = %s where id = %s", (sid, other))
    w = StudioWorker(store, files, ai, "generation")
    lk = w.film_look(store.one("studio_projects", {"id": other}))
    assert lk["light"] == "neon rain", "a new episode uses the series' look, not its own genre's"


def test_repeated_scenes_are_re_planned_and_a_time_jump_dissolves(env):
    client, db, store, files, ai, gen, ren, u, pid = film(env)
    q, s = produce(client, gen, ren, u, pid)
    assert s["status"] == "done"
    rows = db.execute("select c.position, string_agg(s.shot_type, ',' order by s.position) from public.studio_scenes c "
                      "join public.studio_shots s on s.scene_id = c.id where c.project_id = %s group by c.position "
                      "order by c.position", (pid,)).fetchall()
    openings = [p.split(",")[0] for _pos, p in rows]
    assert len(set(openings)) >= 2, openings
    rep = client.get(f"/api/v1/studio/projects/{pid}/film-report", headers=H(u)).json()
    assert not any("exactly the same shots" in p["problem"] for p in rep["problems"])
    planned = sum(x["duration_seconds"] for x in TALKS)
    assert rep["render"]["duration_seconds"] < planned + 6, "the time jump overlaps (dissolves) instead of adding black"


# ------------------------------------------------------------------ look off vs on

def test_benchmark_look_off_versus_on(env, tmp_path, monkeypatch):
    def one(enabled):
        monkeypatch.setattr(look, "ENABLED", enabled)
        client, db, store, files, ai, gen, ren, u, pid = film(env)
        n = len(getattr(ai, "image_log", []))
        q, s = produce(client, gen, ren, u, pid)
        assert s["status"] == "done" and s["spent_credits"] <= s["approved_credits"]
        prompts = picture_prompts(ai, n)
        with_look = [p for p in prompts if "LOOK OF THE FILM" in p]
        blocks = {p.split("LOOK OF THE FILM", 1)[1][:600] for p in with_look}
        rows = db.execute("select c.position, string_agg(s.shot_type, ',' order by s.position) from public.studio_scenes c "
                          "join public.studio_shots s on s.scene_id = c.id where c.project_id = %s group by c.position "
                          "order by c.position", (pid,)).fetchall()
        pats = [p for _x, p in rows]
        rep = client.get(f"/api/v1/studio/projects/{pid}/film-report", headers=H(u)).json()
        monkeypatch.setattr(look, "ENABLED", True)
        return {"pictures": len(prompts), "pictures_with_look": len(with_look), "distinct_look_texts": len(blocks),
                "distinct_openings": len({p.split(',')[0] for p in pats}), "distinct_patterns": len(set(pats)),
                "identical_pattern_runs": sum(1 for p in rep["problems"] if "exactly the same shots" in p["problem"]),
                "film_seconds": rep["render"]["duration_seconds"], "spent": s["spent_credits"]}
    off, on = one(False), one(True)
    card = {"benchmark": "look_bible_ab", "ai": "fake (no paid calls)", "render": "real ffmpeg", "look_off": off,
            "look_on": on, "not_measured": ["whether real pictures actually share one look"]}
    folder = os.environ.get("VIYO_BENCHMARK_DIR") or str(tmp_path)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "look_bible_ab.json"), "w") as f:
        json.dump(card, f, indent=2)
    print("\nBENCHMARK look_bible_ab: " + json.dumps(card))
    assert off["pictures_with_look"] == 0 and on["pictures_with_look"] == on["pictures"] and on["distinct_look_texts"] == 1
    assert on["distinct_openings"] >= off["distinct_openings"] and on["identical_pattern_runs"] == 0
    assert on["identical_pattern_runs"] <= off["identical_pattern_runs"]
    assert on["spent"] == off["spent"] or abs(on["spent"] - off["spent"]) <= 0.1 * off["spent"], "the look costs nothing"
