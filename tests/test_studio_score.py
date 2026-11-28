"""The planned score: cues that follow the story, planned silences, character themes, a small known cost.

Pure checks (moods, grouping, the cue cap, themes, prompts and reuse keys, the film timeline, a real ffmpeg score bed,
pricing), then the pipeline with a fake AI and a fake music maker: the render makes each cue once, places it under its
scenes, charges only the cues it made, reuses unchanged cues, leaves a failed cue silent, respects the creator's own
music and "music off", and the quote holds for the new cues. Then score off vs on."""
import json
import os
import subprocess
import uuid

import pytest

import studio_music
import studio_pricing as pricing
import studio_render as render
import studio_score as score
from conftest import make_user
from test_creator_studio import H, balance, env, grant, run, start, studio_schema  # noqa: F401
from test_video_drama import DAN, MAYA, project, split

pytestmark = pytest.mark.skipif(bool(os.environ.get("STUDIO_PREMIGRATION")), reason="needs the Studio schema")


def sc(i, visual, lines=(), secs=8, transition="cut"):
    return {"id": f"s{i}", "position": i, "visual": visual, "duration_seconds": secs,
            "dialogue": [{"character": c, "text": t, **({"emotion": e} if e else {})} for c, t, e in lines],
            "continuity": {"transition": transition}}


CAST = [{"name": "Maya Lin", "personality": "brave, determined"}, {"name": "Daniel", "personality": "serious, protective"},
        {"name": "Grace", "personality": "warm"}]
FILM = [sc(0, "Maya waits at the bus stop at night", [("Maya", "He's late.", "")]),
        sc(1, "Daniel arrives; they talk quietly", [("Daniel", "They know.", ""), ("Maya", "How?", "")]),
        sc(2, "A car swerves; two men attack and Maya fights them off", [], 12),
        sc(3, "They hide behind the bins, breathing hard", [("Daniel", "Are you hurt?", "")]),
        sc(4, "Maya reads the letter. She says nothing.", [], 6),
        sc(5, "Dawn. They board the bus together", [("Maya", "Let's go.", "")], 8, "time_jump")]


# ------------------------------------------------------------------ the cue sheet

def test_each_scene_gets_a_mood_from_the_story():
    p = {"genre": "Thriller"}
    cues = score.sheet(p, FILM, CAST)
    moods = [(c["positions"], c["mood"]) for c in cues]
    assert moods == [([1, 2], "tension"), ([3], "action"), ([4], "aftermath"), ([5], "silence"), ([6], "finale")]
    assert next(c for c in cues if c["mood"] == "silence")["silent"] is True
    assert [c["mood"] for c in score.sheet({"genre": "Romance"}, FILM[:2], CAST)] == ["tender", "finale"]
    assert score.sheet({"genre": "Romance"}, FILM[:3] + [FILM[5]], CAST)[0]["mood"] == "tender"
    shock = [sc(0, "Grace opens the door", [("Grace", "He's dead.", "devastated")]), sc(1, "After", [], 6)]
    assert score.sheet({"genre": "Drama"}, shock, CAST)[0]["mood"] == "silence", "a shock is held in silence"


def test_a_time_jump_starts_a_new_cue_and_long_films_are_capped():
    same = [sc(0, "Maya walks", [], 12), sc(1, "Maya walks on", [], 12, "time_jump"), sc(2, "end", [], 12)]
    assert len(score.sheet({"genre": "Drama"}, same, CAST)) == 3
    many = []
    for i in range(20):
        many.append(sc(i, "They fight hard" if i % 2 else "They talk", [("Maya", "x", "")], 6))
    cues = score.sheet({"genre": "Drama"}, many, CAST)
    music = [c for c in cues if not c["silent"]]
    assert len(music) <= score.MAX_CUES and len(music) <= -(-120 // score.SECONDS_PER_CUE)
    assert sum(len(c["scenes"]) for c in cues) == 20, "every scene is still covered"
    keep = [sc(0, "They fight", [], 6), sc(1, "She says nothing.", [], 4), sc(2, "They fight again", [], 6),
            sc(3, "They talk", [("Maya", "x", "")], 4)]
    out = score.sheet({"genre": "Drama"}, keep, CAST)
    assert [c["mood"] for c in out].count("silence") == 1, "a planned silence is never merged away"


def test_main_characters_get_a_theme_carried_into_their_cues():
    th = score.themes(CAST, FILM)
    assert list(th) == ["Maya Lin", "Daniel"] and "brass" in th["Maya Lin"] and "low-string" in th["Daniel"]
    present = {s["id"]: ["Maya Lin", "Daniel"] for s in FILM}
    cue = score.sheet({"genre": "Thriller"}, FILM, CAST, present)[0]
    prompt = score.cue_prompt({"genre": "Thriller"}, cue)
    assert "Mood: tension" in prompt and "Intensity 3 of 5" in prompt and "Maya Lin's theme: a bold brass call" in prompt
    full = studio_music.music_prompt({"genre": "Thriller", "logline": "x", "_cue": cue})
    assert "music cue for one part of a film" in full and "No vocals" in full and "Do not imitate" in full


def test_unchanged_cues_are_reused_and_changed_ones_remade():
    p = {"genre": "Thriller", "logline": "a"}
    cues = score.sheet(p, FILM, CAST)
    assert len(score.to_make(p, cues)) == 4, "4 music cues; the silence costs nothing"
    saved = [{"key": score.cue_key(p, c), "path": f"m{c['index']}.mp3"} for c in cues if not c["silent"]]
    p2 = {**p, "story_bible": {"score": {"cues": saved}}}
    assert score.to_make(p2, cues) == []
    assert len(score.to_make({**p2, "genre": "Romance"}, score.sheet({**p2, "genre": "Romance"}, FILM, CAST))) >= 1


def test_the_film_timeline_takes_dissolves_out():
    assert score.timeline([4, 5, 6], [0, 0, 0.8]) == [(0, 4), (4, 9), (8.2, 14.2)]


def test_pricing_holds_for_each_new_cue():
    assert pricing.render_cost(30, music_tracks=3).provider_cost_usd == pytest.approx(0.24)
    assert pricing.render_cost(30, music_generated=True).provider_cost_usd == pytest.approx(0.08)
    assert pricing.render_cost(30, music_tracks=0).provider_cost_usd == 0


def test_the_score_bed_places_cues_and_keeps_planned_silences(tmp_path):
    tones = []
    for f in (220, 440):
        p = str(tmp_path / f"t{f}.mp3")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"sine=f={f}:d=1.5", p], check=True)
        tones.append(p)
    out = str(tmp_path / "bed.wav")
    render.score_bed([(tones[0], 0.0, 2.5), (tones[1], 4.0, 6.0)], 6.0, out)   # 2.5-4.0 s is a planned silence
    info = render.probe(out)
    assert abs(info["duration"] - 6.0) < 0.1
    log = subprocess.run(["ffmpeg", "-i", out, "-af", "silencedetect=n=-45dB:d=1", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    starts = [float(x.split("silence_start: ")[1].split()[0]) for x in log.splitlines() if "silence_start" in x]
    assert any(2.0 <= s <= 2.8 for s in starts), log[-400:]
    assert not any(s < 1.5 for s in starts), "the first cue loops to fill its 2.5 s stretch"


# ------------------------------------------------------------------ the pipeline

def tone(tmp_path, f=220):
    p = tmp_path / f"cue{f}.mp3"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"sine=f={f}:d=4", str(p)], check=True)
    return p.read_bytes()


def scored_film(env, scenes=None, **over):
    client, db, store, files, ai, gen, ren = env
    u = make_user(db)
    grant(client, u, 100000)
    pid = project(client, u, production_style="pictures", genre="Thriller", **over)["id"]
    raw = [{"heading": "EXT. STREET - NIGHT", "visual": s["visual"], "duration_seconds": s["duration_seconds"],
            "dialogue": [{"character": d["character"].split(" ")[0], "text": d["text"]} for d in s["dialogue"]],
            "continuity": s["continuity"]} for s in (scenes or FILM)]
    split(client, ai, gen, u, pid, raw, characters=(MAYA, DAN))
    for sid in [s["id"] for s in client.get(f"/api/v1/studio/projects/{pid}", headers=H(u)).json()["scenes"]]:
        start(client, u, pid, "image", [sid])
        run(gen)
    return client, db, store, files, ai, gen, ren, u, pid


def render_job(client, ren, u, pid):
    r = start(client, u, pid, "render")
    assert r.status_code == 200, r.text
    run(ren)
    return next(j for j in client.get(f"/api/v1/studio/projects/{pid}/jobs", headers=H(u)).json()["jobs"]
                if j["kind"] == "render")


def music_usage(db):
    return float(db.execute("select provider_cost_usd from public.studio_usage u join public.studio_jobs j on j.id = u.job_id "
                            "where j.kind = 'render' and u.status = 'charged' order by u.id desc limit 1").fetchone()[0])


def test_the_render_makes_each_cue_once_places_it_and_charges_only_what_it_made(env, tmp_path):
    client, db, store, files, ai, gen, ren, u, pid = scored_film(env)
    calls = []
    ren.make_music = lambda p, secs: calls.append(p["_cue"]) or tone(tmp_path, 200 + 40 * len(calls))
    sheet = client.get(f"/api/v1/studio/projects/{pid}/score", headers=H(u)).json()
    assert sheet["uses"] == "planned score" and sheet["new_cues"] == 4 and sheet["music_credits"] == pricing.credits_for(0.32)
    assert [c["mood"] for c in sheet["cues"]] == ["tension", "action", "aftermath", "silence", "finale"]
    est = client.post(f"/api/v1/studio/projects/{pid}/estimate", headers=H(u), json={"kind": "render"}).json()
    assert est["total_credits"] >= pricing.credits_for(0.32), "the quote holds for the new cues"
    job = render_job(client, ren, u, pid)
    assert job["status"] == "succeeded", job
    assert [c["mood"] for c in calls] == ["tension", "action", "aftermath", "finale"], "silence makes no music"
    assert music_usage(db) == pytest.approx(0.32, abs=0.01), "4 cues x $0.08, no more"
    bible = db.execute("select story_bible, music_path, music_name from public.studio_projects where id = %s", (pid,)).fetchone()
    cues = bible[0]["score"]["cues"]
    assert sum(1 for c in cues if c.get("path")) == 4 and bible[1] is None and bible[2] == "Viyo Original Score"
    assert client.get(f"/api/v1/studio/projects/{pid}", headers=H(u)).json()["project"]["has_music"] is True
    # A second render reuses every cue: no new music, nothing charged for music.
    assert render_job(client, ren, u, pid)["status"] == "succeeded" and len(calls) == 4
    assert music_usage(db) < 0.01
    assert client.get(f"/api/v1/studio/projects/{pid}/score", headers=H(u)).json()["new_cues"] == 0


def test_a_failed_cue_stays_silent_the_film_is_made_and_only_made_cues_are_charged(env, tmp_path):
    client, db, store, files, ai, gen, ren, u, pid = scored_film(env)
    calls = []

    def maker(p, secs):
        calls.append(p["_cue"]["mood"])
        if p["_cue"]["mood"] == "action":
            raise studio_music.MusicGenerationError("busy", retry=True)
        return tone(tmp_path)
    ren.make_music = maker
    assert render_job(client, ren, u, pid)["status"] == "succeeded"
    assert music_usage(db) == pytest.approx(0.24, abs=0.01)
    cues = db.execute("select story_bible from public.studio_projects where id = %s", (pid,)).fetchone()[0]["score"]["cues"]
    assert next(c for c in cues if c["mood"] == "action").get("failed") is True
    assert client.get(f"/api/v1/studio/projects/{pid}/score", headers=H(u)).json()["new_cues"] == 1, "tried again next time"


def test_the_creators_music_and_music_off_are_respected(env, tmp_path):
    client, db, store, files, ai, gen, ren, u, pid = scored_film(env, scenes=FILM[:2])
    ren.make_music = lambda p, secs: pytest.fail("no score when the creator turned music off")
    client.patch(f"/api/v1/studio/projects/{pid}", headers=H(u), json={"auto_music": False})
    assert client.get(f"/api/v1/studio/projects/{pid}/score", headers=H(u)).json()["uses"] == "no music"
    assert render_job(client, ren, u, pid)["status"] == "succeeded" and music_usage(db) == 0
    client.patch(f"/api/v1/studio/projects/{pid}", headers=H(u), json={"auto_music": True})
    path = f"{u}/{pid}/music/own.mp3"
    files.put(path, tone(tmp_path), "audio/mpeg") if hasattr(files, "put") else None
    db.execute("update public.studio_projects set music_path = %s, music_name = 'My song' where id = %s", (path, pid))
    ren.make_music = lambda p, secs: pytest.fail("the creator's own music wins")
    assert client.get(f"/api/v1/studio/projects/{pid}/score", headers=H(u)).json()["uses"] == "your music"
    assert render_job(client, ren, u, pid)["status"] == "succeeded"


def test_the_kill_switch_makes_one_track_as_before(env, tmp_path, monkeypatch):
    monkeypatch.setattr(score, "ENABLED", False)
    client, db, store, files, ai, gen, ren, u, pid = scored_film(env, scenes=FILM[:3])
    calls = []
    ren.make_music = lambda p, secs: calls.append(p.get("_cue")) or tone(tmp_path)
    assert render_job(client, ren, u, pid)["status"] == "succeeded"
    assert calls == [None] and music_usage(db) == pytest.approx(0.08, abs=0.01)


def test_benchmark_score_off_versus_on(env, tmp_path, monkeypatch):
    def one(enabled):
        monkeypatch.setattr(score, "ENABLED", enabled)
        client, db, store, files, ai, gen, ren, u, pid = scored_film(env)
        calls = []
        ren.make_music = lambda p, secs: calls.append(p.get("_cue")) or tone(tmp_path, 180 + 30 * len(calls))
        job = render_job(client, ren, u, pid)
        assert job["status"] == "succeeded"
        cues = ((db.execute("select story_bible from public.studio_projects where id = %s", (pid,)).fetchone()[0] or {})
                .get("score") or {}).get("cues") or []
        monkeypatch.setattr(score, "ENABLED", True)
        return {"tracks_made": len(calls), "moods": [c["mood"] for c in cues] or ["one looped track"],
                "planned_silences": sum(1 for c in cues if c.get("silent")),
                "themed_cues": sum(1 for c in cues if c.get("themes")),
                "music_cost_usd": round(music_usage(db), 2), "music_credits": pricing.credits_for(0.08 * len(calls))}
    off, on = one(False), one(True)
    card = {"benchmark": "score_ab", "ai": "fake music maker (no paid calls)", "render": "real ffmpeg",
            "score_off": off, "score_on": on,
            "not_measured": ["how real Lyria cues sound", "whether a theme's melody really recurs across cues"]}
    folder = os.environ.get("VIYO_BENCHMARK_DIR") or str(tmp_path)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "score_ab.json"), "w") as f:
        json.dump(card, f, indent=2)
    print("\nBENCHMARK score_ab: " + json.dumps(card))
    assert off["tracks_made"] == 1 and on["tracks_made"] == 4 and on["planned_silences"] == 1
    assert len(set(on["moods"])) == 5 and on["music_cost_usd"] == pytest.approx(0.32, abs=0.01)
