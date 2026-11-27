"""Benchmark productions, made end to end with a fake AI (no paid calls) and a real ffmpeg render:

  short_drama    4 directed talking scenes, one provider failure on the way
  series         a 12-chapter approved novel planned into 3 episodes; every episode produced and rendered
  action_10min   50 scenes (fights, a chase, aftermath) of a 10-minute action film

Each one writes a scorecard (JSON) to $VIYO_BENCHMARK_DIR, or the test's temp folder, and asserts only what must
always hold: nothing charged above the approved budget, every failure refunded and recovered, the film rendered at
about its planned length with sound, characters' reference pictures sent with their shots, the cast and the story
memory carried across episodes, and the whole-film report run on the real render.

What these benchmarks CANNOT measure, and say so in the scorecard: whether real models keep a face the same, whether
action reads well, whether the story is good. Those need the admin-run live benchmark (prepared, not run).
"""
import json
import os
import re
import uuid

import pytest

import ad_qc
import creator_studio as cs
import studio_ai
import studio_novel_api
from conftest import ROOT, connect, make_user
from test_creator_studio import H, balance, env, grant, run, studio_schema  # noqa: F401
from test_story_bible import approve_novel, written_novel
from test_story_bible import bible_schema as _bible_schema  # noqa: F401
from test_story_chapters import chapters_schema as _chapters_schema  # noqa: F401
from test_story_episodes import good_plan, plan
from test_video_drama import DAN, MAYA, detail, project, split

pytestmark = pytest.mark.skipif(bool(os.environ.get("STUDIO_PREMIGRATION")), reason="needs the Studio schema")

NOT_MEASURED = ["real-model face consistency", "real-model action quality", "story quality as viewers feel it",
                "voice likeness in real clips"]


@pytest.fixture(scope="module")
def series_schema(_bible_schema):
    conn = connect()
    conn.execute((ROOT / "creator_studio_episode_plan.sql").read_text())
    conn.close()


def scene(heading, visual, secs, lines=(), who=("Maya", "Daniel"), transition="cut"):
    return {"heading": heading, "visual": visual, "duration_seconds": secs,
            "dialogue": [{"character": c, "text": t} for c, t in lines],
            "continuity": {"transition": transition, "characters": [{"name": n, "in_frame": True} for n in who]}}


SHORT = [
    scene("EXT. ROOFTOP - NIGHT", "Maya waits at the edge in the rain", 6, [("Maya", "You're late.")], ("Maya",), "new_place"),
    scene("EXT. ROOFTOP - NIGHT", "Daniel steps out of the stairwell", 7,
          [("Daniel", "I almost didn't come."), ("Maya", "But you did.")]),
    scene("EXT. ROOFTOP - NIGHT", "They stand close, the city below", 8,
          [("Daniel", "They know about the money."), ("Maya", "Then we leave tonight.")]),
    scene("INT. STAIRWELL - NIGHT", "Footsteps below; Maya pulls Daniel back", 6, [("Maya", "Someone's coming.")],
          transition="continuous"),
]


def action_film():
    """50 scenes of 12 s (the breakdown's longest scene), 10 minutes: set-up, fights, a chase, a fall, the aftermath."""
    beats = [("INT. WAREHOUSE - NIGHT", "Maya creeps between crates, gun drawn", [("Maya", "Stay behind me.")]),
             ("INT. WAREHOUSE - NIGHT", "Two guards turn; Daniel freezes", [("Daniel", "Maya!")]),
             ("INT. WAREHOUSE - NIGHT", "Maya disarms the first guard and throws him into the crates", []),
             ("INT. WAREHOUSE - NIGHT", "The second guard swings a pipe; Maya ducks and strikes", []),
             ("EXT. DOCKS - NIGHT", "They burst out into the rain and run along the pier", [("Daniel", "Which way?")]),
             ("EXT. DOCKS - NIGHT", "A car's headlights swing round behind them", []),
             ("EXT. STREET - NIGHT", "Chase: they sprint between market stalls, the car close behind", []),
             ("EXT. ROOFTOPS - NIGHT", "Maya leaps a gap between roofs; Daniel hesitates, then jumps", [("Maya", "Jump!")]),
             ("EXT. ROOFTOPS - NIGHT", "Daniel slips and hangs from the edge; Maya grabs his wrist", [("Daniel", "Don't let go.")]),
             ("EXT. ROOFTOPS - NIGHT", "Breathing hard, they lie on the roof as sirens pass", [("Maya", "We're alive.")]),
             ("INT. SAFE HOUSE - NIGHT", "Maya cleans a cut on Daniel's arm in silence", []),
             ("INT. SAFE HOUSE - NIGHT", "Daniel finally speaks", [("Daniel", "You've done this before."),
                                                                     ("Maya", "Too many times.")])]
    out = []
    for i in range(50):
        h, v, lines = beats[i % len(beats)]
        out.append(scene(h, f"{v} (part {i // len(beats) + 1})", 12,
                         lines if i < len(beats) else (), transition="new_place" if i % len(beats) == 0 else "cut"))
    return out


def produce(client, gen, ren, u, pid, rounds=40):
    q = client.post(f"/api/v1/studio/projects/{pid}/production/quote", headers=H(u), json={}).json()
    r = client.post(f"/api/v1/studio/projects/{pid}/production", headers=H(u),
                    json={"request_id": uuid.uuid4().hex, "confirm_credits": q["total_credits"], "storyboard_review": False})
    assert r.status_code == 200, r.text
    for _ in range(rounds):
        run(gen)
        run(ren)
        s = client.get(f"/api/v1/studio/projects/{pid}/production", headers=H(u)).json()["production"]
        if s["status"] in ("done", "needs_attention", "needs_approval", "canceled"):
            break
    return q, s


def film_seconds_with_sound(db, files, pid):
    path = db.execute("select path from public.studio_renders where project_id = %s order by version desc limit 1",
                      (pid,)).fetchone()[0]
    import tempfile
    local = os.path.join(tempfile.mkdtemp(), "film.mp4")
    files.get_to(path, local)
    info = ad_qc.probe(local)
    return info["duration"], info["has_audio"]


def scorecard(name, tmp_path, **fields):
    card = {"benchmark": name, "ai": "fake (no paid calls)", "render": "real ffmpeg", "not_measured": NOT_MEASURED, **fields}
    folder = os.environ.get("VIYO_BENCHMARK_DIR") or str(tmp_path)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, f"{name}.json"), "w") as f:
        json.dump(card, f, indent=2, default=str)
    print(f"\nBENCHMARK {name}: " + json.dumps({k: v for k, v in card.items() if k != "not_measured"}, default=str))
    return card


def report_summary(rep):
    m = rep["metrics"]
    return {"status": rep["status"], "problems": {c["key"]: c["count"] for c in rep["categories"]},
            "high": [p["problem"] for p in rep["problems"] if p["severity"] == "high"][:5],
            "found": [f'{p["category"]}: {p["problem"]}' for p in rep["problems"]][:12],
            "pieces": m["pieces"], "directed_scenes": m["directed_scenes"], "distinct_shot_types": m["distinct_shot_types"],
            "largest_type_share": m["largest_type_share"], "average_shot_seconds": m["average_shot_seconds"],
            "fingerprinted": m["fingerprinted"], "screen_time": m["screen_time"]}


def common_checks(client, db, u, pid, q, s, before, film):
    """What must hold for every benchmark."""
    approved = s.get("approved_credits") or q["total_credits"]
    assert s["status"] == "done", s
    assert 0 < s["spent_credits"] <= approved, "never more than the approved budget"
    assert balance(client, u)["held"] == 0 and balance(client, u)["available"] == before - s["spent_credits"]
    rep = client.get(f"/api/v1/studio/projects/{pid}/film-report", headers=H(u)).json()
    assert rep["metrics"]["fingerprinted"] == rep["metrics"]["pieces"], "every piece of the film fingerprinted"
    assert not [p for p in rep["problems"] if p["category"] == "missing"], rep["problems"]
    seconds, sound = film
    return rep, {"quoted_max": q["total_credits"], "approved": approved, "spent": s["spent_credits"],
                 "refunded": s.get("refunded_credits"), "within_budget": s["spent_credits"] <= approved,
                 "estimate_accuracy": round(s["spent_credits"] / q["total_credits"], 2)}, seconds, sound


# ------------------------------------------------------------------ short drama

def test_benchmark_short_drama(env, tmp_path):
    client, db, store, files, ai, gen, ren = env
    ai.varied_pictures = True
    u = make_user(db)
    grant(client, u, 200000)
    pid = project(client, u, production_style="video", target_seconds=30)["id"]
    split(client, ai, gen, u, pid, SHORT, characters=(MAYA, DAN))
    ai.fail["video"] = [studio_ai.ProviderError("boom")]                # one failed step on the way
    before = balance(client, u)["available"]
    q, s = produce(client, gen, ren, u, pid)
    film = film_seconds_with_sound(db, files, pid)
    rep, money, seconds, sound = common_checks(client, db, u, pid, q, s, before, film)
    failed = db.execute("select count(*), coalesce(sum(charged_credits), 0) from public.studio_jobs "
                        "where project_id = %s and status = 'failed'", (pid,)).fetchone()
    assert failed[0] >= 1 and failed[1] == 0, "the failed step was refunded, retried, and the film still finished"
    refs = [x["refs"] for x in ai.image_log if x["refs"]]
    assert refs, "characters' reference pictures go with their shots"
    assert sound and abs(seconds - sum(x["duration_seconds"] for x in SHORT)) <= 12
    scorecard("short_drama", tmp_path, scenes=len(SHORT), film_seconds=round(seconds, 1), has_sound=sound,
              money=money, recovery={"failed_steps": failed[0], "charged_for_failures": failed[1], "finished": True},
              identity={"pictures_with_references": len(refs), "pictures": len(ai.image_log)},
              report=report_summary(rep))


# ------------------------------------------------------------------ 3-episode series

def test_benchmark_series_of_three_episodes(series_schema, env, tmp_path, monkeypatch):
    client, db, store, files, ai, gen, ren = env
    monkeypatch.setattr(cs, "_ai", ai)
    client.app.include_router(studio_novel_api.router)
    ai.varied_pictures = True
    u = make_user(db)
    grant(client, u, 300000)
    sid = written_novel(client, ai, gen, u)
    assert approve_novel(client, u, sid).status_code == 200
    ai.novel_results = {"episode plan": good_plan(3)}
    d = plan(client, u, sid, episodes=3, seconds=60).json()
    made = client.post(f"/api/v1/studio/stories/{sid}/episodes/plan/approve", headers=H(u),
                       json={"version": d["version"]}).json()["created"]
    assert [m["number"] for m in made] == [1, 2, 3]
    from studio_worker import StudioWorker
    w = StudioWorker(store, files, ai, "generation")
    cards, identities = [], []
    for m in made:
        pid = m["project_id"]
        cast = store.select("studio_characters", {"project_id": pid})
        names = sorted(c["name"] for c in cast)
        identities.append({c["name"]: c.get("identity_id") for c in cast})
        first = lambda n: n.split(" ")[0]                                           # noqa: E731
        lead, other = names[0], names[1]
        split(client, ai, gen, u, pid, [
            scene("INT. TAILOR SHOP - DAY", f"{first(lead)} works at the sewing machine", 6,
                  [(lead, "Not today.")], (first(lead),), "new_place"),
            scene("INT. TAILOR SHOP - DAY", f"{first(other)} walks in holding the letter", 7,
                  [(other, "You need to read this."), (lead, "Where did you find it?")],
                  (first(lead), first(other))),
            scene("EXT. MARKET - DUSK", f"{first(lead)} reads the letter alone", 6, [], (first(lead),), "time_jump")],
            characters=[{"name": c["name"], "description": c.get("description") or ""} for c in cast])
        after = sorted(c["name"] for c in store.select("studio_characters", {"project_id": pid}))
        assert after == names, "the novel's cast is reused, never duplicated"
        before = balance(client, u)["available"]
        q, s = produce(client, gen, ren, u, pid)
        rep, money, seconds, sound = common_checks(client, db, u, pid, q, s, before,
                                                   film_seconds_with_sound(db, files, pid))
        ep = w.novel_episode(store.one("studio_projects", {"id": pid}))
        memory = (ep or {}).get("memory", "")
        span = next(e for e in d["plan"] if e["number"] == m["number"])
        happened = [int(x) for x in re.findall(r"Happened \(chapter (\d+)\)", memory)]
        at = int(re.search(r"STORY CANON at chapter (\d+)", memory).group(1))
        assert all(c <= span["chapter_to"] for c in happened), "the writer never sees events from later chapters"
        cards.append({"episode": m["number"], "chapters": [span["chapter_from"], span["chapter_to"]],
                      "film_seconds": round(seconds, 1), "has_sound": sound, "money": money,
                      "memory_at_chapter": at, "memory_events": happened,
                      "memory_in_writer_prompt": "STORY CANON at chapter" in memory, "report": report_summary(rep)})
    # The same people in every episode: each character keeps one identity across the series.
    assert all(i.keys() == identities[0].keys() for i in identities)
    assert all(i == identities[0] for i in identities), identities
    assert all(c["memory_in_writer_prompt"] for c in cards)
    points = [c["memory_at_chapter"] for c in cards]
    assert points == sorted(points) and points[0] < points[-1], "the memory moves forward with the story"
    scorecard("series", tmp_path, episodes=cards, cast_identity_kept=True,
              memory={"no_future_events_leaked": True, "memory_points_by_episode": points,
                      "every_episode_written_from_memory": True})


# ------------------------------------------------------------------ 10-minute action film

def test_benchmark_ten_minute_action_film(env, tmp_path):
    client, db, store, files, ai, gen, ren = env
    ai.varied_pictures = True
    u = make_user(db)
    grant(client, u, 500000)
    scenes = action_film()
    planned = sum(x["duration_seconds"] for x in scenes)
    assert 590 <= planned <= 610
    pid = project(client, u, production_style="video", target_seconds=600, genre="thriller")["id"]
    split(client, ai, gen, u, pid, scenes, characters=(MAYA, DAN))
    before = balance(client, u)["available"]
    q, s = produce(client, gen, ren, u, pid, rounds=80)
    film = film_seconds_with_sound(db, files, pid)
    rep, money, seconds, sound = common_checks(client, db, u, pid, q, s, before, film)
    assert sound and seconds >= planned * 0.85
    shots = db.execute("select c.position, count(s.id) from public.studio_scenes c left join public.studio_shots s "
                       "on s.scene_id = c.id where c.project_id = %s group by c.position order by c.position", (pid,)).fetchall()
    per_scene = [n for _p, n in shots]
    scorecard("action_10min", tmp_path, scenes=len(scenes), planned_seconds=planned, film_seconds=round(seconds, 1),
              has_sound=sound, money=money, report=report_summary(rep),
              action={"shots_per_scene_avg": round(sum(per_scene) / len(per_scene), 2), "shots_per_scene_max": max(per_scene),
                      "note": "Baseline before action intelligence: the director plans at most 4 shots per scene."})
