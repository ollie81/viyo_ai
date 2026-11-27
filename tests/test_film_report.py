"""The whole-film report: free, changes nothing, and names each problem with its fix.

Pure checks on hand-made productions (each problem found, a clean film raises nothing), real ffmpeg fingerprints,
and one end-to-end run: a directed production is rendered (fake AI, real ffmpeg), the render job stores a
fingerprint per shot, and the report finds a reused picture and a film that is older than its plan."""
import os
import subprocess
import uuid

import pytest

import studio_film_report as fr
from test_creator_studio import H, balance, env, grant, run, studio_schema  # noqa: F401

pytestmark = pytest.mark.skipif(bool(os.environ.get("STUDIO_PREMIGRATION")), reason="needs the Studio schema")

CAST = [{"name": "Maya Lin"}, {"name": "Daniel Okoro"}, {"name": "Grace"}]


def scene(i, lines=(), secs=6, mode="video", **kw):
    return {"id": f"s{i}", "position": i - 1, "duration_seconds": secs, "mode": mode, "image_path": f"img{i}",
            "video_path": f"vid{i}" if mode == "video" else None,
            "dialogue": [{"character": c, "text": t} for c, t in lines], **kw}


def shot(sid, k, kind, names, secs=3, lines=(), move="static", **kw):
    return {"id": f"{sid}-{k}", "position": k, "shot_type": kind, "camera_movement": move, "duration_seconds": secs,
            "characters": [{"name": n} for n in names], "dialogue": [{"character": c, "text": t} for c, t in lines],
            "image_path": f"{sid}-{k}.png", "video_path": f"{sid}-{k}.mp4", **kw}


def clean_film():
    """Four directed scenes with varied coverage, lines carried by shots, a held ending."""
    scenes = [scene(1, [("Maya", "You came back.")]), scene(2, [("Daniel", "I never left.")]),
              scene(3, [("Maya", "Then why run?")], secs=7), scene(4, [("Daniel", "To find you.")], secs=7)]
    shots = {
        "s1": [shot("s1", 0, "establishing_wide", ["Maya"], 2.5, move="pan"),
               shot("s1", 1, "close_up", ["Maya"], 3.5, [("Maya", "You came back.")], move="slow push in")],
        "s2": [shot("s2", 0, "medium", ["Daniel"], 4, [("Daniel", "I never left.")]),
               shot("s2", 1, "reaction", ["Maya"], 2, move="slow push in")],
        "s3": [shot("s3", 0, "over_the_shoulder", ["Maya", "Daniel"], 4.5, [("Maya", "Then why run?")], move="dolly"),
               shot("s3", 1, "close_up", ["Daniel"], 2.5)],
        "s4": [shot("s4", 0, "wide_two_shot", ["Maya", "Daniel"], 3, move="pull out"),
               shot("s4", 1, "close_up", ["Daniel"], 4, [("Daniel", "To find you.")])],
    }
    presence = {"s1": ["Maya Lin"], "s2": ["Daniel Okoro", "Maya Lin"], "s3": ["Maya Lin", "Daniel Okoro"],
                "s4": ["Maya Lin", "Daniel Okoro"]}
    return scenes, shots, presence


def report(scenes, shots, presence, cast=CAST[:2], **kw):
    return fr.build({"target_seconds": 26}, scenes, shots, cast, presence, **kw)


def cats(rep):
    return {p["category"] for p in rep["problems"]}


def test_a_clean_film_raises_nothing():
    scenes, shots, presence = clean_film()
    segs = [{"scene": i, "hash": f"{h:016x}", "path": None} for i, h in
            enumerate([0x0F0F0F0F0F0F0F0F, 0xF0F0F0F0F0F0F0F0, 0x00FF00FF00FF00FF, 0xFF00FF00FF00FF00,
                       0x0000FFFF0000FFFF, 0xFFFF0000FFFF0000, 0x123456789ABCDEF0, 0xFEDCBA9876543210])]
    rep = report(scenes, shots, presence, segments=segs)
    assert rep["problems"] == [] and rep["status"] == "ok", rep["problems"]
    assert {c["key"]: c["status"] for c in rep["categories"]} == {c: "ok" for c in fr.CATEGORIES}
    assert rep["free"] is True and "changes nothing" in rep["note"]
    assert rep["metrics"]["distinct_shot_types"] == 6 and rep["metrics"]["screen_time"]["Maya Lin"]["shots"] == 5


def test_duplicates_the_same_file_and_near_identical_frames():
    scenes, shots, presence = clean_film()
    shots["s3"][1]["image_path"] = shots["s3"][1]["video_path"] = shots["s1"][1]["video_path"]
    segs = [{"scene": 1, "shot": 1, "hash": "0f0f0f0f0f0f0f0f"}, {"scene": 2, "shot": 1, "hash": "0f0f0f0f0f0f0f0e"},
            {"scene": 2, "shot": 2, "hash": "f0f0f0f0f0f0f0f0"}]
    rep = report(scenes, shots, presence, segments=segs)
    dup = [p for p in rep["problems"] if p["category"] == "duplicates"]
    assert any("exactly the same picture" in p["problem"] and (p["scene"], p["shot"]) == (3, 2) for p in dup)
    assert any("almost identical to Scene 1, shot 1" in p["problem"] and p["scene"] == 2 for p in dup)
    assert all(p["route"] in fr.ROUTES for p in rep["problems"])
    assert fr.distance("0f0f0f0f0f0f0f0f", "0f0f0f0f0f0f0f0e") == 1


def test_missing_pictures_clips_lines_and_an_outdated_film():
    scenes, shots, presence = clean_film()
    shots["s2"][1]["image_path"] = shots["s2"][1]["video_path"] = None
    shots["s4"][0]["video_path"] = None                       # a video scene with only a still
    shots["s3"][0]["dialogue"] = []                           # "Then why run?" is in no shot
    rep = report(scenes, shots, presence)
    texts = [p["problem"] for p in rep["problems"] if p["category"] == "missing"]
    assert any("Scene 2, shot 2 has no picture yet" in t for t in texts)
    assert any("Scene 4, shot 1 is a video scene but has only a still" in t for t in texts)
    assert any("\"Then why run?\" isn't in any shot" in t for t in texts)
    assert rep["status"] == "problems"
    # The render remembers which file each piece used; a remade shot makes the film outdated.
    scenes, shots, presence = clean_film()
    units = fr._units(scenes, shots)
    segs = [{"scene_id": u["scene_id"], "shot_id": u["shot_id"], "path": u["path"], "hash": None} for u in units]
    render = {"version": 1, "duration_seconds": 26}
    assert "missing" not in cats(report(scenes, shots, presence, segments=segs, render=render))
    shots["s2"][0]["video_path"] = "s2-0-new.mp4"
    out = [p for p in report(scenes, shots, presence, segments=segs, render=render)["problems"] if p["category"] == "missing"]
    assert out and "rendered before scene 2 changed" in out[0]["problem"] and out[0]["route"] == "edit"


def test_camera_repetition_and_a_slideshow():
    scenes = [scene(i, [("Maya", "Line %d." % i)]) for i in range(1, 7)]
    shots = {f"s{i}": [shot(f"s{i}", 0, "medium", ["Maya"], 3, [("Maya", "Line %d." % i)]),
                       shot(f"s{i}", 1, "medium" if i % 2 else "close_up", ["Maya"], 3)] for i in range(1, 7)}
    rep = report(scenes, shots, {f"s{i}": ["Maya Lin"] for i in range(1, 7)})
    cam = [p["problem"] for p in rep["problems"] if p["category"] == "camera"]
    assert any("all open on a medium" in t for t in cam)
    assert any("% of the shots are mediums" in t for t in cam)
    assert any("same camera move (static)" in t for t in cam)
    single = [scene(i, mode="still") for i in range(1, 7)]
    rep = report(single, {}, {s["id"]: [] for s in single})
    assert any("are one single shot" in p["problem"] for p in rep["problems"])


def test_pacing_rushed_lines_static_stills_and_length():
    scenes, shots, presence = clean_film()
    scenes.append(scene(5, [("Maya", " ".join(["word"] * 40))], secs=6, mode="still"))
    scenes.append(scene(6, secs=14, mode="still"))
    rep = fr.build({"target_seconds": 120}, scenes, shots, CAST[:2], {**presence, "s5": ["Maya Lin"], "s6": []})
    pace = [p for p in rep["problems"] if p["category"] == "pacing"]
    assert any("40 words in 6 s" in p["problem"] and p["severity"] == "high" for p in pace)
    assert any("Scene 6 holds one still picture for 14 s" in p["problem"] for p in pace)
    assert any("120 s was planned" in p["problem"] for p in pace)


def test_screen_time_never_seen_missing_from_shots_and_lead_undercovered():
    scenes, shots, presence = clean_film()
    scenes[1]["dialogue"].append({"character": "Grace", "text": "Dinner's ready."})   # Grace speaks, never seen
    shots["s2"][0]["dialogue"].append({"character": "Grace", "text": "Dinner's ready."})
    presence["s3"].append("Grace")                                                   # planned, but in no shot of s3
    rep = report(scenes, shots, presence, cast=CAST)
    st = [p["problem"] for p in rep["problems"] if p["category"] == "screen_time"]
    assert any("Grace speaks 1 line but is never seen" in t for t in st)
    assert any("Scene 3: Grace is in the scene but in none of its shots" in t for t in st)
    assert rep["metrics"]["screen_time"]["Grace"] == {"shots": 0, "lines": 1, "share": 0.0}


def test_continuity_brings_in_the_free_checks_and_shot_check_failures_point_to_the_free_fix():
    scenes, shots, presence = clean_film()
    issues = [{"scene": 3, "severity": "fix", "message": "Maya's coat changed between scenes 2 and 3.", "fix": "Keep the coat."}]
    checks = {"s2-0": {"status": "failed", "checks": [{"label": "Face", "passed": False}]}}
    rep = report(scenes, shots, presence, continuity_issues=issues, shot_checks=checks)
    cont = [p for p in rep["problems"] if p["category"] == "continuity"]
    assert any("coat changed" in p["problem"] for p in cont)
    fix = next(p for p in cont if p["shot"] == 1 and p["scene"] == 2)
    assert fix["route"] == "free_fix" and "face" in fix["problem"] and fix["severity"] == "high"
    assert rep["metrics"]["shots_not_checked"] == 7


def test_the_ending_short_without_the_lead_and_a_weak_verified_score():
    scenes, shots, presence = clean_film()
    shots["s4"] = [shot("s4", 0, "wide_two_shot", ["Daniel"], 1.5, [("Daniel", "To find you.")])]
    presence["s3"] = presence["s4"] = ["Daniel Okoro"]
    scenes[0]["dialogue"] += [{"character": "Maya", "text": "One."}, {"character": "Maya", "text": "Two."}]
    shots["s1"][1]["dialogue"] += [{"character": "Maya", "text": "One."}, {"character": "Maya", "text": "Two."}]
    p = {"target_seconds": 22, "story_bible": {"evaluation": {"categories": {"ending_hook": {
        "status": "verified", "score": 6, "reason": "it stops instead of ending"}}}}}
    rep = fr.build(p, scenes, shots, CAST[:2], presence)
    end = [x["problem"] for x in rep["problems"] if x["category"] == "ending"]
    assert any("ending goes by too fast" in t for t in end)
    assert any("Maya Lin, who drives the story, isn't in the last two scenes" in t for t in end)
    assert any("scored the ending 6/10: it stops instead of ending" in t for t in end)
    assert rep["metrics"]["ending"]["story_check_score"] == 6


def test_fingerprints_match_the_same_picture_and_tell_different_ones_apart(tmp_path):
    from PIL import Image, ImageDraw
    paths = []
    for i, box in enumerate([(10, 10, 200, 300), (10, 10, 200, 300), (300, 200, 500, 500)]):
        img = Image.new("RGB", (540, 960), (30, 30, 30))
        ImageDraw.Draw(img).rectangle(box, fill=(240, 200, 40))
        paths.append(str(tmp_path / f"p{i}.png"))
        img.save(paths[-1])
    a, b, c = (fr.frame_hash(p) for p in paths)
    assert a == b and fr.distance(a, c) > fr.HASH_DUPLICATE
    clip = str(tmp_path / "v.mp4")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-loop", "1", "-t", "2", "-i", paths[0], "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", clip], check=True)
    assert fr.distance(fr.frame_hash(clip, True, 2.0), a) <= fr.HASH_DUPLICATE
    assert fr.frame_hash(str(tmp_path / "missing.png")) is None


# ------------------------------------------------------------------ end to end

def test_a_rendered_production_gets_fingerprints_and_a_free_report(env):
    from test_auto_director import produce, setup
    client, db, store, ai, gen, ren, u, pid = setup(env)
    ai.varied_pictures = True
    produce(client, gen, ren, u, pid)
    before = balance(client, u)
    jobs_before = db.execute("select count(*) from public.studio_jobs").fetchone()[0]
    rep = client.get(f"/api/v1/studio/projects/{pid}/film-report", headers=H(u))
    assert rep.status_code == 200, rep.text
    rep = rep.json()
    m = rep["metrics"]
    assert rep["render"]["version"] == 1 and m["fingerprinted"] == m["pieces"] >= 3
    assert m["directed_scenes"] == 2 and not [p for p in rep["problems"] if p["category"] in ("duplicates", "missing")]
    assert balance(client, u) == before, "the report is free"
    assert db.execute("select count(*) from public.studio_jobs").fetchone()[0] == jobs_before, "and starts nothing"
    # A shot quietly reusing another shot's clip is caught, and the film is now older than its plan.
    rows = db.execute("select s.id, s.video_path from public.studio_shots s join public.studio_scenes c on c.id = s.scene_id "
                      "where c.project_id = %s order by c.position, s.position", (pid,)).fetchall()
    db.execute("update public.studio_shots set video_path = %s where id = %s", (rows[0][1], rows[-1][0]))
    rep = client.get(f"/api/v1/studio/projects/{pid}/film-report", headers=H(u)).json()
    assert any("exactly the same picture" in p["problem"] for p in rep["problems"])
    assert any("rendered before" in p["problem"] for p in rep["problems"])
    other = str(uuid.uuid4())
    assert client.get(f"/api/v1/studio/projects/{other}/film-report", headers=H(u)).status_code == 404


def test_many_jump_cuts_are_one_item_not_dozens():
    scenes = [scene(i, mode="video") for i in range(1, 11)]
    shots = {f"s{i}": [shot(f"s{i}", 0, "close_up" if i % 2 else "insert", ["Maya"], 3),
                       shot(f"s{i}", 1, "medium", ["Maya"], 3, move="pan" if i % 2 else "dolly")] for i in range(1, 11)}
    for i in range(2, 11):
        shots[f"s{i}"][0]["shot_type"] = "medium"
        shots[f"s{i}"][1]["shot_type"] = "medium" if i % 3 else "close_up"
    rep = fr.build({"target_seconds": 60}, scenes, shots, CAST[:1], {s["id"]: ["Maya Lin"] for s in scenes})
    jumps = [p for p in rep["problems"] if "feel like a jump" in p["problem"] or "feel like jumps" in p["problem"]]
    assert len(jumps) == 1 and "cuts between scenes go from one medium" in jumps[0]["problem"]
