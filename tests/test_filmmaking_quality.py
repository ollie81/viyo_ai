"""Filmmaking fixes from the quality audit: still pictures move the way the
scene's camera direction says (not one push-in for everything), the film opens
from black and dips to black for a time jump or flashback, captions follow the
recorded voices instead of being spread by word count, and the production quote
says which scenes would be one static shot.

ffmpeg really renders here, so the filters are proven valid; how good the
real AI pictures and clips look still needs a real-model test.
"""
import os
import shutil
import subprocess

import pytest
from PIL import Image

import studio_render as render
from conftest import make_user
from test_creator_studio import H, env, grant, run, start, studio_schema  # noqa: F401
from test_video_drama import DAN, MAYA, detail, project, split

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")


# ------------------------------------------------------------------ camera movement on stills

@pytest.mark.parametrize("text,shot,move", [
    ("slow push in on Maya", "", "push_in"),
    ("the camera pulls back to reveal the empty ward", "", "pull_out"),
    ("pan left across the beds", "", "pan_left"),
    ("tracking right with Daniel", "", "pan_right"),
    ("crane up over the rooftops", "", "tilt_up"),
    ("locked-off, static frame", "", "static"),
    ("", "establishing_wide", "pan_right"),
    ("", "insert", "static"),
    ("", "", "push_in"),
])
def test_the_planned_camera_movement_decides_how_a_still_moves(text, shot, move):
    assert render.movement(text, shot) == move


def test_different_moves_make_different_frames(tmp_path):
    img = tmp_path / "grad.png"
    im = Image.new("RGB", (1080, 1920))
    im.putdata([(x * 255 // 1080, y * 255 // 1920, 120) for y in range(1920) for x in range(1080)])
    im.save(img)

    def last_frame(move):
        out = str(tmp_path / f"{move}.mp4")
        render.render_scene(render.SceneInput(duration=2.0, image=str(img), move=move), 540, 960, out, str(tmp_path), 0, False)
        frame = str(tmp_path / f"{move}.png")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-sseof", "-0.1", "-i", out, "-frames:v", "1", frame], check=True)
        return Image.open(frame).convert("RGB").resize((8, 8)).getpixel((1, 4))
    left, right = last_frame("pan_left"), last_frame("pan_right")
    assert left[0] < right[0] - 25, "a pan left ends on the left of the picture, a pan right on the right"


# ------------------------------------------------------------------ editing

def test_the_film_opens_from_black_and_a_time_jump_dips_to_black(tmp_path):
    img = tmp_path / "white.png"
    Image.new("RGB", (540, 960), (240, 240, 240)).save(img)
    out = str(tmp_path / "s.mp4")
    render.render_scene(render.SceneInput(duration=2.0, image=str(img), fade_in=0.6, fade_out=0.6, move="static"),
                        540, 960, out, str(tmp_path), 0, False)
    first = str(tmp_path / "first.png")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", out, "-frames:v", "1", first], check=True)
    assert sum(Image.open(first).convert("L").resize((1, 1)).getpixel((0, 0)) for _ in [0]) < 40, "starts black"
    mid = str(tmp_path / "mid.png")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", "1.0", "-i", out, "-frames:v", "1", mid], check=True)
    assert Image.open(mid).convert("L").resize((1, 1)).getpixel((0, 0)) > 200


def test_edit_choices_follow_the_scene_record():
    from studio_worker import StudioWorker
    scenes = [{"id": "a", "position": 0, "continuity": {"camera_movement": "slow pan left"}},
              {"id": "b", "position": 1, "continuity": {"transition": "time_jump"}},
              {"id": "c", "position": 2, "continuity": {"transition": "continuous", "camera": "static wide"},
               "audio_voices": {"x": "Kore", "_timings": [[0.0, 1.2], [1.45, 3.0]]}}]
    a, b, c = (StudioWorker._film_edit(s, scenes, False) for s in scenes)
    assert (a["move"], a["fade_in"], a["fade_out"]) == ("pan_left", 0.6, 0.0), "the opening comes up from black"
    assert b["fade_in"] == 0.5 and b["move"] == "push_in", "a time jump dips to black"
    assert c["fade_out"] == 0.8 and c["move"] == "static" and c["line_times"] == ((0.0, 1.2), (1.45, 3.0))
    assert StudioWorker._film_edit(scenes[2], scenes, True)["line_times"] == (), "a talking clip has its own timing"


# ------------------------------------------------------------------ captions follow the voices

def test_captions_use_the_recorded_line_timings():
    lines = ("Kofi?", "You shouldn't be here, Ama. Go home before they see you.")
    assert render.caption_times(lines, 6.0, ((0.0, 0.7), (0.95, 4.8))) == [(0.0, 0.7), (0.95, 4.8)]
    spread = render.caption_times(lines, 6.0)
    assert spread[0][1] < 1.0, "without timings: by word count"
    assert render.caption_times(lines, 6.0, ((0.0, 1.0),)) == spread, "mismatched timings are ignored"


def test_recording_a_scene_saves_when_each_line_is_heard(env):
    client, db, store, files, ai, gen, ren = env
    u = make_user(db)
    grant(client, u, 3000)
    pid = project(client, u, production_style="pictures")["id"]
    talk = {"heading": "INT. WARD - NIGHT", "visual": "Maya and Daniel", "duration_seconds": 6,
            "dialogue": [{"character": "Maya", "text": "You came back."}, {"character": "Daniel", "text": "I never left, not once."}]}
    split(client, ai, gen, u, pid, [talk], characters=(MAYA, DAN))
    sid = detail(client, u, pid)["scenes"][0]["id"]
    start(client, u, pid, "voice", [sid])
    run(gen)
    t = db.execute("select audio_voices from public.studio_scenes where id = %s", (sid,)).fetchone()[0]["_timings"]
    assert len(t) == 2 and t[0][0] == 0.0 and t[1][0] > t[0][1], "line 2 starts after line 1 ends"
    assert abs(t[1][0] - (t[0][1] + 0.25)) < 0.05, "the gap join_audio puts between lines"
    out = client.get(f"/api/v1/studio/projects/{pid}", headers=H(u)).json()
    assert out["scenes"][0].get("outdated_voices", []) == [], "the timing entry is not mistaken for a character"


# ------------------------------------------------------------------ coverage

def test_the_production_quote_names_scenes_that_would_be_one_static_shot(env):
    client, db, store, files, ai, gen, ren = env
    u = make_user(db)
    grant(client, u, 3000)
    pid = project(client, u, production_style="pictures")["id"]
    talk = {"heading": "INT. WARD", "visual": "Maya and Daniel argue", "duration_seconds": 6,
            "dialogue": [{"character": "Maya", "text": "Why?"}, {"character": "Daniel", "text": "Because."}]}
    quiet = {"heading": "EXT. ROOF", "visual": "Maya alone", "duration_seconds": 3, "dialogue": []}
    split(client, ai, gen, u, pid, [talk, quiet], characters=(MAYA, DAN))
    q = client.post(f"/api/v1/studio/projects/{pid}/production/quote", headers=H(u), json={"direct": False}).json()
    assert q["single_shot_scenes"] == [detail(client, u, pid)["scenes"][0]["id"]]
    assert "1 scene (1) would be filmed as one shot each" in q["advice"][0]
