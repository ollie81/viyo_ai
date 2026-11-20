"""The final film check: a usable film passes (including dark/quiet moments and the film's own fades); a film
with missing voices, a black gap, no sound, missing time or the wrong shape fails with the scene to fix."""
import os
import subprocess

import pytest

import film_qc


def _film(path, segments, w=720, h=1280, audio=True):
    """segments: [(seconds, colour, tone_hz or 0)]."""
    parts, args = [], []
    for i, (sec, colour, hz) in enumerate(segments):
        args += ["-f", "lavfi", "-i", f"color=c={colour}:s={w}x{h}:d={sec}:r=24"]
    for i, (sec, colour, hz) in enumerate(segments):
        src = f"sine=frequency={hz}:duration={sec}" if hz else f"anullsrc=r=24000:cl=mono:d={sec}"
        args += ["-f", "lavfi", "-t", str(sec), "-i", src]
    n = len(segments)
    vf = "".join(f"[{i}:v]" for i in range(n)) + f"concat=n={n}:v=1:a=0[v]"
    af = "".join(f"[{n + i}:a]" for i in range(n)) + f"concat=n={n}:v=0:a=1[a]"
    graph = vf + ";" + af if audio else vf
    cmd = ["ffmpeg", "-v", "error", "-y", *args, "-filter_complex", graph, "-map", "[v]"]
    if audio:
        cmd += ["-map", "[a]", "-c:a", "aac"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", path]
    subprocess.run(cmd, check=True)
    return path


def plan(*speaks, d=2.0, fade_in=0.0, fade_out=0.0, durations=None):
    durations = durations or [d] * len(speaks)
    p = [{"duration": x, "speaks": s, "fade_in": 0.0, "fade_out": 0.0} for s, x in zip(speaks, durations)]
    p[0]["fade_in"], p[-1]["fade_out"] = fade_in, fade_out
    return p


def failed(rep):
    return {i["key"]: i["note"] for i in rep["items"] if i["status"] == "failed"}


def test_a_good_film_passes(tmp_path):
    f = _film(str(tmp_path / "f.mp4"), [(2, "red", 300), (2, "blue", 400), (2, "green", 500)])
    rep = film_qc.inspect_film(f, plan(True, True, True), "9:16")
    assert rep["passed"], failed(rep)


def test_a_quiet_scene_without_dialogue_and_a_dark_scene_pass(tmp_path):
    f = _film(str(tmp_path / "f.mp4"), [(2, "red", 300), (2, "0x303030", 0), (2, "green", 500)])
    assert film_qc.inspect_film(f, plan(True, False, True), "9:16")["passed"]


def test_missing_voices_in_a_speaking_scene_fail_and_name_the_scene(tmp_path):
    f = _film(str(tmp_path / "f.mp4"), [(2, "red", 300), (2, "blue", 0), (2, "green", 500)])
    rep = film_qc.inspect_film(f, plan(True, True, True), "9:16")
    assert not rep["passed"] and "Scene 2" in failed(rep)["voices"]
    assert "credits for this render were returned" in film_qc.explain(rep)


def test_a_black_gap_fails_but_the_films_own_fades_dont(tmp_path):
    f = _film(str(tmp_path / "f.mp4"), [(2, "red", 300), (3, "black", 300), (2, "green", 300)])
    rep = film_qc.inspect_film(f, plan(True, True, True, durations=[2, 3, 2]), "9:16")
    assert "picture" in failed(rep) and "scene 2" in failed(rep)["picture"]
    opening = _film(str(tmp_path / "o.mp4"), [(1.5, "black", 300), (4, "red", 300)])
    assert film_qc.inspect_film(opening, plan(True, True, durations=[1.5, 4], fade_in=1.5), "9:16")["passed"]
    ending = _film(str(tmp_path / "e.mp4"), [(4, "red", 300), (2, "black", 300)])
    assert film_qc.inspect_film(ending, plan(True, True, durations=[4, 2], fade_out=2.0), "9:16")["passed"]


def test_no_sound_when_characters_speak_fails(tmp_path):
    f = _film(str(tmp_path / "f.mp4"), [(2, "red", 300), (2, "blue", 300)], audio=False)
    assert "sound" in failed(film_qc.inspect_film(f, plan(True, True), "9:16"))
    assert film_qc.inspect_film(f, plan(False, False), "9:16")["passed"], "a silent film without dialogue is fine"


def test_missing_time_and_wrong_shape_fail(tmp_path):
    f = _film(str(tmp_path / "f.mp4"), [(2, "red", 300), (2, "blue", 300)])
    assert "length" in failed(film_qc.inspect_film(f, plan(True, True, d=5), "9:16"))
    assert "shape" in failed(film_qc.inspect_film(f, plan(True, True), "16:9"))


def test_an_unreadable_file_fails(tmp_path):
    p = tmp_path / "broken.mp4"
    p.write_bytes(b"not a video")
    rep = film_qc.inspect_film(str(p), plan(True), "9:16")
    assert not rep["passed"] and "plays" in failed(rep)
