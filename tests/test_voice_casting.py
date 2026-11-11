"""Automatic voice casting and permanent voice identity (tests A-L).
Fake AI and a real local Postgres only; nothing here calls a provider."""
import os
import subprocess
import uuid

import pytest

import studio_ai
import studio_casting as casting
import studio_render as render
import studio_worker as sw
from conftest import make_user
from test_creator_studio import H, balance, env, grant, new_project, run, start, studio_schema  # noqa: F401

ADMIN = {"X-Admin-Key": "admin-secret"}


def detail(client, u, pid):
    return client.get(f"/api/v1/studio/projects/{pid}", headers=H(u)).json()


def cast_of(client, u, pid):
    return {c["name"]: c for c in detail(client, u, pid)["characters"]}


def voice_log(ai):
    return [(x["text"], x["voice"], x["style"]) for x in getattr(ai, "voice_log", [])]


def reset_catalog(db):
    """Back to what the migration ships: adult groups only."""
    db.execute("update public.studio_voice_catalog set age_categories = '[\"young_adult\", \"middle_aged\"]'::jsonb, "
               "status = 'available', languages = '[\"all\"]'::jsonb, verified_by = 'provider_docs'")
    db.execute("delete from public.studio_voice_catalog where voice_id like 'test-%'")


def production(client, ai, gen, u, characters, scenes, language="English", credits=5000):
    grant(client, u, credits, kind="purchase")
    pid = new_project(client, u, language=language)
    client.patch(f"/api/v1/studio/projects/{pid}", headers=H(u), json={"script": "INT. HOUSE - DAY\n\nA story.\n"})
    ai.breakdown_result = {"characters": characters, "scenes": scenes}
    assert start(client, u, pid, "breakdown").status_code == 200
    run(gen)
    job = client.get(f"/api/v1/studio/projects/{pid}/jobs", headers=H(u)).json()["jobs"][0]
    assert job["status"] == "succeeded", job
    return pid


def scene(heading, *lines):
    return {"heading": heading, "visual": "v", "duration_seconds": 4,
            "dialogue": [dict(zip(("character", "text", "emotion"), l)) for l in lines]}


LILY = {"name": "Lily", "description": "A 7-year-old girl with braids, curious and lively", "gender": "female",
        "age_category": "young_child", "age_years": 7, "age_confidence": "high", "personality": "curious, energetic",
        "role": "lead"}
SAM = {"name": "Sam", "description": "16-year-old boy, sarcastic, hoodie", "gender": "male",
       "age_category": "older_teen", "age_years": 16, "age_confidence": "high", "personality": "sarcastic, casual"}
ANA = {"name": "Ana", "description": "Woman, 30, a warm and caring nurse", "gender": "female",
       "age_category": "young_adult", "age_years": 30, "age_confidence": "high", "personality": "warm, caring"}
DAN = {"name": "Daniel", "description": "Man, 40, serious and protective", "gender": "male",
       "age_category": "middle_aged", "age_years": 40, "age_confidence": "high", "personality": "serious, protective",
       "role": "lead"}


# ---------------------------------------------------------------- engine (no database)

def test_age_is_read_from_the_story_not_the_name():
    assert casting.infer_age("A 7-year-old girl") == ("young_child", 7, "high")
    assert casting.infer_age("Sixteen-year-old boy")[0] == "older_teen"
    assert casting.infer_age("Grandfather, retired fisherman")[0] == "elderly"
    assert casting.infer_age("Mother of a 7-year-old, tired nurse") == (None, None, "low"), "the adult, not the child"
    assert casting.infer_age("Maya")[0] is None and casting.infer_age("Junior")[0] is None
    p = casting.build_profile("Kid", {"age_category": "middle_aged"}, "a little girl who loves drawing", "English")
    assert p.age_category == "young_child" and p.age_confidence == "low", "text and AI disagree on child: trust the text"
    p = casting.build_profile("X", {}, "someone in a coat", "English")
    assert (p.age_category, p.age_confidence, p.age_years) == ("unknown", "low", None)


def test_casting_is_deterministic_and_respects_age_gender_and_language():
    cat = casting.builtin_catalog()
    p = casting.build_profile("Daniel", DAN, DAN["description"], "English")
    a = casting.cast(p, cat, set(), "seed-1", DAN["description"])
    b = casting.cast(p, cat, set(), "seed-1", DAN["description"])
    assert a == b and a.status == "cast" and a.voice
    assert next(v for v in cat if v.voice_id == a.voice).gender == "male"
    assert a.voice_category == "adult_male" and "adult male" in a.voice_description and "English" in a.voice_description
    assert a.voice != casting.NARRATOR_VOICE
    other = casting.cast(p, cat, {a.voice}, "seed-1", DAN["description"])
    assert other.voice != a.voice, "a second important character gets a different suitable voice"
    child = casting.cast(casting.build_profile("Lily", LILY, LILY["description"], "English"), cat, set(), "s")
    assert child.status == "needs_voice" and child.voice == "", "never an adult voice for a child"
    teen = casting.cast(casting.build_profile("Sam", SAM, SAM["description"], "English"), cat, set(), "s")
    assert teen.status == "needs_voice"
    old = casting.cast(casting.build_profile("Gran", {"age_years": 70, "gender": "female"}, "kind grandmother", "English"),
                       cat, set(), "s")
    assert old.status == "approximate" and old.voice and "closest available" in old.voice_description
    klingon = casting.cast(casting.build_profile("Q", ANA, "", "Klingon"), cat, set(), "s")
    assert klingon.status == "unsupported_language"


# ---------------------------------------------------------------- A, B, C

def test_A_a_child_is_detected_and_never_given_an_adult_voice(env):
    client, db, store, files, ai, gen, ren = env
    reset_catalog(db)
    u = make_user(db)
    pid = production(client, ai, gen, u, [LILY], [scene("INT. HOUSE - DAY", ("Lily", "Where is Mum?", "frightened"))])
    lily = cast_of(client, u, pid)["Lily"]
    assert lily["age_category"] == "young_child" and lily["age_years"] == 7 and lily["age_label"] == "young child"
    assert lily["voice"] == "" and lily["voice_status"] == "needs_voice" and lily["voice_category"] == "child_female"
    s = detail(client, u, pid)["scenes"][0]
    r = start(client, u, pid, "voice", [s["id"]])
    assert r.status_code == 400 and "won't use an adult voice" in r.json()["detail"]["message"]
    assert balance(client, u)["held"] == 0 and not voice_log(ai)

    # An admin listens and verifies a voice for young children: now Lily is cast, once.
    r = client.patch("/api/v1/studio/admin/voices/google/Leda", headers=ADMIN,
                     json={"age_categories": ["young_child", "older_child"], "checked_by": "qa"})
    assert r.status_code == 200 and r.json()["voice"]["verified_by"] == "qa"
    assert start(client, u, pid, "voice", [s["id"]]).status_code == 200
    run(gen)
    lily = cast_of(client, u, pid)["Lily"]
    assert lily["voice"] == "Leda" and lily["voice_status"] == "cast" and lily["voice_description"].startswith("A natural childlike female")
    assert voice_log(ai)[-1][1] == "Leda"
    hist = db.execute("select new_voice, reason from public.studio_voice_changes").fetchall()
    assert hist == [("Leda", "auto_cast")]
    # Not re-cast later, even if the catalog changes.
    client.patch("/api/v1/studio/admin/voices/google/Leda", headers=ADMIN, json={"age_categories": ["young_adult"]})
    start(client, u, pid, "voice", [s["id"]])
    run(gen)
    assert voice_log(ai)[-1][1] == "Leda"
    reset_catalog(db)


def test_A_a_job_already_queued_for_a_child_fails_clearly_and_refunds(env):
    client, db, store, files, ai, gen, ren = env
    reset_catalog(db)
    u = make_user(db)
    pid = production(client, ai, gen, u, [LILY, ANA], [scene("INT. HOUSE - DAY", ("Ana", "Hello"))])
    s = detail(client, u, pid)["scenes"][0]
    before = balance(client, u)
    assert start(client, u, pid, "voice", [s["id"]]).status_code == 200
    # The line changes to Lily's after the job was queued (bypassing the price check).
    db.execute("update public.studio_scenes set dialogue = '[{\"character\": \"Lily\", \"text\": \"Hi\"}]'::jsonb")
    run(gen)
    job = client.get(f"/api/v1/studio/projects/{pid}/jobs", headers=H(u)).json()["jobs"][0]
    assert job["status"] == "failed" and "Lily is a young child" in job["error"] and "credits were returned" in job["error"]
    assert balance(client, u) == before and not voice_log(ai)


def test_B_a_teenager_is_not_given_a_generic_adult_voice(env):
    client, db, store, files, ai, gen, ren = env
    reset_catalog(db)
    u = make_user(db)
    pid = production(client, ai, gen, u, [SAM], [scene("EXT. PARK - DAY", ("Sam", "Whatever."))])
    sam = cast_of(client, u, pid)["Sam"]
    assert sam["age_category"] == "older_teen" and sam["voice"] == "" and sam["voice_status"] == "needs_voice"
    assert sam["voice_category"] == "teen_male"
    client.patch("/api/v1/studio/admin/voices/google/Puck", headers=ADMIN, json={"age_categories": ["older_teen"]})
    client.patch("/api/v1/studio/admin/voices/google/Leda", headers=ADMIN, json={"age_categories": ["older_teen"]})
    voice_all(client, gen, u, pid)
    assert cast_of(client, u, pid)["Sam"]["voice"] == "Puck", "the only verified teen voice that is male"
    reset_catalog(db)


def voice_all(client, gen, u, pid, targets=None):
    r = start(client, u, pid, "voice", targets)
    assert r.status_code == 200, r.text
    run(gen)


def test_C_adults_get_suitable_independent_voices(env):
    client, db, store, files, ai, gen, ren = env
    reset_catalog(db)
    u = make_user(db)
    pid = production(client, ai, gen, u, [ANA, DAN], [scene("INT. WARD - NIGHT", ("Ana", "He's awake."), ("Daniel", "Good."))])
    cast = cast_of(client, u, pid)
    genders = {v.voice_id: v.gender for v in casting.builtin_catalog()}
    assert genders[cast["Ana"]["voice"]] == "female" and genders[cast["Daniel"]["voice"]] == "male"
    assert cast["Ana"]["voice_category"] == "adult_female" and cast["Daniel"]["voice_category"] == "adult_male"
    assert cast["Ana"]["age_years"] == 30 and cast["Daniel"]["age_category"] == "middle_aged"
    assert cast["Ana"]["voice_status"] == cast["Daniel"]["voice_status"] == "cast"
    assert "warm" in cast["Ana"]["voice_description"] or "natural" in cast["Ana"]["voice_description"]
    # Same story, new production: the same casting decisions are made.
    pid2 = production(client, ai, gen, u, [ANA, DAN], [scene("INT. WARD - NIGHT", ("Ana", "He's awake."))])
    cast2 = cast_of(client, u, pid2)
    assert {k: v["voice"] for k, v in cast2.items()} != {} and cast2["Ana"]["voice_status"] == "cast"


# ---------------------------------------------------------------- D, G, H

def test_D_H_same_voice_in_three_scenes_while_emotion_changes(env):
    client, db, store, files, ai, gen, ren = env
    reset_catalog(db)
    u = make_user(db)
    pid = production(client, ai, gen, u, [ANA], [
        scene("INT. A - DAY", ("Ana", "We won!", "happy")),
        scene("INT. B - DAY", ("Ana", "He's gone.", "sad")),
        scene("INT. C - DAY", ("Ana", "Get out!", "angry")),
        scene("INT. D - NIGHT", ("Ana", "Who's there?", "frightened"))])
    voice_all(client, gen, u, pid)
    ana = cast_of(client, u, pid)["Ana"]
    log = voice_log(ai)
    assert {v for _t, v, _s in log} == {ana["voice"]}, "one speaker"
    styles = {t: s for t, _v, s in log}
    assert [styles[x] for x in ("We won!", "He's gone.", "Get out!", "Who's there?")] == \
        [f"{ana['voice_style']}, {e}, felt but held back, not overplayed" for e in ("happy", "sad", "angry", "frightened")]
    assert ana["voice_style"] == "warm", "a short, stable note from the personality"


def test_G_regenerating_a_line_keeps_the_voice_and_the_old_recording(env):
    client, db, store, files, ai, gen, ren = env
    reset_catalog(db)
    u = make_user(db)
    pid = production(client, ai, gen, u, [DAN], [scene("INT. A - DAY", ("Daniel", "Stay here.", "serious"))])
    voice_all(client, gen, u, pid)
    s = detail(client, u, pid)["scenes"][0]
    first = db.execute("select audio_path from public.studio_scenes where id=%s", (s["id"],)).fetchone()[0]
    voice_all(client, gen, u, pid, [s["id"]])
    second = db.execute("select audio_path from public.studio_scenes where id=%s", (s["id"],)).fetchone()[0]
    assert first != second
    assert files.get(first), "the earlier recording file is still there"
    assert db.execute("select count(*) from public.studio_assets where kind='audio'").fetchone()[0] == 2
    assert len({v for _t, v, _s in voice_log(ai)}) == 1


# ---------------------------------------------------------------- E, F, J

def test_E_F_episode_2_and_episode_55_reuse_the_same_identity(env, monkeypatch):
    client, db, store, files, ai, gen, ren = env
    import creator_studio as cs
    monkeypatch.setattr(cs, "_rate_limit", lambda *a, **k: None)   # 54 episodes in a row
    reset_catalog(db)
    u = make_user(db)
    pid = production(client, ai, gen, u, [ANA, DAN], [scene("INT. A - DAY", ("Ana", "Hi"), ("Daniel", "Hey"))])
    ep1 = cast_of(client, u, pid)
    history_before = db.execute("select count(*) from public.studio_voice_changes").fetchone()[0]
    last = pid
    for n in range(2, 56):
        r = client.post(f"/api/v1/studio/projects/{last}/next-episode", headers=H(u))
        assert r.status_code == 200, r.text
        last = r.json()["id"]
        if n == 2:
            ep2 = cast_of(client, u, last)
            for name in ("Ana", "Daniel"):
                assert ep2[name]["voice"] == ep1[name]["voice"] and ep2[name]["identity_id"] == ep1[name]["id"]
                assert ep2[name]["age_category"] == ep1[name]["age_category"]
                assert ep2[name]["voice_description"] == ep1[name]["voice_description"]
    assert detail(client, u, last)["project"]["title"].endswith("Episode 55")
    ep55 = cast_of(client, u, last)
    for name in ("Ana", "Daniel"):
        assert ep55[name]["voice"] == ep1[name]["voice"] and ep55[name]["identity_id"] == ep1[name]["id"]
    # Episode 55's own scene list: the people are recognised, nobody is re-cast.
    client.patch(f"/api/v1/studio/projects/{last}", headers=H(u), json={"script": "INT. B\n\nANA\nAgain.\n"})
    ai.breakdown_result = {"characters": [ANA, DAN], "scenes": [scene("INT. B", ("Ana", "Again."))]}
    start(client, u, last, "breakdown")
    run(gen)
    voice_all(client, gen, u, last)
    assert voice_log(ai)[-1][1] == ep1["Ana"]["voice"]
    assert db.execute("select count(*) from public.studio_voice_changes").fetchone()[0] == history_before, \
        "no new voice assignments anywhere in the series"


def test_J_a_deliberate_voice_change_applies_to_the_whole_series_and_keeps_old_audio(env):
    client, db, store, files, ai, gen, ren = env
    reset_catalog(db)
    u = make_user(db)
    pid = production(client, ai, gen, u, [ANA], [scene("INT. A - DAY", ("Ana", "Hi"))])
    voice_all(client, gen, u, pid)
    ep2 = client.post(f"/api/v1/studio/projects/{pid}/next-episode", headers=H(u)).json()["id"]
    ep3 = client.post(f"/api/v1/studio/projects/{ep2}/next-episode", headers=H(u)).json()["id"]
    old_audio = db.execute("select audio_path from public.studio_scenes where project_id=%s", (pid,)).fetchone()[0]
    ana2 = cast_of(client, u, ep2)["Ana"]
    new_voice = "Sulafat" if ana2["voice"] != "Sulafat" else "Despina"
    r = client.patch(f"/api/v1/studio/characters/{ana2['id']}", headers=H(u), json={"voice": new_voice})
    assert r.status_code == 409 and r.json()["detail"]["scenes"] == 1, "episode 1's recording counts"
    r = client.patch(f"/api/v1/studio/characters/{ana2['id']}", headers=H(u),
                     json={"voice": new_voice, "confirm_voice_change": True}).json()
    assert r["voice_status"] == "manual" and r["other_episodes_affected"] == 1 and r["rerecord_scene_ids"] == []
    for p in (pid, ep2, ep3):
        assert cast_of(client, u, p)["Ana"]["voice"] == new_voice
    assert db.execute("select audio_path from public.studio_scenes where project_id=%s", (pid,)).fetchone()[0] == old_audio
    assert files.get(old_audio)
    assert detail(client, u, pid)["scenes"][0]["voice_outdated"] == ["Ana"]
    assert db.execute("select reason from public.studio_voice_changes order by changed_at desc limit 1").fetchone()[0] == "creator"


# ---------------------------------------------------------------- I

def test_I_provider_failure_refunds_and_a_retry_uses_the_same_voice(env):
    client, db, store, files, ai, gen, ren = env
    reset_catalog(db)
    u = make_user(db)
    pid = production(client, ai, gen, u, [DAN], [scene("INT. A - DAY", ("Daniel", "Now."))])
    s = detail(client, u, pid)["scenes"][0]
    before = balance(client, u)
    ai.fail["voice"] = [studio_ai.ProviderError("boom")]
    voice_all(client, gen, u, pid, [s["id"]])
    job = client.get(f"/api/v1/studio/projects/{pid}/jobs", headers=H(u)).json()["jobs"][0]
    assert job["status"] == "failed" and balance(client, u) == before
    voice_all(client, gen, u, pid, [s["id"]])
    log = voice_log(ai)
    assert len(log) == 2 and log[0][1] == log[1][1] == cast_of(client, u, pid)["Daniel"]["voice"]
    assert sw.NARRATOR_VOICE not in {v for _t, v, _s in log}


def test_I_an_unavailable_voice_is_a_clear_error_not_a_substitute(env):
    client, db, store, files, ai, gen, ren = env
    reset_catalog(db)
    u = make_user(db)
    pid = production(client, ai, gen, u, [DAN], [scene("INT. A - DAY", ("Daniel", "Now."))])
    dan = cast_of(client, u, pid)["Daniel"]
    r = client.patch(f"/api/v1/studio/admin/voices/google/{dan['voice']}", headers=ADMIN, json={"status": "unavailable"})
    assert r.json()["characters_using"] >= 1 and "keep it" in r.json()["note"]
    assert cast_of(client, u, pid)["Daniel"]["voice"] == dan["voice"], "never silently changed"
    s = detail(client, u, pid)["scenes"][0]
    r = start(client, u, pid, "voice", [s["id"]])
    assert r.status_code == 400 and "isn't available" in r.json()["detail"]["message"]
    names = {v["name"] for v in client.get("/api/v1/studio/voices", headers=H(u)).json()["voices"]}
    assert dan["voice"] not in names, "unavailable voices aren't offered or cast"
    reset_catalog(db)


# ---------------------------------------------------------------- K

@pytest.mark.parametrize("language,expected", [("English", "English"), ("French", "French"),
                                               ("Swahili", "Swahili"), ("Kinyarwanda", "Kinyarwanda")])
def test_K_supported_languages(env, language, expected):
    client, db, store, files, ai, gen, ren = env
    reset_catalog(db)
    u = make_user(db)
    pid = production(client, ai, gen, u, [ANA], [scene("INT. A", ("Ana", "..."))], language=language)
    ana = cast_of(client, u, pid)["Ana"]
    assert ana["voice"] and ana["voice_language"] == expected and expected in ana["voice_description"]


def test_K_an_unsupported_language_fails_clearly(env):
    client, db, store, files, ai, gen, ren = env
    reset_catalog(db)
    u = make_user(db)
    pid = production(client, ai, gen, u, [ANA], [scene("INT. A", ("Ana", "Amahoro"))], language="Kirundi")
    ana = cast_of(client, u, pid)["Ana"]
    assert ana["voice"] == "" and ana["voice_status"] == "unsupported_language"
    s = detail(client, u, pid)["scenes"][0]
    r = start(client, u, pid, "voice", [s["id"]])
    assert r.status_code == 400 and "Kirundi" in r.json()["detail"]["message"] and "doesn't support" in r.json()["detail"]["message"]


# ---------------------------------------------------------------- L

def test_L_dialogue_scenes_use_only_the_character_voice(env, monkeypatch):
    client, db, store, files, ai, gen, ren = env
    reset_catalog(db)
    u = make_user(db)
    pid = production(client, ai, gen, u, [DAN], [scene("INT. A", ("Daniel", "Go.")),
                                                 {"heading": "EXT. B", "visual": "rain", "duration_seconds": 4, "dialogue": []}])
    sc = detail(client, u, pid)["scenes"]
    start(client, u, pid, "image")
    run(gen)
    voice_all(client, gen, u, pid, [sc[0]["id"]])
    for s in sc:
        client.patch(f"/api/v1/studio/scenes/{s['id']}", headers=H(u), json={"mode": "video"})
        start(client, u, pid, "video", [s["id"]])
        run(gen)
        if s["id"] == sc[0]["id"]:
            assert "No dialogue and no speech" in ai.last_video_prompt
    seen = {}
    real = render.render_film

    def spy(scenes, *a, **kw):
        seen["scenes"] = scenes
        return real(scenes, *a, **kw)

    monkeypatch.setattr(render, "render_film", spy)
    assert start(client, u, pid, "render").status_code == 200
    run(ren)
    job = client.get(f"/api/v1/studio/projects/{pid}/jobs", headers=H(u)).json()["jobs"][0]
    assert job["status"] == "succeeded", job
    assert [s.video_audio for s in seen["scenes"]] == [False, True], "video sound only where nobody speaks"


def _band_db(path, hz):
    """Mean level of a narrow band around `hz`."""
    out = subprocess.run(["ffmpeg", "-v", "info", "-i", path, "-af",
                          f"bandpass=f={hz}:width_type=h:w=60,bandpass=f={hz}:width_type=h:w=60,volumedetect",
                          "-f", "null", "-"], capture_output=True, text=True).stderr
    line = next(l for l in out.splitlines() if "mean_volume" in l)
    return float(line.split("mean_volume:")[1].split("dB")[0])


def test_L_the_video_models_own_speech_is_never_mixed_under_dialogue(tmp_path):
    """The fake video's own sound is a 300 Hz tone; the character's voice is
    a 1000 Hz tone. In a dialogue scene only the voice may be heard; in a
    scene without dialogue the video's sound stays as ambience."""
    clip, _ = studio_ai.FakeStudioAI().video("x", None, "9:16", 3)
    vpath = tmp_path / "v.mp4"
    vpath.write_bytes(clip)
    voice = tmp_path / "voice.wav"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "sine=f=1000:d=2", "-ar", "24000", str(voice)],
                   check=True)
    w, h = render.canvas("9:16")
    talk = tmp_path / "talk.mp4"
    render.render_scene(render.SceneInput(duration=3, video=str(vpath), audio=str(voice), audio_seconds=2,
                                          lines=("Go.",), video_audio=False), w, h, str(talk), str(tmp_path), 0, False)
    quiet = tmp_path / "quiet.mp4"
    render.render_scene(render.SceneInput(duration=3, video=str(vpath)), w, h, str(quiet), str(tmp_path), 1, False)
    assert _band_db(str(talk), 1000) > -30, "the character's voice is there"
    assert _band_db(str(talk), 300) < -55, "no second voice from the video model under the dialogue"
    assert _band_db(str(quiet), 300) > -45, "ambient sound kept when nobody talks"


# ---------------------------------------------------------------- creator added characters, gallery

def test_characters_added_by_hand_are_cast_automatically(env):
    client, db, store, files, ai, gen, ren = env
    reset_catalog(db)
    u = make_user(db)
    grant(client, u, 100)
    pid = new_project(client, u)
    kid = client.post(f"/api/v1/studio/projects/{pid}/characters", headers=H(u),
                      json={"name": "Tom", "description": "A 9-year-old boy who loves football"}).json()
    assert kid["age_category"] == "older_child" and kid["voice"] == "" and kid["voice_status"] == "needs_voice"
    adult = client.post(f"/api/v1/studio/projects/{pid}/characters", headers=H(u),
                        json={"name": "Grace", "description": "Woman in her 30s, a confident lawyer"}).json()
    assert adult["voice"] and adult["voice_status"] == "cast" and adult["voice_category"] == "adult_female"


def test_admin_voice_gallery(env):
    client, db, store, files, ai, gen, ren = env
    reset_catalog(db)
    g = client.get("/api/v1/studio/admin/voices", headers=ADMIN).json()
    assert len(g["voices"]) == 30 and g["age_groups"][0] == "infant"
    assert all(v["age_categories"] == ["young_adult", "middle_aged"] and v["verified_by"] == "provider_docs" for v in g["voices"])
    assert client.get("/api/v1/studio/admin/voices").status_code in (401, 403)
    assert client.patch("/api/v1/studio/admin/voices/google/Kore", headers=ADMIN,
                        json={"age_categories": ["toddler"]}).status_code == 400
    assert client.patch("/api/v1/studio/admin/voices/google/Kore", headers=ADMIN,
                        json={"languages": ["Klingon"]}).status_code == 400
    r = client.post("/api/v1/studio/admin/voices", headers=ADMIN,
                    json={"voice_id": "test-kid-1", "gender": "female", "age_categories": ["young_child"],
                          "notes": "from the extended library, checked by ear"})
    assert r.status_code == 200 and r.json()["voice"]["verified_by"] == "admin"
    assert client.post("/api/v1/studio/admin/voices", headers=ADMIN, json={"voice_id": "test-kid-1"}).status_code == 409
    reset_catalog(db)
