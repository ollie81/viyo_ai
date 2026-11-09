"""Studio stage 0: Omni 1.2 video, a new scene list never loses paid work,
no silent voice swaps, and the production language reaches the breakdown.
Fake AI only: nothing here calls a real provider."""
import base64
import types
import uuid

import pytest

import studio_ai
import studio_pricing as pricing
import studio_worker as sw
from conftest import make_user
from test_creator_studio import H, approve_story, balance, env, grant, new_project, run, start, studio_schema  # noqa: F401


def detail(client, u, pid):
    return client.get(f"/api/v1/studio/projects/{pid}", headers=H(u)).json()


def produced(client, ai, gen, u, **project):
    """A project with a script, a scene list, portraits, scene pictures and voices."""
    grant(client, u, 5000, kind="purchase")
    pid = new_project(client, u, **project)
    assert start(client, u, pid, "script").status_code == 200
    run(gen)
    assert start(client, u, pid, "breakdown").status_code == 200
    run(gen)
    start(client, u, pid, "portrait")
    run(gen)
    start(client, u, pid, "image")
    run(gen)
    start(client, u, pid, "voice")
    run(gen)
    d = detail(client, u, pid)
    assert all(s["has_audio"] for s in d["scenes"]) and all(s["image_url"] for s in d["scenes"])
    return pid


def scene_rows(db, pid):
    return db.execute("select id::text, position, heading, image_path, video_path, audio_path, dialogue "
                      "from public.studio_scenes where project_id = %s order by position", (pid,)).fetchall()


# ---------------------------------------------------------------- 1. video model and price

def test_video_uses_gemini_omni_1_2_flash_everywhere():
    import ads_studio
    import studio
    assert pricing.VIDEO_MODEL == "gemini-omni-1.2-flash"
    assert studio.VEO_MODEL == "gemini-omni-1.2-flash" and studio.VEO_PRICE_PER_SEC_USD_CENTS == 7
    assert {t["model"] for t in ads_studio.VEO_TIERS.values()} == {"gemini-omni-1.2-flash"}
    assert {t["price_per_sec_cents"] for t in ads_studio.VEO_TIERS.values()} == {7}


def test_video_price_matches_omni_1_2_list_price():
    # $12 per million output tokens x 5,792 tokens per second of 720p = $0.0695/s
    assert pricing.video_cost(10).provider_cost_usd == pytest.approx(0.695, abs=0.001)
    assert pricing.video_cost(1).provider_cost_usd == pytest.approx(0.0695, abs=0.0001)
    # Clip length asked for = the scene's length, within Omni's 3-15 s.
    assert [pricing.video_seconds_for(s) for s in (2, 3, 5.2, 8, 15, 20, None)] == [3, 3, 6, 8, 15, 15, 8]
    assert pricing.estimate_video(6).units["video_seconds"] == 6


def test_google_video_request_names_the_model_and_the_clip_length(monkeypatch):
    sent = {}

    class Interactions:
        def create(self, **kw):
            sent.update(kw)
            return types.SimpleNamespace(output_video=types.SimpleNamespace(data=base64.b64encode(b"not-a-real-mp4").decode()),
                                         usage=None)

    g = object.__new__(studio_ai.GoogleStudioAI)
    g.client = types.SimpleNamespace(interactions=Interactions())
    data, cost = g.video("a quiet street", b"png", "9:16", 6)
    assert sent["model"] == "gemini-omni-1.2-flash"
    assert sent["response_format"] == {"type": "video", "aspect_ratio": "9:16", "resolution": "720p", "duration": "6s"}
    assert data == b"not-a-real-mp4"
    assert cost.units["video_seconds"] == 6.0
    assert cost.provider_cost_usd == pytest.approx(6 * 0.0695 + 1e-12, abs=0.001)


def test_a_retired_or_wrong_model_gives_a_clear_error():
    e = studio_ai._classify(Exception("404 NOT_FOUND: models/gemini-omni-9 is not found"))
    assert isinstance(e, studio_ai.ProviderError) and "credits were returned" in str(e)
    assert isinstance(studio_ai._classify(Exception("429 RESOURCE_EXHAUSTED")), studio_ai.ProviderBusy)


def test_video_estimate_and_job_use_the_scene_length(env):
    client, db, store, files, ai, gen, ren = env
    u = make_user(db)
    grant(client, u, 2000)
    pid = new_project(client, u)
    s = client.post(f"/api/v1/studio/projects/{pid}/scenes", headers=H(u), json={"visual": "rain", "duration_seconds": 6}).json()
    est = client.post(f"/api/v1/studio/projects/{pid}/estimate", headers=H(u), json={"kind": "video", "target_ids": [s["id"]]}).json()
    assert est["total_credits"] == pricing.estimate_video(6).credits
    assert start(client, u, pid, "video", [s["id"]]).status_code == 200
    run(gen)
    assert ("video", "9:16", 6) in ai.calls
    job = client.get(f"/api/v1/studio/projects/{pid}/jobs", headers=H(u)).json()["jobs"][0]
    assert job["status"] == "succeeded" and job["charged_credits"] <= est["total_credits"]


def test_a_failed_video_returns_the_credits(env):
    client, db, store, files, ai, gen, ren = env
    u = make_user(db)
    grant(client, u, 2000)
    pid = new_project(client, u)
    s = client.post(f"/api/v1/studio/projects/{pid}/scenes", headers=H(u), json={"visual": "rain"}).json()
    ai.fail["video"] = studio_ai.ProviderError("The AI model rejected the request (404). Your credits were returned.")
    start(client, u, pid, "video", [s["id"]])
    run(gen)
    job = client.get(f"/api/v1/studio/projects/{pid}/jobs", headers=H(u)).json()["jobs"][0]
    assert job["status"] == "failed"
    assert balance(client, u)["available"] == 2000 and balance(client, u)["held"] == 0


# ---------------------------------------------------------------- 2. paid work is never lost

def test_a_new_scene_list_must_be_confirmed_and_nothing_is_held_until_then(env):
    client, db, store, files, ai, gen, ren = env
    u = make_user(db)
    pid = produced(client, ai, gen, u)
    before = balance(client, u)
    r = start(client, u, pid, "breakdown")
    assert r.status_code == 409
    d = r.json()["detail"]
    assert d["code"] == "confirm_replace" and d["scenes"] == 2 and d["with_pictures"] == 2 and d["with_voice"] == 2
    assert "saved, not deleted" in d["message"]
    assert balance(client, u) == before


def test_rerunning_the_same_breakdown_keeps_every_scene_and_its_work(env):
    client, db, store, files, ai, gen, ren = env
    u = make_user(db)
    pid = produced(client, ai, gen, u)
    before = scene_rows(db, pid)
    assets = db.execute("select count(*) from public.studio_assets").fetchone()[0]
    approve_story(client, u, pid)
    r = client.post(f"/api/v1/studio/projects/{pid}/jobs", headers=H(u),
                    json={"kind": "breakdown", "request_id": uuid.uuid4().hex, "confirm_replace": True})
    assert r.status_code == 200
    run(gen)
    assert scene_rows(db, pid) == before, "same ids, same pictures, same voice recordings"
    assert db.execute("select count(*) from public.studio_scene_archive").fetchone()[0] == 0
    assert db.execute("select count(*) from public.studio_assets").fetchone()[0] == assets
    job = client.get(f"/api/v1/studio/projects/{pid}/jobs", headers=H(u)).json()["jobs"][0]
    assert job["status"] == "succeeded"


def test_changed_and_new_scenes_never_disconnect_paid_assets(env):
    client, db, store, files, ai, gen, ren = env
    u = make_user(db)
    pid = produced(client, ai, gen, u)
    (s1, p1, h1, img1, _, aud1, _), (s2, p2, h2, img2, _, aud2, _) = scene_rows(db, pid)
    assets = db.execute("select count(*) from public.studio_assets").fetchone()[0]

    ai.breakdown_result = {
        "characters": [{"name": "Maya", "description": "different text", "gender": "female"},
                       {"name": "Daniel", "description": "", "gender": "male"},
                       {"name": "Rosa", "description": "Woman in her 60s, grey bun", "gender": "female"}],
        "scenes": [
            {"heading": "INT. HOSPITAL - DAY", "visual": "Rosa at a desk", "duration_seconds": 3,
             "dialogue": [{"character": "Rosa", "text": "Next patient."}]},
            {"heading": "INT. ROOFTOP - NIGHT", "visual": "Maya on a rooftop", "duration_seconds": 4,
             "dialogue": [{"character": "Maya", "text": "They said I'd never come back."}]},          # unchanged
            {"heading": "EXT. STREET - NIGHT", "visual": "Daniel running", "duration_seconds": 3,
             "dialogue": [{"character": "Daniel", "text": "Maya! Please wait!"}]},                     # new line
        ]}
    approve_story(client, u, pid)
    client.post(f"/api/v1/studio/projects/{pid}/jobs", headers=H(u),
                json={"kind": "breakdown", "request_id": uuid.uuid4().hex, "confirm_replace": True})
    run(gen)
    rows = scene_rows(db, pid)
    assert [r[1] for r in rows] == [0, 1, 2]
    new, kept, changed = rows
    assert new[3] is None and new[5] is None                                   # added, nothing to keep
    assert kept[0] == s1 and kept[3] == img1 and kept[5] == aud1               # untouched, moved down
    assert changed[0] != s2 and changed[3] == img2 and changed[5] is None      # same place: picture kept, old words not reused
    arch = db.execute("select scene_id::text, image_path, audio_path, scene->'dialogue' "
                      "from public.studio_scene_archive where project_id = %s", (pid,)).fetchall()
    assert len(arch) == 1 and arch[0][0] == s2 and arch[0][1] == img2 and arch[0][2] == aud2
    assert arch[0][3] == [{"character": "Daniel", "text": "Maya! Wait!"}]
    assert db.execute("select count(*) from public.studio_assets").fetchone()[0] == assets, "no file record lost"
    cast = {c["name"]: c for c in detail(client, u, pid)["characters"]}
    assert set(cast) == {"Maya", "Daniel", "Rosa"}
    assert cast["Maya"]["description"] == "Woman in her 20s, short black hair, red jacket", "existing look not overwritten"
    voices = [c["voice"] for c in cast.values()]
    assert len(set(voices)) == 3 and all(v in studio_ai.VOICE_NAMES for v in voices)


def test_existing_characters_keep_their_voice_through_a_new_scene_list(env):
    client, db, store, files, ai, gen, ren = env
    u = make_user(db)
    pid = produced(client, ai, gen, u)
    db.execute("update public.studio_characters set voice = 'Sulafat' where name = 'Maya'")
    approve_story(client, u, pid)
    client.post(f"/api/v1/studio/projects/{pid}/jobs", headers=H(u),
                json={"kind": "breakdown", "request_id": uuid.uuid4().hex, "confirm_replace": True})
    run(gen)
    assert db.execute("select voice from public.studio_characters where name = 'Maya'").fetchone()[0] == "Sulafat"


def test_a_failed_breakdown_leaves_the_project_as_it_was_and_refunds(env):
    client, db, store, files, ai, gen, ren = env
    u = make_user(db)
    pid = produced(client, ai, gen, u)
    before, bal = scene_rows(db, pid), balance(client, u)
    ai.fail["breakdown"] = studio_ai.ProviderError("boom")
    approve_story(client, u, pid)
    client.post(f"/api/v1/studio/projects/{pid}/jobs", headers=H(u),
                json={"kind": "breakdown", "request_id": uuid.uuid4().hex, "confirm_replace": True})
    run(gen)
    assert scene_rows(db, pid) == before
    assert balance(client, u) == bal


def test_the_scene_update_is_all_or_nothing(env):
    """If any part of applying a scene list fails, nothing changes."""
    client, db, store, files, ai, gen, ren = env
    u = make_user(db)
    pid = produced(client, ai, gen, u)
    before = scene_rows(db, pid)
    with pytest.raises(Exception):
        store.rpc("studio_apply_breakdown", {
            "p_project": pid, "p_owner": str(u), "p_job": None, "p_characters": [], "p_describe": [], "p_keep": [],
            "p_archive": [before[0][0], before[1][0]],
            "p_insert": [{"position": 0, "heading": "X", "visual": "x", "dialogue": [], "duration_seconds": 99,
                          "mode": "still"}]})       # 99 s breaks a check constraint after the archive step ran
    assert scene_rows(db, pid) == before
    assert db.execute("select count(*) from public.studio_scene_archive").fetchone()[0] == 0


def test_a_first_breakdown_needs_no_confirmation(env):
    client, db, store, files, ai, gen, ren = env
    u = make_user(db)
    grant(client, u, 2000)
    pid = new_project(client, u)
    start(client, u, pid, "script")
    run(gen)
    assert start(client, u, pid, "breakdown").status_code == 200


def test_plan_matches_duplicate_places_in_order():
    current = [{"id": "a", "position": 0, "heading": "INT. KITCHEN - DAY", "dialogue": [{"character": "A", "text": "Hi"}],
                "image_path": "a.png"},
               {"id": "b", "position": 1, "heading": "INT. KITCHEN - DAY", "dialogue": [{"character": "A", "text": "Bye"}],
                "image_path": "b.png", "audio_path": "b.m4a"}]
    incoming = [{"heading": "int. kitchen — day", "visual": "", "duration_seconds": 3,
                 "dialogue": [{"character": "A (V.O.)", "text": "Bye!"}]},
                {"heading": "INT. KITCHEN - DAY", "visual": "", "duration_seconds": 3,
                 "dialogue": [{"character": "A", "text": "Something new"}]}]
    plan = sw.plan_breakdown(current, incoming)
    assert plan["keep"] == [{"id": "b", "position": 0}]       # punctuation and (V.O.) don't count as changes
    assert plan["insert"][0]["image_path"] == "a.png" and plan["insert"][0]["position"] == 1
    assert plan["archive"] == ["a"]


# ---------------------------------------------------------------- 3. voices

def _voiced_scene(client, u, pid, speaker="Maya", text="Hello there"):
    return client.post(f"/api/v1/studio/projects/{pid}/scenes", headers=H(u),
                       json={"visual": "x", "dialogue": [{"character": speaker, "text": text}]}).json()


def test_an_unknown_saved_voice_stops_the_job_and_refunds(env, capsys):
    client, db, store, files, ai, gen, ren = env
    u = make_user(db)
    grant(client, u, 300)
    pid = new_project(client, u)
    client.post(f"/api/v1/studio/projects/{pid}/characters", headers=H(u), json={"name": "Maya", "voice": "Kore"})
    s = _voiced_scene(client, u, pid)
    assert start(client, u, pid, "voice", [s["id"]]).status_code == 200
    db.execute("update public.studio_characters set voice = 'OldRetiredVoice' where name = 'Maya'")
    run(gen)
    job = client.get(f"/api/v1/studio/projects/{pid}/jobs", headers=H(u)).json()["jobs"][0]
    assert job["status"] == "failed" and "Maya's voice" in job["error"] and "credits were returned" in job["error"]
    assert not [c for c in ai.calls if c[0] == "voice"], "never voiced with a different voice"
    assert balance(client, u) == {"available": 300, "purchased": 0, "promotional": 300, "held": 0}
    assert db.execute("select audio_path from public.studio_scenes where id = %s", (s["id"],)).fetchone()[0] is None
    assert "unavailable voice 'OldRetiredVoice'" in capsys.readouterr().out
    # ...and the API refuses before holding anything next time.
    r = start(client, u, pid, "voice", [s["id"]])
    assert r.status_code == 400 and "Choose a new voice" in r.json()["detail"]["message"]


def test_a_speaker_missing_from_the_cast_is_refused_not_narrated(env):
    client, db, store, files, ai, gen, ren = env
    u = make_user(db)
    grant(client, u, 300)
    pid = new_project(client, u)
    s = _voiced_scene(client, u, pid, speaker="Ghost")
    r = start(client, u, pid, "voice", [s["id"]])
    assert r.status_code == 400 and "isn't in your cast" in r.json()["detail"]["message"]
    assert balance(client, u)["available"] == 300 and not ai.calls


def test_narrator_and_screenplay_extensions(env):
    client, db, store, files, ai, gen, ren = env
    u = make_user(db)
    grant(client, u, 300)
    pid = new_project(client, u)
    client.post(f"/api/v1/studio/projects/{pid}/characters", headers=H(u), json={"name": "Maya", "voice": "Kore"})
    s = client.post(f"/api/v1/studio/projects/{pid}/scenes", headers=H(u), json={"visual": "x", "dialogue": [
        {"character": "NARRATOR", "text": "Years later."}, {"character": "MAYA (V.O.)", "text": "I remember."}]}).json()
    assert start(client, u, pid, "voice", [s["id"]]).status_code == 200
    run(gen)
    assert [c for c in ai.calls if c[0] == "voice"] == [("voice", sw.NARRATOR_VOICE), ("voice", "Kore")]


def test_every_character_gets_a_voice_when_created_and_old_empty_ones_are_saved(env):
    client, db, store, files, ai, gen, ren = env
    u = make_user(db)
    grant(client, u, 300)
    pid = new_project(client, u)
    c = client.post(f"/api/v1/studio/projects/{pid}/characters", headers=H(u), json={"name": "Maya"}).json()
    assert c["voice"] in studio_ai.VOICE_NAMES
    db.execute("update public.studio_characters set voice = '' where id = %s", (c["id"],))   # made before this fix
    s = _voiced_scene(client, u, pid)
    start(client, u, pid, "voice", [s["id"]])
    run(gen)
    saved = db.execute("select voice from public.studio_characters where id = %s", (c["id"],)).fetchone()[0]
    assert saved in studio_ai.VOICE_NAMES and saved != sw.NARRATOR_VOICE
    assert ("voice", saved) in ai.calls
    s2 = _voiced_scene(client, u, pid, text="Again")
    start(client, u, pid, "voice", [s2["id"]])
    run(gen)
    assert [c for c in ai.calls if c[0] == "voice"] == [("voice", saved), ("voice", saved)], "same voice every time"


def test_a_voice_provider_failure_mid_scene_refunds(env):
    client, db, store, files, ai, gen, ren = env
    u = make_user(db)
    grant(client, u, 300)
    pid = new_project(client, u)
    client.post(f"/api/v1/studio/projects/{pid}/characters", headers=H(u), json={"name": "Maya", "voice": "Kore"})
    s = client.post(f"/api/v1/studio/projects/{pid}/scenes", headers=H(u), json={"visual": "x", "dialogue": [
        {"character": "Maya", "text": "One"}, {"character": "Maya", "text": "Two"}]}).json()
    real, calls = ai.voice, []

    def second_line_fails(text, voice, style=""):
        calls.append(text)
        if len(calls) == 2:
            raise studio_ai.ProviderError("boom")
        return real(text, voice, style)

    ai.voice = second_line_fails
    start(client, u, pid, "voice", [s["id"]])
    run(gen)
    assert calls == ["One", "Two"]
    assert db.execute("select audio_path from public.studio_scenes where id = %s", (s["id"],)).fetchone()[0] is None
    assert balance(client, u) == {"available": 300, "purchased": 0, "promotional": 300, "held": 0}


# ---------------------------------------------------------------- 4. language

def test_the_project_language_reaches_the_breakdown_and_dialogue_is_kept_as_written(env):
    client, db, store, files, ai, gen, ren = env
    u = make_user(db)
    grant(client, u, 2000)
    pid = new_project(client, u, language="Kinyarwanda")
    client.patch(f"/api/v1/studio/projects/{pid}", headers=H(u),
                 json={"script": "INT. ISOKO - UMUNSI\n\nAMANI\nMuraho, Keza! Amakuru yawe?\n"})
    lines = [{"character": "Amani", "text": "Muraho, Keza! Amakuru yawe?"}]
    ai.breakdown_result = {"characters": [{"name": "Amani", "description": "Young man, green shirt", "gender": "male"}],
                           "scenes": [{"heading": "INT. ISOKO - UMUNSI", "visual": "A busy market, Amani waves",
                                       "dialogue": lines, "duration_seconds": 4}]}
    assert start(client, u, pid, "breakdown").status_code == 200
    run(gen)
    assert ai.last_breakdown_brief["language"] == "Kinyarwanda"
    prompt = ai.last_breakdown_prompt
    assert "The production language is Kinyarwanda" in prompt
    assert "Never translate dialogue" in prompt and "write them in English" in prompt
    assert detail(client, u, pid)["scenes"][0]["dialogue"] == lines


def test_breakdown_prompt_defaults_to_english():
    assert "The production language is English" in studio_ai.breakdown_prompt({"kind": "episode"}, "x")
