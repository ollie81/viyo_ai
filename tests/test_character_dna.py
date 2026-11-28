"""Character DNA and distinct casts.

Pure checks (similarity, family, the cast check inside a drama and across a creator's dramas, picture fingerprints,
returning characters, the brief given to a new identity), then the pipeline with a fake AI: new identities are told
who is already in the drama, look-alikes are pointed out in the production quote and by the cast-check endpoint, and
the DNA card gathers everything about one character (and only its owner can read it)."""
import os
import uuid

import pytest

import studio_dna as dna
from conftest import make_user
from test_creator_studio import H, env, grant, run, start, studio_schema  # noqa: F401
from test_video_drama import DAN, MAYA, project, split

pytestmark = pytest.mark.skipif(bool(os.environ.get("STUDIO_PREMIGRATION")), reason="needs the Studio schema")


def who(name, hair, skin, build, outfit, gender="female", age="young_adult", desc="", cid=None, identity_id=None):
    return {"id": cid or name, "name": name, "description": desc, "gender_presentation": gender, "age_category": age,
            "identity_id": identity_id,
            "identity": {"basic": {"skin_tone": skin, "build": build}, "face": {"hair": hair, "face_shape": "oval",
                                                                                "eyes": "dark brown almond"},
                         "default_outfit": outfit}}


AMA = who("Ama Mensah", "short black curly hair", "deep brown", "slim, tall", "red jacket, jeans")
ABENA = who("Abena", "short black curly hair", "deep brown", "slim tall", "red jacket and jeans")
KOFI = who("Kofi", "shaved head", "light brown", "broad, heavy", "grey suit", gender="male", age="middle_aged")


def test_look_alikes_are_scored_by_what_shows_on_screen():
    score, same = dna.similarity(AMA, ABENA)
    assert score >= 0.8 and {"hair", "skin tone", "build", "clothes"} <= set(same)
    low, _ = dna.similarity(AMA, KOFI)
    assert low < dna.LOOK_ALIKE
    assert dna.similarity({"name": "x"}, {"name": "y"})[0] == 0.0, "nothing to compare yet"


def test_family_is_recognised_and_softened():
    sister = {**ABENA, "description": "the younger sister of Ama"}
    assert dna.related(AMA, sister) and dna.related(AMA, {**ABENA, "name": "Abena Mensah"})
    assert not dna.related(AMA, ABENA)
    pairs = dna.cast_check([AMA, sister])["pairs"]
    assert pairs and pairs[0]["severity"] == "low" and "family" in pairs[0]["problem"]


def test_the_cast_check_names_what_matches_and_what_to_change():
    out = dna.cast_check([AMA, ABENA, KOFI])
    assert len(out["pairs"]) == 1
    p = out["pairs"][0]
    assert (p["a"], p["b"], p["scope"], p["severity"]) == ("Ama Mensah", "Abena", "this drama", "high")
    assert "look alike" in p["problem"] and p["fix"].startswith("Give Abena a different hair colour")


def test_the_same_reference_picture_is_flagged_whatever_the_text_says():
    out = dna.cast_check([AMA, KOFI], fingerprints={"Ama Mensah": "0f0f0f0f0f0f0f0f", "Kofi": "0f0f0f0f0f0f0f0e"})
    assert out["pairs"][0]["same_picture"] and "almost the same reference picture" in out["pairs"][0]["problem"]


def test_other_dramas_returning_characters_and_clear_repeats():
    other = {**ABENA, "id": "o1", "name": "Efua", "project_title": "Kente Nights"}
    out = dna.cast_check([AMA], [other])
    assert out["pairs"][0]["scope"] == 'your drama "Kente Nights"' and "Each drama's cast should be its own" in \
        out["pairs"][0]["fix"]
    same_person = {**AMA, "id": "o2", "identity_id": "Ama Mensah", "project_title": "Episode 2"}
    assert dna.cast_check([AMA], [same_person])["pairs"] == [], "a returning character is the same person on purpose"
    assert dna.cast_check([AMA], [{**KOFI, "project_title": "x"}])["pairs"] == []


def test_a_new_identity_is_told_who_is_already_cast():
    block = dna.distinct_block([AMA, KOFI, {"name": "New", "identity": {}}], "New")
    assert "CHARACTERS ALREADY IN THIS DRAMA" in block and "Ama Mensah: deep brown; short black curly hair" in block
    assert "Kofi" in block and "unless the story says they are family" in block
    assert dna.distinct_block([{"name": "New"}], "New") == ""


# ------------------------------------------------------------------ the pipeline

def film(env):
    client, db, store, files, ai, gen, ren = env
    u = make_user(db)
    grant(client, u, 100000)
    pid = project(client, u, production_style="pictures", genre="Drama")["id"]
    scenes = [{"heading": "INT. FLAT - NIGHT", "visual": "Maya and Daniel at the table", "duration_seconds": 6,
               "dialogue": [{"character": "Maya", "text": "You came back.", "emotion": "hopeful"},
                            {"character": "Daniel", "text": "I had to."}]}]
    split(client, ai, gen, u, pid, scenes, characters=(MAYA, DAN))
    return client, db, store, files, ai, gen, ren, u, pid


def test_new_identities_are_told_the_cast_and_look_alikes_reach_the_quote(env, monkeypatch):
    client, db, store, files, ai, gen, ren, u, pid = film(env)
    seen = []
    real = ai.identity
    monkeypatch.setattr(ai, "identity", lambda name, desc, ctx="": seen.append((name, ctx)) or real(name, desc, ctx))
    chars = client.get(f"/api/v1/studio/projects/{pid}", headers=H(u)).json()["characters"]
    for c in chars:                                     # each character's reference sheet (the fake gives one face)
        r = client.post(f"/api/v1/studio/projects/{pid}/jobs", headers=H(u),
                        json={"kind": "portrait", "target_ids": [c["id"]], "request_id": uuid.uuid4().hex, "views": ["front_full"]})
        assert r.status_code == 200, r.text
        run(gen)
    assert len(seen) == 2 and "CHARACTERS ALREADY IN THIS DRAMA" not in seen[0][1]
    assert "CHARACTERS ALREADY IN THIS DRAMA" in seen[1][1] and seen[0][0] in seen[1][1], \
        "the second character is told who is already cast"
    check = client.get(f"/api/v1/studio/projects/{pid}/cast-check", headers=H(u)).json()
    assert check["checked"] == 2 and check["pairs"], "the fake AI drew the same identity twice: flagged"
    pair = check["pairs"][0]
    assert {pair["a"], pair["b"]} == {"Maya", "Daniel"} and pair["fix"].startswith("Give ")
    q = client.post(f"/api/v1/studio/projects/{pid}/production/quote", headers=H(u), json={}).json()
    assert any("look alike" in a or "reference picture" in a for a in q["advice"]), "pointed out before paying"


def test_the_dna_card_gathers_everything_and_only_for_the_owner(env):
    client, db, store, files, ai, gen, ren, u, pid = film(env)
    maya = next(c for c in client.get(f"/api/v1/studio/projects/{pid}", headers=H(u)).json()["characters"]
                if c["name"] == "Maya")
    client.post(f"/api/v1/studio/projects/{pid}/jobs", headers=H(u),
                json={"kind": "portrait", "target_ids": [maya["id"]], "request_id": uuid.uuid4().hex, "views": ["front_full"]})
    run(gen)
    db.execute("update public.studio_projects set story_bible = %s where id = %s",
               ('{"saga": {"characters": [{"name": "Maya", "role": "lead", "wants": "the truth", "arc": "from fear to '
                'courage"}], "relationships": [{"between": "Maya and Daniel", "kind": "love", "tension": "his lie"}]}}', pid))
    r = client.get(f"/api/v1/studio/characters/{maya['id']}/dna", headers=H(u))
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["name"] == "Maya" and d["look"]["identity"]["face"]["hair"]
    assert next(v for v in d["look"]["reference_views"] if v["view"] == "front_full")["url"]
    assert d["voice"]["status"] in ("cast", "needs_voice") and d["person"]["personality"] == "brave, warm"
    assert d["story"] == {"role": "lead", "wants": "the truth", "arc": "from fear to courage"}
    assert d["relationships"] == [{"between": "Maya and Daniel", "kind": "love", "tension": "his lie"}]
    assert d["emotional_state"] == {"emotion": "hopeful", "scene": 1} and d["appears_in_scenes"] == [1]
    assert set(d["consistency"]) >= {"shots", "checked", "flagged"} and isinstance(d["look_alikes"], list)
    stranger = make_user(db)
    assert client.get(f"/api/v1/studio/characters/{maya['id']}/dna", headers=H(stranger)).status_code == 404
