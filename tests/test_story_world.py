"""The story world and the larger story: built once, labelled honestly,
correctable by the creator, inherited by later episodes, and passed into the
story plan, screenplay, story check, breakdown, character pictures, scene and
shot pictures, video clips, places/objects, outfits, music and the assistant.

Fake AI and local Postgres: the worlds below are scripted. These tests prove
that the right world reaches the right step and that wrong details are caught;
they do NOT prove the real model knows Rwandan or Japanese history. That needs
a real-Gemini run (with web search), opt-in.
"""
import os
import uuid

import pytest

import creator_studio as cs
import studio_music
import studio_saga as saga_mod
import studio_story as story
import studio_story_check as sc
import studio_world as wb
from conftest import make_user
from test_creator_studio import H, balance, env, grant, new_project, run, start, studio_schema  # noqa: F401
from test_episode_storytelling import Room, script_for

pytestmark = pytest.mark.skipif(bool(os.environ.get("STUDIO_PREMIGRATION")), reason="needs the story SQL")

RWANDA_1500 = {
    "setting": {"country": "Rwanda", "region": "central highlands", "place": "a royal court (ibwami) near Nyanza",
                "era": "around 1500, the early kingdom", "year": 1500,
                "summary": "A small highland kingdom of hills and banana groves, ruled by the mwami and his court."},
    "society": {"power": "the mwami (king) and court ritualists", "social_structure": "lineages and clans; cattle mean wealth",
                "family": "patrilineal households in enclosed compounds"},
    "culture": {"traditions": ["poetry recited at court", "cattle as gifts that bind loyalty"],
                "respect_and_greetings": "measured speech, indirectness before elders and the king",
                "how_emotion_is_shown": "restraint in public; grief and anger kept private"},
    "language": {"spoken": "an early form of Kinyarwanda", "naming": "names carrying a meaning or a wish",
                 "forms_of_address": ["Nyagasani (Lord) for the king"]},
    "material": {"architecture": "round thatched houses of reeds and grass inside woven fences",
                 "clothing": "barkcloth and cow-hide garments; no woven cotton cloth",
                 "clothing_by_status": "finer hides and ornaments for the court",
                 "materials": ["barkcloth", "cow hide", "reed", "clay"], "objects": ["milk pots", "spears", "woven baskets"]},
    "technology": {"available": ["iron working"], "transport": "on foot",
                   "not_available": ["firearms", "woven cotton cloth", "wheeled carts"]},
    "environment": {"landscape": "steep green hills, mist in the valleys", "light": "soft morning mist, hard noon sun",
                    "sounds": ["cattle lowing", "drums from the court"]},
    "visual": {"palette": "ochre, deep green, cream of milk", "textures": "barkcloth, woven reed"},
    "sound": {"ambience": ["wind in banana leaves"], "music": "drums and voice", "instruments": ["ingoma drums", "inanga"]},
    "facts": [{"claim": "Cattle were central to wealth and political ties", "status": "documented", "source": 1},
              {"claim": "The exact court rituals of 1500 are recorded", "status": "documented", "source": 0},
              {"claim": "The prince's rival lineage", "status": "invented"},
              {"claim": "Dynastic dates before the 1600s", "status": "uncertain", "note": "oral tradition, disputed"}],
    "unknown": ["Precise dates of reigns before the 17th century"],
    "avoid": ["kente cloth", "Yoruba gele head-wraps"],
}
JAPAN_1600 = {"setting": {"country": "Japan", "place": "a castle town in Owari", "era": "late Sengoku, 1590s", "year": 1595,
                          "summary": "War between lords; samurai households and merchants."},
              "material": {"clothing": "kimono and hakama; armour for war", "architecture": "castle keeps, tiled roofs"},
              "technology": {"available": ["matchlock guns (tanegashima)"], "not_available": ["printing press newspapers"]},
              "facts": [{"claim": "Matchlocks arrived in 1543", "status": "documented", "source": 1}],
              "unknown": ["The fictional lord's clan"], "avoid": ["Chinese qipao", "Korean hanbok", "hanbok"]}
LAGOS_TODAY = {"setting": {"country": "Nigeria", "place": "Yaba, Lagos", "era": "today", "year": 2025,
                           "summary": "A tech hub with danfo buses and traffic."},
               "material": {"clothing": "ankara prints, jeans, agbada for occasions"},
               "facts": [], "unknown": [], "avoid": ["kinyarwanda"]}
FICTION = {"setting": {"country": "", "place": "the floating city of Oru", "era": "an invented age", "year": 0,
                       "summary": "Inspired by medieval Swahili coast trading towns."},
           "material": {"clothing": "dyed cotton wraps"}, "facts": [{"claim": "Oru floats", "status": "invented"}],
           "unknown": [], "avoid": ["Viking helmets"]}
SOURCES = [{"title": "Encyclopaedia of Rwanda", "url": "https://example.org/rwanda"}]


@pytest.fixture
def studio(env, monkeypatch):
    client, db, store, files, ai, gen, ren = env
    monkeypatch.setattr(cs, "_ai", ai)
    monkeypatch.setattr(cs, "_files", files)
    u = make_user(db)
    grant(client, u, 50000)
    return client, db, store, files, ai, gen, ren, u


def build(client, u, pid, mode="grounded", setting="Rwanda around 1500"):
    r = client.post(f"/api/v1/studio/projects/{pid}/world", headers=H(u),
                    json={"request_id": uuid.uuid4().hex, "mode": mode, "setting": setting})
    assert r.status_code == 200, r.text
    return r.json()


def rwanda(studio, **over):
    client, db, store, files, ai, gen, ren, u = studio
    ai.world_result, ai.world_sources = RWANDA_1500, SOURCES
    pid = new_project(client, u, title="The Drum Prince", logline="A fictional prince in Rwanda around 1500 must "
                      "choose between his father's throne and the woman his clan has sworn to destroy.", **over)
    return pid, build(client, u, pid)


# ------------------------------------------------------------------ building the world

def test_a_historical_rwanda_world_is_researched_and_labelled_honestly(studio):
    client, db, store, files, ai, gen, ren, u = studio
    before = balance(client, u)["available"]
    pid, out = rwanda(studio)
    w = out["world"]
    assert w["mode"] == "grounded" and w["researched"] is True and w["sources"][0]["url"] == SOURCES[0]["url"]
    statuses = [f["status"] for f in w["facts"]]
    assert statuses == ["documented", "unsourced", "invented", "uncertain"], \
        "a 'documented' claim with no source is downgraded, never shown as history"
    assert w["facts"][0]["source"]["title"] == "Encyclopaedia of Rwanda"
    assert out["check"]["facts"]["unsourced"] == 1 and any(p["kind"] == "unsourced" for p in out["check"]["problems"])
    prompt = ai.world_prompts[-1]
    assert "never treat a continent, region or country as one culture" in prompt and "Never label a guess as documented" in prompt
    assert "Required for every real place" in prompt
    assert "Historically grounded fiction" in prompt and "Rwanda around 1500" in prompt
    assert 0 < before - balance(client, u)["available"] <= cs.pricing.estimate_world(True).credits


def test_the_creator_chooses_grounded_alternate_fictional_or_modern(studio):
    client, db, store, files, ai, gen, ren, u = studio
    pid = new_project(client, u)
    for mode, words in (("alternate", "one thing went differently"), ("fictional", "inspired by a real culture"),
                        ("modern", "A real place today")):
        ai.world_result = FICTION if mode == "fictional" else LAGOS_TODAY
        build(client, u, pid, mode=mode, setting="x")
        assert words in ai.world_prompts[-1]
    assert ai.calls[-1] == ("build_world", "modern")
    w = client.get(f"/api/v1/studio/projects/{pid}/world", headers=H(u)).json()
    assert w["world"]["mode"] == "modern" and set(w["modes"]) == {"grounded", "alternate", "fictional", "modern"}


def test_a_failed_world_build_charges_nothing(studio):
    client, db, store, files, ai, gen, ren, u = studio
    pid = new_project(client, u)
    ai.fail = {"build_world": RuntimeError("down")}
    before = balance(client, u)["available"]
    r = client.post(f"/api/v1/studio/projects/{pid}/world", headers=H(u), json={"request_id": uuid.uuid4().hex})
    assert r.status_code == 502 and "weren't charged" in r.text and balance(client, u)["available"] == before


# ------------------------------------------------------------------ the world's own checks

def test_the_world_check_catches_thin_overconfident_and_self_contradicting_worlds():
    w = wb.clean_world({**RWANDA_1500, "unknown": [], "facts": [{"claim": "x", "status": "documented", "source": 1}],
                        "material": {**RWANDA_1500["material"], "clothing": "jeans and cotton shirts"}},
                       "grounded", SOURCES, researched=True)
    kinds = [p["kind"] for p in wb.evaluate(w)["problems"]]
    assert "overconfident" in kinds, "a historical world that admits no uncertainty is suspicious"
    assert "anachronism" in kinds, "jeans in 1500 inside the world itself"
    thin = wb.evaluate(wb.clean_world({"setting": {"country": "X", "place": "y", "era": "z", "year": 1200, "summary": "s"}},
                                      "grounded"))
    assert any(p["kind"] == "thin" for p in thin["problems"])


@pytest.mark.parametrize("world,mode,text,anachronisms,confusions", [
    (RWANDA_1500, "grounded", "Gakire checks his phone, then wraps a kente cloth around his shoulders.", ["phone"], ["kente cloth"]),
    (RWANDA_1500, "grounded", "His guards carry firearms beside a wheeled cart.", ["firearms", "wheeled carts"], []),
    (JAPAN_1600, "grounded", "The ashigaru fire matchlock guns; a lady in a hanbok watches.", [], ["hanbok"]),
    (JAPAN_1600, "grounded", "A samurai in kimono reads a letter by lamplight.", [], []),
    ({"setting": {"country": "Egypt", "place": "Thebes", "era": "New Kingdom", "year": -1300, "summary": "s"},
      "avoid": ["toga", "camel caravans"]}, "grounded",
     "The scribe rides a bicycle past a man in a toga.", ["bicycle"], ["toga"]),
    (LAGOS_TODAY, "modern", "Tunde checks his phone in a danfo bus.", [], []),
    (FICTION, "fictional", "On Oru, a merchant checks a phone-like shell; a guard wears Viking helmets.", [], ["viking helmets"]),
])
def test_wrong_details_are_caught_for_each_world(world, mode, text, anachronisms, confusions):
    w = wb.clean_world(world, mode)
    assert wb.anachronisms(text, w) == sorted(anachronisms)
    assert [c.lower() for c in wb.confusions(text, w)] == confusions


# ------------------------------------------------------------------ corrections

def test_creator_corrections_override_everything_and_survive_a_rebuild(studio):
    client, db, store, files, ai, gen, ren, u = studio
    pid, _ = rwanda(studio)
    client.post(f"/api/v1/studio/projects/{pid}/world/approve", headers=H(u), json={"approved": True})
    r = client.post(f"/api/v1/studio/projects/{pid}/world/correct", headers=H(u), json={
        "field": "material.clothing", "value": "barkcloth wraps (impuzu) for women; cow-hide for men",
        "note": "My grandmother's description"})
    assert r.status_code == 200
    w = r.json()["world"]
    assert w["material"]["clothing"].startswith("barkcloth wraps") and w["approved"] is False, "re-approval needed"
    text = wb.block(w, "image")
    assert text.startswith("CREATOR CORRECTIONS (override everything below): material.clothing: barkcloth wraps")
    build(client, u, pid)
    assert "CREATOR CORRECTIONS (these override everything):\n- material.clothing: barkcloth wraps" in ai.world_prompts[-1]
    w = client.get(f"/api/v1/studio/projects/{pid}/world", headers=H(u)).json()["world"]
    assert w["corrections"][0]["note"] == "My grandmother's description"


# ------------------------------------------------------------------ the world reaches production

def test_the_world_reaches_the_story_plan_screenplay_and_story_check(studio):
    client, db, store, files, ai, gen, ren, u = studio
    pid, _ = rwanda(studio)
    room = Room()
    ai.story_fns = (room.text, room.json)
    start(client, u, pid, "script")
    run(gen)
    plan_p = [p[1] for p in room.prompts if p[0] == "json" and p[2] is story.PLAN_SCHEMA][0]
    assert "STORY WORLD (Historically grounded fiction): a royal court (ibwami) near Nyanza" in plan_p
    assert "[documented] Cattle were central to wealth" in plan_p and "[unsourced] The exact court rituals" in plan_p
    assert "Unknown or disputed: Precise dates of reigns" in plan_p and "IMMERSION (use the world" in plan_p
    draft_p = [p[1] for p in room.prompts if p[0] == "text"][0]
    assert "STORY WORLD" in draft_p
    judge = [p[1] for p in room.prompts if p[0] == "json" and p[2] is sc.SCHEMA][0]
    assert "- authenticity: quote a line or action" in judge and "- immersion:" in judge
    bible = client.get(f"/api/v1/studio/projects/{pid}", headers=H(u)).json()["project"]["story_bible"]
    assert bible["world"]["setting"]["country"] == "Rwanda", "writing the story keeps the world"
    assert set(sc.required(False, True)) >= {"authenticity", "immersion"}
    assert bible["evaluation"]["status"] == "incomplete" and "authenticity" in bible["evaluation"]["unverified"], \
        "the scripted judge didn't score authenticity: that's incomplete, never a pass"


def test_an_anachronism_in_the_screenplay_is_a_failed_check():
    w = wb.clean_world(RWANDA_1500, "grounded")
    script = "INT. COURT - NIGHT\n\nGAKIRE\nBring me my phone, the king is coming.\n"
    checks = story.structure_checks(script, {"kind": "episode", "world": w}, [])
    assert any(c["area"] == "authenticity" and "phone" in c["problem"] for c in checks)


def test_the_world_reaches_breakdown_character_pictures_scenes_video_places_outfits_and_music(studio):
    client, db, store, files, ai, gen, ren, u = studio
    pid, _ = rwanda(studio, production_style="video")
    client.patch(f"/api/v1/studio/projects/{pid}", headers=H(u), json={"script": "INT. COURT - DAY\n\nGakire waits.\n"})
    ai.breakdown_result = {"characters": [{"name": "Gakire", "description": "Prince, 19, lean", "gender": "male"}],
                           "scenes": [{"heading": "INT. COURT - DAY", "visual": "Gakire waits by the fence", "duration_seconds": 5,
                                       "dialogue": [{"character": "Gakire", "text": "Father will not see me."}]}]}
    start(client, u, pid, "breakdown")
    run(gen)
    bp = ai.last_breakdown_prompt
    assert "round thatched houses" in bp and "NEVER SHOW: firearms, woven cotton cloth, wheeled carts, kente cloth" in bp
    assert "Every scene's look, clothing, objects and place follow this world." in bp
    start(client, u, pid, "portrait")
    run(gen)
    portrait = ai.image_log[-1]["prompt"]
    assert "barkcloth and cow-hide garments" in portrait and "Nyagasani" in portrait and "NEVER SHOW" in portrait
    sid = client.get(f"/api/v1/studio/projects/{pid}", headers=H(u)).json()["scenes"][0]["id"]
    start(client, u, pid, "image", [sid])
    run(gen, times=3)
    scene_p = [x["prompt"] for x in ai.image_log if "Gakire waits" in x["prompt"]][-1]
    assert "Look of the world: architecture: round thatched houses" in scene_p and "palette: ochre" in scene_p
    assert "NEVER SHOW: firearms" in scene_p
    start(client, u, pid, "video", [sid])
    run(gen, times=3)
    assert "Background sound: wind in banana leaves, cattle lowing" in ai.last_video_prompt
    assert "restraint in public" in ai.last_video_prompt, "how people behave in this world"
    start(client, u, pid, "elements")
    run(gen)
    assert "STORY WORLD" in ai.last_elements_prompt and "milk pots" in ai.last_elements_prompt
    start(client, u, pid, "outfits")
    run(gen)
    assert "barkcloth and cow-hide garments" in ai.last_outfit_prompt
    p = client.get(f"/api/v1/studio/projects/{pid}", headers=H(u)).json()["project"]
    row = db.execute("select story_bible from public.studio_projects where id = %s", (pid,)).fetchone()[0]
    music = studio_music.music_prompt({**p, "story_bible": row})
    assert "ingoma drums" in music and "Musical world" in music


def test_a_world_alone_does_not_block_breaking_a_pasted_script_into_scenes(studio):
    client, db, store, files, ai, gen, ren, u = studio
    pid, _ = rwanda(studio)
    client.patch(f"/api/v1/studio/projects/{pid}", headers=H(u), json={"script": "INT. COURT - DAY\n\nGakire waits.\n"})
    r = client.post(f"/api/v1/studio/projects/{pid}/jobs", headers=H(u),
                    json={"kind": "breakdown", "request_id": uuid.uuid4().hex})
    assert r.status_code == 200, r.text


# ------------------------------------------------------------------ the larger story

SAGA = {"premise": "A prince caught between his father's throne and a forbidden love.",
        "main_storyline": "Gakire's rise and fall at court.",
        "characters": [{"name": "Gakire", "role": "lead", "history": "raised by his mother's clan", "wants": "the throne",
                        "fears_losing": "his mother", "hides": "he is not the king's son", "inner_conflict": "duty vs love",
                        "arc": "from obedient son to his own man"},
                       {"name": "Uwase", "role": "lead", "history": "daughter of the rival clan", "wants": "peace",
                        "fears_losing": "her brother", "hides": "she spies for her clan", "inner_conflict": "love vs blood",
                        "arc": "from spy to ally"}],
        "relationships": [{"between": "Gakire and Uwase", "kind": "forbidden love"}],
        "secrets": [{"secret": "Gakire is not the king's son", "kept_by": "the queen mother", "reveal_episode": 2},
                    {"secret": "Uwase spies for her clan", "kept_by": "Uwase", "reveal_episode": 5}],
        "mysteries": [{"question": "Who poisoned the drummer?", "answer": "the queen mother", "reveal_episode": 6}],
        "turning_points": [{"episode": 2, "what": "the king learns of the love"}, {"episode": 4, "what": "exile"}],
        "seasons": [{"number": 1, "arc": "love and exile", "ends_with": "Gakire crowned in exile"}],
        "future_seasons": ["The return"], "open_questions": ["Will Uwase choose her clan?"]}


def test_the_larger_story_is_planned_with_the_chosen_length_and_checked(studio):
    client, db, store, files, ai, gen, ren, u = studio
    pid, _ = rwanda(studio)
    ai.saga_result = SAGA
    r = client.post(f"/api/v1/studio/projects/{pid}/saga", headers=H(u),
                    json={"request_id": uuid.uuid4().hex, "mode": "episodes", "count": 5})
    assert r.status_code == 200, r.text
    assert r.json()["problems"] == ["Planned beyond the 5 episodes: episode(s) [6]."]
    assert "a limited series of exactly 5 episodes" in ai.saga_prompts[-1]
    assert "STORY WORLD" in ai.saga_prompts[-1], "the larger story is planned inside the world"
    shallow = saga_mod.clean({**SAGA, "characters": [{"name": "X", "role": "lead", "wants": "y"}] * 2}, {"mode": "open"})
    assert any("without enough depth" in p for p in saga_mod.problems(shallow))


def test_episode_2_inherits_the_world_and_follows_the_larger_story(studio):
    client, db, store, files, ai, gen, ren, u = studio
    pid, _ = rwanda(studio)
    ai.saga_result = SAGA
    client.post(f"/api/v1/studio/projects/{pid}/saga", headers=H(u), json={"request_id": uuid.uuid4().hex, "mode": "open"})
    room = Room()
    ai.story_fns = (room.text, room.json)
    start(client, u, pid, "script")
    run(gen)
    ep2 = client.post(f"/api/v1/studio/projects/{pid}/next-episode", headers=H(u)).json()["id"]
    client.patch(f"/api/v1/studio/projects/{ep2}", headers=H(u), json={"logline": "The king finds out."})
    w = client.get(f"/api/v1/studio/projects/{ep2}/world", headers=H(u)).json()
    assert w["inherited"] is True and w["world"]["setting"]["country"] == "Rwanda" and w["saga"]["premise"]
    start(client, u, ep2, "script")
    run(gen)
    plans = [p[1] for p in room.prompts if p[0] == "json" and p[2] is story.PLAN_SCHEMA and "YOUR PREVIOUS" not in p[1]]
    p2 = plans[-1]
    assert "This is episode 2." in p2 and "Turning point due IN THIS EPISODE: the king learns of the love" in p2
    assert "Secret to reveal now: Gakire is not the king's son" in p2
    assert "Secrets that must STAY hidden in this episode (hint at most): Uwase spies for her clan" in p2
    assert "STORY WORLD (Historically grounded fiction)" in p2, "the series' world, without building it again"
    assert sum(1 for c in ai.calls if c[0] == "build_world") == 1, "researched once, reused"
    b2 = client.get(f"/api/v1/studio/projects/{ep2}", headers=H(u)).json()["project"]["story_bible"]
    assert "Gakire is not the king's son" in b2["saga"]["revealed"]
    assert "- Gakire (lead): wants the throne; afraid of losing his mother; hides he is not the king's son" in p2


# ------------------------------------------------------------------ the assistant

def test_the_assistant_uses_the_world_keeps_labels_and_can_correct_it(studio):
    client, db, store, files, ai, gen, ren, u = studio
    pid, _ = rwanda(studio)
    client.patch(f"/api/v1/studio/projects/{pid}", headers=H(u), json={"script": "INT. COURT - DAY\n\nGakire waits.\n"})
    ai.breakdown_result = {"characters": [{"name": "Gakire", "description": "Prince", "gender": "male"}],
                           "scenes": [{"heading": "INT. COURT - DAY", "visual": "Gakire wears a kente cloth and checks a phone",
                                       "duration_seconds": 5, "dialogue": []}]}
    start(client, u, pid, "breakdown")
    run(gen)
    ai.chat_result = {"reply": "Barkcloth fits; kente is Ghanaian.", "memory": [], "proposals": [
        {"type": "correct_world", "title": "Women wear impuzu", "summary": "x",
         "data": {"field": "material.clothing", "value": "impuzu barkcloth wraps", "note": "creator"}},
        {"type": "build_world", "title": "Research again", "summary": "x", "data": {"mode": "grounded"}}]}
    r = client.post(f"/api/v1/studio/projects/{pid}/assistant/chat", headers=H(u),
                    json={"message": "Would a prince in 1500 Rwanda wear kente?", "request_id": uuid.uuid4().hex})
    assert r.status_code == 200, r.text
    prompt = ai.chat_prompts[-1]
    assert "STORY WORLD (grounded, NOT approved yet, researched with web search, 1 sources)" in prompt
    assert "- [unsourced] The exact court rituals of 1500 are recorded" in prompt
    assert "Unknown/disputed: Precise dates of reigns" in prompt
    assert "WORLD CHECK scene 1: Things that don't exist in around 1500, the early kingdom: phone." in prompt
    assert "WORLD CHECK scene 1: Details from another culture: kente cloth." in prompt
    assert "for any country, region or invented world alike" in prompt
    assert "Never present an invented or uncertain detail as history" in prompt
    fix, rebuild = r.json()["reply"]["proposals"]
    assert fix["paid"] is False and rebuild["paid"] is True and rebuild["job"] == "world"
    msg = r.json()["reply"]["id"]
    assert client.post(f"/api/v1/studio/assistant/messages/{msg}/proposals/0/apply", headers=H(u)).status_code == 200
    w = client.get(f"/api/v1/studio/projects/{pid}/world", headers=H(u)).json()["world"]
    assert w["material"]["clothing"] == "impuzu barkcloth wraps" and w["corrections"][0]["note"] == "creator"


def test_a_detail_mentioned_only_as_absent_is_not_a_mistake():
    w = wb.clean_world(RWANDA_1500, "grounded")
    assert wb.anachronisms("Barkcloth, never woven cotton cloth; no firearms at court.", w) == []
    assert wb.anachronisms("A servant drapes woven cotton cloth over the stool.", w) == ["woven cotton cloth"]


# Every country and place is handled the same way: the confusions come from each world's own research.
PLACES = [
    ({"setting": {"country": "Peru", "place": "Cusco", "era": "Inca Empire, 1500", "year": 1500, "summary": "s"},
      "avoid": ["sombrero", "Aztec feathered headdress"], "technology": {"not_available": ["horses"]}},
     "A runner in a sombrero rides horses toward the Aztec feathered headdress seller.",
     ["horses"], ["aztec feathered headdress", "sombrero"]),
    ({"setting": {"country": "Norway", "place": "a fjord farm", "era": "Viking Age, 900s", "year": 950, "summary": "s"},
      "avoid": ["horned helmets"]}, "The jarl wears horned helmets and checks his wristwatch.", ["wristwatch"], ["horned helmets"]),
    ({"setting": {"country": "South Korea", "place": "Mapo, Seoul", "era": "today", "year": 2025, "summary": "s"},
      "avoid": ["kimono", "Chinese lanterns for every street"]}, "She checks her phone; her mother wears a kimono.", [], ["kimono"]),
    ({"setting": {"country": "India", "place": "Agra", "era": "Mughal court, 1640s", "year": 1645, "summary": "s"},
      "avoid": ["Bollywood dance number"]}, "Courtiers break into a Bollywood dance number lit by a light bulb.",
     ["light bulb"], ["bollywood dance number"]),
]


@pytest.mark.parametrize("world,text,anach,conf", PLACES)
def test_any_country_is_checked_the_same_way_from_its_own_researched_world(world, text, anach, conf):
    w = wb.clean_world(world, "grounded")
    assert wb.anachronisms(text, w) == sorted(anach)
    assert wb.confusions(text, w) == sorted(conf)


def test_no_country_has_a_built_in_list_and_a_world_without_confusions_is_flagged():
    assert not hasattr(wb, "CONFUSIONS"), "the same rule for every place"
    bare = wb.clean_world({**RWANDA_1500, "avoid": []}, "grounded")
    assert wb.confusions("He wears kente cloth.", bare) == [], "nothing is assumed about any country"
    assert any(p["kind"] == "no_confusions" for p in wb.evaluate(bare)["problems"])
    fictional = wb.clean_world({**FICTION, "avoid": []}, "fictional")
    assert not any(p["kind"] == "no_confusions" for p in wb.evaluate(fictional)["problems"])
