"""Story Intelligence stage 2: interview -> outline -> creator approval -> story memory. Real Postgres, the Studio's
real credit path, a fake AI (no paid calls)."""
import copy
import uuid

import pytest

import creator_studio as cs
import studio_ai
import studio_novel as nv
import studio_novel_api
from conftest import ROOT, connect, make_user
from studio_canon import CanonRepo
from test_creator_studio import H, balance, env, grant, new_project, studio_schema  # noqa: F401


@pytest.fixture(scope="module")
def novel_schema(studio_schema):
    conn = connect()
    for f in ("creator_studio_canon.sql", "creator_studio_novel.sql", "creator_studio_novel.sql"):   # twice: safe
        conn.execute((ROOT / f).read_text())
    conn.close()


@pytest.fixture
def st(novel_schema, env, monkeypatch):
    client, db, store, files, ai, gen, ren = env
    monkeypatch.setattr(cs, "_ai", ai)
    client.app.include_router(studio_novel_api.router)      # same app, same signed-in test users
    u = make_user(db)
    grant(client, u, 5000)
    return client, db, store, ai, u


IDEA = ("A young seamstress in Kigali finds a letter sewn into a wedding dress that proves her late mother was "
        "betrayed by the richest family in town. I want it to feel hopeful but heartbreaking.")


def start(client, u, **over):
    pid = new_project(client, u, title="The Dress")
    body = {"project_id": pid, "idea": IDEA, "length": "short", "episodes": 30, **over}
    r = client.post("/api/v1/studio/stories", headers=H(u), json=body)
    assert r.status_code == 200, r.text
    return r.json()["story"]


def ask(client, u, sid, **body):
    return client.post(f"/api/v1/studio/stories/{sid}/interview", headers=H(u),
                       json={"request_id": uuid.uuid4().hex, **body})


def turn(topic, text, understood=()):
    return {"understood": [dict(zip(("topic", "value", "creator_words"), x)) for x in understood],
            "question": {"topic": topic, "text": text, "why": "", "options": ["A", "B"]}}


def outline(n=12, **over):
    o = {
        "title": "The Dress", "logline": "A seamstress unpicks a family's lie.", "premise": "Amani finds a letter.",
        "genre": "family drama", "tone": "hopeful, heartbreaking", "audience_feeling": "hope through grief",
        "themes": ["truth", "inheritance"], "point_of_view": "close third person", "setting_summary": "Kigali today",
        "characters": [
            {"name": "Amani Uwase", "role": "protagonist", "age": "24", "appearance": "tall, a burn scar on her left hand",
             "voice": "soft, quick when angry", "personality": "stubborn, funny", "want": "clear her mother's name",
             "need": "forgive herself", "flaw": "trusts no one", "wound": "found her mother's body", "secret": "",
             "arc": "from suspicion to trust"},
            {"name": "Eric Habimana", "role": "love interest", "age": "27", "appearance": "neat, gold watch",
             "voice": "low", "personality": "charming", "want": "escape his family", "need": "courage",
             "flaw": "cowardice", "wound": "a cruel father", "secret": "his father wrote the letter", "arc": "stands up"},
            {"name": "Mama Grace", "role": "mentor", "age": "60", "appearance": "grey braids", "voice": "warm",
             "personality": "careful", "want": "protect Amani", "need": "tell the truth", "flaw": "fear",
             "wound": "", "secret": "she sewed the letter in", "arc": "confesses"}],
        "relationships": [{"a": "Amani", "b": "Mama Grace", "side": "family", "value": "like a daughter"},
                          {"a": "Amani Uwase", "b": "Eric Habimana", "side": "romance", "value": "strangers"}],
        "places": [{"name": "the tailor shop", "description": "a narrow shop in Nyamirambo"}],
        "objects": [{"name": "the letter", "description": "yellowed paper, sewn into a hem", "why_it_matters": "proof"}],
        "secrets": [{"name": "Who wrote the letter", "truth": "Eric's father wrote it", "known_by": ["Mama Grace"],
                     "revealed_in_chapter": 9, "revealed_to": ["Amani"]}],
        "chapters": [{"number": i, "title": f"Chapter {i}", "purpose": "turn", "summary": f"Things happen {i}.",
                      "beats": ["a", "b", "c"], "pov": "Amani", "emotion": "dread", "setups": [], "payoffs": []}
                     for i in range(1, n + 1)],
        "threads": [{"kind": "foreshadowing", "description": "The gold watch stops at midnight", "planted_chapter": 2,
                     "payoff_chapter": 11}],
        "ending": {"kind": "bittersweet", "description": "She wins the case and loses Eric."},
        "assumptions": ["Set in today's Kigali"],
        "kept_from_creator": [{"topic": "feeling", "how": "Hope through grief in every chapter."},
                              {"topic": "protagonist", "how": "Amani is the seamstress."},
                              {"topic": "goal", "how": "Clearing her mother's name."},
                              {"topic": "genre_tone", "how": "A family drama with a love story."},
                              {"topic": "setting", "how": "Kigali, today."},
                              {"topic": "ending", "how": "Bittersweet: she wins and loses."}]}
    o.update(over)
    return o


def interview_to_done(client, u, ai, sid):
    ai.novel_results = {"interview question": [
        turn("goal", "What does she want most?", [("protagonist", "a young seamstress", "A young seamstress in Kigali"),
                                                    ("feeling", "hopeful but heartbreaking", "hopeful but heartbreaking"),
                                                    ("setting", "Paris in 1920", "she lives in Paris")]),
        turn("genre_tone", "Romance or thriller?", [("goal", "clear her mother's name", "clear my mother's name")]),
        turn("ending", "How does it end?", [("genre_tone", "family drama", "a family drama")]),
        turn("setting", "Where and when?", [("ending", "bittersweet", "bittersweet")]),
        turn("", "", [("setting", "Kigali today", "Kigali, today")]),
    ]}
    r = ask(client, u, sid)
    assert r.status_code == 200, r.text
    s = r.json()["story"]
    for answer in ("She wants to clear my mother's name.", "It's a family drama with a love story.",
                   "Bittersweet.", "Kigali, today."):
        assert s["interview"]["question"], s
        r = ask(client, u, sid, answer=answer)
        assert r.status_code == 200, r.text
        s = r.json()["story"]
    return s


# ------------------------------------------------------------------------------------------- the interview

def test_interview_keeps_only_what_the_creator_really_said(st):
    client, db, store, ai, u = st
    s = start(client, u)
    ai.novel_results = {"interview question": turn("goal", "What does she want most?", [
        ("protagonist", "a young seamstress", "A young seamstress in Kigali"),
        ("feeling", "hopeful but heartbreaking", "hopeful, but heartbreaking!"),      # punctuation doesn't matter
        ("setting", "Paris in 1920", "she lives in Paris")])}                           # never said: not kept
    before = balance(client, u)["available"]
    r = ask(client, u, s["id"])
    assert r.status_code == 200, r.text
    s = r.json()["story"]
    assert set(s["interview"]["known"]) == {"protagonist", "feeling"}
    assert s["interview"]["unverified"] == ["setting"]
    assert "setting" in s["interview"]["missing"]
    assert s["interview"]["question"]["topic"] == "goal"
    assert s["interview"]["question"]["options"] == ["A", "B"]
    assert balance(client, u)["available"] < before                         # a question is a paid step
    assert "A young seamstress in Kigali" in ai.novel_prompts[-1][1]         # the idea is in the prompt


def test_a_question_about_something_already_known_is_replaced(st):
    client, db, store, ai, u = st
    s = start(client, u)
    ai.novel_results = {"interview question": turn("feeling", "How should it feel?",
                                                   [("feeling", "hopeful", "hopeful but heartbreaking")])}
    s = ask(client, u, s["id"]).json()["story"]
    q = s["interview"]["question"]
    assert q["topic"] == "protagonist" and q["text"] == nv.TOPICS["protagonist"]


def test_full_flow_idea_to_approved_outline_and_story_memory(st):
    client, db, store, ai, u = st
    s = start(client, u)
    s = interview_to_done(client, u, ai, s["id"])
    assert s["interview"]["ready"] and s["interview"]["missing"] == []
    assert s["interview"]["known"]["setting"]["creator_words"] == "Kigali, today"
    # Once ready, asking again is free and asks nothing.
    calls = len(ai.novel_prompts)
    before = balance(client, u)["available"]
    assert ask(client, u, s["id"]).json()["story"]["interview"]["question"] is None
    assert len(ai.novel_prompts) == calls and balance(client, u)["available"] == before

    ai.novel_results = {"novel outline": outline()}
    r = client.post(f"/api/v1/studio/stories/{s['id']}/outline", headers=H(u), json={"request_id": uuid.uuid4().hex})
    assert r.status_code == 200, r.text
    s = r.json()["story"]
    assert s["stage"] == "outline" and s["outline_version"] == 1 and s["problems"] == []
    prompt = ai.novel_prompts[-1][1]
    assert "exactly 12 chapters" in prompt and "about 30 drama episodes" in prompt
    assert "Kigali, today" in prompt and "WHAT THE CREATOR DECIDED" in prompt
    assert "GENRE CRAFT" in prompt

    r = client.post(f"/api/v1/studio/stories/{s['id']}/outline/approve", headers=H(u), json={"version": 1})
    assert r.status_code == 200, r.text
    assert r.json()["story"]["stage"] == "outline_approved"
    mem = r.json()["memory"]
    assert mem["entities"] == 6 and mem["threads"] == 2          # 3 people, a place, an object, a secret

    canon = CanonRepo(store, s["id"], str(u)).load()
    amani, grace, secret = canon.eid("Amani Uwase"), canon.eid("Mama Grace"), canon.eid("Who wrote the letter")
    assert all(canon.by_id[i]["status"] == "approved" and canon.by_id[i]["origin"] == "import" for i in (amani, grace, secret))
    st1 = canon.state_at(1)
    assert st1.traits_of(amani)["appearance"] == "tall, a burn scar on her left hand"
    assert st1.knows(grace, secret) and not st1.knows(amani, secret)
    assert [f["value"] for f in st1.relationships_of(amani) if f["attribute"] == "family"] == ["like a daughter"]
    reveal = [t for t in canon.threads if t["description"].startswith("Reveal: Who wrote the letter")][0]
    assert reveal["due_chapter"] == 9 and reveal["status"] == "approved"
    facts = [f for f in canon.facts if f["subject_id"] == amani]
    assert facts and all(f["source_ref"] == "outline v1" and f["source_quote"] and f["decided_by"] == "creator"
                         for f in facts)
    # The memory works for writing straight away: chapter 5 with Amani gets her scar and the secret kept from her.
    got = canon.retrieve({"at": 5, "text": "Amani opens the shop early."})
    assert any("burn scar" in f["statement"] for f in got["facts"])
    assert got["secrets"] and got["secrets"][0]["kept_from"] == ["Amani Uwase"]


def test_skip_and_finish_turn_open_topics_into_listed_assumptions(st):
    client, db, store, ai, u = st
    s = start(client, u)
    ai.novel_results = {"interview question": turn("ending", "How should it end?")}
    ask(client, u, s["id"])
    r = ask(client, u, s["id"], skip=True, finish=True)
    assert r.status_code == 200, r.text
    s = r.json()["story"]
    assert s["interview"]["ready"] and s["interview"]["turns"][0]["skipped"]
    ai.novel_results = {"novel outline": outline(kept_from_creator=[])}
    s = client.post(f"/api/v1/studio/stories/{s['id']}/outline", headers=H(u),
                    json={"request_id": uuid.uuid4().hex}).json()["story"]
    prompt = ai.novel_prompts[-1][1]
    assert "NOT DECIDED BY THE CREATOR" in prompt and "ending" in prompt.split("NOT DECIDED BY THE CREATOR")[1]


def test_interview_stops_after_the_maximum_number_of_questions(st):
    client, db, store, ai, u = st
    s = start(client, u)
    ai.novel_results = {"interview question": lambda p: turn("twist", "Any twist?")}
    s = ask(client, u, s["id"]).json()["story"]
    while s["interview"]["question"]:                       # answering returns the next question
        r = ask(client, u, s["id"], answer="Not sure yet.")
        assert r.status_code == 200, r.text
        s = r.json()["story"]
    assert s["interview"]["asked"] == nv.MAX_QUESTIONS
    final = ask(client, u, s["id"]).json()["story"]
    assert final["interview"]["ready"] and final["interview"]["asked"] <= nv.MAX_QUESTIONS
    assert final["interview"]["question"] is None


def test_an_answer_is_kept_and_nothing_charged_when_the_ai_fails(st):
    client, db, store, ai, u = st
    s = start(client, u)
    ai.novel_results = {"interview question": turn("goal", "What does she want?")}
    ask(client, u, s["id"])
    before = balance(client, u)["available"]
    ai.fail = {"novel_json": studio_ai.ProviderError("down")}
    r = ask(client, u, s["id"], answer="To clear my mother's name.")
    assert r.status_code == 502 and "weren't charged" in r.json()["detail"]
    assert balance(client, u)["available"] == before
    got = client.get(f"/api/v1/studio/stories/{s['id']}", headers=H(u)).json()["story"]
    assert got["interview"]["turns"][0]["answer"] == "To clear my mother's name."


def test_no_credits_no_question(st):
    client, db, store, ai, u = st
    poor = make_user(db)
    s = start(client, poor)
    r = ask(client, poor, s["id"])
    assert r.status_code == 402


def test_only_the_owner_sees_and_changes_a_story(st):
    client, db, store, ai, u = st
    s = start(client, u)
    other = make_user(db)
    assert client.get(f"/api/v1/studio/stories/{s['id']}", headers=H(other)).status_code == 404
    assert ask(client, other, s["id"]).status_code == 404
    assert client.post(f"/api/v1/studio/stories/{s['id']}/outline/approve", headers=H(other),
                       json={"version": 1}).status_code == 404


def test_one_story_per_production(st):
    client, db, store, ai, u = st
    s = start(client, u)
    again = client.post("/api/v1/studio/stories", headers=H(u),
                        json={"project_id": s["project_id"], "idea": "A totally different idea here."})
    assert again.json()["existing"] is True and again.json()["story"]["id"] == s["id"]
    assert client.get(f"/api/v1/studio/projects/{s['project_id']}/story", headers=H(u)).json()["story"]["id"] == s["id"]


# ------------------------------------------------------------------------------------------- the outline

def _ready_story(client, u, ai):
    s = start(client, u)
    ai.novel_results = {"interview question": turn("goal", "?")}
    ask(client, u, s["id"])
    return ask(client, u, s["id"], skip=True, finish=True).json()["story"]


def test_a_plan_with_problems_is_sent_back_once_and_the_better_one_kept(st):
    client, db, store, ai, u = st
    s = _ready_story(client, u, ai)
    bad = outline(n=10, secrets=[{"name": "x", "truth": "y", "known_by": ["Nobody Known"], "revealed_in_chapter": 40}])
    ai.novel_results = {"novel outline": [bad, outline()]}
    s = client.post(f"/api/v1/studio/stories/{s['id']}/outline", headers=H(u),
                    json={"request_id": uuid.uuid4().hex}).json()["story"]
    fix = ai.novel_prompts[-1][1]
    assert "YOUR PLAN HAS THESE PROBLEMS" in fix and "it needs exactly 12" in fix and "outside the book" in fix
    assert s["problems"] == [] and len(s["outline"]["chapters"]) == 12


def test_a_worse_fix_is_not_used_and_blocking_problems_stop_approval(st):
    client, db, store, ai, u = st
    s = _ready_story(client, u, ai)
    first = outline(n=11)
    worse = outline(n=3, ending={"kind": "", "description": ""})
    ai.novel_results = {"novel outline": [first, worse]}
    s = client.post(f"/api/v1/studio/stories/{s['id']}/outline", headers=H(u),
                    json={"request_id": uuid.uuid4().hex}).json()["story"]
    assert len(s["outline"]["chapters"]) == 11
    assert s["blocking"] == ["The plan has 11 chapters; it needs exactly 12."]
    r = client.post(f"/api/v1/studio/stories/{s['id']}/outline/approve", headers=H(u), json={"version": 1})
    assert r.status_code == 409 and r.json()["detail"]["problems"]


def test_warnings_dont_block_and_the_creator_can_edit_then_approve_the_new_version(st):
    client, db, store, ai, u = st
    s = _ready_story(client, u, ai)
    o = outline(relationships=[{"a": "Amani", "b": "A Stranger", "side": "work", "value": "rival"}])
    ai.novel_results = {"novel outline": [o, o]}
    s = client.post(f"/api/v1/studio/stories/{s['id']}/outline", headers=H(u),
                    json={"request_id": uuid.uuid4().hex}).json()["story"]
    assert s["problems"] and not s["blocking"]                       # a warning only
    edited = copy.deepcopy(s["outline"])
    edited["relationships"] = [{"a": "Amani", "b": "Eric Habimana", "side": "work", "value": "rival"}]
    r = client.put(f"/api/v1/studio/stories/{s['id']}/outline", headers=H(u), json={"outline": edited})
    assert r.status_code == 200 and r.json()["story"]["outline_version"] == 2 and r.json()["story"]["problems"] == []
    stale = client.post(f"/api/v1/studio/stories/{s['id']}/outline/approve", headers=H(u), json={"version": 1})
    assert stale.status_code == 409 and "changed" in stale.json()["detail"]
    ok = client.post(f"/api/v1/studio/stories/{s['id']}/outline/approve", headers=H(u), json={"version": 2})
    assert ok.status_code == 200
    # Approved: no silent replanning or editing over the story memory.
    assert client.post(f"/api/v1/studio/stories/{s['id']}/outline", headers=H(u),
                       json={"request_id": uuid.uuid4().hex}).status_code == 409
    assert client.put(f"/api/v1/studio/stories/{s['id']}/outline", headers=H(u), json={"outline": edited}).status_code == 409
    assert client.post(f"/api/v1/studio/stories/{s['id']}/outline/approve", headers=H(u), json={"version": 2}).status_code == 409


def test_outline_before_the_interview_is_ready_is_refused(st):
    client, db, store, ai, u = st
    s = start(client, u)
    r = client.post(f"/api/v1/studio/stories/{s['id']}/outline", headers=H(u), json={"request_id": uuid.uuid4().hex})
    assert r.status_code == 409


def test_seeding_twice_adds_nothing_new(st):
    client, db, store, ai, u = st
    s = _ready_story(client, u, ai)
    repo = CanonRepo(store, s["id"], str(u))
    first = nv.seed_canon(repo, nv.clean_outline(outline()), 1)
    again = nv.seed_canon(repo, nv.clean_outline(outline()), 2)
    assert first["entities"] == 6 and again["entities"] == 0 and again["facts"] == 0


# ------------------------------------------------------------------------------------------- pure helpers

def test_said_ignores_case_and_punctuation_and_works_in_any_language():
    text = "Umukobwa w'umudozi i Kigali. Elle veut la vérité!"
    assert nv.said("umukobwa w'umudozi", text)
    assert nv.said("elle veut la VÉRITÉ", text)
    assert not nv.said("la vérité absolue", text)
    assert not nv.said("", text)


def test_outline_problems_name_each_structural_fault():
    story = {"length": "short", "interview": {"known": {"ending": {"value": "x", "creator_words": "x"}}}}
    o = nv.clean_outline(outline(kept_from_creator=[], threads=[{"kind": "setup", "description": "gun on the wall", "planted_chapter": 8,
                                           "payoff_chapter": 3}],
                                 chapters=[{"number": i, "title": "", "summary": "s", "beats": ["a", "b"],
                                            "pov": "Ghost Writer"} for i in range(1, 13)]))
    p = nv.outline_problems(o, story)
    assert any("must pay off after chapter 8" in x for x in p)
    assert any("told by \"Ghost Writer\"" in x for x in p)
    assert any("keeps what the creator decided about \"ending\"" in x for x in p)
    assert nv.blocking(p) == []
