"""Action intelligence: genres, detection, sequences, the 8-shot limit, spatial continuity, feasibility flags,
duplicate shots, and the per-shot picture prompt. Pure checks first (no database), then the real pipeline with a
fake AI: the prompts each shot's picture gets, the quote, the budget, refunds, the storyboard gate and the API."""
import os
import uuid

import pytest

import studio_action as act
import studio_shots as ss
import studio_story

CAST = ["Maya", "Daniel"]


def scene(visual, secs=12, lines=(), sid="s1", pos=0, **c):
    return {"id": sid, "position": pos, "heading": "INT. WAREHOUSE - NIGHT", "visual": visual, "duration_seconds": secs,
            "dialogue": [{"character": a, "text": t} for a, t in lines],
            "continuity": {"characters": [{"name": n, "in_frame": True} for n in CAST], **c}}


FIGHT = scene("Maya disarms the first guard and throws him into the crates")
DRAMA = scene("Maya and Daniel sit at the kitchen table, not looking at each other", 8,
              [("Maya", "You lied to me."), ("Daniel", "To keep you safe.")])


# ------------------------------------------------------------------ genres and detection

def test_the_seven_action_genres_join_the_writers_room_and_old_genres_stay():
    for g, key in [("Action", "action"), ("Adventure", "adventure"), ("Superhero", "superhero"),
                   ("Martial arts", "martial_arts"), ("Spy thriller", "spy_thriller"),
                   ("Survival thriller", "survival_thriller"), ("Science fiction", "science_fiction")]:
        craft = studio_story.genre_craft(g)
        assert key in craft.split("\n")[0], (g, craft[:60])
        assert studio_story.GENRES[key] in craft
    assert studio_story.genre_craft("Thriller").startswith("GENRE CRAFT (thriller)")
    assert studio_story.genre_craft("Romance").startswith("GENRE CRAFT (romance)")
    assert studio_story.genre_craft("").startswith("GENRE CRAFT (drama)")
    assert "Never action for its own sake" in studio_story.GENRES["action"]


@pytest.mark.parametrize("visual,expected", [
    ("Maya disarms the first guard and throws him into the crates", True),
    ("Chase: they sprint between market stalls, the car close behind", True),
    ("Daniel slips and falls from the roof edge, Maya grabs his wrist", True),
    ("Maya runs down the empty road", False),                 # one weak word: still a drama moment
    ("Maya and Daniel sit at the kitchen table", False),
    ("Maya reads the letter by the window", False),
])
def test_action_scenes_are_detected_and_quiet_scenes_are_not(visual, expected):
    assert act.is_action(scene(visual)) is expected


# ------------------------------------------------------------------ sequences

def test_consecutive_action_scenes_form_a_sequence_with_beats_and_a_consequence():
    scenes = [scene("They walk into the warehouse quietly", sid="a", pos=0),
              scene("A guard attacks; Maya ducks and strikes", sid="b", pos=1),
              scene("Two more guards attack; Daniel is grabbed", sid="c", pos=2),
              scene("The fight turns: Maya loses her gun and is cornered", sid="d", pos=3),
              scene("Maya kicks the last guard through the window", sid="e", pos=4),
              scene("Maya cleans a cut on Daniel's arm in silence", sid="f", pos=5)]
    seqs = act.sequences(scenes)
    assert len(seqs) == 1 and seqs[0]["scenes"] == ["b", "c", "d", "e"] and seqs[0]["consequence_scene"] == "f"
    assert [seqs[0]["beats"][k] for k in "bcde"] == ["setup", "escalation", "reversal", "climax"]
    assert act.scene_beat(scenes[5], scenes)["beat"] == "consequence"
    assert act.scene_beat(scenes[0], scenes) is None
    one = [scene("A guard attacks Maya", sid="x")]
    assert act.scene_beat(one[0], one)["beat"] == "whole"


def test_the_director_brief_for_action_and_none_for_drama():
    scenes = [FIGHT]
    block = act.planner_block(FIGHT, act.scene_beat(FIGHT, scenes))
    for words in ("ONE-SCENE ACTION", "up to 8", "SPATIAL CONTINUITY", "end_position", "movement_direction",
                  "environment_change", "WHAT VIDEO MODELS DO BADLY", "No action for its own sake"):
        assert words in block, words
    assert act.planner_block(DRAMA, act.scene_beat(DRAMA, [DRAMA])) == ""
    p_action = ss.planner_prompt({"kind": "short"}, FIGHT, 1, 1, None, None, [], limit=8, action=block)
    p_drama = ss.planner_prompt({"kind": "short"}, DRAMA, 1, 1, None, None, [])
    assert "Choose 1 to 8 shots" in p_action and "Choose 1 to 4 shots" in p_drama
    assert "ACTION SCENE" not in p_drama and "its OWN moment" in p_drama


# ------------------------------------------------------------------ the shot limit and the director's choice

def test_action_scenes_may_have_up_to_8_shots_drama_keeps_4_and_the_director_chooses():
    assert act.max_shots(FIGHT) == 8 and act.max_shots(DRAMA) == 4
    assert ss.estimate_shots(FIGHT) == 8 and ss.estimate_shots(DRAMA) <= 4           # the quote's upper bound
    nine = {"shots": [{"type": t, "description": f"Shot {k}: something else happens", "characters": [],
                       "line_indices": [], "frame_prompt": f"Shot {k}", "video_prompt": "x", "reason": "r",
                       "duration_seconds": 1.5} for k, t in enumerate(list(ss.SHOT_TYPES) * 2)][:9]}
    assert len(ss.clean_plan(nine, FIGHT, CAST, act.max_shots(FIGHT))) == 8
    assert len(ss.clean_plan(nine, DRAMA, CAST, act.max_shots(DRAMA))) == 4
    # The (fake) director doesn't always use the maximum: a short, simple action scene gets fewer shots.
    short = scene("A guard attacks Maya", secs=6)
    big = scene("Maya disarms the guard, kicks him into the crates, dodges a swing and tackles the second guard")
    assert len(ss.default_plan(short, CAST)["shots"]) == 3
    assert len(ss.default_plan(big, CAST)["shots"]) == 6
    assert len(ss.default_plan(FIGHT, CAST)["shots"]) < 8


def test_the_kill_switch_restores_the_earlier_behaviour(monkeypatch):
    monkeypatch.setattr(act, "ENABLED", False)
    assert not act.is_action(FIGHT) and act.max_shots(FIGHT) == 4 and ss.estimate_shots(FIGHT) <= 4
    assert act.planner_block(FIGHT, {"beat": "whole", "sequence": 1, "position": 1, "of": 1}) == ""


# ------------------------------------------------------------------ spatial continuity

def shot(kind, who, beat="", move="none", desc=None, secs=2.0, env=""):
    return {"shot_type": kind, "description": desc or f"{kind} of {', '.join(n for n, *_ in who)}",
            "frame_prompt": desc or f"{kind} of {', '.join(n for n, *_ in who)}", "duration_seconds": secs,
            "characters": [{"name": n, "position": p, **({"end_position": e} if e else {})} for n, p, e in who],
            "_action": {k: v for k, v in (("beat", beat), ("movement_direction", move), ("environment_change", env)) if v and v != "none"}}


def test_spatial_continuity_screen_side_ends_meet_starts_and_chase_direction():
    good = [shot("establishing_wide", [("Maya", "left", "left"), ("Daniel", "right", "right")], "setup", "left_to_right"),
            shot("medium", [("Maya", "left", "centre")], "escalation", "left_to_right"),
            shot("medium", [("Maya", "centre", "right")], "climax", "left_to_right"),
            shot("reaction", [("Maya", "right", "")], "consequence")]
    assert act.spatial_review(good) == []
    jump = [shot("medium", [("Maya", "left", "")]), shot("close_up", [("Maya", "right", "")])]
    assert act.spatial_review(jump)[0]["kind"] == "screen_direction" and "jumps from the left to the right" in \
        act.spatial_review(jump)[0]["problem"]
    turn = [shot("medium", [("Maya", "left", "")], move="left_to_right"),
            shot("medium", [("Maya", "left", "")], move="right_to_left")]
    assert any(p["kind"] == "movement_direction" for p in act.spatial_review(turn))


def test_action_review_wants_geography_a_reaction_and_beats_in_order():
    flat = [shot("medium", [("Maya", "left", "")], "climax", desc="Maya punches"),
            shot("medium", [("Maya", "left", "")], "setup", desc="Maya waits"),
            shot("insert", [], "escalation", desc="The gun on the floor")]
    kinds = {p["kind"] for p in act.action_review(FIGHT, flat)}
    assert {"geography", "consequence", "beats"} <= kinds


# ------------------------------------------------------------------ feasibility

@pytest.mark.parametrize("desc,kind", [
    ("Maya and the guards grapple on the floor", "multi_contact"),
    ("The car swerves and crashes into the barrier", "vehicle"),
    ("Daniel picks the lock with shaking fingers", "hands"),
    ("Maya reads the message on the screen", "text"),
    ("The crowd surges toward the gate", "crowd"),
    ("Her body glows with power and she transforms", "transform"),
    ("Maya leaps between the rooftops", "fall"),
    ("The camera does a 360 orbit around them", "camera"),
])
def test_difficult_shots_are_flagged_with_a_simpler_alternative(desc, kind):
    flags = act.feasibility({"description": desc, "duration_seconds": 3})
    hit = next(f for f in flags if f["kind"] == kind)
    assert hit["issue"] and hit["alternative"] and hit["risk"] in ("high", "medium")


def test_long_choreography_and_crowded_action_are_flagged_and_calm_shots_are_not():
    long = act.feasibility({"description": "Maya ducks, kicks the guard, grabs the pipe and strikes the second guard",
                            "duration_seconds": 8})
    assert any(f["kind"] == "long_choreography" for f in long)
    many = act.feasibility({"description": "They fight and run", "duration_seconds": 3,
                            "characters": [{"name": "A"}, {"name": "B"}, {"name": "C"}]})
    assert any(f["kind"] == "many_people" for f in many)
    assert act.feasibility({"description": "Close-up of Maya listening", "duration_seconds": 3}) == []


# ------------------------------------------------------------------ duplicates and each shot's own instructions

def test_the_review_sends_back_two_shots_of_the_same_moment():
    same = [shot("medium", [("Maya", "left", "")], desc="Maya and Daniel face each other in the warehouse"),
            shot("close_up", [("Daniel", "right", "")], desc="Daniel's eyes"),
            shot("medium", [("Maya", "left", "")], desc="Maya and Daniel face each other in the warehouse.")]
    dup = [p for p in ss.plan_quality(DRAMA, same) if p["kind"] == "duplicate"]
    assert dup and "Shots 1 and 3 show the same moment" in dup[0]["problem"]
    different = [same[0], same[1], shot("medium", [("Maya", "left", "")], desc="Maya stands and walks to the door")]
    assert not [p for p in ss.plan_quality(DRAMA, different) if p["kind"] == "duplicate"]


def test_each_shot_is_told_its_own_moment_with_the_scene_as_labelled_context():
    plan = ss.clean_plan(ss.default_plan(FIGHT, CAST), FIGHT, CAST, 8)
    moments = [ss.moment(FIGHT, plan, i) for i in range(len(plan))]
    context = ss.scene_context(FIGHT)
    assert "SCENE CONTEXT" in context and FIGHT["visual"] in context and "ONE moment" in context
    for i, (m, sh) in enumerate(zip(moments, plan)):
        assert m.startswith(f"THIS SHOT (shot {i + 1} of {len(plan)}")
        assert sh["frame_prompt"].rstrip(".") in m, "the shot's own description is kept, not replaced by the scene"
        assert "Who is where" in m or sh["shot_type"] == "insert"
    pairs = [ss.similarity(a, b) for i, a in enumerate(moments) for b in moments[i + 1:]]
    assert len(set(moments)) == len(moments) and max(pairs) < 0.8, "every shot of the scene gets different instructions"
    assert "Movement on screen: left to right." in moments[1] and "ends centre" in moments[1]
    assert "What changes in the place: dust and debris" in next(m for m in moments if "climax" in m)


# ------------------------------------------------------------------ the real pipeline (fake AI, local Postgres)

pytestmark_db = pytest.mark.skipif(bool(os.environ.get("STUDIO_PREMIGRATION")), reason="needs the Studio schema")
from test_creator_studio import H, balance, env, grant, run, studio_schema  # noqa: E402,F401
from test_video_drama import DAN, MAYA, project, split  # noqa: E402

ACTION_SCENES = [
    {"heading": "INT. WAREHOUSE - NIGHT", "visual": "Maya creeps between crates; a guard attacks and she disarms him",
     "duration_seconds": 12, "dialogue": [{"character": "Maya", "text": "Stay behind me."}],
     "continuity": {"transition": "new_place", "purpose": "They must reach the files before the alarm",
                    "characters": [{"name": "Maya", "in_frame": True}, {"name": "Daniel", "in_frame": True}]}},
    {"heading": "INT. WAREHOUSE - NIGHT", "visual": "Two more guards attack; Maya kicks one into the crates",
     "duration_seconds": 12, "dialogue": [],
     "continuity": {"transition": "continuous",
                    "characters": [{"name": "Maya", "in_frame": True}, {"name": "Daniel", "in_frame": True}]}},
    {"heading": "INT. WAREHOUSE - NIGHT", "visual": "Maya and Daniel sit against the wall, breathing hard",
     "duration_seconds": 6, "dialogue": [{"character": "Daniel", "text": "You've done this before."}],
     "continuity": {"transition": "continuous",
                    "characters": [{"name": "Maya", "in_frame": True}, {"name": "Daniel", "in_frame": True}]}},
]


def produce(client, gen, ren, u, pid, storyboard=False, rounds=40):
    q = client.post(f"/api/v1/studio/projects/{pid}/production/quote", headers=H(u), json={}).json()
    r = client.post(f"/api/v1/studio/projects/{pid}/production", headers=H(u),
                    json={"request_id": uuid.uuid4().hex, "confirm_credits": q["total_credits"],
                          "storyboard_review": storyboard})
    assert r.status_code == 200, r.text
    for _ in range(rounds):
        run(gen)
        run(ren)
        s = client.get(f"/api/v1/studio/projects/{pid}/production", headers=H(u)).json()["production"]
        if s["status"] in ("done", "needs_attention", "needs_approval", "storyboard_review", "canceled"):
            break
    return q, s


def setup_film(env, scenes=ACTION_SCENES, credits=300000):
    client, db, store, files, ai, gen, ren = env
    ai.varied_pictures = True
    u = make_user(db)
    grant(client, u, credits)
    pid = project(client, u, production_style="video", genre="Action")["id"]
    split(client, ai, gen, u, pid, [dict(s) for s in scenes], characters=(MAYA, DAN))
    return client, db, store, files, ai, gen, ren, u, pid


from conftest import make_user  # noqa: E402


@pytestmark_db
def test_action_scenes_get_more_shots_priced_in_the_quote_and_kept_inside_the_budget(env, monkeypatch):
    client, db, store, files, ai, gen, ren, u, pid = setup_film(env)
    q_new = client.post(f"/api/v1/studio/projects/{pid}/production/quote", headers=H(u), json={}).json()
    monkeypatch.setattr(act, "ENABLED", False)
    q_old = client.post(f"/api/v1/studio/projects/{pid}/production/quote", headers=H(u), json={}).json()
    monkeypatch.setattr(act, "ENABLED", True)
    label = lambda q: [i["label"] for st in q["stages"] for i in st["items"] if i.get("reserve_only")]  # noqa: E731
    assert "Scene 1: up to 8 shot pictures" in label(q_new) and "Scene 3: up to 4 shot pictures" in label(q_new)
    assert not any("up to 8" in x for x in label(q_old))
    assert q_new["total_credits"] > q_old["total_credits"], "the extra shots are in the price before payment"
    before = balance(client, u)["available"]
    q, s = produce(client, gen, ren, u, pid)
    assert s["status"] == "done" and 0 < s["spent_credits"] <= s["approved_credits"]
    assert balance(client, u)["held"] == 0 and balance(client, u)["available"] == before - s["spent_credits"]
    rows = db.execute("select c.position, count(s.id) from public.studio_scenes c join public.studio_shots s on "
                      "s.scene_id = c.id where c.project_id = %s group by c.position order by c.position", (pid,)).fetchall()
    counts = [n for _p, n in rows]
    assert counts[0] > 4 and counts[1] > 4 and counts[2] <= 4, counts


@pytestmark_db
def test_every_shot_picture_gets_its_own_instructions_with_references(env):
    client, db, store, files, ai, gen, ren, u, pid = setup_film(env)
    produce(client, gen, ren, u, pid)
    shot_prompts = [x for x in ai.image_log if "THIS SHOT (shot" in x["prompt"]]
    n_shots = db.execute("select count(*) from public.studio_shots where project_id = %s", (pid,)).fetchone()[0]
    assert len(shot_prompts) >= n_shots
    prompts = [x["prompt"] for x in shot_prompts]
    assert len(set(prompts)) == len(prompts), "no two shot pictures get the same prompt"
    own = [p.split("SCENE: ", 1)[1].split("\nCAMERA:", 1)[0] for p in prompts]
    worst = max(ss.similarity(a, b) for i, a in enumerate(own) for b in own[i + 1:])
    assert worst < 0.9, worst
    for p in prompts:
        assert "SCENE CONTEXT" in p and "CHARACTERS (permanent identity" in p and "CAMERA:" in p
        assert "SHOT CONTINUITY" in p and "STYLE AND LIGHT" in p
    assert all(x["refs"] > 0 for x in shot_prompts if "Maya" in x["prompt"]), "Character DNA references still sent"
    shots = client.get(f"/api/v1/studio/projects/{pid}", headers=H(u)).json()["scenes"][0]["shots"]
    assert shots[0]["action"]["beat"] == "setup" and "feasibility" in shots[0]["action"]
    assert client.get(f"/api/v1/studio/projects/{pid}", headers=H(u)).json()["scenes"][0]["action"]["beat"] == "setup"


@pytestmark_db
def test_drama_productions_are_planned_and_made_as_before(env):
    client, db, store, files, ai, gen, ren, u, pid = setup_film(env, scenes=[ACTION_SCENES[2]])
    q, s = produce(client, gen, ren, u, pid)
    assert s["status"] == "done"
    assert db.execute("select count(*) from public.studio_shots where project_id = %s", (pid,)).fetchone()[0] <= 4
    made = db.execute("select made_with from public.studio_shots where project_id = %s", (pid,)).fetchall()
    assert all("_action" not in (m or {}) for (m,) in made), "no action data on a drama scene"


@pytestmark_db
def test_the_storyboard_gate_still_stops_an_action_production_before_video(env):
    client, db, store, files, ai, gen, ren, u, pid = setup_film(env)
    q, s = produce(client, gen, ren, u, pid, storyboard=True)
    assert s["status"] == "storyboard_review"
    assert db.execute("select count(*) from public.studio_jobs where project_id = %s and kind = 'shot_video'",
                      (pid,)).fetchone()[0] == 0


@pytestmark_db
def test_a_failed_step_in_an_action_production_is_refunded_and_retried(env):
    import studio_ai
    client, db, store, files, ai, gen, ren, u, pid = setup_film(env)
    ai.fail["image"] = [studio_ai.ProviderError("boom")]
    q, s = produce(client, gen, ren, u, pid)
    failed = db.execute("select count(*), coalesce(sum(charged_credits), 0) from public.studio_jobs where project_id = %s "
                        "and status = 'failed'", (pid,)).fetchone()
    assert s["status"] == "done" and failed[0] >= 1 and int(failed[1]) == 0
