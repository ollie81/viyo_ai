"""Baseline vs action-aware shot planning, on the same action film, in one run (fake AI, real ffmpeg render).

The same 8-scene film (a warehouse fight, a chase across the docks and rooftops, the aftermath) is produced twice:
with action intelligence switched off (the earlier behaviour, including the old shot picture prompts) and on.
Both runs are measured the same way:

  shot variety          distinct shot sizes, the largest share of one size, shots per action scene
  visual distinction    duplicate pairs found by the whole-film report's frame fingerprints
  movement continuity   shots that carry spatial data, and spatial problems found in the plans
  character consistency shot pictures sent with the characters' reference pictures
  scene coverage        action scenes with a wide AND a reaction/close shot, beats covered in order
  prompt duplication    identical shot prompts, and the most similar pair of shot-specific instructions
  estimated cost        the quote's maximum and what was actually charged

What this proves: the pipeline gives the director more room only where the scene is action, prices it before
payment, keeps it inside the budget, and gives every shot its own instructions. What it does NOT prove: that the
real director plans better or that real models draw better films. The fake director follows the brief by
construction, and fake pictures are drawn from the prompt text. Those need the live benchmark.
"""
import json
import os
import uuid

import pytest

import studio_action as act
import studio_shots as ss
from conftest import make_user
from test_creator_studio import H, balance, env, grant, run, studio_schema  # noqa: F401
from test_video_drama import DAN, MAYA, project, split

pytestmark = pytest.mark.skipif(bool(os.environ.get("STUDIO_PREMIGRATION")), reason="needs the Studio schema")


def sc(heading, visual, secs, lines=(), transition="cut", purpose=""):
    return {"heading": heading, "visual": visual, "duration_seconds": secs,
            "dialogue": [{"character": c, "text": t} for c, t in lines],
            "continuity": {"transition": transition, "purpose": purpose,
                           "characters": [{"name": "Maya", "in_frame": True}, {"name": "Daniel", "in_frame": True}]}}


FILM = [
    sc("INT. WAREHOUSE - NIGHT", "Maya and Daniel slip in through a side door, listening", 6,
       [("Maya", "Stay behind me.")], "new_place"),
    sc("INT. WAREHOUSE - NIGHT", "A guard attacks; Maya ducks and strikes him", 12,
       purpose="They must reach the files before the alarm"),
    sc("INT. WAREHOUSE - NIGHT", "Two more guards attack; Maya kicks one into the crates", 12),
    sc("INT. WAREHOUSE - NIGHT", "The fight turns: Maya loses her gun and is cornered by the last guard", 12),
    sc("EXT. DOCKS - NIGHT", "Chase: they sprint along the pier, a car close behind them", 12, transition="new_place"),
    sc("EXT. ROOFTOPS - NIGHT", "Maya leaps a gap between roofs; Daniel jumps after her and she grabs his wrist", 12,
       [("Daniel", "Don't let go.")], "new_place"),
    sc("EXT. ROOFTOPS - NIGHT", "They lie on the roof as sirens pass below", 6, [("Maya", "We're alive.")]),
    sc("INT. SAFE HOUSE - NIGHT", "Maya cleans a cut on Daniel's arm in silence", 8,
       [("Daniel", "You've done this before."), ("Maya", "Too many times.")], "time_jump"),
]


def produce(env, enabled, monkeypatch):
    client, db, store, files, ai, gen, ren = env
    monkeypatch.setattr(act, "ENABLED", enabled)
    ai.varied_pictures = True
    start = len(getattr(ai, "image_log", []))
    u = make_user(db)
    grant(client, u, 500000)
    pid = project(client, u, production_style="video", genre="Action", target_seconds=90)["id"]
    split(client, ai, gen, u, pid, [dict(s) for s in FILM], characters=(MAYA, DAN))
    q = client.post(f"/api/v1/studio/projects/{pid}/production/quote", headers=H(u), json={}).json()
    before = balance(client, u)["available"]
    r = client.post(f"/api/v1/studio/projects/{pid}/production", headers=H(u),
                    json={"request_id": uuid.uuid4().hex, "confirm_credits": q["total_credits"], "storyboard_review": False})
    assert r.status_code == 200, r.text
    for _ in range(60):
        run(gen)
        run(ren)
        s = client.get(f"/api/v1/studio/projects/{pid}/production", headers=H(u)).json()["production"]
        if s["status"] in ("done", "needs_attention", "needs_approval", "canceled"):
            break
    assert s["status"] == "done", s
    assert 0 < s["spent_credits"] <= s["approved_credits"] and balance(client, u)["held"] == 0
    assert balance(client, u)["available"] == before - s["spent_credits"]
    rep = client.get(f"/api/v1/studio/projects/{pid}/film-report", headers=H(u)).json()
    monkeypatch.setattr(act, "ENABLED", True)          # measuring always uses the same yardstick
    return measure(db, ai, pid, q, s, rep, ai.image_log[start:])


def measure(db, ai, pid, q, s, rep, images):
    scenes = db.execute("select id, position, visual, heading, duration_seconds, continuity from public.studio_scenes "
                        "where project_id = %s order by position", (pid,)).fetchall()
    action_scenes, rows = [], {}
    for sid, pos, visual, heading, dur, cont in scenes:
        shots = db.execute("select shot_type, characters, description, frame_prompt, made_with, duration_seconds "
                           "from public.studio_shots where scene_id = %s order by position", (sid,)).fetchall()
        rows[pos] = [{"shot_type": t, "characters": c or [], "description": d, "frame_prompt": f,
                      "duration_seconds": float(sec), "_action": (m or {}).get("_action") or {}} for t, c, d, f, m, sec in shots]
        if act.action_score({"visual": visual, "heading": heading, "continuity": cont}) >= 2:
            action_scenes.append(pos)
    all_shots = [x for v in rows.values() for x in v]
    covered = [p for p in action_scenes if {"establishing_wide", "wide_two_shot"} & {x["shot_type"] for x in rows[p]}
               and {"reaction", "close_up", "extreme_close_up"} & {x["shot_type"] for x in rows[p]}]
    beat_order = [p for p in action_scenes
                  if [x["_action"].get("beat") for x in rows[p] if x["_action"].get("beat")] and
                  [act.BEATS.index(b) for b in [x["_action"].get("beat") for x in rows[p]] if b in act.BEATS] ==
                  sorted(act.BEATS.index(b) for b in [x["_action"].get("beat") for x in rows[p]] if b in act.BEATS)]
    spatial = [x for x in all_shots if any(c.get("end_position") for c in x["characters"]) or
               x["_action"].get("movement_direction")]
    spatial_problems = sum(len(act.spatial_review(v)) for v in rows.values())
    flagged = sum(len(x["_action"].get("feasibility") or []) for x in all_shots)
    shot_imgs = [x for x in images if "SHOT CONTINUITY" in x["prompt"]]
    prompts = [x["prompt"] for x in shot_imgs]
    own = [p.split("SCENE: ", 1)[1].split("\nCAMERA:", 1)[0] if "SCENE: " in p else p for p in prompts]
    sims = [ss.similarity(a, b) for i, a in enumerate(own) for b in own[i + 1:]]
    m = rep["metrics"]
    return {
        "shot_variety": {"shots": len(all_shots), "distinct_types": m["distinct_shot_types"],
                         "largest_type_share": m["largest_type_share"],
                         "shots_per_action_scene": round(sum(len(rows[p]) for p in action_scenes) / len(action_scenes), 2),
                         "max_shots_in_a_scene": max(len(v) for v in rows.values())},
        "visual_distinction": {"duplicate_pairs": m["duplicate_pairs"], "fingerprinted": m["fingerprinted"]},
        "movement_continuity": {"shots_with_spatial_data": len(spatial), "share": round(len(spatial) / len(all_shots), 2),
                                "spatial_problems": spatial_problems, "feasibility_flags": flagged},
        "character_consistency": {"shot_pictures": len(shot_imgs),
                                  "with_references": sum(1 for x in shot_imgs if x["refs"] > 0)},
        "scene_coverage": {"action_scenes": len(action_scenes), "wide_and_reaction": len(covered),
                           "beats_in_order": len(beat_order)},
        "prompt_duplication": {"identical_prompts": len(prompts) - len(set(prompts)),
                               "near_identical_pairs": sum(1 for x in sims if x >= 0.9),
                               "most_similar_pair": round(max(sims), 2) if sims else None},
        "cost": {"quoted_max": q["total_credits"], "spent": s["spent_credits"]},
        "report": {"status": rep["status"], "problems": {c["key"]: c["count"] for c in rep["categories"]}},
    }


def test_baseline_versus_action_aware_planning(env, tmp_path, monkeypatch):
    base = produce(env, False, monkeypatch)
    new = produce(env, True, monkeypatch)
    card = {"benchmark": "action_planning_ab", "ai": "fake (no paid calls)", "render": "real ffmpeg",
            "baseline": base, "action_aware": new,
            "cost_change": {"quoted_max": new["cost"]["quoted_max"] - base["cost"]["quoted_max"],
                            "spent": new["cost"]["spent"] - base["cost"]["spent"],
                            "spent_ratio": round(new["cost"]["spent"] / base["cost"]["spent"], 2)},
            "not_measured": ["real director plan quality", "real model drawing quality", "whether viewers find "
                             "the action clearer"]}
    folder = os.environ.get("VIYO_BENCHMARK_DIR") or str(tmp_path)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "action_planning_ab.json"), "w") as f:
        json.dump(card, f, indent=2)
    print("\nBENCHMARK action_planning_ab: " + json.dumps(card))
    # More coverage only where it's action, every shot its own instructions, inside the budget.
    assert new["shot_variety"]["shots_per_action_scene"] > base["shot_variety"]["shots_per_action_scene"]
    assert new["shot_variety"]["max_shots_in_a_scene"] <= act.MAX_ACTION_SHOTS
    assert new["shot_variety"]["distinct_types"] >= base["shot_variety"]["distinct_types"]
    assert new["scene_coverage"]["wide_and_reaction"] >= base["scene_coverage"]["wide_and_reaction"]
    assert new["scene_coverage"]["beats_in_order"] == new["scene_coverage"]["action_scenes"]
    assert new["movement_continuity"]["share"] > base["movement_continuity"]["share"]
    assert new["movement_continuity"]["spatial_problems"] == 0
    assert new["prompt_duplication"]["identical_prompts"] == 0 and new["prompt_duplication"]["near_identical_pairs"] == 0
    assert new["visual_distinction"]["duplicate_pairs"] <= base["visual_distinction"]["duplicate_pairs"]
    assert new["visual_distinction"]["duplicate_pairs"] == 0
    cc = new["character_consistency"]          # an insert of an object may carry only the place or shot before
    assert cc["with_references"] >= 0.75 * cc["shot_pictures"], cc
    assert new["cost"]["quoted_max"] > base["cost"]["quoted_max"], "the extra shots are priced before payment"
