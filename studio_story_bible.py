"""Story Intelligence stage 6: the approved novel becomes the production's Story Bible.

When the creator approves the whole novel, its story memory (studio_canon) and approved outline are compiled, by
code and without any AI call, into the Story Bible the rest of the Studio already reads:
  story_bible.saga       the larger story every episode is written inside (studio_saga's format): premise, main
                         storyline, characters (want, fear, what they hide, inner conflict, arc, look), relationships,
                         secrets (who keeps them, who knows by the end, when they come out), mysteries (the question,
                         its answer, its clues), turning points, and the planned number of episodes;
  story_bible.knowledge  who knows which secret, and from which chapter;
  story_bible.novel      the novel it came from: its chapters' titles and summaries.
Everything comes from APPROVED or creator-locked memory as it stands at the end of the novel (so a character's age
or a relationship that changed is taken as it ends), falling back to the approved outline only where the memory
says nothing. Proposed (unreviewed) facts never reach the bible.

Characters, places and objects then become the production's cast and elements (studio_characters, studio_elements),
linked back to their memory entity, so their look, voice and identity stay the same in every episode. An existing
character or element with the same name is linked, never duplicated or overwritten.

Episode numbers are filled in by stage 7 (the episode planner); until then reveals are placed by chapter.
"""
from __future__ import annotations

import datetime
from typing import Optional

from studio_canon import CANON, END, Canon

# The last moment of the story (END itself is exclusive: a fact with no end is true *before* END, not at it).
LAST = (END[0] - 1, 0)

TRAIT_MAP = {"role": "role", "age": "age", "history": "wound", "wants": "want", "fears_losing": "need",
             "hides": "hides", "weakness": "flaw", "inner_conflict": "flaw", "arc": "arc", "look": "appearance"}


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _t(x, n: int = 500) -> str:
    return " ".join(str(x or "").split())[:n]


def _outline_character(outline: dict, name: str) -> dict:
    low = name.lower()
    return next((c for c in outline.get("characters") or [] if (c.get("name") or "").lower() == low), {})


def character_profile(canon: Canon, e: dict, outline: dict) -> dict:
    """A character as the novel ends: memory traits first, the outline where the memory is silent."""
    traits = canon.state_at(LAST).traits_of(e["id"])
    plan = _outline_character(outline, e["name"])
    out = {"name": e["name"]}
    for saga_key, trait in TRAIT_MAP.items():
        v = traits.get(trait) or plan.get(trait) or (plan.get("secret") if trait == "hides" else "")
        if v:
            out[saga_key] = _t(v)
    for k in ("voice", "personality"):
        v = traits.get(k) or plan.get(k)
        if v:
            out[k] = _t(v)
    return out


def compile_bible(story: dict, chapters: dict, canon: Canon) -> dict:
    """{"saga", "knowledge", "novel"} for the production's story_bible (free: no AI)."""
    outline = story.get("outline") or {}
    names = canon.names
    live = [e for e in canon.entities if e.get("status") in CANON]
    chars = [e for e in live if e["kind"] == "character"]
    end = canon.state_at(LAST)
    approved_chapters = [chapters[k] for k in sorted(chapters) if chapters[k].get("status") == "approved"]

    relationships = []
    for f in canon.facts:
        if f.get("status") in CANON and f["kind"] == "relationship" and f.get("valid_until_chapter") is None:
            relationships.append({"between": f"{names.get(f.get('subject_id'), '?')} and {names.get(f.get('object_id'), '?')}",
                                  "kind": _t(f.get("attribute") or "bond", 80), "tension": _t(f.get("value"), 300)})

    knowledge, known_by = [], {}
    for f in canon.facts:
        if f.get("status") in CANON and f["kind"] == "knows" and (f.get("value") or "").lower() not in ("no", "false", "forgot", "forgets") \
                and f.get("valid_until_chapter") is None:
            who, what = names.get(f.get("subject_id"), ""), names.get(f.get("object_id"), "")
            if who and what:
                knowledge.append({"character": who, "knows": what, "since_chapter": int(f.get("valid_from_chapter") or 0)})
                known_by.setdefault(what.lower(), []).append(who)

    secrets = []
    for s in outline.get("secrets") or []:
        secrets.append({"secret": _t(s.get("truth") or s.get("name")), "name": _t(s.get("name"), 120),
                        "kept_by": _t(", ".join(s.get("known_by") or []), 300),
                        "known_by": _t(", ".join(sorted(set(known_by.get((s.get("name") or "").lower(), [])))), 300),
                        "reveal_chapter": int(s.get("revealed_in_chapter") or 0), "reveal_episode": 0})

    mysteries, open_threads = [], []
    for t in canon.threads:
        if t.get("status") not in CANON:
            continue
        item = {"question": _t(t["description"]), "clues": _t(t.get("planted_quote"), 300),
                "planted_chapter": int(t.get("planted_chapter") or 0), "reveal_episode": 0}
        if t.get("state") == "resolved":
            item.update(answer=_t(t.get("resolved_quote") or t.get("payoff_plan"), 400),
                        reveal_chapter=int(t.get("resolved_chapter") or 0))
            mysteries.append(item)
        elif t.get("state") == "open":
            item.update(answer=_t(t.get("payoff_plan"), 400))
            open_threads.append(item)

    turning = []
    by_chapter: dict = {}
    for e in sorted((e for e in canon.events if e.get("status") in CANON), key=lambda e: (int(e["chapter"]), int(e.get("seq") or 0))):
        by_chapter.setdefault(int(e["chapter"]), e)
    for n, e in sorted(by_chapter.items()):
        turning.append({"chapter": n, "episode": 0, "what": _t(e["summary"])})

    saga = {
        "premise": _t(outline.get("premise"), 600),
        "main_storyline": _t(" ".join(c.get("summary") or "" for c in approved_chapters), 1200),
        "theme": _t(", ".join(outline.get("themes") or []), 600),
        "storylines": [],
        "characters": [character_profile(canon, e, outline) for e in chars][:30],
        "relationships": relationships[:30],
        "secrets": secrets[:30],
        "mysteries": mysteries[:30],
        "open_threads": open_threads[:30],
        "rivalries": [],
        "turning_points": turning[:60],
        "seasons": [],
        "future_seasons": [],
        "open_questions": [t["question"] for t in open_threads][:10],
        "ending": _t((outline.get("ending") or {}).get("description"), 600),
        "length": {"mode": "episodes" if int(story.get("episodes") or 0) else "open",
                   "count": int(story.get("episodes") or 0)},
        "built_at": _now(), "approved": True, "revealed": [],
        "source": {"kind": "novel", "story_id": story["id"], "chapters": len(approved_chapters)},
    }
    novel = {"story_id": story["id"], "title": story.get("title") or outline.get("title") or "",
             "chapters": [{"number": c["number"], "title": c.get("title") or "", "summary": _t(c.get("summary"), 600)}
                          for c in approved_chapters],
             "places": [e["name"] for e in live if e["kind"] == "place"],
             "objects": [e["name"] for e in live if e["kind"] == "object"]}
    return {"saga": saga, "knowledge": knowledge[:200], "novel": novel, "_end_alive": [
        names.get(e["id"]) for e in chars if end.status_of(e["id"]) not in ("dead",)]}


def merge_into(story_bible: Optional[dict], compiled: dict) -> dict:
    """The production's story_bible with the novel's saga, knowledge and source; everything else is kept."""
    out = dict(story_bible or {})
    out.pop("_assistant_approval", None)          # the screenplay basis changed: an old approval no longer applies
    out["saga"] = compiled["saga"]
    out["knowledge"] = compiled["knowledge"]
    out["novel"] = compiled["novel"]
    return out


def character_description(profile: dict) -> str:
    """The cast card's description: what an image or voice model needs (look first), within the card's limit."""
    parts = [profile.get("look"), f"Age: {profile['age']}" if profile.get("age") else "",
             f"Role: {profile['role']}" if profile.get("role") else "", profile.get("personality"),
             f"Voice: {profile['voice']}" if profile.get("voice") else ""]
    return _t(". ".join(p for p in parts if p), 1200)


def element_rows(canon: Canon, outline: dict) -> list[dict]:
    """Places and objects that should exist as production elements (location / prop), with descriptions."""
    plans = {(p.get("name") or "").lower(): p for p in (outline.get("places") or []) + (outline.get("objects") or [])}
    out = []
    for e in canon.entities:
        if e.get("status") not in CANON or e["kind"] not in ("place", "object"):
            continue
        profile = e.get("profile") or {}
        plan = plans.get(e["name"].lower(), {})
        desc = profile.get("description") or plan.get("description") or ""
        if e["kind"] == "object" and (profile.get("why_it_matters") or plan.get("why_it_matters")):
            desc = f"{desc} ({profile.get('why_it_matters') or plan.get('why_it_matters')})".strip()
        out.append({"entity_id": e["id"], "element_id": e.get("element_id"),
                    "kind": "location" if e["kind"] == "place" else "prop",
                    "name": e["name"][:80], "description": _t(desc, 800)})
    return out


def browse(canon: Canon, story: dict) -> dict:
    """The Story Bible browser: characters with their traits, secrets and who knows them, places, objects, open and
    resolved threads, and the timeline by chapter. Live from the memory (approved and locked only)."""
    outline = story.get("outline") or {}
    names = canon.names
    end = canon.state_at(LAST)
    live = [e for e in canon.entities if e.get("status") in CANON]
    knows: dict = {}
    for f in canon.facts:
        if f.get("status") in CANON and f["kind"] == "knows" and f.get("valid_until_chapter") is None \
                and (f.get("value") or "").lower() not in ("no", "false", "forgot", "forgets"):
            knows.setdefault(f.get("object_id"), []).append({"who": names.get(f.get("subject_id"), ""),
                                                             "since_chapter": int(f.get("valid_from_chapter") or 0)})
    locked = {f.get("subject_id") for f in canon.facts if f.get("status") == "creator_locked"}
    return {
        "characters": [{**character_profile(canon, e, outline), "id": e["id"], "status": end.status_of(e["id"]) or "alive",
                        "where": names.get(end.location_of(e["id"]) or "", "") if end.location_of(e["id"]) else "",
                        "cast_id": e.get("character_id"), "has_locked_facts": e["id"] in locked,
                        "relationships": [{"with": names.get(r.get("object_id"), ""), "kind": r.get("attribute") or "",
                                           "what": r.get("value") or ""} for r in end.relationships_of(e["id"])]}
                       for e in live if e["kind"] == "character"],
        "secrets": [{"id": e["id"], "name": e["name"], "known_by": knows.get(e["id"], [])}
                    for e in live if e["kind"] == "secret"],
        "places": [{"id": e["id"], "name": e["name"], "description": (e.get("profile") or {}).get("description") or "",
                    "element_id": e.get("element_id")} for e in live if e["kind"] == "place"],
        "objects": [{"id": e["id"], "name": e["name"], "description": (e.get("profile") or {}).get("description") or "",
                     "where": names.get(end.location_of(e["id"]) or "", "") if end.location_of(e["id"]) else "",
                     "element_id": e.get("element_id")} for e in live if e["kind"] == "object"],
        "threads": [{"id": t["id"], "kind": t["kind"], "description": t["description"], "state": t.get("state"),
                     "planted_chapter": t.get("planted_chapter"), "resolved_chapter": t.get("resolved_chapter")}
                    for t in canon.threads if t.get("status") in CANON],
        "timeline": [{"chapter": int(e["chapter"]), "summary": e["summary"], "story_time": e.get("story_time") or "",
                      "participants": [names.get(p, "") for p in e.get("participants") or []]}
                     for e in sorted((e for e in canon.events if e.get("status") in CANON),
                                     key=lambda e: (int(e["chapter"]), int(e.get("seq") or 0)))],
    }
