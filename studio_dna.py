"""VIYO Character DNA: one record per character, and a check that a cast doesn't look alike.

Distinct casts (free: no AI call)
  Two characters are compared on what makes people look different on screen: presented gender, age group, skin tone,
  hair, face shape, eyes, build, facial hair, marks and the clothes they wear most. Matching words in matching
  fields add up to a similarity score. Pairs that look alike are flagged with which features match and what to change
  first, inside the drama and against the creator's other dramas (so a creator's productions don't keep producing the
  same generic face). Family is expected to share features: a pair the story makes related is still shown, softened.
  A returning character (the same identity in a series) is the same person on purpose and is never flagged.
  The reference pictures are compared too: two characters whose anchor pictures have (nearly) the same frame
  fingerprint are flagged, whatever their text says.
  New identities are told who is already in the cast so the AI makes the new person clearly different.

Character DNA
  Everything that defines a character, gathered from where it already lives: the identity (face, body, basics), the
  reference views and their approval, outfits, the voice, personality, the story bible's history, wants, fears and
  arc, relationships, the latest emotional state, where they appear, and how their pictures fared in VIYO's checks.

Honest limits
  The look-alike check reads the identity TEXT (and exact picture fingerprints), not faces: two different
  descriptions can still be drawn alike by an image model, and a near-identical description can be drawn differently.
  It is advice for the creator, never a block.
"""
from __future__ import annotations

import re
from typing import Optional

LOOK_ALIKE = 0.6           # in one drama
ACROSS_DRAMAS = 0.75       # against the creator's other dramas: only clear repeats
FIELDS = [  # (label, how to read it, weight)
    ("gender", lambda c, i: c.get("gender_presentation") or "", 1.0),
    ("age", lambda c, i: c.get("age_category") or str((i.get("basic") or {}).get("age") or ""), 1.5),
    ("skin tone", lambda c, i: (i.get("basic") or {}).get("skin_tone") or "", 2.0),
    ("hair", lambda c, i: (i.get("face") or {}).get("hair") or "", 2.5),
    ("face shape", lambda c, i: (i.get("face") or {}).get("face_shape") or "", 1.0),
    ("eyes", lambda c, i: (i.get("face") or {}).get("eyes") or "", 1.0),
    ("build", lambda c, i: (i.get("basic") or {}).get("build") or (i.get("body") or {}).get("physique") or "", 1.5),
    ("facial hair", lambda c, i: (i.get("face") or {}).get("facial_hair") or "", 1.0),
    ("marks", lambda c, i: (i.get("face") or {}).get("marks") or (i.get("basic") or {}).get("distinguishing") or "", 1.0),
    ("clothes", lambda c, i: i.get("default_outfit") or "", 1.5),
]
CHANGE_FIRST = {"hair": "a different hair colour, length or style", "skin tone": "a different skin tone",
                "build": "a different build or height", "clothes": "a different signature outfit or colour",
                "face shape": "a different face shape", "facial hair": "different facial hair",
                "marks": "a distinguishing mark (a scar, freckles, glasses)", "eyes": "different eyes",
                "age": "a different age", "gender": "different presentation"}
_FAMILY = r"(sister|brother|twin|mother|father|mum|mom|dad|son|daughter|cousin|aunt|uncle|grandmother|grandfather|sibling)"
_STOP = {"and", "with", "the", "a", "an", "of", "in", "on", "slightly", "very", "quite", "none", "no", "set", "her", "his"}


def _words(text: str) -> set:
    return set(re.findall(r"[a-z]+", (text or "").lower())) - _STOP


def similarity(a: dict, b: dict) -> tuple[float, list[str]]:
    """(0..1, matching features) of two characters' identities."""
    ia, ib = a.get("identity") or {}, b.get("identity") or {}
    score = total = 0.0
    same = []
    for label, read, w in FIELDS:
        wa, wb = _words(read(a, ia)), _words(read(b, ib))
        if not wa or not wb:
            continue
        overlap = len(wa & wb) / len(wa | wb)
        total += w
        score += w * overlap
        if overlap >= 0.5:
            same.append(label)
    return (round(score / total, 2) if total else 0.0), same


def related(a: dict, b: dict) -> bool:
    """The story makes them family (siblings may share features): a family word next to the other's name, or a
    shared surname."""
    na, nb = (a.get("name") or "").split(), (b.get("name") or "").split()
    if len(na) > 1 and len(nb) > 1 and na[-1].lower() == nb[-1].lower():
        return True
    for x, y in ((a, nb), (b, na)):
        text = f"{x.get('description') or ''} {x.get('personality') or ''}".lower()
        if y and re.search(rf"{_FAMILY}\W+(of\W+)?{re.escape(y[0].lower())}|{re.escape(y[0].lower())}'s\W+{_FAMILY}", text):
            return True
    return False


def _suggest(b: dict, same: list[str]) -> str:
    first = [CHANGE_FIRST[f] for f in ("hair", "clothes", "build", "skin tone", "marks", "facial hair", "face shape")
             if f in same][:2]
    return f"Give {b.get('name')} {' and '.join(first) if first else 'one clearly different feature'}."


def cast_check(cast: list[dict], others: Optional[list[dict]] = None, fingerprints: Optional[dict] = None) -> dict:
    """Look-alike pairs in this cast, and against the creator's other dramas' characters (others: rows with
    'project_title'). fingerprints: {character id: anchor picture hash}."""
    import studio_film_report as fr
    pairs = []
    for i in range(len(cast)):
        for j in range(i + 1, len(cast)):
            a, b = cast[i], cast[j]
            score, same = similarity(a, b)
            fa, fb = (fingerprints or {}).get(a["id"]), (fingerprints or {}).get(b["id"])
            picture = bool(fa and fb and fr.distance(fa, fb) <= fr.HASH_DUPLICATE)
            fam = related(a, b)
            if score >= LOOK_ALIKE or picture:
                pairs.append({"a": a["name"], "b": b["name"], "scope": "this drama", "score": score, "same": same,
                              "same_picture": picture, "related": fam,
                              "severity": "low" if fam and not picture else "high" if picture or score >= 0.8 else "medium",
                              "problem": (f"{a['name']} and {b['name']} have almost the same reference picture."
                                          if picture else f"{a['name']} and {b['name']} look alike: same "
                                          f"{', '.join(same) or 'overall look'}.")
                                         + (" The story makes them family, so some likeness is expected." if fam else ""),
                              "fix": _suggest(b, same)})
    seen = {c.get("identity_id") or c["id"] for c in cast}
    for c in cast:
        for o in others or []:
            if (o.get("identity_id") or o["id"]) in seen:          # a returning character: the same person on purpose
                continue
            score, same = similarity(c, o)
            if score >= ACROSS_DRAMAS:
                pairs.append({"a": c["name"], "b": o["name"], "scope": f"your drama \"{o.get('project_title') or ''}\"",
                              "score": score, "same": same, "same_picture": False, "related": False,
                              "severity": "low",
                              "problem": f"{c['name']} looks like {o['name']} from your drama "
                                         f"\"{o.get('project_title') or ''}\": same {', '.join(same)}.",
                              "fix": _suggest(c, same) + " Each drama's cast should be its own."})
    pairs.sort(key=lambda p: ({"high": 0, "medium": 1, "low": 2}[p["severity"]], -p["score"]))
    return {"pairs": pairs, "checked": len(cast), "compared_with_other_dramas": len(others or []),
            "note": "Compares the characters' written identities and their reference pictures. It's advice: two "
                    "different descriptions can still be drawn alike, so look at the pictures too."}


def distinct_block(cast: list[dict], name: str) -> str:
    """Told to the AI that writes a new character's identity: who is already in this drama."""
    others = [c for c in cast if c.get("name") != name and (c.get("identity") or {}).get("face")]
    if not others:
        return ""
    import studio_identity as ident
    lines = []
    for c in others[:6]:
        i = c.get("identity") or {}
        bits = [x for x in ((i.get("basic") or {}).get("skin_tone"), (i.get("face") or {}).get("hair"),
                            (i.get("basic") or {}).get("build"), i.get("default_outfit")) if x]
        lines.append(f"- {c['name']}: {'; '.join(bits)[:220]}")
    return ("CHARACTERS ALREADY IN THIS DRAMA (make this person clearly different from them at a glance: hair, "
            "build, colouring and clothes, unless the story says they are family):\n" + "\n".join(lines))


# ------------------------------------------------------------------ the DNA card

def dna(ch: dict, *, urls: dict, outfits: list[dict], bible: dict, scenes: list[dict], shots: list[dict],
        appears_in: list[int], check_pairs: list[dict]) -> dict:
    import studio_identity as ident
    views = ch.get("reference_views") or {}
    status = ident.sheet_status(views)
    first = (ch.get("name") or "").split(" ")[0].lower()
    saga = (bible.get("saga") or {})
    story = next((c for c in (saga.get("characters") or []) + (bible.get("characters") or [])
                  if (c.get("name") or "").split(" ")[0].lower() == first), {})
    rels = [r for r in saga.get("relationships") or [] if first in (r.get("between") or "").lower()]
    emotion = None
    for s in sorted(scenes, key=lambda s: s.get("position") or 0):
        for d in s.get("dialogue") or []:
            if (d.get("character") or "").split(" ")[0].lower() == first and d.get("emotion"):
                emotion = {"emotion": d["emotion"], "scene": (s.get("position") or 0) + 1}
    mine = [sh for sh in shots if any((c.get("name") or "").split(" ")[0].lower() == first for c in sh.get("characters") or [])]
    checked = [sh for sh in mine if ((sh.get("inspection") or {}).get("result"))]
    failed = [sh for sh in checked if (((sh.get("inspection") or {}).get("result") or {}).get("problems"))
              and any(p.get("kind") in ("character", "identity", "face", "outfit") and not p.get("intentional")
                      for p in sh["inspection"]["result"]["problems"])]
    return {
        "id": ch["id"], "name": ch["name"], "identity_id": ch.get("identity_id") or ch["id"],
        "locked": bool(ch.get("identity_locked_at")),
        "look": {"identity": ch.get("identity") or {}, "guessed": ident.guessed_fields(ch.get("identity")),
                 "reference_views": [{"view": v, "label": ident.VIEW_LABEL[v], "status": status[v],
                                      "url": urls.get((views.get(v) or {}).get("path") or "")} for v in ident.VIEWS],
                 "outfits": [{"name": o.get("name"), "description": o.get("description") or ""} for o in outfits]},
        "voice": {"voice": ch.get("voice"), "description": ch.get("voice_description") or "",
                  "status": ch.get("voice_status") or ("cast" if ch.get("voice") else "needs_voice"),
                  "locked": bool(ch.get("voice_locked_at"))},
        "person": {"age": ch.get("age_years") or ch.get("age_category"), "gender": ch.get("gender_presentation"),
                   "personality": ch.get("personality") or "", "description": ch.get("description") or ""},
        "story": {k: story.get(k) for k in ("role", "history", "wants", "fears_losing", "hides", "strength", "weakness",
                                             "inner_conflict", "arc", "want", "need", "flaw", "secret") if story.get(k)},
        "relationships": [{"between": r.get("between"), "kind": r.get("kind"), "tension": r.get("tension")} for r in rels],
        "emotional_state": emotion,
        "appears_in_scenes": appears_in,
        "consistency": {"shots": len(mine), "checked": len(checked), "flagged": len(failed),
                        "note": "Shots where VIYO's picture check found this character drawn differently from the "
                                "approved references." if failed else ""},
        "look_alikes": [p for p in check_pairs if ch["name"] in (p["a"], p["b"])],
    }
