"""Viyo's story world: one structured, reusable model of WHERE and WHEN a story
happens, built once per production (or series) and passed into every step
that writes or draws it.

A world has a setting (country, region, place, era and a mode: historically
grounded, alternate history, a fictional world inspired by a real culture, or
a modern real setting), society, culture and manners, language and names,
the material world (buildings, clothes, objects, food, work), technology and
transport (and what does NOT exist yet), environment and sound, history,
a visual identity, FACTS each labelled documented / interpretation / invented
/ uncertain (documented ones carry their source when the world was researched
with web search), what is unknown, and the confusions to avoid for THIS place
(details of neighbouring or often-confused cultures, researched for every
setting the same way: no country has a special list).

The creator can correct it; corrections override everything and are shown
first to every model. Nothing here proves historical accuracy: it says where a
claim comes from and how sure it is.

Used by: the story plan, screenplay, breakdown, character pictures, scene and
shot pictures, video clips, places/objects, outfits, music, the assistant and
the story check (studio_story_check: authenticity and immersion).
"""
from __future__ import annotations

import datetime
import json
import re
from typing import Optional

MODES = {
    "grounded": "Historically grounded fiction: invented characters and events inside a real place and period, "
                "true to what is documented; anything uncertain is labelled.",
    "alternate": "Alternate history: a real place and period where one thing went differently; the change is "
                 "deliberate and everything else stays true to the period.",
    "fictional": "A fictional world inspired by a real culture or period: free invention, but the inspiration is "
                 "respected and not mixed up with other cultures.",
    "modern": "A real place today: contemporary, specific and true to that place, not a generic city.",
}
STATUSES = ("documented", "interpretation", "invented", "uncertain")

_S = {"type": "string"}
_L = {"type": "array", "items": _S}
WORLD_SCHEMA = {
    "type": "object",
    "properties": {
        "setting": {"type": "object", "properties": {
            "country": _S, "region": _S, "place": _S, "era": _S,
            "year": {"type": "integer", "description": "A representative year (negative = BCE). 0 if fictional/unknown."},
            "summary": {"type": "string", "description": "Two sentences: where and when, and what life is like."}},
            "required": ["country", "place", "era", "year", "summary"]},
        "society": {"type": "object", "properties": {
            "power": _S, "social_structure": _S, "family": _S, "gender_roles": _S, "wealth_and_status": _S,
            "law_and_conflict": _S}},
        "culture": {"type": "object", "properties": {
            "traditions": _L, "respect_and_greetings": _S, "how_emotion_is_shown": _S, "beliefs": _S, "taboos": _L,
            "storytelling_and_arts": _S}},
        "language": {"type": "object", "properties": {
            "spoken": _S, "naming": _S, "example_names": _L, "forms_of_address": _L,
            "words": {"type": "array", "items": {"type": "object", "properties": {
                "word": _S, "meaning": _S}, "required": ["word", "meaning"]}}}},
        "material": {"type": "object", "properties": {
            "architecture": _S, "settlements": _S, "interiors": _S, "clothing": _S, "clothing_by_status": _S,
            "hair_and_adornment": _S, "materials": _L, "objects": _L, "food": _S, "work_and_trade": _S, "daily_life": _S}},
        "technology": {"type": "object", "properties": {
            "available": _L, "transport": _S,
            "not_available": {"type": "array", "items": _S, "description": "Things that must NOT appear (anachronisms)."}}},
        "environment": {"type": "object", "properties": {
            "geography": _S, "climate": _S, "landscape": _S, "plants_and_animals": _S, "light": _S, "sounds": _L}},
        "history": {"type": "object", "properties": {"conflicts": _S, "conditions": _S, "recent_events": _S}},
        "visual": {"type": "object", "properties": {
            "palette": _S, "textures": _S, "signature_details": _L}},
        "sound": {"type": "object", "properties": {"ambience": _L, "music": _S, "instruments": _L}},
        "facts": {"type": "array", "items": {"type": "object", "properties": {
            "claim": _S, "status": {"type": "string", "enum": list(STATUSES)},
            "source": {"type": "integer", "description": "The number of the research source behind a documented claim (0 if none)."},
            "note": _S}, "required": ["claim", "status"]}},
        "unknown": {"type": "array", "items": _S, "description": "What isn't known or is disputed, said plainly."},
        "avoid": {"type": "array", "items": _S,
                  "description": "Confusions to avoid: other cultures' dress, words, buildings or customs that don't belong here."},
    },
    "required": ["setting", "society", "culture", "language", "material", "technology", "environment", "facts", "unknown",
                 "avoid"],
}

# Things that didn't exist before a year (approximate, conservative). "camera"
# is left out: it's a film word in screenplays.
INVENTIONS = {
    "smartphone": 2007, "mobile phone": 1983, "cellphone": 1983, "phone": 1876, "telephone": 1876, "laptop": 1981,
    "computer": 1945, "internet": 1983, "email": 1971, "television": 1927, "tv": 1927, "radio": 1895,
    "car": 1886, "motorcycle": 1885, "truck": 1896, "bus": 1895, "airplane": 1903, "aeroplane": 1903,
    "helicopter": 1939, "train": 1825, "railway": 1825, "bicycle": 1817, "electricity": 1880, "electric": 1880,
    "light bulb": 1879, "lightbulb": 1879, "plastic": 1907, "jeans": 1873, "t-shirt": 1913, "zipper": 1913,
    "sneakers": 1917, "wristwatch": 1810, "photograph": 1826, "photo": 1826, "cigarette": 1830, "tarmac": 1902,
    "rifle": 1500, "pistol": 1500, "revolver": 1836, "machine gun": 1884, "sunglasses": 1929, "refrigerator": 1913,
}

_SECTIONS = ("setting", "society", "culture", "language", "material", "technology", "environment", "history",
             "visual", "sound")


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _t(x, n=600) -> str:
    return re.sub(r"\s+", " ", str(x or "")).strip()[:n]


def _list(x, n=12, size=200) -> list[str]:
    return [_t(v, size) for v in (x or [])[:n] if _t(v, size)] if isinstance(x, list) else []


# ------------------------------------------------------------------ building

def research_prompt(brief: dict, idea: str, mode: str) -> str:
    return f"""Research the setting of this story so a film can be made true to its place and time. Use reliable
sources (academic, museum, encyclopedic). Separate what is documented from what historians interpret and from what
is unknown or disputed. Be specific to this exact place and period: never borrow from neighbouring cultures.

STORY IDEA: {idea}
SETTING WANTED: {brief.get('setting') or '(take it from the idea)'}
MODE: {MODES.get(mode, MODES['grounded'])}

Cover: power and social structure; family and how people address each other; customs, respect, beliefs and taboos;
names and language; buildings and settlements; clothing by status, gender and age, hair and adornment; materials,
objects, food, work and trade; technology and transport available, and what did not exist yet; landscape, climate,
light and everyday sounds; music and instruments; conflicts and conditions of the time. Cite your sources."""


def world_prompt(brief: dict, idea: str, mode: str, research: str = "", sources: Optional[list] = None,
                 corrections: Optional[list] = None) -> str:
    src = "\n".join(f"[{i}] {s.get('title') or ''} {s.get('url') or ''}" for i, s in enumerate(sources or [], start=1))
    fix = "\n".join(f"- {c['field']}: {c['value']}" + (f" ({c['note']})" if c.get("note") else "") for c in corrections or [])
    return f"""You are the world builder of a film studio. Build the story world for this production as structured
data that writers, costume, set, camera and sound departments will all use.

STORY IDEA: {idea}
SETTING WANTED: {brief.get('setting') or '(take it from the idea)'}
MODE: {MODES.get(mode, MODES['grounded'])}
LANGUAGE OF THE FILM: {brief.get('language') or 'English'}
{('CREATOR CORRECTIONS (these override everything):' + chr(10) + fix + chr(10)) if fix else ''}
{('RESEARCH NOTES:' + chr(10) + research[:12000] + chr(10)) if research else 'No web research was done: label claims honestly (interpretation or uncertain unless you are certain).' + chr(10)}
{('SOURCES:' + chr(10) + src + chr(10)) if src else ''}
RULES:
- Be specific to THIS place and period, wherever it is. Never default to American, European or modern life, and
  never treat a continent, region or country as one culture: a story keeps its own people's dress, words, food,
  buildings and customs, not their neighbours'.
- facts: the claims the film relies on, each labelled: documented (with the source number when there is one),
  interpretation (a reasonable reading), invented (made up for the story), uncertain (disputed or unknown).
  Never label a guess as documented.
- unknown: say plainly what isn't known.
- avoid: 5-15 concrete things that are often wrongly shown for THIS place and period: the dress, words, buildings,
  food and customs of neighbouring or commonly confused cultures, and common stereotypes or film clichés about it.
  Specific words (a garment's name, a building type), not general advice. Required for every real place.
- technology.not_available: everyday things that did not exist yet and must never appear.
- For an alternate history, name the one deliberate change in setting.summary; for a fictional world, say which
  real culture inspires it and keep that inspiration coherent.
- Visual and sound details are concrete (materials, colours, light, sounds), so pictures and music can follow them."""


def clean_world(raw, mode: str, sources: Optional[list] = None, researched: bool = False) -> dict:
    """Valid fields only. A 'documented' claim without a research source is
    downgraded to 'unsourced' (shown as not verified)."""
    raw = raw if isinstance(raw, dict) else {}
    out = {"mode": mode if mode in MODES else "grounded", "researched": researched,
           "sources": [{"title": _t(s.get("title"), 200), "url": _t(s.get("url"), 400)} for s in (sources or [])[:20]
                       if isinstance(s, dict) and s.get("url")],
           "built_at": _now(), "corrections": [], "approved": False}
    for sec in _SECTIONS:
        val = raw.get(sec) if isinstance(raw.get(sec), dict) else {}
        clean = {}
        for k, v in val.items():
            if isinstance(v, list):
                if v and isinstance(v[0], dict):
                    clean[k] = [{kk: _t(vv, 200) for kk, vv in d.items() if isinstance(d, dict)} for d in v[:20] if isinstance(d, dict)]
                else:
                    clean[k] = _list(v, 20)
            elif isinstance(v, (int, float)) and not isinstance(v, bool):
                clean[k] = int(v)
            elif _t(v):
                clean[k] = _t(v, 900)
        out[sec] = clean
    facts = []
    for f in raw.get("facts") or []:
        if not isinstance(f, dict) or not _t(f.get("claim")):
            continue
        status = f.get("status") if f.get("status") in STATUSES else "uncertain"
        item = {"claim": _t(f["claim"], 300), "status": status, "note": _t(f.get("note"), 200)}
        n = f.get("source") if isinstance(f.get("source"), int) else 0
        if status == "documented":
            if researched and 1 <= n <= len(out["sources"]):
                item["source"] = out["sources"][n - 1]
            else:
                item["status"] = "unsourced"      # claimed as documented, but nothing backs it
        facts.append(item)
    out["facts"] = facts[:40]
    out["unknown"] = _list(raw.get("unknown"), 15, 300)
    out["avoid"] = _list(raw.get("avoid"), 20, 120)
    return out


def year(world: dict) -> Optional[int]:
    y = (world.get("setting") or {}).get("year")
    return y if isinstance(y, int) and y != 0 else None


def apply_correction(world: dict, field: str, value: str, note: str = "") -> dict:
    """A creator's correction ("setting.era", "material.clothing", "avoid", ...). Kept in the list and written
    into the field; corrections are shown first to every model."""
    world = json.loads(json.dumps(world or {}))
    field, value = _t(field, 80), _t(value, 600)
    if not field or not value:
        raise ValueError("A correction needs a field and a value.")
    world.setdefault("corrections", []).append({"field": field, "value": value, "note": _t(note, 300), "at": _now()})
    sec, _, key = field.partition(".")
    if sec in ("avoid", "unknown"):
        world.setdefault(sec, []).append(value)
    elif sec in _SECTIONS and key:
        world.setdefault(sec, {})[key] = value
    world["approved"] = False            # a changed world is approved again by the creator
    return world


# ------------------------------------------------------------------ what each department gets

def _corrections(world: dict) -> str:
    cs = world.get("corrections") or []
    return ("CREATOR CORRECTIONS (override everything below): " + "; ".join(
        f"{c['field']}: {c['value']}" for c in cs[-10:]) + "\n") if cs else ""


def _join(d: dict, keys) -> str:
    return "; ".join(f"{k.replace('_', ' ')}: {', '.join(d[k]) if isinstance(d[k], list) else d[k]}"
                     for k in keys if d.get(k))


def _guard(world: dict) -> str:
    bits = list((world.get("technology") or {}).get("not_available") or [])
    bits += world.get("avoid") or []
    return ("NEVER SHOW: " + ", ".join(bits[:24]) + ".") if bits else ""


def block(world: Optional[dict], purpose: str) -> str:
    """The world for one department: 'story', 'characters', 'image', 'video', 'sound'."""
    if not world or not world.get("setting"):
        return ""
    st = world.get("setting") or {}
    head = (f"STORY WORLD ({MODES.get(world.get('mode'), '').split(':')[0]}): {st.get('place') or ''}, "
            f"{st.get('region') or ''} {st.get('country') or ''}, {st.get('era') or ''}. {st.get('summary') or ''}")
    m, cu, la = world.get("material") or {}, world.get("culture") or {}, world.get("language") or {}
    env, vis, snd = world.get("environment") or {}, world.get("visual") or {}, world.get("sound") or {}
    lines = [_corrections(world) + head]
    if purpose == "story":
        lines += [f"Society: {_join(world.get('society') or {}, ('power', 'social_structure', 'family', 'gender_roles', 'wealth_and_status', 'law_and_conflict'))}",
                  f"Culture: {_join(cu, ('traditions', 'respect_and_greetings', 'how_emotion_is_shown', 'beliefs', 'taboos'))}",
                  f"Language and names: {_join(la, ('spoken', 'naming', 'example_names', 'forms_of_address'))}"
                  + ("; words: " + ", ".join(f"{w.get('word')} ({w.get('meaning')})" for w in la.get("words") or [])[:400] if la.get("words") else ""),
                  f"Daily life: {_join(m, ('work_and_trade', 'food', 'daily_life', 'settlements'))}",
                  f"Technology: {_join(world.get('technology') or {}, ('available', 'transport'))}",
                  f"History: {_join(world.get('history') or {}, ('conflicts', 'conditions', 'recent_events'))}"]
        facts = world.get("facts") or []
        if facts:
            lines.append("Facts (keep the labels; never present invented or uncertain things as history): " + "; ".join(
                f"[{f['status']}] {f['claim']}" for f in facts[:20]))
        if world.get("unknown"):
            lines.append("Unknown or disputed: " + "; ".join(world["unknown"][:8]))
    elif purpose == "characters":
        lines += [f"Names and address: {_join(la, ('naming', 'example_names', 'forms_of_address'))}",
                  f"Clothing: {_join(m, ('clothing', 'clothing_by_status', 'hair_and_adornment', 'materials'))}",
                  f"Manners: {_join(cu, ('respect_and_greetings', 'how_emotion_is_shown'))}"]
    elif purpose == "image":
        lines += [f"Look of the world: {_join(m, ('architecture', 'interiors', 'clothing', 'clothing_by_status', 'hair_and_adornment', 'materials', 'objects'))}",
                  f"Environment: {_join(env, ('landscape', 'climate', 'plants_and_animals', 'light'))}",
                  f"Visual identity: {_join(vis, ('palette', 'textures', 'signature_details'))}"]
    elif purpose == "video":
        lines += [f"Look: {_join(m, ('architecture', 'clothing', 'materials'))}; light: {env.get('light') or ''}",
                  f"How people move and behave: {_join(cu, ('respect_and_greetings', 'how_emotion_is_shown'))}",
                  f"Background sound: {', '.join((snd.get('ambience') or []) + (env.get('sounds') or []))[:300]}"]
    elif purpose == "sound":
        lines = [head, f"Music: {snd.get('music') or ''}; instruments: {', '.join(snd.get('instruments') or [])}",
                 f"Ambience: {', '.join((snd.get('ambience') or []) + (env.get('sounds') or []))}"]
        return "\n".join(x for x in lines if x.split(':', 1)[-1].strip(' ;')) + "\n"
    guard = _guard(world)
    if guard and purpose in ("image", "video", "characters"):
        lines.append(guard)
    return "\n".join(x for x in lines if x.split(':', 1)[-1].strip(' ;')) + "\n"


# ------------------------------------------------------------------ checks that need no AI

def _found(term: str, text: str) -> bool:
    """Whole words, singular or plural ("wheeled carts" matches "a wheeled cart")."""
    t = term.lower().strip()
    stem = t[:-1] if t.endswith("s") and not t.endswith("ss") and len(t) > 4 else t
    low = text.lower()
    for m in re.finditer(rf"(?<![a-z]){re.escape(stem)}(?:s|es)?(?![a-z])", low):
        before = low[max(0, m.start() - 40):m.start()].split()[-4:]
        if not any(w.strip(",.;:()") in ("no", "not", "never", "without", "nor", "instead", "avoid") for w in before):
            return True              # "no woven cotton cloth" is a denial, not a sighting
    return False


def _longest(hits) -> list[str]:
    hits = {h.lower() for h in hits}
    return sorted(h for h in hits if not any(h != o and h in o for o in hits))


def anachronisms(text: str, world: Optional[dict]) -> list[str]:
    """Things in the text that didn't exist yet in the world's year (unless the world lists them as available)."""
    y = year(world or {})
    if not world or y is None or world.get("mode") == "fictional":
        return []
    allowed = " ".join((world.get("technology") or {}).get("available") or []).lower()
    hits = [w for w, since in INVENTIONS.items() if y < since and _found(w, text) and w not in allowed]
    hits += [w for w in (world.get("technology") or {}).get("not_available") or [] if len(w) > 2 and _found(w, text)]
    return _longest(hits)      # "mobile phone" also matches "phone": keep the longest


def confusions(text: str, world: Optional[dict]) -> list[str]:
    """Other cultures' markers in the text: the world's own researched avoid list and the creator's
    corrections. The same rule for every place; no country has a built-in list."""
    if not world:
        return []
    terms = list(world.get("avoid") or [])
    return _longest(t for t in terms if 2 < len(t) <= 40 and _found(t, text))


def check_text(text: str, world: Optional[dict]) -> list[dict]:
    """Problems in a screenplay, scene or prompt text against the world."""
    out = []
    a = anachronisms(text, world)
    if a:
        out.append({"area": "authenticity", "kind": "anachronism",
                    "problem": f"Things that don't exist in {(world.get('setting') or {}).get('era')}: {', '.join(a)}.",
                    "fix": f"Replace {', '.join(a)} with what people of this time and place actually used."})
    c = confusions(text, world)
    if c:
        out.append({"area": "authenticity", "kind": "confusion",
                    "problem": f"Details from another culture: {', '.join(c)}.",
                    "fix": f"Use this culture's own clothing, words and customs instead of {', '.join(c)}."})
    return out


def evaluate(world: Optional[dict]) -> dict:
    """The world's own quality report (no AI): depth per department, honest
    labelling of facts, and contradictions inside the world itself."""
    if not world or not world.get("setting"):
        return {"status": "none", "problems": [], "sections": {}}
    sections = {s: sum(1 for v in (world.get(s) or {}).values() if v) for s in _SECTIONS}
    problems = []
    thin = [s for s in ("setting", "society", "culture", "language", "material", "environment") if sections.get(s, 0) < 2]
    if thin:
        problems.append({"kind": "thin", "problem": f"Thin: {', '.join(thin)}.", "fix": "Build those parts of the world."})
    if world.get("mode") != "fictional" and len(world.get("avoid") or []) < 3:
        problems.append({"kind": "no_confusions", "problem": "Few or no confusions listed for this place, so mix-ups with "
                                                             "neighbouring cultures can't be caught.",
                         "fix": "Rebuild the world, or add what is often wrongly shown for this place."})
    facts = world.get("facts") or []
    counts = {s: sum(1 for f in facts if f["status"] == s) for s in STATUSES + ("unsourced",)}
    if counts["unsourced"]:
        problems.append({"kind": "unsourced", "problem": f"{counts['unsourced']} claim(s) called documented with no "
                                                         "source; shown as not verified.", "fix": "Research them or relabel them."})
    historical = world.get("mode") in ("grounded", "alternate") and (year(world) or 3000) < 1950
    if historical and not world.get("unknown") and not counts["uncertain"]:
        problems.append({"kind": "overconfident", "problem": "A historical world with nothing marked unknown or "
                                                             "uncertain: it's probably claiming more than is known.",
                         "fix": "Say what is disputed or unknown."})
    inner = json.dumps({k: world.get(k) for k in ("material", "technology", "environment", "visual")}, ensure_ascii=False)
    for p in check_text(inner.replace("not_available", ""), {**world, "technology": {**(world.get("technology") or {}),
                                                                                       "not_available": []}}):
        problems.append({"kind": p["kind"], "problem": "In the world itself: " + p["problem"], "fix": p["fix"]})
    status = "needs_work" if problems else "ok"
    return {"status": status, "problems": problems, "sections": sections, "facts": counts,
            "researched": bool(world.get("researched")), "sources": len(world.get("sources") or [])}


def summary_lines(world: Optional[dict]) -> list[str]:
    """For the assistant."""
    if not world or not world.get("setting"):
        return ["STORY WORLD: not built yet (propose build_world before writing a story set in a specific culture or period)."]
    st = world["setting"]
    ev = evaluate(world)
    out = [f"STORY WORLD ({world.get('mode')}, {'approved' if world.get('approved') else 'NOT approved yet'}, "
           f"{'researched with web search, ' + str(ev['sources']) + ' sources' if world.get('researched') else 'NOT researched with web search'}): "
           f"{st.get('place')}, {st.get('country')}, {st.get('era')}. {st.get('summary') or ''}",
           f"  Facts: {ev['facts']['documented']} documented with a source, {ev['facts']['interpretation']} interpretation, "
           f"{ev['facts']['invented']} invented, {ev['facts']['uncertain']} uncertain, {ev['facts']['unsourced']} called "
           "documented without a source (not verified)."]
    for f in (world.get("facts") or [])[:12]:
        out.append(f"  - [{f['status']}] {f['claim']}" + (f" (source: {f['source']['title'] or f['source']['url']})" if f.get("source") else ""))
    if world.get("unknown"):
        out.append("  Unknown/disputed: " + "; ".join(world["unknown"][:6]))
    if world.get("avoid"):
        out.append("  Avoid: " + ", ".join(world["avoid"][:12]))
    for c in (world.get("corrections") or [])[-6:]:
        out.append(f"  Creator corrected {c['field']}: {c['value']}")
    for p in ev["problems"][:4]:
        out.append(f"  World check: {p['problem']}")
    return out
