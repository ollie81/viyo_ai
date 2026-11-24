"""Story Intelligence stage 2: from a creator's idea to an approved novel outline.

1. Interview. VIYO asks a few adaptive questions, one at a time, only about what it doesn't know yet: who the story
   is about, what they want and what stands in the way, the genre and tone, what the audience should feel, where
   and when it happens, and how it ends. The AI chooses the next question and says what it has understood, but only
   what the creator actually said counts: every understood point must come with the creator's own words, which
   are looked up in the idea and the answers. Anything the creator didn't say isn't treated as known. The creator
   can skip a question ("VIYO decides") or start the outline early; anything still open then becomes an
   ASSUMPTION the outline must list for the creator to see, never a hidden choice. At most MAX_QUESTIONS.
2. Outline. One plan for the whole novel: premise, characters (want, need, flaw, wound, secret, arc, appearance,
   voice), relationships, places, important objects, secrets (who knows them, when they come out), chapter by
   chapter (purpose, summary, beats, point of view, emotion, setups, payoffs), threads and foreshadowing (planted
   and paid off), the ending, the assumptions, and how each thing the creator said was kept. A checker finds
   structural problems (wrong chapter count, a secret revealed outside the book, a payoff before its setup,
   a name nobody knows, something the creator said that was lost); the AI is sent back once to fix them, and what
   remains is shown to the creator.
3. Approval. Only the creator approves an outline version. Approval seeds the story canon (studio_canon) with what
   the plan fixes: characters and their lasting traits, places, objects, secrets and who knows them at the start,
   relationships at the start, and the planned reveals and threads. Those records are approved because the
   creator approved the plan, and each keeps the outline text it came from as its evidence. Chapter beats stay
   in the outline: they are a plan, not something the story has said yet.

Pure logic: prompts, schemas, cleaning and checks. The API (studio_novel_api.py) calls the AI through
studio_ai and charges through the Studio's normal paid-call path.
"""
from __future__ import annotations

import datetime
import json
import re
from typing import Optional

MAX_QUESTIONS = 8
LENGTHS = {"short": {"chapters": 12, "words": 2500}, "standard": {"chapters": 24, "words": 3000},
           "long": {"chapters": 36, "words": 3000}}

# What VIYO needs to know before planning a novel, with the plain question it falls back to.
TOPICS = {
    "protagonist": "Who is the story about? Tell me about the main character.",
    "goal": "What does the main character want most, and what stands in the way?",
    "genre_tone": "What kind of story is it: romance, thriller, family drama, comedy...? Light or dark?",
    "feeling": "How should the audience feel while watching and at the end?",
    "setting": "Where and when does it happen?",
    "ending": "How should it end: happy, bittersweet, tragic, or open?",
}
OPTIONAL_TOPICS = {
    "audience": "Who is it for (teenagers, adults, families)?",
    "avoid": "Is there anything you don't want in the story?",
    "relationships": "Who matters most to the main character?",
    "twist": "Is there a secret or twist you already have in mind?",
}
ALL_TOPICS = {**TOPICS, **OPTIONAL_TOPICS}


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _norm(text: str) -> str:
    t = (text or "").lower().replace("’", "'").replace("‘", "'")
    return " ".join(re.findall(r"[^\W_]+(?:'[^\W_]+)?", t))


def said(words: str, creator_text: str) -> bool:
    """True if the creator really said these words (in the idea or an answer), ignoring case and punctuation."""
    w = _norm(words)
    return bool(w) and f" {w} " in f" {_norm(creator_text)} "


def creator_text(story: dict) -> str:
    """Everything the creator has said: the idea and every answer."""
    turns = (story.get("interview") or {}).get("turns") or []
    return "\n".join([story.get("idea") or ""] + [t.get("answer") or "" for t in turns if t.get("answer")])


def turns(story: dict) -> list:
    return list((story.get("interview") or {}).get("turns") or [])


def known(story: dict) -> dict:
    return dict((story.get("interview") or {}).get("known") or {})


def missing(story: dict) -> list:
    k = known(story)
    skipped = {t["question"]["topic"] for t in turns(story) if t.get("skipped") and t.get("question")}
    return [t for t in TOPICS if t not in k and t not in skipped]


def pending(story: dict) -> Optional[dict]:
    """The question waiting for an answer, if any."""
    ts = turns(story)
    return ts[-1] if ts and ts[-1].get("answer") is None and not ts[-1].get("skipped") else None


def ready(story: dict) -> bool:
    return not missing(story) or len(turns(story)) >= MAX_QUESTIONS or bool((story.get("interview") or {}).get("done"))


def record_answer(story: dict, answer: Optional[str], skip: bool = False) -> dict:
    """The interview with the creator's answer (or skip) to the waiting question."""
    iv = dict(story.get("interview") or {})
    ts = list(iv.get("turns") or [])
    if not ts or ts[-1].get("answer") is not None or ts[-1].get("skipped"):
        raise ValueError("There's no question waiting for an answer.")
    last = dict(ts[-1])
    if skip:
        last.update(skipped=True, answered_at=_now())
    else:
        a = (answer or "").strip()
        if not a:
            raise ValueError("Write an answer, or let VIYO decide.")
        last.update(answer=a[:2000], answered_at=_now())
    ts[-1] = last
    iv["turns"] = ts
    return iv


# ------------------------------------------------------------------------------------------- the interview turn

TURN_SCHEMA = {
    "type": "object",
    "properties": {
        "understood": {"type": "array", "items": {"type": "object", "properties": {
            "topic": {"type": "string"}, "value": {"type": "string"}, "creator_words": {"type": "string"}},
            "required": ["topic", "value", "creator_words"]}},
        "question": {"type": "object", "properties": {
            "topic": {"type": "string"}, "text": {"type": "string"}, "why": {"type": "string"},
            "options": {"type": "array", "items": {"type": "string"}}}, "required": ["topic", "text"]},
    },
    "required": ["understood", "question"],
}


def interview_prompt(story: dict, world_text: str = "") -> str:
    lines = ["You are VIYO, a story editor helping a creator turn an idea into a novel that will later become a "
             "drama series. Interview them: ask ONE short, friendly question at a time, about the most important "
             "thing still unknown. Never ask about something they already told you. Offer 2-4 short example "
             "answers (options) that fit THEIR idea, not generic ones. Write in the language they write in.",
             "",
             "Topics you need: " + "; ".join(f"{k} ({v})" for k, v in TOPICS.items()),
             "Useful if natural: " + "; ".join(f"{k} ({v})" for k, v in OPTIONAL_TOPICS.items()),
             "",
             "Also report what you have UNDERSTOOD so far. For each point give the topic, a short value, and "
             "creator_words: words copied EXACTLY from what the creator wrote that say it. If they didn't say it, "
             "don't report it (don't guess, don't infer).",
             "",
             f"THE CREATOR'S IDEA: {story.get('idea') or ''}"]
    if world_text:
        lines += ["", "THE STORY WORLD THEY ALREADY SET UP:", world_text[:2500]]
    ts = [t for t in turns(story) if t.get("answer") or t.get("skipped")]
    if ts:
        lines += ["", "THE INTERVIEW SO FAR:"]
        for t in ts:
            lines.append(f"Q ({t['question'].get('topic')}): {t['question'].get('text')}")
            lines.append("A: " + ("(skipped: VIYO decides)" if t.get("skipped") else t.get("answer") or ""))
    still = missing(story)
    if still:
        lines += ["", "STILL UNKNOWN: " + ", ".join(still)]
    return "\n".join(lines)


def apply_turn(story: dict, raw: dict) -> tuple[dict, dict]:
    """Cleans an AI turn: keeps only understood points the creator really said, and makes sure the question is
    about something still unknown (else asks the next unknown topic plainly). Returns (interview, question)."""
    iv = dict(story.get("interview") or {})
    k = dict(iv.get("known") or {})
    text = creator_text(story)
    rejected = []
    for u in raw.get("understood") or []:
        topic = (u.get("topic") or "").strip().lower()
        if topic not in ALL_TOPICS:
            continue
        if said(u.get("creator_words") or "", text):
            k[topic] = {"value": (u.get("value") or "").strip()[:400],
                        "creator_words": (u.get("creator_words") or "").strip()[:400]}
        else:
            rejected.append(topic)
    iv["known"] = k
    probe = {**story, "interview": iv}
    q = raw.get("question") or {}
    topic = (q.get("topic") or "").strip().lower()
    open_topics = missing(probe)
    asked = {t["question"]["topic"] for t in turns(story) if t.get("question")}
    if topic not in ALL_TOPICS or topic in k or (topic in asked and topic not in open_topics) or not (q.get("text") or "").strip():
        # Not a usable question: ask the next thing still needed plainly, or stop when nothing is needed
        # (an optional topic is only asked when the AI chose it).
        topic = open_topics[0] if open_topics else ""
        q = {"topic": topic, "text": ALL_TOPICS.get(topic, ""), "why": "", "options": []}
    question = {"topic": topic, "text": (q.get("text") or "").strip()[:400], "why": (q.get("why") or "").strip()[:300],
                "options": [str(o).strip()[:120] for o in (q.get("options") or []) if str(o).strip()][:4]}
    iv["unverified"] = sorted(set(rejected))
    return iv, question


def add_question(iv: dict, question: dict) -> dict:
    iv = dict(iv)
    iv["turns"] = list(iv.get("turns") or []) + [{"question": question, "answer": None, "skipped": False, "asked_at": _now()}]
    return iv


# ------------------------------------------------------------------------------------------- the outline

_STR = {"type": "string"}
_STRS = {"type": "array", "items": _STR}
OUTLINE_SCHEMA = {
    "type": "object",
    "properties": {
        "title": _STR, "logline": _STR, "premise": _STR, "genre": _STR, "tone": _STR, "audience_feeling": _STR,
        "themes": _STRS, "point_of_view": _STR, "setting_summary": _STR,
        "characters": {"type": "array", "items": {"type": "object", "properties": {
            "name": _STR, "role": _STR, "age": _STR, "appearance": _STR, "voice": _STR, "personality": _STR,
            "want": _STR, "need": _STR, "flaw": _STR, "wound": _STR, "secret": _STR, "arc": _STR},
            "required": ["name", "role"]}},
        "relationships": {"type": "array", "items": {"type": "object", "properties": {
            "a": _STR, "b": _STR, "side": _STR, "value": _STR}, "required": ["a", "b", "value"]}},
        "places": {"type": "array", "items": {"type": "object", "properties": {"name": _STR, "description": _STR},
                                              "required": ["name"]}},
        "objects": {"type": "array", "items": {"type": "object", "properties": {
            "name": _STR, "description": _STR, "why_it_matters": _STR}, "required": ["name"]}},
        "secrets": {"type": "array", "items": {"type": "object", "properties": {
            "name": _STR, "truth": _STR, "known_by": _STRS, "revealed_in_chapter": {"type": "integer"},
            "revealed_to": _STRS}, "required": ["name", "truth"]}},
        "chapters": {"type": "array", "items": {"type": "object", "properties": {
            "number": {"type": "integer"}, "title": _STR, "purpose": _STR, "summary": _STR, "beats": _STRS,
            "pov": _STR, "emotion": _STR, "setups": _STRS, "payoffs": _STRS},
            "required": ["number", "title", "summary", "beats"]}},
        "threads": {"type": "array", "items": {"type": "object", "properties": {
            "kind": _STR, "description": _STR, "planted_chapter": {"type": "integer"},
            "payoff_chapter": {"type": "integer"}}, "required": ["description", "planted_chapter"]}},
        "ending": {"type": "object", "properties": {"kind": _STR, "description": _STR}},
        "assumptions": _STRS,
        "kept_from_creator": {"type": "array", "items": {"type": "object", "properties": {
            "topic": _STR, "how": _STR}, "required": ["topic", "how"]}},
    },
    "required": ["title", "premise", "characters", "chapters", "ending"],
}


def target(story: dict) -> dict:
    return LENGTHS.get(story.get("length") or "standard", LENGTHS["standard"])


def outline_prompt(story: dict, world_text: str = "", genre_craft: str = "", note: str = "") -> str:
    t = target(story)
    k = known(story)
    open_topics = missing(story) + [x["question"]["topic"] for x in turns(story) if x.get("skipped") and x.get("question")]
    lines = [
        "You are VIYO's head writer. Plan a complete, original NOVEL from the creator's idea. It will later be adapted "
        "into a drama series, so give every chapter a real turn, and make the characters people the audience will "
        "remember. Plan literature, not a list of events: inner lives, subtext, gestures, atmosphere, symbols, "
        "foreshadowing that pays off, cause and effect from one chapter to the next.",
        "",
        f"LENGTH: exactly {t['chapters']} chapters, numbered 1 to {t['chapters']}, about {t['words']} words each when written.",
    ]
    if int(story.get("episodes") or 0):
        lines.append(f"It will become about {story['episodes']} drama episodes: plan enough turns and cliffhanger "
                     "moments for that (several per chapter).")
    lines += ["", f"THE CREATOR'S IDEA (keep it; strengthen it, never swap it for another story): {story.get('idea') or ''}"]
    if k:
        lines += ["", "WHAT THE CREATOR DECIDED (must be kept, in spirit and in detail):"]
        for topic, v in k.items():
            lines.append(f"- {topic}: {v.get('value')} (their words: \"{v.get('creator_words')}\")")
    answers = [x for x in turns(story) if x.get("answer")]
    if answers:
        lines += ["", "THEIR ANSWERS IN FULL:"] + [f"- {x['question'].get('text')} -> {x['answer']}" for x in answers]
    if open_topics:
        lines += ["", "NOT DECIDED BY THE CREATOR (you choose, and list each choice in 'assumptions' so they can "
                  "change it): " + ", ".join(sorted(set(open_topics)))]
    if world_text:
        lines += ["", "THE STORY WORLD (use it; never contradict it):", world_text[:3000]]
    if genre_craft:
        lines += ["", genre_craft]
    lines += ["", "RULES:",
              "- Every main character has a want, a need, a flaw, a wound and an arc; give each a clear appearance "
              "and voice (they will be cast and filmed).",
              "- Every secret says who knows it at the start, and the chapter where it comes out (1 to "
              f"{t['chapters']}), or 0 if it stays hidden.",
              "- Every thread (setup, foreshadowing, mystery, promise, question) is planted in one chapter and paid "
              "off in a LATER chapter.",
              "- Use only names that are in 'characters', 'places' or 'objects'.",
              "- In 'kept_from_creator' say, for EACH thing the creator decided, how the plan keeps it.",
              "- Don't imitate a named existing book, film or series."]
    if note.strip():
        lines += ["", f"THE CREATOR ASKS FOR THIS IN THE NEW VERSION: {note.strip()[:1500]}"]
    return "\n".join(lines)


def _s(v, n=600) -> str:
    return str(v or "").strip()[:n]


def clean_outline(raw: dict) -> dict:
    o = {k: _s(raw.get(k), 1500) for k in ("title", "logline", "premise", "genre", "tone", "audience_feeling",
                                           "point_of_view", "setting_summary")}
    o["title"] = o["title"][:200]
    o["themes"] = [_s(x, 120) for x in raw.get("themes") or [] if _s(x)][:8]
    o["characters"] = [{k: _s(c.get(k), 600) for k in ("name", "role", "age", "appearance", "voice", "personality",
                                                        "want", "need", "flaw", "wound", "secret", "arc")}
                       for c in raw.get("characters") or [] if _s(c.get("name"))][:30]
    for c in o["characters"]:
        c["name"] = c["name"][:120]
    o["relationships"] = [{k: _s(r.get(k), 200) for k in ("a", "b", "side", "value")}
                          for r in raw.get("relationships") or [] if _s(r.get("a")) and _s(r.get("b"))][:60]
    o["places"] = [{"name": _s(p.get("name"), 120), "description": _s(p.get("description"))}
                   for p in raw.get("places") or [] if _s(p.get("name"))][:40]
    o["objects"] = [{"name": _s(p.get("name"), 120), "description": _s(p.get("description")),
                     "why_it_matters": _s(p.get("why_it_matters"))} for p in raw.get("objects") or [] if _s(p.get("name"))][:40]
    o["secrets"] = [{"name": _s(s.get("name"), 120), "truth": _s(s.get("truth")),
                     "known_by": [_s(x, 120) for x in s.get("known_by") or [] if _s(x)],
                     "revealed_in_chapter": int(s.get("revealed_in_chapter") or 0),
                     "revealed_to": [_s(x, 120) for x in s.get("revealed_to") or [] if _s(x)]}
                    for s in raw.get("secrets") or [] if _s(s.get("name"))][:30]
    o["chapters"] = [{"number": int(c.get("number") or 0), "title": _s(c.get("title"), 200),
                      "purpose": _s(c.get("purpose")), "summary": _s(c.get("summary"), 2000),
                      "beats": [_s(b, 400) for b in c.get("beats") or [] if _s(b)][:12], "pov": _s(c.get("pov"), 120),
                      "emotion": _s(c.get("emotion"), 200), "setups": [_s(b, 300) for b in c.get("setups") or [] if _s(b)][:8],
                      "payoffs": [_s(b, 300) for b in c.get("payoffs") or [] if _s(b)][:8]}
                     for c in raw.get("chapters") or []][:60]
    kinds = ("setup", "foreshadowing", "mystery", "promise", "question")
    o["threads"] = [{"kind": (t.get("kind") or "").lower() if (t.get("kind") or "").lower() in kinds else "setup",
                     "description": _s(t.get("description")), "planted_chapter": int(t.get("planted_chapter") or 0),
                     "payoff_chapter": int(t.get("payoff_chapter") or 0)}
                    for t in raw.get("threads") or [] if _s(t.get("description"))][:60]
    e = raw.get("ending") or {}
    o["ending"] = {"kind": _s(e.get("kind"), 60), "description": _s(e.get("description"), 1500)}
    o["assumptions"] = [_s(x, 300) for x in raw.get("assumptions") or [] if _s(x)][:20]
    o["kept_from_creator"] = [{"topic": _s(x.get("topic"), 60).lower(), "how": _s(x.get("how"), 400)}
                              for x in raw.get("kept_from_creator") or [] if _s(x.get("topic"))][:20]
    return o


_NOT_A_NAME = {"", "narrator", "omniscient", "third person", "first person", "multiple", "various", "ensemble"}


def outline_problems(o: dict, story: dict) -> list[str]:
    """What a head writer would send back. Structural only: it can't judge whether the story is good."""
    out = []
    n = target(story)["chapters"]
    chapters = o.get("chapters") or []
    if len(chapters) != n:
        out.append(f"The plan has {len(chapters)} chapters; it needs exactly {n}.")
    if [c["number"] for c in chapters] != list(range(1, len(chapters) + 1)):
        out.append("Chapters must be numbered 1, 2, 3... in order, with no gaps or repeats.")
    for c in chapters:
        if not c.get("summary") or len(c.get("beats") or []) < 2:
            out.append(f"Chapter {c.get('number')} needs a summary and at least two beats.")
    chars = o.get("characters") or []
    names = {_norm(c["name"]) for c in chars}
    first = lambda s: _norm(s).split(" ")[0] if _norm(s) else ""          # "Amani" matches "Amani Uwase"
    firsts = {first(c["name"]) for c in chars}
    known_name = lambda s: _norm(s) in names or first(s) in firsts
    if len(chars) < 2:
        out.append("The plan needs at least two characters.")
    for c in chars:
        if (c.get("role") or "").lower() in ("protagonist", "main", "lead", "main character", "antagonist") or c is chars[0]:
            gaps = [f for f in ("want", "need", "flaw", "arc") if not c.get(f)]
            if gaps:
                out.append(f"{c['name']} needs: {', '.join(gaps)}.")
    places = {_norm(p["name"]) for p in o.get("places") or []}
    for s in o.get("secrets") or []:
        r = s.get("revealed_in_chapter") or 0
        if r < 0 or r > n:
            out.append(f"The secret \"{s['name']}\" is revealed in chapter {r}, outside the book (1 to {n}, or 0).")
        for who in s.get("known_by") or []:
            if not known_name(who) and _norm(who) not in ("nobody", "no one", "none"):
                out.append(f"\"{who}\" knows the secret \"{s['name']}\" but isn't one of the characters.")
    for t in o.get("threads") or []:
        p, q = t.get("planted_chapter") or 0, t.get("payoff_chapter") or 0
        if not (1 <= p <= n):
            out.append(f"The thread \"{t['description'][:60]}\" is planted outside the book.")
        elif q and not (p < q <= n):
            out.append(f"The thread \"{t['description'][:60]}\" must pay off after chapter {p} and by chapter {n}.")
    for r in o.get("relationships") or []:
        for who in (r["a"], r["b"]):
            if not known_name(who):
                out.append(f"The relationship names \"{who}\", who isn't one of the characters.")
    for c in chapters:
        pov = c.get("pov") or ""
        if _norm(pov) not in _NOT_A_NAME and not known_name(pov) and _norm(pov) not in places:
            out.append(f"Chapter {c['number']} is told by \"{pov}\", who isn't one of the characters.")
    if not (o.get("ending") or {}).get("description"):
        out.append("The plan needs an ending.")
    kept = {x["topic"] for x in o.get("kept_from_creator") or []}
    for topic in known(story):
        if topic not in kept:
            out.append(f"Say how the plan keeps what the creator decided about \"{topic}\".")
    return list(dict.fromkeys(out))


BLOCKING = ("chapters; it needs exactly", "numbered 1, 2, 3", "at least two characters", "needs an ending")


def blocking(problems: list[str]) -> list[str]:
    """Problems that make an outline unusable (approval is refused). The rest are shown as warnings."""
    return [p for p in problems if any(b in p for b in BLOCKING)]


def fix_prompt(base: str, outline: dict, problems: list[str]) -> str:
    return (base + "\n\nYOUR PLAN HAS THESE PROBLEMS; return the whole plan fixed (keep everything that was fine):\n"
            + "\n".join(f"- {p}" for p in problems) + "\n\nYOUR PLAN:\n" + json.dumps(outline, ensure_ascii=False)[:30000])


# ------------------------------------------------------------------------------------------- approval -> canon

_TRAITS = ("role", "age", "appearance", "voice", "personality", "want", "need", "flaw", "wound", "arc")


def seed_canon(repo, outline: dict, version: int) -> dict:
    """Puts what an APPROVED outline fixes into the story canon. Idempotent: an entity already there (same name or
    alias) isn't added again, and a fact already in the canon for the same question isn't duplicated."""
    ref = f"outline v{version}"
    canon = repo.load()
    added = {"entities": 0, "facts": 0, "threads": 0}

    # A character is usually called by their first name ("Amani" for "Amani Uwase"): kept as an alias when no
    # other character shares it, so the text finds them.
    firsts = [(c["name"].split() or [""])[0] for c in outline.get("characters") or []]

    def entity(kind, name, quote, profile=None):
        nonlocal canon
        e = canon.entity(name)
        if e:
            return e["id"]
        first = (name.split() or [""])[0]
        aliases = [first] if kind == "character" and first != name and firsts.count(first) == 1 \
            and not canon.entity(first) else []
        row = repo.import_entity(kind, name, quote, ref, profile=profile, aliases=aliases)
        added["entities"] += 1
        canon = repo.load()
        return row["id"]

    def lookup(name):
        e = canon.entity(name)
        if e:
            return e["id"]
        first = _norm(name).split(" ")[0] if _norm(name) else ""
        hits = [x for x in canon.entities if x["kind"] == "character" and _norm(x["name"]).split(" ")[0] == first]
        return hits[0]["id"] if len(hits) == 1 else None

    def fact(f):
        from studio_canon import key_of
        k = key_of(f)
        if k and any(key_of(x) == k and x.get("status") in ("approved", "creator_locked") for x in canon.facts):
            return
        repo.import_fact(f, ref)
        added["facts"] += 1

    for c in outline.get("characters") or []:
        cid = entity("character", c["name"], f"{c['name']}: {c.get('role') or 'character'}")
        for t in _TRAITS:
            if c.get(t):
                fact({"kind": "trait", "subject_id": cid, "attribute": t, "value": c[t][:500],
                      "statement": f"{c['name']} — {t}: {c[t]}"[:600], "source_quote": c[t]})
        if c.get("secret"):
            fact({"kind": "trait", "subject_id": cid, "attribute": "hides", "value": c["secret"][:500],
                  "statement": f"{c['name']} hides: {c['secret']}"[:600], "source_quote": c["secret"]})
    for p in outline.get("places") or []:
        entity("place", p["name"], p.get("description") or p["name"], {"description": p.get("description") or ""})
    for x in outline.get("objects") or []:
        entity("object", x["name"], x.get("description") or x["name"],
               {"description": x.get("description") or "", "why_it_matters": x.get("why_it_matters") or ""})
    canon = repo.load()
    for r in outline.get("relationships") or []:
        a, b = lookup(r["a"]), lookup(r["b"])
        if a and b:
            fact({"kind": "relationship", "subject_id": a, "object_id": b, "attribute": (r.get("side") or "bond")[:80],
                  "value": r["value"][:500], "statement": f"{r['a']} and {r['b']}: {r['value']}"[:600],
                  "source_quote": f"{r['a']} / {r['b']}: {r['value']}"})
    for s in outline.get("secrets") or []:
        sid = entity("secret", s["name"], s.get("truth") or s["name"], {"truth": s.get("truth") or ""})
        canon = repo.load()
        for who in s.get("known_by") or []:
            w = lookup(who)
            if w:
                fact({"kind": "knows", "subject_id": w, "object_id": sid,
                      "statement": f"{who} knows from the start: {s['name']}"[:600], "source_quote": f"known by {who}"})
        if s.get("revealed_in_chapter"):
            repo.import_thread("mystery", f"Reveal: {s['name']} — {s.get('truth') or ''}"[:600], 1,
                               f"secret: {s['name']}", ref, due_chapter=int(s["revealed_in_chapter"]),
                               entities=[i for i in [sid] + [lookup(x) for x in s.get("revealed_to") or []] if i])
            added["threads"] += 1
    for t in outline.get("threads") or []:
        repo.import_thread(t["kind"], t["description"], int(t["planted_chapter"] or 1), t["description"], ref,
                           due_chapter=int(t["payoff_chapter"]) if t.get("payoff_chapter") else None)
        added["threads"] += 1
    return added
