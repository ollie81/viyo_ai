"""Viyo Studio's story engine: a writers' room in three steps, so a simple
idea becomes a story people remember instead of a generic script.

1. Story plan (JSON): sharpens the creator's idea into a premise with a
   real conflict, builds characters with a want, a need, a flaw and a
   secret, and plans the beats: a hook in the first seconds, a reversal
   every few scenes, a twist and an ending that lands (or a cliffhanger).
   For a series it reads the earlier episodes and keeps everyone the same.
2. Screenplay draft from the plan.
3. Script doctor (JSON): scores the draft against a checklist, says what is
   weak, and rewrites the whole screenplay to fix it. Also writes the
   episode summary and the open storylines the next episode must pick up.

The creator's idea is always kept: the engine strengthens it, never swaps
it for a different story.
"""
from __future__ import annotations

import json
from typing import Optional

OPENINGS = ("shocking_discovery", "unexpected_arrival", "betrayal", "confrontation", "mysterious_event",
            "dangerous_secret", "emotional_revelation", "unexpected_decision", "threat", "in_the_middle_of_action")
ENDINGS = ("secret_exposed", "betrayal_discovered", "unexpected_person", "disturbing_news", "relationship_shift",
           "dangerous_decision", "mystery_deepens", "reversal", "threat_arrives", "final_twist", "emotional_payoff")

# What works where: the techniques a genre's audience responds to, and what to avoid.
GENRES = {
    "thriller": "Open on danger or a discovery that puts someone at risk. Tighten with a ticking clock, a hidden threat "
                "and information the viewer has before the hero. End on a threat arriving, a dangerous decision or a reversal.",
    "mystery": "Open on a mysterious event or clue. Answer one question per beat and raise a bigger one. Plant clues "
               "fairly. End on a mystery deepening or a secret exposed, never on random shock.",
    "romance": "Open on longing, an unexpected meeting or a relationship at a breaking point. Build with closeness that "
               "is interrupted, a secret between them, a rival. End on a relationship shift, a confession or a betrayal discovered.",
    "family": "Open on a family secret surfacing or a confrontation at home. Build with loyalty tested, old wounds and who "
              "knows what. End on a secret exposed, a disturbing truth about a parent or sibling, or an unexpected person.",
    "revenge": "Open on the wrong done or the avenger's first move. Build with a plan that tightens and a cost that grows. "
               "End on a reversal, the target discovering the plan, or a dangerous decision.",
    "horror": "Open on something wrong in an ordinary place. Build dread by showing less, not more. End on the threat "
              "arriving or a disturbing discovery.",
    "comedy": "Open on an embarrassing situation or a lie that will snowball. Build with escalating complications. End "
              "on the worst person walking in, or a reversal that makes it worse.",
    "drama": "Open on an emotional revelation or a decision that changes a life. Build with conflict between people who "
             "want different things. End on a relationship shift, disturbing news or a decision the viewer fears.",
}
_GENRE_WORDS = {"thriller": ("thriller", "crime", "action", "suspense", "spy", "heist"),
                "mystery": ("mystery", "detective", "whodunit", "investigation"),
                "romance": ("romance", "romantic", "love", "dating"),
                "family": ("family", "sibling", "mother", "father", "inheritance"),
                "revenge": ("revenge", "vengeance", "betrayal"),
                "horror": ("horror", "ghost", "supernatural", "scary"),
                "comedy": ("comedy", "comic", "funny", "romcom", "rom-com")}


def genre_craft(genre: str, logline: str = "") -> str:
    """The storytelling techniques that fit this genre (and the idea's own words)."""
    text = f"{genre} {logline}".lower()
    picked = [g for g, words in _GENRE_WORDS.items() if any(w in text for w in words)][:2] or ["drama"]
    return "GENRE CRAFT (" + ", ".join(picked) + "):\n" + "\n".join(f"- {GENRES[g]}" for g in picked)


PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "description": "A short, memorable title (keep the creator's if it is good)."},
        "logline": {"type": "string", "description": "One sentence: who wants what, what stands in the way, what is at stake."},
        "premise_twist": {"type": "string", "description": "What makes this story unexpected or high-concept."},
        "theme": {"type": "string"},
        "tone": {"type": "string"},
        "central_conflict": {"type": "string"},
        "stakes": {"type": "string", "description": "What is lost if the hero fails. Personal and concrete."},
        "characters": {"type": "array", "items": {"type": "object", "properties": {
            "name": {"type": "string"},
            "age": {"type": "string", "description": "Age or age range, e.g. '7', 'late 20s', '60s'."},
            "role": {"type": "string", "enum": ["lead", "supporting", "minor"]},
            "want": {"type": "string"}, "need": {"type": "string"},
            "flaw": {"type": "string"}, "secret": {"type": "string"},
            "speech": {"type": "string", "description": "How they talk: rhythm, words they use, what they avoid saying."},
            "look": {"type": "string", "description": "Face, hair, build, clothes. No real people."},
            "arc": {"type": "string"},
            "history": {"type": "string"}, "fears_losing": {"type": "string"},
            "inner_conflict": {"type": "string"}, "outer_conflict": {"type": "string"},
            "vulnerability": {"type": "string"},
        }, "required": ["name", "age", "role", "want", "flaw", "speech", "look"]}},
        "beats": {"type": "array", "items": {"type": "object", "properties": {
            "purpose": {"type": "string", "enum": ["hook", "setup", "inciting", "escalation", "reversal", "twist",
                                                   "dark_moment", "climax", "resolution", "end_hook"],
                        "description": "The FIRST beat must be 'hook' and the LAST beat must be 'end_hook'."},
            "what_happens": {"type": "string"},
            "emotional_turn": {"type": "string", "description": "How the feeling changes, e.g. hope -> dread."},
            "seconds": {"type": "number"},
        }, "required": ["purpose", "what_happens", "emotional_turn", "seconds"]}},
        "ending": {"type": "string", "description": "The final image and why it hits (or the cliffhanger question)."},
        "visual_motifs": {"type": "array", "items": {"type": "string"}},
        "hook_plan": {"type": "object", "properties": {
            "opening_technique": {"type": "string", "enum": list(OPENINGS)},
            "opening_moment": {"type": "string", "description": "Exactly what we see and hear in the first 3-5 seconds."},
            "opening_why": {"type": "string", "description": "Why a viewer can't scroll away, tied to THIS story."},
            "ending_technique": {"type": "string", "enum": list(ENDINGS)},
            "ending_moment": {"type": "string", "description": "The final moment, exactly."},
            "viewer_question": {"type": "string", "description": "The question the viewer is left needing answered."},
        }, "required": ["opening_technique", "opening_moment", "opening_why", "ending_technique", "ending_moment",
                        "viewer_question"]},
        "continues_from": {"type": "string", "description": "Series only: how the first beat picks up the previous "
                                                            "episode's final moment. Empty for a first episode."},
        "questions": {"type": "array", "description": "The questions this episode makes the viewer ask, in order.",
                      "items": {"type": "object", "properties": {
                          "question": {"type": "string"}, "raised_in_beat": {"type": "integer"},
                          "answered_in_beat": {"type": "integer", "description": "0 if it stays open for later."}},
                          "required": ["question", "raised_in_beat", "answered_in_beat"]}},
        "knowledge": {"type": "array", "description": "At the END of this episode: who knows what, who hides what.",
                      "items": {"type": "object", "properties": {
                          "character": {"type": "string"}, "knows": {"type": "string"}, "hides": {"type": "string"},
                          "does_not_know": {"type": "string"}}, "required": ["character", "knows"]}},
        "relationships": {"type": "array", "items": {"type": "object", "properties": {
            "between": {"type": "string"}, "state": {"type": "string", "description": "At the end of this episode."},
            "change": {"type": "string", "description": "What changed in this episode, or empty."}},
            "required": ["between", "state"]}},
        "revelations": {"type": "array", "items": {"type": "string"},
                        "description": "What the VIEWER learns in this episode (so later episodes never re-explain it)."},
        "next_episode": {"type": "array", "items": {"type": "string"},
                         "description": "Planned developments for the next episode (episodes and series)."},
    },
    "required": ["title", "logline", "central_conflict", "stakes", "characters", "beats", "ending", "hook_plan",
                 "questions", "knowledge"],
}

DOCTOR_SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {"type": "object", "properties": {
            "hook": {"type": "integer"}, "ending_hook": {"type": "integer"},
            "conflict": {"type": "integer"}, "characters": {"type": "integer"},
            "dialogue": {"type": "integer"}, "twist": {"type": "integer"}, "ending": {"type": "integer"},
            "visuals": {"type": "integer"}}, "description": "1-10 for the DRAFT, before your rewrite."},
        "fixes": {"type": "array", "items": {"type": "string"}, "description": "The most important changes you made."},
        "screenplay": {"type": "string", "description": "The full rewritten screenplay."},
        "opening_hook": {"type": "string", "description": "Quote the opening hook of YOUR rewrite (first 3-5 seconds)."},
        "ending_hook": {"type": "string", "description": "Quote the ending hook of YOUR rewrite (the final moment)."},
        "episode_summary": {"type": "string", "description": "3-4 sentences: what happens, for the next episode's writer."},
        "open_threads": {"type": "array", "items": {"type": "string"},
                         "description": "Unresolved questions and promises the next episode must pay off."},
    },
    "required": ["scores", "fixes", "screenplay", "opening_hook", "ending_hook", "episode_summary", "open_threads"],
}

HOOK_RULE = """NON-NEGOTIABLE: every story starts with a hook and ends with a hook.
- Opening hook (first 3-5 seconds): a shocking line, a startling image, a mystery or a threat that makes leaving impossible.
- Ending hook (the final moment): for an episode, a cliffhanger question the viewer must have answered; for a film or
  short, a final twist, revelation, line or image that hits hard and stays with the viewer after it ends.
- Never open slowly (no establishing calm, no introductions first) and never end flat (no fading out, no summary)."""

CRAFT = HOOK_RULE + """

What makes short drama unforgettable:
- Open in the middle of trouble: the first 3-5 seconds must make the viewer need to know what happens.
- Every scene has someone who wants something and someone or something in the way. No scene only gives information.
- Turn the story every 15-20 seconds: a reveal, a reversal, a choice with a cost.
- Characters are specific: a want, a flaw, a secret, a way of speaking only they have.
- Dialogue is short and has subtext: people rarely say exactly what they feel. No speeches, no explaining the plot.
- Plant something early that pays off later. The twist must feel surprising and, looking back, inevitable.
- Show emotion in actions and images, not in words.
- End on a strong image or line. An episode ends on a question the viewer can't leave unanswered.
- Avoid cliches: no waking from a dream, no evil twin, no "it was all a misunderstanding".

Mature themes are welcome when the story needs them: violence, crime, death, betrayal, adult romance and desire.
Keep sex scenes between adults implied (a kiss, then cut away), never explicit. Characters under 18 are never
in romantic or sexual situations. No real-world how-to for weapons or crimes."""


def _format(b: dict) -> str:
    return "vertical 9:16 phone video" if b.get("aspect_ratio") == "9:16" else "landscape 16:9 film"


def _record_of(ep: dict) -> dict:
    """An earlier episode's continuity record (older bibles have none: built from what they kept)."""
    rec = dict(ep.get("episode_record") or {})
    rec.setdefault("ending", ep.get("ending_hook") or ep.get("ending") or "")
    rec.setdefault("open_questions", ep.get("open_threads") or [])
    rec.setdefault("end_technique", (ep.get("hook_plan") or {}).get("ending_technique") or "")
    if ep.get("_screenplay_end"):
        rec["final_scene"] = ep["_screenplay_end"]       # the script as it is now (the creator may have edited it)
    return rec


def _continuity(previous: list[dict]) -> str:
    if not previous:
        return ""
    lines = ["THIS IS A SERIES. Earlier episodes, oldest first (keep every established fact, name, relationship and",
             "personality; never rename or change a character):"]
    for n, ep in enumerate(previous[-6:], start=max(1, len(previous) - 5)):
        rec = _record_of(ep)
        lines.append(f"- Episode {n} \"{ep.get('title') or ''}\": {ep.get('episode_summary') or ''}".strip())
        if rec.get("revelations"):
            lines.append("  The viewer already learned: " + "; ".join(rec["revelations"][:6]))
        if rec.get("end_technique"):
            lines.append(f"  Its ending used: {rec['end_technique'].replace('_', ' ')}")
    last = _record_of(previous[-1])
    lines.append("")
    lines.append("THE PREVIOUS EPISODE ENDED EXACTLY LIKE THIS (this episode starts from here):")
    lines.append((last.get("final_scene") or last.get("ending") or "(not recorded)")[:1500])
    if last.get("open_questions"):
        lines.append("Questions the viewer is waiting to have answered: " + "; ".join(map(str, last["open_questions"][:8])))
    knowledge = last.get("knowledge") or []
    if knowledge:
        lines.append("Who knows what at the end of the previous episode (a character only learns more ON SCREEN):")
        for k in knowledge[:10]:
            lines.append(f"- {k.get('character')}: knows {k.get('knows') or '-'}"
                         + (f"; hides {k['hides']}" if k.get("hides") else "")
                         + (f"; doesn't know {k['does_not_know']}" if k.get("does_not_know") else ""))
    for r in (last.get("relationships") or [])[:8]:
        lines.append(f"- Relationship {r.get('between')}: {r.get('state')}")
    if last.get("next_episode"):
        lines.append("Planned for this episode: " + "; ".join(map(str, last["next_episode"][:6])))
    used = [(_record_of(e).get("end_technique") or "") for e in previous[-2:]]
    used = [u for u in used if u]
    lines.append("SERIES RULES: the first beat picks up that final moment (its consequence or the very next moment); "
                 "don't re-explain what the viewer already saw or learned; pay off at least one waiting question and "
                 "raise at least one new one; the larger story must move forward."
                 + (f" Don't end this episode with the same kind of cliffhanger as before ({', '.join(u.replace('_', ' ') for u in used)})."
                    if used else ""))
    cast = {}
    for ep in previous:
        for c in ep.get("characters") or []:
            cast[c.get("name")] = c
    if cast:
        lines.append("Established characters (keep them exactly like this):")
        for c in list(cast.values())[:10]:
            lines.append(f"- {c.get('name')} ({c.get('age', '')}): wants {c.get('want', '')}; flaw: {c.get('flaw', '')}; "
                         f"speaks: {c.get('speech', '')}; looks: {c.get('look', '')}")
    return "\n".join(lines) + "\n\n"


IMMERSION = """IMMERSION (use the world, don't decorate with it):
- People speak, greet, show respect, hide fear and argue the way people of THIS place and time do; names and forms
  of address come from the world. Wealth, family, rank and tradition shape what each person can say and do.
- Show the place through what characters touch, hear and do (work, food, weather, rituals), and through silence,
  looks and gestures, not through explanation. Every detail serves the scene.
- Characters are real people: a history, something they're afraid of losing, a secret, an inner conflict. Their
  choices have consequences; they make mistakes and misunderstand each other believably.
- Never present invented or uncertain things as historical fact; nothing that didn't exist yet appears.

"""


def plan_prompt(b: dict, previous: list[dict]) -> str:
    seconds = int(b.get("target_seconds") or 90)
    scenes = max(3, seconds // 8)
    episode = b.get("kind") in ("episode", "series")
    import studio_saga
    import studio_world
    world = studio_world.block(b.get("world"), "story")
    saga = studio_saga.position(b.get("saga"), len(previous) + 1)
    return f"""You are the head writer of a top short-drama studio. Develop the creator's idea into a story plan.

{world + IMMERSION if world else ''}{saga}{_continuity(previous)}CREATOR'S IDEA (keep its core: the same people, situation and promise; make it stronger, not different):
Title: {b.get('title')}
Idea: {b.get('logline') or '(none given: invent a gripping original story in this genre)'}
Genre: {b.get('genre') or 'drama'}
Format: {b.get('kind')} of about {seconds} seconds, {_format(b)}, about {scenes} scenes.
Language of the final dialogue: {b.get('language') or 'English'}.

{CRAFT}

{genre_craft(b.get('genre') or '', b.get('logline') or '')}

HOOKS THAT FIT THIS STORY (don't force a random shock):
- Opening: pick the technique that grows from THIS idea (a discovery, an arrival, a betrayal, a confrontation, a
  mystery, a dangerous secret, an emotional revelation, a decision...). The first line or image already belongs to the
  conflict. No greetings, no "my name is", no background explained, no calm before the trouble.
- Suspense: every beat turns the feeling (emotional_turn) and either raises a question or answers one (questions:
  raised_in_beat / answered_in_beat). Answer small questions to earn bigger ones. No two beats do the same job; no
  filler conversation; each scene changes something.
- Ending: the final moment grows out of what just happened and leaves one clear question the viewer needs answered
  (viewer_question). It must feel inevitable, not bolted on.
- knowledge: at the end, who knows what and who hides what (later episodes depend on it).

Plan {scenes}-{scenes + 2} beats whose seconds add up to about {seconds}.
The FIRST beat's purpose is "hook" and the LAST beat's purpose is "end_hook".
{'The end hook is a cliffhanger.' if episode else 'The story resolves emotionally, and its final moment is still a hook: a last twist, revelation, line or image that hits hard.'}
Fictional characters only: no real people, brands or copyrighted characters.
Write the plan in English (character names as they would be in the story)."""


def draft_prompt(b: dict, plan: dict, previous: list[dict]) -> str:
    import studio_world
    world = studio_world.block(b.get("world"), "story")
    return f"""Write the screenplay for this story plan.

{world + IMMERSION if world else ''}{_continuity(previous)}STORY PLAN:
{json.dumps(plan, ensure_ascii=False, indent=1)}

{CRAFT}

Rules:
- Standard screenplay format: scene headings (INT./EXT. PLACE - TIME), short vivid action lines, CHARACTER NAMES in caps before dialogue.
- Follow the beats in order; about {int(b.get('target_seconds') or 90)} seconds on screen in total ({_format(b)}).
- Write all dialogue in {b.get('language') or 'English'}, natural for the characters' age and world. Action lines may be in the same language.
- Each character speaks the way the plan says. Keep lines short.
- Return only the screenplay text."""


def doctor_prompt(b: dict, plan: dict, draft: str, previous: Optional[list[dict]] = None) -> str:
    import studio_world
    world = studio_world.block(b.get("world"), "story")
    return f"""You are a demanding script doctor. Here is a story plan and a first-draft screenplay.

{world + IMMERSION if world else ''}{_continuity(previous or [])}STORY PLAN:
{json.dumps(plan, ensure_ascii=False, indent=1)}

DRAFT:
{draft}

{CRAFT}

1. Score the DRAFT 1-10 on: hook (opening), ending_hook, conflict, characters, dialogue, twist, ending, visuals.
2. Rewrite the WHOLE screenplay to fix every weakness. The opening hook and the ending hook must each be a 9 or 10:
   if either is weaker, rewrite it until it is. Also: more conflict and subtext, more memorable lines, a twist that is
   set up. Keep the creator's story, the characters' names
   and the plan's beats. Keep the same length (about {int(b.get('target_seconds') or 90)} seconds) and format.
   Dialogue stays in {b.get('language') or 'English'}.
3. Quote the opening hook and the ending hook of your rewrite.
4. Write the episode summary and the open threads for the next episode's writer (in English)."""


def hooks_ok(plan: dict) -> bool:
    beats = plan.get("beats") or []
    return bool(beats) and beats[0].get("purpose") == "hook" and beats[-1].get("purpose") in ("end_hook", "cliffhanger")


def plan_problems(plan: dict, brief: dict, previous: list[dict]) -> list[str]:
    """What the plan gets wrong that a writer would send back."""
    out = []
    if not hooks_ok(plan):
        out.append("It must start with a 'hook' beat and end with an 'end_hook' beat.")
    hp = plan.get("hook_plan") or {}
    if not (hp.get("opening_moment") or "").strip() or not (hp.get("ending_moment") or "").strip():
        out.append("hook_plan must say exactly what the opening and the final moment are.")
    if previous:
        if not (plan.get("continues_from") or "").strip():
            out.append("continues_from must say how the first beat picks up the previous episode's final moment.")
        last = (_record_of(previous[-1]).get("end_technique") or "")
        if last and hp.get("ending_technique") == last:
            out.append(f"The previous episode already ended with '{last.replace('_', ' ')}'. Choose a different kind "
                       "of ending that still grows out of this story.")
    if brief.get("kind") in ("episode", "series"):
        if not any(not q.get("answered_in_beat") for q in plan.get("questions") or []):
            out.append("An episode must leave at least one question open for the next episode (answered_in_beat 0).")
    beats = plan.get("beats") or []
    jobs = [(b.get("purpose"), (b.get("emotional_turn") or "").lower()) for b in beats]
    if len(jobs) != len(set(jobs)) and len(beats) > 2:
        out.append("Two beats do the same job with the same feeling: make every beat change something.")
    return out


from studio_story_check import CRITERIA, THRESHOLD  # noqa: E402,F401  (the strict check lives there)


def structure_checks(script: str, brief: dict, previous: list[dict]) -> list[dict]:
    """Checks that need no AI: a slow opening, repeated lines, a flat last moment."""
    import re
    out = []
    lines = screenplay_lines(script)
    first = lines[0][1] if lines else ""
    if re.match(r"^\W*(hi|hello|hey|good (morning|evening|afternoon)|my name is|welcome)\b", first, re.I):
        out.append({"area": "opening_hook", "problem": f"The first line is a greeting: \"{first[:80]}\".",
                    "fix": "Open inside the conflict: cut the greeting and start on the moment of trouble."})
    seen = {}
    for who, text in lines:
        key = " ".join(_words(text))
        if len(key.split()) >= 4:
            seen[key] = seen.get(key, 0) + 1
    rep = [k for k, n in seen.items() if n > 1]
    if rep:
        out.append({"area": "dialogue", "problem": f"A line is said more than once: \"{rep[0][:80]}\".",
                    "fix": "Say it once; give the repeat a new line that moves the scene."})
    if previous:
        # The first scene should share someone with the previous episode's last scene.
        # The first scene should share someone or something with the previous episode's last scene:
        # its speakers, the names in it, or its distinctive words (a policeman, the ID card...).
        end = _record_of(previous[-1]).get("final_scene") or ""
        import studio_story_check as sc
        who = {w.split(" ")[0].lower() for w, _t in screenplay_lines(end)}
        names = {w.lower() for w in re.findall(r"\b[A-Z][a-z]{2,}\b", end)} - {"the", "she", "his", "her", "then", "int", "ext"}
        things = {w for w in re.findall(r"[a-z]{6,}", end.lower())} - {"night", "morning", "evening", "suddenly", "through",
                                                                      "corridor", "outside", "inside", "behind", "toward"}
        first = sc.sections(script)[0].lower() if script else ""
        shared = [n for n in who | names | things if re.search(rf"\b{re.escape(n)}\b", first)]
        if (who or names or things) and not shared:
            out.append({"area": "continuity", "problem": "The first scene shares nobody and nothing with the previous "
                                                         f"episode's last scene ({', '.join(sorted(who | names))[:120]}).",
                        "fix": "Open on the people and the moment the previous episode ended on."})
    if brief.get("world"):
        import studio_world
        out += [{"area": c["area"], "problem": c["problem"], "fix": c["fix"]} for c in studio_world.check_text(script, brief["world"])]
    tail = (script or "").strip().splitlines()[-1:] or [""]
    if re.search(r"\b(the end|fade out|fades to black|to be continued)\b", tail[0], re.I) and brief.get("kind") in ("episode", "series"):
        out.append({"area": "ending_hook", "problem": "It ends on a fade-out instead of a moment.",
                    "fix": "End on the final image or line that leaves the viewer's question open."})
    return out


def _last_section(script: str, n: int = 900) -> str:
    parts = [p for p in __import__("re").split(r"\n(?=\s*(?:INT\.|EXT\.|INT/EXT|EXT/INT|I/E))", script or "") if p.strip()]
    return (parts[-1] if parts else script or "").strip()[-n:]


def episode_record(plan: dict, script: str, doc: dict, ev: dict) -> dict:
    """What the next episode must know: how this one ended, what's still open,
    who knows what, relationships, what the viewer learned, what's planned."""
    hp = plan.get("hook_plan") or {}
    open_q = [q["question"] for q in plan.get("questions") or [] if isinstance(q, dict) and not q.get("answered_in_beat")]
    return {"ending": doc.get("ending_hook") or hp.get("ending_moment") or plan.get("ending") or "",
            "final_scene": _last_section(script), "end_technique": hp.get("ending_technique") or "",
            "open_questions": list(dict.fromkeys(open_q + list(doc.get("open_threads") or []) + ev.get("viewer_questions", [])))[:10],
            "knowledge": plan.get("knowledge") or [], "relationships": plan.get("relationships") or [],
            "revelations": plan.get("revelations") or [], "next_episode": plan.get("next_episode") or []}


def develop(text_fn, json_fn, brief: dict, previous: Optional[list[dict]] = None):
    """Plan -> draft -> script doctor -> evaluation (-> one targeted revision
    and a second evaluation if it's weak). `text_fn(prompt) -> (text, in, out)`,
    `json_fn(prompt, schema) -> (dict, in, out)`. Returns (screenplay, bible, in, out)."""
    previous = previous or []
    tin = tout = 0

    def j(prompt, schema):
        nonlocal tin, tout
        d, i, o = json_fn(prompt, schema)
        tin, tout = tin + i, tout + o
        return d

    def t(prompt):
        nonlocal tin, tout
        d, i, o = text_fn(prompt)
        tin, tout = tin + i, tout + o
        return (d or "").strip()

    plan = j(plan_prompt(brief, previous), PLAN_SCHEMA)
    if not plan.get("beats") or not plan.get("characters"):
        raise ValueError("the story plan came back empty")
    problems = plan_problems(plan, brief, previous)
    if problems:
        # Sent back once, with every problem named.
        fix = (plan_prompt(brief, previous) + "\n\nYOUR PREVIOUS PLAN HAS THESE PROBLEMS:\n" +
               "\n".join(f"- {x}" for x in problems) + "\nHere it is; return it fixed:\n" + json.dumps(plan, ensure_ascii=False))
        fixed = j(fix, PLAN_SCHEMA)
        if fixed.get("beats") and fixed.get("characters"):
            plan = fixed
    plan["hooks_planned"] = hooks_ok(plan)
    plan["plan_problems"] = plan_problems(plan, brief, previous)
    draft = t(draft_prompt(brief, plan, previous))
    if not draft:
        raise ValueError("the screenplay draft came back empty")
    try:
        doc = j(doctor_prompt(brief, plan, draft, previous), DOCTOR_SCHEMA)
    except Exception:
        doc = {}
    final = (doc.get("screenplay") or "").strip()
    # A rewrite much shorter than the draft has probably been cut off: keep the draft.
    if len(final) < 0.6 * len(draft):
        final, doc = draft, {**doc, "fixes": doc.get("fixes") or [], "rewrite_used": False}
    else:
        doc["rewrite_used"] = True

    # The FINAL screenplay is judged; only quotes found in it count (studio_story_check).
    import studio_story_check as sc
    import studio_world
    cont = _continuity(previous)
    wtext = studio_world.block(brief.get("world"), "story")
    try:
        evaluation = sc.check(j, brief, plan, final, cont, bool(previous), structure_checks(final, brief, previous),
                              world_text=wtext)
        final, evaluation = sc.improve(j, brief, plan, final, evaluation, cont, bool(previous),
                                       lambda text: structure_checks(text, brief, previous), HOOK_RULE, world_text=wtext)
    except Exception as e:
        evaluation = {"status": "incomplete", "complete": False, "categories": {}, "unverified": list(sc.required(bool(previous))),
                      "below_bar": [], "weak": [], "checks": [], "viewer_questions": [], "script_sha": sc.script_sha(final),
                      "world": bool(wtext),
                      "error": f"The story check couldn't run ({str(e)[:120]}); the screenplay is unchanged.",
                      "summary": "Incomplete: the story check couldn't run. It isn't a pass."}
    bible = {**plan, "draft_scores": doc.get("scores") or {}, "scores": doc.get("scores") or {},
             "fixes": doc.get("fixes") or [],
             "opening_hook": doc.get("opening_hook") or "", "ending_hook": doc.get("ending_hook") or "",
             "episode_summary": doc.get("episode_summary") or "", "open_threads": doc.get("open_threads") or [],
             "rewrite_used": doc.get("rewrite_used", False), "evaluation": evaluation}
    bible["episode_record"] = episode_record(plan, final, doc, evaluation)
    return final, bible, tin, tout


# ------------------------------------------------------------------ dialogue check
import re as _re

_CUE = _re.compile(r"^\s*([A-Z][A-Z0-9 .'\-]{0,40}?)\s*(\((?:V\.O\.|O\.S\.|O\.C\.|CONT'D|CONT’D|cont'd)\))?\s*$")
_NOT_CUE = ("INT.", "EXT.", "INT/EXT", "EXT/INT", "I/E", "CUT TO", "FADE", "DISSOLVE", "SMASH CUT", "THE END", "TITLE")


def screenplay_lines(script: str) -> list[tuple[str, str]]:
    """(SPEAKER, line) for every line of dialogue in a standard screenplay."""
    out, speaker, buf = [], None, []

    def flush():
        if speaker and buf:
            out.append((speaker, " ".join(buf)))

    for raw in (script or "").splitlines():
        line = raw.strip()
        m = _CUE.match(line) if line else None
        if not line:
            flush()
            speaker, buf = None, []
        elif m and not line.upper().startswith(_NOT_CUE) and any(ch.isalpha() for ch in m.group(1)):
            flush()
            speaker, buf = m.group(1).strip().title(), []
        elif line.upper().startswith(_NOT_CUE[:5]):
            flush()                                   # a scene heading ends the speech
            speaker, buf = None, []
        elif speaker and not (line.startswith("(") and line.endswith(")")):
            buf.append(line)
    flush()
    return out


def _words(text: str) -> list[str]:
    return _re.findall(r"[a-z0-9']+", (text or "").lower())


def _close(a: str, b: str) -> bool:
    wa, wb = set(_words(a)), set(_words(b))
    return bool(wa) and bool(wb) and len(wa & wb) / max(len(wa), len(wb)) >= 0.6


def dialogue_check(script: str, scenes: list[dict]) -> dict:
    """Possible dialogue problems after the screenplay was broken into scenes:
    lines of the screenplay no scene says (forgotten), and lines said in more
    than one place (repeated). Fuzzy on purpose: a slightly reworded line counts
    as said. Returns {"forgotten": [(speaker, text)], "repeated": [(speaker, text, [scene numbers])]}."""
    wanted = screenplay_lines(script)
    said = [(n, (d.get("character") or "").strip(), d.get("text") or "")
            for n, s in enumerate(scenes, start=1) for d in s.get("dialogue") or [] if (d.get("text") or "").strip()]
    used = set()
    forgotten = []
    for who, text in wanted:
        hit = next((k for k, (_n, w, t) in enumerate(said) if k not in used and _close(text, t)
                    and (not w or w.split(" ")[0].lower() == who.split(" ")[0].lower())), None)
        if hit is None:
            forgotten.append((who, text))
        else:
            used.add(hit)
    repeated, seen = [], set()
    for k, (n, w, t) in enumerate(said):
        if k in seen:
            continue
        twins = [j for j in range(k + 1, len(said)) if said[j][1].lower() == w.lower() and _close(t, said[j][2])]
        written = sum(1 for who, text in wanted if _close(text, t))
        if twins and len(twins) + 1 > max(1, written):
            seen.update(twins)
            repeated.append((w, t, [n] + [said[j][0] for j in twins]))
    return {"forgotten": forgotten if wanted else [], "repeated": repeated}


def dialogue_lines(check: dict) -> list[str]:
    """For the assistant's project facts."""
    f, r = check.get("forgotten") or [], check.get("repeated") or []
    if not f and not r:
        return ["DIALOGUE CHECK (screenplay vs scenes): every screenplay line is said once in the scenes."]
    out = ["DIALOGUE CHECK (screenplay vs scenes, possible problems: ask whether they are intended):"]
    for who, text in f[:6]:
        out.append(f"- Possibly forgotten: {who}: \"{text[:160]}\" is in the screenplay but no scene says it.")
    if len(f) > 6:
        out.append(f"- ...and {len(f) - 6} more screenplay lines no scene says.")
    for who, text, where in r[:6]:
        out.append(f"- Possibly repeated: {who}: \"{text[:160]}\" is said in scenes {', '.join(map(str, where))}.")
    return out
