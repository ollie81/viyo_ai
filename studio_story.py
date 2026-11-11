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
    },
    "required": ["title", "logline", "central_conflict", "stakes", "characters", "beats", "ending"],
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


def _continuity(previous: list[dict]) -> str:
    if not previous:
        return ""
    lines = ["THIS IS A SERIES. Earlier episodes (keep every established fact, name, relationship and personality;",
             "never rename or change a character; pay off the open threads):"]
    for ep in previous[-6:]:
        lines.append(f"- {ep.get('title') or 'Episode'}: {ep.get('episode_summary') or ''}".strip())
        if ep.get("open_threads"):
            lines.append("  Open threads: " + "; ".join(ep["open_threads"][:6]))
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


def plan_prompt(b: dict, previous: list[dict]) -> str:
    seconds = int(b.get("target_seconds") or 90)
    scenes = max(3, seconds // 8)
    episode = b.get("kind") in ("episode", "series")
    return f"""You are the head writer of a top short-drama studio. Develop the creator's idea into a story plan.

{_continuity(previous)}CREATOR'S IDEA (keep its core: the same people, situation and promise; make it stronger, not different):
Title: {b.get('title')}
Idea: {b.get('logline') or '(none given: invent a gripping original story in this genre)'}
Genre: {b.get('genre') or 'drama'}
Format: {b.get('kind')} of about {seconds} seconds, {_format(b)}, about {scenes} scenes.
Language of the final dialogue: {b.get('language') or 'English'}.

{CRAFT}

Plan {scenes}-{scenes + 2} beats whose seconds add up to about {seconds}.
The FIRST beat's purpose is "hook" and the LAST beat's purpose is "end_hook".
{'The end hook is a cliffhanger.' if episode else 'The story resolves emotionally, and its final moment is still a hook: a last twist, revelation, line or image that hits hard.'}
Fictional characters only: no real people, brands or copyrighted characters.
Write the plan in English (character names as they would be in the story)."""


def draft_prompt(b: dict, plan: dict, previous: list[dict]) -> str:
    return f"""Write the screenplay for this story plan.

{_continuity(previous)}STORY PLAN:
{json.dumps(plan, ensure_ascii=False, indent=1)}

{CRAFT}

Rules:
- Standard screenplay format: scene headings (INT./EXT. PLACE - TIME), short vivid action lines, CHARACTER NAMES in caps before dialogue.
- Follow the beats in order; about {int(b.get('target_seconds') or 90)} seconds on screen in total ({_format(b)}).
- Write all dialogue in {b.get('language') or 'English'}, natural for the characters' age and world. Action lines may be in the same language.
- Each character speaks the way the plan says. Keep lines short.
- Return only the screenplay text."""


def doctor_prompt(b: dict, plan: dict, draft: str) -> str:
    return f"""You are a demanding script doctor. Here is a story plan and a first-draft screenplay.

STORY PLAN:
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


def develop(text_fn, json_fn, brief: dict, previous: Optional[list[dict]] = None):
    """Runs the three steps. `text_fn(prompt) -> (text, in_tokens, out_tokens)`,
    `json_fn(prompt, schema) -> (dict, in_tokens, out_tokens)`.
    Returns (screenplay, bible, total_in, total_out)."""
    previous = previous or []
    plan, i1, o1 = json_fn(plan_prompt(brief, previous), PLAN_SCHEMA)
    if not plan.get("beats") or not plan.get("characters"):
        raise ValueError("the story plan came back empty")
    if not hooks_ok(plan):
        # Every story starts and ends with a hook: send the plan back once.
        fix = (plan_prompt(brief, previous) + "\n\nYOUR PREVIOUS PLAN BROKE THE RULE. It must start with a 'hook' beat "
               "and end with an 'end_hook' beat. Here it is; return it fixed:\n" + json.dumps(plan, ensure_ascii=False))
        fixed, i0, o0 = json_fn(fix, PLAN_SCHEMA)
        i1, o1 = i1 + i0, o1 + o0
        if fixed.get("beats") and fixed.get("characters"):
            plan = fixed
    plan["hooks_planned"] = hooks_ok(plan)
    draft, i2, o2 = text_fn(draft_prompt(brief, plan, previous))
    draft = (draft or "").strip()
    if not draft:
        raise ValueError("the screenplay draft came back empty")
    try:
        doc, i3, o3 = json_fn(doctor_prompt(brief, plan, draft), DOCTOR_SCHEMA)
    except Exception:
        doc, i3, o3 = {}, 0, 0
    final = (doc.get("screenplay") or "").strip()
    # A rewrite much shorter than the draft has probably been cut off: keep the draft.
    if len(final) < 0.6 * len(draft):
        final, doc = draft, {**doc, "fixes": doc.get("fixes") or [], "rewrite_used": False}
    else:
        doc["rewrite_used"] = True
    bible = {**plan, "scores": doc.get("scores") or {}, "fixes": doc.get("fixes") or [],
             "opening_hook": doc.get("opening_hook") or "", "ending_hook": doc.get("ending_hook") or "",
             "episode_summary": doc.get("episode_summary") or "", "open_threads": doc.get("open_threads") or [],
             "rewrite_used": doc.get("rewrite_used", False)}
    return final, bible, i1 + i2 + i3, o1 + o2 + o3


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
