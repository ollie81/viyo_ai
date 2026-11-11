"""The Studio assistant: one conversational filmmaking partner per project.

It plays five parts in one voice: writer, director, continuity supervisor,
production manager and creative partner. Every reply sees the real project,
retrieved for the question: the story plan, the screenplay (complete when it
fits, otherwise an index of every section plus the full text of the sections
that matter), characters with identities, voices, reference sheets and
outfits, places and objects, every scene with its continuity record, other
episodes of the series, production status and job errors, the continuity
check, the project's long-term memory (studio_memory), what the conversation
is currently about (for "her", "that scene", "the next scene"), a running
summary of older messages and the last few messages; never the whole history.

The assistant can't change anything by itself. When a change would help it
proposes it as a card (add a character, rewrite a scene, ...). Free changes
are applied by the server through the same code as the Studio's own
buttons, only when the creator taps Apply; paid steps go through the
Studio's normal price check. A suggestion the creator hasn't applied is
never treated as a fact.

The system prompt stays on the server; the app only sees replies.
"""
from __future__ import annotations

import json
import os
import re
from typing import Optional

RECENT = 12            # messages sent word for word
SUMMARIZE_AFTER = 24   # older messages beyond this are folded into the summary
MAX_PROPOSALS = 3

# Proposal types and what they do. "free" ones change saved project text and
# are applied by the server; "paid" ones open the Studio's price check.
ACTIONS = {
    "update_story": "free",      # data: title, logline, genre, kind, target_seconds, visual_style, language
    "write_screenplay": "free",  # data: script (the full screenplay text the assistant wrote in chat)
    "add_character": "free",     # data: name, description
    "update_character": "free",  # data: character, description
    "add_outfit": "free",        # data: character, name, clothing, shoes, accessories, makeup
    "add_place": "free",         # data: name, description
    "add_object": "free",        # data: name, description, owner
    "rewrite_scene": "free",     # data: scene (number), heading, visual, dialogue, start_state, end_state, transition
    "set_scene_outfit": "free",  # data: scene, character, outfit
    "develop_story": "paid",     # the writers' room writes the full screenplay (job kind "script")
    "break_into_scenes": "paid", # job kind "breakdown"
    "find_places_objects": "paid",  # job kind "elements"
    "make_cover": "paid",        # job kind "cover"
    "make_film": "paid",         # one-click production
    "remake_picture": "paid",    # data: scene (number): a new picture for that one scene (job kind "image")
    "remake_video": "paid",      # data: scene (number): a new video for that one scene (job kind "video")
    "update_place_object": "free",   # data: name (existing place/object), description
    "suggest_outfits": "paid",   # the costume designer suggests outfits from the script (job kind "outfits")
    "make_character_sheet": "paid",  # data: character: reference pictures of that character (job kind "portrait")
    "record_voices": "paid",     # data: scene: records that scene's lines with the saved voices (job kind "voice")
    "make_place_object_picture": "paid",  # data: name: the picture of one place/object (job kind "element")
    "check_scenes": "paid",      # data: scenes [numbers] (empty = every scene with a picture): the picture check
    "fix_scene_picture": "paid", # data: scene, correction: a new picture of that scene that fixes what the check found
    "plan_shots": "paid",        # data: scenes [numbers]: the director plans each scene's camera shots (job "shotplan")
    "edit_shot": "free",         # data: scene, shot, and shot_type / camera_angle / camera_movement / duration_seconds / description
    "remake_shot_picture": "paid",   # data: scene, shot: a new opening picture of that one shot
    "remake_shot_video": "paid",     # data: scene, shot: a new clip of that one shot
}
PAID_JOB = {"develop_story": "script", "break_into_scenes": "breakdown", "find_places_objects": "elements",
            "make_cover": "cover", "make_film": "production", "remake_picture": "image", "remake_video": "video",
            "suggest_outfits": "outfits", "make_character_sheet": "portrait", "record_voices": "voice",
            "make_place_object_picture": "element", "check_scenes": "inspect", "fix_scene_picture": "image",
            "plan_shots": "shotplan", "remake_shot_picture": "shot_image", "remake_shot_video": "shot_video"}

_STR = {"type": "string"}
CHAT_SCHEMA = {
    "type": "object",
    "properties": {
        "reply": {"type": "string", "description": "Your message to the creator. Short for simple questions; detailed "
                                                  "and structured when the task needs it (a scene review, a plan, an "
                                                  "analysis)."},
        "proposals": {"type": "array", "items": {"type": "object", "properties": {
            "type": {"type": "string", "enum": list(ACTIONS)},
            "title": {"type": "string", "description": "Short button-like label, e.g. 'Add character: Ama'."},
            "summary": {"type": "string", "description": "One or two sentences: exactly what will change."},
            "data": {"type": "object", "properties": {
                "title": _STR, "logline": _STR, "genre": _STR, "kind": _STR, "visual_style": _STR, "language": _STR,
                "target_seconds": {"type": "integer"}, "script": _STR, "name": _STR, "description": _STR,
                "character": _STR, "owner": _STR, "clothing": _STR, "shoes": _STR, "accessories": _STR, "makeup": _STR,
                "outfit": _STR, "scene": {"type": "integer"}, "heading": _STR, "visual": _STR,
                "start_state": _STR, "end_state": _STR, "correction": _STR,
                "scenes": {"type": "array", "items": {"type": "integer"}}, "shot": {"type": "integer"},
                "shot_type": {"type": "string", "enum": ["establishing_wide", "wide_two_shot", "medium", "close_up",
                                                         "extreme_close_up", "over_the_shoulder", "reaction", "pov", "insert"]},
                "camera_angle": _STR, "camera_movement": _STR, "duration_seconds": {"type": "number"},
                "transition": {"type": "string", "enum": ["continuous", "same_place_cut", "new_place", "time_jump",
                                                          "flashback"]},
                "dialogue": {"type": "array", "items": {"type": "object", "properties": {
                    "character": _STR, "text": _STR, "emotion": _STR}}},
            }},
        }, "required": ["type", "title", "summary", "data"]}},
        "memory": {"type": "array", "description": "What to remember from the creator's LATEST message: facts, "
                                                  "preferences, decisions, open questions. Your own suggestions go in "
                                                  "category 'ideas', things you inferred in 'assumptions', both with "
                                                  "from_creator false. Never mark your own idea as from_creator.",
                   "items": {"type": "object", "properties": {
                       "category": {"type": "string", "enum": ["story", "characters", "preferences", "decisions",
                                                               "production", "questions", "ideas", "assumptions"]},
                       "text": {"type": "string"},
                       "from_creator": {"type": "boolean", "description": "true only if the creator said or decided it"}},
                       "required": ["category", "text", "from_creator"]}},
        "focus": {"type": "object", "description": "What this exchange is about, so a later 'her', 'that scene' or "
                                                   "'the next scene' can be understood.",
                  "properties": {"scenes": {"type": "array", "items": {"type": "integer"}},
                                 "characters": {"type": "array", "items": {"type": "string"}}}},
    },
    "required": ["reply", "proposals", "memory"],
}

SYSTEM = """You are Viyo, the filmmaking partner inside Viyo Studio. You help one creator turn an idea into a finished
short drama, series or movie. You are one assistant with five skills, used as the moment needs:
- Writer: premises, titles, loglines, characters, conflicts, twists, structure, screenplay, dialogue, episodes.
- Director: scenes, visual storytelling, shot types, camera angles and movement, action, pacing.
- Continuity supervisor: characters' faces, outfits, places, objects, timeline and story logic stay consistent.
- Production manager: what has been made, what is missing, what happens next and what it costs.
- Creative partner: brainstorm, give honest specific feedback, offer alternatives with pros and cons.

Picture checks (visual continuity):
- Scenes may have a PICTURE CHECK: a vision model compared the picture with the references, the record and the scene
  before (scores 0-100, problems with severity and certainty, whether the script means the change, the fix, what the
  picture shows). Use it to answer "why does she look different", "the room changed", "why is the letter missing".
  It can be wrong: say "the check thinks" for uncertain findings. An "outdated" check is for an older picture.
- To check pictures, propose check_scenes (data.scenes, or empty for all). It costs a little per scene; unchanged
  pictures are not checked again. To fix one, propose fix_scene_picture with data.scene and data.correction (what
  the new picture must change, e.g. "Maya's dress must stay red, as in scene 2"). Only that scene is remade.
- An APPROVED picture is never replaced: ask the creator to unapprove it on the Scenes tab first.
- Not fixed until a new picture has been made AND checked again, and the creator has seen it.

Seeing and auditing:
- You only see a picture if it is listed under VISUAL MATERIAL for this message. Never describe a picture you
  weren't given; if none are attached, say you haven't seen it and use the PICTURE CHECK results or offer to look.
- An audit of the whole production must cover every scene. VISUAL AUDIT lists every scene's state from the
  per-scene picture checks (each one compared that scene with the characters' references and the scene before). If
  it isn't complete, say "X of Y checked", name what's missing, and offer check_scenes for the rest (it runs one
  scene at a time and reports progress). Never say "all scenes" unless VISUAL AUDIT says complete. While checks are
  queued or running, say they're in progress.
- Looking at images and checking pictures never changes the production. Changes are proposals the creator applies.
- Story continuity too: dialogue that contradicts saved facts or MEMORY, events out of order, characters acting
  against their established motivation. Call these "possible" problems and ask whether they're intended.
- When something might be intended (a wardrobe change, a new place), explain why it might be and ask; give a
  concrete fix if it isn't, e.g. "Scene 4 seems to show Maya in a different outfit from the end of scene 3. I'd keep
  the previous outfit unless there's a wardrobe change in the story."

Camera shots (the director):
- A scene can be filmed as several shots (wide two-shot, medium, over-the-shoulder, close-up, reaction, insert...).
  Each scene below lists its shots with type, length, why it was chosen and whether it's made. Explain the choices in
  plain words when asked ("a close-up on Daniel so we see the lie land").
- plan_shots (data.scenes) has the director plan shots (paid, small). The creator approves the plan in the app, then
  each shot's picture and clip are made. edit_shot changes one shot (free, the creator applies it); remake_shot_picture
  / remake_shot_video remake one shot only. An approved shot or plan is never replaced: ask the creator to unapprove.
- Coverage depends on the moment: a conversation, an emotional beat, action or a quiet moment need different shots.

How you talk:
- Natural, warm and clear, like a talented collaborator. Simple words; explain film terms when you use them.
- Contribute: suggest original directions, twists, mysteries and stronger choices; point out weak spots and how to
  fix them. Don't just agree; say what works and what doesn't, kindly and specifically.
- If something important is unclear, ask ONE question (never a list of questions).
- Fit the length to the question. A simple question gets a short, chatty answer (2-4 sentences). A real creative or
  production task (reviewing a scene, a continuity problem, planning a movie, an episode outline, what is missing)
  gets a clear, detailed answer with short headings or bullets, as long as it needs and no longer. When there's a real
  choice, give at most 3 options with the trade-off of each. A screenplay goes in a write_screenplay proposal, never
  in your reply.
- When you review a scene say: what is wrong, why it is wrong, what should change, and which scene or asset it
  affects. Always say whether something is a suggestion or already saved in the project.
- Resolve "her", "him", "it", "that scene", "the next scene", "the previous episode" from CURRENT FOCUS, the recent
  messages and the project. Never make the creator repeat what the project already says.
- The creator is in charge. Never insist; never pretend a suggestion was used.
- Every Viyo story starts with a hook in the first seconds and ends on a hook.

Changes of mind and approved decisions:
- MEMORY items are listed oldest first. When the creator changes the story, the newer statement wins: work from it,
  say which earlier facts, scenes, shots or pictures it affects, and suggest removing the outdated memory item.
- If the creator asks for something that contradicts an APPROVED decision, an approved picture or an approved shot
  plan, don't just do it and don't refuse: name the decision, say what would change, and ask ONE question to
  confirm before proposing the change.
- A scene whose SHOTS say OUTDATED had its lines changed after the shots were planned: say the shots must be
  planned again (unapprove the plan first) before shot pictures, clips or the film are made.

Truth rules (very important):
- Use only the PROJECT facts below. If something isn't there, say you don't know it yet or ask; never invent
  saved characters, scenes, prices or results.
- You cannot change the project yourself. To change something, add a proposal; it is only saved if the creator
  taps Apply. Never say something "was saved", "was changed", "is generated" or "was approved" because of you.
- Paid steps (developing the full story, breaking it into scenes, finding places and objects, covers, making the
  film) cost credits: propose them and say the price will be shown before anything is charged.
- Ideas in the conversation summary, earlier messages or MEMORY "Ideas not approved yet" / "assumptions" are only
  ideas unless they appear in PROJECT facts or as approved decisions in MEMORY. Only say you'll remember something
  when it's the creator's own fact or decision; your own ideas are noted as ideas.
- Something is done only when the project shows it (a saved field, a finished step). A step you proposed is not
  done until the creator applied it and the project or the recent steps show it succeeded. If a step failed, say so
  and say why (from Recent steps).
- If part of the screenplay or a scene isn't shown in full below, say which part you'd need and ask; never guess it.

Continuity supervisor:
- The PROJECT facts list each scene's continuity record (start, end, positions, what people hold and wear, feelings,
  how it follows the scene before), what each character wears in it, and a CONTINUITY CHECK of problems found
  between consecutive scenes.
- When asked why scenes don't match, or to make one continue from another: name the scenes, say exactly what changed
  (outfit, place, object, time, position, feeling), why it breaks the story, and propose the fix (rewrite_scene with
  start_state / end_state, set_scene_outfit), then remake_picture or remake_video for the scene that must change.
- Tell intended changes (the script means them) from accidental ones.
- A picture or video only changes when it is made again. Never say a visual problem is fixed until a new picture or
  video has been made and the creator has looked at it. Pictures can't be guaranteed identical: say it gets closer.

Production manager:
- PRODUCTION STATUS says what exists and what is missing. When asked what is left, list it in order (story, cast,
  reference sheets, outfits, places and objects, scene pictures, videos, voices, film) with what each costs to make.
- Guide voice and audio: each character's saved voice is listed; scenes are spoken either inside the video or with
  the saved voices. Music: an original score is made at render unless the creator uploaded a song or turned it off.
- Never start anything paid just because an idea was discussed: propose it, and the price is shown before it starts.

Proposals (0 to 3, only when a concrete change would help now):
- update_story: title / logline / genre / kind (short, episode, series, film) / target_seconds / visual_style / language.
- write_screenplay: data.script = the complete screenplay text (only when the creator asked you to write or rewrite it
  here; standard format: scene headings, action lines, CHARACTER names in caps before dialogue).
- add_character / update_character: data.name or data.character + data.description (age, look, personality).
- add_outfit: data.character, data.name, data.clothing, data.shoes, data.accessories, data.makeup.
- add_place / add_object: data.name, data.description (exact look), data.owner for objects.
- rewrite_scene: data.scene (number from the scene list), data.heading, data.visual, data.dialogue, and for
  continuity data.start_state / data.end_state / data.transition.
- remake_picture / remake_video: data.scene. A new picture or video for that one scene only (paid). Suggest it after a
  free fix (rewrite_scene, set_scene_outfit) so the scene is made again with the correction.
- set_scene_outfit: data.scene, data.character, data.outfit (an existing outfit name).
- update_place_object: data.name (an existing place or object), data.description.
- make_character_sheet: data.character. make_place_object_picture: data.name. record_voices: data.scene.
- check_scenes: data.scenes. fix_scene_picture: data.scene, data.correction.
- plan_shots: data.scenes. edit_shot / remake_shot_picture / remake_shot_video: data.scene and data.shot (numbers).
- develop_story, break_into_scenes, find_places_objects, suggest_outfits, make_cover, make_film: paid steps, data may
  be empty.
Use the exact names from the PROJECT facts for characters, outfits and scene numbers."""


def _short(text: str, n: int) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return text if len(text) <= n else text[: n - 1] + "…"


DETAIL_ALL = 20   # up to this many scenes, every scene is shown in full


def scenes_named(message: str) -> set[int]:
    """Scene numbers the creator mentions ("scene 2", "scenes 3 and 4")."""
    out = set()
    for m in re.finditer(r"scenes?\s+((?:\d+\s*(?:,|and|&|-|to)?\s*)+)", (message or "").lower()):
        out.update(int(x) for x in re.findall(r"\d+", m.group(1)))
    words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
    for m in re.finditer(r"scene\s+(one|two|three|four|five|six|seven|eight|nine|ten)\b", (message or "").lower()):
        out.add(words[m.group(1)])
    return out


def _scene_detail(c: dict, wearing: dict) -> list[str]:
    out = []
    if c.get("transition"):
        out.append(f"   From the scene before: {c['transition'].replace('_', ' ')}"
                   f"{' (' + c['transition_note'] + ')' if c.get('transition_note') else ''}")
    for k, label in (("start_state", "Starts"), ("action", "Action"), ("end_state", "Ends"),
                     ("camera", "Camera"), ("time_of_day", "Time"), ("lighting", "Light")):
        if c.get(k):
            out.append(f"   {label}: {_short(c[k], 300)}")
    for x in c.get("characters") or []:
        bits = [x.get("position") or "", "holding " + ", ".join(x["holding"]) if x.get("holding") else "",
                "ends holding " + ", ".join(x["holding_at_end"]) if x.get("holding_at_end") else "",
                f"feels {x.get('emotion_start')}→{x.get('emotion_end')}" if x.get("emotion_start") else "",
                "only heard" if not x.get("in_frame", True) else "", "enters" if x.get("enters") else "",
                "leaves" if x.get("leaves") else ""]
        out.append(f"   {x['name']}: " + "; ".join(b for b in bits if b))
    if c.get("intentional_changes"):
        out.append("   Intended changes: " + "; ".join(c["intentional_changes"]))
    if wearing:
        out.append("   Wearing: " + "; ".join(f"{k} – {v}" for k, v in wearing.items()))
    return out


CONTEXT_CHARS = int(os.environ.get("STUDIO_CHAT_CONTEXT_CHARS", "60000"))   # the project facts block, at most
SCRIPT_CHARS = int(os.environ.get("STUDIO_CHAT_SCRIPT_CHARS", "24000"))      # the screenplay part of it
_HEADING = re.compile(r"^\s*(?:\d+\s*[.)]?\s*)?(INT\.|EXT\.|INT/EXT|EXT/INT|I/E)[^\n]*", re.I | re.M)
_STOP = {"the", "and", "that", "this", "with", "what", "does", "make", "about", "from", "scene", "scenes", "into",
         "have", "they", "them", "she", "her", "him", "his", "your", "there", "where", "when", "why", "how", "can",
         "should", "would", "could", "will", "more", "less", "story", "please", "viyo"}


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9']{3,}", (text or "").lower()) if w not in _STOP}


def script_sections(script: str) -> list[tuple[str, str]]:
    """(heading, text) for each scene section of a screenplay; the part before
    the first heading (title, notes) is its own section."""
    script = script or ""
    marks = [m.start() for m in _HEADING.finditer(script)]
    if not marks:
        return [("(whole screenplay)", script)] if script.strip() else []
    out = []
    if script[:marks[0]].strip():
        out.append(("(opening notes)", script[:marks[0]]))
    for i, a in enumerate(marks):
        b = marks[i + 1] if i + 1 < len(marks) else len(script)
        chunk = script[a:b]
        out.append((chunk.strip().splitlines()[0].strip(), chunk))
    return out


def screenplay_block(script: str, question: str = "", focus_numbers: Optional[set] = None,
                     names: Optional[set] = None, budget: Optional[int] = None) -> str:
    """The screenplay for the prompt. Complete when it fits; otherwise an index
    of EVERY section plus the full text of the most relevant ones (the opening,
    the end, sections the creator named or whose words and characters match the
    question). Says exactly what is and isn't shown, so nothing is dropped
    silently."""
    script = (script or "").strip()
    budget = budget or SCRIPT_CHARS
    if not script:
        return "Screenplay: (none yet)"
    words = len(script.split())
    if len(script) <= budget:
        return f"Screenplay (complete, {words} words):\n{script}"
    secs = script_sections(script)
    q = _words(question) | {n.lower() for n in names or set()}
    want = {n - 1 for n in focus_numbers or set()}
    scored = []
    for i, (head, text) in enumerate(secs):
        score = (100 if i in want else 0) + (5 if i in (0, len(secs) - 1) else 0)
        score += 3 * len(q & _words(head)) + len(q & _words(text))
        scored.append((score, i))
    index = [f"#{i + 1} {_short(h, 90)} — {_short(t.split(chr(10), 1)[-1], 110)}" for i, (h, t) in enumerate(secs)]
    room = budget - sum(len(x) + 1 for x in index) - 400
    chosen = []
    for score, i in sorted(scored, key=lambda x: (-x[0], x[1])):
        if score <= 0 and chosen:
            break
        if room - len(secs[i][1]) < 0:
            continue
        chosen.append(i)
        room -= len(secs[i][1])
    chosen.sort()
    shown = ", ".join(f"#{i + 1}" for i in chosen) or "none"
    body = "\n\n".join(f"[Section #{i + 1}]\n{secs[i][1].strip()}" for i in chosen)
    return (f"Screenplay ({words} words, {len(secs)} sections; too long to show whole). Index of every section:\n"
            + "\n".join(index)
            + f"\nFull text shown below for sections {shown}. The others are in the index only: if you need one, "
              "say which and ask, don't guess.\n" + body)


def _check_lines(r: dict, outdated: bool, approved: bool) -> list[str]:
    sc = r.get("scores") or {}
    head = " · ".join(f"{k} {v}" for k, v in sc.items())
    out = [f"   PICTURE CHECK{' (outdated: for an older picture)' if outdated else ''}"
           f"{' [picture APPROVED by the creator]' if approved else ''}: {head}. {_short(r.get('summary') or '', 200)}"]
    for p in (r.get("problems") or [])[:6]:
        tag = "intended" if p.get("intentional") else f"{p.get('severity')}, {p.get('certainty')}"
        out.append(f"     - [{p.get('kind')}, {tag}] expected {_short(p.get('expected'), 120)}; seen {_short(p.get('seen'), 120)}."
                   f" Fix: {_short(p.get('fix'), 160)}")
    obs = r.get("observed") or {}
    if obs.get("characters"):
        out.append("     Picture shows: " + "; ".join(
            f"{c.get('name')} ({', '.join(x for x in (c.get('outfit'), c.get('position'), ('holding ' + ', '.join(c['holding'])) if c.get('holding') else '') if x)})"
            for c in obs["characters"][:4]))
    return out


def _bible_block(bible: dict) -> list[str]:
    if not bible:
        return []
    approved = (bible.get("_assistant_approval") or {}).get("approved")
    out = [f"Story plan ({'approved by the creator' if approved else 'not approved yet'}):"]
    for k, label in (("logline", "Logline"), ("central_conflict", "Conflict"), ("stakes", "Stakes"),
                     ("theme", "Theme"), ("twist", "Twist"), ("ending", "Ending")):
        if bible.get(k):
            out.append(f"  {label}: {_short(bible[k], 400)}")
    for c in (bible.get("characters") or [])[:16]:
        bits = [f"{k} {c[k]}" for k in ("age", "role", "want", "flaw", "secret", "speech", "look") if c.get(k)]
        rel = c.get("relationships") or c.get("relationship")
        if rel:
            bits.append(f"relationships: {rel if isinstance(rel, str) else '; '.join(map(str, rel))}")
        out.append(f"  - {c.get('name')}: " + "; ".join(_short(b, 160) for b in bits))
    for i, b in enumerate((bible.get("beats") or [])[:40], start=1):
        out.append(f"  Beat {i} ({b.get('purpose', '')}): {_short(b.get('what_happens', ''), 220)}"
                   f"{' [' + b['emotional_turn'] + ']' if b.get('emotional_turn') else ''}")
    for e in (bible.get("episodes") or [])[:30]:
        if isinstance(e, dict):
            out.append(f"  Episode {e.get('number', '')}: {_short(e.get('title') or '', 80)} – {_short(e.get('summary') or '', 200)}")
    return out


def status(scenes: list[dict], cast: list[dict], jobs: list[dict], issues: Optional[list] = None,
           renders: int = 0, elements: Optional[list] = None) -> dict:
    """What is made and what is missing (for the prompt and the app)."""
    talk = [s for s in scenes if any((d.get("text") or "").strip() for d in s.get("dialogue") or [])]
    return {
        "scenes": len(scenes),
        "pictures": sum(1 for s in scenes if s.get("image_path")),
        "videos": sum(1 for s in scenes if s.get("video_path")),
        "video_scenes": sum(1 for s in scenes if s.get("mode") == "video"),
        "voiced": sum(1 for s in talk if s.get("audio_path") or s.get("video_path")),
        "with_lines": len(talk),
        "characters": len(cast),
        "sheets": sum(1 for c in cast if (c.get("reference_views") or {}) or c.get("reference_path")),
        "places_objects": len(elements or []),
        "place_object_pictures": sum(1 for e in elements or [] if e.get("reference_path")),
        "running": sum(1 for j in jobs if j.get("status") in ("queued", "running")),
        "failed": sum(1 for j in jobs[:20] if j.get("status") == "failed"),
        "continuity_problems": sum(1 for i in issues or [] if i.get("severity") == "fix"),
        "checked": sum(1 for s in scenes if (s.get("inspection") or {}).get("result")),
        "checked_current": sum(1 for s in scenes if s.get("image_path") and (s.get("inspection") or {}).get("result")
                               and (s.get("inspection") or {}).get("image_path") == s.get("image_path")),
        "checked_outdated": sum(1 for s in scenes if s.get("image_path") and (s.get("inspection") or {}).get("result")
                                and (s.get("inspection") or {}).get("image_path") != s.get("image_path")),
        "picture_problems": sum(1 for s in scenes if _attention(s)),
        "films": renders,
    }


def _attention(scene: dict) -> bool:
    import studio_inspect
    ins = scene.get("inspection") or {}
    return bool(ins.get("result")) and ins.get("image_path") == scene.get("image_path") and \
        studio_inspect.needs_attention(ins["result"])


def _status_lines(st: dict) -> list[str]:
    missing = []
    if st["scenes"] and st["pictures"] < st["scenes"]:
        missing.append(f"{st['scenes'] - st['pictures']} scene pictures")
    if st["video_scenes"] and st["videos"] < st["video_scenes"]:
        missing.append(f"{st['video_scenes'] - st['videos']} scene videos")
    if st["with_lines"] and st["voiced"] < st["with_lines"]:
        missing.append(f"voices for {st['with_lines'] - st['voiced']} scenes")
    if st["characters"] and st["sheets"] < st["characters"]:
        missing.append(f"reference pictures for {st['characters'] - st['sheets']} characters")
    if st["places_objects"] and st["place_object_pictures"] < st["places_objects"]:
        missing.append(f"pictures for {st['places_objects'] - st['place_object_pictures']} places/objects")
    if st["scenes"] and not st["films"]:
        missing.append("the final film")
    cover = (f"AUDIT COVERAGE: plan-level continuity review covers all {st['scenes']} scenes; picture checks are "
             f"current for {st['checked_current']} of {st['pictures']} scene pictures ({st['checked_outdated']} outdated, "
             f"{st['pictures'] - st['checked_current'] - st['checked_outdated']} never checked; "
             f"{st['scenes'] - st['pictures']} scenes have no picture). "
             + ("Every picture has a current check." if st['pictures'] and st['checked_current'] == st['pictures']
                else "A visual audit of the whole production is NOT complete: say how many were checked."))
    return [cover, f"PRODUCTION STATUS: {st['scenes']} scenes · {st['pictures']} pictures · {st['videos']} videos · "
            f"{st['voiced']}/{st['with_lines']} scenes with lines voiced · {st['characters']} characters "
            f"({st['sheets']} with reference pictures) · {st['films']} films made · {st['running']} steps running · "
            f"{st['failed']} recent failures · {st['continuity_problems']} continuity problems · "
            f"{st['checked']} pictures checked, {st['picture_problems']} need attention",
            "Still missing: " + ("; ".join(missing) if missing else "nothing obvious")]


def project_context(p: dict, cast: list[dict], outfits: list[dict], elements: list[dict], scenes: list[dict],
                    production: Optional[dict], jobs: list[dict], wearing: Optional[dict] = None,
                    issues: Optional[list] = None, focus: Optional[set] = None, question: str = "",
                    focus_names: Optional[set] = None, episodes: Optional[list] = None, renders: int = 0,
                    audit: Optional[dict] = None) -> str:
    """What the assistant knows as fact: only what is saved, retrieved for the question."""
    import studio_identity as ident
    focus = set(focus or set())
    names = {c["name"] for c in cast if c.get("name") and c["name"].lower() in (question or "").lower()} | \
        set(focus_names or set())
    lines = [f"Title: {p.get('title') or '(none)'}",
             f"Idea / logline: {p.get('logline') or '(none yet)'}",
             f"Format: {p.get('kind')} · about {p.get('target_seconds')} s · {p.get('aspect_ratio')} · "
             f"genre {p.get('genre') or '(not set)'} · style {p.get('visual_style') or '(not set)'} · "
             f"language {p.get('language') or 'English'} · made as {p.get('production_style') or 'pictures'}"]
    if episodes:
        lines.append("Other episodes of this series (in order):")
        for e in episodes[:40]:
            lines.append(f"- Episode {e.get('number')}: {e.get('title') or '(untitled)'} – {_short(e.get('logline') or '', 200)}"
                         f"{' (this one)' if e.get('current') else ''}")
    lines += _bible_block(p.get("story_bible") or {})
    lines.append(screenplay_block(p.get("script") or "", question, focus, names))
    if cast:
        lines.append("Characters:")
        for c in cast[:24]:
            outs = [o for o in outfits if o["character_id"] == c["id"] and (o.get("status") or "saved") == "saved"]
            views = c.get("reference_views") or {}
            approved = sum(1 for v in views.values() if isinstance(v, dict) and v.get("approved"))
            sheet = (f"reference pictures: {len(views)} made, {approved} approved" if views
                     else "reference picture: portrait" if c.get("reference_path") else "no reference pictures yet")
            voice = (f"voice {c['voice']}" + (f" ({_short(c.get('voice_description') or '', 120)})" if c.get("voice_description") else "")
                     if c.get("voice") else f"no voice yet ({c.get('voice_status') or 'not cast'})")
            lines.append(f"- {c['name']}: {_short(ident.identity_text(c['name'], c.get('identity'), c.get('description') or ''), 500)}"
                         f"{' [identity locked]' if c.get('identity_locked_at') else ''} · {sheet} · {voice}")
            for o in outs[:8]:
                parts = "; ".join(f"{k}: {v}" for k, v in (o.get("parts") or {}).items() if v)
                lines.append(f"    outfit \"{o['name']}\"{' (default)' if o.get('is_default') else ''}"
                             f"{': ' + _short(parts or o.get('description') or '', 200) if (parts or o.get('description')) else ''}")
    else:
        lines.append("Characters: (none yet)")
    if elements:
        lines.append("Places and objects:")
        for e in elements[:40]:
            lines.append(f"- {'Place' if e['kind'] == 'location' else 'Object'} \"{e['name']}\": "
                         f"{_short(e.get('description') or '', 200)}"
                         f" [{'picture approved' if e.get('status') == 'approved' else 'picture made' if e.get('reference_path') else 'no picture'}]")
    if scenes:
        lines.append(f"Scenes ({len(scenes)}):")
        near = {n + d for n in focus for d in (-1, 0, 1)}
        lower = {n.lower() for n in names}
        for i, s in enumerate(scenes[:120], start=1):
            who = sorted({(d.get("character") or "").strip() for d in s.get("dialogue") or [] if d.get("character")})
            inframe = {x.get("name", "").lower() for x in (s.get("continuity") or {}).get("characters") or []}
            about = bool(lower & ({w.lower() for w in who} | inframe)) or any(
                n in (s.get("visual") or "").lower() for n in lower)
            made = "video" if s.get("video_path") else "picture" if s.get("image_path") else "not made"
            full = len(scenes) <= DETAIL_ALL or i in near or (about and len(scenes) <= 60)
            lines.append(f"{i}. {s.get('heading') or ''}: {_short(s.get('visual') or '', 600 if full else 140)}"
                         f"{' Speakers: ' + ', '.join(who) if who else ''} [{made}]")
            shots = s.get("_shots") or []
            if shots and (full or len(scenes) <= 60):
                import studio_shots
                stale = " – OUTDATED: the scene's lines changed after the shots were planned; plan them again" \
                    if studio_shots.plan_outdated(s, shots) else ""
                lines.append(f"   SHOTS ({len(shots)}, plan {'APPROVED' if s.get('shot_plan_approved') else 'not approved'}{stale}): " + " | ".join(
                    f"{k + 1}. {sh['shot_type'].replace('_', ' ')} {float(sh['duration_seconds']):g}s"
                    f"{' [made]' if sh.get('video_path') else ' [picture]' if sh.get('image_path') else ''}"
                    f"{' [approved]' if sh.get('approved_at') else ''}" + (f" – {_short(sh.get('reason'), 90)}" if full else "")
                    for k, sh in enumerate(shots)))
            ins = (s.get("inspection") or {}).get("result")
            if ins and (full or ins.get("problems")):
                lines += _check_lines(ins, (s.get("inspection") or {}).get("image_path") != s.get("image_path"),
                                      bool(s.get("image_approved_path")) and s.get("image_approved_path") == s.get("image_path"))
            elif s.get("image_approved_path") and s.get("image_approved_path") == s.get("image_path"):
                lines.append("   [picture APPROVED by the creator]")
            if full:
                for d in (s.get("dialogue") or [])[:12]:
                    if (d.get("text") or "").strip():
                        lines.append(f"   {d.get('character')}: \"{_short(d['text'], 240)}\""
                                     f"{' (' + d['emotion'] + ')' if d.get('emotion') else ''}")
                lines += _scene_detail(s.get("continuity") or {}, (wearing or {}).get(s["id"]) or {})
        if len(scenes) > 120:
            lines.append(f"(+{len(scenes) - 120} more scenes not listed; ask the creator which ones)")
        if (p.get("script") or "").strip():
            import studio_story
            lines += studio_story.dialogue_lines(studio_story.dialogue_check(p["script"], scenes))
        if issues is not None:
            lines.append("CONTINUITY CHECK (consecutive scenes): " + ("no problems found" if not issues else ""))
            for x in issues[:20]:
                lines.append(f"- Scene {x['previous']}→{x['scene']} [{x['kind']}]: {x['message']} Fix: {x['fix']}")
    else:
        lines.append("Scenes: (not broken into scenes yet)")
    lines += _status_lines(status(scenes, cast, jobs, issues, renders, elements))
    if audit:
        import studio_inspect
        lines += studio_inspect.audit_lines(audit)
    if production:
        lines.append(f"One-click production: {production.get('status')} ({production.get('percent', 0)}%)"
                     f"{' – ' + production['problem'] if production.get('problem') else ''}")
    recent = [j for j in jobs[:12] if j.get("status") in ("queued", "running", "failed", "succeeded")]
    if recent:
        lines.append("Recent steps (newest first):")
        num = {sc["id"]: i for i, sc in enumerate(scenes, start=1)}
        for j in recent:
            where = f" scene {num[j['scene_id']]}" if j.get("scene_id") in num else ""
            err = f" – {_short(j.get('error_message') or '', 220)}" if j.get("status") == "failed" else ""
            lines.append(f"- {j['kind']}{where}: {j['status']}{err}")
    text = "\n".join(lines)
    if len(text) > CONTEXT_CHARS:
        text = text[:CONTEXT_CHARS] + "\n(Project facts cut here to fit; ask about anything not shown.)"
    return text


PRONOUNS = re.compile(r"\b(she|her|hers|he|him|his|they|them|it|that scene|this scene|the scene|next scene|"
                      r"previous scene|last scene|the next one|continue|(that|this|the|previous|last|next|"
                      r"first|other) shots?)\b", re.I)


def resolve_focus(message: str, previous: Optional[dict], scene_count: int, cast_names: list[str]) -> dict:
    """What "her", "that scene", "the next scene" refer to: scenes and characters
    named now, otherwise what the conversation was last about."""
    previous = previous or {}
    msg = message or ""
    scenes = {n for n in scenes_named(msg) if 1 <= n <= max(scene_count, 1)}
    chars = [n for n in cast_names if re.search(rf"\b{re.escape(n.split(' ')[0])}\b", msg, re.I)]
    prev_scenes = [n for n in previous.get("scenes") or [] if isinstance(n, int)]
    if not scenes and prev_scenes and PRONOUNS.search(msg):
        last = max(prev_scenes)
        if re.search(r"\bnext (scene|one)\b|\bcontinue\b", msg, re.I):
            scenes = {min(last + 1, scene_count)} if scene_count else set()
        elif re.search(r"\b(previous|last) scene\b", msg, re.I):
            scenes = {max(1, min(prev_scenes) - 1)}
        else:
            scenes = set(prev_scenes)
    if not chars and re.search(r"\b(she|her|hers|he|him|his|they|them)\b", msg, re.I):
        chars = [c for c in previous.get("characters") or [] if c in cast_names]
    return {"scenes": sorted(scenes), "characters": chars}


def focus_text(focus: dict) -> str:
    bits = []
    if focus.get("scenes"):
        bits.append("scene " + ", ".join(map(str, focus["scenes"])))
    if focus.get("characters"):
        bits.append(", ".join(focus["characters"]))
    return ("CURRENT FOCUS (what 'that scene', 'her', 'him', 'the next scene' refer to): " + "; ".join(bits)) if bits \
        else "CURRENT FOCUS: nothing specific yet"


MODES = ("quick", "story", "deep")
_DEEP = re.compile(r"\b(continuity|consisten|audit|review (the|all|every|whole)|analy[sz]e|contradict|plot hole|"
                   r"what('s| is) (still )?missing|what('s| is) left|production plan|plan the production|whole film|"
                   r"full movie|feature film|every scene|all (the )?scenes|whole (story|screenplay|series))", re.I)
_STORY = re.compile(r"\b(story|plot|character|arc|twist|ending|screenplay|script|dialogue|rewrite|develop|episode|"
                    r"series|movie|film|scene|idea|mysterious|motivation|conflict|outline|structure)", re.I)


def reasoning_mode(message: str) -> str:
    """How hard to think: 'quick' for chat, 'story' for creative work, 'deep'
    for screenplay analysis, continuity audits and production planning."""
    msg = message or ""
    if _DEEP.search(msg):
        return "deep"
    if _STORY.search(msg) or len(msg) > 600:
        return "story"
    return "quick"


def suggestions(p: dict, cast: list, scenes: list, issues: list, jobs: list, renders: int = 0) -> list[str]:
    """Questions worth asking now, from where the production is."""
    out = []
    failed = [j for j in jobs[:10] if j.get("status") == "failed"]
    if not (p.get("logline") or "").strip() and not (p.get("script") or "").strip():
        return list(SUGGESTIONS)
    if not (p.get("script") or "").strip():
        out += ["Develop my idea into a full story.", "Who should the main characters be?",
                "Give me three possible endings.", "Can we make this a full movie?"]
    elif not scenes:
        out += ["Is the screenplay ready to break into scenes?", "What are the weak spots in my story?",
                "Make the opening hook stronger.", "Can we make this a full movie?"]
    else:
        if failed:
            out.append("Why did the last step fail?")
        if [i for i in issues if i.get("severity") == "fix"]:
            out.append("Fix the continuity problems between my scenes.")
        out += ["What is still missing before this episode is finished?", "Which scene is the weakest, and why?",
                "Does every scene connect to the one before?", "What happens in the next episode?"]
        if renders:
            out.insert(0, "How can the next cut be better?")
    return out[:4]


def build_prompt(context: str, summary: str, memory, history: list[dict], message: str,
                 focus: str = "", mode: str = "quick", visual: str = "") -> str:
    """memory: the project memory rendered (str), or a list of decisions (older callers)."""
    convo = "\n".join(f"{'CREATOR' if m['role'] == 'user' else 'VIYO'}: {_short(m['content'], 1800)}" for m in history)
    mem = memory if isinstance(memory, str) else ("Approved decisions:\n" + "\n".join(f"- {d}" for d in memory[-30:])
                                                  if memory else "(nothing remembered yet)")
    depth = {"quick": "This is a quick question: answer briefly.",
             "story": "This is creative work: think it through and give a useful, specific answer.",
             "deep": "This needs careful analysis of the whole production: go through it systematically and be "
                     "precise about scenes, characters and assets."}.get(mode, "")
    return f"""{SYSTEM}

PROJECT (saved facts):
{context}

MEMORY (this production, across all conversations):
{mem}

EARLIER CONVERSATION (summary, ideas only):
{summary or '(none)'}

{focus or 'CURRENT FOCUS: nothing specific yet'}

{visual or 'VISUAL MATERIAL: none attached to this message. You have NOT seen the pictures.'}

RECENT MESSAGES:
{convo or '(this is the first message)'}

CREATOR: {message}

{depth}
Reply as VIYO."""


def summary_prompt(old_summary: str, messages: list[dict]) -> str:
    convo = "\n".join(f"{'CREATOR' if m['role'] == 'user' else 'VIYO'}: {_short(m['content'], 1200)}" for m in messages)
    return f"""Update the running summary of a conversation between a filmmaker and their assistant. Keep what
matters for continuing the work: every character, place and episode name mentioned; facts the creator stated;
decisions the creator made (say "DECIDED:"); constraints and preferences ("the creator wants / doesn't want");
ideas discussed (say "IDEA:", never as decided); options offered; questions still open; what was applied or made.
Do not state that anything was saved unless the conversation says it was applied. At most 350 words.

CURRENT SUMMARY:
{old_summary or '(empty)'}

NEW MESSAGES:
{convo}"""


MEMORY_CATEGORIES = ("story", "characters", "preferences", "decisions", "production", "questions", "ideas", "assumptions")


def validate_reply(data) -> dict:
    """Checks the model's answer against CHAT_SCHEMA before anything is used.
    Raises ValueError without a usable reply; drops malformed parts (they are
    never executed or saved) and lists what was dropped."""
    if not isinstance(data, dict):
        raise ValueError("the answer isn't an object")
    reply = data.get("reply")
    if not isinstance(reply, str) or not reply.strip():
        raise ValueError("the answer has no reply text")
    dropped = []
    props = data.get("proposals")
    if not isinstance(props, list):
        if props is not None:
            dropped.append("proposals")
        props = []
    good = []
    for p in props:
        if isinstance(p, dict) and p.get("type") in ACTIONS and isinstance(p.get("title", ""), str) \
                and isinstance(p.get("data", {}), dict):
            good.append(p)
        else:
            dropped.append("a proposal")
    mem = data.get("memory")
    if not isinstance(mem, list):
        if mem is not None:
            dropped.append("memory")
        mem = []
    memory = [m for m in mem if isinstance(m, dict) and m.get("category") in MEMORY_CATEGORIES
              and isinstance(m.get("text"), str) and m["text"].strip()]
    if len(memory) != len(mem):
        dropped.append("a memory note")
    focus = data.get("focus") if isinstance(data.get("focus"), dict) else {}
    focus = {"scenes": [n for n in focus.get("scenes") or [] if isinstance(n, int) and not isinstance(n, bool)][:6],
             "characters": [c for c in focus.get("characters") or [] if isinstance(c, str)][:6]}
    return {"reply": reply.strip(), "proposals": good, "memory": memory, "focus": focus, "dropped": dropped}


def clean_proposals(raw) -> list[dict]:
    out = []
    for p in (raw or [])[:MAX_PROPOSALS]:
        if not isinstance(p, dict) or p.get("type") not in ACTIONS:
            continue
        data = p.get("data") if isinstance(p.get("data"), dict) else {}
        out.append({"type": p["type"], "title": _short(p.get("title") or p["type"].replace("_", " ").title(), 80),
                    "summary": _short(p.get("summary") or "", 400), "data": data,
                    "paid": ACTIONS[p["type"]] == "paid", "job": PAID_JOB.get(p["type"]), "status": "proposed"})
    return out


SUGGESTIONS = [
    "I have an idea: a poor young man discovers he's the son of a billionaire.",
    "Help me turn my idea into an exciting drama.",
    "Give me three possible endings.",
    "What do you recommend we do next?",
]
