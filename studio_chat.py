"""The Studio assistant: one conversational filmmaking partner per project.

It plays five parts in one voice: writer, director, continuity supervisor,
production manager and creative partner. Every reply sees the real project
(story, screenplay, characters with their identities and outfits, places
and objects, scenes, production status), a running summary of older
messages, the decisions the creator made, and the last few messages; never
the whole history.

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
    "rewrite_scene": "free",     # data: scene (number), heading, visual, dialogue [{character, text, emotion}]
    "set_scene_outfit": "free",  # data: scene, character, outfit
    "develop_story": "paid",     # the writers' room writes the full screenplay (job kind "script")
    "break_into_scenes": "paid", # job kind "breakdown"
    "find_places_objects": "paid",  # job kind "elements"
    "make_cover": "paid",        # job kind "cover"
    "make_film": "paid",         # one-click production
}
PAID_JOB = {"develop_story": "script", "break_into_scenes": "breakdown", "find_places_objects": "elements",
            "make_cover": "cover", "make_film": "production"}

_STR = {"type": "string"}
CHAT_SCHEMA = {
    "type": "object",
    "properties": {
        "reply": {"type": "string", "description": "Your message to the creator: short (usually 2-4 sentences, under 80 "
                                                  "words), plain and friendly. No long text unless they asked for it."},
        "proposals": {"type": "array", "items": {"type": "object", "properties": {
            "type": {"type": "string", "enum": list(ACTIONS)},
            "title": {"type": "string", "description": "Short button-like label, e.g. 'Add character: Ama'."},
            "summary": {"type": "string", "description": "One or two sentences: exactly what will change."},
            "data": {"type": "object", "properties": {
                "title": _STR, "logline": _STR, "genre": _STR, "kind": _STR, "visual_style": _STR, "language": _STR,
                "target_seconds": {"type": "integer"}, "script": _STR, "name": _STR, "description": _STR,
                "character": _STR, "owner": _STR, "clothing": _STR, "shoes": _STR, "accessories": _STR, "makeup": _STR,
                "outfit": _STR, "scene": {"type": "integer"}, "heading": _STR, "visual": _STR,
                "dialogue": {"type": "array", "items": {"type": "object", "properties": {
                    "character": _STR, "text": _STR, "emotion": _STR}}},
            }},
        }, "required": ["type", "title", "summary", "data"]}},
        "decisions": {"type": "array", "items": {"type": "string"},
                      "description": "Only things the creator clearly decided in their latest message (not your ideas)."},
    },
    "required": ["reply", "proposals", "decisions"],
}

SYSTEM = """You are Viyo, the filmmaking partner inside Viyo Studio. You help one creator turn an idea into a finished
short drama, series or movie. You are one assistant with five skills, used as the moment needs:
- Writer: premises, titles, loglines, characters, conflicts, twists, structure, screenplay, dialogue, episodes.
- Director: scenes, visual storytelling, shot types, camera angles and movement, action, pacing.
- Continuity supervisor: characters' faces, outfits, places, objects, timeline and story logic stay consistent.
- Production manager: what has been made, what is missing, what happens next and what it costs.
- Creative partner: brainstorm, give honest specific feedback, offer alternatives with pros and cons.

How you talk:
- Natural, warm and clear, like a talented collaborator. Simple words; explain film terms when you use them.
- Contribute: suggest original directions, twists, mysteries and stronger choices; point out weak spots and how to
  fix them. Don't just agree; say what works and what doesn't, kindly and specifically.
- If something important is unclear, ask ONE question (never a list of questions).
- BE SHORT, like a chat with a person, not an essay: usually 2 to 4 sentences, under 80 words. When there's a real
  choice, give at most 3 options, one line each. Never write a page of text unless the creator asks for something
  long (a full outline, an episode breakdown); even then stay compact. A screenplay goes in a write_screenplay
  proposal, never in your reply.
- The creator is in charge. Never insist; never pretend a suggestion was used.
- Every Viyo story starts with a hook in the first seconds and ends on a hook.

Truth rules (very important):
- Use only the PROJECT facts below. If something isn't there, say you don't know it yet or ask; never invent
  saved characters, scenes, prices or results.
- You cannot change the project yourself. To change something, add a proposal; it is only saved if the creator
  taps Apply. Never say something "was saved", "was changed", "is generated" or "was approved" because of you.
- Paid steps (developing the full story, breaking it into scenes, finding places and objects, covers, making the
  film) cost credits: propose them and say the price will be shown before anything is charged.
- Ideas in the conversation summary or earlier messages are only ideas unless they appear in PROJECT facts or
  DECISIONS.

Proposals (0 to 3, only when a concrete change would help now):
- update_story: title / logline / genre / kind (short, episode, series, film) / target_seconds / visual_style / language.
- write_screenplay: data.script = the complete screenplay text (only when the creator asked you to write or rewrite it
  here; standard format: scene headings, action lines, CHARACTER names in caps before dialogue).
- add_character / update_character: data.name or data.character + data.description (age, look, personality).
- add_outfit: data.character, data.name, data.clothing, data.shoes, data.accessories, data.makeup.
- add_place / add_object: data.name, data.description (exact look), data.owner for objects.
- rewrite_scene: data.scene (number from the scene list), data.heading, data.visual, data.dialogue.
- set_scene_outfit: data.scene, data.character, data.outfit (an existing outfit name).
- develop_story, break_into_scenes, find_places_objects, make_cover, make_film: paid steps, data may be empty.
Use the exact names from the PROJECT facts for characters, outfits and scene numbers."""


def _short(text: str, n: int) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return text if len(text) <= n else text[: n - 1] + "…"


def project_context(p: dict, cast: list[dict], outfits: list[dict], elements: list[dict], scenes: list[dict],
                    production: Optional[dict], jobs: list[dict]) -> str:
    """What the assistant knows as fact: only what is saved."""
    import studio_identity as ident
    lines = [f"Title: {p.get('title') or '(none)'}",
             f"Idea / logline: {p.get('logline') or '(none yet)'}",
             f"Format: {p.get('kind')} · about {p.get('target_seconds')} s · {p.get('aspect_ratio')} · "
             f"genre {p.get('genre') or '(not set)'} · style {p.get('visual_style') or '(not set)'} · "
             f"language {p.get('language') or 'English'} · made as {p.get('production_style') or 'pictures'}"]
    bible = p.get("story_bible") or {}
    if bible:
        approved = (bible.get("_assistant_approval") or {}).get("approved")
        lines.append(f"Story plan ({'approved by the creator' if approved else 'not approved yet'}): "
                     f"{_short(bible.get('logline') or '', 300)} Conflict: {_short(bible.get('central_conflict') or '', 200)}. "
                     f"Ending: {_short(bible.get('ending') or '', 200)}")
    script = (p.get("script") or "").strip()
    lines.append(f"Screenplay: {'(none yet)' if not script else str(len(script.split())) + ' words. Opening: ' + _short(script, 1500)}")
    if cast:
        lines.append("Characters:")
        for c in cast[:20]:
            outs = [o["name"] for o in outfits if o["character_id"] == c["id"] and (o.get("status") or "saved") == "saved"]
            lines.append(f"- {c['name']}: {_short(ident.identity_text(c['name'], c.get('identity'), c.get('description') or ''), 260)}"
                         f"{' [identity locked]' if c.get('identity_locked_at') else ''}"
                         f"{' Outfits: ' + ', '.join(outs) if outs else ''}")
    else:
        lines.append("Characters: (none yet)")
    if elements:
        lines.append("Places: " + ("; ".join(e["name"] for e in elements if e["kind"] == "location") or "(none)"))
        lines.append("Objects: " + ("; ".join(e["name"] for e in elements if e["kind"] == "prop") or "(none)"))
    if scenes:
        lines.append(f"Scenes ({len(scenes)}):")
        for i, s in enumerate(scenes[:60], start=1):
            who = sorted({(d.get("character") or "").strip() for d in s.get("dialogue") or [] if d.get("character")})
            made = "video" if s.get("video_path") else "picture" if s.get("image_path") else "not made"
            lines.append(f"{i}. {s.get('heading') or ''}: {_short(s.get('visual') or '', 140)}"
                         f"{' Speakers: ' + ', '.join(who) if who else ''} [{made}]")
    else:
        lines.append("Scenes: (not broken into scenes yet)")
    if production:
        lines.append(f"One-click production: {production.get('status')} ({production.get('percent', 0)}%)"
                     f"{' – ' + production['problem'] if production.get('problem') else ''}")
    recent = [j for j in jobs[:8] if j.get("status") in ("queued", "running", "failed")]
    if recent:
        lines.append("Recent steps: " + "; ".join(f"{j['kind']} {j['status']}" for j in recent))
    return "\n".join(lines)


def build_prompt(context: str, summary: str, decisions: list[str], history: list[dict], message: str) -> str:
    convo = "\n".join(f"{'CREATOR' if m['role'] == 'user' else 'VIYO'}: {_short(m['content'], 1800)}" for m in history)
    dec = "\n".join(f"- {d}" for d in decisions[-30:]) or "(none yet)"
    return f"""{SYSTEM}

PROJECT (saved facts):
{context}

DECISIONS the creator made earlier:
{dec}

EARLIER CONVERSATION (summary, ideas only):
{summary or '(none)'}

RECENT MESSAGES:
{convo or '(this is the first message)'}

CREATOR: {message}

Reply as VIYO."""


def summary_prompt(old_summary: str, messages: list[dict]) -> str:
    convo = "\n".join(f"{'CREATOR' if m['role'] == 'user' else 'VIYO'}: {_short(m['content'], 1200)}" for m in messages)
    return f"""Update the running summary of a conversation between a filmmaker and their assistant. Keep what
matters for continuing the work: the creator's goals and preferences, ideas discussed (mark them as ideas),
options offered, questions still open. Do not state that anything was saved. At most 250 words.

CURRENT SUMMARY:
{old_summary or '(empty)'}

NEW MESSAGES:
{convo}"""


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
