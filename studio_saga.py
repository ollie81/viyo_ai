"""The saga: the larger story planned BEFORE episodes are written, so a series
grows instead of repeating itself.

It holds the premise, the main and supporting storylines, character arcs (with
history, fears, secrets, inner and outer conflicts and how each changes), the
relationship map, secrets (who knows, when they come out), long mysteries
(their hidden answers and when they're revealed), rivalries, planned turning
points and revelations by episode, season arcs, possible future seasons and
open questions. The creator chooses the length: a number of episodes, seasons,
or open-ended.

Each episode's plan is told where it sits in the saga (studio_story), and the
episode records track what has already been revealed, so a secret isn't
revealed twice and a turning point isn't skipped.
"""
from __future__ import annotations

import datetime
import json
import re
from typing import Optional

_S = {"type": "string"}
_I = {"type": "integer"}
SCHEMA = {
    "type": "object",
    "properties": {
        "premise": _S, "main_storyline": _S, "theme": _S,
        "storylines": {"type": "array", "items": {"type": "object", "properties": {
            "name": _S, "what": _S, "episodes": _S}, "required": ["name", "what"]}},
        "characters": {"type": "array", "items": {"type": "object", "properties": {
            "name": _S, "role": _S, "age": _S, "history": _S, "wants": _S, "fears_losing": _S, "hides": _S,
            "regret": _S, "strength": _S, "weakness": _S, "inner_conflict": _S, "outer_conflict": _S,
            "vulnerability": _S, "arc": {"type": "string", "description": "Who they are at the start, what changes them, who they become."},
            "look": _S}, "required": ["name", "role", "wants", "fears_losing", "hides", "inner_conflict", "arc"]}},
        "relationships": {"type": "array", "items": {"type": "object", "properties": {
            "between": _S, "kind": _S, "tension": _S, "how_it_changes": _S}, "required": ["between", "kind"]}},
        "secrets": {"type": "array", "items": {"type": "object", "properties": {
            "secret": _S, "kept_by": _S, "known_by": _S, "reveal_episode": _I, "consequence": _S},
            "required": ["secret", "kept_by", "reveal_episode"]}},
        "mysteries": {"type": "array", "items": {"type": "object", "properties": {
            "question": _S, "answer": _S, "clues": _S, "reveal_episode": _I}, "required": ["question", "answer", "reveal_episode"]}},
        "rivalries": {"type": "array", "items": {"type": "object", "properties": {"between": _S, "over": _S},
                                                 "required": ["between", "over"]}},
        "turning_points": {"type": "array", "items": {"type": "object", "properties": {
            "episode": _I, "what": _S}, "required": ["episode", "what"]}},
        "seasons": {"type": "array", "items": {"type": "object", "properties": {
            "number": _I, "title": _S, "episodes": _I, "arc": _S, "ends_with": _S},
            "required": ["number", "arc", "ends_with"]}},
        "future_seasons": {"type": "array", "items": _S},
        "open_questions": {"type": "array", "items": _S},
    },
    "required": ["premise", "main_storyline", "characters", "relationships", "secrets", "mysteries", "turning_points",
                 "seasons"],
}
LENGTHS = ("episodes", "seasons", "open")


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def length_text(length: dict) -> str:
    mode = (length or {}).get("mode") or "open"
    n = (length or {}).get("count") or 0
    if mode == "episodes" and n:
        return f"a limited series of exactly {n} episodes with a complete ending"
    if mode == "seasons" and n:
        return f"{n} season(s); plan each season's arc and how it ends"
    return "an open-ended series: plan the first season in detail and where later seasons could go, without a fixed end"


def prompt(brief: dict, world_text: str, length: dict, existing_cast: list[str]) -> str:
    return f"""You are the showrunner. Before any episode is written, plan the larger story so it can run for
many episodes without repeating itself or stretching artificially.

IDEA: {brief.get('logline') or brief.get('title')}
GENRE: {brief.get('genre') or 'drama'} · each episode about {int(brief.get('target_seconds') or 60)} seconds
LENGTH: {length_text(length)}
{world_text}{('Characters already in the story (keep them): ' + ', '.join(existing_cast) + chr(10)) if existing_cast else ''}
Plan:
- premise, main storyline and theme; 2-4 supporting storylines that feed the main one.
- characters who feel like real people: a personal history, what they want, what they're afraid of losing, what
  they hide, a regret, a strength and a weakness, an inner and an outer conflict, a vulnerability, and an arc (they
  change, sometimes through mistakes and misunderstandings). Nobody exists only to deliver information. Names,
  manners and looks belong to the story world.
- relationships (family, love, rivalry, debt, loyalty) with their tension and how they change.
- secrets: who keeps each, who knows, the episode it comes out, and what it costs.
- long mysteries: the question, the hidden answer, the clues, the episode it's revealed (some stay open for later seasons).
- turning points by episode number, spread out so every few episodes something big changes.
- seasons: each one's arc and how it ends; possible future seasons; open questions.
Decisions have consequences; no reset to the status quo; each revelation raises a new question."""


def clean(raw, length: dict) -> dict:
    raw = raw if isinstance(raw, dict) else {}

    def t(x, n=600):
        return re.sub(r"\s+", " ", str(x or "")).strip()[:n]

    def items(key, fields, ints=()):
        out = []
        for d in (raw.get(key) or [])[:30]:
            if isinstance(d, dict):
                row = {f: (d.get(f) if isinstance(d.get(f), int) and f in ints else t(d.get(f), 500)) for f in fields if d.get(f) not in (None, "")}
                if row:
                    out.append(row)
        return out
    saga = {"premise": t(raw.get("premise")), "main_storyline": t(raw.get("main_storyline"), 1200), "theme": t(raw.get("theme")),
            "storylines": items("storylines", ("name", "what", "episodes")),
            "characters": items("characters", ("name", "role", "age", "history", "wants", "fears_losing", "hides", "regret",
                                               "strength", "weakness", "inner_conflict", "outer_conflict", "vulnerability",
                                               "arc", "look")),
            "relationships": items("relationships", ("between", "kind", "tension", "how_it_changes")),
            "secrets": items("secrets", ("secret", "kept_by", "known_by", "reveal_episode", "consequence"), ("reveal_episode",)),
            "mysteries": items("mysteries", ("question", "answer", "clues", "reveal_episode"), ("reveal_episode",)),
            "rivalries": items("rivalries", ("between", "over")),
            "turning_points": sorted(items("turning_points", ("episode", "what"), ("episode",)), key=lambda x: x.get("episode") or 0),
            "seasons": items("seasons", ("number", "title", "episodes", "arc", "ends_with"), ("number", "episodes")),
            "future_seasons": [t(x, 300) for x in (raw.get("future_seasons") or [])[:6] if t(x)],
            "open_questions": [t(x, 300) for x in (raw.get("open_questions") or [])[:10] if t(x)],
            "length": {"mode": (length or {}).get("mode") if (length or {}).get("mode") in LENGTHS else "open",
                       "count": int((length or {}).get("count") or 0)},
            "built_at": _now(), "approved": False, "revealed": []}
    return saga


def problems(saga: dict) -> list[str]:
    """What a showrunner would send back (no AI)."""
    out = []
    if len(saga.get("characters") or []) < 2:
        out.append("Fewer than two developed characters.")
    shallow = [c["name"] for c in saga.get("characters") or [] if sum(1 for k in ("history", "fears_losing", "hides",
               "inner_conflict", "arc") if c.get(k)) < 4]
    if shallow:
        out.append(f"Characters without enough depth: {', '.join(shallow)}.")
    eps = [tp.get("episode") for tp in saga.get("turning_points") or [] if tp.get("episode")]
    if len(eps) < 2:
        out.append("Fewer than two turning points.")
    if eps and len(eps) != len(set(eps)):
        out.append("Two turning points in the same episode: spread them out.")
    n = (saga.get("length") or {}).get("count") or 0
    if (saga.get("length") or {}).get("mode") == "episodes" and n:
        late = [e for e in eps if e > n] + [x.get("reveal_episode") for x in (saga.get("secrets") or []) + (saga.get("mysteries") or [])
                                            if (x.get("reveal_episode") or 0) > n]
        if late:
            out.append(f"Planned beyond the {n} episodes: episode(s) {sorted(set(late))}.")
    if not saga.get("secrets") and not saga.get("mysteries"):
        out.append("No secrets or mysteries to carry the series.")
    return out


def position(saga: Optional[dict], episode: int) -> str:
    """What THIS episode must do in the larger story (for the episode plan)."""
    if not saga or not saga.get("premise"):
        return ""
    revealed = set(saga.get("revealed") or [])
    due = [tp["what"] for tp in saga.get("turning_points") or [] if tp.get("episode") == episode]
    secrets = [s for s in saga.get("secrets") or [] if s.get("reveal_episode") == episode and s["secret"] not in revealed]
    mysteries = [m for m in saga.get("mysteries") or [] if m.get("reveal_episode") == episode and m["question"] not in revealed]
    keep = [s["secret"] for s in saga.get("secrets") or [] if (s.get("reveal_episode") or 0) > episode and s["secret"] not in revealed]
    keep_q = [m["question"] for m in saga.get("mysteries") or [] if (m.get("reveal_episode") or 0) > episode]
    season = next((s for s in saga.get("seasons") or [] if s.get("number") == 1), {})
    lines = [f"THE LARGER STORY ({length_text(saga.get('length') or {})}): {saga.get('premise')}",
             f"Main storyline: {saga.get('main_storyline')}",
             f"This is episode {episode}." + (f" Season arc: {season.get('arc')}" if season.get("arc") else "")]
    if due:
        lines.append("Turning point due IN THIS EPISODE: " + "; ".join(due))
    if secrets:
        lines.append("Secret to reveal now: " + "; ".join(f"{s['secret']} (kept by {s.get('kept_by')}; cost: {s.get('consequence') or '-'})" for s in secrets))
    if mysteries:
        lines.append("Mystery answered now: " + "; ".join(f"{m['question']} -> {m['answer']}" for m in mysteries))
    if keep:
        lines.append("Secrets that must STAY hidden in this episode (hint at most): " + "; ".join(keep[:6]))
    if keep_q:
        lines.append("Mysteries still open (deepen, don't answer): " + "; ".join(keep_q[:6]))
    if revealed:
        lines.append("Already revealed earlier (never reveal again): " + "; ".join(sorted(revealed)[:10]))
    for c in (saga.get("characters") or [])[:8]:
        lines.append(f"- {c['name']} ({c.get('role', '')}): wants {c.get('wants', '')}; afraid of losing "
                     f"{c.get('fears_losing', '')}; hides {c.get('hides', '')}; inner conflict {c.get('inner_conflict', '')}; "
                     f"arc {c.get('arc', '')}")
    return "\n".join(lines) + "\n\n"


def mark_revealed(saga: dict, episode: int) -> dict:
    """After an episode is written: what it was planned to reveal counts as revealed."""
    saga = json.loads(json.dumps(saga))
    done = set(saga.get("revealed") or [])
    done |= {s["secret"] for s in saga.get("secrets") or [] if s.get("reveal_episode") == episode}
    done |= {m["question"] for m in saga.get("mysteries") or [] if m.get("reveal_episode") == episode}
    saga["revealed"] = sorted(done)
    return saga


def summary_lines(saga: Optional[dict]) -> list[str]:
    if not saga or not saga.get("premise"):
        return ["LARGER STORY: not planned yet (propose plan_saga for a series)."]
    out = [f"LARGER STORY ({length_text(saga.get('length') or {})}, {'approved' if saga.get('approved') else 'NOT approved yet'}): "
           f"{saga['premise']}", f"  Main storyline: {saga.get('main_storyline')}"]
    for s in (saga.get("seasons") or [])[:4]:
        out.append(f"  Season {s.get('number')}: {s.get('arc')} – ends with {s.get('ends_with')}")
    for tp in (saga.get("turning_points") or [])[:8]:
        out.append(f"  Episode {tp.get('episode')}: {tp.get('what')}")
    for sct in (saga.get("secrets") or [])[:6]:
        out.append(f"  Secret ({sct.get('kept_by')}, out in episode {sct.get('reveal_episode')}"
                   f"{', ALREADY REVEALED' if sct['secret'] in (saga.get('revealed') or []) else ''}): {sct['secret']}")
    for p in problems(saga)[:3]:
        out.append(f"  Saga check: {p}")
    return out
