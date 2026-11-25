"""Story Intelligence stage 7: the episode planner. An approved novel becomes N drama episodes.

The plan
  One AI call reads the approved chapters' summaries and the outline's beats and splits the novel by dramatic shape,
  not page count: each episode covers a stretch of chapters (a long chapter can be shared by several episodes, in
  order), and has a title, what happens, its beats, an opening hook, a turn and a cliffhanger. Code checks the plan:
  exactly N episodes, in order, covering every chapter with no gap and no going back, each with a hook and a
  cliffhanger (the last episode resolves instead). If the check finds problems the plan is sent back once. The
  creator can edit titles, summaries and beats, then approves one exact version.

Source passages are never invented
  An episode's source text is cut from the approved chapters themselves, by code: the chapters it covers, and when
  several episodes share a chapter, that chapter's paragraphs split between them in order. No AI quotes are trusted,
  so an episode can only be adapted from words the novel really contains.

Approving the plan
  Creates one ordinary episode production per planned episode, in a new series of its own, each with the same
  settings, cast (voices, portraits, identities, outfits) and places as the novel's production, and the novel's
  Story Bible with every reveal placed in its episode. studio_story_episodes records which chapters each episode
  covers, so "the story at episode 30" is an exact point in the memory.

Writing an episode
  The existing writers' room (studio_story) is given the episode's plan, its source passages and the story memory
  AT THAT EPISODE (who is alive, who knows what, open threads, locked facts first), in addition to the episodes
  before it. Everything after the screenplay (breakdown, shots, voices, video, render, quality checks) is unchanged.
"""
from __future__ import annotations

import re
from typing import Optional

MAX_SOURCE_WORDS = 3500        # how much of the novel one episode's writer sees word for word
_STR = {"type": "string"}
PLAN_SCHEMA = {
    "type": "object",
    "properties": {"episodes": {"type": "array", "items": {"type": "object", "properties": {
        "number": {"type": "integer"}, "title": _STR, "chapter_from": {"type": "integer"},
        "chapter_to": {"type": "integer"}, "summary": _STR, "beats": {"type": "array", "items": _STR},
        "hook": _STR, "turn": _STR, "cliffhanger": _STR},
        "required": ["number", "title", "chapter_from", "chapter_to", "summary", "beats", "hook", "cliffhanger"]}}},
    "required": ["episodes"],
}


def _t(x, n: int = 600) -> str:
    return re.sub(r"\s+", " ", str(x or "")).strip()[:n]


def plan_prompt(story: dict, chapters: list[dict], episodes: int, seconds: int, note: str = "") -> str:
    outline = story.get("outline") or {}
    plans = {int(c.get("number") or 0): c for c in outline.get("chapters") or []}
    lines = [
        f"Adapt the approved novel \"{story.get('title') or outline.get('title') or ''}\" into EXACTLY {episodes} "
        f"short-drama episodes of about {seconds} seconds each.",
        f"Premise: {outline.get('premise') or ''}",
        f"Genre and tone: {outline.get('genre') or ''}; {outline.get('tone') or ''}.",
        f"How it ends: {(outline.get('ending') or {}).get('description') or ''}",
        "",
        "Split by dramatic shape, not by page count. Each episode needs: an opening hook (the first moment already "
        "belongs to the conflict), a turn in the middle, and a cliffhanger that makes the viewer need the next "
        "episode. The LAST episode resolves the story instead of a cliffhanger (its cliffhanger field says how it "
        "ends).",
        "Each episode covers chapter_from..chapter_to (inclusive). Episodes go in order and together cover every "
        f"chapter from 1 to {len(chapters)} with no gap and no going back. With more episodes than chapters, several "
        "episodes share a chapter (the next one starts at the same chapter the previous one ended in).",
        "Keep what the novel established: names, who knows what and when, deaths, and the order of revelations. "
        "Don't invent new main characters.",
        "",
        "THE CHAPTERS (as approved):",
    ]
    for c in chapters:
        p = plans.get(int(c["number"]), {})
        lines.append(f"{c['number']}. {c.get('title') or ''}: {_t(c.get('summary'), 900)}"
                     + (f" Beats: {' | '.join(p.get('beats') or [])}" if p.get("beats") else ""))
    if note.strip():
        lines += ["", f"THE CREATOR ASKS: {note.strip()[:1500]}"]
    return "\n".join(lines)


def clean_plan(raw: dict) -> list[dict]:
    out = []
    for i, e in enumerate((raw or {}).get("episodes") or []):
        if not isinstance(e, dict):
            continue
        try:
            a, b = int(e.get("chapter_from") or 0), int(e.get("chapter_to") or 0)
        except (TypeError, ValueError):
            a, b = 0, 0
        out.append({"number": i + 1, "title": _t(e.get("title"), 120), "chapter_from": a, "chapter_to": b,
                    "summary": _t(e.get("summary"), 1200), "beats": [_t(x, 300) for x in (e.get("beats") or []) if _t(x)][:12],
                    "hook": _t(e.get("hook"), 400), "turn": _t(e.get("turn"), 400), "cliffhanger": _t(e.get("cliffhanger"), 400)})
    return out


def plan_problems(plan: list[dict], chapters: int, episodes: int) -> list[str]:
    """What's wrong with a plan (empty = fine): count, order and coverage, and each episode's dramatic shape."""
    out = []
    if len(plan) != episodes:
        out.append(f"The plan has {len(plan)} episodes; it must have exactly {episodes}.")
    prev_to = 0
    for e in plan:
        n, a, b = e["number"], e["chapter_from"], e["chapter_to"]
        if not (1 <= a <= b <= chapters):
            out.append(f"Episode {n} covers chapters {a}-{b}, outside 1-{chapters}.")
        elif prev_to and a not in (prev_to, prev_to + 1):
            out.append(f"Episode {n} starts at chapter {a} but the one before ended at chapter {prev_to} "
                       "(no gaps, no going back).")
        elif not prev_to and a != 1:
            out.append("The first episode must start at chapter 1.")
        prev_to = max(prev_to, b)
        if not e["title"]:
            out.append(f"Episode {n} has no title.")
        if not e["hook"]:
            out.append(f"Episode {n} has no opening hook.")
        if not e["cliffhanger"]:
            out.append(f"Episode {n} has no cliffhanger (or, for the last one, its ending).")
        if len(e["beats"]) < 2:
            out.append(f"Episode {n} needs at least two beats.")
    if plan and prev_to != chapters:
        out.append(f"The episodes end at chapter {prev_to}; they must reach chapter {chapters}.")
    return out


def fix_prompt(base: str, plan: list[dict], problems: list[str]) -> str:
    import json
    return (base + "\n\nYOUR PREVIOUS PLAN HAS THESE PROBLEMS:\n" + "\n".join(f"- {p}" for p in problems) +
            "\nHere it is; return it fixed:\n" + json.dumps({"episodes": plan}, ensure_ascii=False))


def episode_for_chapter(plan: list[dict], chapter: int) -> int:
    """The first episode that covers a chapter (0 if none)."""
    return next((e["number"] for e in plan if e["chapter_from"] <= chapter <= e["chapter_to"]), 0)


def _paragraphs(text: str) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n|\n", text or "") if p.strip()]
    if len(paras) <= 1:                           # one long block: split by sentences into ~8 parts
        sentences = re.split(r"(?<=[.!?])\s+", (text or "").strip())
        size = max(1, len(sentences) // 8)
        paras = [" ".join(sentences[i:i + size]) for i in range(0, len(sentences), size)]
    return [p for p in paras if p]


def source_passages(plan: list[dict], number: int, texts: dict, max_words: int = MAX_SOURCE_WORDS) -> list[dict]:
    """The novel's own words for one episode: each chapter it covers, and a shared chapter split by paragraphs
    between the episodes that share it, in order. Trimmed evenly to max_words."""
    e = next((x for x in plan if x["number"] == number), None)
    if not e:
        return []
    out = []
    for c in range(e["chapter_from"], e["chapter_to"] + 1):
        sharing = [x["number"] for x in plan if x["chapter_from"] <= c <= x["chapter_to"]]
        paras = _paragraphs(texts.get(c) or "")
        if not paras:
            continue
        k, i = len(sharing), sharing.index(number)
        start, end = round(i * len(paras) / k), round((i + 1) * len(paras) / k)
        part = paras[start:max(end, start + 1)]
        out.append({"chapter": c, "part": f"{i + 1} of {k}" if k > 1 else "", "text": "\n".join(part)})
    words = sum(len(p["text"].split()) for p in out)
    if words > max_words and out:
        keep = max_words / words
        for p in out:
            w = p["text"].split()
            p["text"] = " ".join(w[:max(60, int(len(w) * keep))]) + " …"
    return out


def place_reveals(saga: dict, plan: list[dict]) -> dict:
    """The novel's saga with every reveal and turning point placed in the episode that covers its chapter."""
    s = dict(saga or {})
    for key, field in (("secrets", "reveal_chapter"), ("mysteries", "reveal_chapter"), ("turning_points", "chapter")):
        s[key] = [{**x, ("episode" if key == "turning_points" else "reveal_episode"):
                   episode_for_chapter(plan, int(x.get(field) or 0)) if x.get(field) else 0} for x in s.get(key) or []]
    s["length"] = {"mode": "episodes", "count": len(plan)}
    s["episodes"] = [{"number": e["number"], "title": e["title"], "chapters": [e["chapter_from"], e["chapter_to"]]}
                     for e in plan]
    return s


def block(episode: dict, memory: str = "") -> str:
    """What the writers' room is told about the episode it is adapting (studio_story.plan_prompt / draft_prompt)."""
    if not episode:
        return ""
    lines = [f"THIS EPISODE ADAPTS THE APPROVED NOVEL — episode {episode.get('number')} of {episode.get('of')}: "
             f"\"{episode.get('title') or ''}\" (chapters {episode.get('chapter_from')}-{episode.get('chapter_to')}).",
             f"What happens: {episode.get('summary') or ''}",
             "Beats, in order: " + " | ".join(episode.get("beats") or []),
             f"Opening hook: {episode.get('hook') or ''}"]
    if episode.get("turn"):
        lines.append(f"The turn: {episode['turn']}")
    lines.append(("How it ends: " if episode.get("last") else "Cliffhanger: ") + (episode.get("cliffhanger") or ""))
    lines.append("Stay faithful to the novel: the same people, events, order of revelations and who knows what. "
                 "Compress and dramatise for the screen; don't add new main characters or reveal anything early.")
    if memory:
        lines += ["", memory[:5000]]
    passages = episode.get("passages") or []
    if passages:
        lines += ["", "THE NOVEL'S OWN WORDS FOR THIS EPISODE (adapt from these):"]
        for p in passages:
            lines.append(f"[Chapter {p['chapter']}{' part ' + p['part'] if p.get('part') else ''}]\n{p['text']}")
    return "\n".join(lines) + "\n\n"
