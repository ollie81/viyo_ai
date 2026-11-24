"""Story Intelligence stage 3: writing the novel chapter by chapter, and reading each chapter back into the story
memory.

Writing
  Each chapter is written from three things, never from the AI's memory of earlier chapters: the approved outline
  (this chapter's purpose, summary, beats, point of view and emotion, with its neighbours for flow), the story
  canon AT THIS CHAPTER (studio_canon.retrieve: who is alive and where, who knows which secret, relationships,
  lasting traits, open threads, threads due, recent events; creator-locked facts first), and how the previous
  chapter actually ended (its last passage and summary). The prompt asks for literature: point of view held,
  scenes not summaries, sensory atmosphere, gestures and silences that carry feeling, subtext in dialogue, distinct
  voices, foreshadowing planted and paid off as planned, no secret revealed before its chapter, and an ending that
  pulls the reader on.
  Chapters are written in order, and chapter N+1 can only be written once the creator approved chapter N, so every
  chapter is written from approved memory.

Reading back
  After writing, the chapter is read again (a second AI call) for what it establishes: new characters, places and
  objects; facts (status, location, relationships, knowledge, traits); events; new threads; threads it resolves;
  and claims (who appears, speaks, acts, knows or is somewhere, marking flashbacks, dreams and lies). Everything is a
  PROPOSAL (studio_canon.CanonRepo.ingest) and must quote the chapter; a quote that isn't in the text is rejected.
  WHERE a fact happens inside the chapter (its seq) is worked out from where its quote is in the text, not taken
  from the AI. The claims are checked against the canon plus the chapter's own new facts (studio_canon.check), so a
  dead character speaking or a secret known too early is reported before the creator approves anything.

Pure logic here: prompts, schemas, positions and cleaning. The job runs in studio_worker.do_chapter.
"""
from __future__ import annotations

import re
from typing import Optional

SEQ_SCALE = 1000          # a chapter's beats are numbered 0..999 by where they are in the text
TAIL_WORDS = 700          # how much of the previous chapter the writer sees word for word
MIN_LENGTH_SHARE = 0.5    # a chapter shorter than half its target is treated as cut off


def words(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def tail(text: str, n: int = TAIL_WORDS) -> str:
    w = (text or "").split()
    return " ".join(w[-n:])


def chapter_plan(outline: dict, n: int) -> Optional[dict]:
    return next((c for c in outline.get("chapters") or [] if int(c.get("number") or 0) == n), None)


def clean_text(text: str, title: str = "") -> str:
    """The chapter's prose: no markdown fences, no repeated "Chapter N: title" heading at the top."""
    t = (text or "").strip()
    t = re.sub(r"^```[a-z]*\s*|\s*```$", "", t).strip()
    lines = t.splitlines()
    if lines and re.match(r"^\s*(#+\s*)?(chapter|chapitre|igice)\b", lines[0], re.I):
        lines = lines[1:]
    elif lines and title and lines[0].strip().strip("#*_ ").lower() == title.strip().lower():
        lines = lines[1:]
    return "\n".join(lines).strip()


def seq_of(quote: str, text: str) -> Optional[int]:
    """Where in the chapter the quote is (0..999), or None if it isn't there."""
    from studio_story_check import _norm, find_quote
    at = find_quote(quote, text)
    if at is None:
        return None
    total = max(1, len(" " + _norm(text) + " "))
    return min(SEQ_SCALE - 1, int(at * SEQ_SCALE / total))


def write_prompt(story: dict, outline: dict, n: int, memory: str, prev: Optional[dict], world_text: str = "",
                 craft: str = "", note: str = "", target_words: int = 3000) -> str:
    plan = chapter_plan(outline, n) or {}
    total = len(outline.get("chapters") or [])
    before, after = chapter_plan(outline, n - 1), chapter_plan(outline, n + 1)
    chars = "; ".join(f"{c['name']} ({c.get('role') or ''}: {c.get('personality') or ''}; voice: {c.get('voice') or ''}; "
                      f"looks: {c.get('appearance') or ''})" for c in outline.get("characters") or [])
    lines = [
        f"You are writing chapter {n} of {total} of the novel \"{outline.get('title') or story.get('title') or ''}\".",
        f"Premise: {outline.get('premise') or ''}",
        f"Genre and tone: {outline.get('genre') or ''}; {outline.get('tone') or ''}. The reader should feel: "
        f"{outline.get('audience_feeling') or ''}. Themes: {', '.join(outline.get('themes') or [])}.",
        f"Point of view of the novel: {outline.get('point_of_view') or 'close third person'}.",
        f"Characters: {chars}",
        "",
        f"THIS CHAPTER — {n}. {plan.get('title') or ''}",
        f"Purpose: {plan.get('purpose') or ''}",
        f"What happens: {plan.get('summary') or ''}",
        "Beats, in order: " + " | ".join(plan.get("beats") or []),
        f"Point of view: {plan.get('pov') or 'as the novel'}. Emotion to leave the reader with: {plan.get('emotion') or ''}.",
    ]
    if plan.get("setups"):
        lines.append("Plant here (subtly, so it pays off later): " + " | ".join(plan["setups"]))
    if plan.get("payoffs"):
        lines.append("Pay off here: " + " | ".join(plan["payoffs"]))
    if before:
        lines.append(f"The chapter before was planned as: {before.get('summary') or ''}")
    if after:
        lines.append(f"The next chapter will be: {after.get('summary') or ''} (lead towards it; don't write it)")
    if prev:
        lines += ["", f"HOW CHAPTER {n - 1} ACTUALLY ENDED (continue from here; keep every detail consistent):",
                  f"Summary: {prev.get('summary') or ''}", f"Its last passage: …{tail(prev.get('text') or '')}"]
    if memory:
        lines += ["", memory]
    if world_text:
        lines += ["", "THE STORY WORLD (never contradict it):", world_text[:2500]]
    if craft:
        lines += ["", craft]
    lines += ["", "HOW TO WRITE IT:",
              f"- About {target_words} words of finished prose. Write the chapter only: no title, no notes.",
              "- Scenes, not summary: let moments play out with action, dialogue and silence.",
              "- Hold the point of view: only what this character can see, hear, know and feel.",
              "- Show feeling through the body, gestures, small actions and what people don't say; let dialogue "
              "carry subtext, and give each character their own way of speaking.",
              "- Ground every scene in its place and time with a few precise sensory details (light, sound, smell, "
              "texture), and keep the world's facts.",
              "- Keep every fact in the STORY CANON true. A character only knows what the canon says they know; no "
              "secret comes out before its planned chapter. Characters who are dead stay dead (a flashback must be "
              "clearly marked as one).",
              "- Introduce a new named character only if the chapter truly needs one.",
              "- End the chapter with a pull: a question, a turn, a decision or an image the reader can't leave.",
              "- Don't imitate a named existing book or author."]
    if note.strip():
        lines += ["", f"THE CREATOR ASKS FOR THIS IN THIS VERSION: {note.strip()[:1500]}"]
    return "\n".join(lines)


_STR = {"type": "string"}
EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": _STR,
        "entities": {"type": "array", "items": {"type": "object", "properties": {
            "name": _STR, "kind": _STR, "quote": _STR}, "required": ["name", "kind", "quote"]}},
        "facts": {"type": "array", "items": {"type": "object", "properties": {
            "kind": _STR, "subject": _STR, "subject_kind": _STR, "object": _STR, "object_kind": _STR,
            "attribute": _STR, "value": _STR, "statement": _STR, "quote": _STR},
            "required": ["kind", "subject", "statement", "quote"]}},
        "events": {"type": "array", "items": {"type": "object", "properties": {
            "summary": _STR, "participants": {"type": "array", "items": _STR}, "story_time": _STR, "quote": _STR},
            "required": ["summary", "quote"]}},
        "threads": {"type": "array", "items": {"type": "object", "properties": {
            "kind": _STR, "description": _STR, "entities": {"type": "array", "items": _STR}, "quote": _STR},
            "required": ["description", "quote"]}},
        "resolved_threads": {"type": "array", "items": {"type": "object", "properties": {
            "ref": _STR, "quote": _STR}, "required": ["ref", "quote"]}},
        "claims": {"type": "array", "items": {"type": "object", "properties": {
            "type": _STR, "subject": _STR, "object": _STR, "mode": _STR, "as_of_chapter": {"type": "integer"},
            "quote": _STR}, "required": ["type", "subject", "quote"]}},
    },
    "required": ["summary", "facts", "claims"],
}


def extract_prompt(n: int, text: str, names: list[str], open_threads: list[dict]) -> str:
    lines = [
        f"Read chapter {n} of a novel and report what it ESTABLISHES, for the story's memory. Every item needs a "
        "quote: words copied EXACTLY from the chapter (at least 4 words) that show it. Report only what the text "
        "says or clearly shows; never what you expect or infer.",
        "",
        "Known names (use exactly these spellings for them): " + ", ".join(names),
        "",
        "- summary: 3-5 sentences of what happens.",
        "- entities: NEW named characters, places or objects that matter (kind: character, place, object, secret).",
        "- facts: kind is one of status (alive, dead, missing, imprisoned...), location (subject is at object), "
        "relationship (attribute = family / romance / work / loyalty, value = what it is), knows (subject learns or "
        "knows the secret named in object), trait (attribute = e.g. age, scar, fear; value). Only CHANGES or NEW "
        "facts from this chapter. Not facts inside a dream, a lie, a rumour or a flashback.",
        "- events: what happens that later chapters must remember.",
        "- threads: new setups, foreshadowing, mysteries or promises the story will owe the reader.",
        "- resolved_threads: earlier threads this chapter pays off (use the ref).",
        "- claims: every time a known character appears, speaks or acts (type appears / speaks / acts), knows a "
        "secret (type knows, object = the secret), or a character or important object is somewhere (type located, "
        "object = the place). mode: present, flashback (give as_of_chapter: the chapter whose time it shows, 0 = "
        "before the book), dream, vision, lie, rumor or hypothetical.",
    ]
    if open_threads:
        lines += ["", "OPEN THREADS (ref: description):"] + [f"{t['ref']}: {t['description']}" for t in open_threads]
    lines += ["", f"CHAPTER {n}:", text]
    return "\n".join(lines)


def claims_from(raw: dict, n: int, text: str) -> list[dict]:
    """The read-back's claims in studio_canon.check's form, placed by where their quote is in the chapter."""
    out = []
    for c in raw.get("claims") or []:
        seq = seq_of(c.get("quote") or "", text)
        if seq is None:
            continue                                     # no evidence in the text: not a claim
        mode = (c.get("mode") or "present").lower()
        item = {"type": (c.get("type") or "").lower(), "subject": c.get("subject") or "", "object": c.get("object") or "",
                "chapter": n, "seq": seq, "mode": mode, "quote": c.get("quote") or ""}
        if mode in ("flashback", "flashforward") and c.get("as_of_chapter") is not None:
            item["as_of"] = int(c["as_of_chapter"])
        out.append(item)
    return out


def placed(items: list[dict], text: str) -> list[dict]:
    """Facts/events with their seq taken from where their quote is (items whose quote isn't found keep seq 0 and
    are rejected by the canon's own quote check)."""
    out = []
    for it in items or []:
        s = seq_of(it.get("quote") or "", text)
        out.append({**it, "seq": s if s is not None else 0})
    return out
