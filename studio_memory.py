"""The assistant's long-term memory of one production (one owner, one project).

Kept on the project (studio_projects.assistant_memory), so every conversation
about the production shares it and a new chat doesn't start from nothing.
Facts are sorted into categories, and the assistant's own suggestions are kept
apart from what the creator decided:

  story        permanent story facts the creator stated or approved
  characters   character facts and relationships
  preferences  how the creator likes to work, tone, style, things to avoid
  decisions    approved creative decisions (creator said so, or applied a proposal)
  production   production decisions (format, style, what to make next, budget)
  questions    open questions still to settle
  ideas        the assistant's suggestions, NOT approved yet (never treated as facts)
  assumptions  things the assistant inferred but the creator never said (never facts)

Each item: {id, text, source ('creator' | 'applied' | 'assistant'), at}.
The creator can add, correct and remove items.
"""
from __future__ import annotations

import datetime
import re
import uuid
from typing import Optional

CATEGORIES = {
    "story": "Story facts",
    "characters": "Characters & relationships",
    "preferences": "Your preferences",
    "decisions": "Approved decisions",
    "production": "Production decisions",
    "questions": "Open questions",
    "ideas": "Ideas not approved yet",
    "assumptions": "Viyo's assumptions (not confirmed)",
}
LIMITS = {"ideas": 20, "questions": 20, "assumptions": 20}
_CONFIRM = re.compile(r"^\W*(yes|yeah|yep|ok|okay|sure|agreed|i agree|do it|let'?s do (it|that)|go (with|for) (it|that)|"
                      r"sounds good|perfect|approved?|that'?s (it|right|good))\b", re.I)
_STOPWORDS = {"the", "and", "that", "this", "with", "from", "have", "will", "into", "they", "their", "there", "about",
              "should", "would", "could", "very", "really", "just", "also", "make", "want"}
DEFAULT_LIMIT = 40
FACTS = ("story", "characters", "preferences", "decisions", "production")


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _text(x, n=300) -> str:
    return re.sub(r"\s+", " ", str(x or "")).strip()[:n]


def clean(memory) -> dict:
    out = {}
    for cat in CATEGORIES:
        items = []
        for it in (memory or {}).get(cat) or [] if isinstance(memory, dict) else []:
            if isinstance(it, str):
                it = {"text": it}
            if not isinstance(it, dict) or not _text(it.get("text")):
                continue
            items.append({"id": str(it.get("id") or uuid.uuid4().hex[:10]), "text": _text(it.get("text")),
                          "source": it.get("source") if it.get("source") in ("creator", "applied", "assistant") else "creator",
                          "at": it.get("at") or _now()})
        out[cat] = items[-LIMITS.get(cat, DEFAULT_LIMIT):]
    return out


def _same(a: str, b: str) -> bool:
    norm = lambda s: re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()
    return norm(a) == norm(b)


def add(memory: dict, category: str, text: str, source: str = "creator") -> tuple[dict, Optional[dict]]:
    """Adds a fact (no duplicates). The assistant's own suggestions can only go
    to 'ideas'; facts need the creator. Returns (memory, item or None)."""
    memory = clean(memory)
    text = _text(text)
    if category not in CATEGORIES or not text:
        return memory, None
    if source == "assistant" and category in FACTS:
        # Never a fact without the creator: a suggested decision is an idea, an
        # inferred story/character detail is an assumption.
        category = "assumptions" if category in ("story", "characters") else "ideas"
    if any(_same(text, it["text"]) for it in memory[category]):
        return memory, None
    if category in FACTS:
        # Once decided, it's no longer just an idea or an open question.
        for cat in ("ideas", "questions", "assumptions"):
            memory[cat] = [it for it in memory[cat] if not _same(it["text"], text)]
    item = {"id": uuid.uuid4().hex[:10], "text": text, "source": source, "at": _now()}
    memory[category] = (memory[category] + [item])[-LIMITS.get(category, DEFAULT_LIMIT):]
    return memory, item


def find(memory: dict, item_id: str) -> Optional[tuple[str, dict]]:
    for cat, items in clean(memory).items():
        for it in items:
            if it["id"] == item_id:
                return cat, it
    return None


def edit(memory: dict, item_id: str, text: Optional[str] = None, category: Optional[str] = None) -> dict:
    """The creator corrects an item (text and/or category)."""
    memory = clean(memory)
    hit = find(memory, item_id)
    if not hit:
        raise KeyError(item_id)
    cat, it = hit
    memory[cat] = [x for x in memory[cat] if x["id"] != item_id]
    new = dict(it, text=_text(text) if text is not None else it["text"], source="creator", at=_now())
    target = category if category in CATEGORIES else cat
    memory[target] = (memory[target] + [new])[-LIMITS.get(target, DEFAULT_LIMIT):]
    return memory


def remove(memory: dict, item_id: str) -> dict:
    memory = clean(memory)
    if not find(memory, item_id):
        raise KeyError(item_id)
    return {cat: [x for x in items if x["id"] != item_id] for cat, items in memory.items()}


def _content(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9']{4,}", (text or "").lower()) if w not in _STOPWORDS}


def from_creator(text: str, creator_message: Optional[str], memory: Optional[dict] = None) -> bool:
    """Is this really something the creator said or confirmed? The model's own
    claim isn't enough: most of its words must be in the creator's message, or
    the creator plainly agreed ("yes", "let's do that") to an idea already noted."""
    if creator_message is None:
        return True          # older callers that don't pass the message
    words = _content(text)
    said = _content(creator_message)
    if words and len(words & said) / len(words) >= 0.5:
        return True
    if _CONFIRM.search(creator_message or "") and memory:
        return any(_same(text, it["text"]) or (_content(it["text"]) and _content(it["text"]) <= words | said)
                   for it in clean(memory)["ideas"][-5:])
    return False


def apply_updates(memory: dict, updates, creator_message: Optional[str] = None) -> tuple[dict, list[dict]]:
    """What the assistant noted from the creator's latest message:
    [{category, text, from_creator}]. Facts need the creator (checked against
    their actual message); anything else is kept as an idea or assumption.
    Returns (memory, what was added)."""
    added = []
    memory = clean(memory)
    for u in (updates or [])[:8]:
        if not isinstance(u, dict) or not isinstance(u.get("text"), str):
            continue
        claimed = u.get("from_creator") is True
        source = "creator" if claimed and from_creator(u["text"], creator_message, memory) else "assistant"
        memory, item = add(memory, str(u.get("category") or ""), u["text"], source)
        if item:
            added.append(item)
    return clean(memory), added


def render(memory: dict) -> str:
    """For the prompt: facts first, then open questions, then ideas (labelled as such)."""
    memory = clean(memory)
    lines = []
    for cat, label in CATEGORIES.items():
        items = memory[cat]
        if not items:
            continue
        tag = " (NOT approved: only ideas)" if cat == "ideas" else ""
        lines.append(f"{label}{tag}:")
        lines += [f"- {it['text']}" for it in items[-25:]]
    return "\n".join(lines) or "(nothing remembered yet)"


def count(memory: dict) -> int:
    return sum(len(v) for v in clean(memory).values())
