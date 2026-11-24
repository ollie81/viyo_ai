"""The story canon: a permanent, time-aware memory of one story, the novel and every episode adapted from it.

It remembers who exists, what is true about them and WHEN it is true, so a 55-episode drama can't forget what
chapter 3 established or bring back someone who died in chapter 5.

Time
  A point in the story is (chapter, seq): a chapter and a beat inside it, in reading order. Each fact is true from
  valid_from (inclusive) until valid_until (exclusive; empty = still true). An episode is a stretch of chapters
  (studio_story_episodes), so "the state at episode 30" is an exact point.

Trust
  Every record (entity, fact, event, thread) has a status:
    proposed        an AI suggestion: kept with its evidence, shown for review, NEVER used as canon;
    approved        the creator accepted it;
    creator_locked  the creator fixed it: it beats any conflicting record, whatever its date;
    retired         no longer used, kept for history.
  Only approved and creator_locked records are canon. Only the creator can approve, lock or retire (the database
  refuses an approved or locked row without a decision on record). An AI fact must carry a quote that is found in
  the passage it came from (the same check the story check uses: studio_story_check.find_quote), or it isn't kept.

What it answers
  state_at(point)   the story as it stands at that point: who is alive, where people and objects are, who knows
                    which secret, relationships, traits, rules, open threads, what has happened so far;
  retrieve(task)    only what one writing task needs: the entities involved (named, or found in its text), their
                    state, every lasting fact about them however early it was established, locked facts first,
                    open threads, relevant secrets and recent events;
  check(claims)     contradictions in a new passage: a dead character acting, a character knowing a secret before
                    learning it, an object in two places, a character in two places at once. Flashbacks and
                    flash-forwards are checked against the time they show; dreams, lies, rumours and visions are not
                    checked as physical fact; an exception the creator approved silences that one rule for that
                    character and stretch of story, and is reported as such (never silently);
  audit()           problems inside the canon itself (two facts that can't both be true, a cause after its effect,
                    an overdue thread).

Pure logic over one loaded snapshot (Canon); CanonRepo does the reads and writes through the Studio's Store
(Supabase in production, PgStore in tests). Nothing here calls an AI provider, costs money or changes the
production pipeline.
"""
from __future__ import annotations

import datetime
import re
from dataclasses import dataclass, field
from typing import Iterable, Optional

STATUSES = ("proposed", "approved", "creator_locked", "retired")
CANON = ("approved", "creator_locked")
ENTITY_KINDS = ("character", "place", "object", "secret", "group")
FACT_KINDS = ("status", "location", "relationship", "knows", "trait", "rule")
THREAD_KINDS = ("setup", "foreshadowing", "mystery", "promise", "question")
RULES = ("dead_character_active", "knowledge_leak", "object_two_places", "object_location_conflict",
         "character_two_places", "cause_after_effect")

# Which facts can only have one value at a time, and what makes two of them "the same question".
_KEY = {"status": ("subject_id",), "location": ("subject_id",), "trait": ("subject_id", "attribute"),
        "relationship": ("subject_id", "object_id", "attribute"), "knows": ("subject_id", "object_id")}
_TIER = {"creator_locked": 3, "approved": 2, "pending": 1, "proposed": 0}

ACTIVE_CLAIMS = ("appears", "speaks", "acts")
TIME_SHIFTS = ("flashback", "flashforward")            # checked against the time the passage shows
NOT_PHYSICAL = ("dream", "vision", "lie", "rumor", "hypothetical", "imagined")   # not checked as fact
DEAD = ("dead",)
NOT_KNOWING = ("no", "false", "forgot", "forgets")

MAX_SEQ = 10 ** 6
END = (10 ** 9, 0)


class CanonError(ValueError):
    pass


class CanonConflict(CanonError):
    """Approving this would contradict a creator-locked fact."""
    def __init__(self, message: str, conflicts: list):
        super().__init__(message)
        self.conflicts = conflicts


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def norm(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", (text or "").lower().replace("’", "'")))


def as_point(x) -> tuple[int, int]:
    """(chapter, seq) from a tuple, a chapter number or {"chapter", "seq"}."""
    if x is None:
        return END
    if isinstance(x, tuple):
        return (int(x[0]), int(x[1]) if len(x) > 1 else 0)
    if isinstance(x, dict):
        return (int(x.get("chapter") or 0), int(x.get("seq") or 0))
    return (int(x), 0)


def valid_from(r: dict) -> tuple[int, int]:
    return (int(r.get("valid_from_chapter") or 0), int(r.get("valid_from_seq") or 0))


def valid_until(r: dict) -> tuple[int, int]:
    if r.get("valid_until_chapter") is None:
        return END
    return (int(r["valid_until_chapter"]), int(r.get("valid_until_seq") or 0))


def valid_at(r: dict, p: tuple[int, int]) -> bool:
    return valid_from(r) <= p < valid_until(r)


def overlaps(a: dict, b: dict) -> bool:
    return valid_from(a) < valid_until(b) and valid_from(b) < valid_until(a)


def key_of(f: dict) -> Optional[tuple]:
    cols = _KEY.get(f.get("kind"))
    return (f["kind"],) + tuple(f.get(c) or "" for c in cols) if cols else None


def _rank(f: dict) -> tuple:
    """Which of two answers to the same question holds: a creator-locked fact always wins; otherwise the one that
    started latest (the story moved on); at the same moment, approved beats pending beats proposed."""
    tier = _TIER.get(f.get("_tier") or f.get("status"), -1)
    return (tier == _TIER["creator_locked"], valid_from(f), tier, f.get("decided_at") or "", f.get("created_at") or "")


def _answer(f: dict) -> tuple:
    return ((f.get("value") or "").strip().lower(), f.get("object_id") or "")


@dataclass
class State:
    """The story at one point. Built only from canon (plus, when checking a new passage, its own pending facts)."""
    point: tuple[int, int]
    names: dict
    facts: list = field(default_factory=list)          # the winning fact for every question, plus rules
    overridden: list = field(default_factory=list)     # (loser, winner): a fact that lost to a stronger one
    events: list = field(default_factory=list)          # canon events that have happened by this point
    threads: list = field(default_factory=list)         # canon threads open at this point

    def _one(self, kind: str, subject: str, **match) -> Optional[dict]:
        for f in self.facts:
            if f["kind"] == kind and f.get("subject_id") == subject and all(f.get(k) == v for k, v in match.items()):
                return f
        return None

    def status_of(self, eid: str) -> Optional[str]:
        f = self._one("status", eid)
        return (f.get("value") or "").lower() if f else None

    def location_of(self, eid: str) -> Optional[str]:
        f = self._one("location", eid)
        return (f.get("object_id") or f.get("value")) if f else None

    def knows(self, character: str, secret: str) -> bool:
        f = self._one("knows", character, object_id=secret)
        return bool(f) and (f.get("value") or "").lower() not in NOT_KNOWING

    def relationships_of(self, eid: str) -> list[dict]:
        return [f for f in self.facts if f["kind"] == "relationship" and eid in (f.get("subject_id"), f.get("object_id"))]

    def traits_of(self, eid: str) -> dict:
        return {f.get("attribute"): f.get("value") for f in self.facts if f["kind"] == "trait" and f.get("subject_id") == eid}

    def name(self, eid: Optional[str]) -> str:
        return self.names.get(eid or "", eid or "")

    def lines(self, about: Optional[Iterable[str]] = None) -> list[str]:
        keep = set(about) if about is not None else None
        out = []
        for f in self.facts:
            ids = {f.get("subject_id"), f.get("object_id")}
            if keep is None or ids & keep or (f["kind"] == "rule" and not f.get("subject_id")):
                out.append(describe(f, self.names))
        return out


def describe(f: dict, names: dict) -> str:
    """One plain sentence for a fact, for prompts and review screens."""
    if f.get("statement"):
        return f["statement"]
    s, o, v, a = names.get(f.get("subject_id") or "", ""), names.get(f.get("object_id") or "", ""), f.get("value") or "", f.get("attribute") or ""
    k = f.get("kind")
    if k == "status":
        return f"{s} is {v}."
    if k == "location":
        return f"{s} is at {o or v}."
    if k == "relationship":
        return f"{s} — {a or v} — {o}."
    if k == "knows":
        return f"{s} {'does not know' if v.lower() in NOT_KNOWING else 'knows'} the secret: {o}."
    if k == "trait":
        return f"{s}: {a} = {v}."
    return v or a


class Canon:
    """One story's canon, loaded once. Every question is answered from this snapshot."""

    def __init__(self, story_id: str, entities=(), facts=(), events=(), threads=(), exceptions=(), episodes=()):
        self.story_id = story_id
        self.entities = [dict(e) for e in entities]
        self.facts = [dict(f) for f in facts]
        self.events = [dict(e) for e in events]
        self.threads = [dict(t) for t in threads]
        self.exceptions = [dict(x) for x in exceptions]
        self.episodes = sorted((dict(e) for e in episodes), key=lambda e: e["number"])
        self.by_id = {e["id"]: e for e in self.entities}
        self.names = {e["id"]: e["name"] for e in self.entities}

    # ------------------------------------------------------------------ names and points

    def entity(self, ref) -> Optional[dict]:
        """An entity by id, name or alias (not retired)."""
        if not ref:
            return None
        if ref in self.by_id:
            return self.by_id[ref]
        n = norm(ref)
        for e in self.entities:
            if e.get("status") == "retired":
                continue
            if norm(e["name"]) == n or n in {norm(a) for a in e.get("aliases") or []}:
                return e
        return None

    def eid(self, ref) -> Optional[str]:
        e = self.entity(ref)
        return e["id"] if e else None

    def mentioned_in(self, text: str) -> list[str]:
        """Entities whose name or alias appears in the text (whole words)."""
        hay = " " + norm(text) + " "
        out = []
        for e in self.entities:
            if e.get("status") == "retired":
                continue
            for n in [e["name"]] + list(e.get("aliases") or []):
                if n and f" {norm(n)} " in hay:
                    out.append(e["id"])
                    break
        return out

    def episode(self, number: int) -> dict:
        for e in self.episodes:
            if e["number"] == number:
                return e
        raise CanonError(f"Episode {number} isn't mapped to chapters yet.")

    def at_episode(self, number: int, end: bool = False) -> tuple[int, int]:
        """The point where an episode starts (or, with end=True, its last beat)."""
        e = self.episode(number)
        return (int(e["chapter_to"]), MAX_SEQ) if end else (int(e["chapter_from"]), 0)

    def point(self, at) -> tuple[int, int]:
        if isinstance(at, dict) and at.get("episode") is not None:
            return self.at_episode(int(at["episode"]), bool(at.get("end")))
        return as_point(at)

    # ------------------------------------------------------------------ the state at a point

    def state_at(self, at, pending: Iterable[dict] = (), include_proposed: bool = False) -> State:
        """The story at a point. Canon only; [pending] (a new passage's own facts, not yet reviewed) can be laid
        on top for checking that passage, ranked below anything approved. [include_proposed] adds unreviewed AI
        facts at the lowest rank, for review screens only; never for writing."""
        p = self.point(at)
        pool = [f for f in self.facts if f.get("status") in CANON]
        pool += [{**f, "_tier": "pending"} for f in pending]
        if include_proposed:
            pool += [f for f in self.facts if f.get("status") == "proposed"]
        live = [f for f in pool if valid_at(f, p)]
        best, overridden, rules = {}, [], []
        for f in live:
            k = key_of(f)
            if k is None:
                rules.append(f)
                continue
            cur = best.get(k)
            if cur is None or _rank(f) > _rank(cur):
                if cur is not None:
                    overridden.append((cur, f))
                best[k] = f
            else:
                overridden.append((f, cur))
        events = [e for e in self.events if e.get("status") in CANON and (int(e["chapter"]), int(e.get("seq") or 0)) <= p]
        threads = [t for t in self.threads if t.get("status") in CANON
                   and (int(t["planted_chapter"]), int(t.get("planted_seq") or 0)) <= p
                   and not (t.get("state") in ("resolved", "dropped") and t.get("resolved_chapter") is not None
                            and int(t["resolved_chapter"]) <= p[0])
                   and not (t.get("state") == "dropped" and t.get("resolved_chapter") is None)]
        facts = sorted(best.values(), key=lambda f: (f["kind"], valid_from(f))) + rules
        return State(p, self.names, facts, [(a, b) for a, b in overridden if _answer(a) != _answer(b)], events, threads)

    # ------------------------------------------------------------------ what a task needs

    def retrieve(self, task: dict) -> dict:
        """Only what one writing task needs. task = {"at": point or {"episode": n}, "entities": [names or ids],
        "text": outline/scene text to scan for names, "limit": max facts}. Canon only, creator-locked first."""
        p = self.point(task.get("at"))
        limit = int(task.get("limit") or 60)
        st = self.state_at(p)
        involved = []
        for r in task.get("entities") or []:
            i = self.eid(r)
            if i and i not in involved:
                involved.append(i)
        for i in self.mentioned_in(task.get("text") or ""):
            if i not in involved:
                involved.append(i)
        main = set(involved)
        # One step out: the people and places they're tied to right now, and the secrets they know or are kept from.
        near = set()
        for i in involved:
            for f in st.relationships_of(i):
                near |= {f.get("subject_id"), f.get("object_id")}
            loc = st.location_of(i)
            if loc in self.by_id:
                near.add(loc)
        near -= main | {None}

        scored = []
        for f in st.facts:
            ids = {f.get("subject_id"), f.get("object_id")}
            global_rule = f["kind"] == "rule" and not f.get("subject_id")
            if not (ids & main or global_rule or (ids & near and f["kind"] in ("status", "relationship"))):
                continue
            prio = 0 if f.get("status") == "creator_locked" else (1 if ids & main or global_rule else 3)
            if f["kind"] == "trait" and prio == 1:
                prio = 2          # lasting traits after the current state, however early they were established
            scored.append((prio, f))
        # Secrets in play: anyone involved who knows a secret, and the involved people it's kept from.
        secrets = {f.get("object_id") for f in st.facts if f["kind"] == "knows" and f.get("subject_id") in main}
        # ...and secrets an open thread ties to someone here (a planned reveal to them): the writer must know
        # what they don't know yet.
        for t in st.threads:
            ids = set(t.get("entities") or [])
            if ids & main:
                secrets |= {i for i in ids if self.by_id.get(i, {}).get("kind") == "secret"}
        secret_lines = []
        for s in secrets:
            if not s:
                continue
            knowers = [self.names.get(c, c) for c in main if self.by_id.get(c, {}).get("kind") == "character" and st.knows(c, s)]
            kept_from = [self.names.get(c, c) for c in main if self.by_id.get(c, {}).get("kind") == "character" and not st.knows(c, s)]
            truth = (self.by_id.get(s, {}).get("profile") or {}).get("truth") or self.names.get(s, s)
            secret_lines.append({"secret": self.names.get(s, s), "truth": truth, "known_by": knowers, "kept_from": kept_from})
        scored.sort(key=lambda x: (x[0], x[1]["kind"], valid_from(x[1])))
        facts = [f for _, f in scored][:limit]
        threads = [t for t in st.threads if not t.get("entities") or set(t["entities"]) & main
                   or (t.get("due_chapter") is not None and int(t["due_chapter"]) <= p[0] + 2)]
        events = [e for e in st.events if set(e.get("participants") or []) & main or e.get("location_id") in main]
        events = sorted(events, key=lambda e: (int(e["chapter"]), int(e.get("seq") or 0)))[-12:]
        return {"point": {"chapter": p[0], "seq": p[1]}, "entities": [self.names.get(i, i) for i in involved],
                "facts": [_public(f, self.names) for f in facts], "secrets": secret_lines,
                "threads": [_thread(t) for t in threads], "events": [_event(e, self.names) for e in events]}

    def render(self, task: dict) -> str:
        """retrieve() as a prompt block: what the writer must keep true."""
        r = self.retrieve(task)
        lines = [f"STORY CANON at chapter {r['point']['chapter']} (keep every fact true; LOCKED facts were fixed by the "
                 "creator and must never change):"]
        for f in r["facts"]:
            tag = "LOCKED " if f["status"] == "creator_locked" else ""
            lines.append(f"- {tag}{f['statement']} (from chapter {f['source_chapter'] if f['source_chapter'] is not None else '-'})")
        for s in r["secrets"]:
            lines.append(f"- SECRET \"{s['secret']}\": {s['truth']}. Known by: {', '.join(s['known_by']) or 'nobody here'}."
                         + (f" Kept from: {', '.join(s['kept_from'])}." if s["kept_from"] else ""))
        for t in r["threads"]:
            lines.append(f"- OPEN {t['kind']}: {t['description']}" + (f" (pay off by chapter {t['due_chapter']})" if t["due_chapter"] else ""))
        for e in r["events"]:
            lines.append(f"- Happened (chapter {e['chapter']}): {e['summary']}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ contradictions in a new passage

    def _excepted(self, rule: str, subject: Optional[str], chapter: int) -> Optional[dict]:
        for x in self.exceptions:
            if x["rule"] != rule or (x.get("subject_id") and x["subject_id"] != subject):
                continue
            if int(x.get("from_chapter") or 0) <= chapter and (x.get("until_chapter") is None or chapter <= int(x["until_chapter"])):
                return x
        return None

    def check(self, claims: Iterable[dict], pending: Iterable[dict] = ()) -> dict:
        """Contradictions in a new passage. Each claim: {"type": appears|speaks|acts|knows|located, "subject",
        "object" (the secret, or the place/holder), "chapter", "seq", "mode" (present, flashback, flashforward,
        dream, vision, lie, rumor, hypothetical, imagined), "as_of" (the point a flashback shows), "quote"}.
        [pending] are the passage's own new facts (a death or a move in the same passage)."""
        pending = list(pending)
        issues, excepted, skipped = [], [], []
        seen_at: dict = {}

        def report(rule, claim, subject, reason, evidence=None):
            item = {"rule": rule, "subject": self.names.get(subject, subject), "subject_id": subject,
                    "chapter": claim.get("chapter"), "seq": claim.get("seq") or 0, "mode": claim.get("mode") or "present",
                    "quote": claim.get("quote") or "", "reason": reason,
                    "evidence": _evidence(evidence, self.names) if evidence else None}
            x = self._excepted(rule, subject, int(claim.get("chapter") or 0))
            if x:
                excepted.append({**item, "exception": x.get("reason")})
            else:
                issues.append(item)

        for c in claims:
            mode = (c.get("mode") or "present").lower()
            subject = self.eid(c.get("subject"))
            if subject is None:
                skipped.append({"claim": c, "reason": "unknown character or object"})
                continue
            if mode in NOT_PHYSICAL:
                skipped.append({"claim": c, "reason": f"{mode}: not checked as fact"})
                continue
            here = as_point((c.get("chapter") or 0, c.get("seq") or 0))
            at = self.point(c.get("as_of")) if mode in TIME_SHIFTS and c.get("as_of") is not None else here
            st = self.state_at(at, pending=[f for f in pending if valid_from(f) <= at])
            kind = c.get("type")
            name = self.names.get(subject, subject)
            if kind in ACTIVE_CLAIMS:
                status = st.status_of(subject)
                if status in DEAD:
                    f = st._one("status", subject)
                    report("dead_character_active", c, subject,
                           f"{name} is dead from chapter {valid_from(f)[0]} but {kind} here"
                           + (f" (the {mode} shows chapter {at[0]})" if at != here else "") + ".", f)
            elif kind == "knows":
                secret = self.eid(c.get("object"))
                if secret and not st.knows(subject, secret):
                    learned = next((f for f in self.facts if f.get("status") in CANON and f["kind"] == "knows"
                                    and f.get("subject_id") == subject and f.get("object_id") == secret
                                    and (f.get("value") or "").lower() not in NOT_KNOWING), None)
                    when = f" (learns it in chapter {valid_from(learned)[0]})" if learned else " (never learns it in the canon)"
                    report("knowledge_leak", c, subject,
                           f"{name} knows \"{self.names.get(secret, secret)}\" here before learning it{when}.", learned)
            elif kind == "located":
                place = self.eid(c.get("object")) or c.get("object")
                is_object = self.by_id.get(subject, {}).get("kind") == "object"
                k = (subject, at)
                if k in seen_at and seen_at[k][0] != place:
                    report("object_two_places" if is_object else "character_two_places", c, subject,
                           f"{name} is at {self.names.get(place, place)} and at {self.names.get(seen_at[k][0], seen_at[k][0])} "
                           "at the same moment.", None)
                seen_at.setdefault(k, (place, c))
                if is_object:
                    where = st.location_of(subject)
                    if where and place and where != place:
                        f = st._one("location", subject)
                        report("object_location_conflict", c, subject,
                               f"{name} is at {self.names.get(where, where)} (since chapter {valid_from(f)[0]}) but appears "
                               f"at {self.names.get(place, place)} with no move in between.", f)
        return {"issues": issues, "excepted": excepted, "skipped": skipped}

    def conflicts_for(self, fact: dict) -> list[dict]:
        """Canon facts a proposed fact contradicts (same question, overlapping time, different answer), with a
        creator-locked one marked as blocking. A return from death is a conflict too."""
        out = []
        k = key_of(fact)
        if k is None:
            return out
        for f in self.facts:
            if f.get("status") not in CANON or f.get("id") == fact.get("id") or key_of(f) != k:
                continue
            if not overlaps(f, fact) or _answer(f) == _answer(fact):
                continue
            locked = f.get("status") == "creator_locked"
            resurrects = fact["kind"] == "status" and _answer(f)[0] in DEAD and _answer(fact)[0] not in DEAD \
                and valid_from(fact) > valid_from(f)
            if valid_from(fact) == valid_from(f):
                reason = "creator-locked" if locked else "gives a different answer for the same moment"
            elif locked:
                reason = "creator-locked"         # a locked fact beats any change, so the change would never show
            elif resurrects:
                reason = "returns from the dead"
            else:
                continue                          # a change over time (later) or an earlier stretch: not a contradiction
            out.append({"fact": _evidence(f, self.names), "blocking": locked, "reason": reason})
        return out

    # ------------------------------------------------------------------ problems inside the canon

    def audit(self, current_chapter: Optional[int] = None) -> list[dict]:
        out = []
        canon = [f for f in self.facts if f.get("status") in CANON]
        groups: dict = {}
        for f in canon:
            k = key_of(f)
            if k is not None:
                groups.setdefault(k, []).append(f)
        for k, fs in groups.items():
            for i, a in enumerate(fs):
                for b in fs[i + 1:]:
                    same = ((a.get("value") or "").lower(), a.get("object_id")) == ((b.get("value") or "").lower(), b.get("object_id"))
                    if not same and valid_from(a) == valid_from(b) and _TIER[a["status"]] == _TIER[b["status"]] and overlaps(a, b):
                        obj = self.by_id.get(a.get("subject_id"), {}).get("kind") == "object" and a["kind"] == "location"
                        out.append({"rule": "object_two_places" if obj else "conflicting_facts",
                                    "subject": self.names.get(a.get("subject_id"), ""),
                                    "facts": [_evidence(a, self.names), _evidence(b, self.names)],
                                    "reason": "Two canon facts give different answers from the same moment."})
        ev = {e["id"]: e for e in self.events if e.get("status") in CANON}

        def order(e):
            return float(e["story_order"]) if e.get("story_order") is not None else float(e["chapter"]) + int(e.get("seq") or 0) / MAX_SEQ
        for e in ev.values():
            for cid in e.get("causes") or []:
                c = ev.get(cid)
                if c and order(c) > order(e) and not self._excepted("cause_after_effect", None, int(e["chapter"])):
                    out.append({"rule": "cause_after_effect", "subject": e["summary"][:80],
                                "reason": f"\"{c['summary'][:80]}\" causes it but happens later in story time."})
        if current_chapter is not None:
            for t in self.threads:
                if t.get("status") in CANON and t.get("state") == "open" and t.get("due_chapter") is not None \
                        and int(t["due_chapter"]) < current_chapter:
                    out.append({"rule": "overdue_thread", "subject": t["description"][:80],
                                "reason": f"Was due to pay off by chapter {t['due_chapter']}."})
        return out


def _public(f: dict, names: dict) -> dict:
    return {"id": f.get("id"), "kind": f["kind"], "statement": describe(f, names), "status": f.get("status"),
            "valid_from": {"chapter": valid_from(f)[0], "seq": valid_from(f)[1]},
            "source_chapter": f.get("source_chapter"), "source_quote": f.get("source_quote") or ""}


def _evidence(f: dict, names: dict) -> dict:
    return {**_public(f, names), "source_ref": f.get("source_ref") or "", "origin": f.get("origin")}


def _thread(t: dict) -> dict:
    return {"id": t.get("id"), "kind": t["kind"], "description": t["description"], "planted_chapter": t["planted_chapter"],
            "planted_quote": t.get("planted_quote") or "", "due_chapter": t.get("due_chapter"), "due_episode": t.get("due_episode")}


def _event(e: dict, names: dict) -> dict:
    return {"id": e.get("id"), "chapter": e["chapter"], "story_time": e.get("story_time") or "", "summary": e["summary"],
            "participants": [names.get(p, p) for p in e.get("participants") or []], "source_quote": e.get("source_quote") or ""}


# ====================================================================== reads and writes

TABLES = {"entities": "studio_story_entities", "facts": "studio_story_facts", "events": "studio_story_events",
          "threads": "studio_story_threads", "exceptions": "studio_story_exceptions", "episodes": "studio_story_episodes"}


def create_story(store, owner_id: str, title: str = "", project_id: Optional[str] = None,
                 series_id: Optional[str] = None) -> dict:
    return store.insert("studio_stories", [{"owner_id": owner_id, "title": title[:200], "project_id": project_id,
                                            "series_id": series_id}])[0]


class CanonRepo:
    """One story's canon in the database. AI input can only ever be proposed; only the creator decides."""

    def __init__(self, store, story_id: str, owner_id: str):
        self.store, self.story_id, self.owner_id = store, story_id, owner_id

    def load(self) -> Canon:
        f = {"story_id": self.story_id}
        return Canon(self.story_id, **{k: self.store.select(t, f) for k, t in TABLES.items()})

    def _base(self) -> dict:
        return {"story_id": self.story_id, "owner_id": self.owner_id}

    @staticmethod
    def _decision(by: str) -> dict:
        if by != "creator":
            raise CanonError("Only the creator can approve, lock or retire canon.")
        return {"decided_by": "creator", "decided_at": _now()}

    def _get(self, table: str, rid: str) -> dict:
        row = self.store.one(table, {"id": rid, "story_id": self.story_id})
        if not row:
            raise CanonError("Not found in this story.")
        return row

    # ---- entities

    def add_entity(self, kind: str, name: str, *, aliases: Iterable[str] = (), profile: Optional[dict] = None,
                   source_quote: str = "", first_chapter: Optional[int] = None, by: str = "ai", lock: bool = False) -> dict:
        if kind not in ENTITY_KINDS:
            raise CanonError(f"Unknown kind: {kind}")
        name = (name or "").strip()[:120]
        if not name:
            raise CanonError("An entity needs a name.")
        row = {**self._base(), "kind": kind, "name": name, "aliases": [a.strip()[:120] for a in aliases if a and a.strip()],
               "profile": profile or {}, "source_quote": (source_quote or "")[:1000], "first_chapter": first_chapter,
               "origin": "creator" if by == "creator" else "ai", "status": "proposed"}
        if by == "creator":
            row.update(status="creator_locked" if lock else "approved", **self._decision("creator"))
        return self.store.insert("studio_story_entities", [row])[0]

    # ---- facts

    def _fact_row(self, fact: dict) -> dict:
        kind = fact.get("kind")
        if kind not in FACT_KINDS:
            raise CanonError(f"Unknown fact kind: {kind}")
        if kind in _KEY and not fact.get("subject_id"):
            raise CanonError(f"A {kind} fact needs a subject.")
        if kind in ("relationship", "knows") and not fact.get("object_id"):
            raise CanonError(f"A {kind} fact needs an object.")
        f0, u0 = as_point((fact.get("valid_from_chapter") or 0, fact.get("valid_from_seq") or 0)), None
        if fact.get("valid_until_chapter") is not None:
            u0 = as_point((fact["valid_until_chapter"], fact.get("valid_until_seq") or 0))
            if u0 <= f0:
                raise CanonError("A fact must end after it starts.")
        cols = ("kind", "subject_id", "attribute", "object_id", "value", "statement", "valid_from_chapter", "valid_from_seq",
                "valid_until_chapter", "valid_until_seq", "source_chapter", "source_ref", "source_quote", "event_id",
                "confidence", "supersedes")
        row = {**self._base(), **{c: fact.get(c) for c in cols if fact.get(c) is not None}}
        row.setdefault("valid_from_chapter", f0[0])
        row.setdefault("valid_from_seq", f0[1])
        for c, n in (("attribute", 80), ("value", 500), ("statement", 600), ("source_ref", 200), ("source_quote", 1000)):
            if c in row:
                row[c] = str(row[c])[:n]
        return row

    def propose_fact(self, fact: dict, passage: Optional[str] = None) -> tuple[Optional[dict], list]:
        """An AI fact: stored as proposed (never canon) only with a quote found in its passage. Returns (row or
        None, conflicts with canon). A conflict with a creator-locked fact is kept but marked blocking."""
        from studio_story_check import find_quote
        quote = (fact.get("source_quote") or "").strip()
        if not quote:
            return None, [{"reason": "no evidence: every AI fact needs a quote from the story"}]
        if passage is not None and find_quote(quote, passage) is None:
            return None, [{"reason": "the quote isn't in the passage", "quote": quote}]
        row = self._fact_row({**fact, "source_quote": quote})
        row.update(origin="ai", status="proposed")
        conflicts = self.load().conflicts_for(row)
        return self.store.insert("studio_story_facts", [row])[0], conflicts

    def creator_fact(self, fact: dict, lock: bool = False) -> dict:
        """A fact the creator states: canon at once (locked if asked)."""
        row = self._fact_row({**fact, "source_ref": fact.get("source_ref") or "creator"})
        row.update(origin="creator", status="creator_locked" if lock else "approved", **self._decision("creator"))
        return self.store.insert("studio_story_facts", [row])[0]

    def approve(self, fact_id: str, by: str = "creator") -> dict:
        """The creator accepts a proposed fact. Refused if it contradicts a creator-locked fact. Its entities are
        accepted with it."""
        decision = self._decision(by)
        f = self._get("studio_story_facts", fact_id)
        if f["status"] == "retired":
            raise CanonError("A retired fact can't be approved.")
        canon = self.load()
        blocking = [c for c in canon.conflicts_for(f) if c["blocking"]]
        if blocking:
            raise CanonConflict("This contradicts a fact the creator locked.", blocking)
        for eid in (f.get("subject_id"), f.get("object_id")):
            if eid and canon.by_id.get(eid, {}).get("status") == "proposed":
                self.store.update("studio_story_entities", {"id": eid}, {"status": "approved", **decision})
        return self.store.update("studio_story_facts", {"id": fact_id}, {"status": "approved", **decision})[0]

    def lock(self, fact_id: str, by: str = "creator") -> dict:
        decision = self._decision(by)
        f = self._get("studio_story_facts", fact_id)
        if f["status"] == "retired":
            raise CanonError("A retired fact can't be locked.")
        return self.store.update("studio_story_facts", {"id": fact_id}, {"status": "creator_locked", **decision})[0]

    def retire(self, fact_id: str, reason: str, by: str = "creator", force_locked: bool = False) -> dict:
        decision = self._decision(by)
        f = self._get("studio_story_facts", fact_id)
        if f["status"] == "creator_locked" and not force_locked:
            raise CanonError("This fact is locked. Unlock it first if you really want to retire it.")
        return self.store.update("studio_story_facts", {"id": fact_id},
                                 {"status": "retired", "retired_reason": (reason or "")[:400], **decision})[0]

    # ---- events, threads, exceptions, episodes

    def add_event(self, chapter: int, summary: str, *, seq: int = 0, story_order: Optional[float] = None,
                  story_time: str = "", participants: Iterable[str] = (), location_id: Optional[str] = None,
                  causes: Iterable[str] = (), consequences: str = "", source_quote: str = "", by: str = "ai") -> dict:
        if by != "creator" and not (source_quote or "").strip():
            raise CanonError("An AI event needs a quote from the story.")
        row = {**self._base(), "chapter": int(chapter), "seq": int(seq), "story_order": story_order,
               "story_time": story_time[:120], "summary": summary[:600], "participants": list(participants),
               "location_id": location_id, "causes": list(causes), "consequences": consequences[:600],
               "source_quote": (source_quote or "")[:1000], "origin": "creator" if by == "creator" else "ai", "status": "proposed"}
        if by == "creator":
            row.update(status="approved", **self._decision("creator"))
        return self.store.insert("studio_story_events", [row])[0]

    def approve_event(self, event_id: str, by: str = "creator") -> dict:
        self._get("studio_story_events", event_id)
        return self.store.update("studio_story_events", {"id": event_id}, {"status": "approved", **self._decision(by)})[0]

    def add_thread(self, kind: str, description: str, planted_chapter: int, *, planted_quote: str = "",
                   entities: Iterable[str] = (), payoff_plan: str = "", due_chapter: Optional[int] = None,
                   due_episode: Optional[int] = None, by: str = "ai") -> dict:
        if kind not in THREAD_KINDS:
            raise CanonError(f"Unknown thread kind: {kind}")
        if by != "creator" and not (planted_quote or "").strip():
            raise CanonError("An AI thread needs the quote where it's planted.")
        row = {**self._base(), "kind": kind, "description": description[:600], "planted_chapter": int(planted_chapter),
               "planted_quote": (planted_quote or "")[:1000], "entities": list(entities), "payoff_plan": payoff_plan[:600],
               "due_chapter": due_chapter, "due_episode": due_episode, "origin": "creator" if by == "creator" else "ai",
               "status": "proposed"}
        if by == "creator":
            row.update(status="approved", **self._decision("creator"))
        return self.store.insert("studio_story_threads", [row])[0]

    def approve_thread(self, thread_id: str, by: str = "creator") -> dict:
        self._get("studio_story_threads", thread_id)
        return self.store.update("studio_story_threads", {"id": thread_id}, {"status": "approved", **self._decision(by)})[0]

    def resolve_thread(self, thread_id: str, chapter: int, quote: str = "", dropped: bool = False) -> dict:
        self._get("studio_story_threads", thread_id)
        return self.store.update("studio_story_threads", {"id": thread_id},
                                 {"state": "dropped" if dropped else "resolved", "resolved_chapter": int(chapter),
                                  "resolved_quote": (quote or "")[:1000]})[0]

    def add_exception(self, rule: str, reason: str, *, subject_id: Optional[str] = None, from_chapter: int = 0,
                      until_chapter: Optional[int] = None, by: str = "creator") -> dict:
        self._decision(by)
        if rule not in RULES:
            raise CanonError(f"Unknown rule: {rule}")
        if not (reason or "").strip():
            raise CanonError("Say why this is intentional.")
        return self.store.insert("studio_story_exceptions", [{**self._base(), "rule": rule, "reason": reason[:400],
                                                              "subject_id": subject_id, "from_chapter": int(from_chapter),
                                                              "until_chapter": until_chapter}])[0]

    def set_episode(self, number: int, chapter_from: int, chapter_to: int, project_id: Optional[str] = None) -> dict:
        if chapter_to < chapter_from:
            raise CanonError("An episode must end at or after the chapter it starts in.")
        existing = self.store.one("studio_story_episodes", {"story_id": self.story_id, "number": int(number)})
        fields = {"chapter_from": int(chapter_from), "chapter_to": int(chapter_to), "project_id": project_id}
        if existing:
            return self.store.update("studio_story_episodes", {"id": existing["id"]}, fields)[0]
        return self.store.insert("studio_story_episodes", [{**self._base(), "number": int(number), **fields}])[0]

    # ---- imports from a plan the creator approved (an outline): approved, with the plan text as evidence

    def import_entity(self, kind: str, name: str, quote: str, source_ref: str, profile: Optional[dict] = None,
                      aliases: Iterable[str] = ()) -> dict:
        if kind not in ENTITY_KINDS:
            raise CanonError(f"Unknown kind: {kind}")
        return self.store.insert("studio_story_entities", [{
            **self._base(), "kind": kind, "name": name.strip()[:120], "profile": profile or {},
            "aliases": [a.strip()[:120] for a in aliases if a and a.strip()],
            "source_quote": (quote or name)[:1000], "origin": "import", "status": "approved",
            **self._decision("creator")}])[0]

    def import_fact(self, fact: dict, source_ref: str) -> dict:
        row = self._fact_row({**fact, "source_ref": source_ref, "source_quote": (fact.get("source_quote") or source_ref)})
        row.update(origin="import", status="approved", **self._decision("creator"))
        return self.store.insert("studio_story_facts", [row])[0]

    def import_thread(self, kind: str, description: str, planted_chapter: int, quote: str, source_ref: str,
                      due_chapter: Optional[int] = None, entities: Iterable[str] = ()) -> dict:
        if kind not in THREAD_KINDS:
            raise CanonError(f"Unknown thread kind: {kind}")
        return self.store.insert("studio_story_threads", [{
            **self._base(), "kind": kind, "description": description[:600], "planted_chapter": int(planted_chapter),
            "planted_quote": (quote or source_ref)[:1000], "entities": list(entities),
            "payoff_plan": f"planned in {source_ref}"[:600], "due_chapter": due_chapter, "origin": "import",
            "status": "approved", **self._decision("creator")}])[0]

    # ---- AI extraction intake

    def ingest(self, extraction: dict, passage: str, chapter: int, source_ref: str = "") -> dict:
        """Takes what an extractor (an AI reading one passage) found and keeps it as PROPOSALS only. Names are
        matched to existing entities (new ones are proposed with their quote). A fact whose quote isn't in the
        passage is rejected and reported. Returns {"facts", "entities", "events", "threads", "rejected", "conflicts"}."""
        from studio_story_check import find_quote
        out = {"facts": [], "entities": [], "events": [], "threads": [], "rejected": [], "conflicts": []}
        canon = self.load()

        def ref(name: str, kind: str = "character", quote: str = "") -> Optional[str]:
            nonlocal canon
            if not name:
                return None
            e = canon.entity(name)
            if e:
                return e["id"]
            if not quote or find_quote(quote, passage) is None:
                out["rejected"].append({"what": f"new {kind} {name}", "reason": "no quote from the passage"})
                return None
            row = self.add_entity(kind, name, source_quote=quote, first_chapter=chapter)
            out["entities"].append(row)
            canon = self.load()
            return row["id"]

        for e in extraction.get("entities") or []:
            ref(e.get("name"), e.get("kind") or "character", e.get("quote") or "")
        for f in extraction.get("facts") or []:
            quote = f.get("quote") or ""
            subject = ref(f.get("subject"), f.get("subject_kind") or "character", quote)
            obj = ref(f.get("object"), f.get("object_kind") or ("secret" if f.get("kind") == "knows" else "place"), quote) \
                if f.get("object") else None
            if f.get("subject") and not subject:
                continue
            row, conflicts = self.propose_fact({"kind": f.get("kind"), "subject_id": subject, "object_id": obj,
                                                "attribute": f.get("attribute") or "", "value": f.get("value") or "",
                                                "statement": f.get("statement") or "",
                                                "valid_from_chapter": chapter, "valid_from_seq": int(f.get("seq") or 0),
                                                "source_chapter": chapter, "source_ref": source_ref,
                                                "source_quote": quote, "confidence": f.get("confidence")}, passage)
            if row is None:
                out["rejected"].append({"what": f.get("statement") or f.get("kind"), "reason": conflicts[0]["reason"]})
                continue
            out["facts"].append(row)
            if conflicts:
                out["conflicts"].append({"fact": row["id"], "statement": describe(row, canon.names), "with": conflicts})
        for e in extraction.get("events") or []:
            quote = e.get("quote") or ""
            if find_quote(quote, passage) is None:
                out["rejected"].append({"what": e.get("summary"), "reason": "the quote isn't in the passage"})
                continue
            out["events"].append(self.add_event(chapter, e.get("summary") or "", seq=int(e.get("seq") or 0),
                                                story_order=e.get("story_order"), story_time=e.get("story_time") or "",
                                                participants=[i for i in (canon.eid(p) for p in e.get("participants") or []) if i],
                                                source_quote=quote))
        for t in extraction.get("threads") or []:
            quote = t.get("quote") or ""
            if find_quote(quote, passage) is None:
                out["rejected"].append({"what": t.get("description"), "reason": "the quote isn't in the passage"})
                continue
            out["threads"].append(self.add_thread(t.get("kind") or "question", t.get("description") or "", chapter,
                                                  planted_quote=quote, payoff_plan=t.get("payoff_plan") or "",
                                                  entities=[i for i in (canon.eid(p) for p in t.get("entities") or []) if i]))
        return out
