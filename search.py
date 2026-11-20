"""VIYO search: dramas, movies, short films, series, episodes, videos, creators and genres in one box.

  * Partial words ("kiga" finds "Kigali Nights"), small spelling mistakes ("romence" finds Romance), and several
    words in any order.
  * Describe-a-story: words like "scary", "falls in love", "revenge", "detective" are mapped to VIYO's genres, so
    "a girl who falls in love with a billionaire" finds Romance and Billionaire dramas. This is a keyword
    dictionary, not an AI model: it costs nothing per search and is honest about what it understands.
  * Ranking: how well the words match (title above description), then quality and freshness a little. Only public
    content; blocked creators are left out.
  * Searches by signed-in viewers are recorded (search text only) to improve their For You page; turning
    personalization off stops that being used.
The catalogue is cached for a few minutes; with the pg_trgm indexes in intelligence.sql, this can move into the
database when the catalogue becomes large.
"""
from __future__ import annotations

import datetime as _dt
import difflib
import re
import time
from typing import Optional

from fastapi import APIRouter, Depends

import activity
import recommend as rec

router = APIRouter(prefix="/api/v1", tags=["search"])

CACHE_SECONDS = 180
_cache: dict = {}

GENRE_WORDS = {
    "Romance": ["love", "romance", "romantic", "kiss", "heart", "crush", "dating", "wedding", "marriage", "falls"],
    "Horror": ["scary", "horror", "ghost", "haunted", "demon", "zombie", "fear", "creepy"],
    "Comedy": ["funny", "comedy", "laugh", "hilarious", "joke", "humor", "humour"],
    "Revenge": ["revenge", "betrayal", "betrayed", "payback", "avenge"],
    "Billionaire": ["billionaire", "ceo", "rich", "wealthy", "millionaire", "tycoon"],
    "Thriller": ["thriller", "suspense", "chase", "danger", "tense"],
    "Mystery": ["mystery", "secret", "clue", "puzzle", "missing", "disappear"],
    "Crime Drama": ["crime", "police", "detective", "murder", "gang", "heist", "criminal"],
    "Fantasy": ["magic", "fantasy", "dragon", "witch", "wizard", "kingdom", "spell"],
    "Werewolf": ["werewolf", "wolf", "alpha", "pack", "moon"],
    "Family": ["family", "mother", "father", "sister", "brother", "parents", "children", "kids"],
    "Hidden Identity": ["hidden", "identity", "disguise", "undercover", "pretend", "secretly"],
    "Science Fiction": ["future", "space", "robot", "alien", "scifi", "sci", "technology", "ai"],
    "Historical Drama": ["history", "historical", "king", "queen", "ancient", "war", "empire"],
    "Action": ["action", "fight", "explosion", "battle", "martial"],
    "Documentary": ["documentary", "real", "true", "story", "nature"],
}
TYPE_WORDS = {"movie": ["movie", "film"], "short_film": ["short", "film"], "short_drama": ["drama", "series", "episodes"]}
STOP = {"a", "an", "the", "and", "or", "of", "to", "in", "on", "with", "who", "that", "for", "about", "is", "it",
        "her", "his", "their", "she", "he", "they", "i", "want", "something", "show", "me", "find", "like"}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", (s or "").lower())).strip()


def _tokens(s: str) -> list[str]:
    return [t for t in _norm(s).split() if t not in STOP]


def word_score(q: str, words: list[str]) -> float:
    """1 exact, 0.8 prefix, 0.6 close spelling (no match: 0)."""
    best = 0.0
    for w in words:
        if w == q:
            return 1.0
        if len(q) >= 2 and w.startswith(q):
            best = max(best, 0.8)
        elif len(q) >= 4 and len(w) >= 4 and difflib.SequenceMatcher(None, q, w).ratio() >= 0.8:
            best = max(best, 0.6)
    return best


def story_genres(query: str) -> list[str]:
    toks = set(_tokens(query))
    hits = [(g, sum(1 for w in words if w in toks)) for g, words in GENRE_WORDS.items()]
    return [g for g, n in sorted(hits, key=lambda x: -x[1]) if n]


def score_doc(q_tokens: list[str], fields: dict, genres: list[str]) -> float:
    """fields: {name: (text, weight)}. Every query word contributes its best match across fields."""
    if not q_tokens:
        return 0.0
    words = {name: _norm(text).split() for name, (text, _w) in fields.items()}
    total = 0.0
    for q in q_tokens:
        total += max((word_score(q, words[n]) * w for n, (_t, w) in fields.items()), default=0.0)
    score = total / len(q_tokens)
    genre = _norm(fields.get("genre", ("", 0))[0])
    if genres and genre and any(_norm(g) == genre for g in genres):
        score += 0.6 if genre == _norm(genres[0]) else 0.3
    return score


def _catalogue(db) -> dict:
    hit = _cache.get("cat")
    if hit and time.time() - hit[0] < CACHE_SECONDS:
        return hit[1]
    now = _dt.datetime.now(_dt.timezone.utc)
    items, key_of = rec.load_items(db, now)
    stats, _ = rec.load_stats(db, key_of, now, rec.flagged_users(db))
    creators = {}
    try:
        ids = list({it.creator_id for it in items})
        for p in (db.table("profiles").select("id,username,display_name,avatar_url").execute().data or []):
            creators[p["id"]] = p
    except Exception:
        pass
    cat = {"items": items, "stats": stats, "creators": creators, "now": now}
    _cache["cat"] = (time.time(), cat)
    return cat


def search(db, query: str, blocked: set, limit: int = 20) -> dict:
    cat = _catalogue(db)
    q_tokens = _tokens(query)[:8]
    genres = story_genres(query)
    results = []
    for it in cat["items"]:
        if it.creator_id in blocked or it.kind == "ad":
            continue
        cr = cat["creators"].get(it.creator_id, {})
        fields = {"title": (it.title, 3.0), "genre": (it.genre, 2.0), "creator": (f"{cr.get('username', '')} {cr.get('display_name', '')}", 1.5),
                  "text": (it.text, 1.0), "type": (" ".join(TYPE_WORDS.get(it.content_type, [])), 0.8)}
        s = score_doc(q_tokens, fields, genres)
        if s < 0.45:
            continue
        s += 0.4 * rec.quality(cat["stats"].get(it.key, rec.Stats())) + 0.2 * rec.freshness(it, cat["now"])
        kind = {"series": {"movie": "movie", "short_film": "short_film", "ai_film": "movie"}.get(it.content_type, "drama"),
                "video": "video", "photo": "photo"}.get(it.kind, it.kind)
        results.append({"kind": kind, "post_id": it.post_id, "series_id": it.key[2:] if it.kind == "series" else None,
                        "title": it.title, "genre": it.genre or None, "creator_id": it.creator_id, "score": round(s, 3)})
    results.sort(key=lambda r: r["score"], reverse=True)
    creators = []
    for cid, p in cat["creators"].items():
        if cid in blocked:
            continue
        s = score_doc(q_tokens, {"u": (p.get("username") or "", 3.0), "d": (p.get("display_name") or "", 2.5)}, [])
        if s >= 0.55:
            creators.append({"id": cid, "username": p.get("username"), "display_name": p.get("display_name"),
                             "avatar_url": p.get("avatar_url"), "score": round(s, 3)})
    creators.sort(key=lambda c: c["score"], reverse=True)
    return {"query": query, "results": results[:limit], "creators": creators[:10],
            "understood_genres": genres[:3]}


@router.get("/search")
async def get_search(q: str = "", user_id: str = Depends(activity._viewer)):
    q = (q or "").strip()[:100]
    if len(q) < 2:
        return {"query": q, "results": [], "creators": [], "understood_genres": []}
    db = activity._db()
    try:
        blocked = {r["blocked_id"] for r in (db.table("blocks").select("*").eq("blocker_id", user_id).execute().data or [])}
    except Exception:
        blocked = set()
    out = search(db, q, blocked)
    activity.record(user_id, "search", query=q)
    return out
