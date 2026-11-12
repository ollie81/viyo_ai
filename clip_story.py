"""Multi-moment storytelling across the WHOLE video.

The whole-video map (video_map, one call that already reads the full
transcript) lists "threads": two moments in different parts of the video
that tell one story together - a setup and its payoff, a question and its
answer, a claim and its example, a problem and its solution, a callback.
Selection reads the video in ~10 minute windows, so it can't see those
pairs; this pass links them to the clips that were chosen.

A link becomes a joined clip only when code says it is safe:
  * both parts are whole sentences ending a thought, 4-25 s each, played in
    the order they were said, at least 8 s apart in the source;
  * the part that comes second doesn't open by leaning on something the
    viewer won't hear ("that", "he", "so"...);
  * neither part is an intro / outro / ad, and the partner isn't inside
    another chosen clip;
  * the whole clip stays within the 60 s limit;
and then the editorial review must CONFIRM that, heard back to back, it keeps
exactly what the speaker meant (`join_faithful`). Unconfirmed joins are undone
(the clip goes back to its single moment). No words are added or reordered:
the viewer sees "LATER IN THE VIDEO" where the parts meet, and the creator
sees each part's source time.
"""
from __future__ import annotations

from typing import Optional

import clip_edit
import clip_selection as cs

MIN_PART, MAX_PART = 4.0, 25.0


def _overlap_share(a: tuple, b: tuple) -> float:
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    return max(0, hi - lo + 1) / max(1, a[1] - a[0] + 1)


def _boilerplate(video_map: dict, a: int, b: int) -> bool:
    return any(not (b < x["start_sentence"] or a > x["end_sentence"]) for x in video_map.get("avoid") or [])


def link(pool: list, sentences: list, words: list, video_map: Optional[dict], video_seconds: float,
         max_seconds: float = cs.MAX_CLIP_SECONDS) -> dict:
    """Joins thread partners onto chosen clips in `pool` (in place). Returns facts."""
    threads = (video_map or {}).get("threads") or []
    facts = {"threads": len(threads), "linked": 0, "refused": {}}
    if not threads:
        return facts
    taken = [(c.start_sentence, c.end_sentence) for c in pool]

    def refuse(why):
        facts["refused"][why] = facts["refused"].get(why, 0) + 1

    for c in pool:
        if (c.structure or {}).get("joined"):
            continue
        main = tuple((c.structure or {}).get("main") or (c.start_sentence, c.end_sentence))
        for th in threads:
            (a0, a1), (b0, b1) = (tuple(p) for p in th["parts"])
            if _overlap_share((a0, a1), main) >= 0.5:
                partner, partner_first = (b0, b1), False
            elif _overlap_share((b0, b1), main) >= 0.5:
                partner, partner_first = (a0, a1), True
            else:
                continue
            p0, p1 = partner
            if not (p1 < main[0] or p0 > main[1]):
                refuse("partner_inside_clip")
                continue
            if any(not (p1 < x0 or p0 > x1) for x0, x1 in taken if (x0, x1) != (c.start_sentence, c.end_sentence)):
                refuse("partner_in_another_clip")
                continue
            dur = sentences[p1].end - sentences[p0].start
            if not (MIN_PART <= dur <= MAX_PART):
                refuse("partner_length")
                continue
            if not sentences[p1].terminal:
                refuse("partner_ends_mid_sentence")
                continue
            second_first = main[0] if partner_first else p0
            if clip_edit.leans_back(sentences[second_first].text) and not partner_first:
                refuse("second_part_leans_back")
                continue
            if partner_first and clip_edit.leans_back(sentences[p0].text):
                refuse("partner_leans_back")
                continue
            if video_map and _boilerplate(video_map, p0, p1):
                refuse("partner_is_boilerplate")
                continue
            first_end = sentences[p1].end if partner_first else sentences[main[1]].end
            second_start = sentences[main[0]].start if partner_first else sentences[p0].start
            if second_start - first_end < clip_edit.MIN_JOIN_GAP_SECONDS:
                refuse("too_close")
                continue
            ranges = [tuple(r) for r in (c.structure or {}).get("ranges") or [main]]
            total = sum(sentences[y].end - sentences[x].start for x, y in ranges) + dur
            if total > max_seconds:
                refuse("too_long")
                continue
            c.structure = dict(c.structure or {})
            c.structure.setdefault("main", list(main))
            c.structure["unjoined"] = cs.snapshot(c)
            cs.set_ranges(c, ranges + [partner], sentences, words, video_seconds)
            order = [tuple(r) for r in sorted(ranges + [partner])]
            k = order.index(tuple(partner))
            c.structure.update(joined=list(partner), relation=th["relation"], why=th.get("why", ""),
                               link="thread", needs_confirmation=True,
                               # where the viewer sees "LATER IN THE VIDEO": the start of the part after the jump
                               joined_at=c.parts[k + 1][0] if partner_first else c.parts[k][0])
            taken.append(partner)
            facts["linked"] += 1
            break
    return facts


def settle(pool: list) -> dict:
    """After the review: joins it didn't confirm as faithful are undone. Returns counts."""
    out = {"confirmed": 0, "undone": 0}
    for c in pool:
        st = c.structure or {}
        if not st.get("joined"):
            continue
        rv = c.review or {}
        faithful = rv.get("join_faithful")
        if faithful is False or (st.get("needs_confirmation") and faithful is not True):
            cs.unjoin(c)
            out["undone"] += 1
        else:
            c.structure.pop("needs_confirmation", None)
            out["confirmed"] += 1
    return out
