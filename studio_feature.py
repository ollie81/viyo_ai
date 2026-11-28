"""VIYO feature films: one 30-60 minute movie, made as parts of at most 10 minutes, joined at the end.

How a feature is made
  An approved novel is planned as PARTS of one continuous movie (studio_adapt with feature=True): no recap, no
  "previously", no cliffhanger at every break; each part begins exactly where the last one ended, and only the last
  part ends the movie. Approving the plan creates a master production (the finished movie) and one production per
  part, all in the same series: they share the cast and places (linked to the same identities), the look
  (studio_look), and each part's writer gets its own chapters and the story memory at that point (studio_canon).
  Every part is made with the normal pipeline, quote, budget, storyboard review and refunds.

Joining
  A part's render keeps its exact original file (as well as the streaming copy), so the master's "render" joins
  the parts end to end without re-encoding: same size, frame rate and sound format, no quality loss, minutes not
  hours. Part breaks have no dip to black (only the movie's first moment fades in and its last fades out). The
  joined movie goes through the same final check as any film; a failed join is refunded and can run again. The
  join waits until every part is rendered, and says which ones aren't.

Honest limits
  Joining and memory are tested with a fake AI and real ffmpeg on short parts. A real 60-minute movie on the
  production server (disk, memory and time for the download, join and upload) has NOT been tested yet; nothing
  here should be advertised as feature-length until that has passed.
"""
from __future__ import annotations

import math
import os
from typing import Optional

MAX_PART_SECONDS = 600         # the existing production pipeline's limit
MIN_MINUTES, MAX_MINUTES = 15, 90


def parts_for(minutes: int) -> tuple[int, int]:
    """(number of parts, seconds per part) for a feature of this length."""
    total = max(MIN_MINUTES, min(MAX_MINUTES, int(minutes))) * 60
    n = math.ceil(total / MAX_PART_SECONDS)
    return n, min(MAX_PART_SECONDS, math.ceil(total / n))


def info(project: Optional[dict]) -> dict:
    """The feature record on a production: {id, part, of, master, title} or {}."""
    return ((project or {}).get("story_bible") or {}).get("feature") or {}


def is_master(project: Optional[dict]) -> bool:
    return bool(info(project).get("master"))


def is_part(project: Optional[dict]) -> bool:
    return bool(info(project).get("part")) and not is_master(project)


PLAN_NOTE = (
    "THESE ARE THE PARTS OF ONE CONTINUOUS FEATURE FILM, NOT EPISODES. Viewers watch them as one movie, with no breaks. "
    "So: no recap, no 'previously', no title card, and no cliffhanger at the end of a part. Put each part break at a "
    "natural scene change. In each part, 'hook' is how the part BEGINS: it continues directly from the moment the "
    "previous part ended (part 1 opens the movie). 'cliffhanger' is the moment the part HANDS OVER to the next one: a "
    "scene ending, not a shock. The last part's 'cliffhanger' is the movie's ending. The whole movie needs one "
    "dramatic shape across all parts: set-up in the first quarter, a midpoint turn, the low point, the climax and the "
    "resolution near the end.")


def writer_block(episode: dict) -> str:
    """What a part's writer is told, instead of the episode rules."""
    k, n = episode.get("number"), episode.get("of")
    return (f"THIS IS PART {k} OF {n} OF ONE CONTINUOUS FEATURE FILM (chapters {episode.get('chapter_from')}-"
            f"{episode.get('chapter_to')}): \"{episode.get('title') or ''}\".\n"
            f"What happens: {episode.get('summary') or ''}\n"
            "Beats, in order: " + " | ".join(episode.get("beats") or []) + "\n"
            + ("It opens the movie: " if k == 1 else "It begins exactly where part {} ended: ".format(k - 1 if k else "?"))
            + f"{episode.get('hook') or ''}\n"
            + ("It ends the movie: " if episode.get("last") else "It hands over to the next part with: ")
            + f"{episode.get('cliffhanger') or ''}\n"
            "No recap, no 'previously', no title card, no cliffhanger at the end of a part (only the last part ends the "
            "movie). Stay faithful to the novel: the same people, events, order of revelations and who knows what.")


def fades(project: Optional[dict], first: bool, last: bool) -> tuple[bool, bool]:
    """(fade in at this scene, fade out at this scene): a feature fades in only at the movie's start and out only at
    its end, never at a part break."""
    f = info(project)
    if not f or f.get("master"):
        return first, last
    return first and int(f.get("part") or 0) == 1, last and int(f.get("part") or 0) == int(f.get("of") or 0)


def join(parts: list[str], out: str, tmp: str) -> None:
    """The parts end to end, without re-encoding (they share size, frame rate and sound format)."""
    import studio_render as render
    listing = os.path.join(tmp, "parts.txt")
    with open(listing, "w") as f:
        f.writelines(f"file '{p}'\n" for p in parts)
    render._run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", listing, "-c", "copy",
                 "-movflags", "+faststart", out], "joining the parts of the feature")


def missing_parts(parts: list[dict], renders: dict) -> list[int]:
    """Part numbers without a render that can be joined (renders: project id -> latest render with a source)."""
    return [int(info(p).get("part") or 0) for p in sorted(parts, key=lambda p: int(info(p).get("part") or 0))
            if not (renders.get(p["id"]) or {}).get("source")]
