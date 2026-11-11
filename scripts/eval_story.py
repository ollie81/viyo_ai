"""Evaluates the writers' room against REAL Gemini: one 3-episode short drama
from one idea, through the exact pipeline the Studio uses (studio_story:
plan -> checks -> draft -> script doctor -> story check -> at most one revision),
with each episode given the previous ones the way the Studio gives them.

It prints each episode's opening and ending, the story check (with the lines it
quoted), and checks across episodes that need no AI:
  hooks       every plan starts with a hook beat and ends with an end-hook beat
  continues   episodes 2 and 3 say how they pick up the previous ending
  varied      no two episodes in a row end with the same kind of cliffhanger
  open        every episode leaves at least one question open
  cast        the same character names in every episode
  bar         the story check put no area below the bar (reported, not hidden)

No database, no pictures, no videos. It spends real Gemini credits (about
$0.05-0.08 an episode) and refuses to run unless both are given:
  GEMINI_API_KEY in the environment, and --approve-spend on the command line.

  python scripts/eval_story.py --approve-spend [--max-usd 0.40] [--genre "Family thriller"]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import studio_ai  # noqa: E402
import studio_pricing as pricing  # noqa: E402
import studio_story  # noqa: E402

IDEA = "A woman discovers that her missing brother has been admitted to the hospital under a false name."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--approve-spend", action="store_true", help="required: this calls real Gemini")
    ap.add_argument("--max-usd", type=float, default=0.40)
    ap.add_argument("--genre", default="Family thriller")
    ap.add_argument("--episodes", type=int, default=3)
    args = ap.parse_args()
    if not args.approve_spend or not os.environ.get("GEMINI_API_KEY"):
        print("Not run: needs GEMINI_API_KEY and --approve-spend (this spends real Gemini credits).")
        return 2
    ai = studio_ai.GoogleStudioAI()
    previous, total, results = [], 0.0, []
    for n in range(1, args.episodes + 1):
        brief = {"title": f"False Name – Episode {n}", "logline": IDEA if n == 1 else "The story continues.",
                 "genre": args.genre, "kind": "episode", "aspect_ratio": "9:16", "target_seconds": 60,
                 "language": "English", "visual_style": "cinematic"}
        worst = pricing.estimate_script(60, len(IDEA)).total_usd
        if total + worst > args.max_usd:
            print(f"Stopping before episode {n}: it could pass the ${args.max_usd:.2f} cap.")
            break
        t = time.time()
        try:
            text, bible, cost = ai.develop_script(brief, previous)
        except Exception as e:
            print(f"Episode {n}: ERROR {str(e)[:200]}")
            break
        total += cost.total_usd
        hp = bible.get("hook_plan") or {}
        ev = bible.get("evaluation") or {}
        print(f"\n===== EPISODE {n} (${cost.total_usd:.4f}, {time.time() - t:.0f}s) =====")
        print(f"Opening ({hp.get('opening_technique')}): {hp.get('opening_moment')}")
        print(f"Ending ({hp.get('ending_technique')}): {hp.get('ending_moment')}")
        print(f"Viewer question: {hp.get('viewer_question')}")
        if n > 1:
            print(f"Continues from: {bible.get('continues_from')}")
        for k, v in (ev.get("scores") or {}).items():
            print(f"  {k}: {v['score']}/10 – {v['reason']} [quoted: {v['evidence'][:100]}]")
        if ev.get("revised"):
            print(f"  Revised once to fix: {ev.get('fixed')} (before: {ev.get('before')})")
        if ev.get("revision_rejected"):
            print(f"  {ev['revision_rejected']}")
        if ev.get("error"):
            print(f"  Story check: {ev['error']}")
        print("\n--- screenplay ---\n" + text[:3000])
        results.append(bible)
        previous.append({**bible, "title": brief["title"], "_screenplay_end": studio_story._last_section(text)})

    print("\nRESULTS")
    checks = {
        "hooks": all(b.get("hooks_planned") for b in results),
        "continues": all((b.get("continues_from") or "").strip() for b in results[1:]),
        "varied": all((results[i].get("hook_plan") or {}).get("ending_technique") !=
                      (results[i - 1].get("hook_plan") or {}).get("ending_technique") for i in range(1, len(results))),
        "open": all((b.get("episode_record") or {}).get("open_questions") for b in results),
        "cast": bool(results) and all(lead in {c.get("name") for c in b.get("characters") or []} for b in results
                                      for lead in [next((c.get("name") for c in results[0].get("characters") or []
                                                         if c.get("role") == "lead"), None)] if lead),
        "bar": all(not (b.get("evaluation") or {}).get("below_bar") for b in results),
    }
    for k, v in checks.items():
        print(f"- {k}: {'pass' if v else 'FAIL'}")
    for i, b in enumerate(results, start=1):
        sc = {k: v["score"] for k, v in ((b.get("evaluation") or {}).get("scores") or {}).items()}
        print(f"- episode {i} story check: {sc or 'not run'}")
    print(f"\nEpisodes: {len(results)} · checks passed {sum(checks.values())}/{len(checks)} · total cost ${total:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
