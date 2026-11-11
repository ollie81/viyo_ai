"""Evaluates the writers' room against REAL Gemini: one 3-episode short drama
from one idea, through the exact pipeline the Studio uses (studio_story:
plan -> checks -> draft -> script doctor -> story check -> at most one revision),
with each episode given the previous ones the way the Studio gives them.

It prints each episode's opening and ending, the story check (with the lines it
quoted), and checks across episodes that need no AI:
  - the plan has hooks; the story check is complete (every category verified)
  - every quote the check used is really in the generated script (re-checked here)
  - the opening is quoted from scene 1 and the ending from the last scene
  - episodes 2 and 3 continue (plan, verified continuity, and the first scene
    shares people or things with the previous last scene)
  - cliffhangers vary; a question is left open; the lead is kept
  - each episode passed the bar (reported, not hidden)

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
import studio_story_check as sc  # noqa: E402

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
    previous, total, results, results_text = [], 0.0, [], []
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
        print(f"Story check: {ev.get('status')} – {ev.get('summary')}")
        for k, c in (ev.get("categories") or {}).items():
            if c.get("status") == "verified":
                print(f"  {k}: {c['score']}/10 (scene {c.get('scene')}) – {c['reason']} [quote: {c['quote'][:90]}]")
            else:
                print(f"  {k}: NOT VERIFIED – {c.get('why')}")
        for c in ev.get("checks") or []:
            print(f"  script check: {c['area']}: {c['problem']}")
        rev = ev.get("revision") or {}
        if rev.get("attempted"):
            print(f"  revision of scene(s) {rev.get('scenes')}: {'kept' if rev.get('kept') else 'not kept'} – {rev.get('why')}")
        if ev.get("error"):
            print(f"  {ev['error']}")
        results_text.append(text)
        print("\n--- screenplay ---\n" + text[:3000])
        results.append(bible)
        previous.append({**bible, "title": brief["title"], "_screenplay_end": studio_story._last_section(text)})

    print("\nRESULTS")
    def quotes_real(b, text):
        """Re-verify every quote against the generated script, independently of the stored status."""
        cats = (b.get("evaluation") or {}).get("categories") or {}
        return all(sc.find_quote(c["quote"], text) is not None for c in cats.values() if c.get("status") == "verified")

    checks = {
        "hooks planned": all(b.get("hooks_planned") for b in results),
        "check complete (every category verified)": all((b.get("evaluation") or {}).get("complete") for b in results),
        "quotes really in the script": all(quotes_real(b, t) for b, t in zip(results, results_text)),
        "opening quoted from scene 1": all(((b.get("evaluation") or {}).get("categories") or {}).get("opening_hook", {}).get("scene") == 1
                                           for b in results),
        "ending quoted from the last scene": all(
            ((b.get("evaluation") or {}).get("categories") or {}).get("ending_hook", {}).get("scene") == len(sc.sections(t))
            for b, t in zip(results, results_text)),
        "continues (plan + verified continuity)": all((b.get("continues_from") or "").strip() and
                                                      ((b.get("evaluation") or {}).get("categories") or {}).get("continuity", {}).get("status") == "verified"
                                                      for b in results[1:]),
        "first scene picks up the last one": all(not any(c["area"] == "continuity" for c in (b.get("evaluation") or {}).get("checks") or [])
                                                 for b in results[1:]),
        "varied cliffhangers": all((results[i].get("hook_plan") or {}).get("ending_technique") !=
                                   (results[i - 1].get("hook_plan") or {}).get("ending_technique") for i in range(1, len(results))),
        "open question left": all((b.get("episode_record") or {}).get("open_questions") for b in results),
        "lead kept": bool(results) and all(lead in {c.get("name") for c in b.get("characters") or []} for b in results
                                           for lead in [next((c.get("name") for c in results[0].get("characters") or []
                                                              if c.get("role") == "lead"), None)] if lead),
        "passed the bar": all((b.get("evaluation") or {}).get("status") == "passed" for b in results),
    }
    for k, v in checks.items():
        print(f"- {k}: {'pass' if v else 'FAIL'}")
    for i, b in enumerate(results, start=1):
        ev = b.get("evaluation") or {}
        print(f"- episode {i} story check: {ev.get('status')} {sc.scores(ev) or ''}"
              f"{' unverified: ' + ', '.join(ev.get('unverified') or []) if ev.get('unverified') else ''}")
    print(f"\nEpisodes: {len(results)} · checks passed {sum(checks.values())}/{len(checks)} · total cost ${total:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
