"""Evaluates the story-world builder against REAL Gemini (with Google Search
grounding for real places): historical Rwanda around 1500, late-Sengoku Japan,
modern Lagos and a fictional world, plus (with --with-story) one Rwandan
episode written inside its world and judged by the strict story check.

Checks that need no AI, on what the real model returned:
  researched     real places were researched with web search and have sources
  labelled       documented claims carry a source (unsourced ones are counted)
  honest         a historical world admits what is unknown or uncertain
  no_confusion   the world doesn't borrow another culture's markers
  no_anachronism the world's own look doesn't contain things that didn't exist yet
  depth          no thin sections (setting, society, culture, language, material, environment)
  story          (--with-story) the episode's story check is complete, authenticity is verified by a real quote,
                 and the screenplay has no anachronisms or confusions

No database, no pictures. It spends real Gemini credits (about $0.05-0.10 a
world with search; the story adds about $0.09) and refuses to run unless both
are given: GEMINI_API_KEY in the environment, and --approve-spend.

  python scripts/eval_world.py --approve-spend [--max-usd 0.60] [--with-story]
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import studio_ai  # noqa: E402
import studio_pricing as pricing  # noqa: E402
import studio_story_check as sc  # noqa: E402
import studio_world as wb  # noqa: E402

CASES = [
    ("Rwanda ~1500", "grounded", "Rwanda around 1500, a royal court in the central highlands",
     "A fictional prince must choose between his father's throne and the woman his clan has sworn to destroy."),
    ("Feudal Japan", "grounded", "Owari province, Japan, 1590s",
     "A merchant's daughter hides a wounded samurai from the lord who wants him dead."),
    ("Modern Lagos", "modern", "Yaba, Lagos, Nigeria, today",
     "A young coder discovers her startup's investor is laundering money for her father."),
    ("Fictional", "fictional", "a floating trading city inspired by the medieval Swahili coast",
     "A navigator's apprentice finds a map that should not exist."),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--approve-spend", action="store_true", help="required: this calls real Gemini")
    ap.add_argument("--max-usd", type=float, default=0.60)
    ap.add_argument("--with-story", action="store_true")
    args = ap.parse_args()
    if not args.approve_spend or not os.environ.get("GEMINI_API_KEY"):
        print("Not run: needs GEMINI_API_KEY and --approve-spend (this spends real Gemini credits).")
        return 2
    ai = studio_ai.GoogleStudioAI()
    total, rows, worlds = 0.0, [], {}
    for name, mode, setting, idea in CASES:
        worst = pricing.estimate_world(mode != "fictional").total_usd
        if total + worst > args.max_usd:
            print(f"Stopping before {name}: it could pass the ${args.max_usd:.2f} cap.")
            break
        try:
            world, cost = ai.build_world({"setting": setting, "language": "English", "title": name, "logline": idea},
                                         idea, mode)
        except Exception as e:
            print(f"{name}: ERROR {str(e)[:200]}")
            continue
        total += cost.total_usd
        worlds[name] = world
        ev = wb.evaluate(world)
        look = " ".join(str(world.get(k)) for k in ("material", "environment", "visual"))
        res = {
            "researched": mode == "fictional" or (world.get("researched") and len(world.get("sources") or []) > 0),
            "labelled": ev["facts"]["unsourced"] == 0,
            "honest": mode in ("modern", "fictional") or bool(world.get("unknown")) or ev["facts"]["uncertain"] > 0,
            "no_confusion": not wb.confusions(look, world),
            "no_anachronism": not wb.anachronisms(look, world),
            "depth": not any(p["kind"] == "thin" for p in ev["problems"]),
        }
        rows.append((name, res, cost.total_usd))
        st = world.get("setting") or {}
        print(f"\n===== {name} ({mode}, ${cost.total_usd:.4f}) =====")
        print(f"{st.get('place')}, {st.get('country')}, {st.get('era')} ({st.get('year')}): {st.get('summary')}")
        print(f"Sources: {len(world.get('sources') or [])} · facts {ev['facts']}")
        for f in (world.get("facts") or [])[:10]:
            print(f"  [{f['status']}] {f['claim']}" + (f"  <- {f['source']['url']}" if f.get("source") else ""))
        print("Unknown:", "; ".join(world.get("unknown") or []) or "(none)")
        print("Avoid:", ", ".join(world.get("avoid") or []) or "(none)")
        print("--- what the picture models get ---\n" + wb.block(world, "image")[:1200])
    if args.with_story and "Rwanda ~1500" in worlds:
        brief = {"title": "The Drum Prince", "logline": CASES[0][3], "genre": "Historical drama", "kind": "episode",
                 "aspect_ratio": "9:16", "target_seconds": 60, "language": "English", "world": worlds["Rwanda ~1500"]}
        if total + pricing.estimate_script(60, 200).total_usd <= args.max_usd:
            text, bible, cost = ai.develop_script(brief, [])
            total += cost.total_usd
            ev = bible.get("evaluation") or {}
            auth = (ev.get("categories") or {}).get("authenticity") or {}
            res = {"story": ev.get("complete") and auth.get("status") == "verified" and sc.find_quote(auth.get("quote", ""), text)
                   is not None and not wb.check_text(text, worlds["Rwanda ~1500"])}
            rows.append(("Rwanda episode", res, cost.total_usd))
            print(f"\n===== Rwanda episode (${cost.total_usd:.4f}) =====\nStory check: {ev.get('status')} – {ev.get('summary')}")
            for k, c in (ev.get("categories") or {}).items():
                print(f"  {k}: " + (f"{c['score']}/10 [quote: {c['quote'][:80]}]" if c.get("status") == "verified" else f"NOT VERIFIED – {c.get('why')}"))
            print("World problems in the script:", wb.check_text(text, worlds["Rwanda ~1500"]) or "none")
            print("\n--- screenplay ---\n" + text[:2500])
        else:
            print("Skipping the story: it could pass the cap.")
    print("\nRESULTS")
    passed = checks = 0
    for name, res, usd in rows:
        checks += len(res)
        passed += sum(bool(v) for v in res.values())
        print(f"- {name}: " + ", ".join(f"{k} {'pass' if v else 'FAIL'}" for k, v in res.items()) + f"  ${usd:.4f}")
    print(f"\nChecks passed {passed}/{checks} · total cost ${total:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
