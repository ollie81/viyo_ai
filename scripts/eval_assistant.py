"""Evaluates the Studio assistant against REAL Gemini on a few small,
controlled test projects (no database, no generation jobs).

It builds the exact prompt the Studio would send (studio_chat.project_context
+ build_prompt), asks the real model, and scores each answer:
  facts       expected facts appear (factual accuracy)
  continuity  the continuity problem is found and named (continuity)
  action      the right proposal type is chosen, with the right target (action correctness)
  honest      it doesn't claim something was saved or made (usefulness/honesty)
and prints the cost of every call. Text replies only: no pictures, videos or
voices are made.

This spends real Gemini credits (a few cents for the whole run). It refuses to
run unless both are given:
  GEMINI_API_KEY in the environment, and --approve-spend on the command line.

  python scripts/eval_assistant.py --approve-spend [--max-usd 0.50]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import studio_ai  # noqa: E402
import studio_chat  # noqa: E402
import studio_continuity  # noqa: E402
import studio_memory  # noqa: E402
import studio_pricing as pricing  # noqa: E402


def scene(n, heading, visual, dialogue=(), **cont):
    return {"id": f"s{n}", "position": n - 1, "heading": heading, "visual": visual,
            "dialogue": [{"character": c, "text": t} for c, t in dialogue], "continuity": cont,
            "image_path": f"p{n}.png", "made_with": {}}


MAYA = {"id": "c1", "name": "Maya", "description": "nurse, 28, short black hair", "voice": "Kore",
        "voice_description": "warm, low"}
DANIEL = {"id": "c2", "name": "Daniel", "description": "her brother, 35, beard, grey coat", "voice": "Charon"}

CASES = [
    {
        "name": "outfit continuity",
        "project": {"title": "The Return", "logline": "A nurse finds her missing brother in her ward.", "kind": "episode",
                    "script": "INT. WARD - NIGHT\nMaya checks a patient.\nINT. WARD - NIGHT\nShe sees his face: Daniel.\n"},
        "cast": [MAYA, DANIEL],
        "outfits": [{"id": "o1", "character_id": "c1", "name": "Blue scrubs", "status": "saved"},
                    {"id": "o2", "character_id": "c1", "name": "Red dress", "status": "saved"}],
        "scenes": [scene(1, "INT. WARD - NIGHT", "Maya checks a patient's chart", transition="new_place"),
                   scene(2, "INT. WARD - NIGHT", "Maya recognises Daniel in the bed", transition="continuous")],
        "wearing": {"s1": {"c1": "o1"}, "s2": {"c1": "o2"}},
        "message": "Why does Maya look different in scene 2?",
        "expect_facts": ["scene 2", "red dress", "scrubs"],
        "expect_action": ("set_scene_outfit", {"scene": 2}),
    },
    {
        "name": "what is missing",
        "project": {"title": "The Return", "logline": "A nurse finds her brother.", "kind": "episode",
                    "script": "INT. WARD - NIGHT\nMaya sees Daniel.\n"},
        "cast": [MAYA, DANIEL], "outfits": [],
        "scenes": [scene(1, "INT. WARD - NIGHT", "Maya sees Daniel", [("Maya", "Daniel?")])],
        "wearing": {},
        "message": "What is still missing before this episode is finished?",
        "expect_facts": ["voice", "film"],
        "expect_action": None,
    },
    {
        "name": "reference resolution + memory",
        "project": {"title": "The Return", "logline": "A nurse finds her brother.", "kind": "episode", "script": ""},
        "cast": [MAYA, DANIEL], "outfits": [], "scenes": [], "wearing": {},
        "memory": [("characters", "Maya is Daniel's younger sister"), ("preferences", "No happy ending")],
        "focus": {"scenes": [], "characters": ["Maya"]},
        "message": "Make her more mysterious, but remember what I said about the ending.",
        "expect_facts": ["maya", "ending"],
        "expect_action": None,
    },
]

BANNED = ("has been saved", "i've saved", "i have saved", "i changed", "i've updated", "has been generated", "is now fixed")


def build(case):
    p = dict(case["project"], aspect_ratio="9:16", target_seconds=60, language="English")
    outfits = case["outfits"]
    names = {o["id"]: o["name"] for o in outfits}
    issues = studio_continuity.review(case["scenes"], case["wearing"], names, {c["id"]: c["name"] for c in case["cast"]},
                                      {})
    wearing = {sid: {next(c["name"] for c in case["cast"] if c["id"] == cid): names[oid] for cid, oid in row.items()}
               for sid, row in case["wearing"].items()}
    focus = case.get("focus") or studio_chat.resolve_focus(case["message"], {}, len(case["scenes"]),
                                                          [c["name"] for c in case["cast"]])
    context = studio_chat.project_context(p, case["cast"], outfits, [], case["scenes"], None, [], wearing=wearing,
                                          issues=issues, focus=set(focus.get("scenes") or []), question=case["message"],
                                          focus_names=set(focus.get("characters") or []))
    mem = {}
    for cat, text in case.get("memory") or []:
        mem, _ = studio_memory.add(mem, cat, text)
    mode = studio_chat.reasoning_mode(case["message"])
    return studio_chat.build_prompt(context, "", studio_memory.render(mem), [], case["message"],
                                    studio_chat.focus_text(focus), mode), mode


def score(case, data):
    text = (data.get("reply") or "").lower()
    out = {"facts": all(f in text for f in case["expect_facts"]),
           "honest": not any(b in text for b in BANNED)}
    want = case["expect_action"]
    props = data.get("proposals") or []
    if want:
        kind, fields = want
        out["action"] = any(p.get("type") == kind and all((p.get("data") or {}).get(k) == v for k, v in fields.items())
                            for p in props)
    if "continuity" in case["name"]:
        out["continuity"] = "scene 2" in text and ("outfit" in text or "dress" in text)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--approve-spend", action="store_true", help="required: this calls real Gemini")
    ap.add_argument("--max-usd", type=float, default=0.50)
    args = ap.parse_args()
    if not args.approve_spend or not os.environ.get("GEMINI_API_KEY"):
        print("Not run: needs GEMINI_API_KEY and --approve-spend (this spends real Gemini credits).")
        return 2
    ai = studio_ai.GoogleStudioAI()
    total, results = 0.0, []
    for case in CASES:
        prompt, mode = build(case)
        guess = pricing.text_cost(len(prompt) // 4, studio_ai.chat_thinking(mode)[1] + 1200).total_usd
        if total + guess > args.max_usd:
            print(f"Stopping before '{case['name']}': it could pass the ${args.max_usd:.2f} cap.")
            break
        t = time.time()
        try:
            data, cost = ai.chat(prompt, mode=mode)
        except Exception as e:
            results.append((case["name"], {"error": str(e)[:200]}, 0.0, 0.0))
            continue
        total += cost.total_usd
        results.append((case["name"], score(case, data), cost.total_usd, time.time() - t))
        print(f"\n--- {case['name']} ({mode}) ---\n{data.get('reply')}\nproposals: "
              f"{[p.get('type') for p in data.get('proposals') or []]}")
    print("\nRESULTS")
    passed = checks = 0
    for name, sc, usd, secs in results:
        if "error" in sc:
            print(f"- {name}: ERROR {sc['error']}")
            continue
        checks += len(sc)
        passed += sum(sc.values())
        print(f"- {name}: {sc}  ${usd:.4f}  {secs:.1f}s")
    print(f"\nScore: {passed}/{checks} checks · total cost ${total:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
