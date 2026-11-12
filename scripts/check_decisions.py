"""One real OpenAI Decisions API request, to confirm the integration works with this account.

Costs well under $0.001 (about 300 input tokens; output isn't charged). It is NOT run by the tests.

    OPENAI_API_KEY=sk-... python scripts/check_decisions.py --approve-spend

Prints the validated answers, the usage and the cost, or the exact error.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import decisions as dc  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--approve-spend", action="store_true", help="required: this makes one paid API request")
args = p.parse_args()
if not args.approve_spend:
    sys.exit("This makes one real (tiny) paid request. Re-run with --approve-spend.")

questions = [
    dc.predicate("standalone", "Can a viewer understand this clip without any other context?"),
    dc.score("hook", "How strongly does the first sentence capture attention?",
             [("Poor", "Filler or a greeting."), ("Average", "On topic but flat."),
              ("Good", "A clear claim or question."), ("Excellent", "Surprising and vivid.")]),
]
clip = ("Nobody tells you that quitting your job is the easy part. The hard part starts three weeks later, "
        "when the money stops. That one phone call paid my rent for six months.")
try:
    res = dc.Client(retries=0).decide(clip, questions)
except dc.DecisionsUnavailable as e:
    sys.exit(f"FAILED: {e}")
for name, a in res.answers.items():
    print(name, a.type, a.probability if a.type == "predicate" else (a.score, a.level, a.confidence))
print("dropped (malformed or missing):", res.dropped)
print("usage:", res.usage, "cost USD:", res.cost_usd, "seconds:", res.seconds)
