"""Story Intelligence stage 4: VIYO reviews each chapter with verified quotes, revises only weak passages at most
once, and keeps a revision only if it is demonstrably better. Real Postgres and job queue, fake AI."""
import copy

import pytest

import studio_ai
import studio_novel_review as rv
from test_creator_studio import H, balance, env, grant, run, studio_schema  # noqa: F401
from test_story_chapters import approved_story, chapters_schema, nv, write  # noqa: F401


def para(first: str, n: int = 160) -> str:
    return first + " " + " ".join(["The light lay quietly on the worn wooden floor of the shop."] * (n // 11))


P1 = para("Amani found the letter on a Tuesday, folded small inside the hem of the ivory dress.")
P2 = para("Mama Grace told her everything was fine and that she should not worry about anything.")
P3 = para("Eric stood in the doorway and said nothing at all, his watch stopped at midnight.")
CHAPTER = "\n\n".join([P1, P2, P3]) + "\n\n" + " ".join(["More quiet light."] * 500)


def scores(**over):
    base = {"originality": (8, "folded small inside the hem", 1), "character": (8, "Amani found the letter on", 1),
            "emotion": (8, "folded small inside the hem", 1), "plot": (8, "said nothing at all, his watch", 3),
            "dialogue": (8, "folded small inside the hem", 1), "prose": (8, "folded small inside the hem", 1),
            "cinematic": (8, "Eric stood in the doorway", 3), "opening": (8, "Amani found the letter on a Tuesday", 1),
            "ending": (8, "More quiet light. More quiet light.", 4)}
    base.update(over)
    return {"scores": {k: {"score": s, "quote": q, "passage": p, "reason": f"{k} reason"} for k, (s, q, p) in base.items()},
            "weak": [], "strengths": ["A strong opening image."]}


WEAK_EMOTION = {**scores(emotion=(5, "she should not worry about anything", 2)),
                "weak": [{"area": "emotion", "passage": 2, "why": "Mama Grace's reassurance is told, not felt.",
                          "fix": "Show her hands betraying her calm."}]}
FIXED_P2 = para("Mama Grace smoothed the dress twice, though it lay flat, and her hands would not stop shaking.")


# ------------------------------------------------------------------------------------------- pure rules

def test_only_quoted_found_and_well_placed_scores_count():
    ps = rv.passages(CHAPTER)
    assert len(ps) == 4 and ps[0].startswith("Amani found")
    ev = rv.verify(scores(), ps)
    assert ev["status"] == "passed" and ev["categories"]["opening"]["passage"] == 1
    bad = rv.verify(scores(originality=(9, "a sentence that is not in the chapter", 1),
                           ending=(8, "Amani found the letter on a Tuesday", 1)), ps)
    assert bad["status"] == "incomplete"
    assert bad["categories"]["originality"]["why"] == "Its quote isn't in the chapter."
    assert bad["categories"]["ending"]["why"] == "The ending must be quoted from the last passage."
    missing = rv.verify({"scores": {}}, ps)
    assert missing["status"] == "incomplete" and set(missing["unverified"]) == set(rv.required(CHAPTER))


def test_dialogue_is_only_judged_when_the_chapter_has_dialogue():
    assert "dialogue" not in rv.required(CHAPTER)
    talk = CHAPTER + ' "Where is it?" she asked. "Gone," he said. "Gone where?" "Away."'
    assert "dialogue" in rv.required(talk)


def test_weaknesses_are_kept_only_for_verified_areas_below_the_bar():
    ps = rv.passages(CHAPTER)
    raw = {**scores(emotion=(5, "she should not worry about anything", 2)),
           "weak": [{"area": "emotion", "passage": 2, "why": "told", "fix": "show"},
                    {"area": "prose", "passage": 1, "why": "could be better", "fix": "rewrite"},       # prose passed
                    {"area": "opening", "passage": 3, "why": "x", "fix": "y"}]}                     # opening passed
    ev = rv.verify(raw, ps)
    assert ev["status"] == "needs_work" and [w["area"] for w in ev["weak"]] == ["emotion"]
    assert rv.targets(ev) == [2]                                    # a good passage is never a target


def test_a_revision_is_kept_only_when_every_target_improved_and_nothing_else_dropped():
    ps = rv.passages(CHAPTER)
    before = rv.verify(WEAK_EMOTION, ps)
    better = rv.verify(scores(emotion=(8, "her hands would not stop shaking", 2)), rv.splice(ps, {"passages": [{"number": 2, "text": FIXED_P2}]}, [2]))
    assert rv.improved(before, better) == (True, "every targeted area improved and nothing else got worse")
    same = rv.verify(WEAK_EMOTION, ps)
    assert rv.improved(before, same)[1].startswith("emotional depth didn't improve")
    worse = rv.verify(scores(emotion=(8, "folded small inside the hem", 1), prose=(6, "folded small inside the hem", 1)), ps)
    assert "literary quality got worse (8 → 6)" == rv.improved(before, worse)[1]
    incomplete = rv.verify({"scores": {}}, ps)
    assert rv.improved(before, incomplete) == (False, "the new review was less complete")


def test_splice_replaces_only_allowed_passages_and_refuses_bad_rewrites():
    ps = rv.passages(CHAPTER)
    new = rv.splice(ps, {"passages": [{"number": 2, "text": FIXED_P2}, {"number": 1, "text": "sneaky change"}]}, [2])
    assert new[1] == FIXED_P2 and new[0] == ps[0] and new[2:] == ps[2:]
    assert rv.splice(ps, {"passages": [{"number": 2, "text": "too short"}]}, [2]) is None
    assert rv.splice(ps, {"passages": [{"number": 2, "text": ps[1] * 3}]}, [2]) is None
    assert rv.splice(ps, {"passages": []}, [2]) is None


# ------------------------------------------------------------------------------------------- inside the chapter job

READ = {"summary": "Amani finds the letter.", "facts": [], "claims": []}


def reviews(*results):
    seq = list(results)
    return lambda prompt: seq.pop(0) if len(seq) > 1 else seq[0]


def test_a_passing_chapter_is_not_revised(nv):
    client, db, store, ai, gen, u = nv
    sid = approved_story(client, ai, u)
    ai.novel_results = {"chapter": CHAPTER, "chapter review": scores(), "chapter memory": READ}
    write(client, u, sid, 1)
    run(gen)
    got = client.get(f"/api/v1/studio/stories/{sid}/chapters/1", headers=H(u)).json()["chapter"]
    assert got["review"] == "passed" and got["revised"] is None
    assert got["review_report"]["categories"]["opening"]["status"] == "verified"
    assert not [w for w, _ in ai.novel_prompts if w == "chapter revision"]


def test_a_weak_passage_is_revised_and_kept_when_the_second_review_is_better(nv):
    client, db, store, ai, gen, u = nv
    sid = approved_story(client, ai, u)
    ai.novel_results = {"chapter": CHAPTER, "chapter memory": READ,
                        "chapter review": reviews(WEAK_EMOTION, scores(emotion=(8, "her hands would not stop shaking", 2))),
                        "chapter revision": {"passages": [{"number": 2, "text": FIXED_P2}]}}
    write(client, u, sid, 1)
    run(gen)
    got = client.get(f"/api/v1/studio/stories/{sid}/chapters/1", headers=H(u)).json()["chapter"]
    rep = got["review_report"]
    assert got["revised"] is True and rep["status"] == "passed" and rep["first"]["status"] == "needs_work"
    assert rep["revision"]["passages"] == [2] and rep["revision"]["scores_before"]["emotion"] == 5
    assert "her hands would not stop shaking" in got["text"] and "should not worry about anything" not in got["text"]
    assert got["text"].startswith(P1)                                 # untouched passages kept word for word
    rev_prompt = [p for w, p in ai.novel_prompts if w == "chapter revision"][0]
    assert "Rewrite ONLY passages 2" in rev_prompt and "Show her hands betraying her calm." in rev_prompt
    # The memory is read from the FINAL text.
    assert "her hands would not stop shaking" in [p for w, p in ai.novel_prompts if w == "chapter memory"][-1]


def test_a_revision_that_isnt_better_is_thrown_away(nv):
    client, db, store, ai, gen, u = nv
    sid = approved_story(client, ai, u)
    ai.novel_results = {"chapter": CHAPTER, "chapter memory": READ,
                        "chapter review": reviews(WEAK_EMOTION, {**scores(emotion=(5, "her hands would not stop shaking", 2)), "weak": []}),
                        "chapter revision": {"passages": [{"number": 2, "text": FIXED_P2}]}}
    write(client, u, sid, 1)
    run(gen)
    got = client.get(f"/api/v1/studio/stories/{sid}/chapters/1", headers=H(u)).json()["chapter"]
    assert got["revised"] is False and got["review"] == "needs_work"
    assert got["review_report"]["revision"]["why"] == "emotional depth didn't improve (5 → 5)"
    assert "should not worry about anything" in got["text"]          # the original is kept
    assert got["review_report"]["weak"][0]["why"] == "Mama Grace's reassurance is told, not felt."


def test_a_whole_chapter_problem_is_reported_not_patched(nv):
    client, db, store, ai, gen, u = nv
    sid = approved_story(client, ai, u)
    whole = {**scores(plot=(5, "said nothing at all, his watch", 3)),
             "weak": [{"area": "plot", "passage": 0, "why": "Nothing changes.", "fix": "Give Amani a choice."}]}
    ai.novel_results = {"chapter": CHAPTER, "chapter review": whole, "chapter memory": READ}
    write(client, u, sid, 1)
    run(gen)
    got = client.get(f"/api/v1/studio/stories/{sid}/chapters/1", headers=H(u)).json()["chapter"]
    assert got["review"] == "needs_work" and "ask for a new version" in got["review_report"]["revision"]["why"]
    assert not [w for w, _ in ai.novel_prompts if w == "chapter revision"]


def test_an_unverifiable_review_is_never_a_pass_and_triggers_no_rewrite(nv):
    client, db, store, ai, gen, u = nv
    sid = approved_story(client, ai, u)
    fake = scores(**{k: (10, "words the chapter never contains", 1) for k in ("originality", "prose")})
    ai.novel_results = {"chapter": CHAPTER, "chapter review": fake, "chapter memory": READ}
    write(client, u, sid, 1)
    run(gen)
    got = client.get(f"/api/v1/studio/stories/{sid}/chapters/1", headers=H(u)).json()["chapter"]
    assert got["review"] == "incomplete" and set(got["review_report"]["unverified"]) == {"originality", "prose"}
    assert not [w for w, _ in ai.novel_prompts if w == "chapter revision"]


def test_a_failed_review_or_revision_doesnt_lose_the_chapter(nv):
    client, db, store, ai, gen, u = nv
    sid = approved_story(client, ai, u)

    def down(prompt):
        raise studio_ai.ProviderError("down")
    ai.novel_results = {"chapter": CHAPTER, "chapter review": down, "chapter memory": READ}
    write(client, u, sid, 1)
    run(gen)
    got = client.get(f"/api/v1/studio/stories/{sid}/chapters/1", headers=H(u)).json()["chapter"]
    assert got["review"] == "unavailable" and got["words"] > 1000
    ai.novel_results = {"chapter": CHAPTER, "chapter review": WEAK_EMOTION, "chapter revision": down, "chapter memory": READ}
    write(client, u, sid, 1)
    run(gen)
    got = client.get(f"/api/v1/studio/stories/{sid}/chapters/1", headers=H(u)).json()["chapter"]
    assert got["version"] == 2 and got["revised"] is False
    assert got["review_report"]["revision"]["why"] == "the revision couldn't be made"
