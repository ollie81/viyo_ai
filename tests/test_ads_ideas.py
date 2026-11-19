"""Daily ideas and Campaign Universe: ideas are free text made only when asked, limited per day, kept to the
brand's facts, private to the owner; using one starts an ad (not paid until generated) with the idea as its
concept and track; the concept reaches the hook and script prompts."""
import uuid

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import ads_brands
import ads_ideas
import ads_maker
import ads_studio as core
from test_ads_brands import KIVU, brand, env as _brand_env  # noqa: F401
from test_ads_maker import ALICE, BOB, token

B = "/api/v1/ads-maker/brands"


@pytest.fixture
def env(_brand_env, monkeypatch):
    _brand_env["db"].db["ad_ideas"] = []
    calls = []

    def fake_generate(b, campaigns, seen):
        calls.append({"brand": b["id"], "seen": list(seen), "campaigns": len(campaigns)})
        n = len(calls)
        return ads_ideas.clean_ideas([
            ads_ideas.IdeaSchema(title=f"Sunset sips {n}", track="lifestyle", hook="Kigali afternoons hit different.",
                                 concept="Friends share cans at sunset.", beats=["Heat", "Can opens", "Smiles"]),
            ads_ideas.IdeaSchema(title=f"Can close-up {n}", track="product_intro", hook="Meet Kivu Sunrise.",
                                 concept="Only RWF 800.", beats=["Can on ice"]),
            ads_ideas.IdeaSchema(title=f"Miracle {n}", track="weird", hook="Clinically proven to cure thirst forever.",
                                 concept="x", beats=[]),
        ], b), 1
    monkeypatch.setattr(ads_ideas, "_generate", fake_generate)
    app = FastAPI()
    app.include_router(ads_ideas.router)
    app.include_router(ads_brands.router)
    app.include_router(ads_maker.router)
    _brand_env["c"] = TestClient(app)
    _brand_env["calls"] = calls
    return _brand_env


def test_opening_ideas_generates_nothing(env):
    b = brand(env)
    r = env["c"].get(f"{B}/{b['id']}/ideas", headers=token(ALICE)).json()
    assert r["today"] == [] and r["can_get_more"] is True and not env["calls"]
    assert [t["key"] for t in r["tracks"]] == list(ads_ideas.TRACKS)


def test_new_ideas_stick_to_the_facts(env):
    b = brand(env)
    r = env["c"].post(f"{B}/{b['id']}/ideas", headers=token(ALICE))
    assert r.status_code == 200, r.text
    ideas = r.json()["today"]
    titles = [i["title"] for i in ideas]
    assert "Miracle 1" not in titles, "an idea whose hook was an invented claim is dropped"
    assert len(ideas) == 2
    intro = next(i for i in ideas if i["track"] == "product_intro")
    assert intro["concept"] == "Only RWF 800.", "a supplied price stays"
    assert all(i["status"] == "new" for i in ideas)
    assert len(env["calls"]) == 1


def test_unsupported_claims_are_removed_from_ideas():
    b = {"name": "Kivu", "facts": {"prices": ["RWF 800"]}}
    out = ads_ideas.clean_ideas([ads_ideas.IdeaSchema(
        title="Fresh start", track="demo", hook="See how it pours.",
        concept="Pour it over ice. 100% organic and the best in Rwanda.", beats=["Only $2!", "Pour", "Sip"])], b)
    assert out[0]["concept"] == "Pour it over ice." and out[0]["beats"] == ["Pour", "Sip"]
    assert any("organic" in r for r in out[0]["removed"])


def test_daily_limits(env):
    b = brand(env)
    assert env["c"].post(f"{B}/{b['id']}/ideas", headers=token(ALICE)).status_code == 200
    r = env["c"].post(f"{B}/{b['id']}/ideas", headers=token(ALICE))
    assert r.status_code == 200 and r.json()["can_get_more"] is False
    assert env["calls"][1]["seen"][:2] == ["Sunset sips 1", "Can close-up 1"], "earlier ideas aren't repeated"
    assert env["c"].post(f"{B}/{b['id']}/ideas", headers=token(ALICE)).status_code == 429
    assert len(env["calls"]) == 2, "no AI call once the limit is reached"


def test_user_daily_limit_across_brands(env, monkeypatch):
    monkeypatch.setattr(ads_ideas, "IDEA_BATCHES_PER_USER_PER_DAY", 1)
    b1, b2 = brand(env), brand(env, name="Other")
    assert env["c"].post(f"{B}/{b1['id']}/ideas", headers=token(ALICE)).status_code == 200
    assert env["c"].post(f"{B}/{b2['id']}/ideas", headers=token(ALICE)).status_code == 429


def test_ideas_are_private(env):
    b = brand(env)
    env["c"].post(f"{B}/{b['id']}/ideas", headers=token(ALICE))
    idea = env["db"].db["ad_ideas"][0]
    assert env["c"].get(f"{B}/{b['id']}/ideas", headers=token(BOB)).status_code == 404
    assert env["c"].post(f"{B}/{b['id']}/ideas", headers=token(BOB)).status_code == 404
    assert env["c"].put(f"{B}/{b['id']}/ideas/{idea['id']}", json={"status": "saved"}, headers=token(BOB)).status_code == 404
    assert env["c"].post(f"{B}/{b['id']}/ideas/{idea['id']}/use", json={}, headers=token(BOB)).status_code == 404
    # an idea id from another brand doesn't work through this brand
    b2 = brand(env, BOB, name="Bob's")
    assert env["c"].put(f"{B}/{b2['id']}/ideas/{idea['id']}", json={"status": "saved"}, headers=token(BOB)).status_code == 404


def test_save_dismiss_and_saved_ideas_stay(env):
    b = brand(env)
    env["c"].post(f"{B}/{b['id']}/ideas", headers=token(ALICE))
    a, d = env["db"].db["ad_ideas"][0], env["db"].db["ad_ideas"][1]
    assert env["c"].put(f"{B}/{b['id']}/ideas/{a['id']}", json={"status": "saved"}, headers=token(ALICE)).status_code == 200
    env["c"].put(f"{B}/{b['id']}/ideas/{d['id']}", json={"status": "dismissed"}, headers=token(ALICE))
    a["day"] = "2026-01-01"            # saved yesterday: still listed
    r = env["c"].get(f"{B}/{b['id']}/ideas", headers=token(ALICE)).json()
    assert [i["id"] for i in r["saved"]] == [a["id"]] and r["today"] == []
    assert env["c"].put(f"{B}/{b['id']}/ideas/{a['id']}", json={"status": "used"}, headers=token(ALICE)).status_code == 422


def test_using_an_idea_starts_an_ad_with_its_concept_and_track_and_charges_nothing(env):
    b = brand(env)
    env["c"].post(f"{B}/{b['id']}/ideas", headers=token(ALICE))
    idea = next(i for i in env["db"].db["ad_ideas"] if i["track"] == "lifestyle")
    r = env["c"].post(f"{B}/{b['id']}/ideas/{idea['id']}/use", json={"aspect_ratio": "1:1"}, headers=token(ALICE))
    assert r.status_code == 200, r.text
    c = next(x for x in env["db"].db["ad_campaigns"] if x["id"] == r.json()["id"])
    assert c["brand_id"] == b["id"] and c["track"] == "lifestyle" and c["ad_style"] == "lifestyle"
    assert "Sunset sips 1" in c["creative_brief"] and "Kigali afternoons" in c["creative_brief"]
    assert c.get("status", "draft") == "draft" and not env["db"].calls, "no credits held or spent"
    assert idea["status"] == "used" and idea["campaign_id"] == c["id"]
    assert env["c"].post(f"{B}/{b['id']}/ideas/{idea['id']}/use", json={}, headers=token(ALICE)).status_code == 400
    # the brand page groups its ads by track
    got = env["c"].get(f"{B}/{b['id']}", headers=token(ALICE)).json()
    assert got["campaigns"][0]["track"] == "lifestyle"


def test_the_concept_reaches_the_hook_and_script_prompts():
    c = {"target_name": "Kivu", "objective": "awareness", "creative_brief": "Sunset sips. Opening: heat."}
    assert "CREATIVE CONCEPT" in core._objective_text(c) and "Sunset sips" in core._objective_text(c)
    assert core._objective_text({"objective": ""}) == "drive interest and engagement"
    assert "Sunset sips" in core._build_hook_prompt(c, None)


def test_new_ad_with_a_track(env):
    b = brand(env)
    r = env["c"].post(f"{B}/{b['id']}/campaigns", json={"track": "seasonal"}, headers=token(ALICE))
    assert next(x for x in env["db"].db["ad_campaigns"] if x["id"] == r.json()["id"])["track"] == "seasonal"
    r = env["c"].post(f"{B}/{b['id']}/campaigns", json={"track": "bogus"}, headers=token(ALICE))
    assert "track" not in next(x for x in env["db"].db["ad_campaigns"] if x["id"] == r.json()["id"])


def test_prompt_is_built_from_the_brand_and_its_facts(env):
    b = env["db"].db["ad_brands"] = [{"id": "b", "name": "Kivu Sunrise", "promote_target": "physical_product",
                                      "facts": {"prices": ["RWF 800"]}}]
    p = ads_ideas.build_prompt(b[0], [{"track": "lifestyle"}], ["Old idea"])
    assert "RWF 800" in p and "Lifestyle: 1" in p and "Old idea" in p and "Never invent prices" in p
