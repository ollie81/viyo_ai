"""Campaign Universe and daily advertising ideas.

Campaign Universe: every ad of a brand belongs to one of its ongoing campaign tracks (product introduction,
lifestyle, character story, demonstration, seasonal, new concept), so a brand builds a connected series of
ads instead of one-offs.

Daily ideas: new ad concepts for a saved brand, written from the brand's own details and verified facts.
  * Free to the creator: an idea is text only (a title, a hook, the concept, a few story beats). Nothing is
    generated beyond that, and nothing is charged.
  * Generated only when the creator asks (opens their ideas), never by a background job, and at most
    IDEA_BATCHES_PER_BRAND_PER_DAY times per brand and IDEA_BATCHES_PER_USER_PER_DAY times per creator per
    day. Each batch is one cheap text call, logged and kept under the platform's daily AI cap.
  * Ideas follow the same facts rule as ads: anything the brand didn't supply (prices, specs, ingredients,
    results, guarantees...) is removed from every idea; an idea left without a hook is dropped.
  * Using an idea starts a normal ad from the brand with the idea as its creative concept. That ad is paid
    the usual way: the price is shown first and nothing is charged until the creator generates it.
"""
from __future__ import annotations

import datetime as _dt
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

import ad_claims
import ads_studio as core
from ads_brands import _brand_assets, _db, _uuid, owned_brand  # noqa: F401
from creator_studio import creator

router = APIRouter(prefix="/api/v1/ads-maker/brands", tags=["ads_maker"])

TRACKS = {
    "product_intro": ("Product introduction", "product_cinematic",
                      "Introduce the product: what it is and why it matters, the product is the hero."),
    "lifestyle": ("Lifestyle", "lifestyle", "The product in real everyday moments of the people it's for."),
    "character_story": ("Character story", "character",
                        "A short story carried by the brand's character, with the product at the turning point."),
    "demo": ("Demonstration", "demo", "Show exactly how it works or what it does, step by step."),
    "seasonal": ("Seasonal", "social",
                 "Tied to the current season or time of year, without inventing promotions, discounts or events."),
    "new_concept": ("New concept", "auto", "A fresh creative angle the brand hasn't tried yet."),
}
IDEAS_PER_BATCH = 3
IDEA_BATCHES_PER_BRAND_PER_DAY = 2
IDEA_BATCHES_PER_USER_PER_DAY = 6


def _today() -> str:
    return _dt.datetime.now(_dt.timezone.utc).date().isoformat()


class IdeaSchema(BaseModel):
    title: str
    track: str
    hook: str
    concept: str
    beats: list[str]
    fact_ids: list[str] = []


class IdeasSchema(BaseModel):
    ideas: list[IdeaSchema]


_IDEAS_PROMPT = """You are a creative director planning the next short video ads for one brand.

BRAND: {name} ({kind}{subtype})
DESCRIPTION: {description}
AUDIENCE: {audience}
{facts}
{character}
THIS BRAND'S ADS SO FAR, by campaign track: {done}
IDEAS ALREADY SUGGESTED (don't repeat them): {seen}
TODAY: {today} ({season} in the northern hemisphere; the brand may be anywhere, so keep seasons general)

Write exactly {n} new, distinct ad ideas. Prefer tracks this brand has done least. Each idea:
- title: 3-7 words
- track: one of {tracks}
- hook: the first line or moment of the ad, under 15 words
- concept: 1-3 sentences on the story and why it fits the brand
- beats: 3-5 short story beats (scene by scene), ending on the product and the call to action
- fact_ids: the ids of the verified facts the idea relies on (may be empty)
Never invent prices, discounts, promotions, specs, ingredients, health or results claims, awards, guarantees,
events or contact details. An idea that needs a fact the brand didn't supply must work without it."""


def _brand_as_campaign(b: dict) -> dict:
    return {"target_name": b.get("name"), "brand_name": b.get("name"), "target_description": b.get("description"),
            "facts": b.get("facts") or {}, "social_links": b.get("social_links") or [], "contact": b.get("contact") or {},
            "destination_link": b.get("website") or ""}


def _season(month: int) -> str:
    return ("winter", "winter", "spring", "spring", "spring", "summer", "summer", "summer", "autumn", "autumn",
            "autumn", "winter")[month - 1]


def _character_line(b: dict) -> str:
    if not b.get("character_id"):
        return "CHARACTER: none saved (ideas may be product-led or use a new character)."
    rows = _db().table("ad_characters").select("*").eq("id", b["character_id"]).limit(1).execute().data or []
    if not rows:
        return "CHARACTER: none saved."
    r = rows[0]
    return f"CHARACTER (the brand's own, reuse it in character ideas): {r.get('name') or 'unnamed'}, " \
           f"{(r.get('spec') or {}).get('appearance', '')}"


def build_prompt(b: dict, campaigns: list[dict], seen: list[str]) -> str:
    counts = {k: 0 for k in TRACKS}
    for c in campaigns:
        if c.get("track") in counts:
            counts[c["track"]] += 1
    today = _dt.datetime.now(_dt.timezone.utc).date()
    return _IDEAS_PROMPT.format(
        name=b.get("name") or "the brand", kind=core.PRODUCT_KINDS.get(b.get("promote_target") or "", "a product"),
        subtype=f", {b['product_subtype']}" if b.get("product_subtype") else "",
        description=b.get("description") or "(none given)", audience=b.get("target_audience") or "(not given)",
        facts=ad_claims.FactSheet.from_campaign(_brand_as_campaign(b)).prompt_block(),
        character=_character_line(b),
        done=", ".join(f"{TRACKS[k][0]}: {v}" for k, v in counts.items()),
        seen="; ".join(seen[-30:]) or "none", today=today.isoformat(), season=_season(today.month),
        n=IDEAS_PER_BATCH, tracks=", ".join(TRACKS))


def clean_ideas(parsed: list, b: dict) -> list[dict]:
    """The facts rule on every idea; unknown tracks become new_concept; an idea without a hook is dropped."""
    sheet = ad_claims.FactSheet.from_campaign(_brand_as_campaign(b))
    out = []
    for i in parsed:
        hook, r1 = ad_claims.enforce(i.hook or "", sheet)
        concept, r2 = ad_claims.enforce(i.concept or "", sheet)
        # The title is a working name shown only to the creator, never in the ad ("Fresh start" is fine there);
        # everything the ad could say (hook, concept, beats) goes through the facts rule.
        title = (i.title or "").strip()
        beats, removed = [], r1 + r2
        for beat in (i.beats or [])[:5]:
            kept, r = ad_claims.enforce(beat, sheet)
            removed += r
            if kept:
                beats.append(kept[:200])
        if not hook.strip() or not title.strip():
            continue
        track = i.track if i.track in TRACKS else "new_concept"
        out.append({"title": title[:80], "track": track, "hook": hook[:200], "concept": concept[:600], "beats": beats,
                    "removed": removed})
    return out


def _generate(b: dict, campaigns: list[dict], seen: list[str]) -> tuple[list[dict], int]:
    """One text call. Returns (cleaned ideas, cost in cents). Stubbed in tests; never called by a background job."""
    from google.genai import types
    if core._gemini_client is None:
        raise HTTPException(status_code=503, detail="Ideas aren't available right now.")
    prompt = build_prompt(b, campaigns, seen)
    core._check_daily_cap(core._estimate_text_cost_cents(prompt))
    try:
        res = core._gemini_client.models.generate_content(
            model=core.GEMINI_TEXT_MODEL, contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=IdeasSchema))
    except Exception:
        raise HTTPException(status_code=502, detail="Couldn't come up with ideas right now. Try again in a minute.")
    if res.parsed is None:
        raise HTTPException(status_code=502, detail="Couldn't come up with ideas right now. Try again in a minute.")
    cents = core._text_cost_cents(res.usage_metadata)
    core._log_cost(None, "ad_ideas", cents)
    return clean_ideas(res.parsed.ideas, b), cents


def _idea_out(r: dict) -> dict:
    return {"id": r["id"], "title": r.get("title") or "", "track": r.get("track") or "new_concept",
            "track_label": TRACKS.get(r.get("track") or "", TRACKS["new_concept"])[0], "hook": r.get("hook") or "",
            "concept": r.get("concept") or "", "beats": r.get("beats") or [], "status": r.get("status") or "new",
            "campaign_id": r.get("campaign_id"), "day": r.get("day")}


def _ideas(brand_id: str) -> list[dict]:
    return _db().table("ad_ideas").select("*").eq("brand_id", brand_id).order("created_at", desc=True).execute().data or []


def _batches_today(rows: list[dict]) -> int:
    return len({r.get("batch") for r in rows if r.get("day") == _today()})


# ---------------------------------------------------------------- routes

@router.get("/{brand_id}/ideas")
async def list_ideas(brand_id: str, user_id: str = Depends(creator)):
    """Today's ideas and the ones the creator saved. Free; nothing is generated here."""
    b = owned_brand(brand_id, user_id)
    rows = [r for r in _ideas(b["id"]) if r.get("owner_id") == user_id]
    today = [r for r in rows if r.get("day") == _today() and r.get("status") in ("new", "saved")]
    saved = [r for r in rows if r.get("status") == "saved" and r.get("day") != _today()]
    return {"today": [_idea_out(r) for r in today], "saved": [_idea_out(r) for r in saved],
            "can_get_more": _batches_today(rows) < IDEA_BATCHES_PER_BRAND_PER_DAY,
            "tracks": [{"key": k, "label": v[0], "description": v[2]} for k, v in TRACKS.items()]}


@router.post("/{brand_id}/ideas")
async def new_ideas(brand_id: str, user_id: str = Depends(creator)):
    """A new batch of ideas for this brand (free; text only), within the daily limits."""
    b = owned_brand(brand_id, user_id)
    if b.get("archived"):
        raise HTTPException(status_code=400, detail="This brand is archived. Restore it first.")
    rows = [r for r in _ideas(b["id"]) if r.get("owner_id") == user_id]
    if _batches_today(rows) >= IDEA_BATCHES_PER_BRAND_PER_DAY:
        raise HTTPException(status_code=429, detail="That's today's ideas for this brand. New ones tomorrow.")
    mine_today = _db().table("ad_ideas").select("*").eq("owner_id", user_id).eq("day", _today()).execute().data or []
    if len({(r.get("brand_id"), r.get("batch")) for r in mine_today}) >= IDEA_BATCHES_PER_USER_PER_DAY:
        raise HTTPException(status_code=429, detail="That's today's ideas. New ones tomorrow.")
    campaigns = [c for c in (_db().table("ad_campaigns").select("*").eq("brand_id", b["id"]).execute().data or [])
                 if c.get("user_id") == user_id]
    ideas, _cents = _generate(b, campaigns, [r.get("title") or "" for r in rows])
    if not ideas:
        raise HTTPException(status_code=502, detail="Couldn't come up with ideas that stick to your facts. Try again.")
    batch = len({r.get("batch") for r in rows}) + 1
    for i in ideas:
        _db().table("ad_ideas").insert({"brand_id": b["id"], "owner_id": user_id, "day": _today(), "batch": batch,
                                        "title": i["title"], "track": i["track"], "hook": i["hook"],
                                        "concept": i["concept"], "beats": i["beats"], "status": "new"}).execute()
    return await list_ideas(b["id"], user_id)


def _owned_idea(brand: dict, idea_id: str, user_id: str) -> dict:
    rows = _db().table("ad_ideas").select("*").eq("id", _uuid(idea_id)).limit(1).execute().data or []
    if not rows or rows[0].get("owner_id") != user_id or rows[0].get("brand_id") != brand["id"]:
        raise HTTPException(status_code=404, detail="Not found.")
    return rows[0]


class IdeaStatusIn(BaseModel):
    status: str = Field(..., pattern="^(saved|dismissed|new)$")


@router.put("/{brand_id}/ideas/{idea_id}")
async def set_idea_status(brand_id: str, idea_id: str, body: IdeaStatusIn, user_id: str = Depends(creator)):
    """Save an idea for later, dismiss it, or bring it back."""
    b = owned_brand(brand_id, user_id)
    i = _owned_idea(b, idea_id, user_id)
    if i.get("status") == "used":
        raise HTTPException(status_code=400, detail="This idea already became an ad.")
    _db().table("ad_ideas").update({"status": body.status}).eq("id", i["id"]).execute()
    return _idea_out({**i, "status": body.status})


class UseIdeaIn(BaseModel):
    aspect_ratio: str = "9:16"
    duration_seconds: int = 15
    use_character: bool = True


@router.post("/{brand_id}/ideas/{idea_id}/use")
async def use_idea(brand_id: str, idea_id: str, body: UseIdeaIn, user_id: str = Depends(creator)):
    """Starts an ad from the brand with this idea as its creative concept. Nothing is charged here: the ad is
    paid when the creator generates it, after seeing the price."""
    import ads_brands
    b = owned_brand(brand_id, user_id)
    i = _owned_idea(b, idea_id, user_id)
    if i.get("status") == "used" and i.get("campaign_id"):
        raise HTTPException(status_code=400, detail="This idea already became an ad.")
    track = i.get("track") if i.get("track") in TRACKS else "new_concept"
    style = TRACKS[track][1]
    use_character = body.use_character and track != "product_intro"
    campaign = await ads_brands.new_ad_from_brand(b["id"], ads_brands.NewAdIn(
        ad_style=style, aspect_ratio=body.aspect_ratio, duration_seconds=body.duration_seconds,
        use_character=use_character, track=track), user_id)
    beats = "; ".join(i.get("beats") or [])
    brief = f"{i.get('title')}. Opening: {i.get('hook')}. {i.get('concept')}" + (f" Story beats: {beats}." if beats else "")
    _db().table("ad_campaigns").update({"creative_brief": brief[:1500]}).eq("id", campaign["id"]).execute()
    _db().table("ad_ideas").update({"status": "used", "campaign_id": campaign["id"]}).eq("id", i["id"]).execute()
    import ads_maker
    return ads_maker._out(ads_maker.owned_campaign(campaign["id"], user_id))
