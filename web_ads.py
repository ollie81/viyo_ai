"""Google AdSense on the web app (viyo.site/app): one switch, off until the owner turns it on.

The app asks GET /api/v1/web-ads/config; ads load only when WEB_ADS_ENABLED is "true" (Railway variable), which the
owner sets once AdSense has approved viyo.site, ads.txt is live and the consent message is set up in AdSense. The
publisher id and slot are public identifiers (they appear in every page that shows the ad), never credentials.
Web ads are VIYO's own revenue: they are not counted for creators (creator earnings are unchanged)."""
import os
import re

from fastapi import APIRouter

router = APIRouter(prefix="/api/v1", tags=["web-ads"])

DEFAULT_CLIENT = "ca-pub-4161816109605295"
DEFAULT_SLOT = "6118261645"            # ad unit VIYO_Web_Display (responsive display)


def config() -> dict:
    client = (os.environ.get("WEB_ADS_CLIENT") or DEFAULT_CLIENT).strip()
    slot = (os.environ.get("WEB_ADS_SLOT") or DEFAULT_SLOT).strip()
    valid = bool(re.fullmatch(r"ca-pub-\d{10,20}", client) and re.fullmatch(r"\d{6,20}", slot))
    on = (os.environ.get("WEB_ADS_ENABLED") or "").strip().lower() == "true"
    return {"enabled": on and valid, "client": client if valid else None, "slot": slot if valid else None}


@router.get("/web-ads/config")
async def web_ads_config():
    return config()
