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


def _slot(name: str) -> str | None:
    v = (os.environ.get(name) or "").strip()
    return v if re.fullmatch(r"\d{6,20}", v) else None


def config() -> dict:
    """display: the responsive display unit (VIYO_Web_Display). infeed / multiplex: optional extra AdSense units
    (WEB_ADS_INFEED_SLOT + WEB_ADS_INFEED_LAYOUT_KEY, WEB_ADS_MULTIPLEX_SLOT), each used only once its id is set."""
    client = (os.environ.get("WEB_ADS_CLIENT") or DEFAULT_CLIENT).strip()
    slot = (os.environ.get("WEB_ADS_SLOT") or DEFAULT_SLOT).strip()
    valid = bool(re.fullmatch(r"ca-pub-\d{10,20}", client) and re.fullmatch(r"\d{6,20}", slot))
    on = (os.environ.get("WEB_ADS_ENABLED") or "").strip().lower() == "true"
    layout_key = (os.environ.get("WEB_ADS_INFEED_LAYOUT_KEY") or "").strip()
    infeed = _slot("WEB_ADS_INFEED_SLOT")
    if not re.fullmatch(r"[-+0-9a-z]{4,40}", layout_key):
        infeed = None                                   # an in-feed unit needs its layout key too
    return {"enabled": on and valid, "client": client if valid else None, "slot": slot if valid else None,
            "infeed_slot": infeed if valid else None, "infeed_layout_key": layout_key if infeed and valid else None,
            "multiplex_slot": _slot("WEB_ADS_MULTIPLEX_SLOT") if valid else None}


@router.get("/web-ads/config")
async def web_ads_config():
    return config()
