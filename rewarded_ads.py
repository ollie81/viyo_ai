"""
Crediting Viyo Coins for watching a rewarded video ad.

Two ways, chosen by the REWARDED_AD_SSV switch (Railway variable):

  off (default, today's behaviour)
      The app calls POST /coins/rewarded-ad/claim after the SDK reports the
      reward. Nothing proves an ad was watched, so the only defence is the
      daily cap.

  on
      Coins are credited only when Google confirms the reward. The app calls
      POST /coins/rewarded-ad/start for a one-time session, shows the ad with
      that session and its user id attached (ServerSideVerificationOptions),
      and AdMob calls GET /admob/ssv, signed with Google's key. This file
      checks that signature against Google's published keys, then
      rewarded_ad_credit() (rewarded_ads_ssv.sql) credits the coins once per
      session and once per AdMob transaction, inside the daily cap, through
      the same locked ledger step as every other coin change. The app polls
      GET /coins/rewarded-ad/{session} until the coins arrive. With the switch
      on, /claim refuses, so nobody can skip the ad.

Turning it on needs, in this order: rewarded_ads_ssv.sql run, the callback URL
(https://<backend>/api/v1/admob/ssv) set on the ad unit in the AdMob console,
an app release that uses /start, then REWARDED_AD_SSV=on.

Ad coins never become creator cash: creators earn only from bought coins
(wallet.paid_portion), so watching ads can't be turned into money.
"""
import base64
import datetime
import json
import os
import time
import urllib.parse
import urllib.request
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from supabase import create_client, Client

from coins import credit_coins

router = APIRouter(prefix="/api/v1", tags=["rewarded_ads"])

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

supabase_admin: Optional[Client] = None
if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
    supabase_admin = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# How many coins one watched ad is worth, and how many times a day any one
# viewer can claim it. Approved by the owner (Nov 2026), same in every country:
# 5 coins per ad (a 30-coin episode is 6 ads), at most 20 ads (100 coins) a day.
# Sized to what a rewarded ad actually earns, so watching stays a win for the
# viewer and for Viyo. Changing them needs the owner's approval.
REWARDED_AD_COINS = 5
DAILY_AD_CAP = 20

# Google's public keys for verifying reward callbacks. Google rotates them;
# don't keep them longer than 24 hours.
VERIFIER_KEYS_URL = "https://www.gstatic.com/admob/reward/verifier-keys.json"
KEY_CACHE_SECONDS = 12 * 3600

# Viyo's rewarded ad unit(s). A callback for any other ad unit is refused.
# Either the full id (ca-app-pub-...) or just the number after the slash.
AD_UNITS = [u.strip() for u in os.environ.get("REWARDED_AD_UNIT_IDS", "ca-app-pub-4006935524883605/4402469307").split(",")
            if u.strip()]


def ssv_on() -> bool:
    return os.environ.get("REWARDED_AD_SSV", "off").strip().lower() in ("on", "true", "1", "required")


def _today_start_utc() -> datetime.datetime:
    now = datetime.datetime.now(datetime.timezone.utc)
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


async def _get_current_user_id_no_guest(authorization: str = Header(None)) -> str:
    from main import get_current_user_id_no_guest

    return await get_current_user_id_no_guest(authorization)


def _admin():
    if supabase_admin is None:
        raise HTTPException(status_code=503, detail="Coin service is not configured.")
    return supabase_admin


def _rpc(name: str, params: dict) -> dict:
    data = _admin().rpc(name, params).execute().data
    if isinstance(data, list):
        data = data[0] if data else {}
    return data or {}


# --------------------------------------------------------------- old path


class RewardedAdClaimResponse(BaseModel):
    coins_earned: int
    claims_today: int
    daily_cap: int


@router.post("/coins/rewarded-ad/claim", response_model=RewardedAdClaimResponse)
async def claim_rewarded_ad(user_id: str = Depends(_get_current_user_id_no_guest)):
    admin = _admin()
    if ssv_on():
        # Coins now come only from Google's confirmation. An app this old can't
        # attach a session, so it is told to update instead of being trusted.
        raise HTTPException(status_code=409, detail={"error": "update_required",
                                                     "message": "Update Viyo to keep earning coins from ads."})

    try:
        today = (
            admin.table("transactions")
            .select("id")
            .eq("user_id", user_id)
            .eq("type", "rewarded_ad")
            .gte("created_at", _today_start_utc().isoformat())
            .execute()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not check today's ad claims: {e}")

    claims_today = len(today.data or [])
    if claims_today >= DAILY_AD_CAP:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "daily_ad_cap_reached",
                "claims_today": claims_today,
                "daily_cap": DAILY_AD_CAP,
            },
        )

    credit_coins(
        admin, user_id, REWARDED_AD_COINS, "rewarded_ad",
        "Watched a rewarded ad",
    )

    return RewardedAdClaimResponse(
        coins_earned=REWARDED_AD_COINS,
        claims_today=claims_today + 1,
        daily_cap=DAILY_AD_CAP,
    )


# --------------------------------------------------------- verified path


class RewardedAdStartResponse(BaseModel):
    verification: str                 # "server": wait for Google; "client": use /claim (switch off)
    session: Optional[str] = None
    user_id: Optional[str] = None
    coins: int
    claims_today: int
    daily_cap: int


@router.post("/coins/rewarded-ad/start", response_model=RewardedAdStartResponse)
async def start_rewarded_ad(user_id: str = Depends(_get_current_user_id_no_guest)):
    """Call before showing an ad. With verification on, returns the session to attach to the ad."""
    _admin()
    if not ssv_on():
        return RewardedAdStartResponse(verification="client", coins=REWARDED_AD_COINS, claims_today=0,
                                       daily_cap=DAILY_AD_CAP)
    try:
        r = _rpc("rewarded_ad_start", {"p_user_id": user_id, "p_daily_cap": DAILY_AD_CAP})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not start an ad session: {e}")
    if not r.get("ok"):
        if r.get("error") == "daily_cap":
            raise HTTPException(status_code=429, detail={"error": "daily_ad_cap_reached",
                                                         "claims_today": r.get("claims_today"),
                                                         "daily_cap": DAILY_AD_CAP})
        raise HTTPException(status_code=429, detail={"error": r.get("error") or "unavailable",
                                                     "message": "Too many ads started at once. Try again later."})
    return RewardedAdStartResponse(verification="server", session=str(r["session"]), user_id=user_id,
                                   coins=REWARDED_AD_COINS, claims_today=int(r.get("claims_today") or 0),
                                   daily_cap=DAILY_AD_CAP)


class RewardedAdStatusResponse(BaseModel):
    credited: bool
    coins: int = 0


@router.get("/coins/rewarded-ad/{session}", response_model=RewardedAdStatusResponse)
async def rewarded_ad_status(session: str, user_id: str = Depends(_get_current_user_id_no_guest)):
    try:
        r = _rpc("rewarded_ad_status", {"p_session": session, "p_user_id": user_id})
    except Exception:
        raise HTTPException(status_code=404, detail="Unknown ad session.")
    if not r.get("found"):
        raise HTTPException(status_code=404, detail="Unknown ad session.")
    return RewardedAdStatusResponse(credited=bool(r.get("credited")), coins=int(r.get("coins") or 0))


# -------------------------------------------- Google's reward callback

_keys: dict = {}
_keys_at = 0.0


def _fetch_keys() -> dict:
    with urllib.request.urlopen(VERIFIER_KEYS_URL, timeout=10) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return {str(k["keyId"]): k["pem"] for k in body.get("keys", []) if k.get("keyId") is not None and k.get("pem")}


def _key(key_id: str):
    """Google's public key for `key_id`, refreshed every 12 h or when an unknown id shows up."""
    global _keys, _keys_at
    if not _keys or time.time() - _keys_at > KEY_CACHE_SECONDS or key_id not in _keys:
        try:
            _keys, _keys_at = _fetch_keys(), time.time()
        except Exception as e:
            print(f"[WARN] AdMob verifier keys unavailable: {e}")
    pem = _keys.get(key_id)
    if not pem:
        return None
    from cryptography.hazmat.primitives.serialization import load_pem_public_key
    return load_pem_public_key(pem.encode("ascii"))


def _b64url(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def verify_callback(query: str) -> Optional[dict]:
    """Checks Google's ECDSA signature over the raw query string (everything
    before `&signature=`). Returns the parsed parameters, or None if forged."""
    marker = "&signature="
    if marker not in query:
        return None
    message = query[:query.index(marker)].encode("utf-8")
    params = dict(urllib.parse.parse_qsl(query, keep_blank_values=True))
    sig, key_id = params.get("signature"), params.get("key_id")
    if not sig or not key_id:
        return None
    key = _key(key_id)
    if key is None:
        return None
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec
    try:
        key.verify(_b64url(sig), message, ec.ECDSA(hashes.SHA256()))
    except (InvalidSignature, ValueError):
        return None
    return params


def _our_ad_unit(ad_unit: str) -> bool:
    ad_unit = (ad_unit or "").strip()
    return any(ad_unit == u or ad_unit == u.rsplit("/", 1)[-1] for u in AD_UNITS)


@router.get("/admob/ssv", response_class=PlainTextResponse)
async def admob_ssv(request: Request):
    """AdMob's server-side verification callback. Public, but only a request
    signed by Google gets past verify_callback. Answers 200 for anything
    handled (including repeats, so Google stops retrying) and 4xx for forged
    or unusable requests."""
    params = verify_callback(request.scope.get("query_string", b"").decode("utf-8"))
    if params is None:
        raise HTTPException(status_code=403, detail="Invalid signature.")
    session, user_id = params.get("custom_data", ""), params.get("user_id", "")
    if not session and not user_id:
        return "ok"                      # the AdMob console's "verify URL" test: signed, carries no user
    if not _our_ad_unit(params.get("ad_unit", "")):
        raise HTTPException(status_code=400, detail="Unknown ad unit.")
    if not session or not user_id or not params.get("transaction_id"):
        raise HTTPException(status_code=400, detail="Missing session, user or transaction.")
    try:
        r = _rpc("rewarded_ad_credit", {"p_session": session, "p_user_id": user_id,
                                        "p_transaction_id": params["transaction_id"],
                                        "p_coins": REWARDED_AD_COINS, "p_daily_cap": DAILY_AD_CAP})
    except HTTPException:
        raise
    except Exception as e:
        # A database hiccup: a 5xx makes Google retry later.
        raise HTTPException(status_code=503, detail=f"Could not record the reward: {e}")
    if r.get("ok") or r.get("error") in ("already_credited", "duplicate", "daily_cap", "expired"):
        return "ok"                      # handled; retrying wouldn't change anything
    raise HTTPException(status_code=400, detail=str(r.get("error") or "rejected"))
