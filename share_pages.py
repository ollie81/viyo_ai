"""
Public preview pages for shared Viyo posts.

www.viyo.site/p/<post id> is rewritten by Vercel to GET /share/p/<post id>
here. The page carries the post's thumbnail, caption and creator as plain
HTML and Open Graph tags, so WhatsApp, X, Facebook, iMessage and search
engines show a proper preview, and a button opens the post in the app
(www.viyo.site/app/?post=<id>).

Only public posts are shown: private or archived posts get a "not
available" page with noindex. The page never includes a video file URL,
so a drama episode that needs unlocking can't be watched from here.
"""
from __future__ import annotations

import html
import os
import uuid
from typing import Optional

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["share_pages"])

SITE = "https://www.viyo.site"
DEFAULT_IMAGE = f"{SITE}/assets/og.png"

_client = None


def _admin():
    global _client
    if _client is None:
        url, key = os.environ.get("SUPABASE_URL", ""), os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
        if not (url and key):
            return None
        from supabase import create_client
        _client = create_client(url, key)
    return _client


def load_public_post(post_id: str) -> Optional[dict]:
    """The post with its creator and series, or None if it isn't public."""
    admin = _admin()
    if admin is None:
        return None
    res = (admin.table("posts")
           .select("id, post_type, caption, media_url, thumbnail_url, is_private, is_archived, "
                   "episode_number, created_at, profiles(username, display_name), series(title)")
           .eq("id", post_id).limit(1).execute())
    rows = res.data or []
    if not rows:
        return None
    post = rows[0]
    if post.get("is_private") or post.get("is_archived"):
        return None
    return post


def _e(s) -> str:
    return html.escape(str(s or ""), quote=True)


def _shorten(text: str, n: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def render_post_page(post: dict) -> str:
    pid = post["id"]
    prof = post.get("profiles") or {}
    series = post.get("series") or {}
    creator = prof.get("display_name") or (f"@{prof['username']}" if prof.get("username") else "A Viyo creator")
    handle = f"@{prof['username']}" if prof.get("username") else ""
    caption = (post.get("caption") or "").strip()
    first_line = caption.split("\n")[0] if caption else ""
    kind = post.get("post_type") or "post"
    episode = ""
    if series.get("title"):
        episode = f"{series['title']}" + (f", episode {post['episode_number']}" if post.get("episode_number") else "")
    headline = episode or _shorten(first_line, 70) or {"video": "A video on Viyo", "photo": "A photo on Viyo"}.get(kind, "A post on Viyo")
    title = f"{headline} | {creator} on Viyo"
    desc = _shorten(caption, 180) if caption else f"Watch {creator}'s {kind} on Viyo."
    image = post.get("thumbnail_url") or (post.get("media_url") if kind == "photo" else None) or DEFAULT_IMAGE
    url = f"{SITE}/p/{pid}"
    open_url = f"{SITE}/app/?post={pid}"
    og_type = "video.other" if kind == "video" else "article"
    cta = "Watch on Viyo" if kind == "video" else "Open in Viyo"
    body_caption = "".join(f"<p>{_e(line)}</p>" for line in caption.split("\n") if line.strip())
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_e(title)}</title>
<meta name="description" content="{_e(desc)}">
<link rel="canonical" href="{url}">
<meta name="theme-color" content="#0B0B1A">
<link rel="icon" type="image/png" href="{SITE}/assets/favicon.png">
<meta property="og:site_name" content="Viyo">
<meta property="og:type" content="{og_type}">
<meta property="og:url" content="{url}">
<meta property="og:title" content="{_e(title)}">
<meta property="og:description" content="{_e(desc)}">
<meta property="og:image" content="{_e(image)}">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="{_e(title)}">
<meta name="twitter:description" content="{_e(desc)}">
<meta name="twitter:image" content="{_e(image)}">
<style>
  :root {{ color-scheme: dark; }}
  body {{ margin: 0; background: #0B0B1A; color: #EDEDF7; font: 16px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif; }}
  .wrap {{ width: min(520px, 100% - 32px); margin: 0 auto; padding: 20px 0 40px; }}
  .top {{ display: flex; align-items: center; justify-content: space-between; margin-bottom: 18px; }}
  .brand {{ color: #00E5FF; font-weight: 800; font-size: 1.5rem; text-decoration: none; }}
  .card {{ background: #13132B; border: 1px solid #1E1E3A; border-radius: 18px; overflow: hidden; }}
  .media {{ display: block; width: 100%; max-height: 70vh; object-fit: cover; background: #17173A; }}
  .inner {{ padding: 16px 18px 18px; }}
  .who {{ font-weight: 700; margin: 0; }}
  .who span {{ color: #A3A3C2; font-weight: 400; }}
  .ep {{ color: #A855F7; font-weight: 700; font-size: .9rem; margin: 4px 0 0; }}
  .cap p {{ margin: 10px 0 0; color: #CFCFE6; overflow-wrap: anywhere; }}
  .btn {{ display: flex; align-items: center; justify-content: center; min-height: 50px; margin-top: 18px; border-radius: 999px; background: #00E5FF; color: #05121A; font-weight: 800; text-decoration: none; }}
  .btn:focus-visible {{ outline: 2px solid #fff; outline-offset: 3px; }}
  .more {{ text-align: center; color: #A3A3C2; font-size: .9rem; margin-top: 16px; }}
  .more a {{ color: #00E5FF; }}
</style>
</head>
<body>
<div class="wrap">
  <div class="top"><a class="brand" href="{SITE}/?home">Viyo</a></div>
  <article class="card">
    <a href="{open_url}"><img class="media" src="{_e(image)}" alt="{_e(headline)}"></a>
    <div class="inner">
      <p class="who">{_e(creator)} <span>{_e(handle) if handle and handle != creator else ""}</span></p>
      {f'<p class="ep">{_e(episode)}</p>' if episode else ""}
      <div class="cap">{body_caption}</div>
      <a class="btn" href="{open_url}">{cta}</a>
    </div>
  </article>
  <p class="more">Viyo is a creator video platform with short dramas and AI tools. <a href="{SITE}/?home">Learn more</a></p>
</div>
</body>
</html>
"""


NOT_FOUND_PAGE = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Post not available | Viyo</title><meta name="robots" content="noindex">
<style>body{{margin:0;background:#0B0B1A;color:#EDEDF7;font:16px/1.55 -apple-system,Segoe UI,Roboto,Arial,sans-serif;text-align:center;padding:80px 20px}}a{{color:#00E5FF}}</style>
</head><body><h1>This post isn't available</h1><p>It may have been deleted or made private.</p>
<p><a href="{SITE}/app/">Open Viyo</a></p></body></html>"""


@router.get("/share/p/{post_id}", response_class=HTMLResponse)
async def share_post_page(post_id: str):
    try:
        uuid.UUID(post_id)
    except ValueError:
        return HTMLResponse(NOT_FOUND_PAGE, status_code=404)
    try:
        post = load_public_post(post_id)
    except Exception as e:
        print(f"[WARN] share page for {post_id}: {e}")
        post = None
    if post is None:
        return HTMLResponse(NOT_FOUND_PAGE, status_code=404, headers={"Cache-Control": "public, max-age=60"})
    return HTMLResponse(render_post_page(post), headers={"Cache-Control": "public, max-age=300"})
