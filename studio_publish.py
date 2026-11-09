"""Publishing a Studio production to Viyo.

A finished render becomes an episode post in the creator's own Viyo drama
(the same `series` and `posts` rows the upload flow creates), so it shows
on the drama's series page in the Dramas tab and in the main feed, under
the creator's account, and goes live at once. The production's main cover
becomes the drama's poster and the episode's thumbnail. The creator sets
the price: free, or a coin price per episode (the usual first-episodes-free
rule of Viyo Dramas still applies to paid series).

`Publisher` talks to Supabase with the service role; tests swap in a fake.
"""
from __future__ import annotations

import os
import uuid
from typing import Optional

COVERS_BUCKET = "studio-images"     # public, like the old Studio's covers


def content_type_for(kind: str) -> str:
    return {"film": "movie", "short": "short_film"}.get(kind or "", "short_drama")


def orientation_for(aspect_ratio: str) -> str:
    return "vertical" if aspect_ratio == "9:16" else "landscape"


class Publisher:
    """The live implementation (Supabase service role, Bunny, push)."""

    def __init__(self):
        from supabase import create_client
        self.db = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

    def public_image(self, data: bytes, mime: str, owner_id: str) -> str:
        path = f"studio-covers/{owner_id}/{uuid.uuid4().hex}.png"
        self.db.storage.from_(COVERS_BUCKET).upload(path, data, file_options={"content-type": mime})
        return self.db.storage.from_(COVERS_BUCKET).get_public_url(path)

    def series(self, series_id: str) -> Optional[dict]:
        rows = self.db.table("series").select("*").eq("id", series_id).limit(1).execute().data
        return rows[0] if rows else None

    def create_series(self, row: dict) -> dict:
        extra = {k: row.pop(k) for k in ("content_type", "orientation") if k in row}
        try:
            return self.db.table("series").insert({**row, **extra}).execute().data[0]
        except Exception as e:
            if not extra or "column" not in str(e):
                raise
            return self.db.table("series").insert(row).execute().data[0]

    def update_series(self, series_id: str, fields: dict) -> None:
        self.db.table("series").update(fields).eq("id", series_id).execute()

    def last_episode(self, series_id: str) -> int:
        rows = (self.db.table("posts").select("episode_number").eq("series_id", series_id)
                .order("episode_number", desc=True).limit(1).execute().data)
        return int((rows[0] if rows else {}).get("episode_number") or 0)

    def insert_post(self, row: dict) -> dict:
        try:
            return self.db.table("posts").insert(row).execute().data[0]
        except Exception as e:
            if "is_free" not in row or "column" not in str(e):
                raise
            return self.db.table("posts").insert({k: v for k, v in row.items() if k != "is_free"}).execute().data[0]

    def after_post(self, post: dict, owner_id: str, series_id: str, episode: int) -> None:
        """The posting reward and the new-episode notification, best-effort."""
        try:
            self.db.rpc("award_post_creation", {"p_user_id": owner_id, "p_post_id": post["id"],
                                                "p_post_type": "video"}).execute()
        except Exception as e:
            print(f"[WARN] studio publish: award_post_creation failed for {post['id']}: {str(e)[:120]}")
        try:
            from episode_notify import notify_episode_followers
            notify_episode_followers(post["id"], owner_id, series_id, episode)
        except Exception as e:
            print(f"[WARN] studio publish: follower notification failed for {post['id']}: {str(e)[:120]}")

    def bunny_playback(self, video_id: str) -> str:
        import bunny_stream
        return bunny_stream._playback_url(video_id)

    def bunny_upload(self, local_path: str, title: str) -> str:
        import bunny_stream
        return bunny_stream.upload_file(local_path, title)
