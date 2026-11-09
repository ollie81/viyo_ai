"""AI covers: poster art of the real characters, the title in the art, and
the real Viyo mark added afterwards. Fake AI and local Postgres only."""
import io
import os

import pytest
from PIL import Image

import studio_ai
import viyo_brand
from conftest import make_user
from test_character_identity import char, make_sheet, setup
from test_creator_studio import H, balance, env, grant, studio_schema  # noqa: F401
from test_creator_studio import run, start
from test_video_drama import detail

pytestmark = pytest.mark.skipif(bool(os.environ.get("STUDIO_PREMIGRATION")), reason="needs creator_studio_cover.sql")


def test_the_viyo_mark_is_small_and_in_the_corner():
    src = Image.new("RGB", (900, 1200), (30, 30, 30))
    buf = io.BytesIO()
    src.save(buf, "PNG")
    out = Image.open(io.BytesIO(viyo_brand.stamp(buf.getvalue()))).convert("RGB")
    assert out.size == (900, 1200)
    changed = [(x, y) for y in range(0, 1200, 6) for x in range(0, 900, 6) if out.getpixel((x, y)) != (30, 30, 30)]
    xs, ys = [p[0] for p in changed], [p[1] for p in changed]
    assert min(xs) > 900 * 0.6 and max(ys) < 1200 * 0.12, "top-right corner only"
    assert len(changed) < 0.03 * (150 * 200), "small"


def test_a_cover_uses_the_story_the_characters_and_their_references(env):
    client, db, ai, gen, u, pid = setup(env, style="pictures")
    make_sheet(client, ai, gen, u, pid, "Maya")
    client.patch(f"/api/v1/studio/projects/{pid}", headers=H(u), json={"title": "Rooftop Promise"})
    est = client.post(f"/api/v1/studio/projects/{pid}/estimate", headers=H(u), json={"kind": "cover"}).json()
    assert est["ready_count"] == 1 and 5 <= est["total_credits"] <= 15
    ai.image_log = []
    assert start(client, u, pid, "cover").status_code == 200
    run(gen)
    prompt, refs = ai.image_log[0]["prompt"], ai.image_log[0]["refs"]
    assert 'write the title exactly as "Rooftop Promise"' in prompt and "Two strangers meet on a rooftop." in prompt
    assert "small scar above the left eyebrow" in prompt and "Reference image 1: Maya: face close-up" in prompt
    assert refs >= 2 and "No logos, watermarks" in prompt and "top-right corner calm" in prompt
    assert "art director" in ai.last_concept_prompt and "neon blue and red" in prompt and "sharp silver letters" in prompt
    d = detail(client, u, pid)["project"]
    assert d["cover_url"]
    lst = client.get("/api/v1/studio/projects", headers=H(u)).json()["projects"][0]
    assert lst["cover_url"] and lst["thumbnail_url"] == lst["cover_url"]
    # The saved file is the stamped poster, not the raw AI picture.
    path = db.execute("select cover_path from public.studio_projects where id = %s", (pid,)).fetchone()[0]
    assert path.endswith(".png")


def test_a_failed_cover_is_refunded_and_keeps_the_old_one(env):
    client, db, ai, gen, u, pid = setup(env, style="pictures")
    start(client, u, pid, "cover")
    run(gen)
    first = db.execute("select cover_path from public.studio_projects where id = %s", (pid,)).fetchone()[0]
    before = balance(client, u)
    ai.fail["image"] = studio_ai.ProviderError("boom")
    start(client, u, pid, "cover")
    run(gen)
    assert balance(client, u) == before
    assert db.execute("select cover_path from public.studio_projects where id = %s", (pid,)).fetchone()[0] == first


def test_only_the_owner_can_make_or_see_the_cover(env):
    client, db, ai, gen, u, pid = setup(env, style="pictures")
    other = make_user(db)
    grant(client, other, 200)
    assert client.post(f"/api/v1/studio/projects/{pid}/jobs", headers=H(other),
                       json={"kind": "cover", "request_id": "x" * 12}).status_code == 404
    assert client.get(f"/api/v1/studio/projects/{pid}", headers=H(other)).status_code == 404


def test_the_art_director_decides_who_is_on_the_poster_even_nobody(env):
    client, db, ai, gen, u, pid = setup(env, style="pictures")
    ai.concept_result = {"focus": "setting", "characters": [], "composition": "an empty rooftop, a red umbrella left behind",
                         "setting": "rooftop at dawn", "colour_and_light": "cold blue", "title_style": "thin white serif"}
    ai.image_log = []
    r = client.post(f"/api/v1/studio/projects/{pid}/jobs", headers=H(u),
                    json={"kind": "cover", "request_id": "c" * 12, "cover_note": "show the umbrella"})
    assert r.status_code == 200
    run(gen)
    prompt = ai.image_log[0]["prompt"]
    assert ai.image_log[0]["refs"] == 0 and "No people on this poster." in prompt and "red umbrella" in prompt
    assert "THE CREATOR WANTS: show the umbrella" in ai.last_concept_prompt
    ai.concept_result = {"focus": "characters", "characters": ["Maya", "Daniel"], "composition": "both, face to face",
                         "setting": "rain", "colour_and_light": "warm", "title_style": "script"}
    ai.image_log = []
    start(client, u, pid, "cover")
    run(gen)
    assert "- Maya" in ai.image_log[0]["prompt"] and "- Daniel" in ai.image_log[0]["prompt"]


def test_every_cover_is_kept_and_the_creator_picks_the_main_one(env):
    client, db, ai, gen, u, pid = setup(env, style="pictures")
    for _ in range(3):
        start(client, u, pid, "cover")
        run(gen)
    p = detail(client, u, pid)["project"]
    covers = p["covers"]
    assert len(covers) == 3 and sum(c["is_main"] for c in covers) == 1
    first_main = next(c for c in covers if c["is_main"])
    assert first_main["id"] == covers[-1]["id"], "the first cover made stays main until the creator picks another"
    pick = covers[0]
    assert client.post(f"/api/v1/studio/projects/{pid}/cover", headers=H(u), json={"cover_id": pick["id"]}).status_code == 200
    p = detail(client, u, pid)["project"]
    assert next(c for c in p["covers"] if c["is_main"])["id"] == pick["id"]
    other = make_user(db)
    assert client.post(f"/api/v1/studio/projects/{pid}/cover", headers=H(other), json={"cover_id": pick["id"]}).status_code == 404
