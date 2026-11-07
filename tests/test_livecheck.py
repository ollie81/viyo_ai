"""The admin live provider test, run with fakes (no real spending)."""
import time

import studio_ai
import studio_lipsync
import studio_livecheck as lc
from studio_store import LocalFiles
from test_creator_studio import env, studio_schema  # noqa: F401


def test_full_live_check_with_fakes(tmp_path):
    files = LocalFiles(str(tmp_path))
    check = lc.LiveCheck(files, ai=studio_ai.FakeStudioAI(), lipsync=studio_lipsync.FakeLipSync(), cap_usd=5.0)
    report = check.run()
    assert report["status"] == "finished", report
    steps = {s["step"].split(" ")[0]: s for s in report["steps"]}
    assert all(s["status"] == "ok" for s in report["steps"]), [(s["step"], s.get("error")) for s in report["steps"]]
    for n in ("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.", "10."):
        assert n in steps, n
    assert [c["voice_status"] for c in report["casting"]] == ["cast", "cast"]
    assert report["casting"][0]["voice"] != report["casting"][1]["voice"]
    assert steps["6."]["detail"]["refs"] == 3 and "speak on camera" in steps["6."]["detail"]["prompt"]
    assert steps["6."]["detail"]["has_sound"] is True
    names = set(files.list(f"livecheck/{check.id}"))
    for f in ("screenplay.txt", "A-omni-talking.mp4", "B-omni-silent.mp4", "lipsync-video.mp4",
              "talking-portrait.mp4", "mini-film.mp4"):
        assert f in names, f
    latest = lc.latest_report(files, check.id)
    assert latest["status"] == "finished" and latest["spent_usd"] == report["spent_usd"]


def test_the_cap_stops_steps_that_would_go_over(tmp_path):
    files = LocalFiles(str(tmp_path))
    report = lc.LiveCheck(files, ai=studio_ai.FakeStudioAI(), lipsync=studio_lipsync.FakeLipSync(), cap_usd=0.20).run()
    statuses = {s["step"].split(" ")[0]: s["status"] for s in report["steps"]}
    assert statuses["1."] == "ok" and statuses["6."] == "skipped" and statuses["7."] == "skipped"
    assert report["spent_usd"] <= 0.20


def test_a_failing_provider_is_reported_not_hidden(tmp_path):
    ai = studio_ai.FakeStudioAI(fail={"video": studio_ai.ProviderError("Omni said no")})
    report = lc.LiveCheck(LocalFiles(str(tmp_path)), ai=ai, lipsync=studio_lipsync.FakeLipSync()).run()
    a = next(s for s in report["steps"] if s["step"].startswith("6."))
    assert a["status"] == "failed" and "Omni said no" in a["error"]
    assert next(s for s in report["steps"] if s["step"].startswith("10."))["status"] == "ok", "the film still renders"


def test_endpoints_need_the_admin_key_and_return_a_results_link(env, monkeypatch):
    client, db, store, files, ai, gen, ren = env
    monkeypatch.setattr(studio_ai, "GoogleStudioAI", lambda: studio_ai.FakeStudioAI())
    monkeypatch.delenv("SYNC_API_KEY", raising=False)
    assert client.post("/api/v1/studio/admin/live-check", json={"confirm_spend": True}).status_code in (401, 403)
    admin = {"X-Admin-Key": "admin-secret"}
    dry = client.post("/api/v1/studio/admin/live-check", headers=admin, json={}).json()
    assert dry["would_spend_up_to_usd"] == 5.0 and "id" not in dry
    assert client.post("/api/v1/studio/admin/live-check", headers=admin,
                       json={"confirm_spend": True, "cap_usd": 50}).status_code == 422, "never more than $5"
    r = client.post("/api/v1/studio/admin/live-check", headers=admin, json={"confirm_spend": True}).json()
    for _ in range(300):
        rep = client.get(r["results_path"]).json()
        if rep["status"] != "running":
            break
        time.sleep(0.1)
    assert rep["status"] == "finished", rep
    assert rep["file_urls"]["mini-film.mp4"].startswith("https://files.test/livecheck/")
    assert any(s["status"] == "skipped" and "SYNC_API_KEY" in s.get("detail", "") for s in rep["steps"])
    assert client.get("/api/v1/studio/live-check/not-a-real-id-xxxxxxxxxxxx").status_code == 404
    assert client.get("/api/v1/studio/live-check/..%2F..%2Fetc").status_code == 404
