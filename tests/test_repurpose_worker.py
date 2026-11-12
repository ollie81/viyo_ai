"""
End-to-end worker tests: real Postgres (the same SQL functions
production calls), real FFmpeg renders, a local directory standing in
for Supabase Storage. Only the transcriber and the LLM are scripted.
"""
import json
import os
import re
import subprocess
import threading

import pytest

import clip_selection as cs
import clip_render
import repurpose as r
from conftest import connect, make_user, pg_rpc
from repurpose_cleanup import run_maintenance
from repurpose_jobs import JobConfig, JobStore, TransientJobError
from repurpose_pipeline import PipelineDeps
from repurpose_storage import CLIPS_BUCKET, SOURCE_BUCKET, LocalStorage
from repurpose_worker import Worker

STORY = [
    "Nobody tells you that quitting your job is the easy part.",
    "The hard part starts three weeks later when the money stops coming in.",
    "I had eleven hundred dollars left and a landlord who did not care about my dreams.",
    "So I did the one thing every business book tells you not to do.",
    "I called my old boss and asked for freelance work at double my old rate.",
    "He laughed, and then he said yes, because nobody else knew the system.",
    "That one phone call paid my rent for six months.",
    "The lesson is simple, your leverage is whatever only you can do.",
]
OTHER = [
    "Here is what nobody explains about pricing your first product online.",
    "Charge what feels slightly uncomfortable and then raise it after ten sales.",
    "Every time I doubled my price the complaints went down, not up.",
    "Cheap customers ask for the most support and leave the worst reviews.",
    "Expensive customers just want the result and they tell their friends.",
    "So stop competing on price, because there is always someone cheaper.",
    "Compete on being the obvious expert in one tiny corner of the market.",
    "That is the whole strategy, and it works in almost every niche.",
]
FILLER = [f"And this is just some ordinary talk number {i} about the weather today." for i in range(16)]


def script_words(sentences, start=1.0, wps=2.6, pause=0.5):
    words, segments, t = [], [], start
    for s in sentences:
        seg_start = t
        for token in s.split():
            words.append({"word": re.sub(r"[^\w']", "", token), "start": round(t, 3),
                          "end": round(t + 0.32, 3)})
            t += 1 / wps
        segments.append({"start": round(seg_start, 3), "end": round(t, 3), "text": " " + s})
        t += pause
    return words, segments, t


ALL_SENTENCES = STORY + FILLER + OTHER
WORDS, SEGMENTS, SPEECH_END = script_words(ALL_SENTENCES)


@pytest.fixture(scope="module")
def source_video(tmp_path_factory):
    d = tmp_path_factory.mktemp("media")
    path = d / "talk.mp4"
    duration = SPEECH_END + 2
    # Tone only while "speaking", silence in the pauses, like real speech.
    gate = "+".join(f"between(t\\,{seg['start']}\\,{seg['end']})" for seg in SEGMENTS)
    subprocess.run([
        "ffmpeg", "-v", "error", "-y",
        "-f", "lavfi", "-i", f"testsrc2=size=1280x720:rate=30:duration={duration}",
        "-f", "lavfi", "-i", f"aevalsrc=0.2*sin(2*PI*(140+40*sin(2*PI*0.3*t))*t)*({gate}):s=16000:d={duration}",
        "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest", str(path),
    ], check=True)
    silent = d / "silent.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc2=size=640x360:rate=30:duration=20", "-c:v", "libx264",
                    "-preset", "ultrafast", str(silent)], check=True)
    return {"talk": str(path), "silent": str(silent)}


class Scripted:
    """Counts calls so tests can prove what was (not) paid for twice."""

    def __init__(self):
        self.transcribe_calls = 0
        self.llm_calls = 0
        self.transcribe_error = None
        self.weak = False
        self.before_transcribe = None

    def transcribe(self, path, prompt):
        self.transcribe_calls += 1
        if self.before_transcribe:
            self.before_transcribe()
        if self.transcribe_error:
            raise self.transcribe_error
        return {"text": " ".join(ALL_SENTENCES), "language": "english",
                "segments": SEGMENTS, "words": WORDS}

    def llm(self, messages):
        content = messages[-1]["content"]
        prompt = content if isinstance(content, str) else content[0]["text"]
        if "You package short vertical videos" in prompt:
            # Packaging: one cheap call at the end of a job, counted on its own.
            self.package_calls = getattr(self, "package_calls", 0) + 1
            ids = [int(m.group(1)) for m in re.finditer(r"^--- CLIP (\d+)", prompt, re.M)]
            return json.dumps({"clips": [{"id": i, "titles": ["Stop competing on price", "Go viral with pricing"],
                                          "hooks": ["There is always someone cheaper"],
                                          "description": "Why competing on price is a trap.",
                                          "thumbnail": [{"moment": "Speaker explaining pricing", "text": "Stop competing"}],
                                          "tiktok": {"caption": "Stop competing on price.", "hashtags": ["pricing", "fyp"]},
                                          "shorts": {"title": "Stop competing on price", "description": "x",
                                                     "hashtags": ["pricing"]},
                                          "reels": {"caption": "Compete on being the expert.", "hashtags": ["pricing"]}}
                                         for i in ids]})
        self.llm_calls += 1
        if "NO talking" in prompt:
            ids = [int(m.group(1)) for m in re.finditer(r"--- CANDIDATE (\d+)", prompt)]
            return json.dumps({"reviews": [{"id": i, "appeal": 3 if self.weak else 8, "complete": 7,
                                            "what": "Colour bars sweep across the screen.",
                                            "title": "Colour bars sweep across the screen",
                                            "caption": "Mesmerising.", "hashtags": ["satisfying"],
                                            "verdict": "publish"} for i in ids]})
        if "analysing a long video" in prompt:
            return json.dumps({"summary": "A creator tells a story about freelancing and explains pricing.",
                               "format": "talk", "speakers": [], "topics": [], "avoid": []})
        if "strict senior editor" in prompt:
            ids = [int(m.group(1)) for m in re.finditer(r"^--- CLIP (\d+)", prompt, re.M)]
            return json.dumps({"reviews": [{"id": i, "standalone": True, "opening": 8, "ending": 8,
                                            "hook_accurate": None, "visual_ok": None, "visual": "",
                                            "verdict": "publish", "issues": []} for i in ids]})
        ids = {int(m.group(1)): m.group(2) for m in re.finditer(r"^S(\d+) (?:[A-Z] )?\[[^\]]+\] (.*)$", prompt, re.M)}
        story_start = next(i for i, t in ids.items() if t.startswith("Nobody tells you"))
        pricing_start = next(i for i, t in ids.items() if t.startswith("Here is what nobody"))
        good = {"hook": 8, "curiosity": 8, "emotional": 7, "completeness": 8, "standalone": 8, "information": 8}
        weak = {k: 3 for k in good}
        return json.dumps({"candidates": [
            {"start_sentence": story_start, "end_sentence": story_start + 7, "category": "story",
             "scores": weak if self.weak else good, "opens_mid_conversation": False,
             "title": "The phone call that paid my rent", "reason": "Concrete stakes and a twist.",
             "overlay_hook": "", "caption": "Leverage.",
             "hashtags": ["freelance", "career"], "topic": "quitting"},
            {"start_sentence": pricing_start, "end_sentence": pricing_start + 7, "category": "insight",
             "scores": weak if self.weak else dict(good, hook=7), "opens_mid_conversation": False,
             "title": "Stop competing on price", "reason": "Counter-intuitive and actionable.",
             "overlay_hook": "", "caption": "Raise your prices.",
             "hashtags": ["pricing"], "topic": "pricing"},
        ]})


@pytest.fixture
def env(db, tmp_path, source_video):
    config = JobConfig(cost=40, daily_limit=50, max_attempts=3, heartbeat_seconds=1,
                       stale_seconds=60, retry_base_delay_seconds=0, result_retention_days=30,
                       free_trial_daily_cap=0, free_trial_max_seconds=600)
    store = JobStore(pg_rpc(connect()), config)
    storage = LocalStorage(str(tmp_path / "storage"))
    scripted = Scripted()
    deps = PipelineDeps(storage=storage, transcribe_chunk=scripted.transcribe, llm=scripted.llm,
                        max_input_bytes=500 * 1024 * 1024, max_input_seconds=1980, critique=False,
                        analysis=False, review=False, qa=False)
    worker = Worker(store, deps, config, worker_id="test-worker")
    return {"db": db, "store": store, "storage": storage, "scripted": scripted,
            "worker": worker, "media": source_video}


def submit(env, video="talk", balance=100):
    uid = make_user(env["db"], balance)
    path = f"{uid}/upload.mp4"
    env["storage"].put_source(SOURCE_BUCKET, path, env["media"][video])
    env["store"].register_upload(str(uid), SOURCE_BUCKET, path, 1, "video/mp4")
    created = env["store"].create_job(str(uid), source_kind="private_upload", source_bucket=SOURCE_BUCKET,
                                      source_path=path, source_url=None, creative_prompt=None,
                                      client_request_id=f"req-{uid}")
    assert created["ok"], created
    return uid, created["job_id"]


def job_row(env, job_id):
    return env["store"].get_job(job_id)


def balance(env, uid):
    return env["db"].execute("select points_balance from profiles where id=%s", (uid,)).fetchone()[0]


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                          "stream=codec_type,width,height:format=duration", "-of", "json", path],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def local_path(url):
    return url[len("file://"):]


# ---------------------------------------------------------------- happy path

def test_job_runs_end_to_end_and_produces_real_vertical_clips(env):
    uid, job_id = submit(env)
    assert env["worker"].run_once()

    job = job_row(env, job_id)
    assert job["status"] == "succeeded" and job["progress"] == 1, job.get("last_error")
    assert balance(env, uid) == 60, "charged exactly once"

    result = job["result"]
    assert [c["highlight"]["suggested_title"] for c in result["clips"]] == [
        "The phone call that paid my rent", "Stop competing on price"]
    assert result["selection"]["delivered"] == 2
    # Speech is ~2.8 minutes: the density cap allows exactly 2 clips.
    assert result["selection"]["target"] == 2
    # The legacy response model still parses it — old app versions keep working.
    r.RepurposeResponse(**result)

    for clip in result["clips"]:
        info = probe(local_path(clip["processed_video_url"]))
        video = next(s for s in info["streams"] if s["codec_type"] == "video")
        assert (video["width"], video["height"]) == (1080, 1920)
        assert any(s["codec_type"] == "audio" for s in info["streams"])
        h = clip["highlight"]
        assert float(info["format"]["duration"]) == pytest.approx(h["end_time"] - h["start_time"], abs=2.0)
        assert h["transcript"].startswith(("Nobody tells you", "Here is what nobody"))
        assert 0 <= h["score"] <= 100 and h["score_breakdown"]["content"] > 0

    assert [c["status"] for c in job["clips"]] == ["rendered", "rendered"]


def test_full_analysis_pipeline_reviews_renders_and_passes_quality_control(env):
    """Same job with whole-video analysis, editorial review and rendering QC switched on."""
    env["worker"].deps.analysis = env["worker"].deps.review = env["worker"].deps.qa = True
    uid, job_id = submit(env)
    assert env["worker"].run_once()
    job = job_row(env, job_id)
    assert job["status"] == "succeeded", job.get("error_detail")
    sel = job["result"]["selection"]
    assert sel["analysis"]["visual"]["samples"] > 0
    assert sel["analysis"]["video_map"]["summary"]
    assert sel["analysis"]["review"]["reviewed"] is True
    for row in job["clips"]:
        qa = row["render_meta"]["qa"]
        assert qa["checks"]["format"]["ok"] and qa["checks"]["loudness"]["ok"], qa


# ---------------------------------------------------------------- retries & resume

def test_retry_resumes_without_paying_for_transcription_or_selection_again(env, monkeypatch):
    uid, job_id = submit(env)
    real_render = clip_render.render
    attempts = {"n": 0}

    def flaky_render(*a, **kw):
        attempts["n"] += 1
        if attempts["n"] <= 2:
            raise clip_render.RenderError("encoder crashed")
        return real_render(*a, **kw)

    monkeypatch.setattr(clip_render, "render", flaky_render)
    env["worker"].run_once()
    job = job_row(env, job_id)
    assert job["status"] == "queued" and job["error_code"] == "render_failed"
    assert balance(env, uid) == 60

    s = env["scripted"]
    transcribed, llm = s.transcribe_calls, s.llm_calls
    env["worker"].run_once()
    job = job_row(env, job_id)
    assert job["status"] == "succeeded"
    assert s.transcribe_calls == transcribed, "transcript came from the cache"
    assert s.llm_calls == llm, "selection came from the checkpoint"
    assert balance(env, uid) == 60


def test_persistent_transient_failure_exhausts_attempts_then_refunds_once(env):
    uid, job_id = submit(env)
    env["scripted"].transcribe_error = TransientJobError("transcription_unavailable", "down", "503")
    for _ in range(3):
        env["worker"].run_once()
    job = job_row(env, job_id)
    assert job["status"] == "failed" and job["attempts"] == 3
    assert job["refund_amount"] == 40 and balance(env, uid) == 100
    assert env["worker"].run_once() is False
    env["store"].sweep()
    assert balance(env, uid) == 100, "never refunded twice"


# ---------------------------------------------------------------- honest failures

def test_video_without_any_sound_gets_visual_clips(env):
    """No talking and no audio track: clips are chosen from the picture."""
    env["worker"].deps.qa = True
    uid, job_id = submit(env, video="silent")
    assert env["worker"].run_once()
    job = job_row(env, job_id)
    assert job["status"] == "succeeded", job.get("last_error")
    assert env["scripted"].transcribe_calls == 0, "nothing to transcribe"
    assert balance(env, uid) == 60
    result = job["result"]
    assert result["selection"]["mode"] == "visual"
    clip = result["clips"][0]
    assert clip["highlight"]["category"] == "visual"
    assert clip["highlight"]["overlay_hook"] == "Colour bars sweep across the screen"
    info = probe(local_path(clip["processed_video_url"]))
    video = next(s for s in info["streams"] if s["codec_type"] == "video")
    assert (video["width"], video["height"]) == (1080, 1920)
    assert any(s["codec_type"] == "audio" for s in info["streams"]), "silent track added for players"
    r.RepurposeResponse(**result)


def test_silent_video_with_no_good_moment_is_refunded(env):
    env["scripted"].weak = True
    uid, job_id = submit(env, video="silent")
    env["worker"].run_once()
    job = job_row(env, job_id)
    assert job["status"] == "failed" and job["error_code"] == "no_strong_moments"
    assert balance(env, uid) == 100


def test_clips_failing_quality_control_are_dropped_and_refunded(env, monkeypatch):
    """A clip QC rejects goes through the spare path; if none pass, refund."""
    import clip_qa

    class Fail:
        name = "loudness"
        detail = {}

    class Report:
        passed = False
        failures = [Fail()]

        def to_dict(self):
            return {"passed": False, "checks": {"loudness": {"ok": False}}}

    monkeypatch.setattr(clip_qa, "run_qa", lambda *a, **k: Report())
    env["worker"].deps.qa = True
    uid, job_id = submit(env)
    env["worker"].run_once()
    job = job_row(env, job_id)
    assert job["status"] == "failed", job
    assert job["error_code"] == "no_clip_passed_quality_check", job.get("last_error")
    assert "loudness" in (job.get("last_error") or job.get("error_detail") or "")
    assert balance(env, uid) == 100


def test_free_video_longer_than_the_limit_is_refused_before_any_ai_work(env):
    import dataclasses
    env["store"].config = dataclasses.replace(env["store"].config, free_trial_max_seconds=5)
    uid, job_id = submit(env)
    env["db"].execute("update repurpose_jobs set free_trial=true, coins_charged=0 where id=%s", (job_id,))
    env["worker"].run_once()
    job = job_row(env, job_id)
    assert job["status"] == "failed" and job["error_code"] == "free_trial_too_long"
    assert env["scripted"].transcribe_calls == 0, "no transcription paid for"


def test_no_strong_moments_is_reported_and_refunded_never_faked(env):
    env["scripted"].weak = True
    uid, job_id = submit(env)
    env["worker"].run_once()
    job = job_row(env, job_id)
    assert job["status"] == "failed" and job["error_code"] == "no_strong_moments"
    assert "No moment" in job["error"]
    assert job["clips"] == [] and balance(env, uid) == 100


# ---------------------------------------------------------------- ownership

def test_worker_that_loses_its_lease_writes_nothing_more(env):
    uid, job_id = submit(env)
    db = env["db"]

    def steal():
        db.execute("update repurpose_jobs set lease_token = gen_random_uuid() where id=%s", (job_id,))

    env["scripted"].before_transcribe = steal
    env["worker"].run_once()
    job = job_row(env, job_id)
    assert job["status"] == "processing", "still owned by the (hypothetical) new worker"
    assert job["clips"] == [] and job["result"] is None
    assert balance(env, uid) == 60


def test_shutdown_hands_the_job_back_without_counting_the_attempt(env):
    uid, job_id = submit(env)
    worker = env["worker"]
    env["scripted"].before_transcribe = worker.request_shutdown
    worker.run_once()
    job = job_row(env, job_id)
    assert job["status"] == "queued" and job["attempts"] == 0
    assert balance(env, uid) == 60


def test_two_workers_share_the_queue_without_double_processing(env):
    jobs = [submit(env)[1] for _ in range(3)]
    workers = [env["worker"], Worker(env["store"], env["worker"].deps, env["store"].config, "w2")]

    def drain(w):
        with_store = w
        while with_store.run_once():
            pass

    threads = [threading.Thread(target=drain, args=(w,)) for w in workers]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    rows = [job_row(env, j) for j in jobs]
    assert all(j["status"] == "succeeded" and j["attempts"] == 1 for j in rows)


# ---------------------------------------------------------------- cleanup

def test_maintenance_deletes_old_sources_and_expired_results_but_keeps_posted_clips(env):
    uid, job_id = submit(env)
    env["worker"].run_once()
    job = job_row(env, job_id)
    storage, db = env["storage"], env["db"]
    source = storage._path(SOURCE_BUCKET, job["source_path"])
    posted, unposted = job["clips"][0], job["clips"][1]
    db.execute("insert into posts (user_id, media_url) values (%s, %s)", (uid, posted["video_url"]))

    run_maintenance(env["store"], storage)
    assert os.path.exists(source), "source kept within its retention window"

    db.execute("update repurpose_jobs set finished_at = now() - interval '2 days', "
               "expires_at = now() - interval '1 minute' where id=%s", (job_id,))
    report = run_maintenance(env["store"], storage)
    assert report["sources_deleted"] == 1 and not os.path.exists(source)
    assert report["jobs_deleted"] == 1 and job_row(env, job_id) is None
    assert os.path.exists(local_path(posted["video_url"])), "posted clip is the post's media now"
    assert not os.path.exists(local_path(unposted["video_url"]))
    assert report["posted_files_kept"] == 1
    folder = os.path.dirname(local_path(posted["video_url"]))
    assert not [f for f in os.listdir(folder) if "_master" in f or f.endswith(".ass")], \
        "masters and platform caption files go too, even next to a posted clip"


def test_uploads_that_never_became_jobs_are_deleted(env):
    uid = make_user(env["db"])
    path = f"{uid}/abandoned.mp4"
    env["storage"].put_source(SOURCE_BUCKET, path, env["media"]["silent"])
    env["store"].register_upload(str(uid), SOURCE_BUCKET, path, 1, "video/mp4")
    env["db"].execute("update repurpose_uploads set created_at = now() - interval '2 days'")
    report = run_maintenance(env["store"], env["storage"])
    assert report["sources_deleted"] == 1
    assert env["storage"].object_size(SOURCE_BUCKET, path) is None
