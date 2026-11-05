"""
Creator Studio worker: runs queued Studio jobs, one lane at a time
("generation" = AI provider calls, "render" = ffmpeg), alongside the
Repurposer in the existing worker service.

For each job: claim -> heartbeat while working -> do the work and save
the files -> complete (keep the real cost, return the rest of the hold)
or fail (return everything; retry only when the provider was busy).
"""
from __future__ import annotations

import os
import re
import tempfile
import threading
import time
import traceback
from typing import Optional

import studio_ai as ai_mod
import studio_pricing as pricing
import studio_render as render
from studio_store import Files, Store, new_path

NARRATOR_VOICE = "Charon"


def _now() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


class JobFailed(Exception):
    def __init__(self, code: str, message: str, retry: bool = False, absorbed: Optional[pricing.Cost] = None):
        super().__init__(message)
        self.code, self.message, self.retry, self.absorbed = code, message, retry, absorbed


class StudioWorker:
    def __init__(self, store: Store, files: Files, ai, lane: str, heartbeat_seconds: int = 20,
                 stale_seconds: int = 180, bunny_upload=None):
        self.store, self.files, self.ai, self.lane = store, files, ai, lane
        self.bunny_upload = bunny_upload  # (local_path, title) -> Bunny video id
        self.heartbeat_seconds, self.stale_seconds = heartbeat_seconds, stale_seconds
        self._stop = threading.Event()

    # ------------------------------------------------------------ loop

    def run_forever(self, idle_seconds: float = 3.0) -> None:
        while not self._stop.is_set():
            try:
                worked = self.run_once()
            except Exception as e:
                # e.g. the Studio SQL hasn't been run yet: back off quietly.
                print(f"[ERROR] studio {self.lane} loop: {str(e)[:200]}")
                self._stop.wait(60)
                continue
            if not worked:
                self._stop.wait(idle_seconds)

    def stop(self) -> None:
        self._stop.set()

    def run_once(self) -> bool:
        job = self.store.rpc("studio_claim_job", {"p_lane": self.lane, "p_stale_seconds": self.stale_seconds})
        if not job:
            return False
        self.run_job(job)
        return True

    def run_job(self, job: dict) -> None:
        stop = threading.Event()

        def beat():
            while not stop.wait(self.heartbeat_seconds):
                try:
                    if not self.store.rpc("studio_heartbeat", {"p_job_id": job["id"], "p_lease": job["lease"],
                                                               "p_provider_job_id": None}):
                        return
                except Exception:
                    pass

        t = threading.Thread(target=beat, daemon=True)
        t.start()
        try:
            output, cost = self.handle(job)
            credits = min(cost.credits, int(job["estimated_credits"])) if job["kind"] not in ("script", "breakdown") \
                else cost.credits
            self.store.rpc("studio_complete_job", {"p_job_id": job["id"], "p_lease": job["lease"], "p_output": output,
                                                   "p_credits": credits, "p_usage": cost.usage()})
        except JobFailed as e:
            self._fail(job, e.code, e.message, e.retry, e.absorbed)
        except ai_mod.ProviderBusy as e:
            self._fail(job, "provider_busy", "The AI service is busy. Trying again shortly.", True)
            print(f"[WARN] studio job {job['id']} busy: {e}")
        except ai_mod.PolicyBlocked as e:
            self._fail(job, "content_blocked", str(e) or "The AI refused this content. Try rewording it.", False)
        except Exception as e:
            traceback.print_exc()
            self._fail(job, "failed", "Something went wrong making this. Your credits were returned.", False)
        finally:
            stop.set()

    def _fail(self, job, code, message, retry, absorbed=None):
        self.store.rpc("studio_fail_job", {"p_job_id": job["id"], "p_lease": job["lease"], "p_code": code,
                                           "p_message": message[:400], "p_retry": retry,
                                           "p_absorbed_usage": absorbed.usage() if absorbed else None})

    # ------------------------------------------------------------ handlers

    def handle(self, job: dict) -> tuple[dict, pricing.Cost]:
        project = self.store.one("studio_projects", {"id": job["project_id"]})
        if not project or project["owner_id"] != job["owner_id"]:
            raise JobFailed("not_found", "This production no longer exists.")
        fn = getattr(self, f"do_{job['kind']}", None)
        if fn is None:
            raise JobFailed("unknown_job", "Unknown job type.")
        return fn(job, project)

    def _save(self, job, project, kind: str, ext: str, data: bytes, content_type: str, **asset) -> str:
        path = new_path(job["owner_id"], project["id"], kind, ext)
        self.files.put(path, data, content_type)
        self.store.insert("studio_assets", [{"owner_id": job["owner_id"], "project_id": project["id"],
                                             "job_id": job["id"], "kind": kind, "path": path,
                                             "bytes": len(data), **asset}])
        return path

    def do_script(self, job, project):
        brief = {k: project[k] for k in ("title", "logline", "genre", "kind", "aspect_ratio", "target_seconds",
                                         "language", "visual_style")}
        if not ai_mod.moderate_text(f"{project['title']}\n{project['logline']}"):
            raise JobFailed("content_blocked", "This idea can't be made on Viyo. Please change it.")
        text, cost = self.ai.write_script(brief)
        self.store.update("studio_projects", {"id": project["id"]}, {"script": text, "updated_at": _now()})
        return {"chars": len(text)}, cost

    def do_breakdown(self, job, project):
        script = project["script"].strip()
        if not script:
            raise JobFailed("no_script", "Write or paste a script first.")
        if not ai_mod.moderate_text(script):
            raise JobFailed("content_blocked", "This script can't be made on Viyo. Please change it.")
        brief = {k: project.get(k) for k in ("title", "kind", "aspect_ratio", "visual_style", "target_seconds", "language")}
        brief["language"] = brief.get("language") or "English"
        data, cost = self.ai.breakdown(brief, script)
        chars = data.get("characters") or []
        scenes = data.get("scenes") or []
        if not scenes:
            raise JobFailed("empty", "The AI couldn't find any scenes in this script.", absorbed=cost)

        existing = {c["name"].strip().lower(): c for c in self.store.select("studio_characters", {"project_id": project["id"]})}
        used = {c["voice"] for c in existing.values() if c.get("voice")}
        new_chars, describe = [], []
        for i, c in enumerate(chars[:20]):
            name = (c.get("name") or "").strip()[:60]
            if not name or is_narrator(name) or name.lower() in {x["name"].lower() for x in new_chars}:
                continue
            row = existing.get(name.lower())
            if row:
                # An existing character keeps its voice and look. Only an
                # empty description is filled in.
                if not row.get("description") and c.get("description"):
                    describe.append({"id": row["id"], "description": c["description"][:1200]})
                continue
            voice = pick_voice(c.get("gender", ""), used)   # assigned once, here, and kept
            used.add(voice)
            new_chars.append({"name": name, "description": (c.get("description") or "")[:1200],
                              "voice": voice, "position": i})

        incoming = []
        for s in scenes[:80]:
            dialogue = [{"character": (d.get("character") or "")[:60], "text": (d.get("text") or "")[:600]}
                        for d in (s.get("dialogue") or []) if (d.get("text") or "").strip()][:12]
            incoming.append({"heading": (s.get("heading") or "")[:200], "visual": (s.get("visual") or "")[:2000],
                             "dialogue": dialogue,
                             "duration_seconds": min(12, max(2, float(s.get("duration_seconds") or 5)))})
        current = self.store.select("studio_scenes", {"project_id": project["id"]})
        plan = plan_breakdown(current, incoming)
        try:
            r = self.store.rpc("studio_apply_breakdown", {
                "p_project": project["id"], "p_owner": project["owner_id"], "p_job": job["id"],
                "p_characters": new_chars, "p_describe": describe, "p_keep": plan["keep"],
                "p_insert": plan["insert"], "p_archive": plan["archive"]})
        except Exception as e:
            msg = str(e).lower()
            if "studio_apply_breakdown" in msg and ("could not find" in msg or "does not exist" in msg):
                print(f"[ERROR] studio breakdown {job['id']}: creator_studio_breakdown.sql not run: {str(e)[:200]}")
                raise JobFailed("not_ready", "Scene lists are being updated on Viyo. Please try again later. "
                                             "Your credits were returned.", absorbed=cost)
            raise
        if not r or not r.get("ok"):
            raise JobFailed("not_found", "This production no longer exists.", absorbed=cost)
        return {"characters": len(chars), "scenes": len(incoming), "kept": len(plan["keep"]),
                "added": len(plan["insert"]), "set_aside": len(plan["archive"]),
                "kept_pictures": plan["kept_pictures"], "kept_voice": plan["kept_voice"]}, cost

    def do_portrait(self, job, project):
        ch = self.store.one("studio_characters", {"id": job["character_id"]})
        if not ch or ch["project_id"] != project["id"]:
            raise JobFailed("not_found", "That character no longer exists.")
        prompt = (f"Character reference portrait for a {project['genre'] or 'drama'} production. {ch['name']}: "
                  f"{ch['description']}. Waist-up, facing the camera, face fully visible, neutral studio background, "
                  f"soft even light. Style: {project['visual_style'] or 'cinematic, realistic'}. No text.")
        data, mime, cost = self.ai.image(prompt, [], "9:16", _size(project))
        ext = "png" if "png" in mime else "jpg"
        path = self._save(job, project, "portrait", ext, data, mime, character_id=ch["id"])
        self.store.update("studio_characters", {"id": ch["id"]}, {"reference_path": path})
        return {"path": path}, cost

    def do_image(self, job, project):
        scene = self._scene(job, project)
        cast = self.store.select("studio_characters", {"project_id": project["id"]})
        in_scene = characters_in_scene(scene, cast)
        refs = []
        for c in in_scene:
            if c.get("reference_path"):
                try:
                    refs.append(self.files.get(c["reference_path"]))
                except Exception:
                    pass
        notes = " ".join(f"{c['name']}: {c['description']}." for c in in_scene)
        prompt = ai_mod.image_prompt(scene["visual"] or scene["heading"], project["visual_style"],
                                     project["aspect_ratio"], notes)
        data, mime, cost = self.ai.image(prompt, refs[:4], project["aspect_ratio"], _size(project))
        ext = "png" if "png" in mime else "jpg"
        path = self._save(job, project, "image", ext, data, mime, scene_id=scene["id"])
        self.store.update("studio_scenes", {"id": scene["id"]}, {"image_path": path, "updated_at": _now()})
        return {"path": path}, cost

    def do_voice(self, job, project):
        scene = self._scene(job, project)
        lines = [d for d in scene["dialogue"] if (d.get("text") or "").strip()]
        if not lines:
            raise JobFailed("no_dialogue", "This scene has no lines to voice.")
        cast = {c["name"].strip().lower(): c for c in self.store.select("studio_characters", {"project_id": project["id"]})}
        # Work out every line's voice before spending anything. A character
        # keeps its saved voice; a bad or missing one stops the job (credits
        # returned) instead of quietly using another voice.
        voices = []
        for line in lines:
            speaker = speaker_name(line.get("character") or "")
            if not speaker or is_narrator(speaker):
                voices.append(NARRATOR_VOICE)
                continue
            ch = cast.get(speaker.lower())
            if ch is None:
                print(f"[WARN] studio voice job {job['id']}: no character named {speaker!r} in project {project['id']}")
                raise JobFailed("unknown_character",
                                f"\"{speaker}\" speaks in this scene but isn't in your cast. Add them to the cast "
                                "or change the line. Your credits were returned.")
            voice = (ch.get("voice") or "").strip()
            if not voice:
                # Never had a voice (made before voices were assigned):
                # give it one now and save it, so it stays the same from here on.
                voice = pick_voice("", {c.get("voice") for c in cast.values() if c.get("voice")})
                self.store.update("studio_characters", {"id": ch["id"]}, {"voice": voice})
                ch["voice"] = voice
            elif voice not in ai_mod.VOICE_NAMES:
                print(f"[WARN] studio voice job {job['id']}: character {ch['id']} has unknown voice {voice!r}")
                raise JobFailed("invalid_voice",
                                f"{ch['name']}'s voice (\"{voice}\") isn't available. Choose a new voice for "
                                f"{ch['name']} in the cast, then try again. Your credits were returned.")
            voices.append(voice)
        total = pricing.voice_cost(0)
        seconds_total = 0.0
        with tempfile.TemporaryDirectory() as tmp:
            wavs = []
            for i, (line, voice) in enumerate(zip(lines, voices)):
                wav, secs, cost = self.ai.voice(line["text"], voice)
                seconds_total += secs
                p = os.path.join(tmp, f"l{i}.wav")
                with open(p, "wb") as f:
                    f.write(wav)
                wavs.append(p)
            out = os.path.join(tmp, "scene.m4a")
            dur = render.join_audio(wavs, out)
            with open(out, "rb") as f:
                data = f.read()
        total = pricing.voice_cost(seconds_total)
        path = self._save(job, project, "audio", "m4a", data, "audio/mp4", scene_id=scene["id"], duration_seconds=round(dur, 2))
        self.store.update("studio_scenes", {"id": scene["id"]}, {"audio_path": path, "audio_seconds": round(dur, 2),
                                                                 "updated_at": _now()})
        return {"path": path, "seconds": round(dur, 2)}, total

    def do_video(self, job, project):
        scene = self._scene(job, project)
        frame = None
        if scene.get("image_path"):
            try:
                frame = self.files.get(scene["image_path"])
            except Exception:
                frame = None
        cast = self.store.select("studio_characters", {"project_id": project["id"]})
        notes = " ".join(f"{c['name']}: {c['description']}." for c in characters_in_scene(scene, cast))
        prompt = (f"{scene['visual'] or scene['heading']}. {notes} Style: {project['visual_style'] or 'cinematic'}. "
                  "Natural, subtle motion. No text, captions or watermarks. Do not add music.")
        seconds = pricing.video_seconds_for(scene.get("duration_seconds") or 8.0)
        data, cost = self.ai.video(prompt, frame, project["aspect_ratio"], seconds)
        path = self._save(job, project, "video", "mp4", data, "video/mp4", scene_id=scene["id"],
                          duration_seconds=cost.units.get("video_seconds"))
        self.store.update("studio_scenes", {"id": scene["id"]}, {"video_path": path, "mode": "video", "updated_at": _now()})
        return {"path": path}, cost

    def do_render(self, job, project):
        scenes = sorted(self.store.select("studio_scenes", {"project_id": project["id"]}), key=lambda s: s["position"])
        if not scenes:
            raise JobFailed("no_scenes", "Add scenes before rendering.")
        missing = [s["position"] + 1 for s in scenes if not (s.get("image_path") or (s["mode"] == "video" and s.get("video_path")))]
        if missing:
            raise JobFailed("not_ready", f"Scene {missing[0]} has no picture yet. Generate it first.")
        with tempfile.TemporaryDirectory() as tmp:
            inputs = []
            for s in scenes:
                def fetch(path, name):
                    if not path:
                        return None
                    local = os.path.join(tmp, name)
                    self.files.get_to(path, local)
                    return local
                n = s["position"]
                use_video = s["mode"] == "video" and s.get("video_path")
                inputs.append(render.SceneInput(
                    duration=float(s["duration_seconds"]),
                    image=None if use_video else fetch(s["image_path"], f"img{n}"),
                    video=fetch(s["video_path"], f"vid{n}.mp4") if use_video else None,
                    audio=fetch(s.get("audio_path"), f"aud{n}.m4a"),
                    audio_seconds=float(s["audio_seconds"]) if s.get("audio_seconds") else None,
                    lines=tuple(d["text"] for d in s["dialogue"] if d.get("text")),
                    sfx=tuple(x for x in (s.get("sfx") or []) if isinstance(x, str))))
            music = None
            if project.get("music_path"):
                music = os.path.join(tmp, "music.m4a")
                self.files.get_to(project["music_path"], music)
            out = os.path.join(tmp, "film.mp4")
            info = render.render_film(inputs, project["aspect_ratio"], out, tmp, captions=bool(project.get("captions", True)),
                                      music=music, music_volume=float(project.get("music_volume") or 0.25))
            prev = self.store.select("studio_renders", {"project_id": project["id"]}, order="version", desc=True, limit=1)
            version = (prev[0]["version"] + 1) if prev else 1
            path, bunny_id = None, None
            if self.bunny_upload is not None:
                # Films go to Bunny Stream: no file-size limit, built for video.
                try:
                    bunny_id = self.bunny_upload(out, f"{project['title']} v{version}")
                except Exception as e:
                    raise JobFailed("storage_busy", "Couldn't save the film. Trying again.", retry=True) from e
            else:
                path = f"{job['owner_id']}/{project['id']}/render/v{version:03d}.mp4"
                self.files.put_file(path, out, "video/mp4")
        self.store.insert("studio_assets", [{"owner_id": job["owner_id"], "project_id": project["id"], "job_id": job["id"],
                                             "kind": "render", "path": path or f"bunny:{bunny_id}", "bytes": info["bytes"],
                                             "duration_seconds": info["duration"], "width": info["width"], "height": info["height"]}])
        self.store.insert("studio_renders", [{"project_id": project["id"], "owner_id": job["owner_id"], "job_id": job["id"],
                                              "version": version, "path": path, "bunny_video_id": bunny_id,
                                              "aspect_ratio": project["aspect_ratio"],
                                              "width": info["width"], "height": info["height"],
                                              "duration_seconds": info["duration"], "bytes": info["bytes"],
                                              "sha256": info["sha256"]}])
        return {"path": path, "bunny_video_id": bunny_id, "version": version, **info}, \
            pricing.render_cost(info["duration"], info["bytes"])

    def _scene(self, job, project):
        scene = self.store.one("studio_scenes", {"id": job["scene_id"]})
        if not scene or scene["project_id"] != project["id"]:
            raise JobFailed("not_found", "That scene no longer exists.")
        return scene


def _size(project: dict) -> str:
    return "2K" if project.get("image_quality") == "high" else "1K"


def characters_in_scene(scene: dict, cast: list[dict]) -> list[dict]:
    text = " ".join([scene.get("visual") or "", scene.get("heading") or ""] +
                    [d.get("character") or "" for d in scene.get("dialogue") or []]).lower()
    return [c for c in cast if c.get("name") and re.search(rf"\b{re.escape(c['name'].lower())}\b", text)]


# ------------------------------------------------------------ breakdown plan

NARRATOR_NAMES = {"narrator", "narration", "voice over", "voiceover", "vo"}


def speaker_name(raw: str) -> str:
    """'DANIEL (V.O.)' -> 'DANIEL'. Screenplay extensions aren't part of the name."""
    return re.sub(r"\s*\((?:v\.?\s*o\.?|o\.?\s*s\.?|o\.?\s*c\.?|cont'?d|contd)\)\s*", " ", raw, flags=re.I).strip()


def is_narrator(name: str) -> bool:
    return re.sub(r"[^a-z ]", "", (name or "").lower()).strip() in NARRATOR_NAMES


def _norm(text: str) -> str:
    return re.sub(r"[\W_]+", " ", (text or "").lower()).strip()


def _lines_key(dialogue) -> tuple:
    return tuple((_norm(speaker_name(d.get("character") or "")), _norm(d.get("text") or ""))
                 for d in (dialogue or []) if (d.get("text") or "").strip())


def _has_work(s: dict) -> bool:
    return bool(s.get("image_path") or s.get("video_path") or s.get("audio_path"))


def plan_breakdown(current: list[dict], incoming: list[dict]) -> dict:
    """Matches a new scene list against the scenes a project already has.

    - Same place and the same lines: the existing scene is kept as it is,
      with its picture, video and voice. Only its order changes (and, if it
      has no picture yet, its description and length are updated).
    - Same place, different lines: a new scene is made that reuses the old
      picture/video (the place hasn't changed) but not the voice (the words
      have). The old scene is set aside in the archive with everything.
    - Anything else: new scenes are added; old ones are set aside.
    Nothing is deleted outright."""
    pool = sorted(current, key=lambda s: s.get("position") or 0)
    used: set = set()
    keep, insert = [], []
    kept_pictures = kept_voice = 0
    matched: dict[int, dict] = {}
    for i, s in enumerate(incoming):          # pass 1: same place, same lines
        for old in pool:
            if old["id"] in used:
                continue
            if _norm(old.get("heading")) == _norm(s["heading"]) and _lines_key(old.get("dialogue")) == _lines_key(s["dialogue"]):
                used.add(old["id"])
                matched[i] = old
                break
    for i, s in enumerate(incoming):
        old = matched.get(i)
        if old is not None:
            row = {"id": old["id"], "position": i}
            if not (old.get("image_path") or old.get("video_path")):
                row.update(visual=s["visual"], duration_seconds=s["duration_seconds"])
            keep.append(row)
            kept_pictures += bool(old.get("image_path") or old.get("video_path"))
            kept_voice += bool(old.get("audio_path"))
            continue
        row = dict(s, position=i, mode="still", image_path=None, video_path=None)
        for old in pool:                       # pass 2: same place, new lines
            if old["id"] in used or not _norm(s["heading"]) or _norm(old.get("heading")) != _norm(s["heading"]):
                continue
            if old.get("image_path") or old.get("video_path"):
                used.add(old["id"])
                row.update(image_path=old.get("image_path"), video_path=old.get("video_path"),
                           mode=old.get("mode") or "still")
                kept_pictures += 1
            break
        insert.append(row)
    archive = [s["id"] for s in pool if s["id"] not in {k["id"] for k in keep}]
    return {"keep": keep, "insert": insert, "archive": archive,
            "kept_pictures": kept_pictures, "kept_voice": kept_voice}


def pick_voice(gender: str, used: set) -> str:
    g = (gender or "").lower()
    pool = [v[0] for v in ai_mod.VOICES if g in ("male", "female") and v[1] == g] or [v[0] for v in ai_mod.VOICES]
    for v in pool:
        if v not in used:
            return v
    return pool[len(used) % len(pool)]


def start_background(store: Store, files: Files, ai, bunny_upload=None) -> list[StudioWorker]:
    """Starts one thread per lane. Called from the worker service."""
    workers = [StudioWorker(store, files, ai, "generation"),
               StudioWorker(store, files, ai, "render", bunny_upload=bunny_upload)]
    for w in workers:
        threading.Thread(target=w.run_forever, name=f"studio-{w.lane}", daemon=True).start()
    return workers


class LazyGoogleAI:
    """Builds the Gemini client on first use, so a missing key fails (and
    refunds) individual jobs instead of stopping the worker."""

    def __init__(self):
        self._ai = None

    def __getattr__(self, name):
        if self._ai is None:
            self._ai = ai_mod.GoogleStudioAI()
        return getattr(self._ai, name)


def start_production() -> list[StudioWorker]:
    if os.environ.get("STUDIO_WORKER", "on").lower() == "off":
        print("[INFO] Creator Studio worker off (STUDIO_WORKER=off)")
        return []
    from studio_store import SupabaseFiles, SupabaseStore
    from supabase import create_client
    admin = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])
    import bunny_stream
    upload = bunny_stream.upload_file if bunny_stream._configured() else None
    workers = start_background(SupabaseStore(admin), SupabaseFiles(admin), LazyGoogleAI(), bunny_upload=upload)
    print(f"[INFO] Studio films are saved to {'Bunny Stream' if upload else 'Supabase Storage'}")
    print("[INFO] Creator Studio worker started (generation + render lanes)")
    return workers
