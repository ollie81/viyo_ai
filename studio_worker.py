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
import studio_casting as casting
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
                 stale_seconds: int = 180, bunny_upload=None, lipsync=None):
        self.store, self.files, self.ai, self.lane = store, files, ai, lane
        self.lipsync = lipsync  # studio_lipsync.SyncLipSync when switched on
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
            print(f"[INFO] studio job {job['id']} ({job['kind']}) failed: {e.code}: {e.message}")
            self._fail(job, e.code, e.message, e.retry, e.absorbed)
        except ai_mod.ProviderBusy as e:
            self._fail(job, "provider_busy", "The AI service is busy. Trying again shortly.", True)
            print(f"[WARN] studio job {job['id']} busy: {e}")
        except ai_mod.PolicyBlocked as e:
            print(f"[INFO] studio job {job['id']} ({job['kind']}) blocked by the AI: {e}")
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

    def _rollback(self) -> None:
        if hasattr(self.store, "conn"):
            try:
                self.store.conn.rollback()
            except Exception:
                pass

    def _optional_update(self, table, filters, fields) -> None:
        """For columns added by creator_studio_voices.sql: if that hasn't been
        run yet, the recording still counts; only the extra detail is lost."""
        try:
            self.store.update(table, filters, fields)
        except Exception as e:
            print(f"[WARN] studio: couldn't save {list(fields)} on {table} (run creator_studio_voices.sql?): {str(e)[:160]}")
            if hasattr(self.store, "conn"):
                try:
                    self.store.conn.rollback()
                except Exception:
                    pass

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
        develop = getattr(self.ai, "develop_script", None)
        if develop is None:
            text, cost = self.ai.write_script(brief)
            bible = None
        else:
            text, bible, cost = develop(brief, self.earlier_episodes(project))
        self.store.update("studio_projects", {"id": project["id"]}, {"script": text, "updated_at": _now()})
        if bible:
            self._optional_update("studio_projects", {"id": project["id"]}, {"story_bible": bible})
        return {"chars": len(text), "scores": (bible or {}).get("scores"), "rewritten": (bible or {}).get("rewrite_used")}, cost

    def earlier_episodes(self, project) -> list[dict]:
        """The story bibles of the episodes before this one in its series,
        oldest first, so the writers' room keeps the story continuous."""
        sid = project.get("series_id")
        if not sid:
            return []
        eps = [e for e in self.store.select("studio_projects", {"series_id": sid, "owner_id": project["owner_id"]})
               if e["id"] != project["id"] and (e.get("created_at") or "") <= (project.get("created_at") or "~")]
        out = []
        for e in sorted(eps, key=lambda e: e.get("created_at") or ""):
            sb = e.get("story_bible") or {}
            if sb:
                out.append({**sb, "title": e.get("title") or sb.get("title")})
        return out

    def do_breakdown(self, job, project):
        script = project["script"].strip()
        if not script:
            raise JobFailed("no_script", "Write or paste a script first.")
        if not ai_mod.moderate_text(script):
            raise JobFailed("content_blocked", "This script can't be made on Viyo. Please change it.")
        brief = {k: project.get(k) for k in ("title", "kind", "aspect_ratio", "visual_style", "target_seconds", "language",
                                             "story_bible")}
        brief["language"] = brief.get("language") or "English"
        data, cost = self.ai.breakdown(brief, script)
        chars = data.get("characters") or []
        scenes = data.get("scenes") or []
        if not scenes:
            raise JobFailed("empty", "The AI couldn't find any scenes in this script.", absorbed=cost)

        existing = {c["name"].strip().lower(): c for c in self.store.select("studio_characters", {"project_id": project["id"]})}
        earlier = self.series_cast(project)
        used = {c["voice"] for c in list(existing.values()) + list(earlier.values()) if c.get("voice")}
        catalog = self.catalog()
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
            prev = earlier.get(name.lower())
            if prev and prev.get("voice"):
                # The same person earlier in this series: the same voice
                # identity, profile, look and identity id. Never re-cast.
                new_chars.append(dict(series_copy(prev), position=i,
                                      description=prev.get("description") or (c.get("description") or "")[:1200]))
                continue
            desc = (c.get("description") or "")[:1200]
            profile = casting.build_profile(name, c, desc, brief["language"])
            seed = casting.identity_seed(project["owner_id"], project.get("series_id") or project["id"], name)
            d = casting.cast(profile, catalog, used, seed, desc)
            if d.voice:
                used.add(d.voice)
            print(f"[INFO] studio cast {project['id']} {name!r}: {profile.age_category}/{profile.age_confidence} "
                  f"{profile.gender} -> {d.status} {d.voice or '-'}")
            new_chars.append({"name": name, "description": desc, "voice": d.voice, "voice_provider": d.provider,
                              "voice_style": d.voice_style, "voice_language": d.language, "position": i,
                              **profile_fields(profile), "voice_category": d.voice_category,
                              "voice_description": d.voice_description, "voice_status": d.status,
                              "voice_assignment_version": 1 if d.voice else 0})

        incoming = []
        for s in scenes[:80]:
            dialogue = [{"character": (d.get("character") or "")[:60], "text": (d.get("text") or "")[:600],
                         **({"emotion": d["emotion"].strip()[:40]} if (d.get("emotion") or "").strip() else {}),
                         **({"delivery": d["delivery"].strip()[:80]} if (d.get("delivery") or "").strip() else {})}
                        for d in (s.get("dialogue") or []) if (d.get("text") or "").strip()][:12]
            incoming.append({"heading": (s.get("heading") or "")[:200], "visual": (s.get("visual") or "")[:2000],
                             "dialogue": dialogue,
                             "duration_seconds": min(12, max(2, float(s.get("duration_seconds") or 5)))})
        current = self.store.select("studio_scenes", {"project_id": project["id"]})
        plan = plan_breakdown(current, incoming, "video" if project.get("production_style") == "video" else "still")
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

    def catalog(self) -> list:
        return load_catalog(self.store)

    def ensure_voice(self, ch: dict, project: dict, cast_rows: list[dict]) -> str:
        """The character's saved voice, or, if it has never had one (made
        before casting existed, or waiting for a child voice), a voice cast
        now and saved. Raises JobFailed if no suitable voice exists."""
        voice = (ch.get("voice") or "").strip()
        catalog = self.catalog()
        if voice:
            if not any(v.voice_id == voice and v.status == "available" for v in catalog):
                print(f"[WARN] studio: character {ch['id']} has unknown or unavailable voice {voice!r}")
                raise JobFailed("invalid_voice",
                                f"{ch['name']}'s voice (\"{voice}\") isn't available. Choose a new voice for "
                                f"{ch['name']} in the cast, then try again. Your credits were returned.")
            return voice
        profile = profile_from_row(ch, project)
        taken = {c.get("voice") for c in cast_rows if c.get("voice")} | set(self.series_voices(project))
        seed = casting.identity_seed(project["owner_id"], project.get("series_id") or project["id"], ch["name"])
        d = casting.cast(profile, catalog, taken, seed, ch.get("description") or "")
        fields = {**profile_fields(profile), "voice_category": d.voice_category, "voice_status": d.status,
                  "voice_language": d.language}
        if not d.voice:
            self._optional_update("studio_characters", {"id": ch["id"]}, fields)
            print(f"[WARN] studio: no voice for {ch['id']} ({profile.age_category}, {profile.language}): {d.reason}")
            code = "unsupported_language" if d.status == "unsupported_language" else "needs_voice"
            who = casting.AGE_LABEL.get(profile.age_category, "")
            msg = (f"{ch['name']}'s lines can't be voiced in {profile.language} yet: the voice model doesn't support it."
                   if code == "unsupported_language" else
                   f"{ch['name']} is a {who}, and Viyo doesn't have a suitable {who} voice yet. Viyo won't use an adult "
                   f"voice instead. Their lines can be voiced once one is added. Your credits were returned.")
            raise JobFailed(code, msg)
        fields.update(voice=d.voice, voice_provider=d.provider, voice_description=d.voice_description,
                      voice_assignment_version=int(ch.get("voice_assignment_version") or 0) + 1)
        if not (ch.get("voice_style") or "").strip():
            fields["voice_style"] = d.voice_style
        self.store.update("studio_characters", {"id": ch["id"]}, {"voice": d.voice})
        self._optional_update("studio_characters", {"id": ch["id"]}, {k: v for k, v in fields.items() if k != "voice"})
        try:
            self.store.insert("studio_voice_changes", [{"character_id": ch["id"], "project_id": ch["project_id"],
                                                       "owner_id": ch["owner_id"], "old_voice": "", "new_voice": d.voice,
                                                       "reason": "auto_cast"}])
        except Exception as e:
            self._rollback()
            print(f"[WARN] studio: cast history not saved for {ch['id']}: {str(e)[:120]}")
        ch.update(fields)
        print(f"[INFO] studio: cast {ch['id']} {ch['name']!r} -> {d.voice} ({d.status})")
        return d.voice

    def series_voices(self, project) -> list[str]:
        return [c.get("voice") for c in self.series_cast(project).values() if c.get("voice")]

    def series_cast(self, project) -> dict:
        """Characters from the other episodes of this project's series, by
        lower-case name, the most recently edited episode winning."""
        sid = project.get("series_id")
        if not sid:
            return {}
        episodes = [e for e in self.store.select("studio_projects", {"series_id": sid, "owner_id": project["owner_id"]})
                    if e["id"] != project["id"]]
        out: dict = {}
        for e in sorted(episodes, key=lambda e: e.get("updated_at") or ""):
            for c in self.store.select("studio_characters", {"project_id": e["id"]}):
                out[c["name"].strip().lower()] = c
        return out

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
        if scene_speech(scene, project) == "video_model":
            raise JobFailed("spoken_in_video", "The characters speak this scene's lines inside the video, so no separate "
                                               "voice recording is needed. Your credits were returned.")
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
                voices.append((NARRATOR_VOICE, None, _style("", line.get("emotion"), line.get("delivery"))))
                continue
            ch = cast.get(speaker.lower())
            if ch is None:
                print(f"[WARN] studio voice job {job['id']}: no character named {speaker!r} in project {project['id']}")
                raise JobFailed("unknown_character",
                                f"\"{speaker}\" speaks in this scene but isn't in your cast. Add them to the cast "
                                "or change the line. Your credits were returned.")
            voice = self.ensure_voice(ch, project, list(cast.values()))
            voices.append((voice, ch, _style(ch.get("voice_style") or "", line.get("emotion"), line.get("delivery"))))
        total = pricing.voice_cost(0)
        seconds_total = 0.0
        with tempfile.TemporaryDirectory() as tmp:
            wavs = []
            for i, (line, (voice, _ch, style)) in enumerate(zip(lines, voices)):
                wav, secs, cost = self.ai.voice(line["text"], voice, style)
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
        used = {("narrator" if ch is None else ch["id"]): v for v, ch, _s in voices}
        self._optional_update("studio_scenes", {"id": scene["id"]}, {"audio_voices": used})
        if scene.get("lipsync_path"):   # new words: the old lip sync no longer matches (its file is kept)
            self._optional_update("studio_scenes", {"id": scene["id"]}, {"lipsync_path": None})
        for _v, ch, _s in voices:
            if ch is not None and not ch.get("voice_locked_at") and "voice_locked_at" in ch:
                ch["voice_locked_at"] = _now()
                self._optional_update("studio_characters", {"id": ch["id"]}, {"voice_locked_at": ch["voice_locked_at"]})
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
        in_scene = characters_in_scene(scene, cast)
        notes = " ".join(f"{c['name']}: {c['description']}." for c in in_scene)
        lines = [d for d in scene.get("dialogue") or [] if (d.get("text") or "").strip()]
        style = f"Style: {project['visual_style'] or 'cinematic'}. "
        references = None
        if lines and scene_speech(scene, project) == "video_model":
            # The characters talk inside the clip. Omni takes no audio, so
            # each character gets their saved voice description, word for
            # word every time, plus their portrait so the face stays the same.
            seconds = speech_clip_seconds(scene)
            if seconds is None:
                raise JobFailed("too_long", "This scene has too much dialogue for one video clip (15 seconds at most). "
                                            "Split it into two scenes. Your credits were returned.")
            prompt = (f"{scene['visual'] or scene['heading']}. {notes} {style}"
                      f"{talking_direction(lines, cast, project)} "
                      "No subtitles, captions, text or watermarks. Do not add music.")
            references = self._references(scene, in_scene, frame)
        else:
            has_lines = bool(lines)
            # Saved voices speak this scene: the video model must not talk,
            # or a second, different voice appears (its sound isn't used).
            speech = ("No dialogue and no speech: nobody talks on camera; their lines are added separately. "
                      "Characters listen, react and gesture naturally, mouths mostly closed. "
                      if has_lines else "No dialogue. ")
            prompt = (f"{scene['visual'] or scene['heading']}. {notes} {style}"
                      f"Natural, subtle motion. {speech}No text, captions or watermarks. Do not add music.")
            seconds = pricing.video_seconds_for(scene.get("duration_seconds") or 8.0)
            if has_lines and scene_speech(scene, project) == "lipsync":
                need = max(float(scene.get("duration_seconds") or 5), float(scene.get("audio_seconds") or 0) + 0.5,
                           speech_seconds(lines) if not scene.get("audio_seconds") else 0)
                if need > MAX_CLIP:
                    raise JobFailed("too_long", "This scene has too much dialogue for one video clip (15 seconds at "
                                                "most). Split it into two scenes. Your credits were returned.")
                seconds = pricing.video_seconds_for(need)
        data, cost = self.ai.video(prompt, frame, project["aspect_ratio"], seconds, references) if references \
            else self.ai.video(prompt, frame, project["aspect_ratio"], seconds)
        path = self._save(job, project, "video", "mp4", data, "video/mp4", scene_id=scene["id"],
                          duration_seconds=cost.units.get("video_seconds"))
        self.store.update("studio_scenes", {"id": scene["id"]}, {"video_path": path, "mode": "video", "updated_at": _now()})
        if scene.get("lipsync_path"):   # a new clip needs new lip sync (the old file is kept)
            self._optional_update("studio_scenes", {"id": scene["id"]}, {"lipsync_path": None})
        return {"path": path}, cost

    def _references(self, scene, in_scene, frame) -> list:
        """Labelled images for Omni: the scene picture (setting and framing)
        and up to 2 speaking characters' portraits (Google: 3 per character,
        2 characters, 6 images at most)."""
        refs = []
        if frame:
            refs.append(("the scene: its setting, lighting and framing (not a character)", frame))
        speakers = [speaker_name(d.get("character") or "").lower() for d in scene.get("dialogue") or []]
        ordered = sorted(in_scene, key=lambda c: (c["name"].lower() not in speakers, c.get("position") or 0))
        for c in ordered[:2]:
            if c.get("reference_path"):
                try:
                    refs.append((f"{c['name']} (keep this exact face, hair and clothes)", self.files.get(c["reference_path"])))
                except Exception:
                    pass
        return refs

    def do_lipsync(self, job, project):
        """The scene's clip (or picture) with the mouth moving to the
        characters' saved voices. The original clip and voice are kept."""
        scene = self._scene(job, project)
        if self.lipsync is None:
            raise JobFailed("not_ready", "Lip sync isn't switched on yet. Your credits were returned.")
        if scene_speech(scene, project) != "lipsync":
            raise JobFailed("not_lipsync", "This scene doesn't use lip sync. Your credits were returned.")
        if not scene.get("audio_path"):
            raise JobFailed("no_voice", "Record this scene's voices first. Your credits were returned.")
        media = scene.get("video_path") if scene.get("mode") == "video" and scene.get("video_path") else scene.get("image_path")
        if not media:
            raise JobFailed("no_picture", "Make this scene's video or picture first. Your credits were returned.")
        kind = "video" if media == scene.get("video_path") else "image"
        urls = self.files.sign([media, scene["audio_path"]], 3600)

        def started(gid):
            # Saved with the job, so a restarted worker carries on with this
            # sync.so job instead of paying for a second one.
            self.store.rpc("studio_heartbeat", {"p_job_id": job["id"], "p_lease": job["lease"], "p_provider_job_id": gid})

        data, cost, gid = self.lipsync.run(kind, urls[media], urls[scene["audio_path"]],
                                           resume_id=job.get("provider_job_id"), on_started=started)
        path = self._save(job, project, "video", "mp4", data, "video/mp4", scene_id=scene["id"],
                          duration_seconds=cost.units.get("seconds"))
        self.store.update("studio_scenes", {"id": scene["id"]}, {"lipsync_path": path, "updated_at": _now()})
        return {"path": path, "sync_id": gid}, cost

    def do_render(self, job, project):
        scenes = sorted(self.store.select("studio_scenes", {"project_id": project["id"]}), key=lambda s: s["position"])
        if not scenes:
            raise JobFailed("no_scenes", "Add scenes before rendering.")
        missing = [s["position"] + 1 for s in scenes if not (s.get("image_path") or (s["mode"] == "video" and s.get("video_path")))]
        if missing:
            raise JobFailed("not_ready", f"Scene {missing[0]} has no picture yet. Generate it first.")
        unspoken = [s["position"] + 1 for s in scenes if talks_in_video(s, project) and not s.get("video_path")]
        unsynced = [s["position"] + 1 for s in scenes if needs_lipsync(s, project) and not s.get("lipsync_path")]
        if unsynced:
            raise JobFailed("not_ready", f"Scene {unsynced[0]}'s lip sync isn't made yet. Make it first, so the "
                                         "lips match the voices.")
        if unspoken:
            raise JobFailed("not_ready", f"Scene {unspoken[0]}'s talking video isn't made yet. Make it first, so its "
                                         "dialogue isn't missing from the film.")
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
                synced = s.get("lipsync_path") if needs_lipsync(s, project) else None
                use_video = synced or (s["mode"] == "video" and s.get("video_path"))
                spoken_in_clip = bool(synced) or (bool(use_video) and talks_in_video(s, project))
                inputs.append(render.SceneInput(
                    duration=float(s["duration_seconds"]),
                    image=None if use_video else fetch(s["image_path"], f"img{n}"),
                    video=fetch(synced or s["video_path"], f"vid{n}.mp4") if use_video else None,
                    audio=None if spoken_in_clip else fetch(s.get("audio_path"), f"aud{n}.m4a"),
                    audio_seconds=float(s["audio_seconds"]) if s.get("audio_seconds") else None,
                    lines=tuple(d["text"] for d in s["dialogue"] if d.get("text")),
                    video_audio=spoken_in_clip or not any((d.get("text") or "").strip() for d in s["dialogue"]),
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


def _style(voice_style: str, emotion, delivery=None) -> str:
    """The performance for one line: the character's fixed short note plus
    this line's emotion and delivery. This only changes how the line is
    said; the speaker is the saved voice id. Google advises keeping it
    short for a steadier voice."""
    parts = []
    for x in (voice_style or "", emotion or "", delivery or ""):
        x = x.strip()
        if x and x.lower() not in [p.lower() for p in parts]:
            parts.append(x)
    return ", ".join(parts)[:120]


def is_narrator(name: str) -> bool:
    return re.sub(r"[^a-z ]", "", (name or "").lower()).strip() in NARRATOR_NAMES


def _norm(text: str) -> str:
    return re.sub(r"[\W_]+", " ", (text or "").lower()).strip()


def _lines_key(dialogue) -> tuple:
    return tuple((_norm(speaker_name(d.get("character") or "")), _norm(d.get("text") or ""))
                 for d in (dialogue or []) if (d.get("text") or "").strip())


def _has_work(s: dict) -> bool:
    return bool(s.get("image_path") or s.get("video_path") or s.get("audio_path"))


def plan_breakdown(current: list[dict], incoming: list[dict], default_mode: str = "still") -> dict:
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
        row = dict(s, position=i, mode=default_mode, image_path=None, video_path=None)
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


# ------------------------------------------------------------ talking video

MAX_CLIP = pricing.VIDEO_MAX_SECONDS


def scene_speech(scene: dict, project: dict) -> str:
    """Who speaks a scene's lines: 'video_model' (the characters talk in the
    clip) or 'saved_voice' (their permanent voices over the scene)."""
    if scene.get("speech") in ("video_model", "saved_voice", "lipsync"):
        return scene["speech"]
    return "video_model" if project.get("production_style") == "video" and scene.get("mode") == "video" else "saved_voice"


def needs_lipsync(scene: dict, project: dict) -> bool:
    return (scene_speech(scene, project) == "lipsync"
            and any((d.get("text") or "").strip() for d in scene.get("dialogue") or []))


def talks_in_video(scene: dict, project: dict) -> bool:
    return (scene_speech(scene, project) == "video_model"
            and any((d.get("text") or "").strip() for d in scene.get("dialogue") or []))


def speech_seconds(lines: list) -> float:
    """A conservative estimate of how long the lines take to say."""
    words = sum(len((d.get("text") or "").split()) for d in lines)
    return words / 2.4 + 0.5 * len(lines) + 1.0


def speech_clip_seconds(scene: dict):
    """The clip length for a talking scene, or None if the lines can't fit
    in one clip."""
    lines = [d for d in scene.get("dialogue") or [] if (d.get("text") or "").strip()]
    need = max(float(scene.get("duration_seconds") or 5), speech_seconds(lines))
    if need > MAX_CLIP:
        return None
    return pricing.video_seconds_for(need)


def talking_direction(lines: list, cast: list, project: dict) -> str:
    """The dialogue for Omni: who says what, in which voice and how. The
    voice description is the character's saved one, never rewritten."""
    by_name = {c["name"].strip().lower(): c for c in cast}
    lang = project.get("language") or "English"
    out = [f"The characters speak on camera with natural lip movement, in {lang}, saying exactly these words, in this order:"]
    for d in lines:
        who = speaker_name(d.get("character") or "")
        how = ", ".join(x for x in ((d.get("emotion") or "").strip(), (d.get("delivery") or "").strip()) if x)
        if not who or is_narrator(who):
            out.append(f'NARRATOR (off-screen voice-over, nobody on screen speaks it){f" ({how})" if how else ""}: "{d["text"].strip()}"')
            continue
        ch = by_name.get(who.lower())
        voice = ""
        if ch is not None:
            voice = (ch.get("voice_description") or "").strip() or \
                casting.describe_voice(profile_from_row(ch, project), None, lang, ch.get("description") or "")
        out.append(f'{who.upper()} (voice: {voice.rstrip(".")}{f"; delivery: {how}" if how else ""}): "{d["text"].strip()}"')
    out.append("Keep each character's voice exactly as described. Nobody else speaks.")
    return " ".join(out)


PROFILE_KEYS = ("age_category", "age_years", "age_confidence", "gender_presentation", "personality")
SERIES_KEYS = ("name", "voice", "voice_provider", "voice_style", "voice_language", "reference_path", "age_category",
               "age_years", "age_confidence", "gender_presentation", "personality", "voice_category",
               "voice_description", "voice_status", "voice_assignment_version")


def profile_fields(p: "casting.Profile") -> dict:
    return {"age_category": p.age_category, "age_years": p.age_years, "age_confidence": p.age_confidence,
            "gender_presentation": p.gender, "personality": p.personality}


def profile_from_row(ch: dict, project: dict) -> "casting.Profile":
    """The saved profile, or one worked out from the description for
    characters made before casting existed or added by hand."""
    if ch.get("age_category"):
        return casting.Profile(name=ch["name"], age_category=ch["age_category"], age_years=ch.get("age_years"),
                               age_confidence=ch.get("age_confidence") or "medium",
                               gender=ch.get("gender_presentation") or casting.infer_gender(ch.get("description") or ""),
                               personality=ch.get("personality") or "",
                               language=project.get("language") or "English")
    return casting.build_profile(ch["name"], None, ch.get("description") or "", project.get("language") or "English")


def series_copy(prev: dict) -> dict:
    """A character carried into another episode: same identity and voice."""
    out = {k: prev.get(k) for k in SERIES_KEYS if k in prev}
    out["identity_id"] = prev.get("identity_id") or prev["id"]
    return out


def load_catalog(store) -> list:
    try:
        rows = store.select("studio_voice_catalog", {})
    except Exception:
        if hasattr(store, "conn"):
            try:
                store.conn.rollback()
            except Exception:
                pass
        rows = []
    return casting.catalog_from_rows(rows) if rows else casting.builtin_catalog()


def pick_voice(gender: str, used: set) -> str:
    g = (gender or "").lower()
    pool = [v[0] for v in ai_mod.VOICES if g in ("male", "female") and v[1] == g] or [v[0] for v in ai_mod.VOICES]
    for v in pool:
        if v not in used:
            return v
    return pool[len(used) % len(pool)]


def start_background(store: Store, files: Files, ai, bunny_upload=None) -> list[StudioWorker]:
    """Starts one thread per lane. Called from the worker service."""
    lipsync = None
    try:
        import studio_lipsync
        if studio_lipsync.enabled():
            lipsync = studio_lipsync.SyncLipSync()
    except Exception as e:
        print(f"[WARN] studio: lip sync not started: {str(e)[:120]}")
    workers = [StudioWorker(store, files, ai, "generation", lipsync=lipsync),
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
