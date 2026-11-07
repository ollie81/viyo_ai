"""The first real test of every paid provider Viyo Studio uses, run by an
admin from the app (it runs on the server, where the keys live).

It makes one tiny two-scene production with the same code creators use:
script -> scene list -> automatic voice casting -> portraits -> scene
pictures -> character voices -> the same dialogue scene made three ways
(characters talking inside an Omni clip; a silent Omni clip + saved voice +
sync.so lip sync; a talking portrait with sync.so) -> a finished mini film.

Nothing touches creator data or credits: files go to `livecheck/<id>/` in
the private bucket and the report is a JSON file next to them. A hard
spending cap stops it before any step that could go over.
"""
from __future__ import annotations

import datetime
import json
import os
import secrets
import tempfile
import threading
import time
import traceback
from typing import Optional

import studio_ai as ai_mod
import studio_casting as casting
import studio_pricing as pricing
import studio_render as render

BRIEF = {"title": "The Last Bus", "logline": "A young nurse finishing a night shift meets a quiet stranger at an "
                                            "empty bus stop who seems to know her name.",
         "genre": "Drama", "kind": "short", "target_seconds": 30, "aspect_ratio": "9:16", "language": "English",
         "visual_style": "cinematic, realistic, moody night light"}


class Budget:
    def __init__(self, cap_usd: float):
        self.cap, self.spent = cap_usd, 0.0

    def allows(self, estimate_usd: float) -> bool:
        return self.spent + estimate_usd <= self.cap

    def add(self, usd: float) -> None:
        self.spent += max(0.0, usd)


class LiveCheck:
    def __init__(self, files, ai=None, lipsync=None, cap_usd: float = 5.0, check_id: Optional[str] = None,
                 brief: Optional[dict] = None):
        self.files, self.ai, self.lipsync = files, ai, lipsync
        self.id = check_id or secrets.token_urlsafe(18)
        self.budget = Budget(cap_usd)
        self.brief = dict(brief or BRIEF)
        self.report = {"id": self.id, "status": "running", "started_at": _now(), "cap_usd": cap_usd,
                       "models": {"text": pricing.TEXT_MODEL, "image": pricing.IMAGE_MODEL, "voice": pricing.TTS_MODEL,
                                  "video": pricing.VIDEO_MODEL},
                       "steps": [], "spent_usd": 0.0}
        self._paths: dict[str, str] = {}

    # ------------------------------------------------------------ plumbing

    @property
    def prefix(self) -> str:
        return f"livecheck/{self.id}"

    def _save_report(self) -> None:
        self.report["spent_usd"] = round(self.budget.spent, 4)
        data = json.dumps(self.report, indent=1).encode()
        self._seq = getattr(self, "_seq", 0) + 1
        path = f"{self.prefix}/report-{self._seq:05d}.json"
        self.files.put(path, data, "application/json")
        self.report_path = path

    def _put(self, name: str, data: bytes, content_type: str) -> str:
        path = f"{self.prefix}/{name}"
        self.files.put(path, data, content_type)
        self._paths[name] = path
        return path

    def _step(self, name: str, estimate_usd: float, fn):
        """Runs one paid step if the cap allows; records cost, time, files
        and any error. Returns the step's result or None."""
        step = {"step": name, "status": "running", "estimate_usd": round(estimate_usd, 4)}
        self.report["steps"].append(step)
        if not self.budget.allows(estimate_usd):
            step.update(status="skipped", detail=f"Would go over the ${self.budget.cap:.2f} cap.")
            self._save_report()
            return None
        t0 = time.monotonic()
        try:
            result, cost_usd, detail, files = fn()
            self.budget.add(cost_usd)
            step.update(status="ok", cost_usd=round(cost_usd, 4), seconds=round(time.monotonic() - t0, 1),
                        detail=detail, files=files)
            return result
        except Exception as e:
            step.update(status="failed", seconds=round(time.monotonic() - t0, 1),
                        error=f"{type(e).__name__}: {str(e)[:500]}")
            print(f"[WARN] livecheck {self.id} {name}: {traceback.format_exc()[-800:]}")
            return None
        finally:
            self._save_report()

    # ------------------------------------------------------------ the test

    def run(self) -> dict:
        try:
            self._run()
            self.report["status"] = "finished"
        except Exception as e:
            self.report.update(status="crashed", error=f"{type(e).__name__}: {str(e)[:500]}")
        self.report["finished_at"] = _now()
        self._save_report()
        return self.report

    def _run(self) -> None:
        ai = self.ai or ai_mod.GoogleStudioAI()
        b = self.brief

        est = pricing.estimate_script(int(b.get("target_seconds") or 30), len(b.get("title", "")) + len(b.get("logline", "")))
        script = self._step("1. Story plan + screenplay + script doctor (Gemini text, 3 calls)",
                            est.provider_cost_usd * 1.6, lambda: self._script(ai))
        if not script:
            return
        plan = self._step("2. Scene list + ages (Gemini text)", 0.02, lambda: self._breakdown(ai, script))
        if not plan:
            return
        chars, scenes = plan
        cast = self._cast(chars)
        size = "1K"
        portraits = {}
        for c in cast[:2]:
            img = self._step(f"3. Portrait of {c['name']} (image)", pricing.image_cost(1, 400, size).provider_cost_usd,
                             lambda c=c: self._portrait(ai, c))
            if img:
                portraits[c["name"]] = img
        pictures = []
        for i, s in enumerate(scenes[:2]):
            refs = [portraits[n] for n in portraits if n.lower() in (s.get("visual", "") + str(s.get("dialogue"))).lower()]
            pic = self._step(f"4. Scene {i + 1} picture (image, {len(refs)} portrait refs)",
                             pricing.image_cost(1, 400 + 1300 * len(refs), size).provider_cost_usd,
                             lambda s=s, refs=refs, i=i: self._picture(ai, s, refs, i))
            pictures.append(pic)
        voices = []
        for i, s in enumerate(scenes[:2]):
            if not s["dialogue"]:
                voices.append(None)
                continue
            text = " ".join(d["text"] for d in s["dialogue"])
            v = self._step(f"5. Scene {i + 1} voices (Gemini TTS)", pricing.estimate_voice(text).provider_cost_usd,
                           lambda s=s, i=i: self._voices(ai, s, cast, i))
            voices.append(v)

        talk = next((i for i, s in enumerate(scenes[:2]) if s["dialogue"]), 0)
        s = scenes[talk]
        lines = s["dialogue"]
        import studio_worker as sw
        clip_secs = sw.speech_clip_seconds({"dialogue": lines, "duration_seconds": s.get("duration_seconds") or 5}) or 8
        talking = self._step(f"6. A: characters talk inside an Omni clip ({clip_secs} s)",
                             pricing.estimate_video(clip_secs).provider_cost_usd,
                             lambda: self._omni_talking(ai, s, cast, portraits, pictures[talk], clip_secs))
        voice_secs = (voices[talk] or {}).get("seconds") or clip_secs
        silent_secs = pricing.video_seconds_for(max(float(voice_secs) + 0.5, 3))
        silent = self._step(f"7. B: silent Omni clip for lip sync ({silent_secs} s)",
                            pricing.estimate_video(silent_secs).provider_cost_usd,
                            lambda: self._omni_silent(ai, s, cast, pictures[talk], silent_secs))
        lipsync = self.lipsync
        if lipsync is None and os.environ.get("SYNC_API_KEY"):
            import studio_lipsync
            lipsync = studio_lipsync.SyncLipSync()
        synced = None
        if lipsync is None:
            self.report["steps"].append({"step": "8-9. sync.so lip sync", "status": "skipped",
                                         "detail": "SYNC_API_KEY isn't set on this service."})
        else:
            import studio_lipsync
            if silent and voices[talk]:
                synced = self._step("8. B: saved voice + sync.so lip sync on the silent clip",
                                    studio_lipsync.estimate_lipsync(silent_secs).provider_cost_usd,
                                    lambda: self._sync(lipsync, "video", silent["path"], voices[talk]["path"], "lipsync-video.mp4"))
            first = speaker_of(lines[0]) if lines else None
            if first in portraits and voices[talk]:
                self._step(f"9. C: talking portrait of {first} (sync.so, picture + saved voice)",
                           studio_lipsync.estimate_lipsync(voice_secs + 0.5).provider_cost_usd,
                           lambda: self._sync(lipsync, "image", portraits[first]["path"], voices[talk]["first_line_path"],
                                              "talking-portrait.mp4"))
        self._step("10. Finished mini film (FFmpeg on our server)", 0.0,
                   lambda: self._film(scenes[:2], pictures, voices, talking, talk))

    # ------------------------------------------------------------ steps

    def _script(self, ai):
        develop = getattr(ai, "develop_script", None)
        if develop is None:
            text, cost = ai.write_script(self.brief)
            bible = {}
        else:
            text, bible, cost = develop(self.brief, [])
            self.brief["story_bible"] = bible
            self._put("story-plan.json", json.dumps(bible, indent=1, ensure_ascii=False).encode(), "application/json")
        self._put("screenplay.txt", text.encode(), "text/plain")
        detail = {"chars": len(text), "preview": text[:600], "logline": bible.get("logline"),
                  "draft_scores": bible.get("scores"), "doctor_fixes": bible.get("fixes"),
                  "rewrite_used": bible.get("rewrite_used")}
        files = ["screenplay.txt"] + (["story-plan.json"] if bible else [])
        return text, cost.provider_cost_usd, detail, files

    def _breakdown(self, ai, script):
        data, cost = ai.breakdown(self.brief, script)
        self._put("scene-list.json", json.dumps(data, indent=1).encode(), "application/json")
        chars = [c for c in data.get("characters") or [] if (c.get("name") or "").strip()]
        scenes = []
        for s in (data.get("scenes") or [])[:2]:
            scenes.append({"heading": s.get("heading") or "", "visual": s.get("visual") or "",
                           "duration_seconds": min(12, max(2, float(s.get("duration_seconds") or 5))),
                           "dialogue": [d for d in s.get("dialogue") or [] if (d.get("text") or "").strip()][:4]})
        if not scenes:
            raise ai_mod.ProviderError("no scenes came back")
        detail = {"characters": [{k: c.get(k) for k in ("name", "age_category", "age_years", "age_confidence", "gender",
                                                          "personality")} for c in chars],
                  "scenes": [{"heading": s["heading"], "lines": len(s["dialogue"]),
                              "emotions": [d.get("emotion") for d in s["dialogue"]]} for s in scenes]}
        return (chars, scenes), cost.provider_cost_usd, detail, ["scene-list.json"]

    def _cast(self, chars) -> list[dict]:
        catalog, taken, out = casting.builtin_catalog(), set(), []
        for c in chars[:4]:
            p = casting.build_profile(c["name"], c, c.get("description") or "", self.brief["language"])
            d = casting.cast(p, catalog, taken, f"livecheck:{self.id}:{c['name'].lower()}", c.get("description") or "")
            if d.voice:
                taken.add(d.voice)
            out.append({"name": c["name"], "description": c.get("description") or "", "voice": d.voice,
                        "voice_status": d.status, "voice_style": d.voice_style, "voice_description": d.voice_description,
                        "age_category": p.age_category, "age_years": p.age_years, "gender_presentation": p.gender,
                        "personality": p.personality})
        self.report["casting"] = [{k: c[k] for k in ("name", "age_category", "age_years", "voice", "voice_status",
                                                      "voice_description")} for c in out]
        return out

    def _portrait(self, ai, c):
        prompt = (f"Character reference portrait for a drama. {c['name']}: {c['description']}. Waist-up, facing the "
                  f"camera, face fully visible, neutral studio background, soft even light. Style: "
                  f"{self.brief['visual_style']}. No text.")
        data, mime, cost = ai.image(prompt, [], "9:16", "1K")
        name = f"portrait-{c['name'].lower()}.png"
        path = self._put(name, data, mime)
        return {"data": data, "path": path}, cost.provider_cost_usd, {"bytes": len(data)}, [name]

    def _picture(self, ai, s, refs, i):
        prompt = ai_mod.image_prompt(s["visual"] or s["heading"], self.brief["visual_style"], "9:16", "")
        data, mime, cost = ai.image(prompt, [r["data"] for r in refs][:4], "9:16", "1K")
        name = f"scene-{i + 1}.png"
        path = self._put(name, data, mime)
        return {"data": data, "path": path}, cost.provider_cost_usd, {"bytes": len(data), "refs": len(refs)}, [name]

    def _voices(self, ai, s, cast, i):
        import studio_worker as sw
        by = {c["name"].lower(): c for c in cast}
        total, cost_usd, wavs, used = 0.0, 0.0, [], []
        with tempfile.TemporaryDirectory() as tmp:
            for k, d in enumerate(s["dialogue"]):
                who = speaker_of(d)
                c = by.get((who or "").lower())
                voice = c["voice"] if c and c["voice"] else casting.NARRATOR_VOICE
                if c and not c["voice"]:
                    raise ai_mod.ProviderError(f"{c['name']} has no suitable voice ({c['voice_status']})")
                style = sw._style((c or {}).get("voice_style") or "", d.get("emotion"), d.get("delivery"))
                wav, secs, cost = ai.voice(d["text"], voice, style)
                total += secs
                cost_usd += cost.provider_cost_usd
                p = os.path.join(tmp, f"l{k}.wav")
                open(p, "wb").write(wav)
                wavs.append(p)
                used.append({"speaker": who, "voice": voice, "style": style, "seconds": round(secs, 2)})
                if k == 0:
                    first = self._put(f"scene-{i + 1}-line-1.wav", wav, "audio/wav")
            out = os.path.join(tmp, "scene.m4a")
            dur = render.join_audio(wavs, out)
            path = self._put(f"scene-{i + 1}-voices.m4a", open(out, "rb").read(), "audio/mp4")
        return ({"path": path, "seconds": dur, "first_line_path": first}, cost_usd, {"lines": used, "seconds": round(dur, 2)},
                [f"scene-{i + 1}-voices.m4a", f"scene-{i + 1}-line-1.wav"])

    def _omni_talking(self, ai, s, cast, portraits, picture, seconds):
        import studio_worker as sw
        refs = []
        if picture:
            refs.append(("the scene: its setting, lighting and framing (not a character)", picture["data"]))
        for name, p in list(portraits.items())[:2]:
            refs.append((f"{name} (keep this exact face, hair and clothes)", p["data"]))
        prompt = (f"{s['visual'] or s['heading']}. Style: {self.brief['visual_style']}. "
                  f"{sw.talking_direction(s['dialogue'], cast, {'language': self.brief['language']})} "
                  "No subtitles, captions, text or watermarks. Do not add music.")
        data, cost = ai.video(prompt, picture["data"] if picture else None, "9:16", seconds, refs)
        path = self._put("A-omni-talking.mp4", data, "video/mp4")
        with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
            f.write(data)
            f.flush()
            p = render.probe(f.name)
        info = {"has_sound": bool(p.get("has_audio")), "clip_seconds": p.get("duration")}
        return ({"path": path, "data": data}, cost.provider_cost_usd,
                {"seconds": cost.units.get("video_seconds"), "refs": len(refs), "prompt": prompt[:1200], **info},
                ["A-omni-talking.mp4"])

    def _omni_silent(self, ai, s, cast, picture, seconds):
        prompt = (f"{s['visual'] or s['heading']}. Style: {self.brief['visual_style']}. Natural, subtle motion. "
                  "No dialogue and no speech: nobody talks on camera; their lines are added separately. Characters "
                  "listen, react and gesture naturally, mouths mostly closed. No text, captions or watermarks. Do not add music.")
        data, cost = ai.video(prompt, picture["data"] if picture else None, "9:16", seconds)
        path = self._put("B-omni-silent.mp4", data, "video/mp4")
        return {"path": path}, cost.provider_cost_usd, {"seconds": cost.units.get("video_seconds")}, ["B-omni-silent.mp4"]

    def _sync(self, lipsync, kind, media_path, audio_path, name):
        urls = self.files.sign([media_path, audio_path], 3600)
        data, cost, gid = lipsync.run(kind, urls[media_path], urls[audio_path])
        self._put(name, data, "video/mp4")
        return {"name": name}, cost.provider_cost_usd, {"sync_id": gid, **cost.units}, [name]

    def _film(self, scenes, pictures, voices, talking, talk):
        with tempfile.TemporaryDirectory() as tmp:
            inputs = []
            for i, s in enumerate(scenes):
                pic = pictures[i] if i < len(pictures) else None
                if i == talk and talking:
                    vp = os.path.join(tmp, f"v{i}.mp4")
                    open(vp, "wb").write(talking["data"])
                    inputs.append(render.SceneInput(duration=s["duration_seconds"], video=vp, video_audio=True,
                                                    lines=tuple(d["text"] for d in s["dialogue"])))
                    continue
                if not pic:
                    continue
                ip = os.path.join(tmp, f"i{i}.png")
                open(ip, "wb").write(pic["data"])
                audio, secs = None, None
                if voices[i]:
                    audio = os.path.join(tmp, f"a{i}.m4a")
                    self.files.get_to(voices[i]["path"], audio)
                    secs = voices[i]["seconds"]
                inputs.append(render.SceneInput(duration=s["duration_seconds"], image=ip, audio=audio, audio_seconds=secs,
                                                lines=tuple(d["text"] for d in s["dialogue"])))
            if not inputs:
                raise render.RenderError("nothing to put in the film")
            out = os.path.join(tmp, "film.mp4")
            info = render.render_film(inputs, "9:16", out, tmp, captions=True)
            self._put("mini-film.mp4", open(out, "rb").read(), "video/mp4")
        return {}, 0.0, {k: info.get(k) for k in ("duration", "width", "height", "bytes") if k in info}, ["mini-film.mp4"]


def speaker_of(line: dict) -> Optional[str]:
    import studio_worker as sw
    who = sw.speaker_name(line.get("character") or "")
    return None if not who or sw.is_narrator(who) else who


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def start_in_background(files, cap_usd: float) -> LiveCheck:
    check = LiveCheck(files, cap_usd=cap_usd)
    check._save_report()
    threading.Thread(target=check.run, name=f"livecheck-{check.id}", daemon=True).start()
    return check


def latest_report(files, check_id: str) -> Optional[dict]:
    """The newest saved report for a check (reports are written as new files
    because stored files are never overwritten)."""
    lister = getattr(files, "list", None)
    if lister is None:
        return None
    names = sorted(n for n in lister(f"livecheck/{check_id}") if n.startswith("report-"))
    if not names:
        return None
    return json.loads(files.get(f"livecheck/{check_id}/{names[-1]}"))
