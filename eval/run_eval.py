"""
Repeatable real-video evaluation for the AI Repurposer.

    python eval/run_eval.py VIDEO [VIDEO ...] --out eval/out [--transcriber openai|local]

Runs the production pipeline stages on each video, stage by stage, and
writes report.json + report.md with timings, peak memory, transcript
structure, delivery analysis, the selected clips (scores, breakdowns,
titles, transcripts) and technical checks on every rendered file.

Transcription:
  openai  whisper-1, exactly as production (needs OPENAI_API_KEY)
  local   faster-whisper on CPU, adapted to whisper-1's output shape —
          a stand-in for exercising everything downstream on real speech
          when no key is available. Its accuracy is NOT whisper-1's.
Selection and rendering need OPENAI_API_KEY (the selection LLM). Without
it they are reported as blocked — never simulated.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import resource
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("ALLOW_INSECURE_AUTH", "true")

import clip_selection as cs  # noqa: E402
from audio_features import build_baseline, decode_pcm, delivery_score, frame_features, window_stats  # noqa: E402
from transcription import transcribe_file  # noqa: E402


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


class Timer:
    def __init__(self):
        self.stages = {}

    def run(self, name, fn, *a, **kw):
        t0, c0 = time.time(), time.process_time()
        out = fn(*a, **kw)
        self.stages[name] = {"wall_s": round(time.time() - t0, 1),
                             "cpu_s": round(time.process_time() - c0, 1),
                             "peak_rss_mb_so_far": round(peak_rss_mb())}
        return out


def local_transcriber(model_size: str):
    from faster_whisper import WhisperModel
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    def run(chunk_path, prompt):
        segments, info = model.transcribe(chunk_path, word_timestamps=True, initial_prompt=prompt,
                                          vad_filter=False)
        segs, words = [], []
        for s in segments:
            segs.append({"start": s.start, "end": s.end, "text": s.text})
            for w in s.words or []:
                # whisper-1's word list carries no punctuation; match it.
                words.append({"word": re.sub(r"[^\w'’-]", "", w.word.strip()), "start": w.start, "end": w.end})
        return {"text": " ".join(s["text"].strip() for s in segs), "language": info.language,
                "segments": segs, "words": [w for w in words if w["word"]]}
    return run


def openai_transcriber():
    from openai import OpenAI
    from transcription import openai_chunk_transcriber
    return openai_chunk_transcriber(OpenAI(timeout=300, max_retries=2))


def probe(path: str) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                          "stream=codec_type,width,height,r_frame_rate:format=duration", "-of", "json", path],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def evaluate(path: str, args) -> dict:
    report = {"video": os.path.basename(path), "transcriber": args.transcriber}
    timer = Timer()
    info = probe(path)
    duration = float(info["format"]["duration"])
    report["duration_min"] = round(duration / 60, 1)
    report["source"] = {s["codec_type"]: {k: s.get(k) for k in ("width", "height", "r_frame_rate")}
                        for s in info["streams"]}

    transcriber = local_transcriber(args.local_model) if args.transcriber == "local" else openai_transcriber()
    with tempfile.TemporaryDirectory() as work:
        transcript = timer.run("transcribe", transcribe_file, path, duration, work, transcriber)

    words, segments = transcript["words"], transcript["segments"]
    punct = cs.attach_punctuation(words, segments)
    sentences = timer.run("sentences", cs.build_sentences, punct)
    speech = cs.speech_seconds(punct)
    terminal = sum(1 for s in sentences if s.terminal)
    lengths = sorted(s.end - s.start for s in sentences)
    report["transcript"] = {
        "language": transcript.get("language"), "words": len(words), "segments": len(segments),
        "sentences": len(sentences), "terminal_sentence_pct": round(100 * terminal / max(1, len(sentences)), 1),
        "median_sentence_s": round(lengths[len(lengths) // 2], 1) if lengths else 0,
        "max_sentence_s": round(lengths[-1], 1) if lengths else 0,
        "speech_min": round(speech / 60, 1), "clip_target": cs.target_clip_count(speech),
        "sample": [s.text for s in sentences[10:14]],
    }

    samples = timer.run("audio_decode", decode_pcm, path)
    ff = timer.run("audio_frames", frame_features, samples)
    del samples
    baseline = timer.run("audio_baseline", build_baseline, ff, duration, [w["start"] for w in words])

    def delivery(start, end, n):
        return delivery_score(window_stats(ff, start, end, n), baseline)

    # Spread of delivery scores over 30s windows: shows whether the
    # signal discriminates at all on this footage.
    scores = [delivery(t, t + 30, sum(1 for w in words if t <= w["start"] < t + 30))[0]
              for t in range(0, max(1, int(duration) - 30), 30)]
    report["delivery_30s_windows"] = {"min": min(scores), "max": max(scores),
                                      "median": sorted(scores)[len(scores) // 2], "count": len(scores)}

    if not os.environ.get("OPENAI_API_KEY"):
        report["selection"] = "BLOCKED: OPENAI_API_KEY not set — selection and rendering not run"
    else:
        import repurpose as r
        from openai import OpenAI
        llm = r.selection_llm(OpenAI(timeout=300, max_retries=2))
        result = timer.run("selection", cs.run_selection, words, segments, duration, llm, delivery)
        report["selection"] = result.summary()
        report["clips"] = [{
            "rank": i + 1, "score": c.score, "breakdown": c.breakdown, "category": c.category,
            "start": c.start, "end": c.end, "title": c.title, "reason": c.reason,
            "hook_line": c.hook_line, "hook_verified": c.hook_line_verified,
            "transcript": c.transcript,
        } for i, c in enumerate(result.clips)]
        report["rejected"] = result.rejected
        if args.render:
            report["renders"] = render_clips(path, result.clips, transcript, args.out, timer)

    report["stages"] = timer.stages
    report["peak_rss_mb"] = round(peak_rss_mb())
    return report


def render_clips(path, clips, transcript, out_dir, timer):
    import repurpose as r
    rendered = []
    for i, c in enumerate(clips, start=1):
        name = f"{os.path.splitext(os.path.basename(path))[0]}_clip{i}.mp4"
        out = os.path.join(out_dir, name)
        words, segments = transcript["words"], transcript["segments"]
        blocks = r._find_speech_blocks(segments, c.start, c.end)
        blocks, _ = r._remove_filler_words(blocks, words)
        captions = os.path.join(out_dir, f"{name}.ass")
        if not r._generate_ass(words, captions, blocks, c.hook_line):
            captions = None
        timer.run(f"render_clip{i}", r._render_clip, path, out, captions, blocks)
        info = probe(out)
        v = next(s for s in info["streams"] if s["codec_type"] == "video")
        rendered.append({"file": name, "width": v["width"], "height": v["height"],
                         "duration_s": round(float(info["format"]["duration"]), 2),
                         "has_audio": any(s["codec_type"] == "audio" for s in info["streams"])})
    return rendered


def to_markdown(reports: list[dict]) -> str:
    lines = ["# AI Repurposer evaluation", ""]
    for rep in reports:
        lines += [f"## {rep['video']} ({rep['duration_min']} min, transcriber: {rep['transcriber']})", ""]
        t = rep["transcript"]
        lines += [f"- Speech: {t['speech_min']} min, {t['words']} words, {t['sentences']} sentences "
                  f"({t['terminal_sentence_pct']}% end on punctuation; median {t['median_sentence_s']}s, "
                  f"max {t['max_sentence_s']}s) — clip target {t['clip_target']}",
                  f"- Delivery score over 30s windows: {rep['delivery_30s_windows']}",
                  f"- Selection: {rep['selection'] if isinstance(rep['selection'], str) else json.dumps(rep['selection'])}",
                  f"- Peak memory: {rep['peak_rss_mb']} MB", "", "| stage | wall s | cpu s |", "|---|---|---|"]
        lines += [f"| {k} | {v['wall_s']} | {v['cpu_s']} |" for k, v in rep["stages"].items()]
        for c in rep.get("clips", []):
            lines += ["", f"### #{c['rank']} {c['title']} — {c['score']}/100 ({c['category']})",
                      f"{c['start']:.1f}s–{c['end']:.1f}s · hook verified: {c['hook_verified']}",
                      f"> {c['transcript']}", "", f"Why: {c['reason']}", f"Breakdown: `{json.dumps(c['breakdown'])}`"]
        lines.append("")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("videos", nargs="+")
    p.add_argument("--out", default=os.path.join(ROOT, "eval", "out"))
    p.add_argument("--transcriber", choices=["openai", "local"], default="openai")
    p.add_argument("--local-model", default="base")
    p.add_argument("--render", action="store_true")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    reports = []
    for v in args.videos:
        print(f"[eval] {v}", flush=True)
        reports.append(evaluate(v, args))
        with open(os.path.join(args.out, "report.json"), "w") as f:
            json.dump(reports, f, indent=2)
    with open(os.path.join(args.out, "report.md"), "w") as f:
        f.write(to_markdown(reports))
    print(to_markdown(reports))


if __name__ == "__main__":
    main()
