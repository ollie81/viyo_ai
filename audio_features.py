"""
Vocal-delivery signals for clip selection, computed locally with FFmpeg
and numpy (no paid API).

Everything here is RELATIVE to the speaker's own baseline across the
whole video. Absolute loudness is a recording property (mic distance,
gain), not a sign of an interesting moment — a creator who records hot
would otherwise have every window score high. What carries signal is
change against how this person normally sounds: pitch moving more than
usual, dynamics opening up, pace picking up.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Optional

import numpy as np

SAMPLE_RATE = 16000
_FRAME_SECONDS = 0.04
_HOP_SECONDS = 0.02
_MIN_F0_HZ = 70.0
_MAX_F0_HZ = 400.0
# Normalised autocorrelation peak needed to call a frame voiced. Lower
# admits more noise as "pitch"; higher drops breathy speech.
_VOICING_THRESHOLD = 0.45
# Frames this far below the loud end of the recording are treated as
# silence / room noise rather than speech.
_SPEECH_FLOOR_BELOW_P95_DB = 30.0
_BATCH_FRAMES = 4096

# Baseline windows: every window of this length (sliding by the hop)
# gives the distribution a candidate is compared against.
_BASELINE_WINDOW_SECONDS = 30.0
_BASELINE_HOP_SECONDS = 10.0


@dataclass
class FrameFeatures:
    times: np.ndarray       # frame centre, seconds
    rms_db: np.ndarray      # frame loudness, dBFS
    f0_hz: np.ndarray       # NaN where unvoiced
    speech: np.ndarray      # bool, frame above the speech floor


@dataclass
class WindowStats:
    pitch_spread_st: Optional[float]   # std of pitch in semitones
    energy_spread_db: Optional[float]  # std of loudness over speech frames
    level_db: Optional[float]          # median loudness over speech frames
    words_per_second: Optional[float]
    voiced_ratio: float


def decode_pcm(path: str, timeout: Optional[float] = None) -> np.ndarray:
    """Mono 16 kHz float32 samples in [-1, 1]. Raises on decode failure."""
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1",
         "-ar", str(SAMPLE_RATE), "-f", "s16le", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"audio decode failed: {proc.stderr.decode(errors='replace')[-300:]}")
    return np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def frame_features(samples: np.ndarray, sr: int = SAMPLE_RATE) -> FrameFeatures:
    frame = int(sr * _FRAME_SECONDS)
    hop = int(sr * _HOP_SECONDS)
    if len(samples) < frame:
        empty = np.zeros(0)
        return FrameFeatures(empty, empty, empty, empty.astype(bool))

    n_frames = 1 + (len(samples) - frame) // hop
    min_lag = int(sr / _MAX_F0_HZ)
    max_lag = int(sr / _MIN_F0_HZ)
    nfft = 1 << int(np.ceil(np.log2(2 * frame)))
    window = np.hanning(frame).astype(np.float32)

    rms_db = np.empty(n_frames, dtype=np.float32)
    f0 = np.full(n_frames, np.nan, dtype=np.float32)

    for b0 in range(0, n_frames, _BATCH_FRAMES):
        b1 = min(n_frames, b0 + _BATCH_FRAMES)
        idx = (np.arange(b0, b1)[:, None] * hop) + np.arange(frame)[None, :]
        frames = samples[idx]
        frames = frames - frames.mean(axis=1, keepdims=True)

        rms = np.sqrt(np.mean(frames ** 2, axis=1))
        rms_db[b0:b1] = 20 * np.log10(np.maximum(rms, 1e-6))

        spec = np.fft.rfft(frames * window, n=nfft, axis=1)
        ac = np.fft.irfft(np.abs(spec) ** 2, n=nfft, axis=1)[:, : max_lag + 2]
        energy = ac[:, 0]
        valid = energy > 1e-9
        norm = np.zeros_like(ac)
        norm[valid] = ac[valid] / energy[valid, None]

        search = norm[:, min_lag: max_lag + 1]
        best = np.argmax(search, axis=1)
        peak = search[np.arange(len(best)), best]
        lag = best + min_lag

        # Parabolic interpolation around the peak for sub-sample lag.
        left = norm[np.arange(len(lag)), lag - 1]
        right = norm[np.arange(len(lag)), np.minimum(lag + 1, norm.shape[1] - 1)]
        denom = left - 2 * peak + right
        shift = np.divide(0.5 * (left - right), denom, out=np.zeros_like(denom),
                          where=np.abs(denom) > 1e-9)
        refined = lag + np.clip(shift, -1, 1)

        voiced = valid & (peak >= _VOICING_THRESHOLD)
        f0[b0:b1] = np.where(voiced, sr / refined, np.nan)

    times = (np.arange(n_frames) * hop + frame / 2) / sr
    loud_ref = float(np.percentile(rms_db, 95)) if n_frames else 0.0
    speech = rms_db > (loud_ref - _SPEECH_FLOOR_BELOW_P95_DB)
    f0 = np.where(speech, f0, np.nan)
    return FrameFeatures(times=times, rms_db=rms_db, f0_hz=f0, speech=speech)


def _semitones(f0: np.ndarray) -> np.ndarray:
    voiced = f0[~np.isnan(f0)]
    if voiced.size == 0:
        return voiced
    st = 12 * np.log2(voiced / np.median(voiced))
    # Autocorrelation pitch trackers make octave errors; anything more
    # than an octave from the speaker's own median is treated as one.
    return st[np.abs(st) <= 12]


def window_stats(ff: FrameFeatures, start: float, end: float,
                 word_count: Optional[int] = None) -> WindowStats:
    if ff.times.size == 0 or end <= start:
        return WindowStats(None, None, None, None, 0.0)
    lo, hi = np.searchsorted(ff.times, [start, end])
    f0 = ff.f0_hz[lo:hi]
    speech = ff.speech[lo:hi]
    db = ff.rms_db[lo:hi][speech]

    st = _semitones(f0)
    voiced_ratio = float(np.mean(~np.isnan(f0))) if f0.size else 0.0
    return WindowStats(
        pitch_spread_st=float(np.std(st)) if st.size >= 20 else None,
        energy_spread_db=float(np.std(db)) if db.size >= 20 else None,
        level_db=float(np.median(db)) if db.size >= 20 else None,
        words_per_second=(word_count / (end - start)) if word_count is not None else None,
        voiced_ratio=voiced_ratio,
    )


@dataclass
class Baseline:
    """Distribution of each window statistic across the whole video."""
    pitch_spread: tuple[float, float]
    energy_spread: tuple[float, float]
    level: tuple[float, float]
    rate: tuple[float, float]


def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.array([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if arr.size < 2:
        return (float(arr.mean()) if arr.size else 0.0, 0.0)
    return float(arr.mean()), float(arr.std())


def build_baseline(ff: FrameFeatures, duration: float, word_times: list[float]) -> Baseline:
    """
    Slides a fixed window over the whole video and records each
    statistic, skipping windows with almost no speech (an intro with
    music, a long silence) so they don't drag the baseline down.
    """
    starts = np.arange(0, max(duration - _BASELINE_WINDOW_SECONDS, 0) + 1e-9, _BASELINE_HOP_SECONDS)
    if starts.size == 0:
        starts = np.array([0.0])
    wt = np.sort(np.asarray(word_times, dtype=np.float64))

    pitch, energy, level, rate = [], [], [], []
    for s in starts:
        e = min(duration, s + _BASELINE_WINDOW_SECONDS)
        n_words = int(np.searchsorted(wt, e) - np.searchsorted(wt, s))
        if n_words < 10:
            continue
        ws = window_stats(ff, s, e, n_words)
        pitch.append(ws.pitch_spread_st)
        energy.append(ws.energy_spread_db)
        level.append(ws.level_db)
        rate.append(ws.words_per_second)

    return Baseline(
        pitch_spread=_mean_std(pitch),
        energy_spread=_mean_std(energy),
        level=_mean_std(level),
        rate=_mean_std(rate),
    )


def _z(value: Optional[float], dist: tuple[float, float], cap: float = 3.0) -> float:
    if value is None:
        return 0.0
    mean, std = dist
    if std < 1e-6:
        return 0.0
    return float(np.clip((value - mean) / std, -cap, cap))


# Weights over z-scores. Loudness level is deliberately the smallest and
# is capped tighter than the rest: being louder than usual is weak
# evidence on its own, while pitch movement and dynamic range track
# emphasis, emotion and storytelling delivery much more closely.
_W_PITCH = 0.40
_W_ENERGY = 0.35
_W_RATE = 0.15
_W_LEVEL = 0.10


def delivery_score(ws: WindowStats, baseline: Baseline) -> tuple[int, dict]:
    """
    0-100, where 50 is this speaker's typical delivery. Returns the
    score and the z-scores behind it so the breakdown is inspectable.
    """
    z_pitch = _z(ws.pitch_spread_st, baseline.pitch_spread)
    z_energy = _z(ws.energy_spread_db, baseline.energy_spread)
    z_rate = float(np.clip(_z(ws.words_per_second, baseline.rate), -1.5, 1.5))
    z_level = float(np.clip(_z(ws.level_db, baseline.level), -1.0, 1.0))
    raw = _W_PITCH * z_pitch + _W_ENERGY * z_energy + _W_RATE * z_rate + _W_LEVEL * z_level
    score = int(round(100 / (1 + np.exp(-1.6 * raw))))
    return score, {
        "pitch_z": round(z_pitch, 2),
        "energy_z": round(z_energy, 2),
        "rate_z": round(z_rate, 2),
        "level_z": round(z_level, 2),
        "voiced_ratio": round(ws.voiced_ratio, 2),
    }
