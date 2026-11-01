"""
Speaker estimation for the AI Repurposer — local, no paid API.

Voice embeddings (WeSpeaker ResNet34, ONNX, CC-BY-4.0) are computed on
overlapping speech windows and clustered. The result is only trusted when
the clusters are clearly separated; otherwise no labels are produced.
"Unknown" is a valid answer and is never filled in with a guess.

When a VisualTimeline is available, voice clusters are cross-checked
against which camera/person is on screen and the agreement is reported,
so the caller knows whether "the person on screen" can be trusted as "the
person talking" (true for speaker-view recordings, false for a fixed wide
shot).

Measured on a real two-speaker interview (15 min, held-out): the voice
model identified the on-screen person for 275/275 windows.
"""
from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

SR = 16000
MODEL_URL = "https://huggingface.co/onnx-community/wespeaker-voxceleb-resnet34-LM/resolve/main/onnx/model.onnx"
_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATHS = [os.path.join(_HERE, "models", "wespeaker.onnx"), "/tmp/viyo_models/wespeaker.onnx"]

WINDOW_SECONDS = 3.0
HOP_SECONDS = 2.0
MIN_SPEECH_RATIO = 0.7
MAX_SPEAKERS = 5
# Minimum silhouette (cosine) for accepting k >= 2 clusters, and the
# smallest share of windows a cluster may hold. Chosen from real data:
# two real speakers scored ~0.5, one speaker split in two scored < 0.1.
MIN_SILHOUETTE = 0.25
MIN_CLUSTER_SHARE = 0.06
# A sentence is labelled only when this much of its voiced time agrees.
MIN_SENTENCE_CONFIDENCE = 0.6

_session = None
_session_failed = False


# ---------------------------------------------------------------- features

def _mel(f):
    return 1127.0 * np.log(1.0 + f / 700.0)


def _mel_banks(n_mels=80, n_fft=512, sr=SR, low=20.0):
    mels = np.linspace(_mel(low), _mel(sr / 2), n_mels + 2)
    bin_mel = _mel(sr / n_fft * np.arange(n_fft // 2 + 1))
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mels):
        lo, c, hi = mels[i], mels[i + 1], mels[i + 2]
        fb[i] = np.maximum(0, np.minimum((bin_mel - lo) / (c - lo), (hi - bin_mel) / (hi - c)))
    return fb


_FB = _mel_banks()


def fbank(x: np.ndarray) -> Optional[np.ndarray]:
    """Kaldi-style 80-band log-mel features (25 ms / 10 ms, Hamming,
    pre-emphasis 0.97, mean-normalised) — the input WeSpeaker expects."""
    x = x.astype(np.float64) * 32768.0
    n = 1 + (len(x) - 400) // 160
    if n < 9:
        return None
    idx = np.arange(n)[:, None] * 160 + np.arange(400)[None, :]
    f = x[idx]
    f = f - f.mean(axis=1, keepdims=True)
    f = np.concatenate([f[:, :1] * (1 - 0.97), f[:, 1:] - 0.97 * f[:, :-1]], axis=1) * np.hamming(400)
    power = np.abs(np.fft.rfft(f, 512, axis=1)) ** 2
    m = np.log(np.maximum(power @ _FB.T, np.finfo(np.float32).eps))
    return (m - m.mean(axis=0)).astype(np.float32)


# ---------------------------------------------------------------- model

def _find_model() -> Optional[str]:
    for p in MODEL_PATHS:
        if os.path.exists(p) and os.path.getsize(p) > 1_000_000:
            return p
    return None


def ensure_model(timeout: float = 120) -> Optional[str]:
    """Path to the speaker model, downloading it once if the image didn't
    ship with it. None (feature off) if it can't be had."""
    found = _find_model()
    if found:
        return found
    dest = MODEL_PATHS[-1]
    try:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        tmp = dest + ".part"
        with urllib.request.urlopen(MODEL_URL, timeout=timeout) as r, open(tmp, "wb") as out:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
        os.replace(tmp, dest)
        return dest
    except Exception as e:
        print(f"[WARN] speaker model unavailable: {e}")
        return None


def _get_session():
    global _session, _session_failed
    if _session is not None or _session_failed:
        return _session
    try:
        import onnxruntime as ort
        path = ensure_model()
        if not path:
            _session_failed = True
            return None
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        _session = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
    except Exception as e:
        print(f"[WARN] speaker encoder unavailable: {e}")
        _session_failed = True
    return _session


def embed(samples: np.ndarray) -> Optional[np.ndarray]:
    """Unit-length 256-d voice embedding, or None."""
    sess = _get_session()
    if sess is None:
        return None
    feats = fbank(samples)
    if feats is None:
        return None
    e = sess.run(None, {"input_features": feats[None]})[0][0]
    n = np.linalg.norm(e)
    return (e / n).astype(np.float32) if n > 0 else None


# ---------------------------------------------------------------- clustering

def _kmeans_cosine(E: np.ndarray, k: int, seed: int = 0, iters: int = 30) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # k-means++ style seeding on cosine distance.
    centers = [E[rng.integers(len(E))]]
    for _ in range(1, k):
        d = 1 - np.max(E @ np.array(centers).T, axis=1)
        p = d ** 2
        centers.append(E[rng.choice(len(E), p=p / p.sum())] if p.sum() > 0 else E[rng.integers(len(E))])
    C = np.array(centers)
    labels = np.zeros(len(E), dtype=int)
    for _ in range(iters):
        new = np.argmax(E @ C.T, axis=1)
        if np.array_equal(new, labels) and _ > 0:
            break
        labels = new
        for j in range(k):
            m = E[labels == j]
            if len(m):
                c = m.mean(axis=0)
                C[j] = c / max(np.linalg.norm(c), 1e-9)
    return labels


def _silhouette(E: np.ndarray, labels: np.ndarray) -> float:
    D = 1 - E @ E.T
    scores = []
    for i in range(len(E)):
        same = labels == labels[i]
        same[i] = False
        if not same.any():
            continue
        a = D[i, same].mean()
        b = min(D[i, labels == c].mean() for c in set(labels) if c != labels[i])
        scores.append((b - a) / max(a, b, 1e-9))
    return float(np.mean(scores)) if scores else 0.0


def cluster_speakers(E: np.ndarray) -> tuple[np.ndarray, float, int]:
    """(labels, silhouette, k). k=1 when no split is clearly supported."""
    if len(E) < 8:
        return np.zeros(len(E), dtype=int), 0.0, 1
    best = (np.zeros(len(E), dtype=int), 0.0, 1)
    best_sil = MIN_SILHOUETTE
    for k in range(2, min(MAX_SPEAKERS, len(E) // 4) + 1):
        labels = max((_kmeans_cosine(E, k, seed=s) for s in range(3)),
                     key=lambda l: _silhouette(E, l) if len(set(l)) == k else -1)
        if len(set(labels)) != k:
            continue
        shares = [np.mean(labels == j) for j in range(k)]
        if min(shares) < MIN_CLUSTER_SHARE:
            continue
        sil = _silhouette(E, labels)
        if sil > best_sil:
            best, best_sil = (labels, sil, k), sil
    return best


# ---------------------------------------------------------------- result

@dataclass
class SpeakerMap:
    reliable: bool = False
    n_speakers: int = 0
    silhouette: float = 0.0
    method: str = "none"                       # "voice" | "none"
    sentence_labels: list = field(default_factory=list)   # "A"/"B"/... or None per sentence
    window_times: np.ndarray = field(default_factory=lambda: np.zeros(0))
    window_labels: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    speaker_seconds: dict = field(default_factory=dict)
    visual_agreement: Optional[float] = None   # share of windows whose voice cluster matches its camera
    on_screen_is_speaker: bool = False         # visual_agreement high enough to trust the picture
    note: str = ""

    def label_at(self, t: float) -> Optional[str]:
        if not self.reliable or len(self.window_times) == 0:
            return None
        i = int(np.argmin(np.abs(self.window_times - t)))
        if abs(self.window_times[i] - t) > HOP_SECONDS * 1.5:
            return None
        return _letter(int(self.window_labels[i]), self.speaker_order)

    speaker_order: list = field(default_factory=list)

    def summary(self) -> dict:
        return {"reliable": self.reliable, "speakers": self.n_speakers, "method": self.method,
                "silhouette": round(self.silhouette, 3), "visual_agreement": self.visual_agreement,
                "on_screen_is_speaker": self.on_screen_is_speaker, "note": self.note}


def _letter(cluster: int, order: list) -> str:
    return chr(ord("A") + order.index(cluster)) if cluster in order else "?"


def estimate_speakers(samples: np.ndarray, ff, sentences: list, timeline=None,
                      max_seconds: float = 4000) -> SpeakerMap:
    """
    samples: mono 16 kHz audio of the whole source. ff: audio_features
    FrameFeatures (for speech detection). sentences: clip_selection.Sentence
    list (labels are produced per sentence). Never raises.
    """
    try:
        if _get_session() is None:
            return SpeakerMap(note="speaker model unavailable")
        dur = len(samples) / SR
        times, embs = [], []
        t = 0.0
        while t + WINDOW_SECONDS <= min(dur, max_seconds):
            lo, hi = np.searchsorted(ff.times, [t, t + WINDOW_SECONDS])
            if hi > lo and ff.speech[lo:hi].mean() >= MIN_SPEECH_RATIO:
                e = embed(samples[int(t * SR): int((t + WINDOW_SECONDS) * SR)])
                if e is not None:
                    times.append(t + WINDOW_SECONDS / 2)
                    embs.append(e)
            t += HOP_SECONDS
        if len(embs) < 8:
            return SpeakerMap(note="not enough clean speech to tell voices apart")
        E = np.array(embs)
        times = np.array(times)
        labels, sil, k = cluster_speakers(E)

        smap = SpeakerMap(method="voice", n_speakers=k, silhouette=sil,
                          window_times=times, window_labels=labels)
        counts = {c: int(np.sum(labels == c)) for c in set(labels)}
        smap.speaker_order = sorted(counts, key=lambda c: -counts[c])
        smap.speaker_seconds = {_letter(c, smap.speaker_order): round(n * HOP_SECONDS, 1) for c, n in counts.items()}

        if k == 1:
            smap.reliable = True
            smap.note = "one voice detected"
        else:
            smap.reliable = True
            smap.note = f"{k} voices detected"

        if timeline is not None and timeline.usable and k >= 2:
            smap.visual_agreement, smap.on_screen_is_speaker = _visual_agreement(smap, timeline)

        smap.sentence_labels = _label_sentences(smap, sentences)
        return smap
    except Exception as e:
        print(f"[WARN] speaker estimation failed: {e}")
        return SpeakerMap(note=f"speaker estimation failed: {e}")


def _visual_agreement(smap: SpeakerMap, timeline) -> tuple[Optional[float], bool]:
    """Share of voice windows whose cluster matches the camera cluster on
    screen, under the best 1:1 mapping. High => the picture shows whoever is
    talking (speaker view); low => it doesn't (one wide shot)."""
    shot_clusters = sorted({s.cluster for s in timeline.shots})
    if len(shot_clusters) < 2:
        return None, False
    pairs = []
    for t, lab in zip(smap.window_times, smap.window_labels):
        sh = timeline.shot_at(float(t))
        if sh is not None:
            pairs.append((int(lab), sh.cluster))
    if len(pairs) < 8:
        return None, False
    voice = sorted({p[0] for p in pairs})
    best = 0
    # Greedy 1:1 mapping, largest co-occurrence first.
    co = {(v, s): sum(1 for p in pairs if p == (v, s)) for v in voice for s in shot_clusters}
    used_v, used_s, matched = set(), set(), 0
    for (v, s), n in sorted(co.items(), key=lambda kv: -kv[1]):
        if v in used_v or s in used_s:
            continue
        used_v.add(v); used_s.add(s); matched += n
    agreement = matched / len(pairs)
    return round(agreement, 3), agreement >= 0.9


def _label_sentences(smap: SpeakerMap, sentences: list) -> list:
    out = []
    for s in sentences:
        lo, hi = np.searchsorted(smap.window_times, [s.start - HOP_SECONDS / 2, s.end + HOP_SECONDS / 2])
        w = smap.window_labels[lo:hi]
        if len(w) == 0:
            out.append(None)
            continue
        vals, cnt = np.unique(w, return_counts=True)
        top = int(vals[np.argmax(cnt)])
        conf = float(cnt.max() / len(w))
        out.append(_letter(top, smap.speaker_order) if conf >= MIN_SENTENCE_CONFIDENCE else None)
    return out
