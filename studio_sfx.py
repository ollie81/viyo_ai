"""
Built-in sound effects for Viyo Studio, synthesised with ffmpeg on our
own server: no AI call, no cost per use and no third-party licence.

Each effect is a short WAV (48 kHz stereo). "Bed" effects (rain, wind)
are looped to fill the whole scene; the rest play once from the start
of the scene.
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile

# id -> (label, ffmpeg lavfi source, target peak in dBFS, loops for the whole scene)
EFFECTS: dict[str, tuple[str, str, float, bool]] = {
    "whoosh": ("Whoosh", "anoisesrc=c=pink:d=0.9:a=0.9,bandpass=f=1400:w=1200,afade=t=in:d=0.35,afade=t=out:st=0.35:d=0.55", -6.0, False),
    "impact": ("Impact", "aevalsrc='0.95*sin(2*PI*(70-35*t)*t)*exp(-3.5*t)+0.25*(random(0)-0.5)*exp(-18*t)':d=1.6", -3.0, False),
    "heartbeat": ("Heartbeat", "aevalsrc='0.9*sin(2*PI*48*t)*(exp(-22*mod(t,0.95))+0.75*exp(-22*mod(t+0.7,0.95)))':d=4", -6.0, False),
    "rain": ("Rain", "anoisesrc=c=white:d=8:a=0.5,highpass=f=500,lowpass=f=7000,afade=t=in:d=1", -14.0, True),
    "wind": ("Wind", "anoisesrc=c=brown:d=8:a=0.8,bandpass=f=420:w=500,tremolo=f=0.25:d=0.6,afade=t=in:d=1.5", -14.0, True),
    "phone": ("Phone ringing", "aevalsrc='0.35*(sin(2*PI*440*t)+sin(2*PI*480*t))*lt(mod(t,3),1.4)':d=4.5", -9.0, False),
    "knock": ("Door knock", "aevalsrc='0.9*sin(2*PI*120*t)*(exp(-40*t)+exp(-40*abs(t-0.28))*gte(t,0.28)+exp(-40*abs(t-0.56))*gte(t,0.56))':d=1.2", -5.0, False),
    "riser": ("Tension riser", "aevalsrc='0.5*sin(2*PI*(110+220*t*t/9)*t)*(t/3)':d=3,afade=t=out:st=2.7:d=0.3", -8.0, False),
}

_cache_dir = os.path.join(tempfile.gettempdir(), "viyo_sfx")


class UnknownEffect(ValueError):
    pass


def labels() -> list[dict]:
    return [{"id": k, "label": v[0], "loops": v[3]} for k, v in EFFECTS.items()]


def path_for(effect_id: str) -> str:
    """The effect's WAV file, made once and cached."""
    if effect_id not in EFFECTS:
        raise UnknownEffect(effect_id)
    os.makedirs(_cache_dir, exist_ok=True)
    out = os.path.join(_cache_dir, f"{effect_id}.wav")
    if not os.path.exists(out):
        _, src, peak, _ = EFFECTS[effect_id]
        raw = out + ".raw.wav"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", src,
                        "-af", "aformat=channel_layouts=stereo,aresample=48000", raw], check=True)
        # Bring every effect to its target peak, whatever the synth produced.
        r = subprocess.run(["ffmpeg", "-i", raw, "-af", "volumedetect", "-f", "null", "-"],
                           capture_output=True, text=True)
        m = re.search(r"max_volume: (-?[0-9.]+) dB", r.stderr)
        gain = peak - float(m.group(1)) if m else 0.0
        tmp = out + ".part.wav"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", raw, "-af", f"volume={gain:.2f}dB", tmp], check=True)
        os.replace(tmp, out)
        os.remove(raw)
    return out


def loops(effect_id: str) -> bool:
    return EFFECTS[effect_id][3]
