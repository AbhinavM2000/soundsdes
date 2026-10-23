"""Godot-ready Kerala horror pack -> godot/audio/kerala_pack/ (+ manifest.json)

    python3 pipeline/kerala_pack.py

  soundscape_kerala_night.ogg   stereo seamless 60 s loop: monsoon on tiles, night, dying fan, breathing vent
  cry_*.wav                     possession cries from the owner's cloned voice: fry -> tremolo -> shriek, growl
  priest_*.wav                  a priest/oracle gone wrong (owner's voice): trembling, mouth freezing, reversed
                                whispers underneath. Original lines - no real mantras.
  weird_*.wav                   the bell nobody strikes, anklet bells walking, a whisper played backwards
World one-shots are MONO and dry (AudioStreamPlayer3D adds distance + room). The soundscape is 2D stereo.
"""
import json
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from school import bed, cast_ml as C, kerala as K, matter as M, voice as V   # noqa: E402
from school.core import (ROOT, RATE, add, db, fade, hp, limiter, lp, lufs, rng, sec, seamless_loop,
                         true_peak_db, write)                                  # noqa: E402

OUT = ROOT / "godot" / "audio" / "kerala_pack"
ITEMS = []
FRAME = V.FRAME_MS / 1000

PRIEST = {  # original lines (meaning in the manifest)
    "didnt_call_you": ("നിന്റെ പേര് ഞാൻ വിളിച്ചില്ലല്ലോ. പിന്നെ നീ എന്തിനാ ഉത്തരം പറഞ്ഞത്?",
                       "I never called your name. So why did you answer?"),
    "dont_turn": ("തിരിഞ്ഞു നോക്കരുത്. അവൾ നിന്റെ തൊട്ടു പിന്നിലുണ്ട്.", "Don't turn around. She is right behind you."),
    "built_on_grove": ("ഈ സ്കൂൾ പണിതത് ഞങ്ങളുടെ കാവിന്റെ മുകളിലാണ്.", "This school was built on top of our grove."),
    "one_child": ("ഓരോ വർഷവും ഒരു കുട്ടിയെ വേണം. ഈ വർഷം നീ.", "Every year it needs one child. This year, you."),
    "lamp": ("വിളക്ക് കെടുത്തരുത്. അവൾ ഇരുട്ടിലാണ് നടക്കുന്നത്.", "Don't put out the lamp. She walks in the dark."),
    "no_one_breathe": ("എല്ലാവരും ഇരിക്കൂ. ആരും എഴുന്നേൽക്കരുത്. ആരും ശ്വസിക്കരുത്.",
                       "Everyone sit down. Nobody stand up. Nobody breathe."),
    "ammu_still": ("അമ്മു ഇപ്പോഴും ക്ലാസ്സിലുണ്ട്. ഇരുപത് വർഷമായി.", "Ammu is still in class. For twenty years."),
    "waiting": ("അത് ഇവിടെ നിന്ന് പോയിട്ടില്ല. അത് നിന്നെ കാത്തിരിക്കുകയായിരുന്നു.",
                "It never left. It has been waiting for you."),
}


def save(name, x, kind, level, meaning="", loop=False):
    x = np.asarray(x, dtype=np.float64)
    x = hp(x, 30, 2)
    x = x * db(level - lufs(x if x.ndim == 2 else np.vstack([x, x])))
    if true_peak_db(x) > -1.5:
        y = limiter(x if x.ndim == 2 else np.vstack([x, x]), -1.5)
        x = y if x.ndim == 2 else y[0]
    if not loop:
        x = fade(x, 0.002, 0.05)
    path = OUT / name
    write(path, x, bits=16) if path.suffix == ".wav" else write(path, x)
    ITEMS.append({"file": name, "kind": kind, "loop": loop, "channels": 1 if x.ndim == 1 else 2,
                  "seconds": round(x.shape[-1] / RATE, 2), "meaning": meaning})
    print(" ", name, ITEMS[-1]["seconds"], "s", flush=True)


def contour(points, n_seconds, tremolo_hz=0.0, tremolo_semi=0.0, jitter=0.0, seed="c"):
    r = rng("cry", seed)
    t = np.arange(int(n_seconds / FRAME)) * FRAME
    ts, hz = zip(*points)
    base = np.exp(np.interp(t, ts, np.log(hz)))
    trem = tremolo_semi * np.sin(2 * np.pi * tremolo_hz * t + r.uniform(0, 6)) * np.clip(t / 0.5, 0, 1)
    jit = jitter * np.convolve(r.standard_normal(len(t)), np.ones(4) / 4, mode="same")
    return base * 2 ** ((trem + jit) / 12)


def cry(vowel, points, seconds, seed, tremolo_hz=6.5, tremolo_semi=1.2, jitter=0.6, growl=0.45, breath=0.35):
    f0 = contour(points, seconds, tremolo_hz, tremolo_semi, jitter, seed)
    y = V.vocalise(vowel, f0, None, seed=seed, breathiness=breath)
    if growl:   # subharmonic layer: the throat tearing
        g = V.vocalise(vowel, f0 * 0.5, None, seed=seed + "g", breathiness=0.6)
        n = min(len(y), len(g))
        y = y[:n] + growl * g[:n]
    y = np.tanh(2.2 * y / (np.max(np.abs(y)) + 1e-9))
    return fade(y, 0.03, 0.25)


def run(fn):
    try:
        fn()
    except Exception:
        print("FAILED:", fn.__name__)
        traceback.print_exc()


def soundscape():
    n = 64.0
    x = K.monsoon(n, 21, "classroom")
    x = x * db(-34 - lufs(x))
    night = lp(K.night(n, 4), 3000)
    x = x + night * db(-44 - lufs(night))
    fan = K.ceiling_fan(n, "pack", dying=0.6)
    x = x + np.vstack([fan, fan]) * db(-46 - lufs(np.vstack([fan, fan])))
    vent = bed.vent(n, 9, breathing=1.0, rate=0.22)
    x = x + vent * db(-50 - lufs(vent))
    save("soundscape_kerala_night.ogg", seamless_loop(x, sec(60), 4.0), "soundscape", -30,
         "monsoon night in an empty Kerala classroom; the vent breathes", loop=True)


def cries():
    vowel = C.sir("അയ്യോ.")
    save("cry_possession_rise.wav", cry(vowel, [(0, 72), (0.5, 80), (1.6, 300), (2.4, 560), (3.6, 520), (4.2, 330)], 4.4,
                                        "rise", 6.8, 1.6), "cry", -20, "fry into a trembling possession shriek")
    save("cry_ululation.wav", cry(vowel, [(0, 260), (1.5, 420), (3.0, 300), (4.5, 380)], 4.6, "ulul", 5.0, 3.0, 0.4, 0.2),
         "cry", -21, "wavering ritual wail")
    save("cry_shriek_short.wav", cry(vowel, [(0, 220), (0.35, 880), (1.2, 760), (1.6, 500)], 1.7, "shriek", 9.0, 1.0, 1.2, 0.3, 0.5),
         "cry", -19, "sudden shriek")
    save("cry_growl.wav", cry(vowel, [(0, 62), (1.5, 75), (3.0, 58)], 3.2, "growl", 3.0, 0.8, 1.5, 0.9, 0.6),
         "cry", -23, "low guttural growl")
    save("cry_sob_breaking.wav", cry(vowel, [(0, 190), (0.6, 240), (1.0, 150), (1.6, 260), (2.2, 120)], 2.6, "sob", 11.0, 2.0, 1.0, 0.15, 0.55),
         "cry", -24, "sobbing that breaks into something else")
    ayyo = C.sir("അയ്യോ. അയ്യോ.")
    st = V.stretch(ayyo, 2.8)
    save("cry_ayyo_stretched.wav", np.tanh(1.8 * st / (np.max(np.abs(st)) + 1e-9)), "cry", -22,
         "'ayyo' (the Malayali cry of distress) stretched beyond breath")


def priest():
    for key, (text, meaning) in PRIEST.items():
        x = C.sir(text, 0.85)
        d = len(x) / RATE
        t = np.linspace(0, d, 2000)
        r = rng("priest", key)
        curve = 2 ** ((1.1 * np.sin(2 * np.pi * 7.2 * t) * np.clip(t / d * 2, 0, 1) + 0.4 * r.standard_normal(2000)) / 12)
        trembling = V.reshape(x, f0_curve=curve, breath=0.25)       # the oracle's tremor
        frozen = V.freeze(V.stretch(x, 1.7), 0.4, 0.5, monotone=True)
        under = V.whisper(V.reverse_speech(x))
        n = max(len(trembling), len(frozen), len(under))
        mix = np.zeros(n)
        add(mix, trembling, 0)
        add(mix, frozen * 0.35, int(0.15 * RATE))
        add(mix, under * 0.3, 0)
        save("priest_%s.wav" % key, mix, "priest", -21, meaning)
        save("priest_%s_tremor_only.wav" % key, trembling, "priest", -21, meaning + " (trembling, clean)")


def weird():
    save("weird_bell_unstruck.wav", K.iron_bell_unstruck(5, 9.0), "weird", -26, "the school bell ringing with nobody striking it")
    seq = np.zeros(sec(6))
    for i, (t, s) in enumerate(M.walk(8, "lino", 0.6, seed="pack")):
        add(seq, s * 0.6, sec(t))
        add(seq, K.chilanka(700 + i, 0.8), sec(t + 0.01))
    save("weird_chilanka_steps.wav", seq, "weird", -26, "footsteps wearing anklet bells")
    save("weird_ammu_reversed.wav", V.whisper(V.reverse_speech(C.sir("അമ്മു. അമ്മു എവിടെ?"))), "weird", -27,
         "'Ammu, where is Ammu?' whispered backwards")
    save("weird_haajar_whisper.wav", C.kid(C.ANSWER, 4, mood="whisper"), "weird", -27, "a child whispering 'haajar'")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for fn in (soundscape, cries, priest, weird):
        print(fn.__name__, flush=True)
        run(fn)
    (OUT / "manifest.json").write_text(json.dumps({"rate": RATE, "assets": ITEMS}, indent=1, ensure_ascii=False))
    print(len(ITEMS), "files")


if __name__ == "__main__":
    main()
