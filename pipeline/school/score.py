"""The score, built from things that could be in the building: the music room's upright piano, a string
section that sounds like it is rehearsing in another wing, bowed metal (cymbal, brake drum - the building's own
voice), and the assembly hymn.

Harmonic premise: UK mains hum is 50 Hz, a hair above G1 (49.0 Hz). The building hums G. The child's piano
exercise climbs C-D-E-F and never reaches G. The hymn never resolves to G. The strings lean on F# and A-flat a
quarter-tone either side of it. The music is a message that cannot finish.
"""
from functools import lru_cache
import glob
import re

import numpy as np

from .core import (CACHE, RATE, conv, db, fade, hp, load, lp, peak_eq, resample, rng, sec, smooth_noise,
                   varispeed, zhp)
from .matter import spectral_freeze

VSCO = CACHE / "VSCO-2-CE"
NOTE = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5, "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11}


def hz(midi):
    return 440.0 * 2 ** ((midi - 69) / 12)


def midi_of(name):
    m = re.match(r"([A-G]#?)(-?\d)", name)
    return NOTE[m.group(1)] + 12 * (int(m.group(2)) + 1)


# ---------------------------------------------------------------- piano

@lru_cache(maxsize=None)
def _piano(dyn, idx):
    return load(VSCO / "Keys" / "Upright Piano" / ("Player_dyn%d_rr1_%03d.wav" % (dyn, idx)), stereo=True)


def piano(midi, velocity=0.6, length=None, detune_cents=0.0, una_corda=False):
    """One note from the upright. velocity 0..1 picks the dynamic layer and gain; detune for the out-of-tune piano
    that has sat in a damp room for twenty years."""
    keys = np.arange(0, 45, 2)
    key_numbers = 21 + 2 * keys
    i = int(np.argmin(np.abs(key_numbers - midi)))
    dyn = 1 if velocity < 0.4 else (2 if velocity < 0.75 else 3)
    x = _piano(dyn, int(keys[i])).mean(axis=0)
    ratio = 2 ** ((midi - key_numbers[i] + detune_cents / 100) / 12)
    x = resample(x, ratio) if abs(ratio - 1) > 1e-4 else x.copy()
    on = int(np.argmax(np.abs(x) > 0.02 * np.abs(x).max()))
    x = x[max(0, on - sec(0.003)):]
    if length:
        x = fade(x[:sec(length + 0.25)], 0, 0.25)
    if una_corda:
        x = lp(x, 2500)
    return x / (np.max(np.abs(x)) + 1e-9) * (0.25 + 0.75 * velocity)


def exercise(seed, notes=(60, 62, 64, 65), tempo=0.62, hesitate=1.0, stop_after=None, wrong=None, velocity=0.42):
    """A child practising the five-finger exercise: uneven time, uneven touch, a breath before the next note.
    It never plays the G. wrong=(index, midi) replaces one note; stop_after cuts it short."""
    r = rng("exercise", seed)
    notes = list(notes)
    if wrong:
        notes[wrong[0]] = wrong[1]
    if stop_after is not None:
        notes = notes[:stop_after]
    out = np.zeros(sec(len(notes) * tempo * 2 + 6))
    t = 0.05
    for k, m in enumerate(notes):
        v = velocity * r.uniform(0.75, 1.15) * (0.9 if k % 2 else 1.0)
        note = piano(m, v, detune_cents=r.normal(0, 9)) * 0.9
        a = sec(t)
        out[a:a + len(note)] += note[:len(out) - a]
        t += tempo * r.uniform(0.85, 1.25) + (hesitate * r.uniform(0.0, 0.35) if k == len(notes) - 2 else 0)
    return out[:sec(t + 5)]


def sympathetic(trigger, chord=(54, 56, 61, 62, 66, 68), seed="symp", decay=7.0):
    """The music room piano with its damper pedal stuck down: anything loud in the room sets its strings ringing.
    We convolve the trigger with the long sustains of the piano's own strings - a chord the room 'answers' with."""
    r = rng("symp", seed)
    ir = np.zeros(sec(decay))
    for m in chord:
        s = piano(m, 0.5)[sec(0.08):]            # skip the hammer: strings only
        s = s[:len(ir)] * np.exp(-np.arange(min(len(s), len(ir))) / sec(decay / 3))
        ir[:len(s)] += s * r.uniform(0.5, 1.0)
    y = conv(trigger, ir / (np.sqrt(np.sum(ir ** 2)) + 1e-9))
    return y / (np.max(np.abs(y)) + 1e-9) * 0.5


# ---------------------------------------------------------------- strings

@lru_cache(maxsize=None)
def _string_bank(folder):
    files = sorted(glob.glob(str(VSCO / "Strings" / folder / "*.wav")))
    bank = []
    for f in files:
        m = re.search(r"_([A-G]#?\d)_", f.split("/")[-1])
        if m:
            bank.append((midi_of(m.group(1)), f))
    return bank


def string_note(folder, midi, seconds, seed="str", cents_curve=None, dynamic=1):
    """A sustained string note of any length: nearest sample, re-pitched, extended by overlapping grains of its
    own stable middle (so the bow keeps moving), then bent by a cents curve (slow, unstable intonation)."""
    r = rng("string", folder, midi, seed)
    bank = _string_bank(folder)
    candidates = [b for b in bank if ("_v%d" % dynamic) in b[1].split("/")[-1]] or bank
    base, f = min(candidates, key=lambda b: abs(b[0] - midi))
    x = load(f).copy()
    x = x / (np.max(np.abs(x)) + 1e-9)
    on = int(np.argmax(np.abs(x) > 0.05))
    x = x[on:]
    stable = x[sec(1.0):max(sec(1.5), len(x) - sec(1.5))]
    n = sec(seconds + 1.0)
    out = np.zeros(n + sec(2))
    head = x[:sec(1.2)]
    out[:len(head)] += head * np.r_[np.ones(len(head) - sec(0.3)), np.cos(np.linspace(0, np.pi / 2, sec(0.3))) ** 2]
    grain = sec(1.4)
    hop = sec(0.6)
    win = np.hanning(grain)
    pos = sec(0.9)
    while pos < n:
        a = int(r.integers(0, max(1, len(stable) - grain)))
        out[pos:pos + grain] += stable[a:a + grain] * win * 0.85
        pos += hop
    out = out[:n]
    ratio = 2 ** ((midi - base) / 12)
    if cents_curve is not None:
        curve = np.interp(np.arange(n), np.linspace(0, n, len(cents_curve)), cents_curve)
        speed = ratio * 2 ** (curve / 1200)
        y = varispeed(np.pad(out, (0, n)), speed)
    else:
        y = resample(out, ratio)
    y = y[:sec(seconds)]
    return fade(y, 0.4, 1.2)


def cluster(seed, seconds, voices, drift_cents=35.0, swell=True):
    """voices: list of (folder, midi, gain). Each voice drifts on its own slow random intonation curve, so the
    cluster beats, tightens and loosens - it breathes without ever settling."""
    r = rng("cluster", seed)
    out = np.zeros((2, sec(seconds)))
    for k, (folder, m, g) in enumerate(voices):
        curve = (smooth_noise(r, 400, 0.25 * 400 / 30, -1, 1) * drift_cents)
        y = string_note(folder, m, seconds, seed="%s-%d" % (seed, k), cents_curve=curve)
        pan = r.uniform(-0.6, 0.6)
        out[0, :len(y)] += y * g * np.sqrt(0.5 * (1 - pan))
        out[1, :len(y)] += y * g * np.sqrt(0.5 * (1 + pan))
    if swell:
        env = smooth_noise(r, out.shape[1], seconds / 5, 0.35, 1.0)
        out *= env
    return out


# ---------------------------------------------------------------- bowed metal: the building's voice

def bowed_metal(seed, seconds, kind="cymbal", pitch=0.5, motion=0.3):
    """Real bowed cymbal / bowed brake drum / gong scrape, slowed and frozen into a long sustain whose spectrum
    slowly moves (two freezes from different moments, crossfading)."""
    r = rng("bowed", kind, seed)
    files = {
        "cymbal": glob.glob(str(VSCO / "Percussion" / "susCymb1-bow-*.wav")),
        "brake": glob.glob(str(CACHE / "VCSL" / "Idiophones" / "Struck Idiophones" / "Brake Drum" / "*Bowed*.wav")),
        "gong": glob.glob(str(VSCO / "Percussion" / "gongscrape*.wav")),
        "drum": glob.glob(str(VSCO / "Percussion" / "bassdrum_rub*.wav")),
    }[kind]
    x = load(sorted(files)[int(r.integers(len(files)))])
    x = x / (np.max(np.abs(x)) + 1e-9)
    loud = np.where(np.abs(x) > 0.1)[0]
    a, b = (loud[0], loud[-1]) if len(loud) else (0, len(x))
    n = sec(seconds / pitch + 1)
    f1 = spectral_freeze(x[a + (b - a) // 4:a + (b - a) // 4 + 8192], n, r)
    f2 = spectral_freeze(x[a + 2 * (b - a) // 3:a + 2 * (b - a) // 3 + 8192], n, r)
    mix = smooth_noise(r, n, max(1.0, seconds / 6), 0, 1)
    y = f1 * (1 - mix) + f2 * mix
    # include the real bowed attack too, slowed: friction you can hear
    att = resample(x[a:a + sec(3)], pitch)
    y[:len(att)] += att[:n] * 0.6
    y = resample(y, pitch)[:sec(seconds)]
    am = 1 + motion * smooth_noise(r, len(y), 0.7, -1, 1)
    return fade(y * am / (np.max(np.abs(y)) + 1e-9) * 0.5, 1.0, 2.0)


def reverse_swell(x, seconds=4.0, room_decay=3.0, seed="inhale"):
    """The 'inhale': a reversed reverb tail of x that rushes *into* the moment it is cut. Everything is drawn in."""
    r = rng("inhale", seed)
    tail = r.standard_normal(sec(room_decay)) * np.exp(-6.91 * np.arange(sec(room_decay)) / sec(room_decay))
    wet = conv(x[-sec(seconds):] if x.ndim == 1 else x[..., -sec(seconds):], tail / 40)
    wet = wet[..., ::-1]
    return wet[..., -sec(seconds):]


# ---------------------------------------------------------------- the hymn

HYMN = [
    # (word, [midi notes], beats)   - G major, 3/4, it never comes home to G
    ("Here", [62], [1]), ("we", [67], [1]), ("gather", [67, 69], [1, 1]), ("every", [71, 71], [1, 1]),
    ("morning", [69, 67], [1, 2]),
    ("every", [69, 71], [1, 1]), ("name", [72], [1]), ("and", [71], [1]), ("every", [69, 67], [1, 1]),
    ("face", [69], [3]),
    ("no", [74], [1]), ("one", [71], [1]), ("leaves", [67], [1]), ("and", [69], [1]), ("no", [71], [1]),
    ("one", [67], [1]), ("wanders", [64, 62], [1, 2]),
    ("each", [64], [1]), ("of", [67], [1]), ("us", [66], [1]), ("must", [64], [1]), ("know", [62], [1]),
    ("our", [66], [1]), ("place", [66], [6]),
]
