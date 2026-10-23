"""An original Carnatic-style alapana used as a stand-in for a licensed film song, and the tanpura under it.
Scale: Punnagavarali-like (S R1 G2 M1 P D1 N2), sung mostly below and around the tonic, with kampita gamakas
(wide, slow oscillations between neighbouring notes) - the sound of a woman singing to herself in a closed room."""
import numpy as np

from . import voice as V
from .core import RATE, bp, fade, hp, lp, rng, sec

SA = 207.65  # G#3: a comfortable female tonic
SCALE = {"S": 0, "R": 1, "G": 3, "M": 5, "P": 7, "D": 8, "N": 10}
FRAME = V.FRAME_MS / 1000

# (swara, octave offset, beats, gamaka) - original phrase; beats at ~0.55 s
PHRASE = [("P", -1, 2, "k"), ("D", -1, 1, ""), ("N", -1, 2, "k"), ("S", 0, 3, "k"), ("-", 0, 1, ""),
          ("R", 0, 1, ""), ("G", 0, 2, "k"), ("R", 0, 1, ""), ("S", 0, 2, ""), ("N", -1, 3, "k"), ("-", 0, 1, ""),
          ("S", 0, 1, ""), ("G", 0, 1, ""), ("M", 0, 3, "k"), ("G", 0, 1, ""), ("R", 0, 2, "k"), ("S", 0, 1, ""),
          ("N", -1, 2, ""), ("D", -1, 4, "k"), ("-", 0, 2, ""), ("N", -1, 1, ""), ("S", 0, 6, "k"), ("-", 0, 2, "")]


def contour(phrase=PHRASE, beat=0.55, seed="alapana"):
    r = rng("contour", seed)
    f0, mask = [], []
    prev = None
    for sw, octv, beats, gam in phrase:
        n = int(beats * beat / FRAME)
        if sw == "-":
            f0 += [0.0] * n
            mask += [0.0] * n
            prev = None
            continue
        target = 12 * octv + SCALE[sw]
        semis = np.full(n, float(target))
        if prev is not None:                       # glide in from the previous note (jaru)
            g = min(n, int(0.12 / FRAME))
            semis[:g] = np.linspace(prev, target, g)
        if gam == "k":                             # kampita: oscillate up towards the next swara and back
            t = np.arange(n) * FRAME
            depth = r.uniform(0.8, 1.6)
            semis += depth * (0.5 - 0.5 * np.cos(2 * np.pi * r.uniform(3.5, 5.0) * t)) * np.clip(t / 0.3, 0, 1)
        semis += 0.08 * np.sin(2 * np.pi * 5.5 * np.arange(n) * FRAME)
        f0 += list(SA * 2 ** (semis / 12))
        mask += [1.0] * n
        prev = target
    return np.array(f0), np.array(mask)


def tanpura(seconds, seed="tanpura"):
    """Four plucked strings (P, S', S', S) cycling, Karplus-Strong with a jivari-like buzzing bridge."""
    r = rng("tanpura", seed)
    n = sec(seconds)
    out = np.zeros(n + sec(4))
    freqs = [SA * 2 ** (-5 / 12), SA, SA, SA / 2]
    t = 0.0
    k = 0
    while t < seconds:
        f = freqs[k % 4] * (1 + r.normal(0, 0.0005))
        period = int(RATE / f)
        m = sec(4.0)
        buf = r.uniform(-1, 1, period)
        y = np.zeros(m)
        for i in range(m):
            v = 0.4985 * (buf[i % period] + buf[(i + 1) % period])
            buf[i % period] = v
            y[i] = v
        y = y + 0.25 * np.tanh(6 * y)                # bridge buzz
        a = sec(t)
        out[a:a + m] += y * r.uniform(0.7, 1.0)
        t += r.uniform(0.62, 0.72)
        k += 1
    out = hp(out[:n], 70)
    return fade(out / (np.max(np.abs(out)) + 1e-9) * 0.4, 1.0, 1.0)


def alapana(seconds=30.0, seed="alapana"):
    vowel = V.tts("Aaah.", "bf_isabella", 0.5)
    phrases = []
    while sum(len(p) for p in phrases) < sec(seconds):
        f0, mask = contour(seed="%s%d" % (seed, len(phrases)))
        phrases.append(V.vocalise(vowel, f0, mask, seed="%s%d" % (seed, len(phrases))))
    voice = np.concatenate(phrases)[:sec(seconds)]
    drone = tanpura(seconds, seed)
    return voice + 0.35 * drone[:len(voice)]
