"""The Kerala school: its monsoon, its fan, its iron bell - and what used to stand where it was built.

Ordinary sounds first (they must be learned before they can be corrupted):
  iron_bell      the peon's bell: a hung iron rail struck with a rod. One stroke per period, a long run at 3:30.
  ceiling_fan    an old fan on a dying regulator: blade-pass air, motor hum, a bearing tick once per revolution.
  monsoon        rain on Mangalore tiles, eaves overflowing onto laterite, a gutter spouting.
  night          a real Kerala night (frogs, insects, a distant temple) through the open verandah.
Then the kaavu:
  chilanka       anklet bells with a step. The performer is walking the corridor.
  theyyam_*      archival chenda / ilathalam / chant / cries, cut from real performances and heard wrongly.
"""
import glob
import json

import numpy as np

from .core import (CACHE, RATE, ROOT, bp, conv, db, fade, hp, load, lp, peak_eq, resample, rng, sec, smooth_noise,
                   source, zhp)
from . import matter as M

KERALA = CACHE / "kerala"
VSCO = CACHE / "VSCO-2-CE"


# ---------------------------------------------------------------- the school day

def _rail():
    """Impulse response of the bell: a real hammer-struck brake drum (cast iron), pitched down to a heavier rail."""
    x = M.sample("VCSL/Idiophones/Struck Idiophones/Brake Drum/BrakeDrum1_Hammer_v3_rr1_Mid.wav")
    on = int(np.argmax(np.abs(x) > 0.2 * np.abs(x).max()))
    x = resample(x[on:on + sec(4.0)], 0.82)
    return zhp(x, 180, 2) / (np.max(np.abs(x)) + 1e-9)


def iron_bell(seed, strokes=1, interval=0.9, force=1.0, run=False):
    """strokes: number of strikes. run=True: the end-of-day run, fast and accelerating then slowing."""
    r = rng("ironbell", seed)
    g = _rail()
    times, t = [], 0.0
    for i in range(strokes):
        times.append(t)
        if run:
            t += interval * (0.55 + 0.45 * abs(np.sin(np.pi * i / max(1, strokes)))) * r.uniform(0.9, 1.1)
        else:
            t += interval * r.uniform(0.95, 1.08)
    n = sec(t + 4.5)
    exc = np.zeros(n)
    for i, tt in enumerate(times):
        a = sec(tt)
        hit = M.contact(r, r.uniform(0.25, 0.5), 2.2) * force * r.uniform(0.75, 1.0)
        exc[a:a + len(hit)] += hit
        if r.random() < 0.2:                                  # the rod bounces
            exc[a + sec(0.012)] += 0.2
    y = conv(exc, g)[:n]
    rattle = bp(r.standard_normal(n), 2500, 7000) * np.convolve(exc, np.exp(-np.arange(sec(0.05)) / sec(0.008)))[:n] * 0.05
    return fade((y / (np.max(np.abs(y)) + 1e-9) + rattle) * 0.7, 0.0005, 0.4)


def iron_bell_unstruck(seed, seconds=10.0):
    """The bell ringing with nobody striking it: the rail's partials swell out of nothing and will not decay."""
    r = rng("unstruck", seed)
    g = _rail()
    frozen = M.spectral_freeze(g[sec(0.15):sec(0.15) + 8192], sec(seconds + 1), r)
    env = np.interp(np.arange(sec(seconds)), [0, sec(seconds * 0.4), sec(seconds * 0.85), sec(seconds)],
                    [0.0, 0.8, 1.0, 0.0]) ** 1.5
    beat = 1 + 0.25 * np.sin(2 * np.pi * 0.7 * np.arange(sec(seconds)) / RATE)
    return frozen[:sec(seconds)] * env * beat * 0.5


def ceiling_fan(seconds, seed="fan", rpm=260.0, dying=0.3):
    """A 1970s ceiling fan: three blades, a worn bearing, a regulator that can't hold its speed."""
    r = rng("fan", seed)
    n = sec(seconds)
    rate = rpm / 60.0 * (1 - dying * 0.35 * (smooth_noise(r, n, 6.0, 0, 1)))
    phase = np.cumsum(rate / RATE)
    blade = 0.5 + 0.5 * np.cos(2 * np.pi * 3 * phase)
    air = bp(r.standard_normal(n), 120, 2400) * (0.35 + 0.65 * blade ** 2)
    t = np.arange(n) / RATE
    motor = sum(np.sin(2 * np.pi * 50 * k * t + r.uniform(0, 6)) / k ** 1.5 for k in (1, 2, 3, 4))
    motor = motor * (0.6 + 0.4 * blade)
    ticks = np.zeros(n)
    idx = np.where(np.diff(np.floor(phase)) > 0)[0]
    for i in idx:                                    # bearing tick once per turn (not every turn)
        if r.random() < 0.7:
            m = sec(0.006)
            ticks[i:i + m] += hp(r.standard_normal(m), 2000)[:n - i] * np.hanning(m)[:n - i] * r.uniform(0.3, 1.0)
    squeak = np.zeros(n)
    for i in idx[::int(r.integers(3, 6))]:            # and a dry squeak every few turns
        m = sec(0.08)
        hz = r.uniform(1400, 2300) * (1 + 0.1 * np.linspace(0, 1, m))
        s = np.sin(2 * np.pi * np.cumsum(hz) / RATE) * np.hanning(m)
        squeak[i:i + m] += s[:n - i] * r.uniform(0.02, 0.07)
    y = air / (np.std(air) + 1e-9) * 0.25 + 0.03 * motor + 0.12 * M.modal(ticks, [900, 2300, 3700], [.05, .03, .02], [1, .6, .3]) + squeak
    return y / (np.max(np.abs(y)) + 1e-9) * 0.5


def chilanka(seed, force=1.0):
    """A step with anklet bells: dozens of tiny bells, a burst then a settling shimmer."""
    r = rng("chilanka", seed)
    files = sorted(glob.glob(str(VSCO / "Miscellania Raw" / "*" / "sleighbell*_hit*.wav")))
    x = load(files[int(r.integers(len(files)))])
    x = resample(x, r.uniform(1.15, 1.35))[:sec(1.0)]
    x = hp(x, 1500)
    return fade(x / (np.max(np.abs(x)) + 1e-9) * 0.5 * force, 0.0005, 0.25)


def monsoon(seconds, seed=0, where="classroom"):
    """Monsoon on Mangalore tiles: the real rain beds, plus water running off the tile edges in uneven streams and
    a gutter spout splashing onto laterite somewhere outside the window."""
    from . import bed
    r = rng("monsoon", seed)
    base = bed.rain(seconds, seed, where)
    n = base.shape[1]
    # eave run-off: thousands of drops falling ~3 m onto stone/water - the tile edge's own rhythm
    drops = np.zeros(n)
    rate = 40 + 30 * smooth_noise(r, n, 4.0, 0, 1)
    count = int(np.sum(rate) / RATE)
    pos = r.integers(0, n, count)
    drops[pos] = r.uniform(0.2, 1.0, count)
    eave = M.modal(drops, [600, 1300, 2600, 4100], [0.02, 0.015, 0.01, 0.008], [1, .7, .5, .3])
    eave = bp(eave, 300, 6000)
    spout = lp(r.standard_normal(n), 900) * (0.6 + 0.4 * smooth_noise(r, n, 0.8, 0, 1))
    from .space import spread
    extra = spread(eave / (np.std(eave) + 1e-9) * 0.08 + spout / (np.std(spout) + 1e-9) * 0.05, 0.9, "eave%d" % seed)
    if where != "classroom":
        extra = lp(extra, 2500)
    return base + extra * np.std(base) * 1.3


def night(seconds, seed=0, offset=None):
    """A real Kerala night (Varkala): frogs, insects, palms, a temple far off."""
    r = rng("knight", seed)
    full = load(KERALA / "OdayamSoundscapeatnight.flac", stereo=True)
    n = sec(seconds)
    a = sec(offset if offset is not None else r.uniform(5, full.shape[1] / RATE - seconds - 5))
    return zhp(full[:, a:a + n], 40, 2)


# ---------------------------------------------------------------- the kaavu (archival Theyyam)

def tags():
    path = ROOT / "build" / "analysis" / "tags.json"
    return json.loads(path.read_text()) if path.exists() else {}


def find(label, min_score=0.3, files=None):
    """All 10-s windows in the archive whose AudioSet tags include `label`, best first."""
    hits = []
    for f, rows in tags().items():
        if files and f not in files:
            continue
        for start, top in rows:
            for name, score in top:
                if name == label and score >= min_score:
                    hits.append((score, f, start))
    return sorted(hits, reverse=True)


def archive(f, start, seconds):
    return load(KERALA / f, start=start, length=seconds)
