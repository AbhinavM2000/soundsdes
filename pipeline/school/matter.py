"""The physical school: objects, materials and Foley.

Two rules: (1) every impact excites a *body* (modal resonators with decay per mode, or a real recorded object
used as an impulse response), never a bare oscillator; (2) nothing repeats exactly - each call takes a seed and
varies force, contact point (which modes are excited), timing and grit.
"""
from functools import lru_cache
import glob

import numpy as np
from scipy import signal

from .core import (CACHE, RATE, bp, conv, db, fade, hp, load, lp, peak_eq, rng, sec, smooth_noise, source,
                   zhp)

VSCO = CACHE / "VSCO-2-CE"
VCSL = CACHE / "VCSL"


# ---------------------------------------------------------------- modal bodies

def modal(excitation, freqs, decays, gains):
    """Bank of two-pole resonators. decays = 60 dB decay times in seconds."""
    out = np.zeros(len(excitation))
    for f, t60, g in zip(freqs, decays, gains):
        if f >= RATE / 2 - 200:
            continue
        r = np.exp(-6.91 / (t60 * RATE))
        w = 2 * np.pi * f / RATE
        out += signal.lfilter([g * np.sin(w)], [1, -2 * r * np.cos(w), r * r], excitation)
    return out


def contact(r, ms=1.5, hardness=1.0):
    """Force pulse of a contact: soft (knuckle, heel) = long and rounded, hard (metal, chalk) = short."""
    n = max(4, int(ms / 1000 * RATE))
    pulse = np.sin(np.linspace(0, np.pi, n)) ** (1.5 / hardness)
    return pulse * r.uniform(0.8, 1.0)


def body_modes(r, base, ratios, t60, spread=0.03, count=None):
    ratios = np.asarray(ratios, float)
    freqs = base * ratios * (1 + r.normal(0, spread, len(ratios)))
    decays = np.asarray(t60, float) * r.uniform(0.8, 1.2, len(ratios))
    gains = r.uniform(0.3, 1.0, len(ratios)) / np.sqrt(np.arange(1, len(ratios) + 1))
    return freqs, decays, gains


# ---------------------------------------------------------------- recorded material

@lru_cache(maxsize=None)
def sample(rel, stereo=False):
    return load(CACHE / rel, stereo=stereo)


def metal_hits():
    return sorted(glob.glob(str(VSCO / "Miscellania Raw" / "*" / "metal_hit*.wav")))


def real(rel_glob):
    return sorted(glob.glob(str(CACHE / rel_glob)))


# ---------------------------------------------------------------- footsteps

SURFACES = {
    # floor body modes (Hz), decays, heel click brightness, grit probability, creak probability
    "lino": dict(modes=[68, 112, 190, 310, 520], t60=[0.09, 0.07, 0.05, 0.04, 0.03], click=4200, grit=0.35, creak=0.0,
                 squeak=0.07),
    "wood": dict(modes=[95, 160, 240, 380, 610, 900], t60=[0.16, 0.12, 0.1, 0.07, 0.05, 0.04], click=3000, grit=0.2,
                 creak=0.3, squeak=0.0),
    "stone": dict(modes=[55, 140, 260, 470], t60=[0.06, 0.05, 0.04, 0.03], click=6000, grit=0.55, creak=0.0,
                  squeak=0.02),
}


def footstep(seed, surface="lino", weight=1.0, pace=1.0):
    """Heel strike, roll, toe-off, scuff; on old surfaces debris crunch and floorboard creak."""
    s = SURFACES[surface]
    r = rng("step", surface, seed)
    n = sec(0.75)
    exc = np.zeros(n)
    heel = contact(r, r.uniform(2.0, 3.5), 0.8) * weight
    exc[:len(heel)] += heel
    roll = int(r.uniform(0.07, 0.13) / pace * RATE)
    toe = contact(r, r.uniform(1.2, 2.5), 0.9) * weight * r.uniform(0.35, 0.6)
    exc[roll:roll + len(toe)] += toe
    f, d, g = body_modes(r, 1.0, s["modes"], s["t60"], 0.06)
    thud = modal(exc, f, d, g)
    out = thud / (np.max(np.abs(thud)) + 1e-9)
    # sole click: the hard edge of a heel on a hard floor
    click = hp(exc * r.standard_normal(n), s["click"] * 0.5, 2)
    out += click / (np.max(np.abs(click)) + 1e-9) * 0.25
    # scuff: sole sliding as weight transfers
    sc_len = sec(r.uniform(0.04, 0.12))
    sc_at = roll + sec(r.uniform(0.0, 0.03))
    scuff = bp(r.standard_normal(sc_len), 700, 6500) * np.hanning(sc_len) ** 0.7
    out[sc_at:sc_at + sc_len] += scuff * r.uniform(0.04, 0.12)
    if r.random() < s["grit"]:  # plaster grit and glass crumbs under the sole
        for _ in range(int(r.integers(4, 22))):
            at = int(r.uniform(0, 0.16) * RATE)
            m = sec(r.uniform(0.0008, 0.003))
            grain = hp(r.standard_normal(m), r.uniform(2500, 6000)) * np.hanning(m)
            out[at:at + m] += grain * r.uniform(0.03, 0.2)
    if r.random() < s["squeak"]:  # rubber on lino
        m = sec(r.uniform(0.02, 0.06))
        t = np.arange(m) / RATE
        hz = r.uniform(1800, 3200) * (1 + 0.15 * t / t[-1])
        sq = np.sin(2 * np.pi * np.cumsum(hz) / RATE) * np.hanning(m)
        out[roll:roll + m] += sq * r.uniform(0.03, 0.08)
    if r.random() < s["creak"]:
        cr = floor_creak(r.integers(1 << 30), short=True)
        out[sec(0.02):sec(0.02) + len(cr)] += cr[:n - sec(0.02)] * r.uniform(0.1, 0.3)
    return fade(out * 0.6, 0.0005, 0.08)


def walk(count, surface="lino", interval=0.56, seed="walk", weight=1.0, start_foot=0, jitter=0.03, slow_to=None):
    """A walk as (onset_seconds, mono_buffer) pairs; gait has left/right asymmetry and drifts in tempo."""
    r = rng("walk", seed)
    t = 0.0
    out = []
    asym = r.uniform(0.01, 0.04)
    for i in range(count):
        step_interval = interval
        if slow_to is not None:
            step_interval = interval + (slow_to - interval) * (i / max(1, count - 1)) ** 2
        foot = (i + start_foot) % 2
        out.append((t, footstep(int(r.integers(1 << 30)), surface, weight * (1 + (0.1 if foot else -0.05)))))
        t += step_interval * (1 + (asym if foot else -asym)) * (1 + r.normal(0, jitter))
    return out


def floor_creak(seed, short=False):
    """Floorboard creak: stick-slip between boards at a rate that wanders with load, exciting board resonances
    that slide as the contact point moves. Pitched friction, not a buzz."""
    r = rng("creak", seed)
    dur = r.uniform(0.15, 0.4) if short else r.uniform(0.6, 1.8)
    n = sec(dur)
    rate = 40 + 110 * smooth_noise(r, n, dur / 4, 0, 1) ** 1.5
    phase = np.cumsum(rate / RATE)
    pulses = np.zeros(n)
    idx = np.where(np.diff(np.floor(phase)) > 0)[0]
    pulses[idx] = r.uniform(0.4, 1.0, len(idx)) * (0.6 + 0.4 * np.sin(np.pi * idx / n))
    out = np.zeros(n)
    centre = r.uniform(300, 700) * (1 + 0.25 * np.linspace(-1, 1, n) * r.choice([-1, 1]))
    # slowly moving resonance approximated by crossfading three fixed banks
    for k, frac in enumerate((0.0, 0.5, 1.0)):
        c = float(np.interp(frac, [0, 1], [centre[0], centre[-1]]))
        w = np.clip(1 - np.abs(np.linspace(0, 1, n) - frac) * 2, 0, 1)
        out += modal(pulses, [c, c * 2.03, c * 3.1, c * 0.51], [0.05, 0.04, 0.03, 0.08], [1, .5, .25, .6]) * w
    return fade(out / (np.max(np.abs(out)) + 1e-9) * 0.5, 0.01, 0.05)


# ---------------------------------------------------------------- knocks, pipes, doors

THREE = (0.0, 0.34, 0.97)    # the motif: knock . knock ...... knock   (it never changes; that's the point)


def knock(seed, material="door", force=1.0):
    r = rng("knock", material, seed)
    n = sec(1.2 if material == "pipe" else 0.5)
    exc = np.zeros(n)
    if material == "door":       # knuckle on a hollow-core classroom door
        k = contact(r, r.uniform(1.4, 2.2), 0.7) * force
        exc[:len(k)] = k
        f, d, g = body_modes(r, 1.0, [88, 143, 231, 317, 455, 610, 822, 1180, 1590],
                             [0.12, 0.1, 0.08, 0.07, 0.05, 0.045, 0.035, 0.025, 0.02], 0.05)
        y = modal(exc, f, d, g)
        y = y / (np.max(np.abs(y)) + 1e-9)
        y += hp(exc * r.standard_normal(n), 2500) * 0.15            # knuckle skin
        rattle_at = sec(r.uniform(0.008, 0.02))                       # the latch rattles in its keeper
        m = sec(0.012)
        y[rattle_at:rattle_at + m] += bp(r.standard_normal(m), 2000, 7000) * np.exp(-np.arange(m) / 90) * 0.12 * force
    elif material == "pipe":     # water hammer / thermal knock in a cast-iron heating pipe
        k = contact(r, r.uniform(0.6, 1.0), 1.5) * force
        exc[:len(k)] = k
        base = r.uniform(310, 380)
        f, d, g = body_modes(r, base, [1.0, 2.76, 5.40, 8.93, 13.3], [0.9, 0.6, 0.4, 0.25, 0.15], 0.01)
        ring = modal(exc, f, d, g)
        thump = modal(exc, [58, 97], [0.12, 0.08], [1.0, 0.5])
        y = ring / (np.max(np.abs(ring)) + 1e-9) * 0.6 + thump / (np.max(np.abs(thump)) + 1e-9)
    elif material == "wall":     # something knocking from inside plaster on brick
        k = contact(r, r.uniform(3, 5), 0.6) * force
        exc[:len(k)] = k
        y = modal(exc, [72, 118, 180, 260], [0.07, 0.05, 0.04, 0.03], [1, .7, .4, .2])
        y = y / (np.max(np.abs(y)) + 1e-9)
    else:
        raise ValueError(material)
    return fade(y * 0.6, 0.0003, 0.05)


def knock_pattern(seed, material="door", count=3, force=1.0, timing=THREE, tempo=1.0):
    r = rng("pattern", material, seed)
    total = sec(timing[-1] / tempo + 1.5)
    out = np.zeros(total)
    for i in range(count):
        at = sec(timing[i] / tempo * (1 + r.normal(0, 0.015)))
        k = knock(int(r.integers(1 << 30)), material, force * (1.0, 0.85, 1.1)[i % 3] * r.uniform(0.9, 1.05))
        out[at:at + len(k)] += k[:total - at]
    return out


def door_creak(seed, dur=None, pitch=1.0):
    """Hinge creak: stick-slip at hundreds of Hz - genuinely pitched, gliding with the door's speed."""
    r = rng("hinge", seed)
    dur = dur or r.uniform(1.0, 2.6)
    n = sec(dur)
    speed = smooth_noise(r, n, dur / 3, 0.2, 1.0) * np.sin(np.pi * np.linspace(0, 1, n)) ** 0.6
    rate = (90 + 380 * speed) * pitch * (1 + 0.04 * smooth_noise(r, n, 0.03, -1, 1))
    phase = np.cumsum(rate / RATE)
    pulses = np.zeros(n)
    idx = np.where(np.diff(np.floor(phase)) > 0)[0]
    pulses[idx] = speed[idx] * r.uniform(0.6, 1.0, len(idx))
    y = modal(pulses, [620, 1340, 2210, 3130, 4400], [0.03, 0.025, 0.02, 0.015, 0.01], [1, .8, .6, .4, .2])
    y += modal(pulses, [110, 180], [0.15, 0.1], [0.6, 0.4])          # the door leaf itself
    return fade(y / (np.max(np.abs(y)) + 1e-9) * 0.5, 0.02, 0.1)


def door_slam(seed, force=1.0):
    r = rng("slam", seed)
    n = sec(1.5)
    exc = np.zeros(n)
    k = contact(r, 6.0, 0.7) * force
    exc[:len(k)] = k
    f, d, g = body_modes(r, 1.0, [52, 77, 121, 168, 240, 330, 470], [0.25, 0.2, 0.15, 0.12, 0.09, 0.07, 0.05], 0.04)
    y = modal(exc, f, d, g)
    y = y / (np.max(np.abs(y)) + 1e-9)
    latch_at = sec(r.uniform(0.01, 0.03))
    m = sec(0.04)
    latch = modal(np.r_[contact(r, 0.3, 2), np.zeros(m)], [2100, 3700, 5200], [0.03, 0.02, 0.015], [1, .6, .3])
    y[latch_at:latch_at + len(latch)] += latch / (np.max(np.abs(latch)) + 1e-9) * 0.3
    # glass in the door's wired window rattles
    gl = bp(r.standard_normal(sec(0.3)), 2500, 8000) * np.exp(-np.arange(sec(0.3)) / sec(0.05)) * 0.08
    y[sec(0.005):sec(0.005) + len(gl)] += gl
    return fade(y * 0.8, 0.0005, 0.2)


def locker(seed, kind="hit"):
    """Steel lockers: real struck-metal recordings as the body."""
    r = rng("locker", seed)
    hits = metal_hits()
    x = sample(hits[int(r.integers(len(hits)))].replace(str(CACHE) + "/", ""))
    x = x[:sec(2.5)]
    if kind == "rattle":
        out = np.zeros(sec(1.2))
        t = 0
        for i in range(int(r.integers(3, 7))):
            g = 0.5 ** i * r.uniform(0.5, 1)
            seg = x[:sec(0.4)] * g
            out[sec(t):sec(t) + len(seg)] += seg[:len(out) - sec(t)]
            t += r.uniform(0.03, 0.09) * (0.8 ** i)
        x = out
    x = lp(x, r.uniform(3000, 6000)) if kind != "hit" else x
    return fade(x / (np.max(np.abs(x)) + 1e-9) * 0.6, 0.0005, 0.1)


# ---------------------------------------------------------------- classroom objects

def chalk(seed, seconds=6.0, words=None, pressure=1.0):
    """Someone writing on a blackboard: strokes, dots, lifts; friction grain with stick-slip 'catches', the rare
    tonal squeal, and the slate-and-frame body. The rhythm is handwriting, so it reads as language."""
    r = rng("chalk", seed)
    n = sec(seconds)
    out = np.zeros(n)
    t = r.uniform(0.05, 0.3)
    letter = 0
    while t < seconds - 0.3:
        kind = r.random()
        if kind < 0.12:                                   # a dot / full stop: a tap
            m = sec(0.03)
            tap = contact(r, 0.5, 1.8)
            seg = np.zeros(m)
            seg[:len(tap)] = tap
            out[sec(t):sec(t) + m] += seg * 2.0
            t += r.uniform(0.08, 0.2)
            continue
        dur = r.uniform(0.07, 0.38)
        m = sec(dur)
        env = np.sin(np.pi * np.linspace(0, 1, m)) ** 0.5 * pressure
        speed = 0.5 + 0.5 * np.sin(np.pi * np.linspace(0, 1, m) * r.uniform(1, 2))
        grain = r.standard_normal(m)
        slip_rate = r.uniform(90, 400) * (0.6 + speed)
        slip = (np.sin(2 * np.pi * np.cumsum(slip_rate) / RATE) > r.uniform(0.2, 0.8)).astype(float)
        stroke = bp(grain * (0.35 + 0.65 * slip), 1200, 7500) * env
        if r.random() < 0.08:                             # the squeal everybody hates (kept rare and short)
            hz = r.uniform(1900, 3400) * (1 + 0.04 * np.sin(2 * np.pi * r.uniform(5, 11) * np.arange(m) / RATE))
            stroke += np.sin(2 * np.pi * np.cumsum(hz) / RATE) * env * r.uniform(0.15, 0.35)
        out[sec(t):sec(t) + m] += stroke[:n - sec(t)]
        t += dur + r.uniform(0.02, 0.09)
        letter += 1
        if letter % int(r.integers(3, 7)) == 0:            # end of a word
            t += r.uniform(0.25, 0.7)
    board = modal(out, [210, 340, 520, 760, 1150], [0.08, 0.07, 0.05, 0.04, 0.03], [1, .8, .5, .4, .3])
    out = out + board / (np.max(np.abs(board)) + 1e-9) * np.max(np.abs(out)) * 0.4
    return fade(out / (np.max(np.abs(out)) + 1e-9) * 0.5, 0.01, 0.05)


def chair_drag(seed, seconds=2.5, chair="metal"):
    """A tubular-steel school chair dragged across a floor: each of the legs judders (stick-slip at 20-70 Hz,
    rate following how hard it is pulled), exciting the frame; occasional catches. Heard through a ceiling this
    becomes the thing everybody recognises and nobody wants to hear above them."""
    r = rng("chair", seed)
    n = sec(seconds)
    pull = smooth_noise(r, n, seconds / 3, 0.3, 1.0) * np.sin(np.pi * np.linspace(0, 1, n)) ** 0.3
    out = np.zeros(n)
    for leg in range(int(r.integers(2, 4))):
        rate = (18 + 52 * pull) * r.uniform(0.85, 1.15)
        phase = np.cumsum(rate / RATE) + r.random()
        pulses = np.zeros(n)
        idx = np.where(np.diff(np.floor(phase)) > 0)[0]
        pulses[idx] = pull[idx] * r.uniform(0.3, 1.0, len(idx))
        # catches: the leg sticks, then releases with a bigger hit
        for c in r.uniform(0.15, 0.9, int(r.integers(0, 3))) * n:
            c = int(c)
            pulses[c:c + sec(0.08)] = 0
            pulses[min(n - 1, c + sec(0.08))] = 2.0
        f, d, g = body_modes(r, 1.0, [310, 720, 1290, 2150, 3300], [0.12, 0.08, 0.06, 0.04, 0.03], 0.05)
        frame = modal(pulses, f, d, g)
        floor = modal(pulses, [62, 104, 170], [0.1, 0.07, 0.05], [1, .6, .3])
        out += frame / (np.max(np.abs(frame)) + 1e-9) * 0.6 + floor / (np.max(np.abs(floor)) + 1e-9)
    grit = bp(r.standard_normal(n), 800, 5000) * pull * 0.08
    return fade((out / (np.max(np.abs(out)) + 1e-9) + grit) * 0.5, 0.03, 0.15)


def clock_tick(seed, tock=False):
    """Quartz wall clock: the second-hand stepper's click and a tiny rebound, inside a plastic case."""
    r = rng("tick", seed, tock)
    n = sec(0.12)
    exc = np.zeros(n)
    c = contact(r, 0.12, 2.0)
    exc[:len(c)] = c
    reb = sec(r.uniform(0.006, 0.011))
    c = contact(r, 0.08, 2.0)
    exc[reb:reb + len(c)] += c * r.uniform(0.2, 0.4)
    base = 1.0 if not tock else 0.93
    f, d, g = body_modes(r, base, [1180, 2650, 3900, 5600, 820], [0.012, 0.009, 0.007, 0.005, 0.015], 0.01)
    y = modal(exc, f, d, g)
    y = y / (np.max(np.abs(y)) + 1e-9) * 0.5 + hp(exc * r.standard_normal(n), 3000) * 0.5   # the click itself
    return fade(y / (np.max(np.abs(y)) + 1e-9) * 0.4, 0.0001, 0.02)


def clock(seconds, seed="clock", skip_at=None, stop_at=None, stutter_at=None):
    """A second hand. skip_at: a tick goes missing. stutter_at: two ticks close together (it hesitates).
    stop_at: it stops."""
    out = np.zeros(sec(seconds) + sec(0.2))
    t, i = 0.0, 0
    skips = set(np.round(np.atleast_1d(skip_at), 0)) if skip_at is not None else set()
    while t < seconds and (stop_at is None or t < stop_at):
        if round(t) not in skips:
            tk = clock_tick(i % 7, tock=bool(i % 2))
            out[sec(t):sec(t) + len(tk)] += tk
        if stutter_at is not None and abs(t - stutter_at) < 0.5:
            tk = clock_tick(99, tock=True) * 0.7
            out[sec(t + 0.19):sec(t + 0.19) + len(tk)] += tk
        t += 1.0
        i += 1
    return out[:sec(seconds)]


def drip(seed, vessel="bucket"):
    """A drop into shallow water in a metal bucket: impact, the bubble's rising Minnaert chirp, the bucket ring."""
    r = rng("drip", vessel, seed)
    n = sec(1.2)
    y = np.zeros(n)
    f0 = r.uniform(900, 2300)
    m = sec(r.uniform(0.02, 0.05))
    t = np.arange(m) / RATE
    hz = f0 * (1 + r.uniform(0.15, 0.6) * t / t[-1])
    bubble = np.sin(2 * np.pi * np.cumsum(hz) / RATE) * np.exp(-t / (m / RATE / 3))
    y[sec(0.002):sec(0.002) + m] += bubble * r.uniform(0.4, 1.0)
    y[:sec(0.002)] += hp(r.standard_normal(sec(0.002)), 3000) * 0.3
    if vessel == "bucket":
        exc = np.zeros(n)
        exc[:8] = 0.2
        ring = modal(exc, [420, 1180, 2230, 3540, 5010], [0.9, 0.7, 0.5, 0.35, 0.25], [.5, 1, .7, .4, .2])
        y += ring / (np.max(np.abs(ring)) + 1e-9) * r.uniform(0.04, 0.09)
    return fade(y * 0.5, 0.0001, 0.2)


# ---------------------------------------------------------------- the school bell

def _gong():
    """Impulse response of the bell's steel gong: a real struck brake drum (VCSL), high-passed, which has the
    clangorous, inharmonic partial structure of a cast bell gong."""
    x = sample("VCSL/Idiophones/Struck Idiophones/Brake Drum/BrakeDrum2_Hammer3_v2_rr1_Mid.wav")
    on = int(np.argmax(np.abs(x) > 0.2 * np.abs(x).max()))
    x = x[on:on + sec(1.6)]
    x = zhp(x, 450, 2)
    from .core import resample
    x = resample(x, 1.45)                     # smaller gong, higher partials
    return x / (np.max(np.abs(x)) + 1e-9)


def school_bell(seed, ring=3.0, rate=24.0, decay="natural", tail=4.0):
    """Electric bell: a solenoid hammer hitting the gong ~24 times a second. decay='natural' rings off in about
    a second; decay='wrong' keeps singing after the hammer stops, swelling and sagging in pitch, as if the bell
    were being held at the edge of its note by something else."""
    r = rng("bell", seed)
    g = _gong()
    n = sec(ring + tail + 2)
    strikes = np.zeros(n)
    t = 0.0
    while t < ring:
        strikes[sec(t)] += r.uniform(0.6, 1.0) * min(1.0, t / 0.08 + 0.3)
        if r.random() < 0.15:                                 # hammer bounce
            strikes[sec(t + 0.004)] += r.uniform(0.1, 0.3)
        t += (1.0 / rate) * (1 + r.normal(0, 0.03))
    y = conv(strikes, g)[:n]
    solenoid = bp(np.sign(np.sin(2 * np.pi * rate * np.arange(n) / RATE)) * (np.arange(n) < sec(ring)), 180, 1400)
    y = y / (np.max(np.abs(y)) + 1e-9) + 0.012 * solenoid
    if decay == "wrong":
        from .core import varispeed
        hold = y[sec(ring - 0.6):sec(ring - 0.1)]
        # spectral freeze of the last half second of ringing -> a sustained, unnatural bell tone
        frozen = _spectral_freeze(hold, sec(tail + 1), r)
        sag = 1.0 - 0.018 * np.linspace(0, 1, len(frozen)) ** 1.5
        frozen = varispeed(np.pad(frozen, (0, sec(1))), sag)[:sec(tail)]
        env = np.interp(np.arange(len(frozen)), [0, sec(0.4), sec(tail * 0.6), sec(tail)], [0.3, 0.55, 0.75, 0.0])
        a = sec(ring - 0.2)
        y[a:a + len(frozen)] += frozen * env
    return fade(y / (np.max(np.abs(y)) + 1e-9) * 0.7, 0.005, 0.3)


def _spectral_freeze(x, n, r, nfft=8192):
    """Keep the magnitude spectrum of x, randomise phase per frame, overlap-add: an endless sustain of a moment."""
    mag = np.abs(np.fft.rfft(x[:nfft] * np.hanning(min(nfft, len(x))), nfft))
    hop = nfft // 4
    out = np.zeros(n + nfft)
    win = np.hanning(nfft)
    for a in range(0, n, hop):
        ph = np.exp(1j * r.uniform(0, 2 * np.pi, len(mag)))
        out[a:a + nfft] += np.fft.irfft(mag * ph, nfft) * win
    out = out[:n]
    return out / (np.max(np.abs(out)) + 1e-9)


spectral_freeze = _spectral_freeze
