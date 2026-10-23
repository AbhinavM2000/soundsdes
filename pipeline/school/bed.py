"""Environmental beds from real field recordings, re-shaped for the school.

All beds return stereo (2, n) at a nominal level of about -30 dBFS RMS so the mixer, not the source recording,
decides how loud each is. `loop=True` returns a seamless loop of exactly `seconds`.
"""
import numpy as np

from .core import (RATE, bp, db, hp, lp, peak_eq, rms_db, rng, sec, seamless_loop, smooth_noise, source, zhp)
from .space import spread

FIELD = "field/"
REC = {
    "rain_window": ("RegenamFenster090120251515.mp3", 20.0, 1140.0),
    "rain_window2": ("1602140001.mp3", 2.0, 96.0),
    "rain_factory": ("marchienne071012h15.mp3", 2.0, 222.0),
    "storm_roof": ("060319puertopercyloosemetalroof.mp3", 2.0, 122.0),
    "abandoned": ("insidetheabandonedbuilding.wav", 2.0, 210.0),
    "factory_hall": ("20131210intrieurgrandhall2photographes.mp3", 2.0, 294.0),
    "corridor": ("14040302CouloirsERG.mp3", 2.0, 265.0),
    "stairwell": ("LiepajaUniversitystairwell.mp3", 2.0, 178.0),
    "stair_wind": ("windstormstairwell.mp3", 2.0, 137.0),
    "tower_wind": ("Carraigefactorytowerwindy.mp3", 2.0, 152.0),
    "slit_wind": ("matewind.mp3", 2.0, 240.0),
    "duct": ("UmeaReklamABHemvagen.mp3", 2.0, 146.0),
    "shaft": ("CityCountryMejaegerstrlueftungsschacht.mp3", 2.0, 120.0),
    "hangar_vent": ("angarasgrotos.wav", 2.0, 158.0),
    "substation": ("ElektrosPastotesGausmasAkademija.wav", 2.0, 186.0),
    "transformer_rain": ("electricalhumfoxesinrainedit.mp3", 1.0, 59.0),
    "radiator": ("rad27.mp3", 2.0, 296.0),
    "radiator2": ("rad.mp3", 2.0, 118.0),
    "gym": ("DusseldorfGGSTurnhalle.mp3", 2.0, 300.0),
    "yard": ("250214005.WAV", 2.0, 298.0),
    "yard2": ("CappamoreFieldrecording.wav", 2.0, 101.0),
    "tap_drip": ("hotelroomtapdripeditJan12012.mp3", 2.0, 410.0),
}


def raw(name, seconds, seed=0, offset=None):
    """A stretch of a real recording, starting at a seeded random offset; wraps if needed."""
    f, lo, hi = REC[name]
    r = rng("bed", name, seed)
    full = source(FIELD + f, stereo=True)
    hi = min(hi, full.shape[1] / RATE - 0.5)
    n = sec(seconds)
    if offset is None:
        offset = r.uniform(lo, max(lo, hi - seconds)) if hi - lo > seconds else lo
    a = sec(offset)
    seg = full[:, a:a + n]
    while seg.shape[1] < n:                       # recording shorter than requested: continue elsewhere
        b = sec(r.uniform(lo, max(lo + 1, hi - 10)))
        seg = np.hstack([seg, full[:, b:b + n - seg.shape[1]]])
    seg = zhp(seg, 28, 2)
    return seg - seg.mean(axis=1, keepdims=True)


def level(x, target=-30.0):
    return x * db(target - rms_db(x))


def _loop(builder, seconds, loop, crossfade=4.0):
    if not loop:
        return builder(seconds)
    body = builder(seconds + crossfade)
    return seamless_loop(body, sec(seconds), crossfade)


# ---------------------------------------------------------------- weather

def rain(seconds, seed=0, where="classroom", loop=False):
    """Rain as heard from inside: glass, sills, and gutters overflowing somewhere."""
    def build(s):
        a = raw("rain_window", s, seed)
        b = raw("rain_factory", s, seed + 1)
        if where == "classroom":
            mix = level(a, -30) + level(lp(b, 3500), -38)
        elif where == "corridor":                 # windows further away, along one side
            mix = level(lp(a, 2600), -36) + level(lp(b, 2200), -37)
        elif where == "stairwell":                # high broken window: the roof and gutters
            mix = level(lp(a, 3000), -38) + level(raw("storm_roof", s, seed), -36) + level(b, -36)
        elif where == "hall":
            mix = level(lp(raw("storm_roof", s, seed), 2500), -36) + level(lp(b, 2000), -38)
        else:
            mix = level(a, -30)
        return mix
    return _loop(build, seconds, loop)


def wind(seconds, seed=0, kind="slit", loop=False):
    """'slit': a whistling draught through a broken window frame - tonal bands that wander like vowels."""
    def build(s):
        name = {"slit": "slit_wind", "stair": "stair_wind", "tower": "tower_wind"}[kind]
        x = raw(name, s, seed)
        if kind == "slit":
            x = hp(x, 180)
        return level(x, -32)
    return _loop(build, seconds, loop)


# ---------------------------------------------------------------- the building's machinery

def room_tone(seconds, seed=0, kind="classroom", loop=False):
    """The sound of a large empty building at night: very low, mostly below 400 Hz, with distant settling."""
    def build(s):
        if kind in ("classroom", "music_room"):
            x = level(lp(raw("abandoned", s, seed), 2500), -40)
        elif kind == "corridor":
            x = level(lp(raw("abandoned", s, seed), 1800), -40) + level(lp(raw("factory_hall", s, seed), 1200), -44)
        elif kind == "stairwell":
            x = level(lp(raw("stairwell", s, seed), 3000), -40) + level(lp(raw("factory_hall", s, seed + 3), 900), -44)
        elif kind == "hall":
            x = level(lp(raw("factory_hall", s, seed), 2500), -38)
        else:
            x = level(lp(raw("abandoned", s, seed), 2500), -40)
        return x
    return _loop(build, seconds, loop)


def hum(seconds, seed=0, flicker=0.3, loop=False, dropouts=()):
    """One fluorescent fitting still alive on a dying ballast: real substation hum (an actual 50 Hz harmonic
    series with its natural jitter) narrowed to a fitting-sized source, plus the ballast's buzz and the tube's
    irregular flicker. dropouts: [(start, length)] in seconds where the light dies."""
    r = rng("hum", seed)

    def build(s):
        x = raw("substation", s, seed).mean(axis=0)
        x = bp(x, 95, 2600, 2)
        x = peak_eq(x, 100, 4.0, 4.0)
        x = peak_eq(x, 1200, 3.0, 1.0)
        n = len(x)
        t = np.arange(n) / RATE
        buzz = np.sign(np.sin(2 * np.pi * 100 * t)) * (0.5 + 0.5 * smooth_noise(r, n, 0.6, 0, 1))
        x = x / (np.std(x) + 1e-9) + 0.05 * bp(buzz, 800, 6000)
        # flicker: brief irregular dips with a tick when the arc re-strikes
        env = np.ones(n)
        tt = r.uniform(0.5, 6.0)
        while tt < s - 1:
            if r.random() < flicker:
                d = r.uniform(0.04, 0.4)
                a = sec(tt)
                env[a:a + sec(d)] *= r.uniform(0.05, 0.5)
                tick = hp(r.standard_normal(sec(0.004)), 2500) * np.hanning(sec(0.004)) * 3
                x[a + sec(d):a + sec(d) + len(tick)] += tick[:max(0, n - a - sec(d))]
            tt += r.exponential(4.0) + 0.3
        for a, d in dropouts:
            env[sec(a):sec(a + d)] = 0.0
        env = np.convolve(env, np.ones(96) / 96, mode="same")
        return level(x * env, -30)

    y = build(seconds) if not loop else seamless_loop(build(seconds + 4.0), sec(seconds), 4.0)
    return y


def vent(seconds, seed=0, breathing=0.0, rate=0.23, loop=False):
    """Ventilation through sheet-metal ducts. breathing (0..1) imposes a human respiration pattern - inhale shorter
    and brighter than the exhale - until the building appears to breathe."""
    r = rng("vent", seed)

    def build(s):
        x = raw("duct", s, seed).mean(axis=0) * 0.6 + raw("shaft", s, seed).mean(axis=0) * 0.4
        x = bp(x, 60, 5000, 2)
        n = len(x)
        if breathing > 0:
            t = np.arange(n) / RATE
            phase = np.cumsum(rate * (1 + 0.1 * smooth_noise(r, n, 3.0, -1, 1))) / RATE
            frac = phase % 1.0
            inhale = np.where(frac < 0.4, np.abs(np.sin(np.pi * np.minimum(frac, 0.4) / 0.4)) ** 1.5, 0)
            exhale = np.where(frac >= 0.45, np.abs(np.sin(np.pi * np.clip(frac - 0.45, 0, 0.55) / 0.55)) ** 2.2, 0)
            bright = hp(x, 1200)
            breathy = x * (0.25 + 0.75 * exhale) + bright * inhale * 1.3
            x = x * (1 - breathing) + breathy * breathing
        return level(spread(x, 0.6, "vent%d" % seed), -32)

    return build(seconds) if not loop else seamless_loop(build(seconds + 4.0), sec(seconds), 4.0)


def radiator(seconds, seed=0, loop=False):
    def build(s):
        return level(bp(raw("radiator", s, seed), 80, 4000), -38)
    return _loop(build, seconds, loop)


# ---------------------------------------------------------------- memory of people

def yard(seconds, seed=0, loop=False):
    """Real children at break-time. Used only as memory: through glass, through PA, through walls."""
    def build(s):
        return level(raw("yard", s, seed) * 0.6 + raw("yard2", s, seed) * 0.4, -30)
    return _loop(build, seconds, loop)


def gym(seconds, seed=0, loop=False):
    def build(s):
        return level(raw("gym", s, seed), -32)
    return _loop(build, seconds, loop)


def tap_drips(seconds=410.0):
    """The real tap-drip recording, for harvesting individual drops."""
    return raw("tap_drip", seconds, 0, offset=2.0)
