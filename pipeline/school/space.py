"""Spatial engine: HRTF binaural rendering, distance, near field, occlusion, and the rooms of the school.

Conventions: az in degrees, 0 = straight ahead, +90 = right, 180 = behind, -90 = left. el in degrees, + = up.
Distances in metres. Every positioned source is mono; every rendered result is binaural stereo (2, n).

Rooms are hybrid BRIRs: the first ~120 ms come from real dummy-head measurements (Aachen AIR: stairway, lecture
room, Aula Carolina, office), which carry the irregular early reflections that make a space believable; after that a
synthetic late tail with per-octave decay times takes over, so each school space gets exactly the decay we want
(and the corridor gets its flutter echo) without amplifying measurement noise.
"""
from functools import lru_cache

import h5py
import numpy as np
import scipy.io as sio
from scipy import signal

from .core import CACHE, RATE, bp, conv, db, hp, lp, peak_eq, rng, sec, shelf, stereo

SPEED = 343.0
OCTAVES = [125, 250, 500, 1000, 2000, 4000, 8000]
COHERENCE = [0.92, 0.75, 0.35, 0.10, 0.04, 0.02, 0.01]   # diffuse-field interaural coherence per octave


# ---------------------------------------------------------------- HRTF

@lru_cache(maxsize=1)
def _hrtf():
    with h5py.File(CACHE / "hrtf" / "mit_kemar_normal_pinna.sofa", "r") as f:
        ir = f["Data.IR"][:]
        pos = f["SourcePosition"][:]
    ir = signal.resample_poly(ir, 160, 147, axis=-1)          # 44.1k -> 48k
    az = np.radians(pos[:, 0])
    el = np.radians(pos[:, 1])
    vec = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=1)
    nfft = 2048
    spec = np.fft.rfft(ir, nfft, axis=-1)
    return vec, spec, nfft, ir.shape[-1]


def _direction(az, el):
    """Our clockwise azimuth -> SOFA unit vector (SOFA azimuth is counter-clockwise, +90 = left)."""
    a = np.radians(-np.asarray(az, dtype=float))
    e = np.radians(np.clip(el, -40, 90))
    return np.stack([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)], axis=-1)


def hrtf_index(az, el):
    vec = _hrtf()[0]
    return np.argmax(_direction(az, el) @ vec.T, axis=-1)


def _air_mask(freqs, dist):
    """Atmospheric absorption, ~0.1 dB/m at 8 kHz (20 C, 50 % RH)."""
    return db(-0.105 * np.power(freqs / 8000.0, 1.9)[None, :] * dist[:, None])


def _near_field(freqs, az, dist):
    """Below ~1 m the ear nearer the source gets much louder, mostly at low frequencies (the head stops being small
    relative to the distance), and proximity adds warmth. This is what makes a whisper sit *on* the ear."""
    amount = np.clip(1.0 / np.maximum(dist, 0.08) - 1.0, 0, 9) / 9.0          # 0 at >= 1 m, 1 at 10 cm
    lateral = np.sin(np.radians(az))                                          # +1 right, -1 left
    weight = 1.0 / (1.0 + (freqs / 1500.0) ** 2)                              # stronger at low frequencies
    ild = 10.0 * amount[:, None] * weight[None, :]
    right = ild * np.clip(lateral, 0, 1)[:, None] - 0.5 * ild * np.clip(-lateral, 0, 1)[:, None]
    left = ild * np.clip(-lateral, 0, 1)[:, None] - 0.5 * ild * np.clip(lateral, 0, 1)[:, None]
    warm = 5.0 * amount[:, None] * (1.0 / (1.0 + (freqs / 300.0) ** 2))[None, :]
    return db(left + warm), db(right + warm)


def binaural(x, az, el=0.0, dist=1.0, hop=512, propagation=True):
    """Render mono x at a fixed or moving position. az/el/dist may be scalars, arrays sampled per hop, or callables
    of time in seconds. Distance law is 1/r relative to 1 m (capped at 12 cm)."""
    vec, spec, nfft, hlen = _hrtf()
    n = len(x)
    frames = int(np.ceil(n / hop)) + 1
    t = np.arange(frames) * hop / RATE

    def curve(v):
        if callable(v):
            return np.asarray([v(tt) for tt in t], dtype=float)
        v = np.asarray(v, dtype=float)
        return np.full(frames, float(v)) if v.ndim == 0 else np.interp(np.arange(frames), np.linspace(0, frames - 1, len(v)), v)

    A, E, D = curve(az), curve(el), np.maximum(curve(dist), 0.12)
    idx = hrtf_index(A, E)
    freqs = np.fft.rfftfreq(nfft, 1 / RATE)
    gain = (1.0 / D)[:, None] * _air_mask(freqs, D)
    nl, nr = _near_field(freqs, A, D)
    win = np.hanning(2 * hop + 1)[:-1]
    padded = np.pad(x, (hop, 2 * hop + nfft))
    out = np.zeros((2, n + 2 * hop + nfft))
    for f in range(frames):
        a = f * hop
        seg = padded[a:a + 2 * hop] * win
        S = np.fft.rfft(seg, nfft) * gain[f]
        y = np.fft.irfft(np.stack([S * spec[idx[f], 0] * nl[f], S * spec[idx[f], 1] * nr[f]]), nfft)
        out[:, a:a + nfft] += y
    delay = int(D[0] / SPEED * RATE) if propagation else 0
    # HRIRs carry ~0.8 ms bulk delay; keep it so direct and reverb stay time-aligned.
    return np.pad(out[:, hop:hop + n + hlen], ((0, 0), (delay, 0)))


# ---------------------------------------------------------------- measured rooms

def _air(name, channel, *rest):
    tag = "_".join(str(r) for r in rest)
    data = sio.loadmat(CACHE / "rir" / "AIR_1_4" / ("air_binaural_%s_%d_1_%s.mat" % (name, channel, tag)))
    return data["h_air"][0].astype(np.float64)


def _air_pair(name, *rest):
    left, right = _air(name, 0, *rest), _air(name, 1, *rest)
    n = min(len(left), len(right))
    return np.vstack([left[:n], right[:n]])


AIR_SETS = {
    # name: [(distance_m, AIR key tuple), ...] ; stairway/aula carry an azimuth (90 = source to the side)
    "lecture": [(2.25, (1,)), (4.0, (2,)), (5.56, (3,)), (7.1, (4,)), (8.68, (5,)), (10.2, (6,))],
    "stairway": [(1.0, (1, 90)), (2.0, (2, 90)), (3.0, (3, 90))],
    "aula_carolina": [(1.0, (1, 90, 3)), (2.0, (2, 90, 3)), (3.0, (3, 90, 3)), (5.0, (4, 90, 3)),
                      (10.0, (5, 90, 3)), (15.0, (6, 90, 3)), (20.0, (7, 90, 3))],
    "office": [(1.0, (1,)), (2.0, (2,)), (3.0, (3,))],
}


class Room:
    """A school space. `decay` = target T60 per octave (125 Hz..8 kHz)."""

    def __init__(self, name, measured, decay, early_ms=120, flutter=None, tone=None, size=1.0, level=0.0):
        self.name, self.measured, self.decay = name, measured, np.asarray(decay, float)
        self.level = level  # dB: emptied of furniture and people, the school is wetter than the measured rooms
        self.early_ms, self.flutter, self.tone, self.size = early_ms, flutter, tone, size

    @lru_cache(maxsize=32)
    def tail(self, dist_index):
        """Stereo reverberant response (direct sound removed), scaled so that convolving a source with it gives the
        physically right reverb level relative to binaural(x, dist=1)."""
        measured_dist, key = AIR_SETS[self.measured][dist_index]
        h = _air_pair(self.measured, *key)
        onset = int(np.argmax(np.abs(h).max(axis=0) > 0.25 * np.abs(h).max()))
        direct_end = onset + sec(0.0025)
        direct_energy = np.sum(h[:, onset - sec(0.001):direct_end] ** 2) / 2
        early = h[:, direct_end:onset + sec(self.early_ms / 1000)].copy()
        r = rng("room-tail", self.name, dist_index)
        length = sec(float(self.decay.max()) * 1.15 + 0.2)
        late = self._late(r, length)
        # Match the synthetic tail to the measurement per octave over the 60 ms before the hand-over.
        ref = h[:, onset + sec(self.early_ms / 1000 - 0.06):onset + sec(self.early_ms / 1000)]
        start = sec(self.early_ms / 1000) - sec(0.06)
        matched = np.zeros_like(late)
        for i, f in enumerate(OCTAVES):
            lo, hi = f / 1.414, min(f * 1.414, 22000)
            band_late = bp(late, lo, hi, order=3)
            e_ref = np.mean(bp(ref, lo, hi, order=3) ** 2)
            e_syn = np.mean(band_late[:, start:start + sec(0.06)] ** 2) + 1e-20
            matched += band_late * np.sqrt(e_ref / e_syn)
        out = matched.copy()
        xf = sec(0.03)
        e_len = early.shape[1]
        ramp_out = np.ones(e_len)
        ramp_out[-xf:] = np.cos(np.linspace(0, np.pi / 2, xf)) ** 2
        out[:, :e_len] *= 0.0
        out[:, e_len - xf:e_len] = matched[:, e_len - xf:e_len] * (1 - ramp_out[-xf:])
        out[:, :e_len] += early * ramp_out
        if self.flutter:
            out += self._flutter(r, out.shape[1]) * np.sqrt(np.mean(matched[:, start:start + sec(0.06)] ** 2))
        if self.tone:
            for f, g, q in self.tone:
                out = peak_eq(out, f, g, q)
        # Reverb level: the measured file's own direct/reverb ratio at its own distance.
        scale = db(self.level) / (np.sqrt(direct_energy) * measured_dist)
        out *= scale
        delay = np.zeros((2, sec(measured_dist / SPEED)))
        return np.hstack([delay, out])

    def _late(self, r, n):
        t = np.arange(n) / RATE
        out = np.zeros((2, n))
        for i, f in enumerate(OCTAVES):
            lo, hi = f / 1.414, min(f * 1.414, 22000)
            common, side = r.standard_normal(n), r.standard_normal(n)
            c = COHERENCE[i]
            a, b = np.sqrt((1 + c) / 2), np.sqrt((1 - c) / 2)
            pair = np.vstack([a * common + b * side, a * common - b * side])
            env = np.exp(-6.91 * t / self.decay[i])
            out += bp(pair, lo, hi, order=3) * env
        return out

    def _flutter(self, r, n):
        """Flutter echo between the two long walls of a corridor: a pitched, decaying ring."""
        width, decay, level = self.flutter
        period = 2 * width / SPEED
        out = np.zeros((2, n))
        t = 0.004
        while t < decay * 1.2 and sec(t) < n:
            g = np.exp(-6.91 * t / decay)
            out[0, sec(t)] += g * r.uniform(0.7, 1.0)
            out[1, sec(t + r.uniform(0, 0.0004))] += g * r.uniform(0.7, 1.0)
            t += period * r.uniform(0.995, 1.005)
        return bp(out, 900, 5000) * level

    def tail_for(self, dist):
        dists = [d for d, _ in AIR_SETS[self.measured]]
        return self.tail(int(np.argmin(np.abs(np.log(np.asarray(dists) / max(dist, 0.3))))))


ROOMS = {
    "classroom": Room("classroom", "lecture", [0.95, 1.05, 1.15, 1.2, 1.1, 0.95, 0.7], level=6.0),
    "music_room": Room("music_room", "lecture", [0.85, 0.9, 0.95, 0.95, 0.9, 0.8, 0.6], level=4.0),
    "corridor": Room("corridor", "stairway", [1.4, 1.6, 1.8, 1.9, 1.7, 1.35, 0.9], flutter=(2.6, 0.9, 0.35),
                     tone=[(2600, 2.0, 1.2)], level=7.0),
    "stairwell": Room("stairwell", "stairway", [1.9, 2.1, 2.4, 2.5, 2.2, 1.75, 1.15], early_ms=150,
                      flutter=(3.2, 1.3, 0.25), level=7.0),
    "hall": Room("hall", "aula_carolina", [2.8, 2.7, 2.6, 2.4, 2.1, 1.6, 1.0], early_ms=180, level=8.0),
    "cupboard": Room("cupboard", "office", [0.5, 0.45, 0.4, 0.35, 0.3, 0.28, 0.2], early_ms=60),
}


# ---------------------------------------------------------------- placement

OCCLUSION = {
    # kind: (low-pass Hz, order, gain dB, high leak dB)  -- leak models the gap under a door
    "open": None,
    "door": (750.0, 2, -9.0, -32.0),
    "door_ajar": (2600.0, 1, -4.0, -20.0),
    "wall": (320.0, 4, -20.0, None),
    "floor": (180.0, 4, -16.0, None),         # structure-borne, from the floor above / below
    "window": (2200.0, 2, -7.0, -24.0),
    "duct": (1400.0, 2, -6.0, None),
}


def occlude(x, kind):
    spec = OCCLUSION[kind]
    if spec is None:
        return x
    f, order, g, leak = spec
    y = lp(x, f, order) * db(g)
    if leak is not None:
        y = y + hp(x, 3000, 2) * db(leak)
    if kind == "floor":
        y = shelf(y, 90, 4.0, high=False)
    if kind == "duct":
        # a sheet-metal duct: a handful of longitudinal resonances and a hollow mid
        for f0 in (180, 410, 690, 1130):
            y = peak_eq(y, f0, 5.0, 6.0)
    return y


def place(x, room="classroom", az=0.0, el=0.0, dist=2.0, wet=1.0, dry=1.0):
    """Source and listener in the same room."""
    r = ROOMS[room] if isinstance(room, str) else room
    d0 = float(dist if not callable(dist) and np.ndim(dist) == 0 else np.median(np.atleast_1d(
        [dist(t) for t in np.linspace(0, len(x) / RATE, 8)] if callable(dist) else dist)))
    direct = binaural(x, az, el, dist) * dry
    tail = conv(x, r.tail_for(d0)) * wet
    n = max(direct.shape[1], tail.shape[1])
    out = np.zeros((2, n))
    out[:, :direct.shape[1]] += direct
    out[:, :tail.shape[1]] += tail
    return out


def place_remote(x, src_room, src_dist, portal_az, portal_dist, occlusion, listener_room, el=0.0, listener_wet=0.35):
    """Source in another room: it excites its own room, the sound leaks through a portal (door, wall, floor) and
    reaches the listener from the portal's direction, then excites the listener's room a little."""
    r = ROOMS[src_room]
    direct = binaural(x, 30.0, 0.0, src_dist)[0]
    rev = conv(x, r.tail_for(src_dist))[0]
    inside = np.zeros(max(len(direct), len(rev)))                       # an 'ear' at the portal
    inside[:len(direct)] += direct
    inside[:len(rev)] += rev
    leaked = occlude(inside, occlusion)
    return place(leaked, listener_room, portal_az, el, portal_dist, wet=listener_wet)


def place_head(x, az=0.0, el=0.0, dist=0.15):
    """Dry, intimate placement with no room at all: physically impossible inside a reverberant building, which is
    exactly the point for the presence."""
    return binaural(x, az, el, dist)


def spread(x_stereo_or_mono, width=1.0, seed="spread"):
    """Wide diffuse stereo from mono (for distant beds); decorrelates with two short all-pass chains."""
    x = x_stereo_or_mono if x_stereo_or_mono.ndim == 1 else x_stereo_or_mono.mean(axis=0)
    r = rng("spread", seed)
    out = []
    for ch in range(2):
        y = x.copy()
        for _ in range(4):
            d = int(r.integers(40, 400))
            g = r.uniform(0.4, 0.7)
            b = np.zeros(d + 1)
            a = np.zeros(d + 1)
            b[0], b[-1] = -g, 1.0
            a[0], a[-1] = 1.0, -g
            y = signal.lfilter(b, a, y)
        out.append(y)
    mid = x
    return np.vstack([mid * (1 - width) + out[0] * width, mid * (1 - width) + out[1] * width])
