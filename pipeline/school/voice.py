"""Human presence. Every authored line starts as neural TTS (Kokoro) and is then re-performed through the WORLD
vocoder, which separates a voice into pitch (F0), vocal-tract envelope (what the mouth is doing) and aperiodicity
(how breathy it is). Keeping those apart is what lets the human quality survive the processing:

- whisper():   real articulation, excitation replaced by breath. Not formant-filtered noise.
- child():     pitch and vocal-tract length scaled together, the way a smaller body actually differs.
- freeze():    the mouth stops moving while the prosody carries on - a speaker losing the ability to form words.
- stretch():   time-stretch without changing pitch or formants - a voice beyond natural timing.
- sing():      speech re-pitched onto notes with sustained vowels - an assembly hymn sung by spoken voices.
"""
from functools import lru_cache

import numpy as np
import pyworld
from scipy import signal

from .core import CACHE, RATE, bp, db, fade, hp, lp, peak_eq, resample, rng, sec, shelf

FRAME_MS = 5.0


@lru_cache(maxsize=1)
def _kokoro():
    from kokoro_onnx import Kokoro
    return Kokoro(str(CACHE / "kokoro" / "kokoro-v1.0.onnx"), str(CACHE / "kokoro" / "voices-v1.0.bin"))


def voice_blend(*names_and_weights):
    """A new speaker from a weighted mix of style vectors, e.g. voice_blend(("bf_emma", .6), ("af_sky", .4))."""
    k = _kokoro()
    total = sum(w for _, w in names_and_weights)
    return sum(k.get_voice_style(n) * (w / total) for n, w in names_and_weights)


@lru_cache(maxsize=512)
def _tts_cached(text, voice, speed, lang):
    k = _kokoro()
    style = voice if isinstance(voice, str) else np.frombuffer(voice[1], dtype=np.float32).reshape(voice[0])
    audio, sr = k.create(text, voice=style, speed=speed, lang=lang)
    return signal.resample_poly(np.asarray(audio, dtype=np.float64), RATE // 1000, sr // 1000)


def tts(text, voice="bf_emma", speed=1.0, lang="en-gb", trim=True):
    key = voice if isinstance(voice, str) else (voice.shape, np.asarray(voice, np.float32).tobytes())
    x = _tts_cached(text, key, float(speed), lang).copy()
    if trim:
        env = np.abs(x) > 0.004
        if env.any():
            a = max(0, int(np.argmax(env)) - sec(0.03))
            b = min(len(x), len(x) - int(np.argmax(env[::-1])) + sec(0.08))
            x = x[a:b]
    return fade(x, 0.005, 0.02)


# ---------------------------------------------------------------- WORLD

class World:
    def __init__(self, x):
        x = np.ascontiguousarray(x, dtype=np.float64)
        self.n = len(x)
        f0, t = pyworld.harvest(x, RATE, f0_floor=60.0, f0_ceil=700.0, frame_period=FRAME_MS)
        self.f0 = pyworld.stonemask(x, f0, t, RATE)
        self.sp = pyworld.cheaptrick(x, self.f0, t, RATE)
        self.ap = pyworld.d4c(x, self.f0, t, RATE)

    def copy(self):
        other = object.__new__(World)
        other.n, other.f0, other.sp, other.ap = self.n, self.f0.copy(), self.sp.copy(), self.ap.copy()
        return other

    @property
    def frames(self):
        return len(self.f0)

    def synth(self):
        y = pyworld.synthesize(np.ascontiguousarray(self.f0), np.ascontiguousarray(self.sp),
                               np.ascontiguousarray(self.ap), RATE, FRAME_MS)
        return y

    # -- envelope geometry
    def warp_formants(self, factor):
        """factor > 1 = shorter vocal tract (child), < 1 = longer. factor may be a per-frame array."""
        bins = self.sp.shape[1]
        freq = np.arange(bins)
        factor = np.broadcast_to(np.asarray(factor, float), (self.frames,))
        for i in range(self.frames):
            src = np.clip(freq / factor[i], 0, bins - 1)
            self.sp[i] = np.interp(src, freq, self.sp[i])
            self.ap[i] = np.interp(src, freq, self.ap[i])
        return self

    def time_map(self, positions):
        """Resample the frame sequence at fractional frame positions (time-stretch/freeze/reverse)."""
        positions = np.clip(np.asarray(positions, float), 0, self.frames - 1)
        i = positions.astype(int)
        j = np.minimum(i + 1, self.frames - 1)
        w = (positions - i)[:, None]
        voiced = (self.f0[i] > 0) & (self.f0[j] > 0)
        f0 = np.where(voiced, self.f0[i] * (1 - w[:, 0]) + self.f0[j] * w[:, 0], np.where(w[:, 0] < .5, self.f0[i], self.f0[j]))
        self.sp = np.exp(np.log(self.sp[i] + 1e-16) * (1 - w) + np.log(self.sp[j] + 1e-16) * w)
        self.ap = self.ap[i] * (1 - w) + self.ap[j] * w
        self.f0 = f0
        self.n = int(len(positions) * FRAME_MS / 1000 * RATE)
        return self


def analyse(x):
    return World(x)


# ---------------------------------------------------------------- transformations

def whisper(x, formant=1.0, breath_tilt=True):
    """A true whisper: the same mouth, no vocal folds."""
    w = World(x)
    if formant != 1.0:
        w.warp_formants(formant)
    w.f0[:] = 0.0
    w.ap[:] = np.maximum(w.ap, 0.999)
    y = w.synth()
    if breath_tilt:
        y = hp(y, 260, 2)
        y = peak_eq(y, 2400, 3.0, 0.8)
    return fade(y / (np.max(np.abs(y)) + 1e-9) * 0.5, 0.004, 0.03)


def child(x, pitch=1.42, formant=1.16, breath=0.25, seed="child"):
    """Smaller larynx and shorter vocal tract together. A little extra breathiness and pitch instability;
    children's voices are less controlled than adult TTS."""
    w = World(x)
    r = rng("child-jitter", seed)
    wobble = 1.0 + 0.012 * np.convolve(r.standard_normal(w.frames), np.ones(12) / 12, mode="same")
    w.f0 = w.f0 * pitch * wobble
    w.warp_formants(formant)
    w.ap = np.clip(w.ap + breath * (1 - w.ap) * np.linspace(0.3, 1, w.ap.shape[1])[None, :], 0, 1)
    y = w.synth()
    return fade(y / (np.max(np.abs(y)) + 1e-9) * 0.5, 0.004, 0.03)


def stretch(x, factor, keep_consonants=True):
    """Stretch voiced frames by `factor`, consonants (unvoiced frames) by much less, so it stays a voice."""
    w = World(x)
    positions = []
    p = 0.0
    while p < w.frames - 1:
        positions.append(p)
        voiced = w.f0[int(p)] > 0
        p += 1.0 / (factor if (voiced or not keep_consonants) else max(1.0, factor ** 0.3))
    w.time_map(positions)
    y = w.synth()
    return fade(y / (np.max(np.abs(y)) + 1e-9) * 0.5, 0.004, 0.05)


def freeze(x, start=0.5, over=0.6, monotone=False):
    """From fraction `start` of the line the vocal tract glides (over `over` of the remaining time) into the shape it
    had at `start` and stays there: the words keep coming as rhythm and pitch, but the mouth has stopped."""
    w = World(x)
    a = int(start * w.frames)
    # find a voiced frame near `a` to freeze on (a vowel, not a consonant)
    voiced = np.where(w.f0 > 0)[0]
    a = int(voiced[np.argmin(np.abs(voiced - a))]) if len(voiced) else a
    target_sp, target_ap = w.sp[a].copy(), w.ap[a].copy()
    span = max(1, int(over * (w.frames - a)))
    for i in range(a, w.frames):
        m = min(1.0, (i - a) / span)
        m = m * m * (3 - 2 * m)
        w.sp[i] = np.exp(np.log(w.sp[i] + 1e-16) * (1 - m) + np.log(target_sp + 1e-16) * m)
        w.ap[i] = w.ap[i] * (1 - m) + target_ap * m
    if monotone:
        f0v = w.f0[w.f0 > 0]
        if len(f0v):
            w.f0[a:] = np.where(w.f0[a:] > 0, np.median(f0v), 0)
    y = w.synth()
    return fade(y / (np.max(np.abs(y)) + 1e-9) * 0.5, 0.004, 0.05)


def reverse_speech(x):
    """Reversed delivery but forward articulation of each frame: uncanny, still unmistakably a mouth."""
    w = World(x)
    w.time_map(np.arange(w.frames)[::-1].astype(float))
    y = w.synth()
    return fade(y / (np.max(np.abs(y)) + 1e-9) * 0.5, 0.01, 0.03)


def reshape(x, pitch=1.0, formant=1.0, speed=1.0, breath=0.0, f0_curve=None):
    w = World(x)
    if speed != 1.0:
        w.time_map(np.arange(0, w.frames - 1, speed))
    if f0_curve is not None:
        w.f0 = np.where(w.f0 > 0, w.f0 * np.interp(np.linspace(0, 1, w.frames), np.linspace(0, 1, len(f0_curve)), f0_curve), 0)
    w.f0 = w.f0 * pitch
    if formant != 1.0:
        w.warp_formants(formant)
    if breath:
        w.ap = np.clip(w.ap + breath * (1 - w.ap), 0, 1)
    y = w.synth()
    return fade(y / (np.max(np.abs(y)) + 1e-9) * 0.5, 0.004, 0.03)


def sing(words, seed="sing", vibrato=5.2, vibrato_depth=0.012, scoop=0.03, formant=1.0, timing=0.03,
         pitch_error=0.006):
    """Turn spoken words into sung notes. words: list of (audio, [(midi, seconds), ...]). A word with several notes
    is split at the energy dips inside its voiced region (syllables). Consonant frames keep their natural speed and
    are placed *before* the beat, as singers do; the vowel is stretched to fill the note, F0 replaced by the note
    with a scoop and delayed vibrato. Every singer gets their own errors."""
    r = rng("sing", seed)
    total = sum(d for _, notes in words for _, d in notes)
    out = np.zeros(sec(total + 3))
    beat_t = 0.4
    for audio, notes in words:
        w = World(audio)
        energy = np.log(np.sum(w.sp, axis=1) + 1e-12)
        voiced = np.where(w.f0 > 0)[0]
        if len(voiced) < 3:
            voiced = np.arange(w.frames)
        v0, v1 = int(voiced[0]), int(voiced[-1])
        # syllable boundaries: deepest energy dips, spread across the voiced region
        cuts = [v0]
        for k in range(1, len(notes)):
            lo = v0 + int((v1 - v0) * (k - 0.35) / len(notes))
            hi = v0 + int((v1 - v0) * (k + 0.35) / len(notes))
            cuts.append(lo + int(np.argmin(energy[lo:max(lo + 1, hi)])))
        cuts.append(v1 + 1)
        pre = np.arange(0, v0, 1.0)
        post = np.arange(v1 + 1, w.frames, 1.0)
        positions, f0_target = [], []
        for k, (midi, dur) in enumerate(notes):
            a, b = cuts[k], max(cuts[k] + 2, cuts[k + 1])
            seg_frames = max(3, int(dur * 1000 / FRAME_MS) - (len(post) if k == len(notes) - 1 else 0))
            # keep the syllable's first frames (onset consonant) at natural speed, stretch the rest
            onset = min(b - a - 1, 6)
            body = np.linspace(a + onset, b - 1, max(2, seg_frames - onset))
            seg = np.concatenate([np.arange(a, a + onset, 1.0), body])
            positions.append(seg)
            f0_target.append(np.full(len(seg), 440.0 * 2 ** ((midi - 69) / 12) * (1 + r.normal(0, pitch_error))))
        positions = np.concatenate([pre] + positions + [post])
        target = np.concatenate([np.full(len(pre), f0_target[0][0])] + f0_target + [np.full(len(post), f0_target[-1][-1])])
        w.time_map(positions)
        t = np.arange(w.frames) * FRAME_MS / 1000
        # note-local time for scoop/vibrato
        change = np.r_[0, np.where(np.abs(np.diff(target)) > 0.5)[0] + 1]
        local = t - t[change[np.searchsorted(change, np.arange(w.frames), side="right") - 1]]
        vib = 1 + vibrato_depth * np.sin(2 * np.pi * (vibrato + r.normal(0, .3)) * t + r.uniform(0, 6)) * np.clip((local - 0.3) / 0.4, 0, 1)
        sc = 1 - scoop * np.exp(-local / 0.06)
        drift = 1 + 0.003 * np.cumsum(r.standard_normal(w.frames)) / np.sqrt(w.frames)
        glide = np.convolve(target, np.ones(9) / 9, mode="same")       # legato between notes
        glide[:4], glide[-4:] = target[:4], target[-4:]
        w.f0 = np.where(w.f0 > 0, glide * vib * sc * drift, 0)
        if formant != 1.0:
            w.warp_formants(formant)
        y = fade(w.synth(), 0.005, 0.04)
        lead = len(pre) * FRAME_MS / 1000
        at = sec(max(0.0, beat_t - lead + r.normal(0, timing)))
        out[at:at + len(y)] += y[:len(out) - at]
        beat_t += sum(d for _, d in notes)
    return out


# ---------------------------------------------------------------- devices and degradations

def pa_speaker(x, seed="pa", hum=0.012, distortion=1.8):
    """A cheap, old ceiling loudspeaker: narrow band, cone break-up resonance, a bit of rattle, mains hum
    and amplifier hiss riding under the voice."""
    r = rng("pa", seed)
    y = bp(x, 340, 3600, order=2)
    y = peak_eq(y, 1150, 7.0, 2.2)
    y = peak_eq(y, 2900, 4.0, 3.0)
    y = np.tanh(y / (np.max(np.abs(y)) + 1e-9) * distortion) * 0.5
    t = np.arange(len(y)) / RATE
    mains = sum(np.sin(2 * np.pi * 50 * k * t + r.uniform(0, 6)) / k ** 1.3 for k in (1, 2, 3, 5, 7))
    hiss = bp(r.standard_normal(len(y)), 1500, 7000) * 0.006
    return y + hum * mains + hiss


def pa_click(seed="click", on=True):
    """Relay clack and amplifier thump when the PA keys up (or drops)."""
    r = rng("paclick", seed, on)
    n = sec(0.35)
    t = np.arange(n) / RATE
    thump = np.sin(2 * np.pi * 48 * t) * np.exp(-t / 0.05) * (0.5 if on else 0.35)
    clack = bp(r.standard_normal(n), 900, 5000) * np.exp(-t / 0.004) * 0.6
    return pa_speaker(thump + clack, seed=seed, hum=0.0, distortion=2.5)


def tape(x, seed="tape", wow=0.004, flutter=0.0015, hiss=0.01, dropouts=0.0, age=1.0):
    """A cassette that has been in a damp classroom for years."""
    from .core import varispeed, smooth_noise
    r = rng("tape", seed)
    n = len(x)
    t = np.arange(n + sec(2)) / RATE
    speed = 1 + wow * np.sin(2 * np.pi * 0.55 * t + r.uniform(0, 6)) + flutter * np.sin(2 * np.pi * 7.3 * t) \
        + 0.6 * wow * (smooth_noise(r, len(t), 0.7, -1, 1))
    y = varispeed(np.pad(x, (0, sec(2))), speed)[:n]
    y = lp(y, 7000 / age, 2)
    y = hp(y, 70, 2)
    y = np.tanh(y * 1.6) / 1.6
    y += bp(r.standard_normal(len(y)), 2000, 9000) * hiss
    if dropouts:
        env = smooth_noise(r, len(y), 0.09, 0, 1)
        y *= np.clip((env - dropouts) * 6 + 1, 0.05, 1)
    return y


def breath(seed="breath", inhale=1.6, exhale=2.1, depth=1.0, nasal=0.4, voiced=0.0):
    """One human breath cycle built from a real vocal tract: noise through slowly changing mouth/nose resonances,
    the inhale brighter and more turbulent, the exhale softer and lower. Optional faint voicing on the exhale (the
    sound a frightened person makes without meaning to)."""
    r = rng("breath", seed)
    parts = []
    for kind, dur in (("in", inhale), ("out", exhale)):
        n = sec(dur)
        t = np.linspace(0, 1, n)
        noise = r.standard_normal(n)
        env = np.sin(np.pi * t ** (0.7 if kind == "in" else 0.45)) ** (1.6 if kind == "in" else 2.0)
        f1 = r.uniform(500, 800) if kind == "in" else r.uniform(380, 600)
        f2 = r.uniform(1200, 1900)
        mouth = bp(noise, f1 * 0.8, f1 * 1.25) * 0.8 + bp(noise, f2 * 0.85, f2 * 1.15) * 0.6
        nose = bp(noise, 220, 420) * nasal
        air = hp(noise, 2500) * (0.25 if kind == "in" else 0.1)
        y = (mouth + nose + air) * env
        if kind == "out" and voiced:
            hz = r.uniform(95, 150)
            ph = 2 * np.pi * np.cumsum(hz * (1 + 0.03 * np.sin(2 * np.pi * 3 * t * dur)) / RATE)
            buzz = sum(np.sin(k * ph) / k for k in range(1, 18))
            y += bp(buzz, 150, 1800) * env ** 3 * voiced * 0.3
        parts.append(y * (1.0 if kind == "in" else 0.75) * depth)
        parts.append(np.zeros(sec(r.uniform(0.05, 0.25))))
    y = np.concatenate(parts)
    return y / (np.max(np.abs(y)) + 1e-9) * 0.5
