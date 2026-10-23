"""Shared DSP primitives: I/O, filters, envelopes, loudness, true peak, seamless loops."""
import os
import subprocess
import zlib
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

RATE = 48000
ROOT = Path(__file__).resolve().parents[2]
CACHE = Path(os.environ.get("SCHOOL_AUDIO_CACHE", ROOT / ".cache" / "sources"))
BUILD = ROOT / "build"


def sec(seconds):
    return int(round(seconds * RATE))


SEED = int(os.environ.get("SCHOOL_AUDIO_SEED", "1983"))


def rng(*key):
    """Deterministic generator per named purpose, e.g. rng("footstep", 3). Change SCHOOL_AUDIO_SEED for a new take."""
    return np.random.default_rng([zlib.crc32("|".join(map(str, key)).encode()), SEED])


# ---------------------------------------------------------------- I/O

@lru_cache(maxsize=256)
def _load(path, channels):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", str(channels), "-ar", str(RATE),
                          "-f", "f32le", "-"], capture_output=True, check=True).stdout
    x = np.frombuffer(raw, np.float32).astype(np.float64)
    return x.reshape(-1, channels).T if channels > 1 else x


def load(path, stereo=False, start=0.0, length=None):
    """Any audio file -> float64 at RATE. Mono (n,) or stereo (2, n)."""
    x = _load(str(path), 2 if stereo else 1)
    a = sec(start)
    b = None if length is None else a + sec(length)
    return x[..., a:b].copy()


def source(relative, **kwargs):
    return load(CACHE / relative, **kwargs)


def write(path, x, bits=24):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(x, dtype=np.float64)
    data = data.T if data.ndim == 2 else data
    subtype = {16: "PCM_16", 24: "PCM_24", 32: "FLOAT"}[bits]
    if path.suffix == ".wav":
        sf.write(str(path), np.clip(data, -1, 1) if bits != 32 else data, RATE, subtype=subtype)
    elif path.suffix == ".flac":
        sf.write(str(path), np.clip(data, -1, 1), RATE, subtype="PCM_24" if bits >= 24 else "PCM_16")
    elif path.suffix in (".ogg", ".mp3"):
        tmp = path.with_suffix(".tmp.wav")
        sf.write(str(tmp), np.clip(data, -1, 1), RATE, subtype="PCM_24")
        codec = ["-c:a", "libvorbis", "-q:a", "7"] if path.suffix == ".ogg" else ["-c:a", "libmp3lame", "-b:a", "256k"]
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(tmp), *codec, str(path)], check=True)
        tmp.unlink()
    else:
        raise ValueError(path)
    return path


# ---------------------------------------------------------------- filters

def _sos(kind, freq, order=2):
    freq = np.clip(np.asarray(freq, dtype=float), 5.0, RATE / 2 - 100)
    return signal.butter(order, freq, btype=kind, fs=RATE, output="sos")


def lp(x, f, order=2):
    return signal.sosfilt(_sos("lowpass", f, order), x, axis=-1)


def hp(x, f, order=2):
    return signal.sosfilt(_sos("highpass", f, order), x, axis=-1)


def bp(x, lo, hi, order=2):
    return signal.sosfilt(_sos("bandpass", [lo, hi], order), x, axis=-1)


def zlp(x, f, order=2):
    return signal.sosfiltfilt(_sos("lowpass", f, order), x, axis=-1)


def zhp(x, f, order=2):
    return signal.sosfiltfilt(_sos("highpass", f, order), x, axis=-1)


def peak_eq(x, f, gain_db, q=1.0):
    """RBJ peaking biquad."""
    a = 10 ** (gain_db / 40)
    w = 2 * np.pi * f / RATE
    alpha = np.sin(w) / (2 * q)
    b = [1 + alpha * a, -2 * np.cos(w), 1 - alpha * a]
    den = [1 + alpha / a, -2 * np.cos(w), 1 - alpha / a]
    return signal.lfilter(np.array(b) / den[0], np.array(den) / den[0], x, axis=-1)


def shelf(x, f, gain_db, high=True):
    """RBJ shelving biquad (S=1)."""
    a = 10 ** (gain_db / 40)
    w = 2 * np.pi * f / RATE
    cw, alpha = np.cos(w), np.sin(w) / 2 * np.sqrt(2)
    s = 1 if high else -1
    b0 = a * ((a + 1) + s * (a - 1) * cw + 2 * np.sqrt(a) * alpha)
    b1 = -2 * s * a * ((a - 1) + s * (a + 1) * cw)
    b2 = a * ((a + 1) + s * (a - 1) * cw - 2 * np.sqrt(a) * alpha)
    a0 = (a + 1) - s * (a - 1) * cw + 2 * np.sqrt(a) * alpha
    a1 = 2 * s * ((a - 1) - s * (a + 1) * cw)
    a2 = (a + 1) - s * (a - 1) * cw - 2 * np.sqrt(a) * alpha
    return signal.lfilter(np.array([b0, b1, b2]) / a0, np.array([a0, a1, a2]) / a0, x, axis=-1)


def tv_lowpass(x, cutoff):
    """One-pole low-pass with a per-sample cutoff curve (for doors opening, rooms sealing)."""
    cutoff = np.broadcast_to(cutoff, x.shape[-1])
    coef = np.exp(-2 * np.pi * np.clip(cutoff, 10, RATE / 2.2) / RATE)
    out = np.empty_like(x)
    state = np.zeros(x.shape[:-1])
    xs = np.moveaxis(x, -1, 0)
    os_ = np.moveaxis(out, -1, 0)
    for i in range(xs.shape[0]):
        state = (1 - coef[i]) * xs[i] + coef[i] * state
        os_[i] = state
    return out


def tv_lowpass_fast(x, cutoff, block=256):
    """Block-wise time-varying 2nd-order low-pass; adequate for slow automation, much faster than tv_lowpass."""
    cutoff = np.broadcast_to(cutoff, x.shape[-1])
    out = np.zeros_like(x)
    zi = None
    for a in range(0, x.shape[-1], block):
        b = min(a + block, x.shape[-1])
        sos = _sos("lowpass", float(np.mean(cutoff[a:b])), 2)
        if zi is None:
            zi = np.zeros((sos.shape[0],) + x.shape[:-1] + (2,))
            zi = np.moveaxis(zi, -1, 1) if x.ndim > 1 else zi
        seg, zi = signal.sosfilt(sos, x[..., a:b], axis=-1, zi=zi)
        out[..., a:b] = seg
    return out


# ---------------------------------------------------------------- noise, envelopes, curves

def pink(r, n):
    spec = np.fft.rfft(r.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / RATE)
    spec[1:] /= np.sqrt(f[1:])
    spec[0] = 0
    y = np.fft.irfft(spec, n)
    return y / (np.std(y) + 1e-12)


def brown(r, n):
    y = np.cumsum(r.standard_normal(n))
    y = zhp(y, 12)
    return y / (np.std(y) + 1e-12)


def smooth_noise(r, n, step_seconds, lo=0.0, hi=1.0):
    """Eased random curve with a new target every step_seconds (from the legacy pipeline; it was the one good idea)."""
    pts = r.uniform(lo, hi, int(n / (step_seconds * RATE)) + 3)
    pos = np.arange(n) / (step_seconds * RATE)
    i = pos.astype(int)
    t = pos - i
    e = t * t * (3 - 2 * t)
    return pts[i] * (1 - e) + pts[i + 1] * e


def ramp(n, points):
    """Piecewise-linear automation: points = [(seconds, value), ...]."""
    t = np.arange(n) / RATE
    xs, ys = zip(*points)
    return np.interp(t, xs, ys)


def fade(x, fin=0.01, fout=0.01):
    x = x.copy()
    n = x.shape[-1]
    a, b = min(sec(fin), n // 2), min(sec(fout), n // 2)
    if a:
        x[..., :a] *= np.sin(np.linspace(0, np.pi / 2, a)) ** 2
    if b:
        x[..., n - b:] *= np.cos(np.linspace(0, np.pi / 2, b)) ** 2
    return x


def db(x):
    return 10 ** (x / 20)


def rms_db(x):
    return 20 * np.log10(np.sqrt(np.mean(np.square(x))) + 1e-12)


def peak_norm(x, peak_db=-1.0):
    return x / (np.max(np.abs(x)) + 1e-12) * db(peak_db)


def rms_norm(x, target_db=-20.0):
    return x * db(target_db - rms_db(x))


def pad_to(x, n):
    if x.shape[-1] >= n:
        return x[..., :n]
    widths = [(0, 0)] * (x.ndim - 1) + [(0, n - x.shape[-1])]
    return np.pad(x, widths)


def add(dst, src, at):
    """Mix src into dst starting at sample `at` (clipped to dst bounds). Mono src into stereo dst is broadcast."""
    if at >= dst.shape[-1] or at + src.shape[-1] <= 0:
        return dst
    a0 = max(0, -at)
    a = max(at, 0)
    b = min(dst.shape[-1], at + src.shape[-1])
    dst[..., a:b] += src[..., a0:a0 + b - a]
    return dst


def stereo(x):
    return np.vstack([x, x]) if x.ndim == 1 else x


def resample(x, ratio):
    """Change speed/pitch by `ratio` (>1 = higher/shorter), high quality polyphase."""
    from fractions import Fraction
    frac = Fraction(1 / ratio).limit_denominator(2000)
    return signal.resample_poly(x, frac.numerator, frac.denominator, axis=-1)


def varispeed(x, rate_curve):
    """Time-varying playback speed (tape wow, slowing machinery). rate_curve: per-output-sample speed."""
    pos = np.cumsum(rate_curve)
    pos = pos[pos < x.shape[-1] - 2]
    i = pos.astype(int)
    f = pos - i
    # 4-point cubic Hermite interpolation
    xm1 = x[..., np.maximum(i - 1, 0)]
    x0, x1, x2 = x[..., i], x[..., i + 1], x[..., i + 2]
    c1 = 0.5 * (x1 - xm1)
    c2 = xm1 - 2.5 * x0 + 2 * x1 - 0.5 * x2
    c3 = 0.5 * (x2 - xm1) + 1.5 * (x0 - x1)
    return ((c3 * f + c2) * f + c1) * f + x0


def conv(x, h):
    """FFT convolution; mono*mono, mono*stereo IR, stereo*stereo (per channel)."""
    if x.ndim == 1 and h.ndim == 2:
        return np.vstack([signal.fftconvolve(x, h[0]), signal.fftconvolve(x, h[1])])
    if x.ndim == 2 and h.ndim == 2:
        return np.vstack([signal.fftconvolve(x[0], h[0]), signal.fftconvolve(x[1], h[1])])
    if x.ndim == 2:
        return np.vstack([signal.fftconvolve(x[0], h), signal.fftconvolve(x[1], h)])
    return signal.fftconvolve(x, h)


def saturate(x, drive=1.0):
    return np.tanh(x * drive) / np.tanh(drive)


# ---------------------------------------------------------------- loudness / peaks

def lufs(x):
    import pyloudnorm
    data = stereo(x).T if x.ndim == 2 else x
    meter = pyloudnorm.Meter(RATE)
    try:
        return meter.integrated_loudness(data)
    except ValueError:
        return -120.0


def short_term_max(x):
    """Max short-term (3 s) loudness, a better measure of 'how loud does it get' than the integrated value."""
    import pyloudnorm
    meter = pyloudnorm.Meter(RATE, block_size=3.0)
    data = stereo(x).T
    best = -120.0
    for a in range(0, data.shape[0] - sec(3), sec(1)):
        try:
            best = max(best, meter.integrated_loudness(data[a:a + sec(3)]))
        except ValueError:
            pass
    return best


def true_peak_db(x):
    up = signal.resample_poly(stereo(x), 4, 1, axis=-1)
    return 20 * np.log10(np.max(np.abs(up)) + 1e-12)


def limiter(x, ceiling_db=-1.0, release=0.12, lookahead=0.005):
    """Look-ahead peak limiter working on a 4x-oversampled envelope so the true peak stays under the ceiling."""
    x = stereo(x)
    ceiling = db(ceiling_db)
    up = np.max(np.abs(signal.resample_poly(x, 4, 1, axis=-1)), axis=0)
    env = up.reshape(-1, 4).max(axis=1)[:x.shape[-1]]
    env = pad_to(env, x.shape[-1])
    need = np.minimum(1.0, ceiling / np.maximum(env, 1e-9))
    la = sec(lookahead)
    from scipy.ndimage import minimum_filter1d
    need = minimum_filter1d(need, la * 2 + 1, mode="nearest")
    coef = np.exp(-1 / (release * RATE))
    gain = np.empty_like(need)
    g = 1.0
    for i, target in enumerate(need):
        g = target if target < g else target + (g - target) * coef
        gain[i] = g
    gain = np.convolve(gain, np.hanning(la * 2 + 1) / np.hanning(la * 2 + 1).sum(), mode="same")
    gain = np.minimum(gain, need)
    return x * gain


# ---------------------------------------------------------------- loops

def seamless_loop(raw, length, crossfade=4.0):
    """raw has >= length + crossfade samples. Equal-power crossfade of the overhang into the head, which is
    correct for uncorrelated material (ambience) and avoids the -3 dB dip a linear fade causes."""
    n = sec(crossfade)
    loop = raw[..., :length].copy()
    t = np.linspace(0, np.pi / 2, n)
    loop[..., :n] = raw[..., :n] * np.sin(t) + raw[..., length:length + n] * np.cos(t)
    return loop


def loop_seam_jump(loop):
    """Discontinuity at the wrap point relative to typical sample-to-sample motion (1.0 = indistinguishable)."""
    x = stereo(loop)
    d = np.abs(np.diff(x, axis=-1))
    typical = np.percentile(d, 99.5) + 1e-12
    seam = np.max(np.abs(x[:, 0] - x[:, -1]))
    return float(seam / typical)
