"""The people who should not be here, and everything they say.

Writing rule: nobody says anything frightening. The teacher takes the register. A child counts. A hymn about
mornings. The horror is entirely in who is answering, from where, and when.
"""
import functools
import glob
import hashlib
import json

import numpy as np
import soundfile as sf

from . import voice as V
from .core import BUILD, CACHE, RATE, bp, fade, hp, load, lp, resample, rng, sec, shelf

DISK = BUILD / "cache" / "voices"


def cached(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        key = hashlib.sha1(json.dumps([fn.__name__, args, kwargs], default=str).encode()).hexdigest()[:20]
        path = DISK / ("%s-%s.npy" % (fn.__name__, key))
        if path.exists():
            return np.load(path)
        y = fn(*args, **kwargs)
        DISK.mkdir(parents=True, exist_ok=True)
        np.save(path, y)
        return y
    return wrapper


TEACHER = "bf_emma"          # Mrs. Hale. Calm, kind, slightly tired. She has taken this register ten thousand times.
REGISTER = [("Thomas Avery.", "Here."), ("Hannah Bell.", "Here, miss."), ("Daniel Cole.", "Here."),
            ("Sophie Doyle.", None)]
CHILDREN = [  # style blends -> distinct small speakers (all pass through child())
    (("bf_lily", .7), ("af_sky", .3)), (("af_bella", .5), ("bf_alice", .5)), (("af_nicole", .6), ("bf_lily", .4)),
    (("bf_isabella", .5), ("af_sky", .5)), (("af_heart", .6), ("bf_alice", .4)), (("am_puck", .5), ("bf_lily", .5)),
    (("am_liam", .4), ("af_sky", .6)), (("bf_alice", .8), ("af_river", .2)), (("af_kore", .5), ("bf_lily", .5)),
    (("am_echo", .3), ("af_nova", .7)), (("af_aoede", .6), ("bf_emma", .4)), (("am_adam", .3), ("af_jessica", .7)),
    (("af_sarah", .6), ("bf_isabella", .4)), (("am_michael", .35), ("af_alloy", .65)),
]
COUNTER = 0                  # the child who counts


def _style(i):
    return V.voice_blend(*CHILDREN[i % len(CHILDREN)])


@cached
def teacher(text, speed=0.92):
    return V.tts(text, TEACHER, speed)


@cached
def child_says(text, who=0, speed=0.88, pitch=1.38, formant=1.15):
    x = V.tts(text, _style(who), speed)
    return V.child(x, pitch=pitch, formant=formant, seed="kid%d" % who)


@cached
def count_word(n, who=COUNTER, mood="count"):
    words = ["zero", "One.", "Two.", "Three.", "Four.", "Five.", "Six.", "Seven.", "Eight.", "Nine.", "Ten."]
    x = V.tts(words[n], _style(who), 0.78)
    if mood == "count":
        return V.child(x, pitch=1.36, formant=1.15, seed="count%d" % n)
    if mood == "whisper":
        return V.whisper(V.child(x, pitch=1.36, formant=1.15, seed="count%d" % n), formant=1.0)
    raise ValueError(mood)


@cached
def whispered(text, who=None, speed=0.85, formant=1.0):
    style = TEACHER if who is None else _style(who)
    x = V.tts(text, style, speed)
    if who is not None:
        x = V.child(x, pitch=1.3, formant=1.12, seed="w%d" % who)
    return V.whisper(x, formant=formant)


@cached
def teacher_freezing(text, start=0.4, over=0.5, stretch=1.0):
    x = teacher(text, 0.88)
    if stretch != 1.0:
        x = V.stretch(x, stretch)
    return V.freeze(x, start, over, monotone=True)


@cached
def teacher_reversed(text):
    return V.reverse_speech(teacher(text))


@cached
def hymn_voice(singer, beat=0.62, same_voice=False, timing=0.035, formant=1.14, transpose=0):
    """One child singing the hymn. same_voice=True: every singer is the *same* child, perfectly in time."""
    from .score import HYMN
    who = 0 if same_voice else singer
    style = _style(who)
    words = [(V.tts(w, style, 0.8), [(m + transpose, b * beat) for m, b in zip(ms, bs)]) for w, ms, bs in HYMN]
    return V.sing(words, seed="sing%d" % (0 if same_voice else singer), formant=formant,
                  timing=0.0 if same_voice else timing, pitch_error=0.0 if same_voice else 0.007,
                  vibrato_depth=0.009 if same_voice else 0.012)


# ---------------------------------------------------------------- real breath

@functools.lru_cache(maxsize=1)
def _breath_index():
    path = BUILD / "breath_good.json"
    return json.loads(path.read_text()) if path.exists() else []


@cached
def inhale(index):
    """A real human inhalation (from LibriSpeech readers), cleaned, with the >8 kHz air the 16 kHz source lacks
    re-synthesised from its own envelope so it can sit right at the ear."""
    items = _breath_index()
    f, t0, d, _, _ = items[index % len(items)]
    x, sr = sf.read(f)
    seg = x[max(0, int((t0 - 0.05) * sr)):int((t0 + d + 0.05) * sr)]
    from scipy import signal
    y = signal.resample_poly(seg, 3, 1)
    y = hp(y, 120)
    env = np.convolve(np.abs(bp(y, 3000, 7500)), np.ones(240) / 240, mode="same")
    r = rng("air", index)
    air = hp(r.standard_normal(len(y)), 8000) * env * 0.8
    y = y + air
    y = y / (np.max(np.abs(y)) + 1e-9) * 0.5
    return fade(y, 0.03, 0.06)


@cached
def exhale(index, voiced=0.0):
    """Exhalation: a whispered, long 'hhh' from a real vocal tract (stretched whisper of a sigh)."""
    r = rng("exhale", index)
    text = ["Hah.", "Huh.", "Hhh.", "Hoh."][index % 4]
    x = V.tts(text, CHILDREN[index % len(CHILDREN)][0][0], 0.6)
    x = V.stretch(x, r.uniform(2.5, 3.5))
    w = V.whisper(x)
    if voiced:
        w = w + V.freeze(x, 0.0, 0.1) * voiced * 0.3
    n = len(w)
    env = np.sin(np.pi * np.linspace(0, 1, n)) ** 1.2
    return fade(w * env, 0.05, 0.1)
