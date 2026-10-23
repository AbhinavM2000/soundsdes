"""The Kerala cast. Everyone speaks Malayalam (and the bits of school English every Kerala classroom uses).

The teacher - Sir - speaks in the project owner's own voice: Malayalam TTS supplies words and prosody, kNN-VC
(school.clone) re-voices it from the owner's 12.5-minute sample. Children are TTS re-performed through WORLD with
their own pitch, vocal-tract length and breathiness each.

Writing rule unchanged: nobody says anything frightening. Sir takes attendance. A child counts. The pledge.
"""
import functools
import os

import numpy as np
from scipy import signal

from . import voice as V
from .cast import cached
from .core import RATE, fade, rng, sec

ENGINE = os.environ.get("SCHOOL_AUDIO_ML_TTS", "mms")      # "mms" | "parler"

GREETING = "Good morning, children."
CLASS_REPLY = "Good morning, sir."
ATTENDANCE = "ഹാജർ എടുക്കാം."                                # let's take attendance
REGISTER = [("അനിൽ കുമാർ.", "Present, sir."), ("ലക്ഷ്മി.", "ഹാജർ."), ("ഫൈസൽ.", "Present, sir."), ("അമ്മു.", None)]
MISSING = "അമ്മു?"
WHERE = "അമ്മു എവിടെ?"                                       # where is Ammu?
ANSWER = "ഹാജർ."
COUNT = ["", "ഒന്ന്.", "രണ്ട്.", "മൂന്ന്.", "നാല്.", "അഞ്ച്.", "ആറ്.", "ഏഴ്.", "എട്ട്.", "ഒമ്പത്.", "പത്ത്."]
PLEDGE = ["ഇന്ത്യ എന്റെ രാജ്യമാണ്.", "എല്ലാ ഇന്ത്യക്കാരും എന്റെ സഹോദരീസഹോദരന്മാരാണ്.",
          "ഞാൻ എന്റെ രാജ്യത്തെ സ്നേഹിക്കുന്നു."]
PA = {
    "dont_run": "ഓടരുത്. നടന്നു പോകൂ.",                          # don't run. walk.
    "back_to_class": "ക്ലാസ്സിലേക്ക് തിരിച്ചു പോകൂ.",                # go back to your class
    "stay": "അവിടെ തന്നെ നിൽക്കൂ. ആരെങ്കിലും വന്നു കൂട്ടിക്കൊണ്ടു പോകും.",   # stay there. someone will come for you.
    "assembly": "എല്ലാവരും അസംബ്ലിക്ക് വരണം.",                     # everyone come to assembly
    "anyone_left": "ഇനി ആരെയെങ്കിലും വിളിക്കാനുണ്ടോ?",              # is there anyone left to call?
    "thank_you": "നന്ദി.",
    "everyone_here": "ഇപ്പോൾ എല്ലാവരും ഇവിടെയുണ്ട്.",               # now everyone is here
}

KIDS = [  # (pitch ratio from the TTS voice, formant ratio, breathiness)
    (1.30, 1.14, 0.25), (1.42, 1.18, 0.2), (1.22, 1.10, 0.3), (1.50, 1.20, 0.15), (1.35, 1.16, 0.35),
    (1.18, 1.08, 0.2), (1.45, 1.19, 0.3), (1.28, 1.12, 0.1), (1.38, 1.17, 0.25), (1.25, 1.13, 0.2),
    (1.47, 1.21, 0.3), (1.33, 1.15, 0.15), (1.20, 1.11, 0.35), (1.40, 1.18, 0.2),
]


@functools.lru_cache(maxsize=1)
def _mms():
    from transformers import VitsModel, AutoTokenizer
    return VitsModel.from_pretrained("facebook/mms-tts-mal").eval(), AutoTokenizer.from_pretrained("facebook/mms-tts-mal")


@functools.lru_cache(maxsize=1)
def _parler():
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer
    m = ParlerTTSForConditionalGeneration.from_pretrained("ai4bharat/indic-parler-tts").eval()
    return m, AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts"), AutoTokenizer.from_pretrained(
        m.config.text_encoder._name_or_path)


@cached
def ml_tts(text, speaker="Anjali", style="calm", seed=0, speed=1.0, engine=None):
    import torch
    engine = engine or ENGINE
    torch.manual_seed(seed)
    if engine == "parler" and not text.isascii():
        m, tok, dtok = _parler()
        desc = {"calm": "%s speaks in a calm, slightly tired voice at a moderate pace. The recording is very clear and close up.",
                "child": "%s speaks in a high, young voice at a moderate pace. The recording is very clear and close up.",
                "slow": "%s speaks slowly and clearly. The recording is very clear and close up."}[style] % speaker
        di, pi = dtok(desc, return_tensors="pt"), tok(text, return_tensors="pt")
        with torch.no_grad():
            a = m.generate(input_ids=di.input_ids, attention_mask=di.attention_mask, prompt_input_ids=pi.input_ids,
                           prompt_attention_mask=pi.attention_mask)
        x, sr = a.cpu().numpy().squeeze().astype(np.float64), m.config.sampling_rate
    elif text.isascii():
        return V.tts(text, "bf_isabella" if style != "child" else "bf_lily", 0.95 * speed)  # school English
    else:
        m, tok = _mms()
        m.config.speaking_rate = speed
        with torch.no_grad():
            x = m(**tok(text, return_tensors="pt")).waveform[0].numpy().astype(np.float64)
        sr = m.config.sampling_rate
    from fractions import Fraction
    f = Fraction(RATE, sr)
    y = signal.resample_poly(x, f.numerator, f.denominator)
    env = np.abs(y) > 0.01 * np.max(np.abs(y))
    a, b = int(np.argmax(env)), len(y) - int(np.argmax(env[::-1]))
    return fade(y[max(0, a - sec(0.02)):b + sec(0.06)], 0.005, 0.02)


@cached
def sir(text, speed=0.95):
    """Sir: the line in the project owner's voice."""
    from . import clone
    base = ml_tts(text, "Harish", "calm", 1, speed) if not text.isascii() else V.tts(text, "bm_george", speed)
    return clone.to_teacher(base)


@cached
def sir_freezing(text, start=0.35, over=0.45, stretch=1.6):
    x = sir(text, 0.9)
    if stretch != 1.0:
        x = V.stretch(x, stretch)
    return V.freeze(x, start, over, monotone=True)


@cached
def kid(text, who=0, speed=0.95, mood="say"):
    p, f, b = KIDS[who % len(KIDS)]
    x = ml_tts(text, "Anjali", "child", 10 + who, speed)
    y = V.child(x, pitch=p, formant=f, breath=b, seed="mlkid%d" % who)
    if mood == "whisper":
        return V.whisper(y)
    return y


@cached
def class_unison(text, voices=14, spread_ms=60.0, sing_song=True, one_voice=False, seed=0):
    """A class answering together: 'Good morning, sir' or the pledge. Each child slightly early/late and at their
    own pitch; sing_song lifts and draws out the stressed syllables the way every Kerala class does.
    one_voice=True: fourteen children with exactly one voice, perfectly together."""
    r = rng("unison", text, seed)
    parts = []
    for k in range(voices):
        who = 0 if one_voice else k
        x = kid(text, who, 0.85 if sing_song else 0.95)
        if sing_song:
            w = V.World(x)
            t = np.linspace(0, 1, w.frames)
            w.f0 = np.where(w.f0 > 0, w.f0 * (1 + 0.18 * np.sin(np.pi * t) ** 2), 0)
            w.time_map(np.linspace(0, w.frames - 1, int(w.frames * 1.25)))
            x = w.synth()
        delay = 0 if one_voice else int(abs(r.normal(0, spread_ms / 1000)) * RATE)
        detune = 1.0 if not one_voice else 1 + (k - voices / 2) * 0.0007
        if detune != 1.0:
            from .core import resample
            x = resample(x, detune)
        parts.append(np.pad(x, (delay + (int(k * 0.0011 * RATE) if one_voice else 0), 0)))
    n = max(len(p) for p in parts)
    y = sum(np.pad(p, (0, n - len(p))) for p in parts)
    return y / (np.max(np.abs(y)) + 1e-9) * 0.6


@cached
def count_word(n, who=3, mood="say"):
    return kid(COUNT[n], who, 0.8, mood)
