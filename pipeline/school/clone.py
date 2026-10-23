"""Voice cloning: re-voice any line in the project owner's own voice (supplied by them for this purpose).

kNN-VC (Baas, van Niekerk, Kamper 2023): every 20 ms WavLM frame of the source line is replaced by the average of
its nearest frames in the target speaker's recordings, then vocoded. It is language-independent, so Malayalam TTS
lines keep their words and prosody and take on the target's timbre. Works best with 5+ minutes of target speech;
the supplied sample is 12.5 minutes.

    SCHOOL_AUDIO_TEACHER_VOICE=/path/to/voice.mp3   (default: <cache>/user_voice/sample.mp3)
"""
import os
import sys
from functools import lru_cache

import numpy as np
from scipy import signal

from .core import CACHE, RATE, BUILD, load

TARGET = os.environ.get("SCHOOL_AUDIO_TEACHER_VOICE", str(CACHE / "user_voice" / "sample.mp3"))


@lru_cache(maxsize=1)
def _model():
    import torch
    sys.path.insert(0, str(CACHE / "knnvc" / "knn-vc"))
    import hubconf
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    return hubconf.knn_vc(pretrained=True, progress=False, prematched=True, device="cpu")


@lru_cache(maxsize=1)
def _matching_set():
    import torch
    path = BUILD / "cache" / "teacher_matching_set.pt"
    if path.exists():
        return torch.load(path)
    x = load(TARGET)
    x16 = signal.resample_poly(x, 1, 3).astype(np.float32)
    chunk = 16000 * 20
    wavs = [torch.from_numpy(x16[a:a + chunk]) for a in range(0, len(x16) - 16000 * 2, chunk)]
    ms = _model().get_matching_set(wavs)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ms, path)
    return ms


def to_teacher(x, topk=4):
    """x: mono float at RATE -> the same words in the target voice, at RATE."""
    import torch
    m = _model()
    x16 = signal.resample_poly(x, 1, 3).astype(np.float32)
    x16 = x16 / (np.max(np.abs(x16)) + 1e-9) * 0.9
    with torch.inference_mode():
        q = m.get_features(torch.from_numpy(x16)[None])
        y = m.match(q, _matching_set(), topk=topk)
    y = signal.resample_poly(y.cpu().numpy().astype(np.float64), 3, 1)
    return y / (np.max(np.abs(y)) + 1e-9) * 0.5
