"""Build the second horror pass's audible, human-source cues.

python3 make_horror_audio.py [--capture-voices]
Input chant: reference/audio/dies_irae_membeth.ogg, public-domain performance by Membeth.
The optional macOS capture produces original prototype speech using the installed Whisper voice;
raw captures are retained so ordinary rebuilds need only ffmpeg, numpy and scipy.
No movie soundtrack is sampled. See game/assets/ASSETS_AND_LICENSES.md.
"""
from pathlib import Path
import argparse
import subprocess
import numpy as np
from scipy import signal
from scipy.io import wavfile
import make_night_sounds as base

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'reference/audio'
DEST = ROOT / 'godot/game/assets/sounds'
RATE = base.RATE
PHRASES = ["Please. Do not look at my face.", "That is not your teacher.",
           "There is room. For one. More.", "You were here. When it happened."]


def read(path):
    rate, raw = wavfile.read(path)
    x = raw.astype(np.float64) / 32768.0
    if x.ndim == 2:
        x = x.mean(axis=1)
    return signal.resample_poly(x, RATE, rate)


def stretch(x, speed):
    return signal.resample_poly(x, 1000, int(speed * 1000))


def room(x, tail=2.5):
    out = np.pad(x, (0, int(RATE * tail)))
    for seconds, gain in [(0.083, .24), (.173, .21), (.317, .17), (.643, .13), (1.1, .09), (1.9, .05)]:
        at = int(seconds * RATE)
        out[at:at + len(x)] += base.low_pass(x, 2200) * gain
    return out


def chant(raw):
    # Preserve recognisable Gregorian phrasing; an octave shadow and reversed distant answers corrupt it.
    x = stretch(raw[RATE:RATE * 25], .78)
    out = x.copy()
    shadow = stretch(raw[RATE:RATE * 14], .49)
    out[:min(len(out), len(shadow))] += shadow[:len(out)] * .3
    for at in [8, 17, 24]:
        segment = x[int(at * RATE):int((at + 3) * RATE)]
        if len(segment):
            out[int(at * RATE):int(at * RATE) + len(segment)] += base.fade(segment[::-1], .4) * .22
    return base.normalise(base.fade(room(base.low_pass(out, 3200)), 2.0), .72)


def confession(raw):
    x = stretch(raw, .85)
    dark = stretch(raw, .63)
    out = np.pad(x, (0, max(0, len(dark) - len(x))))
    out[:len(dark)] += dark * .28
    out = base.band_pass(out, 180, 6200)
    return base.normalise(base.fade(room(out, 2.5), .04), .65)


def knocks():
    rng = np.random.default_rng(710)
    out = np.zeros(int(4.3 * RATE))
    for start, gain in [(0.1, .7), (.56, .85), (1.32, 1), (3.0, .3)]:
        t = np.arange(int(.42 * RATE)) / RATE
        hit = base.low_pass(rng.standard_normal(len(t)), 1000) * np.exp(-t / .027)
        hit += .3 * np.sin(2 * np.pi * 93 * t) * np.exp(-t / .055)
        at = int(start * RATE)
        out[at:at + len(hit)] += hit * gain
    return base.normalise(base.fade(room(out), .004), .7)


def strings():
    rng = np.random.default_rng(711)
    t = np.arange(8 * RATE) / RATE
    out = np.zeros(len(t))
    for pitch in [184, 197, 207.65, 311.13, 329.63, 348.0]:
        drift = base.smooth_random(rng, len(t), .15, -.018, .018)
        phase = 2 * np.pi * np.cumsum(pitch * (1 + drift)) / RATE
        voice = sum(np.sin(phase * k + rng.uniform(0, 6)) / k for k in range(1, 12))
        voice *= base.smooth_random(rng, len(t), .25, .1, 1)
        out += voice
    bow = base.band_pass(rng.standard_normal(len(t)), 1600, 7000)
    out = base.normalise(out, .6) + base.normalise(bow, .18)
    return base.normalise(base.fade(room(out), 1.3), .65)


def impact():
    rng = np.random.default_rng(712)
    t = np.arange(int(2.8 * RATE)) / RATE
    noise = base.low_pass(rng.standard_normal(len(t)), 1700) * np.exp(-t / .24)
    for f in [38, 57, 83, 121, 173]:
        noise += .12 * np.sin(2 * np.pi * f * t) * np.exp(-t / .45)
    return base.normalise(base.fade(room(noise), .012), .7)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture-voices', action='store_true')
    args = parser.parse_args()
    SOURCE.mkdir(exist_ok=True)
    if args.capture_voices:
        for i, text in enumerate(PHRASES):
            aiff = SOURCE / f'whisper_source_{i}.aiff'
            subprocess.run(['say', '-v', 'Whisper', '-r', '105', '-o', str(aiff), text], check=True)
            subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', str(aiff), '-ar', str(RATE),
                            '-ac', '1', str(SOURCE / f'whisper_source_{i}.wav')], check=True)
    decoded = ROOT / 'build/horror-pass-2/chant-source.wav'
    decoded.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', str(SOURCE / 'dies_irae_membeth.ogg'),
                    '-t', '32', '-ar', str(RATE), '-ac', '1', str(decoded)], check=True)
    clips = {'ritual_chant': chant(read(decoded)), 'door_knocks': knocks(),
             'bowed_dread': strings(), 'presence_impact': impact()}
    for i in range(len(PHRASES)):
        clips[f'confession_{i}'] = confession(read(SOURCE / f'whisper_source_{i}.wav'))
    for name, samples in clips.items():
        base.write_wav(samples, str(DEST / (name + '.wav')))
        print(name, f'{len(samples) / RATE:.2f}s', f'peak {np.max(np.abs(samples)):.2f}')


if __name__ == '__main__':
    main()
