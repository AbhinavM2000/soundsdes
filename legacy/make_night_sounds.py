"""Synthesised night sounds (no recordings, so no licences to track).

    python3 make_night_sounds.py
        -> godot/game/assets/sounds/: drone, wind, crickets, creak_0..5, rain_outdoor, rain_roof, drips,
           thunder_near_0..2, thunder_far_0..2, whisper_0..11, heartbeat, breath_0..1, step_0..3, rumble,
           groan_0..2, debris_0..3, buzz, spark_0..2, crackle, bell_faint
        -> godot/npc/sounds/murmur_0..7.wav  (the zombie murmurs Phase 1 left as a stub)

Voices (murmurs, whispers) are deliberately aperiodic: the owner heard the first tonal versions (frogs, formant
murmurs) as "horn-like pom pom". Murmurs are irregular vocal fry buried in breath; whispers are unvoiced.

Every generator is seeded and returns float32 samples in [-1, 1] at RATE. Loops are made seamless by
crossfading their last LOOP_CROSSFADE seconds into the start, then rotating the loop so it starts where two
neighbouring samples are closest (no step at the seam).
"""
import os
import wave

import numpy
from scipy import signal

HERE = os.path.dirname(os.path.abspath(__file__))
SOUND_DIR = os.path.join(HERE, "godot", "game", "assets", "sounds")
MURMUR_DIR = os.path.join(HERE, "godot", "npc", "sounds")
RATE = 48000
LOOP_CROSSFADE = 2.0


def normalise(samples, peak):
    return (samples / max(float(numpy.abs(samples).max()), 1e-9) * peak).astype(numpy.float32)


def band_pass(samples, low, high, order=2):
    sos = signal.butter(order, [low, high], btype="bandpass", fs=RATE, output="sos")
    return signal.sosfilt(sos, samples)


def low_pass(samples, cutoff, order=2):
    return signal.sosfilt(signal.butter(order, cutoff, btype="lowpass", fs=RATE, output="sos"), samples)


def fade(samples, seconds):
    count = min(int(seconds * RATE), len(samples) // 2)
    ramp = numpy.sin(numpy.linspace(0.0, numpy.pi / 2, count)) ** 2
    samples = samples.copy()
    samples[:count] *= ramp
    samples[len(samples) - count:] *= ramp[::-1]
    return samples


def smooth_random(rng, length, step_seconds, low=0.0, high=1.0):
    """Slowly varying random curve: random points every step_seconds, eased between."""
    points = rng.uniform(low, high, int(length / (step_seconds * RATE)) + 3)
    position = numpy.arange(length) / (step_seconds * RATE)
    index = position.astype(int)
    t = position - index
    eased = t * t * (3.0 - 2.0 * t)
    return points[index] * (1.0 - eased) + points[index + 1] * eased


def make_loop(raw, length):
    """raw has length + crossfade samples; returns a seamless loop of exactly `length` samples."""
    count = int(LOOP_CROSSFADE * RATE)
    loop = raw[:length].copy()
    ramp = numpy.linspace(0.0, 1.0, count)
    loop[:count] = raw[:count] * ramp + raw[length:length + count] * (1.0 - ramp)
    steps = numpy.abs(loop - numpy.roll(loop, 1))
    return numpy.roll(loop, -int(numpy.argmin(steps)))


def pink_noise(rng, length):
    spectrum = numpy.fft.rfft(rng.standard_normal(length))
    frequencies = numpy.fft.rfftfreq(length, 1.0 / RATE)
    spectrum[1:] /= numpy.sqrt(frequencies[1:])
    spectrum[0] = 0.0
    return numpy.fft.irfft(spectrum, length)


def drone(seconds=60.0):
    rng = numpy.random.default_rng(11)
    length = int(seconds * RATE)
    raw_length = length + int(LOOP_CROSSFADE * RATE)
    t = numpy.arange(raw_length) / RATE
    tone = numpy.zeros(raw_length)
    for frequency, gain in ((41.0, 1.0), (55.3, 0.7), (82.5, 0.4)):
        rate = rng.uniform(0.03, 0.07)
        depth = rng.uniform(0.2, 0.5)
        phase = 2 * numpy.pi * numpy.cumsum(frequency + depth * numpy.sin(2 * numpy.pi * rate * t)) / RATE
        tone += gain * numpy.sin(phase)
    brown = low_pass(numpy.cumsum(rng.standard_normal(raw_length)), 120.0)
    brown = brown - low_pass(brown, 20.0)
    tone += 0.6 * brown / max(float(numpy.abs(brown).max()), 1e-9)
    return normalise(make_loop(tone, length), 0.5)


def wind(seconds=60.0):
    rng = numpy.random.default_rng(12)
    length = int(seconds * RATE)
    raw_length = length + int(LOOP_CROSSFADE * RATE)
    noise = pink_noise(rng, raw_length)
    low = band_pass(noise, 220.0, 450.0)
    high = band_pass(noise, 500.0, 950.0)
    drift = smooth_random(rng, raw_length, 6.0)
    gust = 0.25 + 0.75 * smooth_random(rng, raw_length, rng.uniform(3.0, 8.0)) ** 2
    mixed = (low * (1.0 - drift) + high * drift * 0.7) * gust
    return normalise(make_loop(mixed, length), 0.6)


def crickets(seconds=30.0):
    rng = numpy.random.default_rng(13)
    length = int(seconds * RATE)
    raw_length = length + int(LOOP_CROSSFADE * RATE)
    out = numpy.zeros(raw_length)
    for _ in range(5):
        frequency = rng.uniform(4400.0, 4900.0)
        period = rng.uniform(0.6, 1.2)
        gain = rng.uniform(0.3, 1.0)
        pulses = int(rng.integers(3, 6))
        chirp_t = numpy.arange(int(pulses / 30.0 * RATE)) / RATE
        chirp = numpy.sin(2 * numpy.pi * frequency * chirp_t) * numpy.sin(numpy.pi * 30.0 * chirp_t) ** 2
        start = rng.uniform(0.0, period)
        while start * RATE + len(chirp) < raw_length:
            begin = int(start * RATE)
            out[begin:begin + len(chirp)] += gain * chirp * rng.uniform(0.8, 1.0)
            start += period * rng.uniform(0.95, 1.05)
    return normalise(make_loop(out, length), 0.4)


def creak(index):
    rng = numpy.random.default_rng(100 + index)
    length = int(rng.uniform(0.4, 1.2) * RATE)
    impulses = numpy.zeros(length)
    position = 0.0
    rate = rng.uniform(20.0, 60.0)
    while position < length:
        impulses[int(position)] = rng.uniform(0.5, 1.0)
        rate = numpy.clip(rate * rng.uniform(0.9, 1.1), 20.0, 60.0)
        position += RATE / rate * rng.uniform(0.85, 1.15)
    resonance = rng.uniform(400.0, 900.0)
    body = band_pass(impulses, resonance * 0.97, resonance * 1.03) + 0.4 * band_pass(impulses, resonance * 1.94, resonance * 2.06)
    return normalise(fade(body, 0.03), 0.7)


def murmur(index):
    """A raspy, wet groan: irregular low vocal-fry pulses (25-55 Hz, heavy jitter and shimmer) and throat
    turbulence, both through "uh/ugh" formants, with a slow rough growl."""
    rng = numpy.random.default_rng(200 + index)
    length = int(rng.uniform(0.8, 2.0) * RATE)
    t = numpy.arange(length) / RATE
    pulses = numpy.zeros(length)
    position = rng.uniform(0.0, 0.02) * RATE
    rate = rng.uniform(25.0, 55.0)
    while position < length:
        pulses[int(position)] = rng.uniform(0.4, 1.0)
        position += RATE / (rate * rng.uniform(0.7, 1.3))
    fry = signal.lfilter([1.0], [1.0, -0.94], pulses)
    turbulence = rng.standard_normal(length) * (0.6 + 0.4 * smooth_random(rng, length, 0.03))
    source = fry / max(float(numpy.abs(fry).max()), 1e-9) + 0.9 * turbulence / max(float(numpy.abs(turbulence).max()), 1e-9)
    voice = (band_pass(source, *band(rng.uniform(380.0, 600.0), 0.45)) * 1.0
             + band_pass(source, *band(rng.uniform(850.0, 1200.0), 0.35)) * 0.6
             + band_pass(source, *band(rng.uniform(2200.0, 2800.0), 0.3)) * 0.35)
    growl = 0.65 + 0.35 * smooth_random(rng, length, 1.0 / rng.uniform(30.0, 50.0), -1.0, 1.0)
    envelope = numpy.minimum(1.0, t / 0.12) * numpy.exp(-1.0 * t / t[-1])
    return normalise(fade(voice * growl * envelope, 0.05), 0.7)


def drop_ticks(rng, length, per_second, low, high, decay_ms):
    """Random short decaying ticks (raindrops, fire pops): Poisson-timed impulses through a band-pass."""
    impulses = numpy.zeros(length)
    count = rng.poisson(per_second * length / RATE)
    impulses[rng.integers(0, length, count)] = rng.uniform(0.2, 1.0, count)
    ticks = band_pass(impulses, low, high)
    return signal.lfilter([1.0], [1.0, -numpy.exp(-1.0 / (decay_ms / 1000.0 * RATE))], ticks)


def rain_outdoor(seconds=30.0):
    rng = numpy.random.default_rng(21)
    length = int(seconds * RATE)
    raw = length + int(LOOP_CROSSFADE * RATE)
    hiss = band_pass(rng.standard_normal(raw), 1000.0, 9000.0) * (0.8 + 0.2 * smooth_random(rng, raw, 2.0))
    hiss = hiss / max(float(numpy.abs(hiss).max()), 1e-9)
    drops = drop_ticks(rng, raw, 220.0, 3000.0, 8000.0, 4.0)
    drops = drops / max(float(numpy.abs(drops).max()), 1e-9)
    return normalise(make_loop(hiss + 0.5 * drops, length), 0.6)


def rain_roof(seconds=30.0):
    rng = numpy.random.default_rng(22)
    length = int(seconds * RATE)
    raw = length + int(LOOP_CROSSFADE * RATE)
    ticks = low_pass(drop_ticks(rng, raw, 900.0, 300.0, 4000.0, 6.0), 2000.0)
    body = band_pass(rng.standard_normal(raw), 150.0, 600.0) * (0.7 + 0.3 * smooth_random(rng, raw, 1.5))
    mix = ticks / max(float(numpy.abs(ticks).max()), 1e-9) + 0.6 * body / max(float(numpy.abs(body).max()), 1e-9)
    return normalise(make_loop(mix, length), 0.6)


def drips(seconds=20.0):
    """Soft water taps and tiny splashes; no pitched glides (the old loop sounded like laser shots)."""
    rng = numpy.random.default_rng(23)
    length = int(seconds * RATE)
    raw = length + int(LOOP_CROSSFADE * RATE)
    out = numpy.zeros(raw)
    start = rng.uniform(0.1, 0.5)
    while start * RATE + 0.2 * RATE < raw:
        n = int(0.15 * RATE)
        t = numpy.arange(n) / RATE
        noise = rng.standard_normal(n)
        tap = band_pass(noise, 350.0, 1600.0) * numpy.exp(-t / 0.009)
        splash = band_pass(noise, 1800.0, 6500.0) * numpy.exp(-t / 0.025)
        plink = fade(tap + 0.4 * splash, 0.001)
        begin = int(start * RATE)
        out[begin:begin + n] += plink * rng.uniform(0.3, 1.0)
        start += rng.uniform(0.3, 2.0)
    return normalise(make_loop(out, length), 0.5)


def ghost_choir(seconds=24.0):
    """An original wordless, breath-heavy choir heard through a wall. Long, overlapping vowels,
    unstable unisons and a dissonant upper voice; no attacks, pitch sweeps or recognisable melody.
    The runtime supplies the swell envelope and long rests between phrases."""
    rng = numpy.random.default_rng(61410)
    length = int(seconds * RATE)
    t = numpy.arange(length) / RATE
    out = numpy.zeros(length)
    for index, fundamental in enumerate((98.0, 98.7, 146.83, 155.56, 196.0, 197.1)):
        wander = smooth_random(rng, length, 2.7, -0.35, 0.35)
        phase = 2 * numpy.pi * numpy.cumsum(fundamental + wander) / RATE
        voice = numpy.zeros(length)
        for harmonic in range(1, 19):
            frequency = fundamental * harmonic
            # Dark "oh/ah" formants; suppress the fundamental so this never becomes a horn.
            weight = (numpy.exp(-((frequency - 620.0) / 230.0) ** 2)
                      + 0.45 * numpy.exp(-((frequency - 1150.0) / 320.0) ** 2)) / harmonic
            voice += weight * numpy.sin(phase * harmonic + rng.uniform(0.0, 2 * numpy.pi))
        air = band_pass(rng.standard_normal(length), 400.0, 2600.0)
        voice = normalise(voice, 0.25) + normalise(air, 0.28)
        envelope = smooth_random(rng, length, 3.5, 0.05, 1.0) ** 2
        voice *= envelope
        out += voice
        for delay, gain in ((0.193, 0.36), (0.431, 0.25), (0.797, 0.17), (1.31, 0.12)):
            offset = int((delay + index * 0.013) * RATE)
            if offset < length:
                out[offset:] += voice[:-offset] * gain
    return normalise(fade(low_pass(out, 2100.0), 3.0), 0.5)


def rolling_swells(rng, length, count, attack):
    t = numpy.arange(length) / RATE
    envelope = numpy.zeros(length)
    for _ in range(count):
        centre = rng.uniform(attack, t[-1] * 0.8)
        width = rng.uniform(0.3, 1.2)
        envelope += rng.uniform(0.4, 1.0) * numpy.exp(-((t - centre) / width) ** 2)
    envelope *= numpy.minimum(1.0, t / attack) * numpy.exp(-t / (t[-1] * 0.6))
    return envelope / max(float(envelope.max()), 1e-9)


def brown(rng, length):
    noise = numpy.cumsum(rng.standard_normal(length))
    return noise - low_pass(noise, 15.0)


def thunder_near(index):
    rng = numpy.random.default_rng(300 + index)
    length = int(rng.uniform(5.0, 8.0) * RATE)
    crack_length = int(0.03 * RATE)
    crack = signal.sosfilt(signal.butter(2, 2000.0, btype="highpass", fs=RATE, output="sos"),
                           rng.standard_normal(crack_length)) * numpy.hanning(crack_length)
    rumble = low_pass(brown(rng, length), 200.0) * rolling_swells(rng, length, int(rng.integers(3, 7)), 0.15)
    out = 0.55 * rumble / max(float(numpy.abs(rumble).max()), 1e-9)
    out[:crack_length] += crack / max(float(numpy.abs(crack).max()), 1e-9)
    return normalise(fade(out, 0.002), 0.95)


def thunder_far(index):
    rng = numpy.random.default_rng(310 + index)
    length = int(rng.uniform(6.0, 10.0) * RATE)
    rumble = low_pass(brown(rng, length), 120.0) * rolling_swells(rng, length, int(rng.integers(3, 6)), 0.8)
    return normalise(fade(rumble, 0.05), 0.8)


def whisper(index):
    """Unvoiced, unintelligible syllables: breath noise through gliding vowel formants, with sibilant and
    plosive onsets, under a phrase-like rise and fall."""
    rng = numpy.random.default_rng(400 + index)
    length = int(rng.uniform(1.2, 2.8) * RATE)
    out = numpy.zeros(length)
    position = int(rng.uniform(0.02, 0.08) * RATE)
    while position < length - int(0.15 * RATE):
        size = int(rng.uniform(0.12, 0.22) * RATE)
        size = min(size, length - position)
        noise = rng.standard_normal(size)
        f1a, f1b = rng.uniform(300.0, 800.0, 2)
        f2a, f2b = rng.uniform(900.0, 2300.0, 2)
        glide = numpy.linspace(0.0, 1.0, size)
        vowel = (band_pass(noise, *band(f1a, 0.4)) * (1 - glide) + band_pass(noise, *band(f1b, 0.4)) * glide) * 0.7
        vowel += (band_pass(noise, *band(f2a, 0.35)) * (1 - glide) + band_pass(noise, *band(f2b, 0.35)) * glide) * 1.0
        vowel += band_pass(noise, 1800.0, 5000.0) * 0.5
        syllable = vowel * numpy.hanning(size)
        roll = rng.uniform()
        if roll < 0.4:
            hiss = int(rng.uniform(0.04, 0.12) * RATE)
            sibilant = band_pass(rng.standard_normal(hiss), 4000.0, 8000.0) * numpy.hanning(hiss) * 1.4
            syllable = numpy.concatenate([sibilant, syllable])
        elif roll < 0.6:
            burst = int(0.01 * RATE)
            plosive = band_pass(rng.standard_normal(burst), 1000.0, 6000.0) * numpy.hanning(burst) * 1.5
            syllable = numpy.concatenate([plosive, syllable])
        end = min(length, position + len(syllable))
        out[position:end] += syllable[:end - position]
        position = end + int(rng.uniform(0.03, 0.08) * RATE)
    t = numpy.arange(length) / RATE
    phrase = numpy.maximum(numpy.sin(numpy.pi * t / t[-1]), 0.0) ** 0.6
    return normalise(fade(out * phrase, 0.02), 0.6)


def heartbeat():
    rng = numpy.random.default_rng(500)
    length = int(0.9 * RATE)
    out = numpy.zeros(length)
    for start, pitch, size, gain in ((0.0, 55.0, 0.08, 1.0), (0.28, 45.0, 0.07, 0.7)):
        n = int(size * RATE)
        t = numpy.arange(n) / RATE
        thump = numpy.sin(2 * numpy.pi * pitch * t) * numpy.hanning(n)
        thump += 0.3 * numpy.tanh(3.0 * thump)
        thump += 0.15 * low_pass(rng.standard_normal(n), 250.0) * numpy.hanning(n)
        begin = int(start * RATE) + int(0.01 * RATE)
        out[begin:begin + n] += gain * thump
    return normalise(out, 0.8)


def breath(index):
    rng = numpy.random.default_rng(510 + index)
    length = int(rng.uniform(1.2, 1.8) * RATE)
    t = numpy.arange(length) / RATE
    air = band_pass(rng.standard_normal(length), 300.0, 2500.0)
    if index % 2 == 0:
        envelope = (t / t[-1]) ** 1.5 * numpy.minimum(1.0, (t[-1] - t) / 0.08)
    else:
        envelope = numpy.minimum(1.0, t / 0.08) * numpy.exp(-2.5 * t / t[-1])
    return normalise(fade(air * envelope, 0.02), 0.4)


def footstep(index):
    rng = numpy.random.default_rng(520 + index)
    length = int(0.25 * RATE)
    t = numpy.arange(length) / RATE
    click = int(0.004 * RATE)
    out = numpy.zeros(length)
    out[:click] += signal.sosfilt(signal.butter(2, 3000.0, btype="highpass", fs=RATE, output="sos"),
                                  rng.standard_normal(click)) * 0.6
    thud_pitch = rng.uniform(80.0, 150.0)
    out += numpy.sin(2 * numpy.pi * thud_pitch * t) * numpy.exp(-t / 0.02) * 0.8
    out += low_pass(rng.standard_normal(length), 300.0) * numpy.exp(-t / 0.03) * 0.3
    out += band_pass(rng.standard_normal(length), 900.0, 2500.0) * numpy.exp(-t / 0.04) * 0.35
    return normalise(fade(out, 0.002), 0.8)


def rumble(seconds=12.0):
    rng = numpy.random.default_rng(530)
    length = int(seconds * RATE)
    raw = length + int(LOOP_CROSSFADE * RATE)
    t = numpy.arange(raw) / RATE
    deep = low_pass(brown(rng, raw), 70.0)
    deep = deep / max(float(numpy.abs(deep).max()), 1e-9)
    throb = numpy.sin(2 * numpy.pi * numpy.cumsum(24.0 + 6.0 * smooth_random(rng, raw, 2.0, -1.0, 1.0)) / RATE)
    grit = band_pass(rng.standard_normal(raw), 80.0, 220.0) * smooth_random(rng, raw, 0.3)
    mix = deep + 0.5 * throb * (0.6 + 0.4 * smooth_random(rng, raw, 1.0)) + 0.35 * grit / max(float(numpy.abs(grit).max()), 1e-9)
    return normalise(make_loop(mix, length), 0.9)


def groan(index):
    """The building under stress: stick-slip impulses exciting slowly sweeping low resonances."""
    rng = numpy.random.default_rng(540 + index)
    length = int(rng.uniform(2.0, 4.0) * RATE)
    impulses = numpy.zeros(length)
    position = 0.0
    while position < length:
        impulses[int(position)] = rng.uniform(0.5, 1.0)
        position += RATE / rng.uniform(25.0, 70.0)
    start, end = (80.0, 300.0) if rng.uniform() < 0.5 else (300.0, 80.0)
    sweep = numpy.linspace(0.0, 1.0, length)
    out = numpy.zeros(length)
    for step in range(4):
        centre = start + (end - start) * step / 3.0
        weight = numpy.exp(-((sweep - step / 3.0) / 0.25) ** 2)
        out += band_pass(impulses, centre * 0.92, centre * 1.08) * weight
    return normalise(fade(out, 0.1), 0.7)


def debris(index):
    rng = numpy.random.default_rng(550 + index)
    length = int(rng.uniform(0.6, 1.5) * RATE)
    out = numpy.zeros(length)
    for hit in range(int(rng.integers(3, 10))):
        size = int(rng.uniform(0.02, 0.08) * RATE)
        begin = int(rng.uniform(0.0, 0.8) * (length - size))
        low = rng.uniform(500.0, 2000.0)
        burst = band_pass(rng.standard_normal(size), low, min(low * 3.0, 4000.0)) * numpy.exp(-numpy.arange(size) / (0.012 * RATE))
        out[begin:begin + size] += burst * rng.uniform(0.3, 1.0) * (1.0 - begin / length) ** 0.5
    return normalise(fade(out, 0.005), 0.7)


def buzz(seconds=4.0):
    rng = numpy.random.default_rng(560)
    length = int(seconds * RATE)
    raw = length + int(LOOP_CROSSFADE * RATE)
    t = numpy.arange(raw) / RATE
    hum = sum(numpy.sin(2 * numpy.pi * 100.0 * n * t + rng.uniform(0, 6.28)) / n for n in range(1, 13))
    crackle_ticks = drop_ticks(rng, raw, 30.0, 2000.0, 8000.0, 1.0)
    mix = hum / max(float(numpy.abs(hum).max()), 1e-9) + 0.25 * crackle_ticks / max(float(numpy.abs(crackle_ticks).max()), 1e-9)
    return normalise(make_loop(mix, length), 0.5)


def spark(index):
    rng = numpy.random.default_rng(570 + index)
    length = int(0.3 * RATE)
    t = numpy.arange(length) / RATE
    out = signal.sosfilt(signal.butter(2, 3000.0, btype="highpass", fs=RATE, output="sos"), rng.standard_normal(length))
    out *= numpy.exp(-t / 0.05) * 0.5
    out[:int(0.002 * RATE)] += rng.standard_normal(int(0.002 * RATE)) * 1.5
    return normalise(fade(out, 0.001), 0.9)


def crackle(seconds=20.0):
    rng = numpy.random.default_rng(580)
    length = int(seconds * RATE)
    raw = length + int(LOOP_CROSSFADE * RATE)
    pops = numpy.zeros(raw)
    position = 0.0
    while position < raw:
        size = int(rng.uniform(0.001, 0.008) * RATE)
        begin = int(position)
        if begin + size < raw:
            low = rng.uniform(1000.0, 3000.0)
            pops[begin:begin + size] += band_pass(rng.standard_normal(size), low, low * 2.0) * rng.uniform(0.2, 1.0)
        position += RATE / rng.uniform(5.0, 25.0)
    roar = low_pass(rng.standard_normal(raw), 400.0) * (0.7 + 0.3 * smooth_random(rng, raw, 0.5))
    mix = pops / max(float(numpy.abs(pops).max()), 1e-9) + 0.4 * roar / max(float(numpy.abs(roar).max()), 1e-9)
    return normalise(make_loop(mix, length), 0.6)


def bell_faint():
    """An electric school bell heard from far down the building: a 22 Hz striker on a ringing gong, muffled."""
    rng = numpy.random.default_rng(590)
    length = int(2.5 * RATE)
    t = numpy.arange(length) / RATE
    ring = numpy.sin(2 * numpy.pi * 900.0 * t) + 0.6 * numpy.sin(2 * numpy.pi * 1900.0 * t + 1.0)
    striker = (signal.square(2 * numpy.pi * 22.0 * t) + 1.0) / 2.0
    out = low_pass(ring * (0.4 + 0.6 * striker), 1500.0)
    out *= numpy.minimum(1.0, t / 0.05) * numpy.minimum(1.0, (t[-1] - t) / 0.3)
    tail = rng.standard_normal(int(0.8 * RATE)) * numpy.exp(-numpy.arange(int(0.8 * RATE)) / (0.25 * RATE))
    out = signal.fftconvolve(out, tail / numpy.abs(tail).sum() * 40.0)[:length] + 0.5 * out
    return normalise(fade(out, 0.02), 0.4)


def band(centre, width):
    return centre * (1.0 - width / 2.0), centre * (1.0 + width / 2.0)


def write_wav(samples, path):
    pcm = (numpy.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        handle.writeframes(pcm.tobytes())


def main():
    os.makedirs(SOUND_DIR, exist_ok=True)
    os.makedirs(MURMUR_DIR, exist_ok=True)
    stale = os.path.join(SOUND_DIR, "frogs.wav")
    if os.path.exists(stale):
        os.remove(stale)
    files = {name: make() for name, make in (
        ("drone", drone), ("wind", wind), ("crickets", crickets), ("rain_outdoor", rain_outdoor),
        ("rain_roof", rain_roof), ("drips", drips), ("rumble", rumble), ("buzz", buzz), ("crackle", crackle),
        ("heartbeat", heartbeat), ("bell_faint", bell_faint), ("ghost_choir", ghost_choir))}
    for index in range(6):
        files["creak_%d" % index] = creak(index)
    for index in range(3):
        files["thunder_near_%d" % index] = thunder_near(index)
        files["thunder_far_%d" % index] = thunder_far(index)
        files["groan_%d" % index] = groan(index)
        files["spark_%d" % index] = spark(index)
    for index in range(12):
        files["whisper_%d" % index] = whisper(index)
    for index in range(2):
        files["breath_%d" % index] = breath(index)
    for index in range(4):
        files["step_%d" % index] = footstep(index)
        files["debris_%d" % index] = debris(index)
    for name, samples in files.items():
        write_wav(samples, os.path.join(SOUND_DIR, name + ".wav"))
    for index in range(8):
        write_wav(murmur(index), os.path.join(MURMUR_DIR, "murmur_%d.wav" % index))
    print("Wrote %d sounds to %s and 8 murmurs to %s" % (len(files), SOUND_DIR, MURMUR_DIR))


if __name__ == "__main__":
    main()
