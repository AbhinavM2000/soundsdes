# Autopsy of the rejected scripts

`legacy/make_night_sounds.py` and `legacy/make_horror_audio.py` are kept unmodified for reference. Both are
competent engineering wrapped around sound design that cannot frighten anyone. This is why.

## The core failure: every sound announces that it is synthetic

| Legacy sound | How it was made | What the ear actually hears |
|---|---|---|
| `drone` | three sine waves (41 / 55.3 / 82.5 Hz) + low-passed brown noise, 60 s loop | A test tone over a subwoofer. Pure sines do not exist in buildings. It is the "constant bass rumble masquerading as tension" the brief bans, and it never develops: minute 1 is minute 10. |
| `ghost_choir` | additive sines on fixed fundamentals (98, 98.7, 146.8, 155.6, 196, 197.1 Hz), formant-weighted, plus four fixed delay taps | An organ pad. Static chord, no consonants, no breaths, no singers. The fixed taps comb-filter it into a metallic chorus. A choir is people; this is a synthesiser preset. |
| `whisper_*` | band-passed noise with random formant targets every 120-220 ms | "Syllable soup". Real whispers are a *mouth* moving continuously through consonants and vowels; random independent bandpass jumps are noise with a wobble. Fixed syllable length range = audible machine rhythm. |
| `murmur_*` | 25-55 Hz pulse train + noise through three fixed band-passes | Exactly what the owner called it: "horn-like pom pom". Three static formants = one frozen vowel = a horn. |
| `confession_*` (pass 2) | macOS `say -v Whisper` slowed to 0.85, plus a 0.63x octave-down copy | Robotic TTS, then the oldest cliché in horror: pitch it down to make it "demonic". Also not reproducible off macOS. |
| `ritual_chant` | Dies Irae recording, slowed, octave shadow, reversed snippets | Gregorian chant is the most worn-out horror signifier there is. It tells the player "scary music now" - the opposite of making them doubt their ears. |
| `bell_faint` | 900 Hz + 1900 Hz sines gated by a 22 Hz square wave | A smoke alarm. A school bell is a steel gong with dozens of inharmonic partials struck by a hammer; two harmonic sines are a beep. |
| `step_*` | sine thud (80-150 Hz) + click + noise, 4 variants | Identical anatomy every step: no heel/toe, no scuff, no surface, no grit, no gait. Four variants rotate audibly. |
| `creak_*` | impulse train through *fixed* narrow band-passes | A buzz at one pitch. Real creaks are stick-slip whose rate *and* resonance move with load. |
| `heartbeat`, `breath_*` | sine thumps; band-passed noise with an envelope | Heartbeats are a cliché; the breaths are filtered noise with static bands (we verified the same flaw in a first-draft breath here and threw it out). |
| `presence_impact` | noise + 5 sines, exponential decay | A generic trailer "boom". A jump scare with no build-up and no source. |
| `crickets` | sine chirps at ~1 s period, +/-5 % jitter | Wrong location (outdoors), and the near-regular period reveals the loop within seconds. |

## Acoustics: there were no rooms

Both scripts' `room()` is four to six discrete delay taps through a low-pass. Discrete taps at fixed times are a
**flutter/slapback effect**, not a space: no diffusion, no frequency-dependent decay, no early-reflection pattern,
no interaural decorrelation. Every sound was in the same non-room, so nothing could be "behind a door", "in the
gym", or "upstairs". Without believable space, nothing can be *wrong* about space - the brief's most powerful
weapon (spatial paranoia) was unavailable.

## Everything was mono

`write_wav` hard-codes one channel. No direction, no distance, no near-field. A whisper "at your ear" and a door
"down the corridor" were the same signal at different gains.

## Gain staging was destroyed by design

Every generator ends in `normalise(x, peak)` with an arbitrary per-sound peak (whisper 0.6, thunder 0.95). Relative
loudness - the most important relationship in a horror mix - is discarded and has to be reinvented by hand at
runtime. There is no loudness measurement, no true-peak control, no headroom policy.

## Time: predictability everywhere

- Loops of 4-60 s (`buzz` is 4 s). A 4 s loop is identifiable in under a minute.
- Fixed-ratio jitter (`period * uniform(0.95, 1.05)`) produces *periodic* events with a little wobble - the ear
  locks onto the period.
- No state, no memory, no consequences. A sound in minute 20 means exactly what it meant in minute 1.

## Writing: telling instead of implying

"That is not your teacher." "Please. Do not look at my face." "There is room. For one. More." Every line explains
the horror to the player, so there is nothing left for the player to fear. Dread lives in the gap between what is
said and what it implies; these lines close the gap.

## What was worth keeping

- Seeded generators and the idea of variation sets.
- `smooth_random` (eased random curves) - kept as `core.smooth_noise`.
- Seamless-loop construction - kept, but replaced the linear 2 s crossfade (which dips -3 dB on uncorrelated
  ambience) with an equal-power 4 s crossfade and added an automated seam test.
- The pipeline shape: one script renders deterministic assets into the Godot tree.

## The replacement in one paragraph

Real recordings where reality wins (rain, ventilation, transformer hum, children at break, stairwells, a bowed
brake drum, an upright piano, string sections), physical models where control matters (modal synthesis for
doors, pipes, chairs, chalk, clocks, bells, footsteps), and neural speech re-performed through a vocoder so voices
keep their humanity (true whispers, children, a mouth that stops moving). Everything placed binaurally (KEMAR
HRTF) inside hybrid rooms built on real dummy-head measurements. A composition built on one idea - the building
hums G and nothing in it is allowed to finish - and an adaptive system that remembers what the player has heard.
