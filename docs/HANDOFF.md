# Handoff - state of the work

Last updated: 2026-10-23, during the Kerala retarget. Read this first if you are picking the project up.

## What this is

A psychological-horror audio system for a first-person game set in an abandoned school: an offline Python
pipeline that renders binaural assets and a composed showcase, plus a Godot 4 adaptive director. The project is
being retargeted from a British-flavoured school to a **Kerala government school built over a cleared kaavu
(sacred grove)** - Malayalam voices, monsoon, Theyyam. Direction: `docs/PLAN_KERALA.md`. Autopsy of the rejected
legacy scripts: `docs/AUDIT.md`.

## Done

| Area | Where | Status |
|---|---|---|
| Source fetching (field recordings, CC0 instruments, HRTF, rooms, speech, TTS) | `pipeline/fetch_sources.py` | done; Kerala archives fetched by hand so far (see below) |
| DSP core: I/O, filters, loudness, true-peak limiter, seamless loops | `pipeline/school/core.py` | done |
| Spatial engine: KEMAR HRTF binaural, near field, hybrid measured+synthetic rooms, occlusion, portals | `school/space.py` | done |
| Voice: Kokoro TTS + WORLD (whisper, child, freeze, stretch, sing, vocalise), PA speaker, tape, breath | `school/voice.py` | done |
| Foley models: steps, knocks, doors, chalk, chair, clock, drips, electric bell, lockers | `school/matter.py` | done |
| Beds from real recordings | `school/bed.py` | done |
| Score: upright piano, string clusters, bowed metal, sympathetic bloom, reverse swell | `school/score.py` | done |
| British showcase "The Register" (calibrated `LEVEL` mix) | `school/showcase.py` | code done; v1 render was unbalanced (`showcase/drafts/`), calibrated v2 not yet re-rendered |
| QC: LUFS / dBTP / seam / spectral checks, overview plot, ASR on the master | `school/qc.py` | done, not yet run on final |
| Godot asset library generator + manifest | `school/library.py` | code done, not yet rendered |
| Godot director (states 0-5, memory, motifs, hooks) | `godot/audio_system/school_audio.gd` | first version; not yet validated headless, no project.godot yet |
| Distant-room song tool + original Carnatic stand-in | `pipeline/distant_room.py`, `school/carnatic.py` | done; `showcase/distant_room_standin.mp3` |
| Kerala objects: iron bell, ceiling fan, chilanka, monsoon on tiles, Kerala night | `school/kerala.py` | written, untested |
| Malayalam cast (MMS / Indic-Parler TTS, kids, class unison, pledge) | `school/cast_ml.py` | written, untested |
| Teacher in the project owner's voice (kNN-VC) | `school/clone.py` | works; needs pitch match (see next) |

## In flight / next

1. **Archive mining**: `build/analysis/tag.py` runs the AudioSet AST tagger over the Theyyam archives
   (`.cache/sources/kerala/`) -> `build/analysis/tags.json`. Use `kerala.find("Drum roll")`, `"Chant"`,
   `"Yell"`, `"Cymbal"`, `"Crowd"` etc. to cut chenda / ilathalam / thottam / cries.
2. **Teacher voice**: kNN-VC keeps the source prosody, so converted lines sit at ~129 Hz vs the owner's ~149 Hz.
   Scale F0 (WORLD) by ~1.15 before conversion. Test Indic-Parler vs MMS for more natural Malayalam prosody
   (`SCHOOL_AUDIO_ML_TTS=parler`).
3. **Kerala showcase "HAAJAR"** (`school/showcase_kerala.py`, to write) - structure in `docs/PLAN_KERALA.md`;
   reuse `Mix.cue` + `LEVEL` calibration from `school/showcase.py`.
4. Possession cry: `voice.vocalise` on a vowel from Sir's cloned speech, oracle F0 contour (fry -> tremolo ->
   shriek), layered with archival cries.
5. Rebuild library with Kerala assets, update the Godot director profile, add `project.godot`, validate headless
   with the Godot 4.3 binary in `.cache/sources/godot/`.
6. QC, listening guide, integration guide, final commit.

## Environment notes (cloud container)

- Python deps: numpy scipy soundfile pyloudnorm matplotlib pyworld h5py kokoro-onnx faster-whisper torch(cpu)
  torchaudio(cpu, from the PyTorch index - the PyPI build fails to load) transformers parler-tts.
- Source cache: `.cache/sources` (symlink, not committed). Re-fetch with `python3 pipeline/fetch_sources.py`.
  Kerala archives (archive.org NCAA items, Odayam night), OpenSLR 63 Malayalam speech, kNN-VC weights
  (github.com/bshall/knn-vc releases, symlinked into `~/.cache/torch/hub/checkpoints/`) were fetched manually;
  add them to `fetch_sources.py`.
- **The owner's voice sample is not committed** (privacy). It lives at `.cache/sources/user_voice/sample.mp3`;
  set `SCHOOL_AUDIO_TEACHER_VOICE` to point elsewhere. The derived matching set is in `build/cache/` (ignored).
- Licensing: the owner will clear archival and commercial material before release. The film song requested for
  the distant-room effect is **not** downloaded; the tool takes a licensed file.

## How to run

    python3 pipeline/fetch_sources.py
    python3 pipeline/build.py showcase      # British calibrated showcase -> showcase/
    python3 pipeline/distant_room.py --standin --seconds 30
    python3 pipeline/build.py library       # -> godot/audio/ + manifest.json
    python3 pipeline/build.py qc            # -> docs/QC_REPORT.md
