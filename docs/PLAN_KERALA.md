# Retarget: a Kerala school (work in progress)

Status: **in progress**. The engine (binaural rooms, voice resynthesis, Foley models, calibrated mixing, QC,
Godot director) is built and target-neutral. The first showcase draft (`showcase/drafts/`) was British-flavoured;
everything below replaces its cultural layer for a Kerala / Malayali audience. No Hindi anywhere.

## Concept: the school was built where the kaavu stood

A government school in north Malabar, built in the 1970s over a cleared *kaavu* - the sacred grove where the
village's Theyyam was performed every year. The grove was cut, the shrine's stones went into the foundations, the
Theyyam stopped. The building is not haunted by a ghost. It is *occupied by a deity nobody performs any more*, and
it has spent fifty years listening to children, learning them by ear, waiting for someone to stand in its place.

Theyyam is a living tradition. The design treats it with respect: the horror is never "Theyyam is evil" - it is
the displaced, unperformed presence, and the possession of an ordinary school by something the school forgot.
Instruments and vocal styles are *inspired by* the ritual (chenda, ilathalam, chengila, kuzhal, thottam chant,
the possessed oracle's trembling cry), heard wrongly: through walls, from a grove that no longer exists, inside
the PA.

## The rules carry over

1. Nothing supernatural is heard before its mundane twin. Teach the school first.
2. Silence the building's hum to announce the presence.
3. Patterns of three, and the third arriving behind you.
4. Every voice says something ordinary. The horror is who answers, from where, and when.

## Sound world, translated

| British draft | Kerala version |
|---|---|
| Rain on sash windows | **Monsoon** on a Mangalore-tile roof, overflowing eaves, water on laterite; frogs and night insects through open verandah corridors (real Kerala night recording) |
| Fluorescent tube hum | Tube light *and* a ceiling fan on a dying regulator - its slow wobble and squeak, the building's pulse (still 50 Hz: still the note G) |
| Electric school bell | The iron school bell: a hanging iron plate/rail struck with a rod by the peon. Long strokes for the end of day. Later: struck by no one |
| English register, "Here" | Malayalam roll call; pupils answer **"ഹാജർ" (haajar)** / "present, teacher". The missing girl: **Ammu** |
| Counting "one, two, three" | Hide-and-seek counting in Malayalam: **ഒന്ന്, രണ്ട്, മൂന്ന്...** |
| Assembly hymn | The **morning pledge** recited in unison by children ("ഇന്ത്യ എന്റെ രാജ്യമാണ്...") - recitation becomes chant, fourteen children become one voice |
| Piano exercise | A harmonium in the music room (and the school's old cassette player) - the swara exercise *sa ri ga ma...* that never reaches *pa* |
| String cluster | Kept as glue, joined by kuzhal-like reed drones and bowed metal; chenda only from the grove that isn't there |
| Breath, whispers | Malayalam whispers; the velichappadu's (oracle's) trembling breath and **possession cry** at the climax |
| "Thank you. Everyone's here now." | **"നന്ദി. ഇപ്പോൾ എല്ലാവരും ഇവിടെയുണ്ട്."** - Thank you. Now everyone is here. |

## The presence: Theyyam heard through a school

- From far off, inside the walls: **chenda** rolls accelerating the way they do before the deity "arrives",
  **ilathalam** cymbals, a **conch**, the crowd's answering cries - real archival Theyyam recordings, placed in a
  grove that is now the playground, heard through laterite and closed shutters.
- Close: **chilanka** (anklet bells) jingling with footsteps behind you - the performer walking the corridor.
- The PA voice, possessed: the teacher's line dissolves into the oracle's trembling, quavering cry (WORLD vocoder:
  her vocal tract, the deity's pitch contour, tremor at 6-9 Hz, subharmonic growl).
- The scream: an original possession cry (not copied from any film) - a long guttural rise from vocal fry into a
  piercing ululating shriek, built from real voice material. Earned by four minutes of restraint, then cut.

## Requested: a song from a distant room (30 s binaural)

Request: *"Oru Murai Vandhu Paarthaya"* playing from a distant room. That is a commercial film recording
(Manichitrathazhu, 1993), so the pipeline will **not** download it from unofficial uploads. Instead:

- `pipeline/distant_room.py <your_licensed_file> --start 0:00 --seconds 30` renders any track as if playing on an
  old radio/cassette in a locked room upstairs: speaker colouration, tape wow, the room's own reverb, door and
  floor occlusion, a slow binaural drift - and the song stopping when you get close.
- A demo render uses an **original** stand-in: a woman singing a Carnatic-style phrase in a locked room.

## Sources being added

- Archival Theyyam / Padayani / Panchavadyam / Kathakali-instrument recordings (National Culture Audiovisual
  Archives via archive.org) - chenda, maddalam, chengila, ilathalam, kuzhal, thottam, crowd.
- Real Kerala night soundscape (Varkala): palms, crickets, frogs, distant temple.
- OpenSLR 63: real Malayalam speakers (male, female) - real Malayali voices for crowds, murmurs, fragments.
- Malayalam TTS (facebook/mms-tts-mal; testing ai4bharat Indic Parler-TTS for more natural voices), re-performed
  through the WORLD vocoder like every other voice.

Licensing for archival/commercial material is to be cleared by the project owner before release; every source is
listed in `pipeline/fetch_sources.py`.

## Next steps

1. Mine the archival recordings: locate chenda solos, cymbal passages, chants, cries, crowd (spectral + speech
   analysis), cut a source bank.
2. Malayalam cast: roll call, haajar answers, counting, pledge chorus, PA lines, whispers, possession cry.
3. Kerala beds: monsoon on tiles, fan, night insects/frogs, iron bell.
4. Re-compose the showcase on the calibrated mix (`LEVEL` table), Kerala version.
5. `distant_room.py` + stand-in demo.
6. Rebuild the Godot library and director profile with the Kerala assets; QC; docs.
