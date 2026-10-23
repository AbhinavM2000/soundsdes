"""The interactive asset library for Godot -> godot/audio/ + godot/audio/manifest.json

Categories (and how the runtime uses them):
  beds/   stereo seamless loops (OGG, loop flag set at runtime), one per room x layer
  world/  MONO, DRY one-shots for AudioStreamPlayer3D; Godot adds distance + the room's reverb bus
  voice/  MONO lines; PA lines already carry the loudspeaker so the runtime only places speakers
  head/   BINAURAL stereo, pre-rendered with HRTF + near field - for things that happen at your ears, which
          Godot's own panner cannot do convincingly. Played 2D on the Head bus (no reverb).
  music/  stereo score layers; loops where the director cross-fades them by intensity
Sequenced motifs (three knocks, the master clock, counting, the piano exercise) ship as single units so the
director can play them with intention: three knocks, then two, then the third one behind you.

Levels: every file is normalised to a per-category integrated loudness (LEVELS) with a -2 dBTP ceiling, so the
runtime starts from a calibrated mix and only applies intent.
"""
import json

import numpy as np

from . import bed, cast, matter as M, score as S, voice as V
from .core import (RATE, ROOT, db, fade, hp, limiter, lp, lufs, rng, sec, seamless_loop, true_peak_db, write,
                   peak_norm)
from .space import place, place_head, place_remote, spread

OUT = ROOT / "godot" / "audio"
LEVELS = {"bed": -30.0, "world": -20.0, "voice": -22.0, "head": -26.0, "music": -28.0}


class Library:
    def __init__(self):
        self.assets = []

    def add(self, rel, x, category, loop=False, states=(0, 1, 2, 3, 4, 5), cooldown=0.0, tags=(), meta=None,
            level=None):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 2 and x.shape[0] != 2:
            x = x.T
        x = x - (x.mean(axis=-1, keepdims=True) if not loop else 0)
        x = hp(x, 20, 2)
        target = LEVELS[category] if level is None else level
        loud = lufs(x if x.ndim == 2 else np.vstack([x, x]))
        if loud > -70:
            x = x * db(target - loud)
        if true_peak_db(x) > -2.0:
            if loop:
                x = x * db(-2.0 - true_peak_db(x))
            else:
                x = limiter(x if x.ndim == 2 else np.vstack([x, x]), -2.0)
                x = x if x.shape[0] == 2 and rel.startswith(("beds", "head", "music")) else x[0]
        if not loop:
            x = fade(x, 0.0005, 0.02)
        path = OUT / rel
        write(path, x, bits=16) if path.suffix == ".wav" else write(path, x)
        self.assets.append({"id": rel.rsplit(".", 1)[0].replace("/", "."), "file": rel, "category": category,
                            "loop": loop, "channels": 1 if x.ndim == 1 else 2,
                            "seconds": round(x.shape[-1] / RATE, 3), "states": list(states), "cooldown": cooldown,
                            "tags": list(tags), "meta": meta or {}})
        print("  %-48s %6.2fs" % (rel, x.shape[-1] / RATE))

    def save(self):
        (OUT / "manifest.json").write_text(json.dumps({"rate": RATE, "levels_lufs": LEVELS,
                                                       "assets": self.assets}, indent=1))


def loop_of(builder, seconds, crossfade=4.0):
    return seamless_loop(builder(seconds + crossfade), sec(seconds), crossfade)


# ---------------------------------------------------------------- sections

def beds(lib):
    for room in ("classroom", "corridor", "stairwell"):
        lib.add("beds/rain_%s.ogg" % room, bed.rain(120, 11, room, loop=True), "bed", loop=True, states=(0, 1, 2, 3, 5),
                tags=("weather", room))
    for room in ("classroom", "corridor", "stairwell", "hall"):
        lib.add("beds/roomtone_%s.ogg" % room, bed.room_tone(120, 21, room, loop=True), "bed", loop=True, tags=("room", room))
    lib.add("beds/wind_slit.ogg", bed.wind(90, 3, "slit", loop=True), "bed", loop=True, states=(0, 1, 2, 3))
    lib.add("beds/wind_stair.ogg", bed.wind(90, 4, "stair", loop=True), "bed", loop=True, states=(0, 1, 2, 3))
    lib.add("beds/vent.ogg", bed.vent(90, 5, 0.0, loop=True), "bed", loop=True, states=(0, 1, 2, 3))
    lib.add("beds/vent_breathing.ogg", bed.vent(90, 6, 1.0, rate=0.24, loop=True), "bed", loop=True, states=(3, 4),
            meta={"breath_rate_hz": 0.24})
    # the hum is a 3D source: mono loop, played at the fitting
    lib.add("beds/hum_tube.ogg", bed.hum(60, 7, flicker=0.3, loop=True), "bed", loop=True, states=(0, 1, 2, 3),
            tags=("3d", "hum"), meta={"note": "the building's G; silence it to announce the presence"})
    lib.add("beds/radiator.ogg", bed.radiator(90, 8, loop=True).mean(axis=0), "bed", loop=True, tags=("3d",))
    # memory of people: real children, only ever heard through something
    yard = bed.yard(60, 9, loop=True)
    lib.add("beds/yard_through_glass.ogg", lp(yard, 1800) * 0.7, "bed", loop=True, states=(1, 2), level=-38)


def world(lib):
    for surface, count in (("lino", 12), ("wood", 8), ("stone", 8)):
        for i in range(count):
            lib.add("world/step_%s_%02d.wav" % (surface, i), M.footstep(7000 + i, surface), "world",
                    tags=("step", surface), level=-24)
    for material, count in (("door", 6), ("pipe", 6), ("wall", 4)):
        for i in range(count):
            lib.add("world/knock_%s_%02d.wav" % (material, i), M.knock(300 + i, material), "world",
                    tags=("knock", material), meta={"motif_timing": list(M.THREE)})
    for i in range(4):
        lib.add("world/door_creak_%02d.wav" % i, M.door_creak(40 + i), "world", cooldown=20)
    for i in range(3):
        lib.add("world/door_slam_%02d.wav" % i, M.door_slam(50 + i), "world", cooldown=60, level=-17)
    for i in range(4):
        lib.add("world/locker_hit_%02d.wav" % i, M.locker(60 + i, "hit"), "world", cooldown=15)
    for i in range(3):
        lib.add("world/locker_rattle_%02d.wav" % i, M.locker(70 + i, "rattle"), "world", cooldown=15)
    for i in range(4):
        lib.add("world/chalk_%02d.wav" % i, M.chalk(80 + i, 9.0), "world", states=(1, 2, 3), cooldown=90,
                tags=("anomaly",))
    for i in range(4):
        lib.add("world/chair_drag_%02d.wav" % i, M.chair_drag(90 + i, 3.0), "world", states=(2, 3, 4), cooldown=120,
                tags=("anomaly", "ceiling"))
    for i in range(6):
        lib.add("world/floor_creak_%02d.wav" % i, M.floor_creak(100 + i), "world", cooldown=8)
    for i in range(7):
        lib.add("world/clock_tick_%02d.wav" % i, M.clock_tick(i, tock=bool(i % 2)), "world", level=-30,
                tags=("clock",), meta={"note": "master clock: the director ticks every clock in the school together"})
    for i in range(12):
        lib.add("world/drip_%02d.wav" % i, M.drip(200 + i), "world", level=-28, tags=("drip",))
    for i in range(2):
        lib.add("world/bell_natural_%02d.wav" % i, M.school_bell(300 + i, ring=3.0), "world", cooldown=240, level=-18)
        lib.add("world/bell_wrong_%02d.wav" % i, M.school_bell(310 + i, ring=2.6, decay="wrong", tail=9.0), "world",
                states=(2, 3, 4), cooldown=300, level=-18, tags=("anomaly",))
    for i in range(3):
        lib.add("world/pa_click_on_%02d.wav" % i, V.pa_click("on%d" % i, True), "world", level=-26)
        lib.add("world/pa_click_off_%02d.wav" % i, V.pa_click("off%d" % i, False), "world", level=-28)
    # the piano: single notes so the director performs the exercise (and gets stuck) itself
    for m in (60, 62, 64, 65, 67):
        for v in range(2):
            lib.add("world/piano_%d_v%d.wav" % (m, v), S.piano(m, 0.35 + 0.2 * v, detune_cents=(-6, 5)[v]), "world",
                    level=-24, tags=("piano",), meta={"midi": m})
    lib.add("world/piano_sympathetic.wav", S.sympathetic(M.school_bell(320, ring=1.0)[:sec(2)], seed="lib"), "world",
            level=-30, tags=("piano",))


def voices(lib):
    pa_lines = {
        "good_morning": "Good morning, everyone.",
        "registration": "Registration, please.",
        "name_thomas_avery": "Thomas Avery.", "name_hannah_bell": "Hannah Bell.", "name_daniel_cole": "Daniel Cole.",
        "name_sophie_doyle": "Sophie Doyle.",
        "walk_dont_run": "Please walk. Don't run in the corridors.",
        "return_to_class": "Would the pupil in the corridor please go back to their classroom.",
        "stay_where_you_are": "Stay where you are, please. Someone will come and find you.",
        "thank_you": "Thank you.", "everyone_here": "Everyone's here now.",
        "assembly": "Would all pupils make their way to the main hall for assembly.",
    }
    for key, text in pa_lines.items():
        lib.add("voice/pa_%s.wav" % key, V.pa_speaker(cast.teacher(text), seed=key), "voice", tags=("pa",),
                meta={"text": text})
    lib.add("voice/pa_name_sophie_doyle_q.wav", V.pa_speaker(cast.teacher("Sophie Doyle?", 0.9), seed="sq"), "voice",
            tags=("pa",), meta={"text": "Sophie Doyle?"})
    lib.add("voice/pa_anyone_freezing.wav",
            V.pa_speaker(cast.teacher_freezing("Is there anyone I haven't called?", 0.32, 0.4, 1.7), seed="frz"), "voice",
            states=(3, 4), tags=("pa", "anomaly"), meta={"text": "Is there anyone I haven't called? (mouth stops)"})
    murmur = place(fade(bed.yard(10.0, 12).mean(axis=0), 1, 1), "hall", az=0, dist=14).mean(axis=0)
    lib.add("voice/pa_carrier_hall.wav", V.pa_speaker(murmur * 0.5, seed="carrier", hum=0.02, distortion=1.4), "voice",
            states=(1, 2), tags=("pa", "anomaly"), meta={"text": "(an open mic in a hall full of children)"})
    for i, (who, ans) in enumerate([(1, "Here."), (2, "Here, miss."), (3, "Here.")]):
        hall = place(cast.child_says(ans, who), "hall", az=(-25, 30, 5)[i], dist=(9, 12, 16)[i]).mean(axis=0)
        lib.add("voice/pa_answer_%02d.wav" % i, V.pa_speaker(hall * 1.6, seed="ans%d" % i), "voice", tags=("pa",),
                meta={"text": ans})
    for i in range(4):
        lib.add("voice/child_here_%02d.wav" % i, cast.child_says("Here.", i + 1, 0.85), "voice", states=(2, 3, 4),
                tags=("child",), meta={"text": "Here."})
    for n in range(1, 11):
        lib.add("voice/count_%02d.wav" % n, cast.count_word(n), "voice", states=(1, 2, 3), tags=("count",),
                meta={"n": n})
        lib.add("voice/count_whisper_%02d.wav" % n, cast.count_word(n, mood="whisper"), "voice", states=(3, 4),
                tags=("count", "whisper"), meta={"n": n}, level=-26)
    for i, name in enumerate(["Thomas Avery.", "Hannah Bell.", "Daniel Cole.", "Sophie Doyle."]):
        lib.add("voice/whisper_name_%02d.wav" % i, cast.whispered(name), "voice", states=(3, 4), tags=("whisper",),
                meta={"text": name}, level=-26)


def head(lib):
    for i in range(4):
        az = (158, -100, 120, -150)[i]
        lib.add("head/breath_in_%02d.wav" % i, place_head(cast.inhale(i * 3), az=az, el=5, dist=0.3), "head",
                states=(3, 4), cooldown=12, meta={"az": az})
    for i in range(3):
        az = (150, -95, 175)[i]
        lib.add("head/breath_out_%02d.wav" % i, place_head(cast.exhale(i), az=az, el=0, dist=0.35), "head",
                states=(3, 4), cooldown=12, meta={"az": az})
    for i, name in enumerate(["Thomas Avery.", "Hannah Bell.", "Daniel Cole."]):
        w = cast.whispered(name, (None, 2, None)[i])
        start = (120, -60, 200)[i]
        lib.add("head/whisper_circle_%02d.wav" % i,
                place(w, "stairwell", az=lambda t, s=start, k=i: s + t * (50, 45, -55)[k], el=15, dist=0.5, wet=0.12),
                "head", states=(4,), cooldown=30)
    lib.add("head/knock_behind.wav", place_head(M.knock(77, "door", 0.8), az=180, el=4, dist=0.45), "head",
            states=(4, 5), cooldown=600, level=-22, tags=("motif", "third_knock"))
    lib.add("head/sophie_reversed.wav",
            place(V.whisper(cast.teacher_reversed("Sophie Doyle.")), "stairwell", az=lambda t: 90 + t * 40, el=20,
                  dist=0.6, wet=0.2), "head", states=(4,), cooldown=120)
    lib.add("head/count_at_ear.wav", place_head(cast.count_word(10, mood="whisper"), az=35, el=0, dist=0.2), "head",
            states=(4,), cooldown=600, meta={"text": "Ten."})


def music(lib):
    far = S.cluster("lib-far", 94, [("Cello Section/susvib", 54, 0.9), ("Viola Section/trem", 56, 0.5),
                                     ("Solo Contrabass/SusNV", 42, 0.7), ("Violin Section/susVib", 66, 0.35),
                                     ("Violin Section/susVib", 68, 0.3)], drift_cents=45)
    far = place_remote(far.mean(axis=0), "hall", 10, -130, 9, "wall", "corridor")
    lib.add("music/strings_far.ogg", seamless_loop(far, sec(90), 4.0), "music", loop=True, states=(1, 2, 3))
    near = S.cluster("lib-near", 64, [("Cello Section/susvib", 54, 1.0), ("Cello Section/susvib", 55, 0.7),
                                       ("Viola Section/trem", 56, 0.6), ("Violin Section/susVib", 66, 0.5),
                                       ("Violin Section/susVib", 67, 0.45), ("Violin Section/susVib", 68, 0.5),
                                       ("Solo Contrabass/SusNV", 43, 0.8)], drift_cents=60, swell=True)
    near = place(near.mean(axis=0), "stairwell", az=0, dist=7, wet=1.6)
    lib.add("music/strings_near.ogg", seamless_loop(near, sec(60), 4.0), "music", loop=True, states=(3, 4))
    for kind, pitch in (("brake", 0.5), ("cymbal", 0.5), ("drum", 0.7)):
        bm = place(S.bowed_metal("lib-" + kind, 66, kind, pitch=pitch), "stairwell", az=60, el=-30, dist=6, wet=1.4)
        lib.add("music/bowed_%s.ogg" % kind, seamless_loop(bm, sec(60), 4.0), "music", loop=True, states=(2, 3, 4))
    # the assembly: crowd with the hush at a known moment, the hymn, and its corruption
    crowd = bed.yard(14.0, 13).mean(axis=0) * 0.7 + bed.gym(14.0, 3).mean(axis=0) * 0.3
    crowd = crowd * np.r_[np.linspace(0, 1, sec(6)) ** 2, np.ones(sec(8))]
    crowd[sec(12.0):] = 0.0
    lib.add("music/assembly_hush.ogg", place_remote(crowd, "hall", 12, 25, 7, "door_ajar", "stairwell", el=-35),
            "music", states=(2, 3), cooldown=600, meta={"hush_at": 12.0})
    singers = []
    r = rng("lib-choir")
    for k in range(14):
        d = r.uniform(6, 16)
        singers.append(np.pad(cast.hymn_voice(k), (sec(d / 343), 0)) / d)
    n = max(len(s) for s in singers)
    diverse = sum(np.pad(s, (0, n - len(s))) for s in singers)
    lib.add("music/hymn.ogg", place_remote(diverse, "hall", 14, 25, 7, "door_ajar", "stairwell", el=-35), "music",
            states=(2, 3), cooldown=900, meta={"beat": 0.62})
    from .core import resample
    same = cast.hymn_voice(0, same_voice=True)
    unison = np.zeros(len(same) + sec(0.05))
    for k in range(14):
        sh = resample(same, 1 + (k - 7) * 0.0006)
        dl = int(k * 0.0011 * RATE)
        unison[dl:dl + len(sh)] += sh[:len(unison) - dl] / 9
    lib.add("music/hymn_one_voice.ogg", place_remote(unison, "hall", 14, 25, 7, "door_ajar", "stairwell", el=-35),
            "music", states=(3, 4), cooldown=900, meta={"beat": 0.62, "note": "fourteen children, one voice"})
    swell_src = place(S.bowed_metal("swell", 6, "cymbal", 0.5), "stairwell", 0, 0, 3).mean(axis=0)
    lib.add("music/inhale_swell.wav", spread(S.reverse_swell(swell_src, 3.2, 3.0) * np.linspace(0, 1, sec(3.2)) ** 2, 0.8),
            "music", states=(4,), cooldown=300, level=-22, meta={"note": "ends exactly at the cut: play it 3.2 s early"})


def render_all():
    lib = Library()
    for section in (beds, world, voices, head, music):
        print(section.__name__)
        section(lib)
    lib.save()
    print("%d assets -> %s" % (len(lib.assets), OUT))
