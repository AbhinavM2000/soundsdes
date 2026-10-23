"""THE REGISTER - a composed 4'46" binaural showcase. Headphones.

The listener is a pupil who has come back to the school at night. Nothing in it is louder than a door.

  I    0:00  Ordinary abandonment   classroom 2B, rain, a dying fluorescent tube in the corridor, the wall clock
  II   0:40  First anomaly          the clock hesitates; the PA keys up and transmits a room full of children
  III  1:18  Recognition            corridor; a child counts your footsteps, and takes one step you didn't
  IV   2:12  Escalation             the register over the PA; Sophie Doyle doesn't answer - then she does,
                                    behind you; the bell upstairs that will not stop; the hush; the hymn
  V    3:25  Presence               the light dies, the building breathes, the names come to your ears
  VI   3:56  Absence                the third knock. Then nothing. Then the PA, very kind. Then the clock.

Mixing is calibrated, not hand-tuned: every cue is measured and placed at an explicit loudness (its loudest
1-second window, LUFS, before mastering), so the dynamic arc is designed in one table - LEVEL below.
"""
import numpy as np

from . import bed, cast, matter as M, score as S, voice as V
from .core import (RATE, ROOT, add, conv, db, fade, hp, limiter, lp, lufs, ramp, resample, rng, sec,
                   short_term_max, true_peak_db, tv_lowpass_fast, varispeed, write)
from .space import ROOMS, binaural, place, place_head, place_remote, spread

TOTAL = 286.0
STEMS = ("env", "foley", "voice", "score")

# The dynamic arc. Max 1-s loudness per cue (beds: integrated), LUFS, before the master stage.
LEVEL = {
    # world
    "rain_classroom": -35, "rain_corridor": -40, "rain_stairwell": -39, "roomtone": -47, "hum": -44,
    "wind_gust": -40, "wind_stair": -44, "vent": -46, "vent_breathing": -38, "absence_tone": -56,
    # foley
    "steps": -33, "steps_fast": -31, "drip": -45, "drip_far": -52, "clock": -43, "slam_far": -36,
    "pipe_tick": -44, "pipe_knocks": -33, "door_creak": -34, "door_knocks": -31, "chalk": -39,
    "copy_steps": -38, "bell": -27, "chair": -35, "knock_behind": -27, "clock_end": -40,
    # voice
    "pa_click": -38, "pa_carrier": -42, "count": -38, "count_extra": -35, "register": -31, "answer_behind": -32,
    "crowd": -36, "hymn": -33, "breath": -31, "whisper": -31, "pa_freeze": -29, "pa_last": -32,
    # score
    "piano_far": -41, "piano_near": -37, "strings_far": -44, "bloom": -40, "bowed": -40, "strings_presence": -33,
    "swell": -27,
}


class Mix:
    def __init__(self, seconds):
        self.n = sec(seconds)
        self.stems = {s: np.zeros((2, self.n + sec(12))) for s in STEMS}

    def put(self, stem, t, x, gain_db=0.0):
        x = x if x.ndim == 2 else np.vstack([x, x])
        add(self.stems[stem], x * db(gain_db), sec(t))

    def cue(self, stem, t, x, key, offset=0.0):
        """Place a cue so that its loudest 1-second window lands at LEVEL[key] + offset."""
        x = x if x.ndim == 2 else np.vstack([x, x])
        level = cue_loudness(x)
        if level > -90:
            self.put(stem, t, x, LEVEL[key] + offset - level)

    def bed(self, stem, x, key, points, t=0.0):
        """A bed at an integrated level LEVEL[key], with a gain automation curve [(seconds_from_t, dB), ...]."""
        x = x if x.ndim == 2 else np.vstack([x, x])
        g = LEVEL[key] - lufs(x)
        self.put(stem, t, x * db(ramp(x.shape[1], points)), g)


def cue_loudness(x, window=1.0, hop=0.25):
    import pyloudnorm
    meter = pyloudnorm.Meter(RATE, block_size=0.4)
    data = x.T
    if data.shape[0] <= sec(window):
        data = np.vstack([data, np.zeros((sec(window) - data.shape[0] + 1, 2))])
    best = -120.0
    for a in range(0, data.shape[0] - sec(window) + 1, sec(hop)):
        try:
            best = max(best, meter.integrated_loudness(data[a:a + sec(window)]))
        except ValueError:
            pass
    return best


def _sum(*xs):
    n = max(x.shape[1] for x in xs)
    out = np.zeros((2, n))
    for x in xs:
        out[:, :x.shape[1]] += x
    return out


# ---------------------------------------------------------------- recurring gestures

def steps(mix, t0, count, room, surface="lino", interval=0.56, seed="me", key="steps", slow_to=None):
    """The listener's own footsteps (below, slightly ahead). Returns the onset times."""
    onsets = []
    seq = np.zeros(sec(count * interval * 1.4 + 3))
    for t, s in M.walk(count, surface, interval, seed=seed, slow_to=slow_to):
        add(seq, s, sec(t))
        onsets.append(t0 + t)
    mix.cue("foley", t0, place(seq, room, az=0, el=-62, dist=1.45, wet=0.9), key)
    return onsets


def pa_render(line, speakers, room, seed="pa", source_room="hall"):
    """A line through the school PA: the voice is in a room (part of the signal), then each ceiling loudspeaker
    re-radiates it into the listener's space at its own distance - the multi-speaker slapback of a corridor."""
    sig = line
    if source_room:
        wet = conv(line, ROOMS[source_room].tail_for(2.0))[0] * 0.5
        sig = np.zeros(max(len(line), len(wet)))
        sig[:len(line)] += line
        sig[:len(wet)] += wet
    return _sum(*[place(V.pa_speaker(sig, seed="%s%d" % (seed, k)), room, az=az, el=el, dist=dist, wet=0.8)
                  for k, (az, el, dist) in enumerate(speakers)])


def click(mix, t, speakers, room, on=True, seed="c", offset=0.0):
    c = V.pa_click(seed, on)
    y = _sum(*[place(c, room, az=az, el=el, dist=dist, wet=0.8) for az, el, dist in speakers])
    mix.cue("voice", t, y, "pa_click", offset)


CLASS_PA = [(25, 60, 2.6)]
CORRIDOR_PA = [(18, 55, 3.2), (6, 22, 11.0), (3, 10, 21.0)]
NEAR_PA = [(0, 75, 1.6)]


# ---------------------------------------------------------------- the piece

def compose():
    mix = Mix(TOTAL)
    r = rng("showcase")

    # ======================= THE WORLD (beds and their automation) =======================
    mix.bed("env", bed.rain(90, 1, "classroom"), "rain_classroom", [(0, -40), (4, 0), (72, 0), (84, -14), (90, -60)])
    mix.bed("env", bed.rain(90, 2, "corridor"), "rain_corridor", [(0, -60), (6, 0), (80, 0), (90, -60)], t=72)
    # stairwell rain: ducks at the hush (~188 s); from 205 s the stairwell seals and it goes very far away
    rs = bed.rain(80, 3, "stairwell")
    rs = tv_lowpass_fast(rs, ramp(rs.shape[1], [(0, 16000), (49, 16000), (56, 450), (80, 300)]))
    mix.bed("env", rs, "rain_stairwell", [(0, -60), (8, 0), (32.1, 0), (32.4, -7), (38, -3), (49, -4), (56, -9), (80, -11)],
            t=156)
    mix.bed("env", bed.room_tone(84, 1, "classroom"), "roomtone", [(0, -40), (3, 0), (76, 0), (84, -60)])
    mix.bed("env", bed.room_tone(90, 2, "corridor"), "roomtone", [(0, -60), (6, 0), (84, 0), (90, -60)], t=76)
    mix.bed("env", bed.room_tone(80, 3, "stairwell"), "roomtone", [(0, -60), (8, 0), (49, 0), (56, -6), (80, -8)], t=158)

    # The fluorescent tube in the corridor: through the door, overhead, then behind. It dies at 205 s.
    hum_mono = bed.hum(210, 1, flicker=0.35, dropouts=[(96.0, 2.6)])
    hum_mono = hum_mono if hum_mono.ndim == 1 else hum_mono.mean(axis=0)
    path_az = lambda t: 95.0 if t < 79 else (95 - (t - 79) * 18 if t < 84 else (-10 if t < 160 else 180))
    path_dist = lambda t: 7.0 if t < 79 else (max(2.0, 7.0 - (t - 79)) if t < 84 else (2.2 if t < 160 else 2.2 + (t - 160) * 1.3))
    path_el = lambda t: 10 if t < 79 else (60 if t < 160 else 25)
    seg = tv_lowpass_fast(hum_mono[:sec(205)], ramp(sec(205), [(0, 1400), (78, 1400), (83, 9000), (160, 9000), (168, 2500)]))
    hum_placed = binaural(seg, path_az, path_el, path_dist)
    hum_placed *= db(ramp(hum_placed.shape[1], [(0, -40), (4, -8), (78, -8), (84, 0), (160, 0), (170, -6), (204.95, -6),
                                                (205.0, -120)]))
    mix.cue("env", 0, hum_placed, "hum")

    mix.bed("env", bed.wind(14, 1, "slit"), "wind_gust", [(0, -60), (4, -2), (9, 0), (14, -60)], t=27)
    mix.bed("env", bed.wind(78, 2, "stair"), "wind_stair", [(0, -60), (8, 0), (32, 0), (33, -6), (45, -8), (70, -10), (78, -60)],
            t=158)
    mix.bed("env", bed.vent(40, 1, breathing=0.0), "vent", [(0, -60), (10, 0), (30, 0), (40, -60)], t=170)
    mix.bed("env", bed.vent(30, 2, breathing=1.0, rate=0.24), "vent_breathing", [(0, -40), (8, -8), (20, -2), (26, 0), (30, 0)],
            t=206)

    # ======================= I. ORDINARY ABANDONMENT =======================
    # The wall clock - the one regular thing. All the school's clocks run off one master clock.
    clk = M.clock(94.5, stutter_at=44.0)[sec(2):]
    clk_az = lambda t: -55.0 if t < 79 else -55 - (t - 79) * 25
    clk_dist = lambda t: 3.4 if t < 79 else 3.4 + (t - 79) * 1.2
    c_all = _sum(binaural(clk, lambda t: clk_az(t + 2), 25, lambda t: clk_dist(t + 2)),
                 conv(clk, ROOMS["classroom"].tail_for(3.4)) * 0.8)
    c_all = tv_lowpass_fast(c_all, ramp(c_all.shape[1], [(0, 20000), (77, 20000), (82, 1500)]))
    c_all *= db(ramp(c_all.shape[1], [(0, 0), (77, 0), (84, -10)]))
    mix.cue("foley", 2.0, c_all, "clock")

    steps(mix, 6.0, 6, "classroom", interval=0.6, seed="s1")

    t, k = 11.0, 0
    while t < 232:                              # drips into a bucket: irregular, forever
        if t < 80:
            mix.cue("foley", t, place(M.drip(k), "classroom", az=118, el=-30, dist=2.6), "drip", r.uniform(-3, 0))
        else:
            room = "corridor" if t < 160 else "stairwell"
            mix.cue("foley", t, place_remote(M.drip(k), "classroom", 2.6, 175, 7 + (t - 80) * 0.08, "door_ajar", room),
                    "drip_far", r.uniform(-3, 0))
        t += r.choice([1.1, 1.7, 2.3, 3.4, 4.6, 0.6]) * r.uniform(0.8, 1.3)
        k += 1

    mix.cue("foley", 15.5, place_remote(M.door_slam(1, 0.8), "stairwell", 6, 160, 14, "door", "classroom"), "slam_far")
    for tt in (18.2, 19.9, 20.6):           # the heating pipe ticks as it cools...
        mix.cue("foley", tt, place_remote(M.knock(int(tt * 10), "pipe", 0.25), "classroom", 2, -80, 2.2, "wall", "classroom"),
                "pipe_tick", r.uniform(-4, 0))
    # ...and knocks three times. Pipes do that.
    mix.cue("foley", 23.5, place_remote(M.knock_pattern(1, "pipe"), "classroom", 2, -78, 2.2, "wall", "classroom"),
            "pipe_knocks")
    steps(mix, 34.0, 6, "classroom", interval=0.62, seed="s2")

    # ======================= II. FIRST ANOMALY =======================
    # 0:44 the clock hesitates (inside clk). 0:49 the classroom PA keys up. Nobody speaks.
    click(mix, 49.0, CLASS_PA, "classroom", True, "k1")
    murmur = fade(bed.yard(8.0, 3).mean(axis=0), 1.2, 0.8)
    murmur_hall = place(murmur, "hall", az=0, dist=14).mean(axis=0)
    carrier = V.pa_speaker(murmur_hall * 0.5, seed="carrier", hum=0.02, distortion=1.4)
    mix.cue("voice", 49.4, place(fade(carrier, 0.4, 0.4), "classroom", 25, 60, 2.6, wet=0.8), "pa_carrier")
    click(mix, 57.6, CLASS_PA, "classroom", False, "k2", -3)

    # 1:02 upstairs in the music room, a child's five-finger exercise. C D E F. No G.
    mix.cue("score", 62.0, place_remote(S.exercise(1, tempo=0.66), "music_room", 3, 30, 6, "floor", "classroom", el=55),
            "piano_far")
    # 1:12 the pipe again. The same three knocks - from the other wall.
    mix.cue("foley", 71.5, place_remote(M.knock_pattern(2, "pipe"), "classroom", 2, 82, 2.4, "wall", "classroom"),
            "pipe_knocks")

    # 1:18 out into the corridor
    mix.cue("foley", 78.0, place(M.door_creak(2, 1.7), "classroom", az=-8, el=0, dist=1.0), "door_creak")
    steps(mix, 80.0, 3, "classroom", interval=0.6, seed="s3")

    # ======================= III. RECOGNITION =======================
    on = steps(mix, 84.0, 8, "corridor", interval=0.6, seed="s4")
    for i, ts in enumerate(on[2:8]):            # a child far down the corridor counts your steps
        mix.cue("voice", ts + 0.09, place(cast.count_word(i + 1), "corridor", az=-4, el=2, dist=25 - i * 0.6, wet=1.0),
                "count", i * 0.15)
    on2 = steps(mix, 93.0, 2, "corridor", interval=0.62, seed="s5")
    for i, ts in enumerate(on2):
        mix.cue("voice", ts + 0.09, place(cast.count_word(7 + i), "corridor", az=-4, el=2, dist=19 - i * 2, wet=1.0),
                "count", 1.0)
    # ...and the step you didn't take. Closer.
    mix.cue("voice", on2[-1] + 0.71, place(cast.count_word(9), "corridor", az=-3, el=2, dist=12.5, wet=0.9),
            "count_extra")

    # 1:36 the light dies for a moment (hum dropout) and every clock in the building stops (clk ends at 96.5).

    # 1:43 knocks on the classroom door beside you. Only two.
    mix.cue("foley", 103.0, place_remote(M.knock_pattern(3, "door", count=2), "classroom", 1.5, 72, 2.2, "door", "corridor"),
            "door_knocks")
    # 1:50 behind that door someone writes on the blackboard. It ends with a full stop.
    mix.cue("foley", 110.0, place_remote(M.chalk(2, 11.0), "classroom", 3.5, 70, 2.4, "door", "corridor"), "chalk")

    # 1:58 strings: a section rehearsing in another wing, leaning on G from both sides. It comes and goes.
    cl1 = S.cluster("c1", 70, [("Cello Section/susvib", 54, 0.8), ("Viola Section/trem", 56, 0.5),
                                ("Violin Section/susVib", 66, 0.4), ("Violin Section/susVib", 68, 0.35),
                                ("Solo Contrabass/SusNV", 42, 0.35)], drift_cents=45, swell=True)
    cl1 = hp(cl1, 110)
    cl1_far = place_remote(cl1.mean(axis=0), "hall", 10, -130, 9, "wall", "corridor")
    cl1_far *= db(ramp(cl1_far.shape[1], [(0, -40), (14, -6), (30, 0), (52, -2), (66, -20), (70, -60)]))
    mix.cue("score", 118.0, cl1_far, "strings_far")
    # 2:04 the exercise again, nearer. It gets stuck on F.
    mix.cue("score", 124.0, place_remote(S.exercise(2, notes=(60, 62, 64, 65, 65, 65), tempo=0.7, velocity=0.4),
                                         "music_room", 3, -115, 4.5, "wall", "corridor"), "piano_near")

    # ======================= IV. ESCALATION =======================
    # 2:12 the register. Mrs Hale has taken it ten thousand times.
    click(mix, 132.6, CORRIDOR_PA, "corridor", True, "k3")
    t = 133.5
    reg = [(cast.teacher("Good morning, everyone."), 2.6)]
    mix.cue("voice", t, pa_render(reg[0][0], CORRIDOR_PA, "corridor", "gm"), "register", -1)
    t += 2.6
    kids = {1: (-25, 9), 2: (30, 12), 3: (5, 16)}
    for i, (name, answer) in enumerate(cast.REGISTER[:3]):
        mix.cue("voice", t, pa_render(cast.teacher(name), CORRIDOR_PA, "corridor", "n%d" % i), "register")
        t += 1.9
        az, d = kids[i + 1]
        ans = place(cast.child_says(answer, i + 1), "hall", az=az, dist=d).mean(axis=0) * 1.6
        mix.cue("voice", t, pa_render(ans, CORRIDOR_PA, "corridor", "a%d" % i, source_room=None), "register", -4)
        t += 1.5
    mix.cue("voice", t, pa_render(cast.teacher("Sophie Doyle."), CORRIDOR_PA, "corridor", "sd"), "register")
    t += 4.2
    mix.cue("voice", t, pa_render(cast.teacher("Sophie Doyle?", 0.9), CORRIDOR_PA, "corridor", "sq"), "register")
    t += 3.4
    # The answer does not come over the PA. It comes from the corridor. Behind you.
    mix.cue("voice", t, place(cast.child_says("Here.", 2, 0.85), "corridor", az=172, el=0, dist=9.0, wet=1.0), "answer_behind")
    click(mix, t + 2.6, CORRIDOR_PA, "corridor", False, "k4", -3)

    # 2:38 you walk faster. Something walks with you, a little late - and stops a step after you do.
    t_walk = t + 4.5
    on3 = steps(mix, t_walk, 12, "corridor", interval=0.46, seed="s6", key="steps_fast")
    copy = np.zeros(sec(10))
    for i, ts in enumerate(on3 + [on3[-1] + 0.55]):
        add(copy, M.footstep(1000 + i, "lino", 0.9), sec(ts - on3[0] + 0.13))
    mix.cue("foley", on3[0], place(copy, "corridor", az=178, el=-20, dist=lambda tt: 9 - tt * 0.25, wet=1.0), "copy_steps")

    # 2:50 the stairwell
    t_st = on3[-1] + 2.2
    steps(mix, t_st, 3, "stairwell", "stone", interval=0.7, seed="s7")

    # 2:54 the bell - from the floor above, which has not been safe to stand on for years. It will not stop.
    t_bell = t_st + 3.5
    bell_sig = M.school_bell(3, ring=3.2, decay="wrong", tail=9.0)
    mix.cue("foley", t_bell, place(bell_sig, "stairwell", az=-20, el=68, dist=10.5, wet=1.2), "bell")
    # the music-room piano, its damper pedal stuck down, answers the bell
    bloom = S.sympathetic(bell_sig[:sec(4)], seed="b1")
    mix.cue("score", t_bell + 0.3, place_remote(bloom, "music_room", 2, -120, 6, "wall", "stairwell"), "bloom")
    bm = hp(S.bowed_metal("m1", 34, "brake", pitch=0.5), 140)
    bm_placed = place(bm, "stairwell", az=60, el=-40, dist=6, wet=1.5)
    bm_placed *= db(ramp(bm_placed.shape[1], [(0, -30), (6, -8), (14, 0), (16, -40), (17, -90)]))   # gone at the hush
    mix.cue("score", t_bell + 4.0, bm_placed, "bowed")

    # 3:03 the hall below fills with children before assembly ... and goes silent at once.
    t_crowd = t_bell + 9.0
    crowd = bed.yard(12.0, 5).mean(axis=0) * 0.7 + bed.gym(12.0, 2).mean(axis=0) * 0.3
    crowd = crowd * np.r_[np.linspace(0, 1, sec(7)) ** 2, np.ones(sec(5))]
    cut = sec(10.0)
    crowd[cut:] = 0.0
    crowd[cut - sec(0.03):cut] *= np.linspace(1, 0, sec(0.03))
    mix.cue("voice", t_crowd, place_remote(crowd, "hall", 12, 25, 7, "door_ajar", "stairwell", el=-35), "crowd")
    t_hush = t_crowd + 10.0

    # 3:12 after the silence, the hymn. Fourteen children ... and then fourteen children with one voice.
    t_hymn = t_hush + 4.0
    singers = []
    for k in range(14):
        d = r.uniform(6, 16)
        singers.append(np.pad(cast.hymn_voice(k), (sec(d / 343), 0)) / d)
    n = max(len(s) for s in singers)
    diverse = sum(np.pad(s, (0, n - len(s))) for s in singers)
    same = cast.hymn_voice(0, same_voice=True)
    unison = np.zeros(len(same) + sec(0.05))
    for k in range(14):
        sh = resample(same, 1 + (k - 7) * 0.0006)
        dl = int(k * 0.0011 * RATE)
        unison[dl:dl + len(sh)] += sh[:len(unison) - dl] / 9
    beat = 0.62
    m = max(len(diverse), len(unison))
    diverse, unison = np.pad(diverse, (0, m - len(diverse))), np.pad(unison, (0, m - len(unison)))
    w = np.clip((np.arange(m) - sec(17 * beat)) / (sec(19 * beat) - sec(17 * beat)), 0, 1)
    hymn = diverse * (1 - w) + unison * w * 1.3
    # the last word, 'place', is held far beyond any breath, sagging, coming apart
    last = sec(0.4 + 33 * beat)
    hold = M.spectral_freeze(hymn[last + sec(1.0):last + sec(1.0) + 8192], sec(16), rng("hold"))
    hold = varispeed(np.pad(hold, (0, sec(1))), 1.0 - 0.03 * np.linspace(0, 1, len(hold) + sec(1)) ** 1.3)[:sec(15)]
    hold *= np.interp(np.arange(len(hold)), [0, sec(1.5), sec(9), sec(15)], [0.0, 0.5, 0.8, 1.0])
    hymn_full = np.pad(hymn, (0, sec(16)))
    hymn_full[last + sec(1.5):last + sec(1.5) + len(hold)] += hold * np.max(np.abs(hymn)) * 0.7
    mix.cue("voice", t_hymn, place_remote(hymn_full, "hall", 14, 25, 7, "door_ajar", "stairwell", el=-35), "hymn")

    # ======================= V. PRESENCE =======================
    # 3:25 the light dies (hum) and does not come back. The strings are no longer in another wing.
    t_p = 205.0
    cl2 = S.cluster("c2", 31, [("Cello Section/susvib", 54, 0.6), ("Viola Section/trem", 56, 0.6),
                                ("Violin Section/susVib", 66, 0.5), ("Violin Section/susVib", 67, 0.45),
                                ("Violin Section/susVib", 68, 0.5), ("Solo Contrabass/SusNV", 43, 0.4)],
                    drift_cents=60, swell=False)
    cl2 = place(hp(cl2.mean(axis=0), 150), "stairwell", az=0, el=0, dist=7, wet=1.6)
    cl2 *= db(ramp(cl2.shape[1], [(0, -40), (8, -14), (14, -8), (20, -12), (26, -3), (31, 0)]))
    mix.cue("score", t_p, cl2, "strings_presence")

    # breath at your ear. Real breath. The ventilation takes up its rhythm.
    mix.cue("voice", 211.6, place_head(cast.inhale(0), az=158, el=5, dist=0.32), "breath")
    mix.cue("voice", 213.2, place_head(cast.exhale(0), az=150, el=0, dist=0.35), "breath", -7)
    mix.cue("voice", 218.4, place_head(cast.inhale(3), az=-100, el=10, dist=0.26), "breath")
    mix.cue("voice", 220.0, place_head(cast.exhale(1), az=-95, el=10, dist=0.3), "breath", -7)

    # the register, whispered, circling your head
    names = [("Thomas Avery.", lambda tt: 120 + tt * 50, 0.55, None), ("Hannah Bell.", lambda tt: -60 + tt * 45, 0.45, 2),
             ("Daniel Cole.", lambda tt: 200 - tt * 55, 0.5, None)]
    for i, (nm, az_f, d, who) in enumerate(names):
        mix.cue("voice", 221.5 + i * 1.6, place(cast.whispered(nm, who), "stairwell", az=az_f, el=15, dist=d, wet=0.15),
                "whisper", -i * 1.5)
    rev = V.whisper(cast.teacher_reversed("Sophie Doyle."))
    mix.cue("voice", 225.0, place(rev, "stairwell", az=lambda tt: 90 + tt * 40, el=20, dist=0.6, wet=0.2), "whisper", -4)

    # a chair dragged across the floor above you
    mix.cue("foley", 223.0, place_remote(M.chair_drag(3, 3.4), "classroom", 3, 10, 2.8, "floor", "stairwell", el=80), "chair")

    # the PA, right above you now, and she cannot get the words out
    click(mix, 226.8, NEAR_PA, "stairwell", True, "k5")
    mix.cue("voice", 227.4, pa_render(cast.teacher_freezing("Is there anyone I haven't called?", 0.32, 0.4, 1.7), NEAR_PA,
                                      "stairwell", "frz"), "pa_freeze")

    # the inhale: everything is drawn in ... and cut.
    t_cut = 236.0
    pre = sum(mix.stems[s] for s in STEMS)
    swell = S.reverse_swell(pre[:, sec(t_cut - 4.5):sec(t_cut)].mean(axis=0), 3.2, 3.0)
    swell = spread(swell * np.linspace(0, 1, len(swell)) ** 2, 0.8)
    for s in STEMS:
        g = np.ones(mix.stems[s].shape[1])
        g[sec(t_cut):] = 0.0
        g[sec(t_cut) - sec(0.012):sec(t_cut)] = np.linspace(1, 0, sec(0.012))
        mix.stems[s] *= g
    mix.cue("score", t_cut - 3.2, swell, "swell")

    # ======================= VI. ABSENCE =======================
    # the third knock. Behind you. Close enough to touch. No room around it at all.
    k3 = M.knock(77, "door", 0.8)
    mix.cue("foley", t_cut + 1.65, _sum(place_head(k3, az=180, el=4, dist=0.45),
                                        place(k3, "cupboard", az=180, el=4, dist=0.45, dry=0.0, wet=0.25)), "knock_behind")
    # an emptiness that is not silence: the faintest room, nothing in it
    mix.bed("env", lp(bed.room_tone(50, 9, "stairwell"), 900), "absence_tone", [(0, -60), (5, 0), (42, 0), (50, -60)],
            t=t_cut + 2.2)
    mix.cue("foley", t_cut + 13.0, place_remote(M.drip(500), "classroom", 3, 170, 12, "door_ajar", "stairwell"), "drip_far")

    # the PA. Close, clean, warm.
    t_end = t_cut + 20.0
    click(mix, t_end, NEAR_PA, "stairwell", True, "k6", -4)
    mix.cue("voice", t_end + 0.8, pa_render(cast.teacher("Thank you."), NEAR_PA, "stairwell", "ty", source_room="cupboard"),
            "pa_last")
    mix.cue("voice", t_end + 2.3, pa_render(cast.teacher("Everyone's here now."), NEAR_PA, "stairwell", "eh",
                                            source_room="cupboard"), "pa_last")
    click(mix, t_end + 5.0, NEAR_PA, "stairwell", False, "k7", -6)
    # and every clock in the school starts again.
    mix.cue("foley", t_end + 8.0, place(M.clock(16.5), "stairwell", az=-140, el=35, dist=3.2, wet=0.9), "clock_end")
    return mix


def master(mix, target_lufs=-23.0):
    """Gentle master: subsonic clean-up, one gain to the loudness target, true-peak limiter at -1 dBTP.
    No compression - the dynamic arc is the composition."""
    total = hp(sum(mix.stems.values())[:, :mix.n], 22, 2)
    g = db(target_lufs - lufs(total))
    out = limiter(total * g, -1.0)
    stems = {k: hp(v[:, :mix.n], 22, 2) * g for k, v in mix.stems.items()}
    return out, stems


def render(out_dir=None):
    out_dir = out_dir or ROOT / "showcase"
    mix = compose()
    out, stems = master(mix)
    write(out_dir / "the_register_master.flac", out)
    write(out_dir / "the_register_master.mp3", out)
    for k, v in stems.items():
        write(out_dir / "stems" / ("the_register_stem_%s.flac" % k), v)
    report = {"integrated_lufs": round(float(lufs(out)), 2), "true_peak_dbtp": round(float(true_peak_db(out)), 2),
              "short_term_max_lufs": round(float(short_term_max(out)), 2), "seconds": round(out.shape[1] / RATE, 2)}
    print(report)
    return report
