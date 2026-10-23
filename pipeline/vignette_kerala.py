"""HAAJAR (ഹാജർ) - a 90-second Kerala vignette. Headphones.

Monsoon night, classroom. The fan still turns. Sir (the project owner's own voice) takes attendance over the PA.
Ammu does not answer - and then she does, from behind you. Something follows you with anklet bells on. The power
dies. Sir thanks you: now everyone is here. The iron bell rings for class.

    python3 pipeline/vignette_kerala.py   -> showcase/haajar_vignette.{flac,mp3}
"""
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from school import bed, cast, cast_ml as C, kerala as K, matter as M, voice as V   # noqa: E402
from school.core import (ROOT, RATE, add, db, hp, limiter, lp, lufs, ramp, sec, true_peak_db, tv_lowpass_fast,
                         varispeed, write)                                            # noqa: E402
from school.showcase import cue_loudness, pa_render, CLASS_PA                     # noqa: E402
from school.space import place, place_head                                         # noqa: E402

TOTAL = 96.0
OUT = np.zeros((2, sec(TOTAL + 8)))


def put(t, x, target, integrated=False):
    """Place x at time t with its loudest 1-s window (or integrated loudness) at `target` LUFS."""
    x = x if x.ndim == 2 else np.vstack([x, x])
    level = lufs(x) if integrated else cue_loudness(x)
    if level > -90:
        add(OUT, x * db(target - level), sec(t))


def cue(fn):
    try:
        fn()
    except Exception:
        print("cue failed:", fn.__name__)
        traceback.print_exc()


def beds():
    rain = K.monsoon(TOTAL, 1, "classroom")
    rain = tv_lowpass_fast(rain, ramp(rain.shape[1], [(0, 16000), (58, 16000), (64, 600), (TOTAL, 450)]))
    put(0, rain * db(ramp(rain.shape[1], [(0, -40), (3, 0), (58, 0), (64, -9), (TOTAL, -10)])), -36, True)
    night = lp(K.night(TOTAL, 1), 3200)
    put(0, night * db(ramp(night.shape[1], [(0, -40), (4, 0), (57, 0), (60, -60), (TOTAL, -60)])), -46, True)
    tone = bed.room_tone(TOTAL, 1, "classroom")
    put(0, tone * db(ramp(tone.shape[1], [(0, -40), (3, 0), (60, 0), (64, -10), (TOTAL, -10)])), -50, True)


def fan():
    f = K.ceiling_fan(70, "f1", dying=0.5)
    speed = np.r_[np.ones(sec(58)), np.linspace(1, 0, sec(8)) ** 1.7, np.zeros(sec(5))]   # the power dies
    f = varispeed(np.pad(f, (0, sec(4))), np.maximum(speed, 0.0005))[:sec(70)]
    f[sec(66):] *= 0
    put(0, place(f, "classroom", az=10, el=80, dist=1.8), -43)


def clock():
    put(1, place(M.clock(57.0, stutter_at=30.0), "classroom", az=-55, el=25, dist=3.4), -45)


def register():
    put(9.0, pa_render(V.pa_click("k1", True), CLASS_PA, "classroom", "c1", None), -39)
    t = 9.8
    for line, gap in ((C.sir(C.GREETING), 2.6), (C.sir(C.ATTENDANCE), 2.8)):
        put(t, pa_render(line, CLASS_PA, "classroom", "s%d" % int(t)), -31)
        t += gap
    for i, (name, ans) in enumerate(C.REGISTER[:3]):
        put(t, pa_render(C.sir(name), CLASS_PA, "classroom", "n%d" % i), -31)
        t += 2.0
        kid = place(C.kid(ans, i + 1), "hall", az=(-25, 30, 5)[i], dist=(9, 12, 16)[i]).mean(axis=0)
        put(t, pa_render(kid * 1.6, CLASS_PA, "classroom", "a%d" % i, None), -35)
        t += 1.7
    put(t, pa_render(C.sir("അമ്മു."), CLASS_PA, "classroom", "am1"), -31)
    t += 4.5
    put(t, pa_render(C.sir(C.MISSING), CLASS_PA, "classroom", "am2"), -31)
    t += 3.6
    # not on the PA. In the room. Behind you.
    put(t, place(C.kid(C.ANSWER, 4), "classroom", az=172, el=0, dist=4.5, wet=1.0), -33)
    put(t + 2.8, pa_render(V.pa_click("k2", False), CLASS_PA, "classroom", "c2", None), -41)


def walk():
    seq = np.zeros(sec(9))
    onsets = [t for t, s in M.walk(9, "lino", 0.56, seed="me")]
    for t, s in M.walk(9, "lino", 0.56, seed="me"):
        add(seq, s, sec(t))
    put(42.0, place(seq, "classroom", az=0, el=-62, dist=1.45, wet=0.9), -33)
    copy = np.zeros(sec(10))
    for i, t in enumerate(onsets + [onsets[-1] + 0.6]):        # it takes one more step than you
        step = M.footstep(500 + i, "lino", 0.85)
        bells = K.chilanka(600 + i, 0.8)
        add(copy, step, sec(t + 0.14))
        add(copy, bells, sec(t + 0.15))
    put(42.0, place(copy, "classroom", az=176, el=-20, dist=lambda tt: 7.5 - tt * 0.35, wet=1.0), -38)


def presence():
    put(66.0, place_head(cast.inhale(0), az=158, el=5, dist=0.32), -31)
    put(67.6, place_head(cast.exhale(0), az=150, el=0, dist=0.35), -38)
    k = M.knock(77, "door", 0.8)                              # the third knock, close enough to touch
    put(71.0, place_head(k, az=180, el=4, dist=0.45), -28)


def ending():
    put(79.0, pa_render(V.pa_click("k3", True), [(0, 75, 1.6)], "classroom", "c3", None), -42)
    last = C.sir(C.PA["thank_you"] + " " + C.PA["everyone_here"])     # one line: 'nandi' alone is too short
    put(79.8, pa_render(last, [(0, 75, 1.6)], "classroom", "eh", "cupboard"), -32)
    put(85.0, pa_render(V.pa_click("k4", False), [(0, 75, 1.6)], "classroom", "c4", None), -44)
    put(88.0, place(K.iron_bell(3, strokes=3, interval=1.3), "corridor", az=40, el=10, dist=15), -30)


def main():
    for fn in (beds, fan, clock, register, walk, presence, ending):
        print("cue", fn.__name__, flush=True)
        cue(fn)
    mix = hp(OUT[:, :sec(TOTAL)], 22, 2)
    mix = limiter(mix * db(-23 - lufs(mix)), -1.0)
    out = ROOT / "showcase" / "haajar_vignette.flac"
    write(out, mix)
    write(out.with_suffix(".mp3"), mix)
    print("LUFS", round(lufs(mix), 1), "dBTP", round(true_peak_db(mix), 1))


if __name__ == "__main__":
    main()
