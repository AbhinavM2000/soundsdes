"""Render any song as if it were playing in a locked room somewhere else in the school - binaural, headphones.

    python3 pipeline/distant_room.py song.mp3 --start 1:05 --seconds 30 --out showcase/distant_song.flac
    python3 pipeline/distant_room.py --standin --seconds 30          # original stand-in (no licensed audio)

The chain: an old cassette player's speaker (narrow band, wow and flutter, hiss, a cracked cone) -> the room the
player sits in (its own reverb) -> through a closed door and the floor -> into the corridor where you stand, from
a direction that drifts a little as you listen -> the monsoon and the night outside. At the end the tape is
stopped by a hand you never hear arrive.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from school import bed, carnatic, voice as V                                   # noqa: E402
from school.core import ROOT, RATE, bp, fade, limiter, load, lufs, db, peak_eq, sec, varispeed, write  # noqa: E402
from school.space import place, place_remote                                       # noqa: E402


def stamp(text):
    parts = [float(p) for p in text.split(":")]
    return sum(p * 60 ** i for i, p in enumerate(reversed(parts)))


def cassette(x, seed="cassette"):
    y = V.tape(x, seed=seed, wow=0.006, flutter=0.002, hiss=0.012, dropouts=0.12, age=1.6)
    y = bp(y, 170, 5200, 2)
    y = peak_eq(y, 900, 5.0, 1.5)          # small speaker's honk
    y = peak_eq(y, 2600, -4.0, 1.0)
    return np.tanh(y / (np.max(np.abs(y)) + 1e-9) * 1.7) * 0.5


def render(x, seconds, seed="room", stop=True, ambience=True):
    x = x[:sec(seconds)]
    if stop:  # the last 0.7 s: someone presses STOP, the tape winds down
        n = sec(0.7)
        speed = np.r_[np.ones(len(x) - n), np.linspace(1, 0.05, n) ** 1.6]
        x = varispeed(np.pad(x, (0, sec(1))), speed)[:len(x)]
        x[-sec(0.05):] *= np.linspace(1, 0, sec(0.05))
    x = cassette(fade(x, 0.02, 0.0), seed)
    y = place_remote(x, "music_room", 2.2, lambda t: -35 + 8 * np.sin(t / 7.0), 8.5, "door", "corridor", el=20,
                     listener_wet=0.45)
    y = y * db(np.interp(np.arange(y.shape[1]), [0, sec(4), y.shape[1]], [-30, 0, 0]))[None, :]
    y = y * db(-30 - lufs(y))
    if ambience:
        amb = bed.rain(y.shape[1] / RATE + 0.1, 7, "corridor")[:, :y.shape[1]]
        amb = amb * db(-38 - lufs(amb))
        night = bed.room_tone(y.shape[1] / RATE + 0.1, 4, "corridor")[:, :y.shape[1]]
        y = y + amb + night * db(-46 - lufs(night))
    y = y * db(-24 - lufs(y))
    return limiter(y, -1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("song", nargs="?")
    ap.add_argument("--start", default="0:00")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--standin", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args()
    if args.standin or not args.song:
        x = carnatic.alapana(args.seconds + 1)
        out = Path(args.out or ROOT / "showcase" / "distant_room_standin.flac")
    else:
        x = load(args.song, start=stamp(args.start), length=args.seconds + 1)
        out = Path(args.out or ROOT / "showcase" / ("distant_room_" + Path(args.song).stem + ".flac"))
    y = render(x, args.seconds)
    write(out, y)
    write(out.with_suffix(".mp3"), y)
    print(out, round(y.shape[1] / RATE, 1), "s", round(lufs(y), 1), "LUFS")


if __name__ == "__main__":
    main()
