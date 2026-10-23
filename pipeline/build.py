"""Build everything.

    python3 pipeline/fetch_sources.py           # once: download the source material (~2.5 GB, cached)
    python3 pipeline/build.py showcase          # the composed binaural demo + stems  -> showcase/
    python3 pipeline/build.py library           # the interactive asset library       -> godot/audio/
    python3 pipeline/build.py qc                # analysis report                     -> docs/QC_REPORT.md
    python3 pipeline/build.py all

SCHOOL_AUDIO_SEED=<int> renders a different take of every procedural choice; the default (1983) is the
reference take that the QC report describes.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main(argv):
    what = argv[1] if len(argv) > 1 else "all"
    t = time.time()
    if what in ("showcase", "all"):
        from school import showcase
        showcase.render()
    if what in ("library", "all"):
        from school import library
        library.render_all()
    if what in ("qc", "all"):
        from school import qc
        qc.run()
    print("done in %.0f s" % (time.time() - t))


if __name__ == "__main__":
    main(sys.argv)
