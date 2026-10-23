"""Fetch every external source the pipeline uses, and record what it is and what we may do with it.

    python3 pipeline/fetch_sources.py            # download anything missing, verify, write docs/PROVENANCE.md

Sources land in $SCHOOL_AUDIO_CACHE (default: <repo>/.cache/sources), which is never committed.
Only sources whose licence allows redistribution of derived works are rendered into deliverables
(`use: "render"`). Sources marked `use: "reference"` are analysed (e.g. to calibrate reverb decay)
but never mixed into any output file.
"""
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = Path(os.environ.get("SCHOOL_AUDIO_CACHE", ROOT / ".cache" / "sources"))
LOCK = ROOT / "pipeline" / "sources.lock.json"
PROVENANCE = ROOT / "docs" / "PROVENANCE.md"

PD = "Public Domain Mark 1.0"
CC0 = "CC0 1.0"
CCBY3 = "CC BY 3.0"


def aporee(item, filename, title, author, licence=PD, note=""):
    return {"id": item, "kind": "url", "path": "field/" + filename,
            "url": "https://archive.org/download/%s/%s" % (item, filename),
            "page": "https://archive.org/details/" + item, "title": title, "author": author,
            "licence": licence, "use": "render", "note": note}


SOURCES = [
    # Real field recordings from radio aporee ::: maps, mirrored on archive.org. Only PD-mark and CC BY items;
    # every NC / ND / SA item found during research was rejected.
    aporee("aporee_66656_77072", "250214005.WAV", "Dromina National School, Co. Cork - yard time",
           "fionalinnanecomposer", note="Real children at break. The school's memory."),
    aporee("aporee_58722_67377", "CappamoreFieldrecording.wav", "Scoil Chaitriona, Cappamore - lunch break at school",
           "fionalinnanecomposer"),
    aporee("aporee_66241_76587", "RegenamFenster090120251515.mp3", "Mannheim - rain against a window",
           "matthias.feuersenger"),
    aporee("aporee_30986_35627", "1602140001.mp3", "Rissen, Hamburg - rain on the window", "soundski"),
    aporee("aporee_38223_43714", "marchienne071012h15.mp3", "Charleroi - rain in an abandoned factory",
           "Vincent Duseigne", licence=CCBY3),
    aporee("aporee_42774_48769", "insidetheabandonedbuilding.wav", "Utena, Lithuania - inside an abandoned building",
           "alas23/sala"),
    aporee("aporee_40781_46556", "angarasgrotos.wav", "Utena, Lithuania - abandoned hangar ventilation",
           "alas23/sala"),
    aporee("aporee_44314_50392", "matewind.mp3", "Poznan - wind through a sliding-door slit", "maciej janasik"),
    aporee("aporee_15256_17789", "rad27.mp3", "Poznan - radiator language when set to 4", "maciej janasik"),
    aporee("aporee_13339_15579", "rad.mp3", "Poznan - apartment radiator", "maciej janasik"),
    aporee("aporee_16357_18963", "DusseldorfGGSTurnhalle.mp3", "Dusseldorf, GGS Lorick - school gym hall",
           "Frank Schulte"),
    aporee("aporee_33858_38949", "electricalhumfoxesinrainedit.mp3",
           "Weisswasser - transformer house, electrical hum in the rain", "Peter Cusack"),
    aporee("aporee_51842_59208", "ElektrosPastotesGausmasAkademija.wav", "Akademija, Lithuania - electrical substation hum",
           "Marius Paulikas"),
    aporee("aporee_32358_37204", "hotelroomtapdripeditJan12012.mp3", "Montreal - late night tap drip",
           "Matt Rogalsky", licence=CCBY3),
    aporee("aporee_43137_49168", "060319puertopercyloosemetalroof.mp3", "Puerto Percy - storm, loose metal roof",
           "carsten stabenow"),
    aporee("aporee_44286_50364", "LiepajaUniversitystairwell.mp3", "Liepaja University stairwell", "john grzinich", licence="CC BY-NC-ND 3.0"),
    aporee("aporee_44285_50363", "Carraigefactorytowerwindy.mp3", "Liepaja - factory stairwell tower in the wind", "john grzinich", licence="CC BY-NC-ND 3.0"),
    aporee("aporee_9932_11869", "windstormstairwell.mp3", "Torun - stairwell in windstorm", "john grzinich", licence="CC BY-SA 3.0"),
    aporee("aporee_23415_27189", "14040302CouloirsERG.mp3", "Erg, Ixelles - corridor ambience", "Flavien Gillie", licence="CC BY-NC-ND 3.0"),
    aporee("aporee_26803_30911", "20131210intrieurgrandhall2photographes.mp3", "Haren - abandoned Wanson factory hall", "Flavien Gillie", licence="CC BY-NC-ND 3.0"),
    aporee("aporee_3458_4741", "CityCountryMejaegerstrlueftungsschacht.mp3", "Berlin - ventilation shaft", "henrik schroder", licence="CC BY-SA 3.0"),
    aporee("aporee_10191_12143", "UmeaReklamABHemvagen.mp3", "Umea - ventilation duct", "martin hogg", licence="CC BY-SA 3.0"),
    aporee("aporee_49915_56925", "Childrenplayingintheprimaryschoolyard.wav", "Caen - children in the primary school yard", "Guillaume Capsowl Voisin", licence="CC BY-NC 3.0"),
    {"id": "FluorescentLightsAreCool", "kind": "url", "path": "field/Fluorescent.mp3",
     "url": "https://archive.org/download/FluorescentLightsAreCool/Fluorescent.mp3",
     "page": "https://archive.org/details/FluorescentLightsAreCool", "title": "Fluorescent lights are cool",
     "author": "ID Yep", "licence": CC0, "use": "render", "note": ""},

    # CC0 instrument and found-object recordings (Versilian Studios). Pinned commits, sparse checkouts.
    {"id": "VSCO-2-CE", "kind": "git", "path": "VSCO-2-CE", "url": "https://github.com/sgossner/VSCO-2-CE",
     "commit": "440300901dfe9275fd84e0b7763af1f8443ae62e",
     "sparse": ["/Keys/Upright Piano/", "/Keys/Upright Nr1/", "/Keys/Organ/Rode_Blower.wav",
                "/Strings/Cello Section/trem/", "/Strings/Cello Section/susvib/", "/Strings/Solo Contrabass/SusNV/",
                "/Strings/Solo Contrabass/Trem/", "/Strings/Viola Section/trem/", "/Strings/Violin Section/Trem/",
                "/Strings/Violin Section/susVib/", "/Strings/Solo Violin/Trem/", "/Strings/Solo Violin/Arco Vib/",
                "/Percussion/susCymb1-*", "/Percussion/gongscrape*", "/Percussion/bassdrum_rub*",
                "/Percussion/vibraring*", "/Percussion/gongHit*", "/Miscellania Raw/", "/LICENSE", "*.txt"],
     "page": "https://github.com/sgossner/VSCO-2-CE", "title": "VS Chamber Orchestra: Community Edition",
     "author": "Versilian Studios / Simon Dalzell (Ivy Audio)", "licence": CC0, "use": "render",
     "note": "Upright pianos, string sections, bowed cymbal, gong scrape, drum-head friction, brick scrape, "
             "chain grind, metal hits, organ blower."},
    {"id": "VCSL", "kind": "git", "path": "VCSL", "url": "https://github.com/sgossner/VCSL",
     "commit": "c1ea7bcc3c7309650ab0da9d15c9cd1fbc4a4c7e",
     "sparse": ["/Chordophones/Zithers/Upright Piano, Yamaha/", "/Idiophones/Struck Idiophones/Brake Drum/",
                "/Idiophones/Struck Idiophones/Gong 1/", "/Idiophones/Struck Idiophones/Vibraphone/*bowed*",
                "/Idiophones/Friction Idiophones/Wine Glasses/",
                "/Chordophones/Zithers/Psaltery, Bowed and Plucked/*LongBow*",
                "/Idiophones/Struck Idiophones/Hand Bells, Nepalese/", "/Idiophones/Struck Idiophones/Tubular Bells 1/",
                "/LICENSE", "/README.md"],
     "page": "https://github.com/sgossner/VCSL", "title": "Versilian Community Sample Library",
     "author": "Versilian Studios", "licence": CC0, "use": "render",
     "note": "Upright piano with release samples, bowed brake drum, bowed vibraphone, wine glasses, bowed psaltery."},

    # Speech.
    {"id": "kokoro-v1.0", "kind": "url", "path": "kokoro/kokoro-v1.0.onnx",
     "url": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
     "page": "https://huggingface.co/hexgrad/Kokoro-82M", "title": "Kokoro-82M TTS (ONNX export)",
     "author": "hexgrad; ONNX export by thewh1teagle", "licence": "Apache 2.0", "use": "render",
     "note": "Neural TTS used as raw material for every authored line; all output is resynthesised (WORLD) "
             "before use."},
    {"id": "kokoro-voices-v1.0", "kind": "url", "path": "kokoro/voices-v1.0.bin",
     "url": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
     "page": "https://huggingface.co/hexgrad/Kokoro-82M", "title": "Kokoro-82M voice embeddings",
     "author": "hexgrad", "licence": "Apache 2.0", "use": "render", "note": ""},
    {"id": "librispeech-dev-clean", "kind": "url", "path": "speech/dev-clean.tar.gz",
     "url": "https://www.openslr.org/resources/12/dev-clean.tar.gz", "unpack": "speech",
     "page": "https://www.openslr.org/12/", "title": "LibriSpeech dev-clean (read English speech, 40 speakers)",
     "author": "Panayotov et al.; LibriVox volunteers", "licence": "CC BY 4.0", "use": "render",
     "note": "Real human voices for the far-away crowd/assembly babble; never intelligible in the output."},

    # Acoustics.
    {"id": "mit-kemar", "kind": "url", "path": "hrtf/mit_kemar_normal_pinna.sofa",
     "url": "https://sofacoustics.org/data/database/mit/mit_kemar_normal_pinna.sofa",
     "page": "https://sofacoustics.org/data/database/mit/", "title": "MIT KEMAR HRTF, normal pinna (SOFA)",
     "author": "Gardner & Martin, MIT Media Lab (1995)", "licence": "No usage restrictions; citation requested",
     "use": "render", "note": "Binaural rendering of every positioned source. Gardner, W. G., and Martin, K. D. "
                              "(1995). HRTF measurements of a KEMAR. JASA 97, 3907-3908."},
    {"id": "aachen-air-1.4", "kind": "url", "path": "rir/air_1_4.zip", "unpack": "rir",
     "url": "https://www.openslr.org/resources/20/air_database_release_1_4.zip",
     "page": "https://www.openslr.org/20/", "title": "Aachen Impulse Response database 1.4",
     "author": "RWTH Aachen University", "licence": "Not stated (copyright RWTH Aachen)", "use": "render",
     "note": "Real binaural (dummy-head) room responses: stairway, lecture room, Aula Carolina hall."},
]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "school-audio-pipeline/1.0"})
    tmp = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(request, timeout=120) as response, open(tmp, "wb") as out:
        while block := response.read(1 << 20):
            out.write(block)
    tmp.rename(target)


def fetch(source):
    target = CACHE / source["path"]
    if source["kind"] == "git":
        if not (target / ".git").exists():
            subprocess.run(["git", "clone", "-q", "--filter=blob:none", "--no-checkout", source["url"], str(target)],
                           check=True)
        subprocess.run(["git", "-C", str(target), "sparse-checkout", "init", "--no-cone"], check=True)
        subprocess.run(["git", "-C", str(target), "sparse-checkout", "set", *source["sparse"]], check=True)
        subprocess.run(["git", "-C", str(target), "checkout", "-q", source["commit"]], check=True)
        return source["commit"]
    if not target.exists():
        print("downloading", source["url"])
        for attempt in range(4):
            try:
                download(source["url"], target)
                break
            except Exception as error:  # archive.org intermittently 500s
                print("  retry", attempt + 1, error)
                time.sleep(2 ** (attempt + 1))
        else:
            print("  FAILED, skipping", source["id"])
            return "missing"
    if source.get("unpack") and target.suffix == ".zip":
        with zipfile.ZipFile(target) as archive:
            archive.extractall(CACHE / source["unpack"])
    elif source.get("unpack") and target.name.endswith(".tar.gz"):
        if not (CACHE / source["unpack"] / "LibriSpeech").exists():
            subprocess.run(["tar", "-xzf", str(target), "-C", str(CACHE / source["unpack"])], check=True)
    return sha256(target)


def write_provenance(lock):
    lines = ["# Source provenance", "",
             "Generated by `pipeline/fetch_sources.py`. Every external input to the pipeline is listed here with its",
             "licence and checksum. `render` sources may appear (transformed) in deliverables; `reference` sources are",
             "analysed only and never mixed into any output.", "",
             "Everything not listed here is synthesised from scratch by the pipeline (seeded, reproducible).", "",
             "| Source | Author | Licence | Use | Checksum / commit |", "|---|---|---|---|---|"]
    for source in SOURCES:
        lines.append("| [%s](%s) | %s | %s | %s | `%s` |" % (source["title"], source["page"], source["author"],
                                                           source["licence"], source["use"], lock[source["id"]][:16]))
    lines += ["", "## Notes", ""]
    for source in SOURCES:
        if source["note"]:
            lines.append("- **%s** - %s" % (source["title"], source["note"]))
    lines += ["", "## Required attributions", "",
              "- Rain in an abandoned factory by Vincent Duseigne (radio aporee), CC BY 3.0.",
              "- Late night tap drip by Matt Rogalsky (radio aporee), CC BY 3.0.",
              "- LibriSpeech (Panayotov, Chen, Povey, Khudanpur, ICASSP 2015), CC BY 4.0.",
              "- HRTF: Gardner, W. G., and Martin, K. D. (1995). HRTF measurements of a KEMAR. JASA 97, 3907-3908.",
              "- Kokoro-82M by hexgrad, Apache 2.0.", ""]
    PROVENANCE.write_text("\n".join(lines))


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    lock = json.loads(LOCK.read_text()) if LOCK.exists() else {}
    for source in SOURCES:
        digest = fetch(source)
        if source["id"] in lock and lock[source["id"]] != digest:
            sys.exit("checksum changed for %s: %s != %s" % (source["id"], digest, lock[source["id"]]))
        lock[source["id"]] = digest
        print("ok", source["id"], digest[:16])
    LOCK.write_text(json.dumps(lock, indent=2) + "\n")
    write_provenance(lock)


if __name__ == "__main__":
    main()
