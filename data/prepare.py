"""
data/prepare.py  -  Step 1: parse NLTK speeches into talk-level (word, label) data.

Labels: O (nothing follows), COMMA, PERIOD, QUESTION
Split : by TALK, not by sentence – seed 42, 70/15/15 train/val/test
Output: data/talks.pkl, data/stats.json, data/SOURCES.md

NOTE: These are real speech transcripts (NLTK state_union + inaugural corpora).
They are NOT real Whisper ASR output. The "ASR proxy" is lowercased, punct-stripped
text from these speeches. Real Whisper output requires audio – see BLOCKED.md.
"""
import pathlib, pickle, json, re, time, random, sys, hashlib
from collections import Counter

OUT = pathlib.Path(__file__).parent
OUT.mkdir(exist_ok=True)

import nltk
for c in ["state_union", "inaugural"]:
    try:
        nltk.data.find(f"corpora/{c}")
    except LookupError:
        nltk.download(c, quiet=True)

from nltk.corpus import state_union, inaugural

SEED = 42
PUNCT_MAP = {",": "COMMA", ".": "PERIOD", "?": "QUESTION",
             "!": "PERIOD", ";": "PERIOD"}


def parse_talk(text):
    """Convert raw speech text to [(word_lower, label)] pairs.

    Label goes on the word BEFORE the punctuation mark.
    """
    tokens = re.findall(r"[A-Za-z']+|[.,?!;]", text)
    pairs = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if re.match(r"[A-Za-z']", tok):
            label = "O"
            if i + 1 < len(tokens) and tokens[i + 1] in PUNCT_MAP:
                label = PUNCT_MAP[tokens[i + 1]]
                i += 1
            pairs.append((tok.lower(), label))
        i += 1
    return pairs


talks = []
for fid in state_union.fileids():
    pairs = parse_talk(state_union.raw(fid))
    if len(pairs) >= 50:
        talks.append({"id": "su_" + fid.replace("/", "_"),
                       "source": "state_union", "pairs": pairs,
                       "word_count": len(pairs)})

for fid in inaugural.fileids():
    pairs = parse_talk(inaugural.raw(fid))
    if len(pairs) >= 50:
        talks.append({"id": "in_" + fid.replace("/", "_"),
                       "source": "inaugural", "pairs": pairs,
                       "word_count": len(pairs)})

print(f"Loaded {len(talks)} talks, "
      f"{sum(t['word_count'] for t in talks):,} words total")

# split by talk – never by sentence
rng = random.Random(SEED)
rng.shuffle(talks)
n = len(talks)
n_train = int(0.70 * n)
n_val   = int(0.15 * n)

for i, t in enumerate(talks):
    if i < n_train:
        t["split"] = "train"
    elif i < n_train + n_val:
        t["split"] = "val"
    else:
        t["split"] = "test"

label_dist = Counter(lbl for t in talks for _, lbl in t["pairs"])
print("Label distribution:", dict(label_dist))
for sp in ("train", "val", "test"):
    ct = sum(1 for t in talks if t["split"] == sp)
    wc = sum(t["word_count"] for t in talks if t["split"] == sp)
    print(f"  {sp:5s}: {ct} talks, {wc:,} words")

pkl_path = OUT / "talks.pkl"
pickle.dump(talks, pkl_path.open("wb"))
print("Saved", pkl_path)

data_hash = hashlib.md5(pickle.dumps(talks)).hexdigest()[:12]
stats = {
    "timestamp":       time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "seed":            SEED,
    "python":          sys.version.split()[0],
    "nltk":            nltk.__version__,
    "talks_total":     len(talks),
    "talks_train":     sum(1 for t in talks if t["split"] == "train"),
    "talks_val":       sum(1 for t in talks if t["split"] == "val"),
    "talks_test":      sum(1 for t in talks if t["split"] == "test"),
    "words_total":     sum(t["word_count"] for t in talks),
    "label_dist":      dict(label_dist),
    "data_hash_md5":   data_hash,
    "note": ("Text from NLTK state_union + inaugural (public domain). "
             "NOT real Whisper ASR output. See BLOCKED.md for audio/ASR status."),
}
(OUT / "stats.json").write_text(json.dumps(stats, indent=2))
print("Saved", OUT / "stats.json")

(OUT / "SOURCES.md").write_text(
    "# Data Sources\n\n"
    "## Text corpora (punctuation model training)\n"
    "- NLTK `state_union` corpus – US State of the Union addresses (public domain)\n"
    "- NLTK `inaugural` corpus  – US Inaugural Addresses (public domain)\n\n"
    "These are real speech transcripts used as text only.\n"
    "They are NOT audio; they are NOT real Whisper output.\n\n"
    "## Audio files (required for real ASR — currently BLOCKED)\n"
    "Place MP4/WAV files in `videos/`.  See `BLOCKED.md`.\n"
)
print("Saved", OUT / "SOURCES.md")
