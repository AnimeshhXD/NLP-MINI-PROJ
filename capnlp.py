"""
Core corpus and punctuation-model helpers for the closed-caption pipeline.
"""
import re, random, unicodedata
from nltk.corpus import state_union, inaugural

LABELS = ["O", "COMMA", "PERIOD", "QUESTION"]
FILLERS = {"um", "uh", "er", "hmm"}
FUNCTION_WORDS = set("the a an of to and but in on with for that which is are was be by as at or from".split())
CONJ = {"and", "but", "which", "so", "because", "while", "although", "however", "where", "when"}


def load_speeches():
    """Return {speech_id: raw_text} for State of the Union + Inaugural corpora."""
    out = {}
    for f in state_union.fileids():
        out["su_" + f[:-4]] = state_union.raw(f)
    for f in inaugural.fileids():
        out["in_" + f[:-4]] = inaugural.raw(f)
    return out


def clean_raw(text):
    """Conservative cleaning of one raw speech."""
    text = unicodedata.normalize("NFC", text)
    text = (text.replace("\u201c", '"').replace("\u201d", '"')
                .replace("\u2018", "'").replace("\u2019", "'"))
    lines = text.split("\n")
    lines = [line for line in lines if not (sum(c.isupper() for c in line) > 0.8 * max(1, sum(c.isalpha() for c in line))
                                           and len(line.split()) > 3)]
    text = " ".join(lines)
    text = re.sub(r"\[[^\]]*\]|\([^)]*\)", " ", text)
    text = re.sub(r"--+|\u2014", " , ", text)
    text = re.sub(r"\b(?:[A-Za-z]\.){2,}", lambda m: m.group(0).replace(".", ""), text)
    text = re.sub(r"\b(Mr|Mrs|Ms|Dr|St|Jr|Sr|Hon|Gen|Gov|Col|Lt|Sen|Rep)\.", r"\1", text)
    text = re.sub(r"\b([A-HJ-Z])\.(?=\s+[A-Z])", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['\-][A-Za-z0-9]+)*(?:[.,][0-9]+)*|[.,?!;:]")


def to_tokens(text):
    """Return list of (word, punct_label, case_label) from a cleaned reference text."""
    toks = TOKEN_RE.findall(text)
    out = []
    for token in toks:
        if token in ".,?!;:":
            if not out:
                continue
            word, _, case = out[-1]
            label = "QUESTION" if token == "?" else "PERIOD" if token in ".!" else "COMMA"
            out[-1] = (word, label, case)
        else:
            if token.isupper() and len(token) > 1 and token.isalpha():
                case = "UPPER"
            elif token[0].isupper():
                case = "CAP"
            else:
                case = "LOWER"
            out.append((token, "O", case))
    return out


def asr_style(words):
    """Return lower-case ASR-style words with apostrophes removed."""
    return [word.lower().replace("'", "") for word in words]


def split_speeches(ids, seed=42):
    ids = sorted(ids)
    random.Random(seed).shuffle(ids)
    n = len(ids)
    n_tr, n_va = int(0.70 * n), int(0.15 * n)
    return ids[:n_tr], ids[n_tr:n_tr + n_va], ids[n_tr + n_va:]


def word_feats(words, i, window=2, full=True):
    features = {"bias": 1.0, "w": words[i]}
    for distance in range(1, window + 1):
        features["w-%d" % distance] = words[i - distance] if i - distance >= 0 else "<s>"
        features["w+%d" % distance] = words[i + distance] if i + distance < len(words) else "</s>"
    if full:
        word = words[i]
        features["suf3"] = word[-3:]
        features["pre2"] = word[:2]
        features["len"] = str(min(len(word), 10))
        features["digit"] = str(any(char.isdigit() for char in word))
        features["w-1|w"] = features["w-1"] + "|" + word if window >= 1 else ""
        features["w|w+1"] = word + "|" + features["w+1"] if window >= 1 else ""
        if window >= 2:
            features["w-1|w|w+1"] = features["w-1"] + "|" + word + "|" + features["w+1"]
            features["w|w+1|w+2"] = word + "|" + features["w+1"] + "|" + features["w+2"]
            features["w-2|w-1|w"] = features["w-2"] + "|" + features["w-1"] + "|" + word
    return features


def only_word_feats(words, i):
    return {"bias": 1.0, "w": words[i]}


def stream_feats(words, window=2, full=True, fn=None):
    if fn is not None:
        return [fn(words, i) for i in range(len(words))]
    return [word_feats(words, i, window, full) for i in range(len(words))]


def chunk(seq, n=100):
    return [seq[i:i + n] for i in range(0, len(seq), n)]


def rule_baseline(words, period_every=20):
    labels, since = [], 0
    for i, word in enumerate(words):
        since += 1
        label = "O"
        nxt = words[i + 1] if i + 1 < len(words) else ""
        if since >= period_every:
            label, since = "PERIOD", 0
        elif nxt in CONJ and since >= 6:
            label, since = "COMMA", 0
        labels.append(label)
    return labels


def attach_punct(words, labels):
    symbols = {"O": "", "COMMA": ",", "PERIOD": ".", "QUESTION": "?"}
    return [word + symbols[label] for word, label in zip(words, labels)]
