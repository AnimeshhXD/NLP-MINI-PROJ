"""
capnlp.py - core library for the closed-caption post-processing pipeline.

Stages (see report Chapter 6):
  1. load + clean the punctuated reference transcripts
  2. make ASR-style input (lower-case, no punctuation, no apostrophes)
  3. disfluency removal  (fillers, immediate repeats)
  4. punctuation restoration  (rule baseline / logistic regression / CRF)
  5. truecasing + apostrophe restoration (grammar-lite stage)
  6. caption segmentation (42 chars/line, 2 lines, 17 chars/sec) + SRT output
"""
import re, random, unicodedata, collections
from nltk.corpus import state_union, inaugural

LABELS = ["O", "COMMA", "PERIOD", "QUESTION"]
FILLERS = {"um", "uh", "er", "hmm"}
REPEAT_WHITELIST = {"had", "that"}
FUNCTION_WORDS = set("the a an of to and but in on with for that which is are was be by as at or from".split())

# ------------------------------------------------------------------ 1. loading
def load_speeches():
    """Return {speech_id: raw_text} for State of the Union + Inaugural corpora."""
    out = {}
    for f in state_union.fileids():
        out["su_" + f[:-4]] = state_union.raw(f)
    for f in inaugural.fileids():
        out["in_" + f[:-4]] = inaugural.raw(f)
    return out


def clean_raw(text):
    """Conservative cleaning of one raw speech (Chapter 4, Table 4.1)."""
    text = unicodedata.normalize("NFC", text)
    text = (text.replace("\u201c", '"').replace("\u201d", '"')
                .replace("\u2018", "'").replace("\u2019", "'"))
    lines = text.split("\n")
    # drop upper-case title lines such as "PRESIDENT ... ANNUAL MESSAGE ..."
    lines = [l for l in lines if not (sum(c.isupper() for c in l) > 0.8 * max(1, sum(c.isalpha() for c in l))
                                      and len(l.split()) > 3)]
    text = " ".join(lines)
    text = re.sub(r"\[[^\]]*\]|\([^)]*\)", " ", text)         # bracketed annotations
    text = re.sub(r"--+|\u2014", " , ", text)                  # dashes -> comma
    text = re.sub(r"\b(?:[A-Za-z]\.){2,}", lambda m: m.group(0).replace(".", ""), text)  # U.S. -> US
    text = re.sub(r"\b(Mr|Mrs|Ms|Dr|St|Jr|Sr|Hon|Gen|Gov|Col|Lt|Sen|Rep)\.", r"\1", text)
    text = re.sub(r"\b([A-HJ-Z])\.(?=\s+[A-Z])", r"\1", text)  # middle initials
    text = re.sub(r"\s+", " ", text).strip()
    return text


TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['\-][A-Za-z0-9]+)*(?:[.,][0-9]+)*|[.,?!;:]")


def to_tokens(text):
    """Return list of (word, punct_label, case_label) from a cleaned reference text."""
    toks = TOKEN_RE.findall(text)
    out = []
    for t in toks:
        if t in ".,?!;:":
            if not out:
                continue
            w, lab, cs = out[-1]
            new = "QUESTION" if t == "?" else "PERIOD" if t in ".!" else "COMMA"
            out[-1] = (w, new, cs)
        else:
            if t.isupper() and len(t) > 1 and t.isalpha():
                cs = "UPPER"
            elif t[0].isupper():
                cs = "CAP"
            else:
                cs = "LOWER"
            out.append((t, "O", cs))
    return out


def asr_style(words):
    """ASR-style view of the words: lower-case, apostrophes removed (Chapter 4, Table 4.4)."""
    return [w.lower().replace("'", "") for w in words]


def split_speeches(ids, seed=42):
    ids = sorted(ids)
    random.Random(seed).shuffle(ids)
    n = len(ids)
    n_tr, n_va = int(0.70 * n), int(0.15 * n)
    return ids[:n_tr], ids[n_tr:n_tr + n_va], ids[n_tr + n_va:]


def inject_disfluency(words, p_fill=0.03, p_rep=0.02, seed=0):
    """Simulate ASR/spontaneous-speech noise on an ASR-style word list."""
    rng = random.Random(seed)
    out = []
    for w in words:
        if rng.random() < p_fill:
            out.append(rng.choice(sorted(FILLERS - {"hmm"})))
        out.append(w)
        if rng.random() < p_rep:
            out.append(w)
    return out

# ------------------------------------------------------------- 3. disfluencies
def remove_disfluencies(words):
    out = []
    for w in words:
        if w in FILLERS:
            continue
        if out and w == out[-1] and w not in REPEAT_WHITELIST:
            continue
        out.append(w)
    return out

# ------------------------------------------------------ 4. features and models
def word_feats(words, i, window=2, full=True):
    f = {"bias": 1.0, "w": words[i]}
    for d in range(1, window + 1):
        f["w-%d" % d] = words[i - d] if i - d >= 0 else "<s>"
        f["w+%d" % d] = words[i + d] if i + d < len(words) else "</s>"
    if full:
        w = words[i]
        f["suf3"] = w[-3:]
        f["pre2"] = w[:2]
        f["len"] = str(min(len(w), 10))
        f["digit"] = str(any(c.isdigit() for c in w))
        f["w-1|w"] = f["w-1"] + "|" + w if window >= 1 else ""
        f["w|w+1"] = w + "|" + f["w+1"] if window >= 1 else ""
        if window >= 2:
            f["w-1|w|w+1"] = f["w-1"] + "|" + w + "|" + f["w+1"]
            f["w|w+1|w+2"] = w + "|" + f["w+1"] + "|" + f["w+2"]
            f["w-2|w-1|w"] = f["w-2"] + "|" + f["w-1"] + "|" + w
    return f


def only_word_feats(words, i):
    return {"bias": 1.0, "w": words[i]}


def stream_feats(words, window=2, full=True, fn=None):
    if fn is not None:
        return [fn(words, i) for i in range(len(words))]
    return [word_feats(words, i, window, full) for i in range(len(words))]


def chunk(seq, n=100):
    return [seq[i:i + n] for i in range(0, len(seq), n)]

# ---- rule baseline
CONJ = {"and", "but", "which", "so", "because", "while", "although", "however", "where", "when"}


def rule_baseline(words, period_every=20):
    labs, since = [], 0
    for i, w in enumerate(words):
        since += 1
        lab = "O"
        nxt = words[i + 1] if i + 1 < len(words) else ""
        if since >= period_every:
            lab, since = "PERIOD", 0
        elif nxt in CONJ and since >= 6:
            lab, since = "COMMA", 0
        labs.append(lab)
    return labs

# ------------------------------------------------------------ 5. truecase etc.
class TrueCaser:
    def __init__(self):
        self.lex = {}

    def fit(self, token_docs):
        cnt = collections.defaultdict(collections.Counter)
        for doc in token_docs:
            prev = "PERIOD"
            for w, lab, cs in doc:
                if prev not in ("PERIOD", "QUESTION"):          # skip sentence-initial positions
                    cnt[w.lower().replace("'", "")][w.replace("'", "") if cs != "LOWER" else w.lower().replace("'", "")] += 1
                prev = lab
        for k, c in cnt.items():
            form, n = c.most_common(1)[0]
            if form != k:                                        # cased form wins
                tot = sum(c.values())
                if n / tot > 0.5:
                    self.lex[k] = form
        self.lex["i"] = "I"
        return self

    def apply(self, words, labels):
        out, start = [], True
        for w, lab in zip(words, labels):
            t = self.lex.get(w, w)
            if start and t[:1].islower():
                t = t[0].upper() + t[1:]
            out.append(t)
            start = lab in ("PERIOD", "QUESTION")
        return out


def fit_apostrophe_lexicon(token_docs):
    """Map stripped form -> apostrophe form when apostrophe form is more frequent in train."""
    with_a, plain = collections.Counter(), collections.Counter()
    for doc in token_docs:
        for w, _, _ in doc:
            lw = w.lower()
            if "'" in lw:
                with_a[lw] += 1
            else:
                plain[lw] += 1
    lex = {}
    for a, n in with_a.items():
        s = a.replace("'", "")
        if n > plain.get(s, 0) and n >= 2:
            lex[s] = a
    return lex


def attach_punct(words, labels):
    sym = {"O": "", "COMMA": ",", "PERIOD": ".", "QUESTION": "?"}
    return [w + sym[l] for w, l in zip(words, labels)]

# --------------------------------------------------------- 6. caption segmenter
MAX_CHARS, MAX_LINES, MAX_CPS, WPS = 42, 2, 17.0, 2.6


def _best_break(words, lo, hi, ends_punct):
    """Pick break index k (words[:k] | words[k:]) within [lo,hi], preferring clause boundaries."""
    best, best_s = hi, -1e9
    for k in range(lo, hi + 1):
        s = 0.0
        last = words[k - 1]
        nxt = words[k] if k < len(words) else ""
        if last[-1:] in ",.?":
            s += 3
        if nxt.lower() in CONJ:
            s += 2
        if last.lower().strip(",.?") in FUNCTION_WORDS:
            s -= 3
        s -= abs(k - hi) * 0.05
        if s > best_s:
            best, best_s = k, s
    return best


def segment_captions(tokens):
    """tokens: list of final display words (with punctuation attached).
    Returns list of blocks, each a list of lines (strings)."""
    # 1) split at sentence ends
    sents, cur = [], []
    for t in tokens:
        cur.append(t)
        if t[-1:] in ".?":
            sents.append(cur)
            cur = []
    if cur:
        sents.append(cur)
    blocks = []
    cap = MAX_CHARS * MAX_LINES
    for s in sents:
        rest = s
        while rest:
            text = " ".join(rest)
            if len(text) <= cap:
                chunk_w, rest = rest, []
            else:
                # largest k such that first k words fit in cap, then choose best break near it
                k, L = 0, 0
                while k < len(rest) and L + len(rest[k]) + (1 if k else 0) <= cap:
                    L += len(rest[k]) + (1 if k else 0)
                    k += 1
                k = max(k, 1)
                lo = max(1, k - 6)
                k = _best_break(rest, lo, k, True)
                chunk_w, rest = rest[:k], rest[k:]
            # wrap chunk into <=2 lines of <=42 chars
            text = " ".join(chunk_w)
            if len(text) <= MAX_CHARS:
                blocks.append([text])
            else:
                mid = len(chunk_w) // 2
                # find k s.t. both halves fit
                cands = [k for k in range(1, len(chunk_w))
                         if len(" ".join(chunk_w[:k])) <= MAX_CHARS and len(" ".join(chunk_w[k:])) <= MAX_CHARS]
                if cands:
                    def score(k):
                        return ((3 if chunk_w[k - 1][-1:] in ",.?" else 0)
                                + (2 if chunk_w[k].lower() in CONJ else 0)
                                - (3 if chunk_w[k - 1].lower().strip(",.?") in FUNCTION_WORDS else 0)
                                - abs(k - mid) * 0.1)
                    k = max(cands, key=score)
                    blocks.append([" ".join(chunk_w[:k]), " ".join(chunk_w[k:])])
                else:                                             # very long words: hard wrap
                    blocks.append([text[:MAX_CHARS], text[MAX_CHARS:2 * MAX_CHARS]])
    return blocks


def baseline_segments(words):
    """Platform-style baseline: unpunctuated words broken only by a character limit."""
    blocks, cur = [], []
    for w in words:
        trial = " ".join(cur + [w])
        if len(trial) > MAX_CHARS * MAX_LINES:
            blocks.append(_wrap2(cur))
            cur = [w]
        else:
            cur.append(w)
    if cur:
        blocks.append(_wrap2(cur))
    return blocks


def _wrap2(ws):
    line1, L = [], 0
    for i, w in enumerate(ws):
        if L + len(w) + (1 if line1 else 0) <= MAX_CHARS:
            line1.append(w)
            L += len(w) + (1 if line1[:-1] else 0)
        else:
            return [" ".join(line1), " ".join(ws[i:])]
    return [" ".join(line1)]


def block_stats(blocks):
    """Format conformity and linguistic quality of a block list."""
    ok_len = ok_cps = end_fw = 0
    for b in blocks:
        txt = " ".join(b)
        if len(b) <= MAX_LINES and all(len(l) <= MAX_CHARS for l in b):
            ok_len += 1
        dur = max(len(txt.split()) / WPS, 1.0)                    # min display time 1 s
        if len(txt) / dur <= MAX_CPS:
            ok_cps += 1
        if txt.split()[-1].lower().strip(",.?") in FUNCTION_WORDS:
            end_fw += 1
    n = max(len(blocks), 1)
    return {"blocks": len(blocks), "line_conform": ok_len / n, "cps_conform": ok_cps / n, "end_function_word": end_fw / n}


def _ts(t):
    h, m = int(t // 3600), int(t % 3600 // 60)
    s = t % 60
    return "%02d:%02d:%06.3f" % (h, m, s)


def to_srt(blocks, vtt=False):
    out, t = ([] if not vtt else ["WEBVTT", ""]), 0.0
    for i, b in enumerate(blocks, 1):
        txt = " ".join(b)
        dur = max(len(txt.split()) / WPS, 1.0)
        a, z = _ts(t), _ts(t + dur)
        if not vtt:
            a, z = a.replace(".", ","), z.replace(".", ",")
        out += [str(i), "%s --> %s" % (a, z), "\n".join(b), ""]
        t += dur
    return "\n".join(out)
