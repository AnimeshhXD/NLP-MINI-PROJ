"""
task6c.py  --  Task 6c (6 sub-tasks)

1. Punct F1 for Whisper and BERT on this video
2. Hybrid system format metrics vs Whisper raw segments vs baseline
3. WER errors with +-5 word context; apostrophe count
4. Block production function, first 5 blocks, Whisper segment count
5. verify_real.py per-number tolerance (0.5 * 10^-d); rerun
6. Print metrics_all.json, bootstrap.json (PE-in-CI), data_reconciliation.md

NON-NEGOTIABLE: no typed numbers; no simulation claims.
"""
import json, pathlib, re, sys, time, difflib, pickle
import numpy as np
import jiwer
from whisper.normalizers import EnglishTextNormalizer

T0 = time.time()
def log(*a): print(f"[{time.time()-T0:5.0f}s]", *a, flush=True)
SEP = "=" * 70

STEM    = "weekly_2015_vra"
ASR_J   = pathlib.Path(f"asr/{STEM}.json")
REF_TXT = pathlib.Path(f"refs/{STEM}.txt")
CAPS    = pathlib.Path("captions")
RESULTS = pathlib.Path("results_new")

MAX_CHARS = 42; MAX_LINES = 2; MAX_CPS = 17.0; MIN_DUR = 1.0; MAX_DUR = 7.0
PUNCT_SYM = {"O": "", "COMMA": ",", "PERIOD": ".", "QUESTION": "?"}
NORM = EnglishTextNormalizer()

# ──────────────────────────────────────────── helpers ────────────────────────

def asr_clean(w):
    return re.sub(r"[^\w]", "", w.lower())

def get_trailing_punct(word):
    w = word.rstrip()
    if w.endswith(","): return "COMMA"
    if w.endswith("?"):  return "QUESTION"
    if w.endswith(".") or w.endswith("!") or w.endswith(";"):
        return "PERIOD"
    return "O"

def split_lines(text, max_c=MAX_CHARS):
    words, lines, cur = text.split(), [], ""
    for w in words:
        cand = (cur + " " + w).strip()
        if len(cand) <= max_c: cur = cand
        else:
            if cur: lines.append(cur)
            cur = w
    if cur: lines.append(cur)
    return lines

def make_blocks(display_words, word_times):
    blocks = []; i = 0; n = len(display_words)
    while i < n:
        buf, buf_t = [], []
        while i < n:
            word = display_words[i]
            test = " ".join(buf + [word])
            if len(split_lines(test)) <= MAX_LINES:
                buf.append(word); buf_t.append(word_times[i]); i += 1
            else: break
        if not buf:
            buf.append(display_words[i]); buf_t.append(word_times[i]); i += 1
        t_start = buf_t[0][0]; t_last = buf_t[-1][1]
        t_end = max(word_times[i][0] if i < n else t_last, t_last)
        text = " ".join(buf)
        t_end = max(t_end, t_start + len(text) / MAX_CPS)
        t_end = max(t_end, t_start + MIN_DUR)
        t_end = min(t_end, t_start + MAX_DUR)
        blocks.append({"start": round(t_start, 3), "end": round(t_end, 3),
                       "lines": split_lines(text)[:MAX_LINES]})
    for j in range(1, len(blocks)):
        if blocks[j]["start"] < blocks[j-1]["end"]:
            blocks[j]["start"] = blocks[j-1]["end"]
        if blocks[j]["end"] <= blocks[j]["start"]:
            blocks[j]["end"] = round(blocks[j]["start"] + MIN_DUR, 3)
    return blocks

def parse_srt(path):
    blocks = []
    text = path.read_text(encoding="utf-8")
    for part in re.split(r"\n\n+", text.strip()):
        lines = part.strip().splitlines()
        if len(lines) < 2: continue
        m = re.match(r"(\d+:\d+:\d+,\d+)\s*-->\s*(\d+:\d+:\d+,\d+)", lines[1])
        if not m: continue
        def ts(s):
            h, mi, rest = s.split(":", 2); sec, ms = rest.split(",")
            return int(h)*3600 + int(mi)*60 + int(sec) + int(ms)/1000
        content = " ".join(lines[2:])
        blocks.append({"start": ts(m.group(1)), "end": ts(m.group(2)),
                       "text": content, "lines": lines[2:]})
    return blocks

def render_per_word(asr_words, labels, tc, ap):
    out = []; start = True
    for w, l in zip(asr_words, labels):
        t = ap.get(w, w)
        if w not in ap: t = tc.lex.get(w, t)
        if start and t[:1].islower(): t = t[0].upper() + t[1:]
        if w == "i": t = "I"
        out.append(t + PUNCT_SYM[l]); start = l in ("PERIOD", "QUESTION")
    return out

# ──────────────────────────────────────────── BERT ────────────────────────────

log("loading BERT...")
import torch
from transformers import DistilBertForTokenClassification, DistilBertTokenizerFast

_LABELS = ["O", "COMMA", "PERIOD", "QUESTION"]
_WINDOW = 128; _OVERLAP = 16
_tok   = DistilBertTokenizerFast.from_pretrained("models/bert_punct/tokenizer")
_model = DistilBertForTokenClassification.from_pretrained("models/bert_punct/unweighted")
_model.eval()

def bert_predict(words):
    if not words: return []
    enc = _tok(words, is_split_into_words=True, add_special_tokens=False)
    ids = enc["input_ids"]; word_ids = enc.word_ids(); n = len(ids)
    logit_sum = torch.zeros(n, len(_LABELS)); counts = torch.zeros(n)
    start = 0
    while start < n:
        end = min(start+_WINDOW, n); chunk = ids[start:end]; pad = _WINDOW-len(chunk)
        inp  = torch.tensor([[*chunk, *([0]*pad)]])
        mask = torch.tensor([[1]*len(chunk) + [0]*pad])
        with torch.no_grad():
            logits = _model(input_ids=inp, attention_mask=mask).logits[0, :len(chunk)]
        logit_sum[start:end] += logits; counts[start:end] += 1
        if end == n: break
        start = end - _OVERLAP
    avg = logit_sum / counts.unsqueeze(1); preds = avg.argmax(-1).tolist()
    word_preds = ["O"]*len(words); seen = set()
    for pos, wid in enumerate(word_ids):
        if wid is not None and wid < len(words) and wid not in seen:
            seen.add(wid); word_preds[wid] = _LABELS[preds[pos]]
    return word_preds

log("  BERT loaded")

# ──────────────────────────────────────────── truecaser + data ────────────────

sys.path.insert(0, ".")
from capnlp import TrueCaser, fit_apostrophe_lexicon

log("loading data.pkl (project's own training data)...")
D   = pickle.load(open("data.pkl", "rb"))
_tc = TrueCaser().fit(D["train"].values())
_ap = fit_apostrophe_lexicon(D["train"].values())
log("  done")

# ──────────────────────────────────────────── load ASR ────────────────────────

asr_data = json.loads(ASR_J.read_text())
raw_ws = []
for seg in asr_data["segments"]:
    for w in seg.get("words", []):
        txt = w["word"].strip()
        if txt:
            raw_ws.append((txt, float(w["start"]), float(w["end"])))

asr_clean_words = [asr_clean(w) for w, _, _ in raw_ws]
word_times_all  = [(s, e) for _, s, e in raw_ws]
# Filter empty tokens
nonempty       = [(c, t, w) for c, t, w in zip(asr_clean_words, word_times_all,
                                                [r for r, _, _ in raw_ws]) if c]
asr_clean_words  = [x[0] for x in nonempty]
word_times_all   = [x[1] for x in nonempty]
whisper_raw_ws   = [x[2] for x in nonempty]   # raw Whisper word text (e.g. "Hi,")

log(f"  {len(asr_clean_words)} Whisper words, {len(asr_data['segments'])} segments")

ref_raw   = REF_TXT.read_text(encoding="utf-8").strip()
ref_words = ref_raw.split()
ref_clean = [re.sub(r"[^\w]", "", w).lower() for w in ref_words]

log("running BERT...")
bert_labels = bert_predict(asr_clean_words)
log("  done")

# Render BERT display words and build BERT blocks
bert_display = render_per_word(asr_clean_words, bert_labels, _tc, _ap)
bert_blocks  = make_blocks(bert_display, word_times_all)

# ═══════════════════════════════════════ TASK 1: PUNCT F1 ═════════════════════

print(); print(SEP); print("TASK 1  -- punctuation-only F1 (comma/period)"); print(SEP)

# Sequence alignment
sm = difflib.SequenceMatcher(None, ref_clean, asr_clean_words, autojunk=False)

ref_labels_ev   = []   # one entry per evaluation point
wh_labels_ev    = []
bert_labels_ev  = []

for tag, i1, i2, j1, j2 in sm.get_opcodes():
    if tag == "equal":
        for d in range(i2 - i1):
            ref_labels_ev.append(get_trailing_punct(ref_words[i1+d]))
            wh_labels_ev.append(get_trailing_punct(whisper_raw_ws[j1+d]))
            bert_labels_ev.append(bert_labels[j1+d])
    elif tag == "replace":
        for d in range(min(i2-i1, j2-j1)):
            ref_labels_ev.append(get_trailing_punct(ref_words[i1+d]))
            wh_labels_ev.append(get_trailing_punct(whisper_raw_ws[j1+d]))
            bert_labels_ev.append(bert_labels[j1+d])
        # extra ref words (deleted): hyp predicts O
        for d in range(min(i2-i1, j2-j1), i2-i1):
            ref_labels_ev.append(get_trailing_punct(ref_words[i1+d]))
            wh_labels_ev.append("O"); bert_labels_ev.append("O")
        # extra hyp words (inserted): ref is O
        for d in range(min(i2-i1, j2-j1), j2-j1):
            ref_labels_ev.append("O")
            wh_labels_ev.append(get_trailing_punct(whisper_raw_ws[j1+d]))
            bert_labels_ev.append(bert_labels[j1+d])
    elif tag == "delete":
        for d in range(i2-i1):
            ref_labels_ev.append(get_trailing_punct(ref_words[i1+d]))
            wh_labels_ev.append("O"); bert_labels_ev.append("O")
    elif tag == "insert":
        for d in range(j2-j1):
            ref_labels_ev.append("O")
            wh_labels_ev.append(get_trailing_punct(whisper_raw_ws[j1+d]))
            bert_labels_ev.append(bert_labels[j1+d])

def prf(ref, hyp, mark):
    tp = sum(1 for r, h in zip(ref, hyp) if r==mark and h==mark)
    fp = sum(1 for r, h in zip(ref, hyp) if r!=mark and h==mark)
    fn = sum(1 for r, h in zip(ref, hyp) if r==mark and h!=mark)
    P = tp/(tp+fp) if (tp+fp) else 0.0
    R = tp/(tp+fn) if (tp+fn) else 0.0
    F1 = 2*P*R/(P+R) if (P+R) else 0.0
    return {"P": round(P,4), "R": round(R,4), "F1": round(F1,4),
            "TP": tp, "FP": fp, "FN": fn}

wh_comma  = prf(ref_labels_ev, wh_labels_ev, "COMMA")
wh_period = prf(ref_labels_ev, wh_labels_ev, "PERIOD")
bt_comma  = prf(ref_labels_ev, bert_labels_ev, "COMMA")
bt_period = prf(ref_labels_ev, bert_labels_ev, "PERIOD")

ref_commas   = sum(1 for l in ref_labels_ev  if l == "COMMA")
ref_periods  = sum(1 for l in ref_labels_ev  if l == "PERIOD")
wh_commas_n  = sum(1 for l in wh_labels_ev   if l == "COMMA")
wh_periods_n = sum(1 for l in wh_labels_ev   if l == "PERIOD")
bt_commas_n  = sum(1 for l in bert_labels_ev if l == "COMMA")
bt_periods_n = sum(1 for l in bert_labels_ev if l == "PERIOD")

print(f"\n  Reference: {len(ref_words)} words,  "
      f"commas={ref_commas},  periods={ref_periods}")
print(f"  Whisper:   {len(asr_clean_words)} words,  "
      f"commas={wh_commas_n},  periods={wh_periods_n}")
print(f"  BERT:      {len(asr_clean_words)} words,  "
      f"commas={bt_commas_n},  periods={bt_periods_n}")
print()
print(f"  Evaluation points: {len(ref_labels_ev)}")
print()
print(f"  {'System':<12} {'Mark':<7} {'P':>7} {'R':>7} {'F1':>7}   TP/FP/FN")
print("  " + "-"*50)
for sname, mc, mp in [("Whisper", wh_comma, wh_period),
                       ("BERT",    bt_comma, bt_period)]:
    for mark, m in [("COMMA", mc), ("PERIOD", mp)]:
        print(f"  {sname:<12} {mark:<7} {m['P']:>7.4f} {m['R']:>7.4f} {m['F1']:>7.4f}"
              f"   {m['TP']}/{m['FP']}/{m['FN']}")

# ═══════════════════════════════════════ TASK 2: FORMAT METRICS ═══════════════

print(); print(SEP); print("TASK 2  -- hybrid system vs Whisper segments vs baseline"); print(SEP)

FUNCTION_WORDS = {
    "a", "an", "the", "and", "but", "or", "nor", "for", "yet", "so",
    "in", "on", "at", "to", "of", "up", "by", "as", "if", "it", "its",
    "is", "was", "are", "were", "be", "been", "have", "has", "had",
    "do", "does", "did", "not", "this", "that", "these", "those",
    "with", "from", "into", "then", "when", "where", "which", "who",
    "whose", "how", "any", "some", "no", "all", "both", "each",
    "i", "we", "you", "he", "she", "they", "me", "us", "him", "her", "them",
    "my", "our", "your", "his", "their", "about", "after", "before",
    "between", "over", "under", "through", "because", "while", "although",
    "since", "until", "unless", "whether", "just", "also", "too",
    "very", "quite", "really", "will", "would", "could", "should", "may",
    "might", "shall", "can", "than", "more", "there", "here", "what",
    "am", "then", "now", "only", "even",
}

def fmt_metrics(blocks, name):
    n = len(blocks)
    if n == 0: return {}
    w2x42 = fwd = fwend = midword = 0
    durs = []
    for b in blocks:
        text = b.get("text", " ".join(b.get("lines", [])))
        dur  = b["end"] - b["start"]
        durs.append(dur)
        wrapped = split_lines(text)
        if len(wrapped) <= MAX_LINES and all(len(l) <= MAX_CHARS for l in wrapped):
            w2x42 += 1
        if dur > 0 and len(text) / dur <= MAX_CPS:
            fwd += 1
        elif dur <= 0:
            fwd += 1
        words = text.rstrip().split()
        last_clean = re.sub(r"[^\w]", "", words[-1]).lower() if words else ""
        if last_clean in FUNCTION_WORDS:
            fwend += 1
        if any(len(w) > MAX_CHARS for w in words):
            midword += 1
    arr = np.array(durs)
    r = {"n": n,
         "pct_within_2x42": round(100.0*w2x42/n, 1),
         "pct_within_17cps": round(100.0*fwd/n, 1),
         "pct_ends_func": round(100.0*fwend/n, 1),
         "mid_word_breaks": midword,
         "mean_dur_s": round(float(arr.mean()), 2),
         "median_dur_s": round(float(np.median(arr)), 2)}
    print(f"\n  [{name}]  n={n}")
    print(f"    within 2x42 chars:   {r['pct_within_2x42']:>5.1f}%  ({w2x42}/{n})")
    print(f"    within 17 cps:       {r['pct_within_17cps']:>5.1f}%  ({fwd}/{n})")
    print(f"    ends on func word:   {r['pct_ends_func']:>5.1f}%  ({fwend}/{n})")
    print(f"    mid-word breaks:     {midword}")
    print(f"    mean duration:       {r['mean_dur_s']:.2f} s")
    print(f"    median duration:     {r['median_dur_s']:.2f} s")
    return r

# Hybrid: Whisper's own punct+casing -> segmenter
whisper_display = whisper_raw_ws   # e.g. ["Hi,", "everybody.", "The", ...]
hybrid_blocks   = make_blocks(whisper_display, word_times_all)

# Whisper raw segments as captions (one block per segment)
whisper_seg_blocks = [{"start": float(s["start"]), "end": float(s["end"]),
                       "text": s["text"].strip()}
                      for s in asr_data["segments"] if s["text"].strip()]

# Fixed-width baseline (no punct, from task6b)
baseline_blocks = parse_srt(CAPS / f"baseline_{STEM}.srt")

h_r  = fmt_metrics(hybrid_blocks,      "Hybrid (Whisper punct -> segmenter)")
ws_r = fmt_metrics(whisper_seg_blocks,  "Whisper raw segments as captions")
bl_r = fmt_metrics(baseline_blocks,     "Fixed-width baseline (no punct, task6b)")

fmt_results = {"hybrid": h_r, "whisper_segments": ws_r, "baseline": bl_r}

# ═══════════════════════════════════════ TASK 3: WER ERRORS + CONTEXT ═════════

print(); print(SEP); print("TASK 3  -- WER errors with +-5 word context"); print(SEP)

whisper_hyp = " ".join(whisper_raw_ws)
bert_hyp    = " ".join(bert_display)

def wer_context(ref, hyp, label, ctx=5):
    out     = jiwer.process_words(ref, hyp)
    ref_t   = ref.split(); hyp_t   = hyp.split()
    print(f"\n  [{label}]  WER={out.wer*100:.2f}%  "
          f"sub={out.substitutions}  ins={out.insertions}  del={out.deletions}")
    print()
    apos_count = 0
    for chunk in out.alignments[0]:
        if chunk.type == "equal": continue
        ri1, ri2 = chunk.ref_start_idx, chunk.ref_end_idx
        hi1, hi2 = chunk.hyp_start_idx, chunk.hyp_end_idx
        r_words  = ref_t[ri1:ri2]
        h_words  = hyp_t[hi1:hi2]
        # ref context
        rc_lo = max(0, ri1 - ctx); rc_hi = min(len(ref_t), ri2 + ctx)
        pre_r = " ".join(ref_t[rc_lo:ri1])
        post_r= " ".join(ref_t[ri2:rc_hi])
        r_str = " ".join(r_words) if r_words else "[ins]"
        h_str = " ".join(h_words) if h_words else "[del]"
        print(f"  pos {ri1:>3} [{chunk.type:<10}]")
        print(f"    ref: ...{pre_r} >>>{r_str}<<< {post_r}...")
        print(f"    hyp:  {h_str}")
        # apostrophe check
        for rw in (r_words or [""]):
            for hw in (h_words or [""]):
                if rw and hw:
                    rns = rw.lower().replace("'", "")
                    hns = hw.lower().replace("'", "")
                    if rns and hns and rns == hns and rw.lower() != hw.lower():
                        print(f"    [apostrophe diff: ref '{rw}' vs hyp '{hw}']")
                        apos_count += 1
        print()
    print(f"  Apostrophe-related content errors: {apos_count}")
    return apos_count

wh_apos = wer_context(ref_raw, whisper_hyp, "Whisper own punct -- WER_strict")
bt_apos = wer_context(ref_raw, bert_hyp,    "BERT punct -- WER_strict")

# ═══════════════════════════════════════ TASK 4: BLOCK PRODUCTION INFO ════════

print(); print(SEP); print("TASK 4  -- block production: function, first 5 blocks, segment count"); print(SEP)

print(f"\n  Function: make_blocks  (implemented in task6b.py and replicated here)")
print(f"  Algorithm: greedy word-packing; end = next word's Whisper start,")
print(f"             clamped to [t_start + min(len(text)/17, 1), t_start + 7]")
print(f"  Constraints: max 2 lines x 42 chars, max 17 chars/s, min 1 s, max 7 s")
print(f"  No mid-word breaks by construction (breaks only between whitespace tokens)")
print()
print(f"  Whisper segments in asr/{STEM}.json: {len(asr_data['segments'])}")
print(f"  BERT caption blocks produced:        {len(bert_blocks)}")
print()
print(f"  First 5 blocks (BERT system from task6b, re-derived):")
print(f"  {'#':>3}  {'start':>8}  {'end':>8}  {'dur':>6}  text")
print("  " + "-"*70)
for k, b in enumerate(bert_blocks[:5], 1):
    text = " / ".join(b["lines"])
    dur  = b["end"] - b["start"]
    print(f"  {k:>3}  {b['start']:>8.3f}  {b['end']:>8.3f}  {dur:>5.2f}s  {text}")

# ═══════════════════════════════════════ TASK 5: verify_real.py TOLERANCE ═════

print(); print(SEP); print("TASK 5  -- per-number tolerance in verify_real.py + rerun"); print(SEP)

VR = pathlib.Path("verify_real.py")
vr_text = VR.read_text(encoding="utf-8")

OLD_VAL_LINE  = "            val = round(float(n_str), 3)"
NEW_VAL_LINE  = "            val = float(n_str)  # no rounding: tolerance is computed per-digit"
OLD_TOL_LINE  = "            if not any(abs(val - j) <= 0.05 for j in all_json_numbers):"
NEW_TOL_LINE  = ("            d   = len(n_str.split('.')[1]) if '.' in n_str else 0\n"
                 "            tol = 0.5 * (10.0 ** (-d))  # 0.5 ulp for printed digits\n"
                 "            if not any(abs(val - j) <= tol for j in all_json_numbers):")
OLD_COLLECT   = "            all_json_numbers.add(round(float(obj), 3))"
NEW_COLLECT   = "            all_json_numbers.add(float(obj))  # full precision"

changes = [
    (OLD_COLLECT,  NEW_COLLECT),
    (OLD_VAL_LINE, NEW_VAL_LINE),
    (OLD_TOL_LINE, NEW_TOL_LINE),
]

new_vr = vr_text
all_found = True
for old, new in changes:
    if old in new_vr:
        new_vr = new_vr.replace(old, new, 1)
    else:
        print(f"  [WARN] pattern not found: {old[:60]!r}")
        all_found = False

import difflib as _dl
diff = "".join(_dl.unified_diff(
    vr_text.splitlines(keepends=True),
    new_vr.splitlines(keepends=True),
    fromfile="verify_real.py (before)",
    tofile="verify_real.py (after)",
))

print("\nFull diff:")
print(diff if diff else "  (no changes; patterns already updated)")

if diff and all_found:
    VR.write_text(new_vr, encoding="utf-8")
    print("\n  Written.")
elif not diff:
    print("  File already has per-digit tolerance; no change.")

# Also add task6c_results.json to the collector; save results first
task6c_json = {
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "task1_punct_f1": {
        "ref_words":  len(ref_words),
        "ref_commas": ref_commas,
        "ref_periods": ref_periods,
        "whisper_commas_predicted": wh_commas_n,
        "whisper_periods_predicted": wh_periods_n,
        "bert_commas_predicted": bt_commas_n,
        "bert_periods_predicted": bt_periods_n,
        "eval_points": len(ref_labels_ev),
        "whisper": {"comma": wh_comma, "period": wh_period},
        "bert":    {"comma": bt_comma, "period": bt_period},
    },
    "task2_format": fmt_results,
    "task3_apostrophes": {"whisper": wh_apos, "bert": bt_apos},
    "task4_segments": len(asr_data["segments"]),
    "task4_bert_blocks": len(bert_blocks),
    "task4_first5": [{"start": b["start"], "end": b["end"],
                      "text": " ".join(b["lines"])} for b in bert_blocks[:5]],
}
(RESULTS / "task6c_results.json").write_text(
    json.dumps(task6c_json, indent=2), encoding="utf-8")
log("Wrote results_new/task6c_results.json")

# Add task6c collector to verify_real.py
VR2 = VR.read_text(encoding="utf-8")
T6C_LOAD = ('task6c_r = json.loads((RESULTS / "task6c_results.json").read_text(encoding="utf-8")) \\\n'
            '           if (RESULTS / "task6c_results.json").exists() else {}')
T6C_COLLECT = "    collect_nums(task6c_r)"

if "task6c_r" not in VR2:
    vr2_new = VR2.replace(
        'task6b_r = json.loads((RESULTS / "task6b_results.json")',
        T6C_LOAD + '\n' + 'task6b_r = json.loads((RESULTS / "task6b_results.json")'
    )
    vr2_new = vr2_new.replace(
        "    collect_nums(task6b_r)",
        "    collect_nums(task6b_r)\n" + T6C_COLLECT
    )
    VR.write_text(vr2_new, encoding="utf-8")
    print("\n  Added task6c_r to verify_real.py collector.")

# Run verify_real.py
import subprocess
print("\nRunning verify_real.py...")
vr_res = subprocess.run([sys.executable, "verify_real.py"], capture_output=True, text=True)
tail = vr_res.stdout.splitlines()
# Print last 12 lines
print("\n".join(tail[-12:]))
if vr_res.returncode != 0:
    print("STDERR:", vr_res.stderr[-500:])
    print("[FAIL]  verify_real.py exited with", vr_res.returncode)
else:
    print("[OK]  verify_real.py PASS")

# ═══════════════════════════════════════ TASK 6: PRINT JSONS ══════════════════

print(); print(SEP); print("TASK 6  -- metrics_all.json, bootstrap.json (PE-in-CI), data_reconciliation.md")
print(SEP)

metrics = json.loads((RESULTS / "metrics_all.json").read_text())
boot    = json.loads((RESULTS / "bootstrap.json").read_text())

print("\n--- results_new/metrics_all.json ---")
print(json.dumps(metrics, indent=2))

print("\n--- results_new/bootstrap.json (PE-in-CI check) ---")
ci_r = boot.get("ci_results", {})
all_ok = True
for sname, sdata in ci_r.items():
    for mkey, entry in sdata.items():
        pe  = entry.get("point_estimate"); lo = entry.get("CI_95_lo"); hi = entry.get("CI_95_hi")
        ok  = (lo is not None and hi is not None and pe is not None and lo <= pe <= hi)
        if not ok: all_ok = False
        print(f"  {sname}/{mkey}: PE={pe:.4f} CI=[{lo:.4f},{hi:.4f}] {'OK' if ok else 'FAIL'}")
print(f"\nPE-in-CI: all {'PASS' if all_ok else 'FAIL'}")
print("\nFull bootstrap.json:")
print(json.dumps(boot, indent=2))

print("\n--- data_reconciliation.md ---")
print(pathlib.Path("data_reconciliation.md").read_text(encoding="utf-8"))

print(); print(SEP); print("Task 6c DONE"); print(SEP)
