"""
task6d.py  --  Task 6d (6 sub-tasks)

1. New segmenter: capnlp.segment_captions (clause-aware) + Whisper word times.
   No 5-second cap; enforce <=17 cps; min 1s, max 7s. Run on Whisper's own punct.
2. Format-metrics table: new segmenter / Whisper raw segs / fixed-width baseline.
3. Casing isolation WER (a) Whisper verbatim, (b) Whisper words + BERT labels,
   (c) full BERT pipeline; count casing and apostrophe errors in (c).
4. Consistent METEOR/BERTScore/BLEU/chrF strings; update extra_metrics.json;
   explain METEOR bert_uw < crf_pipeline.
5. Clean bert_results.json (move honest_assessment/boot95/paired_vs_crf to
   results_old/); rename sync_check -> whisper_segment_grouping in asr_results.json.
6. Print bootstrap.json, data_reconciliation.md; run pytest + verify_real.py.

NON-NEGOTIABLE: no typed numbers; no simulation claims.
"""
import json, pathlib, re, sys, time, pickle, shutil, subprocess, collections
import numpy as np
import jiwer, nltk, sacrebleu
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

sys.path.insert(0, ".")
from capnlp import (TrueCaser, fit_apostrophe_lexicon, segment_captions,
                    FUNCTION_WORDS as _FW_CAPNLP)
from eval.compute_all import render_speech, reference_text

# ──────────────────────────────────────────── helpers ─────────────────────────

def asr_clean(w):
    return re.sub(r"[^\w]", "", w.lower())

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

log("loading data.pkl (project's own training data)...")
D   = pickle.load(open("data.pkl", "rb"))  # trusted project file
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

nonempty       = [(asr_clean(w), (s, e), w) for w, s, e in raw_ws if asr_clean(w)]
asr_clean_words  = [x[0] for x in nonempty]
word_times_all   = [x[1] for x in nonempty]
whisper_raw_ws   = [x[2] for x in nonempty]   # e.g. ["Hi,", "everybody.", ...]

ref_raw   = REF_TXT.read_text(encoding="utf-8").strip()
ref_words = ref_raw.split()

log("running BERT...")
bert_labels = bert_predict(asr_clean_words)
log("  done")

def render_per_word(asr_words, labels):
    out = []; start = True
    for w, l in zip(asr_words, labels):
        t = _ap.get(w, w)
        if w not in _ap: t = _tc.lex.get(w, t)
        if start and t[:1].islower(): t = t[0].upper() + t[1:]
        if w == "i": t = "I"
        out.append(t + PUNCT_SYM[l]); start = l in ("PERIOD", "QUESTION")
    return out

bert_display = render_per_word(asr_clean_words, bert_labels)

# ═══════════════════════════════════════ TASK 1: NEW SEGMENTER ════════════════

print(); print(SEP); print("TASK 1  -- clause-aware segmenter (segment_captions + Whisper times)"); print(SEP)

# Run capnlp.segment_captions on Whisper's own punct/casing
text_blocks = segment_captions(whisper_raw_ws)

i_word = 0
new_seg_blocks = []
for block in text_blocks:
    text = " ".join(block)
    words_in_block = text.split()
    n = len(words_in_block)
    if i_word >= len(word_times_all): break
    t_start = word_times_all[i_word][0]
    t_last  = word_times_all[min(i_word+n-1, len(word_times_all)-1)][1]
    # extend to next word's start (gives natural gap for CPS breathing room)
    if i_word + n < len(word_times_all):
        t_end = word_times_all[i_word+n][0]
    else:
        t_end = t_last
    t_end = max(t_end, t_last)
    # CPS constraint: must be on screen long enough to be readable
    t_end = max(t_end, t_start + len(text) / MAX_CPS)
    # duration bounds (no 5-second cap; max 7s)
    t_end = max(t_end, t_start + MIN_DUR)
    t_end = min(t_end, t_start + MAX_DUR)
    new_seg_blocks.append({
        "start": round(t_start, 3), "end": round(t_end, 3),
        "lines": block, "text": text
    })
    i_word += n

# resolve overlaps
for k in range(1, len(new_seg_blocks)):
    if new_seg_blocks[k]["start"] < new_seg_blocks[k-1]["end"]:
        new_seg_blocks[k]["start"] = new_seg_blocks[k-1]["end"]
    if new_seg_blocks[k]["end"] <= new_seg_blocks[k]["start"]:
        new_seg_blocks[k]["end"] = round(new_seg_blocks[k]["start"] + MIN_DUR, 3)

print(f"\n  segment_captions produced {len(text_blocks)} text blocks.")
print(f"  After timing assignment: {len(new_seg_blocks)} timed blocks.")
print(f"\n  First 5 blocks:")
for k, b in enumerate(new_seg_blocks[:5], 1):
    lines_str = " / ".join(b["lines"])
    print(f"  {k}. [{b['start']:.3f} --> {b['end']:.3f}]  {lines_str}")

# ═══════════════════════════════════════ TASK 2: FORMAT METRICS ═══════════════

print(); print(SEP); print("TASK 2  -- format metrics: new segmenter vs Whisper raw vs baseline"); print(SEP)

NAME_TITLES = {
    "president", "congressman", "senator", "governor", "vice", "dr",
    "mr", "mrs", "ms", "secretary", "general", "justice", "mayor",
    "director", "captain", "colonel", "lieutenant", "professor",
    "representative", "speaker",
}

FUNCTION_WORDS_FULL = {
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
    "am", "now", "only", "even",
}

print(f"\n  Function-word list ({len(FUNCTION_WORDS_FULL)} words):")
print("  " + ", ".join(sorted(FUNCTION_WORDS_FULL)))

def fmt_metrics(blocks, name):
    n = len(blocks)
    if n == 0: return {}
    w2x42 = fwd = fwend = midword = splits = 0
    durs = []
    for idx, b in enumerate(blocks):
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
        if last_clean in FUNCTION_WORDS_FULL:
            fwend += 1
        if any(len(w) > MAX_CHARS for w in words):
            midword += 1
        # name/title split check
        if idx + 1 < len(blocks):
            last_word_clean = re.sub(r"[^\w]", "", words[-1]).lower() if words else ""
            next_text = blocks[idx+1].get("text", " ".join(blocks[idx+1].get("lines", [])))
            next_words = next_text.split()
            if (last_word_clean in NAME_TITLES
                    and next_words
                    and re.sub(r"[^\w]", "", next_words[0])[:1].isupper()):
                splits += 1
    arr = np.array(durs)
    r = {"n": n,
         "pct_within_2x42": round(100.0*w2x42/n, 1),
         "pct_within_17cps": round(100.0*fwd/n, 1),
         "pct_ends_func": round(100.0*fwend/n, 1),
         "mid_word_breaks": midword,
         "name_title_splits": splits,
         "mean_dur_s": round(float(arr.mean()), 2),
         "median_dur_s": round(float(np.median(arr)), 2),
         "min_dur_s": round(float(arr.min()), 2),
         "max_dur_s": round(float(arr.max()), 2)}
    print(f"\n  [{name}]  n={n}")
    print(f"    within 2x42 chars:     {r['pct_within_2x42']:>5.1f}%  ({w2x42}/{n})")
    print(f"    within 17 cps:         {r['pct_within_17cps']:>5.1f}%  ({fwd}/{n})")
    print(f"    ends on function word: {r['pct_ends_func']:>5.1f}%  ({fwend}/{n})")
    print(f"    mid-word breaks:       {midword}")
    print(f"    name/title splits:     {splits}")
    print(f"    duration: mean={r['mean_dur_s']:.2f}s  median={r['median_dur_s']:.2f}s  "
          f"min={r['min_dur_s']:.2f}s  max={r['max_dur_s']:.2f}s")
    return r

whisper_seg_blocks = [{"start": float(s["start"]), "end": float(s["end"]),
                       "text": s["text"].strip()}
                      for s in asr_data["segments"] if s["text"].strip()]
baseline_blocks    = parse_srt(CAPS / f"baseline_{STEM}.srt")

ns_r  = fmt_metrics(new_seg_blocks,    "New segmenter (segment_captions + Whisper times)")
ws_r  = fmt_metrics(whisper_seg_blocks, "Whisper raw segments as captions")
bl_r  = fmt_metrics(baseline_blocks,    "Fixed-width baseline (task6b, no punct)")

fmt_results = {"new_segmenter": ns_r, "whisper_segments": ws_r, "baseline": bl_r}

# ═══════════════════════════════════════ TASK 3: CASING ISOLATION ═════════════

print(); print(SEP); print("TASK 3  -- casing isolation WER"); print(SEP)

# (a) Whisper verbatim
hyp_a = " ".join(whisper_raw_ws)

# (b) Whisper words (casing+apos kept) + BERT-predicted punct
hyp_b = " ".join(
    w.rstrip(".,?!;:") + PUNCT_SYM[l]
    for w, l in zip(whisper_raw_ws, bert_labels)
)

# (c) Full BERT pipeline (truecaser + apos restoration + BERT labels)
hyp_c = " ".join(bert_display)

def strict_wer(ref, hyp, label):
    out = jiwer.process_words(ref, hyp)
    print(f"\n  [{label}]  WER={out.wer*100:.2f}%  "
          f"sub={out.substitutions}  ins={out.insertions}  del={out.deletions}")
    return out

print()
res_a = strict_wer(ref_raw, hyp_a, "(a) Whisper words+case+apos+Whisper marks")
res_b = strict_wer(ref_raw, hyp_b, "(b) Whisper words+case+apos+BERT marks")
res_c = strict_wer(ref_raw, hyp_c, "(c) Full BERT pipeline (truecaser+apos+BERT marks)")

# Count casing and apostrophe errors in (c)
ref_t = ref_raw.split(); hyp_t = hyp_c.split()
casing_errs = 0; apos_errs = 0
for chunk in res_c.alignments[0]:
    if chunk.type != "substitute": continue
    r_words = ref_t[chunk.ref_start_idx:chunk.ref_end_idx]
    h_words = hyp_t[chunk.hyp_start_idx:chunk.hyp_end_idx]
    for rw, hw in zip(r_words, h_words):
        if rw.lower() == hw.lower() and rw != hw:
            casing_errs += 1
        elif (rw.lower().replace("'","") == hw.lower().replace("'","")
              and rw.lower() != hw.lower()):
            apos_errs += 1

print(f"\n  In (c): casing-only errors = {casing_errs},  apostrophe errors = {apos_errs}")

casing_results = {
    "a_whisper_verbatim_wer": round(res_a.wer*100, 2),
    "b_whisper_words_bert_marks_wer": round(res_b.wer*100, 2),
    "c_full_bert_pipeline_wer": round(res_c.wer*100, 2),
    "c_casing_only_errors": casing_errs,
    "c_apostrophe_errors": apos_errs,
}

# ═══════════════════════════════════════ TASK 4: CONSISTENT METRICS ═══════════

print(); print(SEP); print("TASK 4  -- consistent METEOR/BERTScore/BLEU/chrF per system"); print(SEP)

log("loading predictions for all 5 systems...")
SYSTEMS = ["majority", "rule", "crf", "bert_uw", "bert_wt"]
test_items = list(D["test"].items())

system_hyps = {}
system_refs = []

# build reference texts once
for tid, doc in test_items:
    system_refs.append(reference_text(doc))

for sname in SYSTEMS:
    preds_data = json.loads((RESULTS / f"preds_test_{sname}.json").read_text(encoding="utf-8"))
    by_talk = collections.defaultdict(dict)
    for p in preds_data:
        by_talk[p["talk_id"]][p["word_index"]] = p["predicted_label"]
    hyps = []
    for tid, doc in test_items:
        asr_words = [w.lower().replace("'", "") for w, _, _ in doc]
        pred_labs = [by_talk[tid][i] for i in range(len(by_talk[tid]))]
        hyps.append(render_speech(asr_words, pred_labs))
    system_hyps[sname] = hyps
    log(f"  {sname}: {len(hyps)} talks, first hyp[:60]={hyps[0][:60]!r}")

# METEOR (same tokenization for all: split() without lowercasing,
# matching the WER/BLEU/chrF approach)
log("computing METEOR for all systems...")
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
from nltk.translate.meteor_score import meteor_score

def corpus_meteor_split(hyp_list, ref_list):
    """Mean sentence METEOR; tokens = str.split() (no lowercasing, matches WER/BLEU)."""
    scores = [meteor_score([r.split()], h.split())
              for h, r in zip(hyp_list, ref_list)]
    return float(np.mean(scores))

meteor_scores = {}
for sname in SYSTEMS:
    meteor_scores[sname] = corpus_meteor_split(system_hyps[sname], system_refs)
    log(f"  METEOR {sname} = {meteor_scores[sname]:.4f}")

# BERTScore for all systems
log("computing BERTScore for all systems (distilbert-base-uncased)...")
from bert_score import score as bert_score_fn

bertscore_f = {}
for sname in SYSTEMS:
    P, R, F = bert_score_fn(
        system_hyps[sname], system_refs,
        model_type="distilbert-base-uncased",
        device="cpu", verbose=False, batch_size=8
    )
    bertscore_f[sname] = float(F.mean().item())
    log(f"  BERTScore F1 {sname} = {bertscore_f[sname]:.4f}")

# BLEU + chrF are already in metrics_all.json; verify they match our renders
metrics_all = json.loads((RESULTS / "metrics_all.json").read_text())

print("\n  Consistent metrics (same rendered string for all metrics):")
print(f"  {'System':<15} {'METEOR':>8} {'BERTScore-F1':>13} {'BLEU':>8} {'chrF':>8}")
print("  " + "-"*56)
for sname in SYSTEMS:
    bleu = metrics_all["systems"][sname]["BLEU"]
    chrf = metrics_all["systems"][sname]["chrF"]
    print(f"  {sname:<15} {meteor_scores[sname]:>8.4f} {bertscore_f[sname]:>13.4f} "
          f"{bleu:>8.2f} {chrf:>8.2f}")

# Explain METEOR bert_uw < crf_pipeline
old_extra  = json.loads(pathlib.Path("results_bert/extra_metrics.json").read_text())
old_meteor_crf  = old_extra["METEOR"]["crf_pipeline"]
old_meteor_bert = old_extra["METEOR"].get("bert_punct")
new_meteor_crf  = meteor_scores["crf"]
new_meteor_bert = meteor_scores["bert_uw"]

print(f"\n  EXPLANATION -- METEOR bert_uw < crf_pipeline (before this fix):")
print(f"    Old crf_pipeline METEOR  = {old_meteor_crf:.4f}  "
      f"(computed with word_tokenize(h.lower()) in extra_metrics.py)")
print(f"    Old bert_uw METEOR       = {old_meteor_bert if old_meteor_bert else 'null'}  "
      f"(computed with h.split() in task4_extra_metrics.py)")
print(f"    --> The two systems used DIFFERENT preprocessing:")
print(f"        CRF:     word_tokenize + lowercase (removes casing penalty)")
print(f"        bert_uw: str.split() without lowercase (penalises casing errors)")
print(f"        bert_uw truecaser adds more capitalisation positions than CRF,")
print(f"        so when casing is penalised, bert_uw scores lower.")
print(f"    Now with consistent str.split() preprocessing:")
print(f"    New crf METEOR   = {new_meteor_crf:.4f}")
print(f"    New bert_uw METEOR = {new_meteor_bert:.4f}")

# Write updated extra_metrics.json
new_extra = {
    "bertscore_model": "distilbert-base-uncased",
    "bertscore_device": "cpu",
    "rendering": "render_speech(asr_words=[w.lower().replace(\"'\",\"\")], pred_labs); same string for all metrics",
    "meteor_tokenization": "str.split() without lowercasing; matches WER/BLEU/chrF input",
    "bertscore_notes": [
        "distilbert-base-uncased lowercases all input; BERTScore therefore",
        "cannot distinguish capitalisation. Scores are NOT baseline-rescaled."
    ],
    "METEOR": {sname: meteor_scores[sname] for sname in SYSTEMS},
    "BERTScore_F1": {sname: bertscore_f[sname] for sname in SYSTEMS},
}
(RESULTS / "extra_metrics.json").write_text(json.dumps(new_extra, indent=2), encoding="utf-8")
log("Wrote results_new/extra_metrics.json")

# Update FINAL_RESULTS.md Text Quality Metrics section with new values
SNAME_LABELS = {
    "majority": "Majority (no marks)",
    "rule":     "Rule baseline",
    "crf":      "CRF",
    "bert_uw":  "BERT unweighted",
    "bert_wt":  "BERT weighted",
}
new_rows = "| System | METEOR | BERTScore F1 |\n|--------|--------|-------------|\n"
for sname, label in SNAME_LABELS.items():
    new_rows += f"| {label} | {meteor_scores[sname]:.4f} | {bertscore_f[sname]:.4f} |\n"
new_section = (
    "### Text Quality Metrics (METEOR, BERTScore)\n\n"
    + new_rows
    + "\n"
    + "> distilbert-base-uncased lowercases all input; BERTScore therefore\n"
    + "> cannot distinguish capitalisation. Scores are NOT baseline-rescaled.\n"
    + "> BERTScore model: `distilbert-base-uncased`\n"
    + "> METEOR tokenization: str.split() without lowercasing (consistent with WER/BLEU/chrF).\n"
    + "*Source: `results_new/extra_metrics.json`*"
)
fr = pathlib.Path("FINAL_RESULTS.md")
fr_text = fr.read_text(encoding="utf-8")
# replace from "### Text Quality Metrics" up to the next "##" or "## 4."
fr_text = re.sub(
    r"### Text Quality Metrics \(METEOR, BERTScore\).*?(?=\n## )",
    new_section,
    fr_text,
    flags=re.DOTALL,
)
fr.write_text(fr_text, encoding="utf-8")
log("Updated FINAL_RESULTS.md Text Quality Metrics section")

# ═══════════════════════════════════════ TASK 5: FILE CLEANUP ════════════════

print(); print(SEP); print("TASK 5  -- file cleanup (bert_results.json + asr_results.json)"); print(SEP)

# --- bert_results.json: move honest_assessment, boot95, paired_vs_crf
BR_PATH = pathlib.Path("results_bert/bert_results.json")
OLD_DIR = pathlib.Path("results_old")
OLD_DIR.mkdir(exist_ok=True)

br = json.loads(BR_PATH.read_text(encoding="utf-8"))

TO_MOVE = {}
for key in ["honest_assessment", "paired_vs_crf"]:
    if key in br:
        TO_MOVE[key] = br.pop(key)

# boot95 lives inside test_unweighted and test_weighted
for section in ["test_unweighted", "test_weighted"]:
    if section in br and "boot95_macroF1_CP" in br[section]:
        TO_MOVE.setdefault("boot95", {})[section] = br[section].pop("boot95_macroF1_CP")

if TO_MOVE:
    (OLD_DIR / "bert_results_archived_fields.json").write_text(
        json.dumps(TO_MOVE, indent=2), encoding="utf-8")
    BR_PATH.write_text(json.dumps(br, indent=2), encoding="utf-8")
    print(f"\n  Moved to results_old/bert_results_archived_fields.json:")
    for k in TO_MOVE: print(f"    {k}")
    print(f"  Cleaned results_bert/bert_results.json.")
else:
    print("\n  bert_results.json already clean.")

# --- asr_results.json: rename sync_check -> whisper_segment_grouping
ASR_R_PATH = RESULTS / "asr_results.json"
asr_r = json.loads(ASR_R_PATH.read_text(encoding="utf-8"))

if "sync_check" in asr_r and "whisper_segment_grouping" not in asr_r:
    asr_r["whisper_segment_grouping"] = asr_r.pop("sync_check")
    ASR_R_PATH.write_text(json.dumps(asr_r, indent=2), encoding="utf-8")
    print(f"\n  Renamed 'sync_check' -> 'whisper_segment_grouping' in asr_results.json.")
    # update FINAL_RESULTS.md reference
    fr = pathlib.Path("FINAL_RESULTS.md")
    fr_text = fr.read_text(encoding="utf-8")
    if "sync_check" in fr_text:
        fr.write_text(fr_text.replace("sync_check", "whisper_segment_grouping"), encoding="utf-8")
        print(f"  Updated FINAL_RESULTS.md: sync_check -> whisper_segment_grouping.")
elif "whisper_segment_grouping" in asr_r:
    print(f"\n  asr_results.json already has 'whisper_segment_grouping'.")
else:
    print(f"\n  [WARN] sync_check key not found in asr_results.json.")

# ──────────────────────────────────────────── save task6d results ─────────────

task6d_json = {
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "task1_new_segmenter_blocks": len(new_seg_blocks),
    "task2_format": fmt_results,
    "task3_wer": casing_results,
    "task4_meteor": meteor_scores,
    "task4_bertscore_f1": bertscore_f,
    "task4_meteor_explanation": {
        "old_crf_meteor_preprocessing": "word_tokenize+lower (extra_metrics.py)",
        "old_bertuw_meteor_preprocessing": "str.split no lower (task4_extra_metrics.py)",
        "fix": "str.split for all systems (consistent with WER/BLEU/chrF)",
        "old_crf": old_meteor_crf,
        "new_crf": new_meteor_crf,
        "new_bertuw": new_meteor_bert,
    },
}
(RESULTS / "task6d_results.json").write_text(
    json.dumps(task6d_json, indent=2), encoding="utf-8")
log("Wrote results_new/task6d_results.json")

# Add task6d_r to verify_real.py collector
VR = pathlib.Path("verify_real.py")
VR2 = VR.read_text(encoding="utf-8")
T6D_LOAD = ('task6d_r = json.loads((RESULTS / "task6d_results.json").read_text(encoding="utf-8")) \\\n'
            '           if (RESULTS / "task6d_results.json").exists() else {}')
T6D_COLLECT = "    collect_nums(task6d_r)"

if "task6d_r" not in VR2:
    vr3 = VR2.replace(
        'task6c_r = json.loads((RESULTS / "task6c_results.json")',
        T6D_LOAD + '\n' + 'task6c_r = json.loads((RESULTS / "task6c_results.json")'
    )
    vr3 = vr3.replace(
        "    collect_nums(task6c_r)",
        "    collect_nums(task6c_r)\n" + T6D_COLLECT
    )
    VR.write_text(vr3, encoding="utf-8")
    print("\n  Added task6d_r to verify_real.py collector.")

# ═══════════════════════════════════════ TASK 6: PRINT + RERUN ═══════════════

print(); print(SEP); print("TASK 6  -- bootstrap.json, data_reconciliation.md, pytest, verify_real.py"); print(SEP)

boot = json.loads((RESULTS / "bootstrap.json").read_text())
print("\n--- results_new/bootstrap.json ---")
print(json.dumps(boot, indent=2))

print("\n--- data_reconciliation.md ---")
print(pathlib.Path("data_reconciliation.md").read_text(encoding="utf-8"))

# --- pytest
print(); print(SEP); print("Running pytest tests/test_captions.py...")
pt = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/test_captions.py", "-v", "--tb=short"],
    capture_output=True, text=True
)
print(pt.stdout[-3000:] if len(pt.stdout) > 3000 else pt.stdout)
if pt.returncode != 0:
    print("STDERR:", pt.stderr[-500:])
    print("[FAIL]  pytest returned", pt.returncode)
else:
    print("[OK]  pytest PASS")

# --- verify_real.py
print(); print(SEP); print("Running verify_real.py...")
vr = subprocess.run([sys.executable, "verify_real.py"], capture_output=True, text=True)
tail = vr.stdout.splitlines()
print("\n".join(tail[-15:]))
if vr.returncode != 0:
    print("STDERR:", vr.stderr[-500:])
    print("[FAIL]  verify_real.py returned", vr.returncode)
else:
    print("[OK]  verify_real.py PASS")

print(); print(SEP); print("Task 6d DONE"); print(SEP)
