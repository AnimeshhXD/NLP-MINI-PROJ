"""
task1_predictions.py  -  OVERNIGHT FIX JOB Task 1

(a) Print per-split label counts from data/talks.pkl; reconcile vs FINAL_RESULTS totals;
    write data_reconciliation.md
(b) Save per-token test predictions for ALL 5 systems to results_new/preds_test_<system>.json
    Format: list of {talk_id, word_index, word, true_label, predicted_label}
    Systems: majority, rule, crf, bert_uw, bert_wt
    Reuse saved BERT models. Retrain CRF (window=3, c1=0.5, c2=0.01, 200 iter).
(c) Assert all systems have identical test token sequences.

Input words for ALL systems: asr_style (lowercase, apostrophes removed) — same as BERT training.
"""
import pickle, json, pathlib, sys, hashlib, time
import numpy as np
import sklearn_crfsuite
import joblib

T0 = time.time()
def log(*a): print("[%5.0fs]" % (time.time() - T0), *a, flush=True)

# ------------------------------------------------------------------ paths
DATA_PKL   = pathlib.Path("data.pkl")
TALKS_PKL  = pathlib.Path("data/talks.pkl")
RESULTS    = pathlib.Path("results_new")
RESULTS.mkdir(exist_ok=True)
MODELS_DIR = pathlib.Path("models/bert_punct")

from capnlp import word_feats, stream_feats, rule_baseline, asr_style, chunk

LABELS = ["O", "COMMA", "PERIOD", "QUESTION"]
L2ID   = {l: i for i, l in enumerate(LABELS)}

flat = lambda L: [x for s in L for x in s]

# ================================================================== (a) reconciliation
log("(a) loading data.pkl and data/talks.pkl")
D     = pickle.load(DATA_PKL.open("rb"))
Dt    = pickle.load(TALKS_PKL.open("rb")) if TALKS_PKL.exists() else None

import collections

def count_split(split_dict, has_case=True):
    """Count words and label distribution for one split dict."""
    n_words = 0
    cnt = collections.Counter()
    for doc in split_dict.values():
        for tok in doc:
            word, label = (tok[0], tok[1]) if has_case else (tok[0], tok[1])
            n_words += 1
            cnt[label] += 1
    return n_words, dict(cnt)

log("data.pkl (old pipeline, BERT training data):")
for split in ("train", "val", "test"):
    nw, cnt = count_split(D[split], has_case=True)
    log(f"  {split:5s}: {len(D[split]):3d} talks, {nw:6d} words, {cnt}")

log("\ndata/talks.pkl (new pipeline, data/prepare.py):")
dt_counts = {}
if Dt is not None:
    log(f"  type: {type(Dt).__name__}")
    if isinstance(Dt, dict) and "train" in Dt:
        for split in ("train", "val", "test"):
            first_doc = next(iter(Dt[split].values()))
            first_tok = first_doc[0]
            has_case = (isinstance(first_tok, (list, tuple)) and len(first_tok) == 3)
            nw, cnt = count_split(Dt[split], has_case=has_case)
            dt_counts[split] = {"talks": len(Dt[split]), "words": nw, "labels": cnt}
            log(f"  {split:5s}: {len(Dt[split]):3d} talks, {nw:6d} words, {cnt}")
    elif isinstance(Dt, (list, tuple)):
        log(f"  data/talks.pkl is a {type(Dt).__name__} of length {len(Dt)}; structure varies from data.pkl")
        log(f"  (different pipeline — counts shown in data/stats.json from prepare.py)")
    else:
        log(f"  unexpected structure: {type(Dt)}")
else:
    log("  data/talks.pkl not found")

# Print test IDs for both datasets
log("\ndata.pkl test talk IDs:", sorted(D["test"].keys()))
if Dt is not None and isinstance(Dt, dict) and "test" in Dt:
    log("data/talks.pkl test talk IDs:", sorted(Dt["test"].keys()))

# ------------------------------------------------------------------ write data_reconciliation.md
data_pkl_counts = {}
for split in ("train", "val", "test"):
    nw, cnt = count_split(D[split])
    data_pkl_counts[split] = {"talks": len(D[split]), "words": nw, "labels": cnt}

rec_lines = []
rec_lines.append("# Data Reconciliation")
rec_lines.append("")
rec_lines.append("## Summary")
rec_lines.append("")
rec_lines.append("FINAL_RESULTS.md (Section 1) shows totals from **data/talks.pkl** (the new pipeline),")
rec_lines.append("but all model results (BERT, CRF) were produced from **data.pkl** (the old pipeline).")
rec_lines.append("The two files differ in tokenisation method, preprocessing, and test-split assignment,")
rec_lines.append("so their label counts and talk IDs do not match.")
rec_lines.append("")
rec_lines.append("## data.pkl (old pipeline — used by BERT and CRF)")
rec_lines.append("")
rec_lines.append("Created by `prep.py` using `capnlp.clean_raw()` + `capnlp.to_tokens()`.")
rec_lines.append("- Removes UPPER-CASE title lines")
rec_lines.append("- Converts dashes to commas")
rec_lines.append("- Removes abbreviation periods (U.S. → US)")
rec_lines.append("- IDs have no `.txt` extension (e.g. `su_2001-GWBush`)")
rec_lines.append("- Stores `(word, label, case_label)` triples")
rec_lines.append("")
rec_lines.append("| Split | Talks | Words | O | COMMA | PERIOD | QUESTION |")
rec_lines.append("|-------|-------|-------|---|-------|--------|---------|")
for split in ("train", "val", "test"):
    c = data_pkl_counts[split]
    l = c["labels"]
    rec_lines.append(f"| {split} | {c['talks']} | {c['words']:,} | {l.get('O',0):,} | {l.get('COMMA',0):,} | {l.get('PERIOD',0):,} | {l.get('QUESTION',0):,} |")
total_w = sum(v["words"] for v in data_pkl_counts.values())
total_l = collections.Counter()
for v in data_pkl_counts.values():
    total_l.update(v["labels"])
rec_lines.append(f"| **total** | **{sum(v['talks'] for v in data_pkl_counts.values())}** | **{total_w:,}** | {total_l.get('O',0):,} | {total_l.get('COMMA',0):,} | {total_l.get('PERIOD',0):,} | {total_l.get('QUESTION',0):,} |")
rec_lines.append("")
rec_lines.append(f"Test talks: `{', '.join(sorted(D['test'].keys()))}`")
rec_lines.append("")

rec_lines.append("## data/talks.pkl (new pipeline — used for FINAL_RESULTS.md Section 1)")
rec_lines.append("")
rec_lines.append("Created by `data/prepare.py` using `re.findall(r\"[A-Za-z']+|[.,?!;]\", text)`.")
rec_lines.append("- No title-line removal, no dash conversion")
rec_lines.append("- IDs include `.txt` extension (e.g. `su_2001-GWBush.txt`)")
rec_lines.append("- Stores `(word, label)` pairs (no case_label)")
rec_lines.append("- Produces a different test set (different seed-42 shuffle due to different ID set)")
if dt_counts:
    rec_lines.append("")
    rec_lines.append("| Split | Talks | Words | O | COMMA | PERIOD | QUESTION |")
    rec_lines.append("|-------|-------|-------|---|-------|--------|---------|")
    for split in ("train", "val", "test"):
        c = dt_counts[split]
        l = c["labels"]
        rec_lines.append(f"| {split} | {c['talks']} | {c['words']:,} | {l.get('O',0):,} | {l.get('COMMA',0):,} | {l.get('PERIOD',0):,} | {l.get('QUESTION',0):,} |")
else:
    # Load from data/stats.json if available
    stats_p = pathlib.Path("data/stats.json")
    if stats_p.exists():
        stats = json.loads(stats_p.read_text())
        rec_lines.append(f"- Total: {stats.get('talks_total','?')} talks, {stats.get('words_total','?'):,} words")
        rec_lines.append(f"- Label distribution: {stats.get('label_dist','N/A')}")
    else:
        rec_lines.append("- See `data/stats.json` for detailed counts (run `python data/prepare.py`)")
rec_lines.append("")

rec_lines.append("## Why the counts differ")
rec_lines.append("")
rec_lines.append("1. **Tokenisation**: `to_tokens()` uses a more conservative regex (`TOKEN_RE`) that")
rec_lines.append("   excludes standalone punctuation differently; `prepare.py` keeps more tokens.")
rec_lines.append("2. **Preprocessing**: `clean_raw()` removes all-caps title lines and converts")
rec_lines.append("   dashes to commas, changing the label distribution.")
rec_lines.append("3. **Different test sets**: Both pipelines shuffle with `seed=42` but use")
rec_lines.append("   different ID sets (with vs without `.txt` suffix), producing different splits.")
rec_lines.append("")
rec_lines.append("## Impact on overnight tasks")
rec_lines.append("")
rec_lines.append("All overnight prediction and evaluation tasks use **data.pkl** as ground truth,")
rec_lines.append("because BERT was trained on data.pkl and the spec says do not retrain BERT.")
rec_lines.append("FINAL_RESULTS.md Section 1 will be updated to reflect data.pkl counts.")

pathlib.Path("data_reconciliation.md").write_text("\n".join(rec_lines), encoding="utf-8")
log("Wrote data_reconciliation.md")

# ================================================================== (b) predictions
log("\n(b) generating test predictions from data.pkl test split")

test_ids   = sorted(D["test"].keys())
test_docs  = [(tid, D["test"][tid]) for tid in test_ids]  # keep sorted order

# Prepare asr_style tokens and true labels for test split
test_asr   = []   # list of (talk_id, [asr_words], [true_labels])
for tid, doc in test_docs:
    words  = [w for w, _, _ in doc]
    labels = [l for _, l, _ in doc]
    asr_w  = asr_style(words)
    test_asr.append((tid, asr_w, labels))

# ------------------------------------------------------------------ majority baseline
log("  majority (no-marks) baseline")
preds_majority = []
for tid, asr_w, labels in test_asr:
    for i, (w, tl) in enumerate(zip(asr_w, labels)):
        preds_majority.append({"talk_id": tid, "word_index": i, "word": w,
                                "true_label": tl, "predicted_label": "O"})

# ------------------------------------------------------------------ rule baseline
log("  rule baseline")
preds_rule = []
for tid, asr_w, labels in test_asr:
    preds = rule_baseline(asr_w)
    for i, (w, tl, pl) in enumerate(zip(asr_w, labels, preds)):
        preds_rule.append({"talk_id": tid, "word_index": i, "word": w,
                           "true_label": tl, "predicted_label": pl})

# ------------------------------------------------------------------ CRF: retrain on train split
log("  CRF: retraining (window=3, c1=0.5, c2=0.01, 200 iter) on data.pkl train split")

# Build train features
train_X, train_Y = [], []
for doc in D["train"].values():
    ws = asr_style([w for w, _, _ in doc])
    ys = [l for _, l, _ in doc]
    train_X.append(stream_feats(ws, window=3, full=True))
    train_Y.append(ys)

crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.5, c2=0.01,
                            max_iterations=200, all_possible_transitions=True)
crf.fit([c for s in train_X for c in chunk(s)],
        [c for s in train_Y for c in chunk(s)])
log("  CRF fitted")

# Save CRF model
crf_path = pathlib.Path("models/crf_task1.joblib")
crf_path.parent.mkdir(exist_ok=True)
joblib.dump(crf, crf_path)
log(f"  CRF saved to {crf_path}")

# Predict on test
log("  CRF predicting on test")
preds_crf = []
for tid, asr_w, labels in test_asr:
    feats  = stream_feats(asr_w, window=3, full=True)
    chunks = chunk(feats)
    cp = flat(crf.predict(chunks))[:len(asr_w)]
    for i, (w, tl, pl) in enumerate(zip(asr_w, labels, cp)):
        preds_crf.append({"talk_id": tid, "word_index": i, "word": w,
                          "true_label": tl, "predicted_label": pl})

# ------------------------------------------------------------------ BERT: load saved models
log("  BERT: loading models")
import torch
from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification

WINDOW_BERT = 128
OVERLAP_BERT = 16
BERT_LABELS = ["O", "COMMA", "PERIOD", "QUESTION"]
BERT_L2ID   = {l: i for i, l in enumerate(BERT_LABELS)}
BERT_ID2L   = {i: l for i, l in enumerate(BERT_LABELS)}

tok = DistilBertTokenizerFast.from_pretrained(str(MODELS_DIR / "tokenizer"))
model_uw = DistilBertForTokenClassification.from_pretrained(str(MODELS_DIR / "unweighted"))
model_w  = DistilBertForTokenClassification.from_pretrained(str(MODELS_DIR / "weighted"))
model_uw.eval(); model_w.eval()
log("  BERT models loaded")

def encode_speech(words, labels):
    enc = tok(words, is_split_into_words=True, add_special_tokens=False)
    ids = enc["input_ids"]
    wids = enc.word_ids()
    aligned_labels = []
    prev_wid = None
    for wid in wids:
        if wid is None:
            aligned_labels.append(-100)
        elif wid != prev_wid:
            aligned_labels.append(BERT_L2ID[labels[wid]])
        else:
            aligned_labels.append(-100)
        prev_wid = wid
    return ids, aligned_labels

def predict_speech_bert(model, words):
    ids, _ = encode_speech(words, ["O"] * len(words))
    enc = tok(words, is_split_into_words=True, add_special_tokens=False)
    word_ids = enc.word_ids()
    n = len(ids)
    logit_sum = torch.zeros(n, 4)
    counts    = torch.zeros(n)
    start = 0
    while start < n:
        end = min(start + WINDOW_BERT, n)
        chunk_ids = ids[start:end]
        pad = WINDOW_BERT - len(chunk_ids)
        input_t = torch.tensor([chunk_ids + [0]*pad], dtype=torch.long)
        with torch.no_grad():
            logits = model(input_ids=input_t).logits[0, :len(chunk_ids)]
        logit_sum[start:end] += logits
        counts[start:end]    += 1
        if end == n: break
        start = end - OVERLAP_BERT
    avg   = logit_sum / counts.unsqueeze(1)
    preds = avg.argmax(-1).tolist()
    word_preds = []
    prev_wid = None
    for i, wid in enumerate(word_ids):
        if wid is None: continue
        if wid != prev_wid:
            word_preds.append(BERT_ID2L[preds[i]])
        prev_wid = wid
    return word_preds

log("  BERT unweighted predicting on test")
preds_bert_uw = []
for j, (tid, asr_w, labels) in enumerate(test_asr):
    if j % 5 == 0:
        log(f"    talk {j+1}/{len(test_asr)}: {tid}")
    bp = predict_speech_bert(model_uw, asr_w)[:len(asr_w)]
    for i, (w, tl, pl) in enumerate(zip(asr_w, labels, bp)):
        preds_bert_uw.append({"talk_id": tid, "word_index": i, "word": w,
                              "true_label": tl, "predicted_label": pl})

log("  BERT weighted predicting on test")
preds_bert_wt = []
for j, (tid, asr_w, labels) in enumerate(test_asr):
    if j % 5 == 0:
        log(f"    talk {j+1}/{len(test_asr)}: {tid}")
    bp = predict_speech_bert(model_w, asr_w)[:len(asr_w)]
    for i, (w, tl, pl) in enumerate(zip(asr_w, labels, bp)):
        preds_bert_wt.append({"talk_id": tid, "word_index": i, "word": w,
                              "true_label": tl, "predicted_label": pl})

# ================================================================== (c) assert identical token sequences
log("\n(c) asserting identical test token sequences")
def token_seq(preds_list):
    return [(p["talk_id"], p["word_index"], p["word"], p["true_label"]) for p in preds_list]

seq_maj  = token_seq(preds_majority)
seq_rule = token_seq(preds_rule)
seq_crf  = token_seq(preds_crf)
seq_buw  = token_seq(preds_bert_uw)
seq_bwt  = token_seq(preds_bert_wt)

for name, seq in [("rule", seq_rule), ("crf", seq_crf), ("bert_uw", seq_buw), ("bert_wt", seq_bwt)]:
    if seq != seq_maj:
        mismatch = [(i, a, b) for i, (a, b) in enumerate(zip(seq_maj, seq)) if a != b]
        print(f"FATAL: token sequence mismatch between majority and {name}")
        print("First 5 mismatches:", mismatch[:5])
        sys.exit(1)
log("  all 5 systems have identical test token sequences [OK]")

# ================================================================== save predictions
for name, preds in [("majority", preds_majority), ("rule",     preds_rule),
                    ("crf",      preds_crf),       ("bert_uw",  preds_bert_uw),
                    ("bert_wt",  preds_bert_wt)]:
    p = RESULTS / f"preds_test_{name}.json"
    p.write_text(json.dumps(preds, ensure_ascii=False), encoding="utf-8")
    log(f"  saved {p}  ({len(preds)} tokens)")

# ================================================================== summary
log("\nTask 1 DONE")
log(f"  data_reconciliation.md written")
log(f"  CRF saved to {crf_path}")
log(f"  Prediction files: {list(RESULTS.glob('preds_test_*.json'))}")
log(f"  Test tokens: {len(preds_majority)}")
