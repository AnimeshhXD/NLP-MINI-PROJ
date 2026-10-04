"""
eval/compute_all.py  -  OVERNIGHT FIX JOB Task 2

Reads results_new/preds_test_<system>.json for each of the 5 systems.
Computes for each system:
  - per-class P/R/F1 (COMMA, PERIOD, QUESTION)
  - macro-F1(COMMA+PERIOD)  [macroF1_CP]
  - sentence-boundary F1    [PERIOD+QUESTION merged as "S"]
  - strict WER              [cased, punctuated text]
  - content WER             [case/punct/apostrophes stripped]
  - BLEU                    [sacrebleu corpus_bleu]
  - chrF                    [sacrebleu corpus_chrf]

ONE rendering function for every system:
  asr_words + predicted_labels → truecased + apos-restored + punct-attached text

Reference:  original cased words from data.pkl + true_labels attached (no restoration).
            (Same logic as bert_punct.py text_metrics)

Writes results_new/metrics_all.json with timestamp, seed, library versions, input hashes.
"""
import pickle, json, pathlib, hashlib, time, sys, collections
import numpy as np
import jiwer, sacrebleu
from sklearn.metrics import precision_recall_fscore_support

T0 = time.time()
def log(*a): print("[%5.0fs]" % (time.time() - T0), *a, flush=True)

# ------------------------------------------------------------------ imports
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from capnlp import (TrueCaser, fit_apostrophe_lexicon, attach_punct)

import torch, transformers, sklearn_crfsuite
import sklearn

RESULTS   = pathlib.Path(__file__).parent.parent / "results_new"
DATA_PKL  = pathlib.Path(__file__).parent.parent / "data.pkl"

# ------------------------------------------------------------------ load data
log("loading data.pkl")
D = pickle.load(DATA_PKL.open("rb"))

# truecaser + apostrophe lexicon fitted on train split
log("fitting TrueCaser and apostrophe lexicon on train split")
tc   = TrueCaser().fit(D["train"].values())
apos = fit_apostrophe_lexicon(D["train"].values())

# ------------------------------------------------------------------ rendering
PUNCT_SYM = {"O": "", "COMMA": ",", "PERIOD": ".", "QUESTION": "?"}

def render_speech(asr_words, labels):
    """
    Convert (asr_words, predicted_labels) to a displayed text string.
    Same logic as bert_punct.py text_metrics():
      1. apostrophe restoration
      2. truecasing (lexicon lookup)
      3. sentence-initial capitalisation
      4. always capitalise 'i'
      5. attach punctuation
    """
    out   = []
    start = True
    for w, l in zip(asr_words, labels):
        t = apos.get(w, w)                          # apostrophe restoration
        if w not in apos:
            t = tc.lex.get(w, t)                    # truecasing
        if start and t[:1].islower():
            t = t[0].upper() + t[1:]                # sentence-initial cap
        if w == "i":
            t = "I"
        out.append(t)
        start = l in ("PERIOD", "QUESTION")
    return " ".join(w + PUNCT_SYM[l] for w, l in zip(out, labels))

def reference_text(doc):
    """Original cased text from data.pkl with true labels attached (no restoration)."""
    return " ".join(w + PUNCT_SYM[l] for w, l, _ in doc)

def strip_content(s):
    """Strip case, punctuation, apostrophes for content WER."""
    return " ".join("".join(c for c in s.lower() if c.isalnum() or c in " '-").split()).replace("'", "")

# ------------------------------------------------------------------ metrics
LABELS_3 = ["COMMA", "PERIOD", "QUESTION"]

def compute_prf(yt, yp):
    """Per-class + macro F1 for 4-label classification."""
    pr, rc, f, sup = precision_recall_fscore_support(yt, yp, labels=LABELS_3, zero_division=0)
    out = {}
    for c, p, r, f1, n in zip(LABELS_3, pr, rc, f, sup):
        out[c] = {"P": float(p), "R": float(r), "F1": float(f1), "n": int(n)}
    out["macroF1_CP"] = float(np.mean(f[:2]))   # COMMA + PERIOD only
    out["macroF1"]    = float(np.mean(f))        # all 3 classes
    # sentence-boundary F1: PERIOD+QUESTION merged as "S"
    ys = ["S" if v in ("PERIOD", "QUESTION") else "N" for v in yt]
    ps = ["S" if v in ("PERIOD", "QUESTION") else "N" for v in yp]
    _, _, sf, _ = precision_recall_fscore_support(ys, ps, labels=["S"], zero_division=0)
    out["sentF1"] = float(sf[0])
    return out

def compute_text_metrics(test_items, preds_by_talk):
    """
    test_items: list of (talk_id, doc) from D["test"].items() in dict order
    preds_by_talk: dict {talk_id: [predicted_label, ...]}
    Returns strictWER, contentWER, BLEU, chrF
    """
    hyps  = []
    refs  = []
    for tid, doc in test_items:
        asr_words = [w.lower().replace("'", "") for w, _, _ in doc]
        true_labs = [l for _, l, _ in doc]
        pred_labs = preds_by_talk[tid]
        hyps.append(render_speech(asr_words, pred_labs))
        refs.append(reference_text(doc))
    strip = strip_content
    return {
        "strictWER":  float(100 * jiwer.wer(refs, hyps)),
        "contentWER": float(100 * jiwer.wer([strip(r) for r in refs],
                                             [strip(h) for h in hyps])),
        "BLEU":       float(sacrebleu.corpus_bleu(hyps, [refs]).score),
        "chrF":       float(sacrebleu.corpus_chrf(hyps, [refs]).score),
    }

if __name__ == "__main__":
    # ------------------------------------------------------------------ build test_items in dict order
    test_items = list(D["test"].items())   # preserves insertion order
    SYSTEMS = ["majority", "rule", "crf", "bert_uw", "bert_wt"]

    log("computing metrics for all systems")
    all_metrics = {}

    for sname in SYSTEMS:
        log(f"  {sname}")
        preds_path = RESULTS / f"preds_test_{sname}.json"
        if not preds_path.exists():
            log(f"    ERROR: {preds_path} not found")
            sys.exit(1)

        preds_data = json.loads(preds_path.read_text(encoding="utf-8"))

        by_talk = collections.defaultdict(dict)
        for p in preds_data:
            by_talk[p["talk_id"]][p["word_index"]] = p["predicted_label"]

        preds_by_talk = {}
        for tid, idx_map in by_talk.items():
            preds_by_talk[tid] = [idx_map[i] for i in range(len(idx_map))]

        yt = [p["true_label"]      for p in preds_data]
        yp = [p["predicted_label"] for p in preds_data]

        prf  = compute_prf(yt, yp)
        txm  = compute_text_metrics(test_items, preds_by_talk)
        all_metrics[sname] = {**prf, **txm}
        log(f"    macroF1_CP={prf['macroF1_CP']:.4f}  sentF1={prf['sentF1']:.4f}  "
            f"WER={txm['strictWER']:.2f}%  BLEU={txm['BLEU']:.1f}")

    def md5(path):
        return hashlib.md5(pathlib.Path(path).read_bytes()).hexdigest()

    input_hashes = {}
    for sname in SYSTEMS:
        p = RESULTS / f"preds_test_{sname}.json"
        input_hashes[f"preds_test_{sname}.json"] = md5(p)
    input_hashes["data.pkl"] = md5(DATA_PKL)

    out = {
        "timestamp":      time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed":           42,
        "library_versions": {
            "jiwer":     getattr(jiwer, "__version__", None) or __import__("importlib.metadata", fromlist=["version"]).version("jiwer"),
            "sacrebleu": sacrebleu.__version__,
            "sklearn":   sklearn.__version__,
            "torch":     torch.__version__,
            "transformers": transformers.__version__,
        },
        "input_file_hashes": input_hashes,
        "systems": all_metrics,
    }

    out_path = RESULTS / "metrics_all.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    log("Task 2 DONE")
