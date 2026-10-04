"""
task3_bootstrap.py  -  OVERNIGHT FIX JOB Task 3

Bootstrap confidence intervals over TALKS (not tokens).
- 20 test talks, 2000 resamples, seed 0
- For each resample: draw 20 talks with replacement
  → pool per-talk counts → compute metric from pooled counts
- Classification (macroF1_CP, sentF1): pool per-talk confusion counts
- WER: pool per-talk edit counts and ref-word counts (same spirit as pooling confusion)
- BLEU/chrF: mean of per-talk sentence scores (fast approximation; stated in output)
- Paired diffs: BERT_uw-CRF, BERT_wt-CRF, BERT_uw-BERT_wt (same resamples)
- p reported as "<0.0005" if no resample crosses zero (2000 resamples)
- Asserts every point estimate lies within its 95% CI
- Writes results_new/bootstrap.json
"""
import pickle, json, pathlib, sys, collections, time
import numpy as np
import jiwer, sacrebleu

T0 = time.time()
def log(*a): print("[%5.0fs]" % (time.time() - T0), *a, flush=True)

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from capnlp import (TrueCaser, fit_apostrophe_lexicon)
from eval.compute_all import render_speech, reference_text, strip_content

RESULTS   = pathlib.Path("results_new")
DATA_PKL  = pathlib.Path("data.pkl")
SYSTEMS   = ["majority", "rule", "crf", "bert_uw", "bert_wt"]
N_RESAMPLES = 2000
SEED      = 0
ALPHA     = 0.05   # 95% CI
METRIC_NAMES = ["macroF1_CP", "sentF1", "strictWER", "BLEU", "chrF"]

log("loading data")
D = pickle.load(DATA_PKL.open("rb"))
test_ids  = list(D["test"].keys())
test_docs = {tid: D["test"][tid] for tid in test_ids}
n_talks   = len(test_ids)
log(f"  {n_talks} test talks")

# ------------------------------------------------------------------ per-talk precomputation
log("precomputing per-talk stats for all systems")

def talk_confusion(yt, yp):
    res = {}
    for c in ("COMMA", "PERIOD", "QUESTION"):
        tp = sum(1 for a, b in zip(yt, yp) if a == c and b == c)
        fp = sum(1 for a, b in zip(yt, yp) if a != c and b == c)
        fn = sum(1 for a, b in zip(yt, yp) if a == c and b != c)
        res[c] = (tp, fp, fn)
    ys = ["S" if v in ("PERIOD", "QUESTION") else "N" for v in yt]
    ps = ["S" if v in ("PERIOD", "QUESTION") else "N" for v in yp]
    tp = sum(1 for a, b in zip(ys, ps) if a == "S" and b == "S")
    fp = sum(1 for a, b in zip(ys, ps) if a != "S" and b == "S")
    fn = sum(1 for a, b in zip(ys, ps) if a == "S" and b != "S")
    res["S"] = (tp, fp, fn)
    return res

def f1(tp, fp, fn):
    d = 2*tp + fp + fn
    return 2*tp / d if d > 0 else 0.0

# per_talk_data[sname] = array of n_talks dicts with precomputed values
per_talk_data = {}

for sname in SYSTEMS:
    p = RESULTS / f"preds_test_{sname}.json"
    raw = json.loads(p.read_text(encoding="utf-8"))
    by_talk = collections.defaultdict(dict)
    for px in raw:
        by_talk[px["talk_id"]][px["word_index"]] = px["predicted_label"]

    rows = []
    for tid in test_ids:
        doc       = test_docs[tid]
        pred_labs = [by_talk[tid][i] for i in range(len(by_talk[tid]))]
        true_labs = [l for _, l, _ in doc]
        asr_w     = [w.lower().replace("'", "") for w, _, _ in doc]

        conf = talk_confusion(true_labs, pred_labs)

        hyp = render_speech(asr_w, pred_labs)
        ref = reference_text(doc)

        # WER edit counts: wer_fraction × n_ref_words ≈ edit_count
        n_ref_words = len(ref.split())
        wer_frac    = jiwer.wer([ref], [hyp])
        wer_edits   = int(round(wer_frac * n_ref_words))

        # sentence-level BLEU/chrF (fast)
        bleu_s = float(sacrebleu.sentence_bleu(hyp, [ref]).score)
        chrf_s = float(sacrebleu.sentence_chrf(hyp, [ref]).score)

        rows.append({
            "conf":       conf,
            "wer_edits":  wer_edits,
            "n_ref_words":n_ref_words,
            "bleu":       bleu_s,
            "chrf":       chrf_s,
        })
    per_talk_data[sname] = rows
    log(f"  {sname} precomputed")

# ------------------------------------------------------------------ point estimates (from full test set)
log("computing point estimates")

def pool_metrics(rows):
    """Compute metrics from a list of per-talk row dicts."""
    pool = {k: [0,0,0] for k in ("COMMA","PERIOD","QUESTION","S")}
    for r in rows:
        for k in pool:
            for j in range(3):
                pool[k][j] += r["conf"][k][j]
    c_f1 = f1(*pool["COMMA"])
    p_f1 = f1(*pool["PERIOD"])
    s_f1 = f1(*pool["S"])
    m_cp = (c_f1 + p_f1) / 2.0
    tot_edits = sum(r["wer_edits"]   for r in rows)
    tot_words = sum(r["n_ref_words"] for r in rows)
    wer  = 100 * tot_edits / max(tot_words, 1)
    bleu = float(np.mean([r["bleu"] for r in rows]))
    chrf = float(np.mean([r["chrf"] for r in rows]))
    return {"macroF1_CP": m_cp, "sentF1": s_f1,
            "strictWER": wer, "BLEU": bleu, "chrF": chrf}

point_est = {}
for sname in SYSTEMS:
    point_est[sname] = pool_metrics(per_talk_data[sname])
    m = point_est[sname]
    log(f"  {sname}: macroF1_CP={m['macroF1_CP']:.4f}  sentF1={m['sentF1']:.4f}  "
        f"WER={m['strictWER']:.2f}%  BLEU={m['BLEU']:.1f}")

# ------------------------------------------------------------------ convert to numpy arrays for speed
# For each system: arrays of shape (n_talks,)
np_data = {}
for sname in SYSTEMS:
    rows = per_talk_data[sname]
    np_data[sname] = {
        "tp_comma":  np.array([r["conf"]["COMMA"][0]  for r in rows], dtype=np.int64),
        "fp_comma":  np.array([r["conf"]["COMMA"][1]  for r in rows], dtype=np.int64),
        "fn_comma":  np.array([r["conf"]["COMMA"][2]  for r in rows], dtype=np.int64),
        "tp_period": np.array([r["conf"]["PERIOD"][0] for r in rows], dtype=np.int64),
        "fp_period": np.array([r["conf"]["PERIOD"][1] for r in rows], dtype=np.int64),
        "fn_period": np.array([r["conf"]["PERIOD"][2] for r in rows], dtype=np.int64),
        "tp_s":      np.array([r["conf"]["S"][0]      for r in rows], dtype=np.int64),
        "fp_s":      np.array([r["conf"]["S"][1]      for r in rows], dtype=np.int64),
        "fn_s":      np.array([r["conf"]["S"][2]      for r in rows], dtype=np.int64),
        "wer_edits": np.array([r["wer_edits"]         for r in rows], dtype=np.int64),
        "n_words":   np.array([r["n_ref_words"]       for r in rows], dtype=np.int64),
        "bleu":      np.array([r["bleu"]              for r in rows], dtype=np.float64),
        "chrf":      np.array([r["chrf"]              for r in rows], dtype=np.float64),
    }

def metrics_from_arrays(arr, idx):
    """Compute pooled metrics from numpy arrays and resample index."""
    def pool(key): return arr[key][idx].sum()
    def f1_pool(tp_k, fp_k, fn_k):
        tp, fp, fn = pool(tp_k), pool(fp_k), pool(fn_k)
        d = 2*tp + fp + fn
        return 2*tp / d if d > 0 else 0.0
    c_f1 = f1_pool("tp_comma",  "fp_comma",  "fn_comma")
    p_f1 = f1_pool("tp_period", "fp_period", "fn_period")
    s_f1 = f1_pool("tp_s",      "fp_s",      "fn_s")
    m_cp = (c_f1 + p_f1) / 2.0
    tot_e = pool("wer_edits"); tot_w = pool("n_words")
    wer   = 100 * tot_e / max(tot_w, 1)
    bleu  = float(arr["bleu"][idx].mean())
    chrf  = float(arr["chrf"][idx].mean())
    return np.array([m_cp, s_f1, wer, bleu, chrf])  # order matches METRIC_NAMES

METRIC_IDX = {m: i for i, m in enumerate(METRIC_NAMES)}

# ------------------------------------------------------------------ bootstrap resamples
log(f"running {N_RESAMPLES} bootstrap resamples (seed={SEED})")
rng = np.random.RandomState(SEED)
resample_idx = rng.choice(n_talks, size=(N_RESAMPLES, n_talks), replace=True)

# bs_vals[sname] shape (N_RESAMPLES, len(METRIC_NAMES))
bs_vals = {s: np.zeros((N_RESAMPLES, len(METRIC_NAMES))) for s in SYSTEMS}

for r in range(N_RESAMPLES):
    if r % 500 == 0:
        log(f"  resample {r}/{N_RESAMPLES}  elapsed={time.time()-T0:.0f}s")
    idx = resample_idx[r]
    for sname in SYSTEMS:
        bs_vals[sname][r] = metrics_from_arrays(np_data[sname], idx)

log("bootstrap done")

# ------------------------------------------------------------------ CIs + assertions
log("computing CIs")
ci_results = {}
for sname in SYSTEMS:
    ci_results[sname] = {}
    for mn in METRIC_NAMES:
        mi   = METRIC_IDX[mn]
        arr  = bs_vals[sname][:, mi]
        lo   = float(np.percentile(arr, 100 * ALPHA / 2))
        hi   = float(np.percentile(arr, 100 * (1 - ALPHA / 2)))
        pe   = point_est[sname][mn]
        inside = lo <= pe <= hi
        ci_results[sname][mn] = {
            "point_estimate":  float(pe),
            "bootstrap_mean":  float(np.mean(arr)),
            "CI_95_lo":        lo,
            "CI_95_hi":        hi,
            "point_in_CI":     bool(inside),
        }
        if not inside:
            log(f"  WARNING: {sname}/{mn} PE={pe:.4f} outside CI [{lo:.4f},{hi:.4f}]")

# ------------------------------------------------------------------ paired diffs
PAIRS = [("bert_uw","crf"), ("bert_wt","crf"), ("bert_uw","bert_wt")]
paired = {}
for sa, sb in PAIRS:
    key = f"{sa}_minus_{sb}"
    paired[key] = {}
    for mn in METRIC_NAMES:
        mi = METRIC_IDX[mn]
        diff_arr = bs_vals[sa][:, mi] - bs_vals[sb][:, mi]
        obs = point_est[sa][mn] - point_est[sb][mn]
        p_raw = float(np.mean(diff_arr <= 0)) if obs >= 0 else float(np.mean(diff_arr >= 0))
        p_str = "<0.0005" if p_raw < 0.0005 else f"{p_raw:.4f}"
        lo = float(np.percentile(diff_arr, 100 * ALPHA / 2))
        hi = float(np.percentile(diff_arr, 100 * (1 - ALPHA / 2)))
        paired[key][mn] = {"obs_diff": float(obs), "CI_lo": lo, "CI_hi": hi, "p_value": p_str}
    log(f"  {key}: macroF1_CP diff={paired[key]['macroF1_CP']['obs_diff']:.4f}  "
        f"p={paired[key]['macroF1_CP']['p_value']}")

# ------------------------------------------------------------------ assert all PEs inside CIs
n_fail = sum(1 for s in SYSTEMS for mn in METRIC_NAMES
             if not ci_results[s][mn]["point_in_CI"])
if n_fail:
    log(f"FATAL: {n_fail} point estimates outside their 95% CI")
    sys.exit(1)
log("All point estimates lie within their 95% CIs [OK]")

# ------------------------------------------------------------------ write output
out = {
    "method":        "bootstrap_over_talks",
    "n_resamples":   N_RESAMPLES,
    "seed":          SEED,
    "ci_level":      0.95,
    "n_test_talks":  n_talks,
    "note_bleu_chrf":"BLEU and chrF bootstrapped as mean of per-talk sentence scores",
    "ci_results":    ci_results,
    "paired_diffs":  paired,
}
out_path = RESULTS / "bootstrap.json"
out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
log(f"Wrote {out_path}")
log("Task 3 DONE")
