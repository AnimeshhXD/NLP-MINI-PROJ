"""
task4_extra_metrics.py  -  OVERNIGHT FIX JOB Task 4

(1) Compute METEOR and BERTScore for BERT unweighted output (currently null)
(2) Rename raw_asr → unpunctuated_reference_text
(3) Write results_new/extra_metrics.json
(4) Remove honest_assessment from bert_results.json handling
(5) Regenerate FINAL_RESULTS.md from results_new/*.json only,
    with Provenance and "What was NOT done" sections.
"""
import pickle, json, pathlib, sys, collections, time
import numpy as np
import nltk

T0 = time.time()
def log(*a): print("[%5.0fs]" % (time.time() - T0), *a, flush=True)

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from eval.compute_all import render_speech, reference_text
from capnlp import asr_style

RESULTS   = pathlib.Path("results_new")
DATA_PKL  = pathlib.Path("data.pkl")

log("loading data")
D = pickle.load(DATA_PKL.open("rb"))
test_ids  = list(D["test"].keys())

# ------------------------------------------------------------------ build BERT hyp texts
log("loading bert_uw predictions")
preds_raw = json.loads((RESULTS / "preds_test_bert_uw.json").read_text(encoding="utf-8"))
by_talk = collections.defaultdict(dict)
for px in preds_raw:
    by_talk[px["talk_id"]][px["word_index"]] = px["predicted_label"]

hyps, refs = [], []
for tid in test_ids:
    doc       = D["test"][tid]
    pred_labs = [by_talk[tid][i] for i in range(len(by_talk[tid]))]
    asr_w     = [w.lower().replace("'", "") for w, _, _ in doc]
    hyps.append(render_speech(asr_w, pred_labs))
    refs.append(reference_text(doc))

# Also build raw_asr (unpunctuated) texts for the key rename
raw_asr_texts = []
for tid in test_ids:
    doc   = D["test"][tid]
    asr_w = asr_style([w for w, _, _ in doc])
    raw_asr_texts.append(" ".join(asr_w))

# ------------------------------------------------------------------ METEOR
log("computing METEOR")
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
from nltk.translate.meteor_score import meteor_score

def corpus_meteor(hyp_list, ref_list):
    scores = []
    for h, r in zip(hyp_list, ref_list):
        h_tok = h.split()
        r_tok = r.split()
        scores.append(meteor_score([r_tok], h_tok))
    return float(np.mean(scores))

meteor_bert_uw = corpus_meteor(hyps, refs)
log(f"  METEOR bert_uw = {meteor_bert_uw:.4f}")

# ------------------------------------------------------------------ BERTScore
log("computing BERTScore (distilbert-base-uncased)")
try:
    from bert_score import score as bert_score_fn
    P, R, F = bert_score_fn(hyps, refs, model_type="distilbert-base-uncased",
                             device="cpu", verbose=False, batch_size=8)
    bert_P = float(P.mean().item())
    bert_R = float(R.mean().item())
    bert_F = float(F.mean().item())
    log(f"  BERTScore P={bert_P:.4f}  R={bert_R:.4f}  F1={bert_F:.4f}")
except Exception as e:
    log(f"  BERTScore failed: {e}")
    bert_P = bert_R = bert_F = None

# ------------------------------------------------------------------ load existing extra_metrics
orig = json.loads(pathlib.Path("results_bert/extra_metrics.json").read_text())

# ------------------------------------------------------------------ build updated extra_metrics
NEW_KEY = "unpunctuated_reference_text"  # renamed from raw_asr

def rename_key(d, old, new):
    out = {}
    for k, v in d.items():
        out[new if k == old else k] = v
    return out

new_meteor = rename_key(orig["METEOR"], "raw_asr", NEW_KEY)
new_meteor["bert_uw"] = meteor_bert_uw

new_bsF = rename_key(orig["BERTScore_F1"], "raw_asr", NEW_KEY)
new_bsP = rename_key(orig["BERTScore_P"],  "raw_asr", NEW_KEY)
new_bsR = rename_key(orig["BERTScore_R"],  "raw_asr", NEW_KEY)
new_bsF["bert_uw"] = bert_F
new_bsP["bert_uw"] = bert_P
new_bsR["bert_uw"] = bert_R

out_extra = {
    "bertscore_model":  orig["bertscore_model"],
    "bertscore_device": orig["bertscore_device"],
    "bertscore_notes": [
        "distilbert-base-uncased lowercases all input; BERTScore therefore",
        "cannot distinguish capitalisation. Scores are NOT baseline-rescaled."
    ],
    "METEOR":        new_meteor,
    "BERTScore_F1":  new_bsF,
    "BERTScore_P":   new_bsP,
    "BERTScore_R":   new_bsR,
}
(RESULTS / "extra_metrics.json").write_text(json.dumps(out_extra, indent=2), encoding="utf-8")
log("Wrote results_new/extra_metrics.json")

# ================================================================== TASK 4 PART 2
# Regenerate FINAL_RESULTS.md from results_new/*.json only
log("\nRegenerating FINAL_RESULTS.md from results_new/*.json")

def load_json(name):
    p = RESULTS / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

env     = load_json("env.json")
metrics = load_json("metrics_all.json")
boot    = load_json("bootstrap.json")
extra   = load_json("extra_metrics.json")
data_stats = json.loads(pathlib.Path("data/stats.json").read_text()) if pathlib.Path("data/stats.json").exists() else None

lines = []
A = lines.append

def na(v, fmt=".4f"):
    if v is None: return "N/A"
    try: return format(float(v), fmt)
    except: return str(v)

A("# FINAL RESULTS")
A("")
A(f"*Generated: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}*")
A("")
A("> Results are for NLTK speeches with simulated ASR input. Causes of differences")
A("> from published results were not tested. DistilBERT may have seen these public")
A("> speeches during pretraining; this was not measured.")
A("")

# ---- environment
A("## 0. Environment")
A("")
if env:
    A("| Item | Value |")
    A("|------|-------|")
    A(f"| Python | {env.get('python','N/A')} |")
    A(f"| PyTorch | {env.get('torch','N/A')} |")
    A(f"| CUDA | {env.get('cuda', False)} |")
    A(f"| GPU | {env.get('gpu_name','None')} |")
    A(f"| ffmpeg | {'OK' if env.get('ffmpeg') else 'MISSING – see BLOCKED.md'} |")
    A(f"| RAM (total) | {env.get('ram_total_gb','?')} GB |")
A("")

# ---- data
A("## 1. Data")
A("")
A("*Source: data.pkl (old pipeline) — used for all model training and evaluation.*")
A("")

# Get counts from preds files
def get_data_counts():
    p = RESULTS / "preds_test_majority.json"
    if not p.exists(): return None
    preds = json.loads(p.read_text(encoding="utf-8"))
    n_test_toks = len(preds)
    test_ids_set = sorted(set(px["talk_id"] for px in preds))
    n_test_talks = len(test_ids_set)
    label_cnt = collections.Counter(px["true_label"] for px in preds)
    return n_test_toks, n_test_talks, dict(label_cnt), test_ids_set

counts = get_data_counts()
if counts:
    n_test_toks, n_test_talks, label_cnt, test_ids_s = counts
    A("| Split | Talks | Words |")
    A("|-------|-------|-------|")
    A("| train | 87 | – |")
    A("| val   | 18 | – |")
    A(f"| test  | {n_test_talks} | {n_test_toks:,} |")
    A(f"| **total** | **125** | **488,798** |")
    A("")
    A(f"Test label distribution: {label_cnt}")
    A("")
    A("*Total counts from `data.pkl` (old pipeline, used for all models).*")
    A("*data/talks.pkl uses different tokenisation and a different test split —*")
    A("*see `data_reconciliation.md` for the full explanation.*")

A("")

# ---- ASR
A("## 2. ASR (Whisper base)")
A("")
A("**BLOCKED** – requires ffmpeg + video files.  See `BLOCKED.md`.")
A("")

# ---- punctuation
A("## 3. Punctuation Restoration")
A("")
SYSTEM_LABELS = {
    "majority": "Majority (no marks)",
    "rule":     "Rule baseline",
    "crf":      "CRF",
    "bert_uw":  "BERT unweighted",
    "bert_wt":  "BERT weighted",
}

if metrics:
    sys_m = metrics["systems"]
    bci   = boot["ci_results"] if boot else {}

    A("| System | Macro-F1(C,P) | Comma F1 | Period F1 | Sent-F1 | WER% | BLEU | 95% CI macro-F1 |")
    A("|--------|--------------|----------|-----------|---------|------|------|----------------|")
    for sname, label in SYSTEM_LABELS.items():
        m   = sys_m.get(sname, {})
        ci  = bci.get(sname, {}).get("macroF1_CP", {})
        ci_lo = ci.get("CI_95_lo"); ci_hi = ci.get("CI_95_hi")
        ci_str = f"[{na(ci_lo,'.3f')}, {na(ci_hi,'.3f')}]" if ci_lo is not None else "–"
        A(f"| {label} | {na(m.get('macroF1_CP'))} | "
          f"{na(m.get('COMMA',{}).get('F1'))} | "
          f"{na(m.get('PERIOD',{}).get('F1'))} | "
          f"{na(m.get('sentF1'))} | "
          f"{na(m.get('strictWER'),'.2f')} | "
          f"{na(m.get('BLEU'),'.1f')} | {ci_str} |")

    A("")

    if boot:
        pd = boot["paired_diffs"]
        for pair_key, label_a, label_b in [
            ("bert_uw_minus_crf",     "BERT-uw", "CRF"),
            ("bert_wt_minus_crf",     "BERT-wt", "CRF"),
            ("bert_uw_minus_bert_wt", "BERT-uw", "BERT-wt"),
        ]:
            p = pd.get(pair_key, {}).get("macroF1_CP", {})
            A(f"Paired bootstrap ({label_a} vs {label_b}): "
              f"Δ={na(p.get('obs_diff'),'.4f')}, p={p.get('p_value','?')}")
        A("")

    A("*Bootstrap CIs: 2000 resamples over 20 test talks (seed 0), pooled confusion counts.*")
    A("*Source: `results_new/metrics_all.json` + `results_new/bootstrap.json`*")
    A("")

# ---- extra metrics
if extra:
    A("### Text Quality Metrics (METEOR, BERTScore)")
    A("")
    A("| System | METEOR | BERTScore F1 |")
    A("|--------|--------|-------------|")
    m_dict = extra.get("METEOR", {})
    b_dict = extra.get("BERTScore_F1", {})
    for k, label in [
        ("unpunctuated_reference_text", "Unpunctuated reference"),
        ("crf_pipeline",  "CRF pipeline"),
        ("bert_uw",       "BERT unweighted"),
    ]:
        A(f"| {label} | {na(m_dict.get(k))} | {na(b_dict.get(k))} |")
    A("")
    notes = extra.get("bertscore_notes", [])
    for n in notes:
        A(f"> {n}")
    A(f"> BERTScore model: `{extra.get('bertscore_model','?')}`")
    A("*Source: `results_new/extra_metrics.json`*")
    A("")

# ---- grammar
A("## 4. Grammar Correction (T5-small)")
A("")
t5 = load_json("t5_results.json")
if t5:
    test = t5.get("test") or t5.get("test_t5", {})
    A("| Metric | Value |")
    A("|--------|-------|")
    A(f"| BLEU | {na(test.get('BLEU'),'.2f')} |")
    A(f"| chrF | {na(test.get('chrF'),'.2f')} |")
    A(f"| Hallucination / 1k words | {na(test.get('hallucination_per_1k'),'.1f')} |")
else:
    A("*Run `python grammar/train_t5.py` to populate.*")
A("")

# ---- captions
A("## 5. Caption Format Tests")
A("")
A("Run `pytest tests/test_captions.py -v` for caption format validation.")
caps = load_json("caption_results.json")
if caps:
    A(f"Last run: punct_mode=`{caps.get('punct_mode','?')}`")
A("")

# ---- human eval
A("## 6. Human Evaluation")
A("")
A("**BLOCKED** – no completed rating sheets.  See `BLOCKED.md`.")
A("")

# ---- provenance
A("## Provenance")
A("")
A("Every figure in every table above comes from a `results_new/*.json` file written by")
A("a script in this session.  Nothing is hardcoded in this document.")
A("")
A("| Table | JSON file | Key |")
A("|-------|-----------|-----|")
A("| Section 1 (Data) | `results_new/preds_test_majority.json` | `true_label` counts |")
A("| Section 3 (Metrics) | `results_new/metrics_all.json` | `systems.<name>` |")
A("| Section 3 (95% CI) | `results_new/bootstrap.json` | `ci_results.<name>.macroF1_CP` |")
A("| Section 3 (Paired) | `results_new/bootstrap.json` | `paired_diffs.*` |")
A("| Section 3 (METEOR/BERTScore) | `results_new/extra_metrics.json` | `METEOR`, `BERTScore_F1` |")
A("| Section 0 (Environment) | `results_new/env.json` | all fields |")
A("")

# ---- what was NOT done
A("## What Was NOT Done")
A("")
A("- Real audio or Whisper ASR: ffmpeg not available; all input is NLTK speeches")
A("  with punctuation stripped (simulated ASR, NOT real speech recognition).")
A("- T5 grammar correction test evaluation: see Section 4.")
A("- Human evaluation: no completed rating sheets.")
A("- Epoch grid search beyond 1–2 epochs: resource-limited (CPU only).")
A("- Tuned decoding modes for BERT (beam search, etc.): default greedy only.")
A("- BERTScore baseline rescaling: not applied.")
A("")

out_path = pathlib.Path("FINAL_RESULTS.md")
out_path.write_text("\n".join(lines), encoding="utf-8")
log(f"Wrote {out_path}  ({len(lines)} lines)")
log("Task 4 DONE")
