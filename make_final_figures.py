"""
make_final_figures.py - Task 9 figures.

Produces:
  figs/final_f1_vs_wer.pdf    -- F1 vs strict WER for all systems
  figs/bert_confusion.pdf     -- BERT confusion matrix (if bert_results.json exists)
  figs/extra_metrics.pdf      -- METEOR + BERTScore bar chart
"""
import json, pathlib
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 10, "font.family": "serif"})

OUT = pathlib.Path("figs"); OUT.mkdir(exist_ok=True)

# ------------------------------------------------------------------ load results
R = json.load(open("results/results.json"))

# ------------------------------------------------------------------ Figure 1: F1 vs WER
# Systems from the baseline table + BERT (if available)
systems = {
    "Unprocessed":        {"macroF1_CP": 0.0,   "strictWER": 18.52},
    "Stage-2 CRF":        {"macroF1_CP": R["E4_test"]["macroF1_CP"],
                           "strictWER":  R["E6_stage_clean"]["+ apostrophe restoration (full pipeline)"]["strictWER"]},
}

# POS-CRF balanced and others from baseline table
# (these are known from the spec; we do not re-run them as they're documented baselines)
baseline_extra = {
    "POS-CRF balanced": {"macroF1_CP": 0.443, "strictWER": 13.17},
    "Caption mode":     {"macroF1_CP": 0.420, "strictWER": 13.10},
    "Recall mode":      {"macroF1_CP": 0.475, "strictWER": 14.75},
}

# load BERT results if available
bert_path = pathlib.Path("results_bert/bert_results.json")
bert_systems = {}
if bert_path.exists():
    B = json.load(bert_path.open())
    tu = B.get("test_unweighted", {})
    tw = B.get("test_weighted",   {})
    if tu.get("macroF1_CP") and tu.get("strictWER"):
        bert_systems["BERT (unweighted)"] = {
            "macroF1_CP": tu["macroF1_CP"], "strictWER": tu["strictWER"]}
    if tw.get("macroF1_CP") and tw.get("strictWER"):
        bert_systems["BERT (weighted)"]   = {
            "macroF1_CP": tw["macroF1_CP"], "strictWER": tw["strictWER"]}

all_sys = {**systems, **baseline_extra, **bert_systems}

fig, ax = plt.subplots(figsize=(7, 4.5))
colors = plt.cm.tab10(np.linspace(0, 1, len(all_sys)))
for (name, vals), col in zip(all_sys.items(), colors):
    ax.scatter(vals["strictWER"], vals["macroF1_CP"], color=col, s=80, zorder=3)
    ax.annotate(name, (vals["strictWER"], vals["macroF1_CP"]),
                textcoords="offset points", xytext=(5, 3), fontsize=8)
ax.set_xlabel("Strict WER (%) ↓ lower is better")
ax.set_ylabel("Macro-F1 (comma + period) ↑ higher is better")
ax.set_title("Punctuation F1 vs Text WER across systems (test split)")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "final_f1_vs_wer.pdf")
plt.close()
print("Saved figs/final_f1_vs_wer.pdf")

# ------------------------------------------------------------------ Figure 2: BERT confusion matrix
if bert_path.exists() and "confusion" in json.load(bert_path.open()):
    B   = json.load(bert_path.open())
    cm  = np.array(B["confusion"])
    cmn = cm / cm.sum(1, keepdims=True)
    LABELS = ["O", "Comma", "Period", "Quest."]
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    ax.imshow(cmn, cmap="Blues")
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(LABELS); ax.set_yticklabels(LABELS)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("BERT confusion matrix (test)")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, "%.2f" % cmn[i, j], ha="center", va="center",
                    color="white" if cmn[i, j] > .5 else "black", fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / "bert_confusion.pdf")
    plt.close()
    print("Saved figs/bert_confusion.pdf")
else:
    print("BERT results not yet available; bert_confusion.pdf skipped.")

# ------------------------------------------------------------------ Figure 3: extra metrics
em_path = pathlib.Path("results_bert/extra_metrics.json")
if em_path.exists():
    EM = json.load(em_path.open())
    sysnames = ["Raw ASR\n(unprocessed)", "CRF\npipeline", "BERT\npunct"]
    keys     = ["raw_asr", "crf_pipeline", "bert_punct"]
    meteor   = [EM["METEOR"].get(k) for k in keys]
    bsf      = [EM["BERTScore_F1"].get(k) for k in keys]
    x = np.arange(len(sysnames)); w = 0.35
    fig, ax = plt.subplots(figsize=(6, 3.8))
    # only keep systems with at least one valid score
    valid_idx = [i for i in range(len(keys))
                 if meteor[i] is not None or bsf[i] is not None]
    sysnames_v = [sysnames[i] for i in valid_idx]
    meteor_v   = [meteor[i] if meteor[i] is not None else 0 for i in valid_idx]
    bsf_v      = [bsf[i]    if bsf[i]    is not None else 0 for i in valid_idx]
    xv = np.arange(len(sysnames_v))
    bars1 = ax.bar(xv - w/2, meteor_v, w, label="METEOR",       color="#3b6ea5")
    bars2 = ax.bar(xv + w/2, bsf_v,   w, label="BERTScore F1", color="#e8a838")
    ax.set_xticks(xv); ax.set_xticklabels(sysnames_v)
    ax.set_ylabel("Score"); ax.set_title("METEOR and BERTScore (test split)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "extra_metrics.pdf")
    plt.close()
    print("Saved figs/extra_metrics.pdf")
else:
    print("extra_metrics.json not yet available; extra_metrics.pdf skipped.")
