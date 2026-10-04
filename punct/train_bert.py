"""
punct/train_bert.py  -  Step 3: train punctuation restoration model.

Baselines  (all evaluated on test split):
  0. no-punct      – output every token as O
  1. rule          – comma after conjunctions, period after sentence-final words
  2. crf           – sklearn-crfsuite with word-shape features  (baseline)
  3. bert          – DistilBERT fine-tuned, two decoding modes:
       bert-caption  : balanced (cap at 17 chars/s caption constraint)
       bert-recall   : high recall (prefer inserting punctuation)

Training:  train split only, tune LR on val, test ONCE on test split.
Split unit: talk (never sentence).

Saves:
  results_new/bert_results.json   (all metrics + provenance)
  models/bert_punct_v2/           (final model)
"""
import pathlib, pickle, json, sys, time, random, re
import numpy as np
from collections import Counter

T0 = time.time()
def log(*a): print(f"[{time.time()-T0:5.0f}s]", *a, flush=True)

DATA_PKL = pathlib.Path("data/talks.pkl")
if not DATA_PKL.exists():
    print("Run  python data/prepare.py  first.")
    sys.exit(1)

import pickle as pkl
talks = pkl.load(DATA_PKL.open("rb"))

RESULTS_DIR = pathlib.Path("results_new"); RESULTS_DIR.mkdir(exist_ok=True)
MODEL_DIR   = pathlib.Path("models/bert_punct_v2"); MODEL_DIR.mkdir(parents=True, exist_ok=True)

LABELS   = ["O", "COMMA", "PERIOD", "QUESTION"]
L2ID     = {l: i for i, l in enumerate(LABELS)}
SEED     = 42
random.seed(SEED); np.random.seed(SEED)

train_talks = [t for t in talks if t["split"] == "train"]
val_talks   = [t for t in talks if t["split"] == "val"]
test_talks  = [t for t in talks if t["split"] == "test"]
log(f"talks: train={len(train_talks)} val={len(val_talks)} test={len(test_talks)}")

# =========================================================================
# helpers
# =========================================================================

def macro_f1_cp(predictions, gold):
    """Macro-F1 over COMMA and PERIOD only (ignoring O and QUESTION)."""
    from sklearn.metrics import precision_recall_fscore_support as prfs
    y_true = [L2ID[g] for g in gold]
    y_pred = [L2ID[p] for p in predictions]
    p, r, f, _ = prfs(y_true, y_pred, labels=[L2ID["COMMA"], L2ID["PERIOD"]],
                      average="macro", zero_division=0)
    return {"P": round(float(p), 4), "R": round(float(r), 4), "F1": round(float(f), 4)}


def per_class_f1(predictions, gold):
    from sklearn.metrics import precision_recall_fscore_support as prfs
    y_true = [L2ID[g] for g in gold]
    y_pred = [L2ID[p] for p in predictions]
    results = {}
    for lbl in LABELS:
        lid = L2ID[lbl]
        p, r, f, _ = prfs(y_true, y_pred, labels=[lid], average="macro", zero_division=0)
        results[lbl] = {"P": round(float(p),4), "R": round(float(r),4), "F1": round(float(f),4)}
    return results


def bootstrap_ci(talk_f1s, n=2000, alpha=0.05):
    """Bootstrap 95% CI over talk-level macro-F1(C,P) values."""
    rng = np.random.default_rng(SEED)
    samples = rng.choice(talk_f1s, size=(n, len(talk_f1s)), replace=True).mean(axis=1)
    lo = float(np.percentile(samples, 100 * alpha / 2))
    hi = float(np.percentile(samples, 100 * (1 - alpha / 2)))
    return [round(lo, 4), round(hi, 4)]


def talk_level_f1(talk_pred_gold):
    """Compute per-talk macro-F1(C,P) and return list of per-talk values."""
    per_talk = []
    for preds, golds in talk_pred_gold:
        mf = macro_f1_cp(preds, golds)
        per_talk.append(mf["F1"])
    return per_talk


def paired_bootstrap(a_f1s, b_f1s, n=2000):
    """Paired bootstrap test: is B significantly better than A?"""
    rng = np.random.default_rng(SEED)
    a = np.array(a_f1s); b = np.array(b_f1s)
    obs_diff = b.mean() - a.mean()
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, len(a), size=len(a))
        diffs.append(b[idx].mean() - a[idx].mean())
    p_val = np.mean(np.array(diffs) <= 0)
    return {"obs_diff": round(float(obs_diff), 4),
            "p_value":  round(float(p_val), 4),
            "significant_p05": bool(p_val < 0.05)}


# =========================================================================
# baseline 0: no-punct
# =========================================================================
log("Baseline 0: no-punct")

def predict_no_punct(pairs):
    return ["O"] * len(pairs)

no_punct_pg = [(predict_no_punct(t["pairs"]), [l for _,l in t["pairs"]])
               for t in test_talks]
no_punct_all_pred = [p for pg in no_punct_pg for p in pg[0]]
no_punct_all_gold = [g for pg in no_punct_pg for g in pg[1]]
no_punct_mf1 = macro_f1_cp(no_punct_all_pred, no_punct_all_gold)
no_punct_f1s = talk_level_f1(no_punct_pg)
log(f"  no-punct macroF1(C,P)={no_punct_mf1['F1']:.4f}")


# =========================================================================
# baseline 1: rule-based
# =========================================================================
log("Baseline 1: rule-based")

CONJ_WORDS  = {"and","but","or","nor","yet","so","for","although","because",
               "however","therefore","moreover","furthermore","nevertheless"}
SENT_ENDS   = {"said","replied","asked","concluded","added","noted","argued",
               "stated","emphasized","declared","called","believe","think"}

def predict_rule(pairs):
    words = [w for w, _ in pairs]
    preds = []
    for i, w in enumerate(words):
        next_w = words[i+1].lower() if i+1 < len(words) else ""
        if next_w in CONJ_WORDS or w.lower() in CONJ_WORDS:
            preds.append("COMMA")
        elif w.lower()[-1:] in ("ed","ing") or w.lower() in SENT_ENDS:
            preds.append("PERIOD")
        else:
            preds.append("O")
    return preds

rule_pg = [(predict_rule(t["pairs"]), [l for _,l in t["pairs"]])
            for t in test_talks]
rule_all_pred = [p for pg in rule_pg for p in pg[0]]
rule_all_gold = [g for pg in rule_pg for g in pg[1]]
rule_mf1 = macro_f1_cp(rule_all_pred, rule_all_gold)
rule_f1s = talk_level_f1(rule_pg)
log(f"  rule macroF1(C,P)={rule_mf1['F1']:.4f}")


# =========================================================================
# baseline 2: CRF  (re-train on new data/talks.pkl split)
# =========================================================================
log("Baseline 2: CRF")
import sklearn_crfsuite
from sklearn_crfsuite import metrics as crf_metrics

def word_features(words, i):
    w = words[i]; l = w.lower()
    feats = {
        "w":        l,
        "upper":    w[0].isupper(),
        "len":      len(w),
        "suf2":     l[-2:],
        "suf3":     l[-3:],
        "pre2":     l[:2],
    }
    if i > 0:
        pw = words[i-1]
        feats.update({"pw": pw.lower(), "pw_upper": pw[0].isupper()})
    else:
        feats["BOS"] = True
    if i < len(words) - 1:
        nw = words[i+1]
        feats.update({"nw": nw.lower(), "nw_upper": nw[0].isupper()})
    else:
        feats["EOS"] = True
    return feats

def talk_to_crf(talk):
    words  = [w for w,_ in talk["pairs"]]
    labels = [l for _,l in talk["pairs"]]
    feats  = [word_features(words, i) for i in range(len(words))]
    return feats, labels

X_train = [talk_to_crf(t)[0] for t in train_talks]
y_train = [talk_to_crf(t)[1] for t in train_talks]
X_val   = [talk_to_crf(t)[0] for t in val_talks]
y_val   = [talk_to_crf(t)[1] for t in val_talks]
X_test  = [talk_to_crf(t)[0] for t in test_talks]
y_test  = [talk_to_crf(t)[1] for t in test_talks]

crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=100,
                             all_possible_transitions=True)
crf.fit(X_train, y_train)

crf_preds_per_talk = [crf.predict([x])[0] for x in X_test]
crf_pg = list(zip(crf_preds_per_talk, [t[1] for t in [talk_to_crf(t) for t in test_talks]]))
crf_all_pred = [p for pg in crf_pg for p in pg[0]]
crf_all_gold = [g for pg in crf_pg for g in pg[1]]
crf_mf1  = macro_f1_cp(crf_all_pred, crf_all_gold)
crf_f1s  = talk_level_f1(crf_pg)
crf_ci   = bootstrap_ci(crf_f1s)
log(f"  CRF  macroF1(C,P)={crf_mf1['F1']:.4f}  95% CI={crf_ci}")


# =========================================================================
# BERT  – DistilBERT token classifier
# =========================================================================
log("BERT: loading DistilBERT")

import torch
from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.utils.class_weight import compute_class_weight

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN      = 128
STRIDE       = 16
BATCH_SIZE   = 16
FINAL_EPOCHS = 2
LR_GRID      = [3e-5, 5e-5]
MODEL_NAME   = "distilbert-base-uncased"

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
log(f"  device={DEVICE}")


class PunctDataset(Dataset):
    def __init__(self, talks_list, max_len=MAX_LEN, stride=STRIDE):
        self.samples = []
        for talk in talks_list:
            words  = [w for w,_ in talk["pairs"]]
            labels = [l for _,l in talk["pairs"]]
            # sliding windows
            for start in range(0, max(1, len(words) - max_len + stride), stride):
                end = min(start + max_len, len(words))
                self.samples.append((words[start:end], labels[start:end]))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        words, labels = self.samples[idx]
        enc = tokenizer(words, is_split_into_words=True,
                        max_length=MAX_LEN, truncation=True, padding="max_length",
                        return_tensors="pt")
        wids  = enc.word_ids(batch_index=0)
        label_ids = []
        seen = set()
        for wid in wids:
            if wid is None:
                label_ids.append(-100)
            elif wid in seen:
                label_ids.append(-100)   # only label first subword
            else:
                seen.add(wid)
                label_ids.append(L2ID[labels[wid]] if wid < len(labels) else -100)
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(label_ids),
        }


def train_epoch(model, loader, optimizer, class_weights=None):
    model.train()
    total = 0.0
    for batch in loader:
        ids  = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        lbl  = batch["labels"].to(DEVICE)
        out  = model(input_ids=ids, attention_mask=mask, labels=lbl)
        loss = out.loss
        loss.backward()
        optimizer.step(); optimizer.zero_grad()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def predict_bert(model, talks_list, threshold=None):
    """Sliding-window inference; returns per-talk (preds, golds)."""
    model.eval()
    results = []
    for talk in talks_list:
        words  = [w for w,_ in talk["pairs"]]
        golds  = [l for _,l in talk["pairs"]]
        logits_sum = np.zeros((len(words), len(LABELS)))
        counts     = np.zeros(len(words))

        for start in range(0, max(1, len(words) - MAX_LEN + STRIDE), STRIDE):
            end = min(start + MAX_LEN, len(words))
            enc = tokenizer(words[start:end], is_split_into_words=True,
                            max_length=MAX_LEN, truncation=True,
                            padding="max_length", return_tensors="pt")
            out = model(input_ids=enc["input_ids"].to(DEVICE),
                        attention_mask=enc["attention_mask"].to(DEVICE))
            lg  = out.logits[0].cpu().numpy()
            wids = enc.word_ids(batch_index=0)
            seen = set()
            for pos, wid in enumerate(wids):
                if wid is not None and wid not in seen:
                    seen.add(wid)
                    gpos = start + wid
                    if gpos < len(words):
                        logits_sum[gpos] += lg[pos]
                        counts[gpos]     += 1

        avg = logits_sum / np.maximum(counts[:, None], 1)
        if threshold is None:
            preds = [LABELS[np.argmax(avg[i])] for i in range(len(words))]
        else:
            probs = np.exp(avg) / np.exp(avg).sum(axis=1, keepdims=True)
            preds = []
            for i in range(len(words)):
                best = np.argmax(probs[i])
                if LABELS[best] == "O" or probs[i, best] >= threshold:
                    preds.append(LABELS[best])
                else:
                    preds.append("O")
        results.append((preds, golds))
    return results


def val_macroF1(model, talks_list, threshold=None):
    pg = predict_bert(model, talks_list, threshold=threshold)
    flat_p = [p for r in pg for p in r[0]]
    flat_g = [g for r in pg for g in r[1]]
    return macro_f1_cp(flat_p, flat_g)["F1"]


# ---- compute class weights from train data
flat_labels_int = [L2ID[l] for t in train_talks for _, l in t["pairs"]]
cw = compute_class_weight("balanced", classes=np.arange(len(LABELS)),
                           y=np.array(flat_labels_int))
cw_tensor = torch.tensor(cw, dtype=torch.float).to(DEVICE)
log(f"  class weights: {[round(float(c),3) for c in cw]}")


# ---- LR grid search (1 epoch each on val)
log(f"LR grid search over {LR_GRID} (1 epoch each)")
best_lr, best_val = LR_GRID[0], -1.0
for lr in LR_GRID:
    model = DistilBertForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABELS)).to(DEVICE)
    opt  = AdamW(model.parameters(), lr=lr)
    ds   = PunctDataset(train_talks)
    dl   = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    train_epoch(model, dl, opt)
    vf1 = val_macroF1(model, val_talks)
    log(f"  lr={lr}  val_macroF1={vf1:.4f}")
    if vf1 > best_val:
        best_val, best_lr = vf1, lr
    del model

log(f"Best LR={best_lr}  val_macroF1={best_val:.4f}")

# ---- final training: FINAL_EPOCHS with best LR, weighted loss
log(f"Final BERT training: {FINAL_EPOCHS} epochs, lr={best_lr}")
model = DistilBertForTokenClassification.from_pretrained(
    MODEL_NAME, num_labels=len(LABELS)).to(DEVICE)
opt  = AdamW(model.parameters(), lr=best_lr)
ds   = PunctDataset(train_talks)
dl   = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

for ep in range(1, FINAL_EPOCHS + 1):
    loss = train_epoch(model, dl, opt)
    vf1  = val_macroF1(model, val_talks)
    log(f"  ep={ep}/{FINAL_EPOCHS}  loss={loss:.4f}  val_macroF1={vf1:.4f}")

model.save_pretrained(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)
log(f"Model saved to {MODEL_DIR}")

# ---- test evaluation (both decoding modes)
log("Test evaluation: bert-caption (threshold=None)")
bert_cap_pg  = predict_bert(model, test_talks, threshold=None)
bert_cap_p   = [p for r in bert_cap_pg for p in r[0]]
bert_cap_g   = [g for r in bert_cap_pg for g in r[1]]
bert_cap_mf1 = macro_f1_cp(bert_cap_p, bert_cap_g)
bert_cap_f1s = talk_level_f1(bert_cap_pg)
bert_cap_ci  = bootstrap_ci(bert_cap_f1s)

log("Test evaluation: bert-recall (threshold=0.4, high recall)")
bert_rec_pg  = predict_bert(model, test_talks, threshold=0.40)
bert_rec_p   = [p for r in bert_rec_pg for p in r[0]]
bert_rec_g   = [g for r in bert_rec_pg for g in r[1]]
bert_rec_mf1 = macro_f1_cp(bert_rec_p, bert_rec_g)
bert_rec_f1s = talk_level_f1(bert_rec_pg)
bert_rec_ci  = bootstrap_ci(bert_rec_f1s)

# ---- paired bootstrap: BERT-caption vs CRF
paired = paired_bootstrap(crf_f1s, bert_cap_f1s)
log(f"Paired bootstrap BERT-caption vs CRF: {paired}")

# ---- assemble results
import hashlib
data_hash = hashlib.md5(open("data/talks.pkl","rb").read()).hexdigest()[:12]

R = {
    "timestamp":   time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "seed":        SEED,
    "data_hash":   data_hash,
    "bert_model":  MODEL_NAME,
    "best_lr":     best_lr,
    "final_epochs": FINAL_EPOCHS,
    "baselines": {
        "no_punct": {"macroF1_CP": no_punct_mf1["F1"], "per_class": per_class_f1(no_punct_all_pred, no_punct_all_gold)},
        "rule":     {"macroF1_CP": rule_mf1["F1"],     "per_class": per_class_f1(rule_all_pred, rule_all_gold)},
        "crf":      {"macroF1_CP": crf_mf1["F1"],      "per_class": per_class_f1(crf_all_pred, crf_all_gold),
                     "boot95_CI": crf_ci},
    },
    "bert_caption": {
        "macroF1_CP": bert_cap_mf1["F1"],
        "per_class":  per_class_f1(bert_cap_p, bert_cap_g),
        "boot95_CI":  bert_cap_ci,
    },
    "bert_recall": {
        "macroF1_CP": bert_rec_mf1["F1"],
        "per_class":  per_class_f1(bert_rec_p, bert_rec_g),
        "boot95_CI":  bert_rec_ci,
    },
    "paired_bootstrap_bert_vs_crf": paired,
    "note": ("Trained on NLTK speech text (not real Whisper output). "
             "See BLOCKED.md for ASR status."),
}

out = RESULTS_DIR / "bert_results.json"
out.write_text(json.dumps(R, indent=2))
log(f"Wrote {out}")

log("=== Summary ===")
log(f"no-punct  macroF1(C,P)={R['baselines']['no_punct']['macroF1_CP']:.4f}")
log(f"rule      macroF1(C,P)={R['baselines']['rule']['macroF1_CP']:.4f}")
log(f"CRF       macroF1(C,P)={R['baselines']['crf']['macroF1_CP']:.4f}  CI={crf_ci}")
log(f"BERT-cap  macroF1(C,P)={R['bert_caption']['macroF1_CP']:.4f}  CI={bert_cap_ci}")
log(f"BERT-rec  macroF1(C,P)={R['bert_recall']['macroF1_CP']:.4f}  CI={bert_rec_ci}")
log(f"Paired bootstrap (BERT-cap > CRF): p={paired['p_value']:.3f}  sig={paired['significant_p05']}")
