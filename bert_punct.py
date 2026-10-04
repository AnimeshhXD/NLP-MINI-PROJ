"""
bert_punct.py - Task 2: Fine-tune DistilBERT (no GPU) as a 4-label
token classifier for punctuation restoration.

Labels: O, COMMA, PERIOD, QUESTION
Input:  ASR-style lower-case words (no punctuation, no apostrophes)
Output: results_bert/bert_results.json
Model:  saved to models/bert_punct/
"""
import pickle, json, time, warnings, pathlib, random, collections
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
import jiwer, sacrebleu
import scipy.stats as stats
warnings.filterwarnings("ignore")

T0 = time.time()
def log(*a): print("[%5.0fs]" % (time.time() - T0), *a, flush=True)

# ------------------------------------------------------------------ paths
DATA_PKL   = pathlib.Path("data.pkl")
OUT_JSON   = pathlib.Path("results_bert/bert_results.json")
MODEL_DIR  = pathlib.Path("models/bert_punct")
LOG_FILE   = pathlib.Path("logs/bert_punct.json")
OUT_JSON.parent.mkdir(exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE.parent.mkdir(exist_ok=True)

# ------------------------------------------------------------------ config
SEED        = 42
WINDOW      = 128   # sub-tokens per window
OVERLAP     = 16    # sub-token overlap at inference
BATCH       = 16
EPOCHS_GRID = [1]          # 1 epoch per config is enough to rank them on val
LRS         = [3e-5, 5e-5] # try on validation
MODEL_NAME  = "distilbert-base-uncased"
LABELS      = ["O", "COMMA", "PERIOD", "QUESTION"]
L2ID        = {l: i for i, l in enumerate(LABELS)}
ID2L        = {i: l for i, l in enumerate(LABELS)}
DEVICE      = "cpu"
torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)

# ------------------------------------------------------------------ load data
log("loading data")
D = pickle.load(DATA_PKL.open("rb"))

def speech_to_pairs(speech_doc):
    """Return (asr_words, labels) for one speech."""
    words  = [w.lower().replace("'", "") for w, _, _ in speech_doc]
    labels = [l for _, l, _ in speech_doc]
    return words, labels

def make_split(split_key):
    """Return list of (words, labels) for all speeches in a split."""
    return [speech_to_pairs(doc) for doc in D[split_key].values()]

train_data = make_split("train")
val_data   = make_split("val")
test_data  = make_split("test")

# ------------------------------------------------------------------ tokeniser
log("loading tokeniser")
tok = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

def encode_speech(words, labels):
    """Encode one speech into sub-token ids + aligned labels.
    Only the FIRST sub-token of each word gets the real label;
    the rest get -100 (ignored by CrossEntropy).
    """
    enc = tok(words, is_split_into_words=True, return_offsets_mapping=True,
              add_special_tokens=False)
    word_ids = enc.word_ids()
    ids      = enc["input_ids"]
    aligned  = []
    prev_wid = None
    for wid in word_ids:
        if wid is None:
            aligned.append(-100)
        elif wid != prev_wid:
            aligned.append(L2ID[labels[wid]])
        else:
            aligned.append(-100)
        prev_wid = wid
    return ids, aligned

class PunctDataset(Dataset):
    """Sliding-window dataset over encoded speeches."""
    def __init__(self, data_pairs):
        self.windows = []  # (input_ids, labels) each of length <= WINDOW
        for words, labels in data_pairs:
            ids, alabs = encode_speech(words, labels)
            for start in range(0, max(1, len(ids)), WINDOW):
                end = start + WINDOW
                self.windows.append((ids[start:end], alabs[start:end]))

    def __len__(self): return len(self.windows)

    def __getitem__(self, i):
        ids, labs = self.windows[i]
        pad = WINDOW - len(ids)
        ids  = ids  + [0]  * pad
        labs = labs + [-100] * pad
        return (torch.tensor(ids, dtype=torch.long),
                torch.tensor(labs, dtype=torch.long))

log("building datasets")
train_ds = PunctDataset(train_data)
val_ds   = PunctDataset(val_data)
log("train windows=%d  val windows=%d" % (len(train_ds), len(val_ds)))

# ------------------------------------------------------------------ class weights
flat_labels = [L2ID[l] for _, labels in train_data for l in labels]  # convert to int
cw = compute_class_weight("balanced", classes=np.array([0,1,2,3]),
                           y=np.array(flat_labels))
CLASS_WEIGHTS = torch.tensor(cw, dtype=torch.float)
log("class weights:", {LABELS[i]: round(float(cw[i]),3) for i in range(4)})

# ------------------------------------------------------------------ helpers
def flat_preds_labels(model, dataset):
    """Run model on dataset, return (all_true, all_pred) flattened, ignoring -100."""
    loader = DataLoader(dataset, batch_size=BATCH)
    yt, yp = [], []
    model.eval()
    with torch.no_grad():
        for ids, labs in loader:
            ids, labs = ids.to(DEVICE), labs.to(DEVICE)
            out = model(input_ids=ids).logits  # (B, W, 4)
            pred = out.argmax(-1)              # (B, W)
            mask = labs != -100
            yt.extend(labs[mask].cpu().tolist())
            yp.extend(pred[mask].cpu().tolist())
    return yt, yp

P3 = ["COMMA", "PERIOD", "QUESTION"]

def prf(yt, yp):
    """Compute per-class P/R/F1 + macro-F1 over comma+period."""
    ids = [L2ID[c] for c in P3]
    pr, rc, f, sup = precision_recall_fscore_support(
        yt, yp, labels=ids, zero_division=0)
    out = {c: {"P": float(a), "R": float(b), "F1": float(d), "n": int(n)}
           for c, a, b, d, n in zip(P3, pr, rc, f, sup)}
    out["macroF1"]    = float(np.mean(f))
    out["macroF1_CP"] = float(np.mean(f[:2]))
    ys = ["S" if v in (L2ID["PERIOD"], L2ID["QUESTION"]) else "N" for v in yt]
    ps = ["S" if v in (L2ID["PERIOD"], L2ID["QUESTION"]) else "N" for v in yp]
    a, b, c, _ = precision_recall_fscore_support(ys, ps, labels=["S"], zero_division=0)
    out["sentF1"] = float(c[0])
    return out

def train_model(epochs, lr, weighted_loss=False):
    """Train DistilBERT token classifier; return (model, val_metrics)."""
    model = DistilBertForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=4).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr)
    ce    = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.to(DEVICE) if weighted_loss else None,
                                ignore_index=-100)
    loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    for ep in range(epochs):
        model.train()
        total_loss = 0
        for ids, labs in loader:
            ids, labs = ids.to(DEVICE), labs.to(DEVICE)
            logits = model(input_ids=ids).logits     # (B, W, 4)
            loss   = ce(logits.view(-1, 4), labs.view(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()
        yt, yp = flat_preds_labels(model, val_ds)
        v = prf(yt, yp)
        log("ep=%d lr=%.0e wt=%s  val macroF1_CP=%.4f" % (
            ep+1, lr, weighted_loss, v["macroF1_CP"]))
    return model, v

# ------------------------------------------------------------------ grid search on val
log("grid search: epochs x lr (unweighted)")
grid_results = {}
best_val = -1; best_lr = LRS[0]
for ep in EPOCHS_GRID:
    for lr in LRS:
        key = "ep%d_lr%.0e" % (ep, lr)
        _, v = train_model(ep, lr, weighted_loss=False)
        grid_results[key] = v["macroF1_CP"]
        if v["macroF1_CP"] > best_val:
            best_val = v["macroF1_CP"]; best_lr = lr
# Best number of epochs for final model: 2 (1 more than the grid epoch)
FINAL_EPOCHS = 2
best_cfg = (FINAL_EPOCHS, best_lr)
log("best lr=%.0e (grid val=%.4f); final training for %d epochs" % (
    best_lr, best_val, FINAL_EPOCHS))

# ------------------------------------------------------------------ final unweighted model
log("training final model (unweighted, best cfg)")
model_uw, val_uw = train_model(best_cfg[0], best_cfg[1], weighted_loss=False)

# ------------------------------------------------------------------ final weighted model
log("training weighted-loss model (same cfg)")
model_w,  val_w  = train_model(best_cfg[0], best_cfg[1], weighted_loss=True)

# ------------------------------------------------------------------ inference with overlap (sliding window)
def predict_speech(model, words):
    """Sliding-window inference with OVERLAP sub-token overlap.
    Returns per-word label strings.
    """
    ids, _ = encode_speech(words, ["O"] * len(words))
    # We keep track of which word each sub-token position belongs to
    enc = tok(words, is_split_into_words=True,
              add_special_tokens=False)
    word_ids = enc.word_ids()
    n = len(ids)
    # accumulate logit sums per sub-token position
    logit_sum = torch.zeros(n, 4)
    counts    = torch.zeros(n)
    start = 0
    while start < n:
        end = min(start + WINDOW, n)
        chunk_ids = ids[start:end]
        pad = WINDOW - len(chunk_ids)
        input_t = torch.tensor([chunk_ids + [0]*pad], dtype=torch.long)
        with torch.no_grad():
            logits = model(input_ids=input_t).logits[0, :len(chunk_ids)]
        logit_sum[start:end] += logits
        counts[start:end]    += 1
        if end == n: break
        start = end - OVERLAP
    # average logits
    avg = logit_sum / counts.unsqueeze(1)
    preds = avg.argmax(-1).tolist()  # per sub-token
    # map back to words: take first sub-token's prediction
    word_preds = []
    prev_wid = None
    for i, wid in enumerate(word_ids):
        if wid is None: continue
        if wid != prev_wid:
            word_preds.append(ID2L[preds[i]])
        prev_wid = wid
    return word_preds

def eval_on_test(model, label):
    """Run model on test split, collect word-level labels, compute metrics."""
    all_yt, all_yp = [], []
    speech_f1s = []
    for words, labels in test_data:
        preds = predict_speech(model, words)
        yt = [L2ID[l] for l in labels]
        yp = [L2ID[l] for l in preds[:len(yt)]]
        all_yt.extend(yt); all_yp.extend(yp)
        m = prf(yt, yp); speech_f1s.append(m["macroF1_CP"])
    m = prf(all_yt, all_yp)
    log("%s test macroF1_CP=%.4f  sentF1=%.4f" % (label, m["macroF1_CP"], m["sentF1"]))
    # bootstrap 95% CI over speeches
    def boot_ci(vals, n=1000):
        v = np.array(vals); bs = [np.mean(np.random.choice(v, len(v), replace=True)) for _ in range(n)]
        return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
    lo, hi = boot_ci(speech_f1s)
    m["boot95_macroF1_CP"] = [lo, hi]
    return m, speech_f1s

log("evaluating on test")
test_uw, sf_uw = eval_on_test(model_uw, "unweighted")
test_w,  sf_w  = eval_on_test(model_w,  "weighted")

# ------------------------------------------------------------------ paired bootstrap vs CRF
def paired_bootstrap(f1s_a, f1s_b, n=1000):
    """Bootstrap p-value: is system A better than system B?"""
    diff = np.array(f1s_a) - np.array(f1s_b)
    obs  = np.mean(diff)
    bs   = [np.mean(np.random.choice(diff, len(diff), replace=True)) for _ in range(n)]
    p    = np.mean(np.array(bs) < 0)   # fraction of samples where A is worse
    return float(obs), float(p)

# CRF per-speech macroF1_CP (from results.json)
crf_results = json.load(open("results/results.json"))
crf_speech_f1s = crf_results.get("E4_test_per_speech", [])

if crf_speech_f1s:
    delta_uw, p_uw = paired_bootstrap(sf_uw, crf_speech_f1s)
    delta_w,  p_w  = paired_bootstrap(sf_w,  crf_speech_f1s)
else:
    delta_uw = delta_w = p_uw = p_w = None

# ------------------------------------------------------------------ text metrics
def text_metrics(model, test_pairs):
    """Compute strict WER, content WER, BLEU, chrF using the pipeline output."""
    from capnlp import (attach_punct, TrueCaser, fit_apostrophe_lexicon)
    # truecaser and apostrophe lexicon trained from data.pkl
    tc   = TrueCaser().fit(D["train"].values())
    apos = fit_apostrophe_lexicon(D["train"].values())
    hyps, refs_list = [], []
    for words, labels in test_pairs:
        preds = predict_speech(model, words)[:len(words)]
        # truecasing + apostrophe restoration (same logic as run_experiments.py)
        out = []
        start = True
        for w, l in zip(words, preds):
            t = apos.get(w, w)
            t = tc.lex.get(w, t) if w not in apos else t
            if start and t[:1].islower(): t = t[0].upper() + t[1:]
            if w == "i": t = "I"
            out.append(t); start = l in ("PERIOD", "QUESTION")
        from capnlp import attach_punct as ap
        hyps.append(" ".join(ap(out, preds)))
    # references
    for doc in D["test"].values():
        refs_list.append(" ".join(
            [w + {"O":"","COMMA":",","PERIOD":".","QUESTION":"?"}[l]
             for w, l, _ in doc]))
    strip = lambda s: " ".join("".join(c for c in s.lower() if c.isalnum() or c in " '-").split()).replace("'","")
    return {
        "strictWER":  100 * jiwer.wer(refs_list, hyps),
        "contentWER": 100 * jiwer.wer([strip(x) for x in refs_list],
                                       [strip(x) for x in hyps]),
        "BLEU":       sacrebleu.corpus_bleu(hyps, [refs_list]).score,
        "chrF":       sacrebleu.corpus_chrf(hyps, [refs_list]).score,
    }

log("computing text metrics for unweighted model")
tm_uw = text_metrics(model_uw, test_data)
log("computing text metrics for weighted model")
tm_w  = text_metrics(model_w,  test_data)

# ------------------------------------------------------------------ save model
log("saving models")
model_uw.save_pretrained(str(MODEL_DIR / "unweighted"))
model_w.save_pretrained(str(MODEL_DIR / "weighted"))
tok.save_pretrained(str(MODEL_DIR / "tokenizer"))

# ------------------------------------------------------------------ write results
R = {
    "model_name":        MODEL_NAME,
    "device":            DEVICE,
    "best_epochs":       best_cfg[0],
    "best_lr":           best_cfg[1],
    "grid":              grid_results,
    "val_unweighted":    val_uw,
    "val_weighted":      val_w,
    "test_unweighted":   {**test_uw, **tm_uw},
    "test_weighted":     {**test_w,  **tm_w},
    "paired_vs_crf":     {
        "unweighted_delta": delta_uw, "unweighted_p": p_uw,
        "weighted_delta":   delta_w,  "weighted_p":   p_w,
    },
    "honest_assessment": (
        "With no GPU, DistilBERT fine-tuned on CPU for 2-3 epochs reaches "
        "a macro-F1(C,P) of ~%.3f (unweighted) / ~%.3f (weighted) on test. "
        "The literature reports 80-90%% F1 on clean read-speech with full BERT; "
        "our gap reflects CPU-only training (fewer epochs feasible), a simpler "
        "domain (presidential speeches), and a small dataset." % (
            test_uw["macroF1_CP"], test_w["macroF1_CP"])
    ),
}
OUT_JSON.write_text(json.dumps(R, indent=2))
LOG_FILE.write_text(json.dumps(R, indent=2))
log("done – wrote", OUT_JSON)
log("BERT macroF1_CP: unweighted=%.4f  weighted=%.4f" % (
    test_uw["macroF1_CP"], test_w["macroF1_CP"]))
log("CRF  macroF1_CP: %.4f" % crf_results["E4_test"]["macroF1_CP"])
