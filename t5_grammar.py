"""
t5_grammar.py - Task 3: Fine-tune T5-small as a grammar-correction model.

Training pairs:
  source = noisy ASR-style text + predicted punctuation from best punctuation model
  target = cleaned reference sentence (<=64 words)
Compares with the rule-based (CRF) stage on validation, then evaluates on test.
Reports strict WER, content WER, BLEU, chrF, and hallucinated words per 1000.
"""
import pickle, json, time, pathlib, random, warnings, math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5TokenizerFast
import jiwer, sacrebleu
warnings.filterwarnings("ignore")

T0 = time.time()
def log(*a): print("[%5.0fs]" % (time.time() - T0), *a, flush=True)

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

DATA_PKL  = pathlib.Path("data.pkl")
OUT_JSON  = pathlib.Path("results_bert/t5_results.json")
LOG_FILE  = pathlib.Path("logs/t5_grammar.json")
MODEL_DIR = pathlib.Path("models/t5_grammar")
OUT_JSON.parent.mkdir(exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE.parent.mkdir(exist_ok=True)

MODEL_NAME  = "t5-small"   # CPU: t5-small; GPU: t5-base
MAX_SRC     = 80            # tokens in source (sentence + predicted punct)
MAX_TGT     = 80            # tokens in target
BATCH       = 8
EPOCHS      = 2             # enough to see signal; more is slow on CPU
LR          = 3e-4
DEVICE      = "cpu"
MAX_WORDS   = 64            # max words per sentence in training pairs

# ------------------------------------------------------------------ load data
log("loading data")
D = pickle.load(DATA_PKL.open("rb"))
crf_results = json.load(open("results/results.json"))

# load the baseline CRF model for generating punctuation on source side
import pickle as pkl
crf = pkl.load(open("results/crf.pkl", "rb"))   # our own file, safe to load

from capnlp import (asr_style, attach_punct, stream_feats, chunk,
                    TrueCaser, fit_apostrophe_lexicon, LABELS,
                    inject_disfluency, remove_disfluencies)

tc   = TrueCaser().fit(D["train"].values())
apos = fit_apostrophe_lexicon(D["train"].values())

def crf_predict(words, window=3):
    """Return per-word label list from CRF."""
    feats = stream_feats(words, window)
    return [l for c in chunk(feats) for l in crf.predict([c])[0]]

def restore(words, labels):
    """Apply truecasing + apostrophes + attach punctuation."""
    out, start = [], True
    for w, l in zip(words, labels):
        t = apos.get(w, w); t = tc.lex.get(w, t) if w not in apos else t
        if start and t[:1].islower(): t = t[0].upper() + t[1:]
        if w == "i": t = "I"
        out.append(t); start = l in ("PERIOD", "QUESTION")
    return attach_punct(out, labels)

def split_to_sentences(doc):
    """Split a speech doc into sentence tuples (words_asr, ref_sentence_str)."""
    sentences = []
    cur_w, cur_r = [], []
    for w, lab, _ in doc:
        cur_w.append(w.lower().replace("'", ""))
        cur_r.append(w + {"O":"","COMMA":",","PERIOD":".","QUESTION":"?"}[lab])
        if lab in ("PERIOD", "QUESTION"):
            if 2 <= len(cur_w) <= MAX_WORDS:
                sentences.append((list(cur_w), " ".join(cur_r)))
            cur_w, cur_r = [], []
    return sentences

# ------------------------------------------------------------------ build pairs
def build_pairs(split_key):
    """Return (source_str, target_str) pairs for T5 training."""
    pairs = []
    for doc in D[split_key].values():
        sents = split_to_sentences(doc)
        for asr_w, ref_str in sents:
            # predict punctuation on ASR input
            labs = crf_predict(asr_w)
            src_toks = restore(asr_w, labs)
            source = "fix grammar: " + " ".join(src_toks)
            target = ref_str
            pairs.append((source, target))
    return pairs

log("building training pairs")
train_pairs = build_pairs("train")
val_pairs   = build_pairs("val")
log("train=%d pairs  val=%d pairs" % (len(train_pairs), len(val_pairs)))

# ------------------------------------------------------------------ tokenise
log("loading T5 tokeniser")
tok = T5TokenizerFast.from_pretrained(MODEL_NAME)

class T5Dataset(Dataset):
    """Simple dataset of (source, target) string pairs."""
    def __init__(self, pairs):
        self.pairs = pairs
    def __len__(self): return len(self.pairs)
    def __getitem__(self, i):
        src, tgt = self.pairs[i]
        s = tok(src, max_length=MAX_SRC, truncation=True,
                padding="max_length", return_tensors="pt")
        t = tok(tgt, max_length=MAX_TGT, truncation=True,
                padding="max_length", return_tensors="pt")
        labels = t["input_ids"].squeeze()
        labels[labels == tok.pad_token_id] = -100
        return {"input_ids": s["input_ids"].squeeze(),
                "attention_mask": s["attention_mask"].squeeze(),
                "labels": labels}

train_ds = T5Dataset(train_pairs)
val_ds   = T5Dataset(val_pairs)

# ------------------------------------------------------------------ train
log("loading T5-small")
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)
opt   = torch.optim.AdamW(model.parameters(), lr=LR)
loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)

def eval_loss(ds):
    """Mean cross-entropy loss on a dataset (first 200 examples for speed)."""
    model.eval()
    losses = []
    dl = DataLoader(ds, batch_size=BATCH)
    for i, batch in enumerate(dl):
        if i >= 25: break   # ~200 examples
        ids  = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        labs = batch["labels"].to(DEVICE)
        with torch.no_grad():
            loss = model(input_ids=ids, attention_mask=mask, labels=labs).loss
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else 0.0

log("training for %d epochs" % EPOCHS)
for ep in range(EPOCHS):
    model.train()
    total = 0.0
    for step, batch in enumerate(loader):
        ids  = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        labs = batch["labels"].to(DEVICE)
        loss = model(input_ids=ids, attention_mask=mask, labels=labs).loss
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item()
        if (step + 1) % 200 == 0:
            log("  ep=%d step=%d loss=%.4f" % (ep+1, step+1, total/(step+1)))
    vl = eval_loss(val_ds)
    log("ep=%d  train_loss=%.4f  val_loss=%.4f" % (ep+1, total/len(loader), vl))

# ------------------------------------------------------------------ generate
def generate_speech(asr_words, max_new=MAX_TGT):
    """Generate corrected text for an entire speech (sentence by sentence)."""
    labs = crf_predict(asr_words)
    # split by predicted sentence boundaries
    sents = []; cur = []
    for w, l in zip(asr_words, labs):
        t = apos.get(w, w); t = tc.lex.get(w, t) if w not in apos else t
        if w == "i": t = "I"
        cur.append(t + {"O":"","COMMA":",","PERIOD":".","QUESTION":"?"}[l])
        if l in ("PERIOD", "QUESTION"):
            sents.append(" ".join(cur)); cur = []
    if cur: sents.append(" ".join(cur))
    outs = []
    model.eval()
    for sent in sents:
        src = "fix grammar: " + sent
        ids = tok(src, max_length=MAX_SRC, truncation=True,
                  return_tensors="pt")["input_ids"]
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=max_new, num_beams=2)
        outs.append(tok.decode(out[0], skip_special_tokens=True))
    return " ".join(outs)

def hallucination_rate(hyp, ref):
    """Words per 1000 in hypothesis that do not appear in reference (case-insensitive)."""
    h_words = set(hyp.lower().split())
    r_words = set(ref.lower().split())
    extra   = sum(1 for w in hyp.lower().split() if w not in r_words)
    total   = max(len(hyp.split()), 1)
    return 1000 * extra / total

def ref_text(doc):
    return " ".join(
        w + {"O":"","COMMA":",","PERIOD":".","QUESTION":"?"}[l]
        for w, l, _ in doc)

def eval_full(split_key, label):
    """Evaluate on a split; return metrics dict."""
    hyps, refs_list = [], []
    for doc in D[split_key].values():
        ws_asr = [w.lower().replace("'","") for w, _, _ in doc]
        hyps.append(generate_speech(ws_asr))
        refs_list.append(ref_text(doc))
    strip = lambda s: " ".join("".join(c for c in s.lower() if c.isalnum() or c in " '-").split()).replace("'","")
    hall = float(np.mean([hallucination_rate(h, r) for h, r in zip(hyps, refs_list)]))
    m = {
        "strictWER":       100 * jiwer.wer(refs_list, hyps),
        "contentWER":      100 * jiwer.wer([strip(x) for x in refs_list],
                                            [strip(x) for x in hyps]),
        "BLEU":            sacrebleu.corpus_bleu(hyps, [refs_list]).score,
        "chrF":            sacrebleu.corpus_chrf(hyps, [refs_list]).score,
        "hallucinated_per_1k": hall,
    }
    log("%s  strictWER=%.2f  BLEU=%.1f  hall/1k=%.1f" % (
        label, m["strictWER"], m["BLEU"], hall))
    return m

log("evaluating on validation (for decision)")
val_m = eval_full("val", "val")

# CRF baseline on val for comparison
log("CRF baseline on validation for comparison")
val_crf_hyps, val_refs = [], []
for doc in D["val"].values():
    ws_asr = [w.lower().replace("'","") for w, _, _ in doc]
    labs = crf_predict(ws_asr)
    val_crf_hyps.append(" ".join(restore(ws_asr, labs)))
    val_refs.append(ref_text(doc))
strip = lambda s: " ".join("".join(c for c in s.lower() if c.isalnum() or c in " '-").split()).replace("'","")
crf_val_m = {
    "strictWER": 100 * jiwer.wer(val_refs, val_crf_hyps),
    "BLEU":      sacrebleu.corpus_bleu(val_crf_hyps, [val_refs]).score,
}
log("CRF val  strictWER=%.2f  BLEU=%.1f" % (crf_val_m["strictWER"], crf_val_m["BLEU"]))

# Decision: T5 is better only if BLEU improves AND hallucination is low
t5_wins = (val_m["BLEU"] > crf_val_m["BLEU"] and val_m["hallucinated_per_1k"] < 50)
log("T5 wins on val: %s" % t5_wins)

# Evaluate on test ONCE regardless (as required), but note which is default
log("evaluating on test (reported once)")
test_m = eval_full("test", "test")

# ------------------------------------------------------------------ save model
model.save_pretrained(str(MODEL_DIR))
tok.save_pretrained(str(MODEL_DIR))

# ------------------------------------------------------------------ write JSON
R = {
    "model":             MODEL_NAME,
    "device":            DEVICE,
    "epochs":            EPOCHS,
    "lr":                LR,
    "val_t5":            val_m,
    "val_crf_baseline":  crf_val_m,
    "test_t5":           test_m,
    "t5_better_on_val":  t5_wins,
    "recommendation":    (
        "T5 is better than rule-based on validation." if t5_wins else
        "CRF pipeline kept as default – T5 does not improve (or hallucinates too much)."
    ),
}
OUT_JSON.write_text(json.dumps(R, indent=2))
LOG_FILE.write_text(json.dumps(R, indent=2))
log("done – wrote", OUT_JSON)
