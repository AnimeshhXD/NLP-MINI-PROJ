"""
grammar/train_t5.py  -  Step 4: T5-small grammar / disfluency correction.

Input  (source): CRF-punctuated text from the ASR proxy
                 (lowercased, punct-stripped, then CRF-repunctuated)
Output (target): Reference (original speech text, sentence-level)

⚠  HONEST NOTE: Without real Whisper output (BLOCKED – no ffmpeg),
   we use a text proxy: the reference text lowercased and stripped
   of punctuation, as if Whisper had transcribed it. This is a fair
   training signal for the grammar model but is NOT real ASR output.
   The model is evaluated on the same text-proxy test set.

Saves:
  results_new/t5_results.json   (metrics + provenance)
  models/t5_grammar_v2/         (final model)
"""
import pathlib, pickle, json, sys, time, re, random
import numpy as np

T0 = time.time()
def log(*a): print(f"[{time.time()-T0:5.0f}s]", *a, flush=True)

DATA_PKL = pathlib.Path("data/talks.pkl")
if not DATA_PKL.exists():
    print("Run  python data/prepare.py  first.")
    sys.exit(1)

talks = pickle.load(DATA_PKL.open("rb"))

RESULTS_DIR = pathlib.Path("results_new"); RESULTS_DIR.mkdir(exist_ok=True)
MODEL_DIR   = pathlib.Path("models/t5_grammar_v2"); MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED); np.random.seed(SEED)

# ---- CRF trained on train split (for generating source text) ----
import sklearn_crfsuite

def word_features(words, i):
    w = words[i]; l = w.lower()
    f = {"w": l, "suf2": l[-2:], "len": len(w), "upper": w[0].isupper()}
    if i > 0:   f["pw"] = words[i-1].lower()
    else:        f["BOS"] = True
    if i < len(words)-1: f["nw"] = words[i+1].lower()
    else:        f["EOS"] = True
    return f

train_talks = [t for t in talks if t["split"] == "train"]
val_talks   = [t for t in talks if t["split"] == "val"]
test_talks  = [t for t in talks if t["split"] == "test"]

log("Fitting CRF for source text generation…")
LABELS = ["O", "COMMA", "PERIOD", "QUESTION"]

X_tr = [[word_features([w for w,_ in t["pairs"]], i)
          for i in range(len(t["pairs"]))] for t in train_talks]
y_tr = [[l for _,l in t["pairs"]] for t in train_talks]

crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=0.1,
                             max_iterations=100, all_possible_transitions=True)
crf.fit(X_tr, y_tr)
log("CRF fitted")


def apply_crf(pairs):
    """Apply CRF to (word, _) pairs; return CRF-punctuated string."""
    words  = [w for w,_ in pairs]
    feats  = [word_features(words, i) for i in range(len(words))]
    preds  = crf.predict([feats])[0]
    out = []
    for w, p in zip(words, preds):
        out.append(w)
        if p == "COMMA":   out.append(",")
        elif p == "PERIOD":  out.append(".")
        elif p == "QUESTION": out.append("?")
    return " ".join(out)


def pairs_to_ref(pairs):
    """Reconstruct reference string from (word, label) pairs."""
    out = []
    for w, l in pairs:
        out.append(w)
        if l == "COMMA":    out.append(",")
        elif l == "PERIOD":   out.append(".")
        elif l == "QUESTION": out.append("?")
    return " ".join(out)


# build sentence-level pairs (source=CRF-output, target=reference)
def make_sentence_pairs(talk_list):
    pairs = []
    for t in talk_list:
        ref   = pairs_to_ref(t["pairs"])
        src   = apply_crf(t["pairs"])
        # split into sentence chunks (split on . or ?)
        ref_sents = re.split(r"(?<=[.?])\s+", ref)
        src_sents = re.split(r"(?<=[.?])\s+", src)
        n = min(len(ref_sents), len(src_sents))
        for rs, ss in zip(ref_sents[:n], src_sents[:n]):
            rs, ss = rs.strip(), ss.strip()
            if 3 <= len(rs.split()) <= 64 and 3 <= len(ss.split()) <= 64:
                pairs.append(("fix grammar: " + ss, rs))
    return pairs

log("Building sentence pairs…")
train_pairs = make_sentence_pairs(train_talks)
val_pairs   = make_sentence_pairs(val_talks)
test_pairs  = make_sentence_pairs(test_talks)
log(f"  pairs: train={len(train_pairs)}  val={len(val_pairs)}  test={len(test_pairs)}")

# ---- T5 fine-tuning ----
import torch
from transformers import T5ForConditionalGeneration, T5TokenizerFast
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
MAX_SRC  = 80
MAX_TGT  = 80
BATCH    = 8
EPOCHS   = 2
LR       = 3e-4
T5_NAME  = "t5-small"

tokenizer = T5TokenizerFast.from_pretrained(T5_NAME)
log(f"T5 tokenizer loaded, device={DEVICE}")


class T5Dataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        enc = tokenizer(src, max_length=MAX_SRC, truncation=True,
                        padding="max_length", return_tensors="pt")
        with tokenizer.as_target_tokenizer():
            dec = tokenizer(tgt, max_length=MAX_TGT, truncation=True,
                            padding="max_length", return_tensors="pt")
        lbl = dec["input_ids"].clone()
        lbl[lbl == tokenizer.pad_token_id] = -100
        return {"input_ids":      enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels":         lbl.squeeze(0)}


model = T5ForConditionalGeneration.from_pretrained(T5_NAME).to(DEVICE)
opt   = AdamW(model.parameters(), lr=LR)
ds    = T5Dataset(train_pairs)
dl    = DataLoader(ds, batch_size=BATCH, shuffle=True)
log(f"Training T5 for {EPOCHS} epochs, {len(ds)} samples")

for ep in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    for step, batch in enumerate(dl, 1):
        ids  = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        lbl  = batch["labels"].to(DEVICE)
        out  = model(input_ids=ids, attention_mask=mask, labels=lbl)
        out.loss.backward()
        opt.step(); opt.zero_grad()
        total_loss += out.loss.item()
        if step % 100 == 0:
            log(f"  ep={ep} step={step} loss={total_loss/step:.4f}")
    log(f"ep={ep} done  avg_loss={total_loss/len(dl):.4f}")

model.save_pretrained(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)
log(f"Saved {MODEL_DIR}")


# ---- evaluate ----
def hallucination_rate(src, hyp):
    """Words in hypothesis not in source (per 1000 hypothesis words)."""
    sw = set(re.findall(r"[a-z']+", src.lower()))
    hw = re.findall(r"[a-z']+", hyp.lower())
    hall = sum(1 for w in hw if w not in sw)
    return 1000 * hall / max(len(hw), 1)


@torch.no_grad()
def generate_batch(srcs):
    model.eval()
    enc = tokenizer(srcs, max_length=MAX_SRC, truncation=True,
                    padding=True, return_tensors="pt").to(DEVICE)
    out = model.generate(enc["input_ids"], attention_mask=enc["attention_mask"],
                          max_new_tokens=MAX_TGT, num_beams=4, early_stopping=True)
    return [tokenizer.decode(o, skip_special_tokens=True) for o in out]


log("Generating test predictions…")
BATCH_INF = 16
test_hyps = []
for i in range(0, len(test_pairs), BATCH_INF):
    batch_src = [s for s,_ in test_pairs[i:i+BATCH_INF]]
    test_hyps.extend(generate_batch(batch_src))

test_refs = [r for _,r in test_pairs]
test_srcs = [s for s,_ in test_pairs]

import sacrebleu
bleu = sacrebleu.corpus_bleu(test_hyps, [test_refs]).score
chrf = sacrebleu.corpus_chrf(test_hyps, [test_refs]).score
hall = np.mean([hallucination_rate(s, h) for s,h in zip(test_srcs, test_hyps)])
log(f"BLEU={bleu:.1f}  chrF={chrf:.1f}  hallucination/1k={hall:.1f}")

import hashlib
data_hash = hashlib.md5(open("data/talks.pkl","rb").read()).hexdigest()[:12]

R = {
    "timestamp":   time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "seed":        SEED,
    "data_hash":   data_hash,
    "t5_model":    T5_NAME,
    "lr":          LR,
    "epochs":      EPOCHS,
    "train_pairs": len(train_pairs),
    "test_pairs":  len(test_pairs),
    "test": {
        "BLEU":              round(bleu, 2),
        "chrF":              round(chrf, 2),
        "hallucination_per_1k": round(float(hall), 2),
    },
    "note": ("Source text is CRF output on ASR proxy (lowercased reference), "
             "NOT real Whisper output. See BLOCKED.md for ASR status."),
}
out = RESULTS_DIR / "t5_results.json"
out.write_text(json.dumps(R, indent=2))
log(f"Wrote {out}")
log(f"=== T5 test: BLEU={bleu:.1f}  chrF={chrf:.1f}  hall/1k={hall:.1f} ===")
