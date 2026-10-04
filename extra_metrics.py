"""
extra_metrics.py - Task 7: METEOR and BERTScore for best system vs baselines.

BERTScore: uses distilbert-base-uncased (no GPU; state which in output).
METEOR:    uses nltk WordNet (already downloaded).

Evaluates on the test split. Writes results_bert/extra_metrics.json.
"""
import pickle, json, pathlib, time, warnings
import numpy as np
warnings.filterwarnings("ignore")

T0 = time.time()
def log(*a): print("[%5.0fs]" % (time.time() - T0), *a, flush=True)

OUT_JSON = pathlib.Path("results_bert/extra_metrics.json")
LOG_FILE = pathlib.Path("logs/extra_metrics.json")
OUT_JSON.parent.mkdir(exist_ok=True)
LOG_FILE.parent.mkdir(exist_ok=True)

# ------------------------------------------------------------------ load baseline output
import pickle as pkl
# data.pkl is our own generated file – safe to load
D   = pkl.load(open("data.pkl", "rb"))
crf = pkl.load(open("results/crf.pkl", "rb"))   # our own file

from capnlp import (asr_style, attach_punct, stream_feats, chunk,
                    TrueCaser, fit_apostrophe_lexicon, LABELS)
from data_loader import load_tc_apos

tc, apos = load_tc_apos()

def restore(words, labels):
    """Truecasing + apostrophes + punctuation attachment."""
    out, start = [], True
    for w, l in zip(words, labels):
        t = apos.get(w, w); t = tc.lex.get(w, t) if w not in apos else t
        if start and t[:1].islower(): t = t[0].upper() + t[1:]
        if w == "i": t = "I"
        out.append(t); start = l in ("PERIOD", "QUESTION")
    return attach_punct(out, labels)

def crf_predict(words, window=3):
    """Run CRF on a word list."""
    feats = stream_feats(words, window)
    return [l for c in chunk(feats) for l in crf.predict([c])[0]]

def ref_text(doc):
    return " ".join(
        w + {"O":"","COMMA":",","PERIOD":".","QUESTION":"?"}[l]
        for w, l, _ in doc)

test_docs = list(D["test"].values())
refs = [ref_text(d) for d in test_docs]

# ------------------------------------------------------------------ build system outputs
log("building system outputs")
# 1. Raw ASR (no processing)
raw_hyps = [" ".join(w.lower().replace("'","") for w, _, _ in d) for d in test_docs]

# 2. CRF full pipeline
crf_hyps = []
for doc in test_docs:
    ws = [w.lower().replace("'","") for w, _, _ in doc]
    labs = crf_predict(ws)
    crf_hyps.append(" ".join(restore(ws, labs)))

# 3. BERT (if available)
bert_hyps = None
bert_available = pathlib.Path("models/bert_punct/unweighted").exists()
if bert_available:
    log("loading BERT model")
    from transformers import DistilBertForTokenClassification, DistilBertTokenizerFast
    import torch
    WINDOW = 128; OVERLAP = 16
    LABELS_LIST = ["O", "COMMA", "PERIOD", "QUESTION"]
    ID2L = {i: l for i, l in enumerate(LABELS_LIST)}
    bert_m = DistilBertForTokenClassification.from_pretrained("models/bert_punct/unweighted")
    bert_t = DistilBertTokenizerFast.from_pretrained("models/bert_punct/tokenizer")
    bert_m.eval()

    def predict_bert(words):
        enc = bert_t(words, is_split_into_words=True, add_special_tokens=False)
        ids, word_ids = enc["input_ids"], enc.word_ids()
        n = len(ids)
        logit_sum = torch.zeros(n, 4); counts = torch.zeros(n)
        start = 0
        while start < n:
            end = min(start + WINDOW, n)
            ci = ids[start:end]; pad = WINDOW - len(ci)
            t  = torch.tensor([ci + [0]*pad], dtype=torch.long)
            with torch.no_grad():
                logits = bert_m(input_ids=t).logits[0, :len(ci)]
            logit_sum[start:end] += logits; counts[start:end] += 1
            if end == n: break
            start = end - OVERLAP
        preds = (logit_sum / counts.unsqueeze(1)).argmax(-1).tolist()
        word_preds, prev_wid = [], None
        for i, wid in enumerate(word_ids):
            if wid is None: continue
            if wid != prev_wid: word_preds.append(ID2L[preds[i]])
            prev_wid = wid
        return word_preds

    bert_hyps = []
    for doc in test_docs:
        ws = [w.lower().replace("'","") for w, _, _ in doc]
        labs = predict_bert(ws)[:len(ws)]
        bert_hyps.append(" ".join(restore(ws, labs)))

# ------------------------------------------------------------------ METEOR
log("computing METEOR")
import nltk
nltk.download("punkt_tab", quiet=True)   # needed for word_tokenize
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize

def corpus_meteor(hyps, refs_list):
    """Mean sentence-level METEOR over the corpus."""
    scores = []
    for h, r in zip(hyps, refs_list):
        h_tok = word_tokenize(h.lower())
        r_tok = word_tokenize(r.lower())
        scores.append(meteor_score([r_tok], h_tok))
    return float(np.mean(scores))

meteor_raw  = corpus_meteor(raw_hyps,  refs)
meteor_crf  = corpus_meteor(crf_hyps,  refs)
meteor_bert = corpus_meteor(bert_hyps, refs) if bert_hyps else None
log("METEOR  raw=%.4f  crf=%.4f  bert=%s" % (
    meteor_raw, meteor_crf, "%.4f" % meteor_bert if meteor_bert else "N/A"))

# ------------------------------------------------------------------ BERTScore
log("computing BERTScore (distilbert-base-uncased, CPU)")
from bert_score import score as bertscore

BERT_MODEL = "distilbert-base-uncased"   # CPU – no GPU available

def compute_bertscore(hyps, refs_list):
    """Return mean F1 BERTScore."""
    P, R, F1 = bertscore(hyps, refs_list, model_type=BERT_MODEL, verbose=False,
                          device="cpu")
    return float(F1.mean()), float(P.mean()), float(R.mean())

log("BERTScore for raw ASR")
bs_f_raw,  bs_p_raw,  bs_r_raw  = compute_bertscore(raw_hyps, refs)
log("BERTScore for CRF pipeline")
bs_f_crf,  bs_p_crf,  bs_r_crf  = compute_bertscore(crf_hyps, refs)
if bert_hyps:
    log("BERTScore for BERT system")
    bs_f_bert, bs_p_bert, bs_r_bert = compute_bertscore(bert_hyps, refs)
else:
    bs_f_bert = bs_p_bert = bs_r_bert = None

log("BERTScore F1  raw=%.4f  crf=%.4f  bert=%s" % (
    bs_f_raw, bs_f_crf, "%.4f" % bs_f_bert if bs_f_bert else "N/A"))

# ------------------------------------------------------------------ write JSON
R = {
    "bertscore_model": BERT_MODEL,
    "bertscore_device": "cpu",
    "METEOR": {
        "raw_asr":      meteor_raw,
        "crf_pipeline": meteor_crf,
        "bert_punct":   meteor_bert,
    },
    "BERTScore_F1": {
        "raw_asr":      bs_f_raw,
        "crf_pipeline": bs_f_crf,
        "bert_punct":   bs_f_bert,
    },
    "BERTScore_P": {
        "raw_asr":      bs_p_raw,
        "crf_pipeline": bs_p_crf,
        "bert_punct":   bs_p_bert,
    },
    "BERTScore_R": {
        "raw_asr":      bs_r_raw,
        "crf_pipeline": bs_r_crf,
        "bert_punct":   bs_r_bert,
    },
}
OUT_JSON.write_text(json.dumps(R, indent=2))
LOG_FILE.write_text(json.dumps(R, indent=2))
log("done – wrote", OUT_JSON)
