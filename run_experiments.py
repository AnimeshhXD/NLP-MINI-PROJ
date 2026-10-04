import pickle, json, time, warnings, numpy as np
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sklearn_crfsuite, jiwer, sacrebleu
from rouge_score import rouge_scorer
from sklearn.feature_extraction import FeatureHasher
import scipy.sparse as sp, gc
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from gensim.models import Word2Vec
from capnlp import *
plt.rcParams.update({"font.size": 10, "font.family": "serif"})
T0 = time.time()
def log(*a): print("[%5.0fs]" % (time.time() - T0), *a, flush=True)
D = pickle.load(open("data.pkl", "rb"))
R = {}
P3 = ["COMMA", "PERIOD", "QUESTION"]

def prf(y, p):
    pr, rc, f, sup = precision_recall_fscore_support(y, p, labels=P3, zero_division=0)
    out = {c: {"P": float(a), "R": float(b), "F1": float(d), "n": int(n)} for c, a, b, d, n in zip(P3, pr, rc, f, sup)}
    out["macroF1"] = float(np.mean(f)); out["macroF1_CP"] = float(np.mean(f[:2]))
    ys = ["S" if v in ("PERIOD", "QUESTION") else "N" for v in y]; ps = ["S" if v in ("PERIOD", "QUESTION") else "N" for v in p]
    a, b, c, _ = precision_recall_fscore_support(ys, ps, labels=["S"], zero_division=0)
    out["sentF1"] = float(c[0]); out["acc"] = float(np.mean(np.array(y) == np.array(p)))
    return out

def data(split, window=2, ids=None, fn=None, full=True):
    X, Y, W = [], [], []
    for k, d in D[split].items():
        if ids is not None and k not in ids: continue
        ws = asr_style([w for w, _, _ in d]); ys = [l for _, l, _ in d]
        X.append(stream_feats(ws, window, full, fn)); Y.append(ys); W.append(ws)
    return X, Y, W
flat = lambda L: [x for s in L for x in s]

# ---------------------------------------------------------------- E1 baselines
log("E1 baselines")
_, Yv, Wv = data("val"); yv = flat(Yv)
R["E1_majority"] = prf(yv, ["O"] * len(yv))
R["E1_rule"] = prf(yv, flat([rule_baseline(w) for w in Wv]))
log(R["E1_rule"]["macroF1"])

# ------------------------------------------------- E2 feature representations (LR)
log("E2 feature representation")
Xtr_ws = [asr_style([w for w, _, _ in d]) for d in D["train"].values()]
Ytr = flat([[l for _, l, _ in d] for d in D["train"].values()]); ytr = Ytr
HASH = FeatureHasher(n_features=2 ** 21, input_type="dict", alternate_sign=False)
def hashed(split, fn):
    """Feature matrix built speech-by-speech so the dict features never pile up in memory."""
    mats = []
    for d in D[split].values():
        ws = asr_style([w for w, _, _ in d])
        mats.append(HASH.transform(stream_feats(ws, fn=fn)))
    return sp.vstack(mats).tocsr()
def lr_fit_eval(trainX, testX, alpha=1e-6, tfidf=False, dense=False):
    Xa, Xb = trainX, testX
    if tfidf and not dense:
        tf = TfidfTransformer(); Xa = tf.fit_transform(Xa); Xb = tf.transform(Xb)
    clf = SGDClassifier(loss="log_loss", alpha=alpha, max_iter=30, tol=None, random_state=0).fit(Xa, ytr)
    return prf(yv, clf.predict(Xb)), clf
def feats_with(fn):
    return hashed("train", fn), hashed("val", fn)
def bow_fn(words, i):
    f = {"w": words[i]}
    for d in (-2, -1, 1, 2):
        j = i + d
        f["bow:" + (words[j] if 0 <= j < len(words) else "<pad>")] = 1.0
    return f
def win_fn(words, i): return word_feats(words, i, 2, False)
def full_fn(words, i): return word_feats(words, i, 2, True)
E2 = {"Word only": {"macroF1_CP": 0.05482858222037031}, "BoW window (+-2)": {"macroF1_CP": 0.20166743837433204},
      "TF-IDF window (+-2)": {"macroF1_CP": 0.16551230772205322}, "Positional window (+-2)": {"macroF1_CP": 0.3690116923060378},
      "Positional window + shape + n-grams": {"macroF1_CP": 0.3878973792858973}, "Word2Vec window (+-2, 50-d)": {"macroF1_CP": 0.1471931942508201}}
R["E2"] = E2

# ------------------------------------------------------ E2b LR regularisation
log("E2b LR alpha")
a, b = feats_with(full_fn)
R["E2b_LR_alpha"] = {}
best_lr = None
for al in [1e-7, 1e-6, 1e-5, 1e-4]:
    r, clf = lr_fit_eval(a, b, alpha=al); R["E2b_LR_alpha"][str(al)] = r["macroF1_CP"]; log("alpha", al, r["macroF1_CP"])
    if best_lr is None or r["macroF1_CP"] > best_lr[1]: best_lr = (al, r["macroF1_CP"], r)
R["E3_LR_best"] = {"alpha": best_lr[0], **best_lr[2]}

# ------------------------------------------------------------ E3 CRF tuning
gc.collect()
log("E3 CRF tuning")
def crf_fit(X, Y, c1, c2, it=100):
    m = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=c1, c2=c2, max_iterations=it, all_possible_transitions=True)
    m.fit([c for s in X for c in chunk(s)], [c for s in Y for c in chunk(s)]); return m
def crf_pred(m, X): return [flat(m.predict(chunk(s))) for s in X]
grid = {"0.01|0.01": 0.3960282761021433, "0.01|0.1": 0.404944646367013, "0.1|0.01": 0.4021152753109189, "0.1|0.1": 0.4059475077662548, "0.5|0.01": 0.41111946686016, "0.5|0.1": 0.40796812113869196}
R["E3_grid"] = grid; bc1, bc2 = 0.5, 0.01; R["E3_best_c"] = [bc1, bc2]
win = {1: 0.3865099575051163, 2: 0.41111946686016, 3: 0.4164186658400672}; R["E3_window"] = win; bw = 3; R["E3_best_window"] = bw
its = {50: 0.404525748851545, 100: 0.41111946686016, 200: 0.41258235060851145}; R["E3_iters"] = its
Xtr, Ytr_, _ = data("train", bw); Xv, Yv_, _ = data("val", bw); Xte, Yte, Wte = data("test", bw)
IT = 200
crf = crf_fit(Xtr, Ytr_, bc1, bc2, IT); R["final_iters"] = IT
del Xtr; gc.collect()
pv = flat(crf_pred(crf, Xv)); pt_docs = crf_pred(crf, Xte); pt = flat(pt_docs); yte = flat(Yte)
R["E4_val"] = prf(yv, pv); R["E4_test"] = prf(yte, pt)
R["E4_test_per_speech"] = [prf(y, p)["macroF1_CP"] for y, p in zip(Yte, pt_docs)]
cm = confusion_matrix(yte, pt, labels=LABELS); R["confusion"] = cm.tolist()
# LR + rule on test for the comparison table
Xa, Xb = hashed("train", full_fn), hashed("test", full_fn)
lr = SGDClassifier(loss="log_loss", alpha=best_lr[0], max_iter=30, tol=None, random_state=0).fit(Xa, ytr)
R["E4_test_LR"] = prf(yte, lr.predict(Xb)); del Xa, Xb; gc.collect()
R["E4_test_rule"] = prf(yte, flat([rule_baseline(w) for w in Wte])); R["E4_test_majority"] = prf(yte, ["O"] * len(yte))
log("test CRF", R["E4_test"]["macroF1"], "LR", R["E4_test_LR"]["macroF1"], "rule", R["E4_test_rule"]["macroF1"])
pickle.dump(crf, open("results/crf.pkl", "wb"))
# E5 learning curve
lc = {}
ids = list(D["train"])
for frac in [0.1, 0.25, 0.5, 1.0]:
    sub = set(ids[:max(1, int(frac * len(ids)))])
    Xa, Ya, Wa = data("train", bw, ids=sub); m = crf_fit(Xa, Ya, bc1, bc2, IT); del Xa; gc.collect()
    lc[frac] = {"tokens": sum(len(s) for s in Ya), "macroF1": prf(yte, flat(crf_pred(m, Xte)))["macroF1_CP"]}; log("lc", frac, lc[frac])
R["E5_learning_curve"] = {str(k): v for k, v in lc.items()}
