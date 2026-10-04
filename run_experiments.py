import pickle, json, time, collections, warnings, numpy as np
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
        f["bow:" + (words[j] if 0 <= j < len(words) else "<pad>")] = 1.0   # position-free bag of window words
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
# learning curve
lc = {}
ids = list(D["train"])
for frac in [0.1, 0.25, 0.5, 1.0]:
    sub = set(ids[:max(1, int(frac * len(ids)))])
    Xa, Ya, Wa = data("train", bw, ids=sub); m = crf_fit(Xa, Ya, bc1, bc2, IT); del Xa; gc.collect()
    lc[frac] = {"tokens": sum(len(s) for s in Ya), "macroF1": prf(yte, flat(crf_pred(m, Xte)))["macroF1_CP"]}; log("lc", frac, lc[frac])
R["E5_learning_curve"] = {str(k): v for k, v in lc.items()}

# ------------------------------------------------------------- pipeline (E6)
log("E6 pipeline")
caser = TrueCaser().fit(D["train"].values()); apos = fit_apostrophe_lexicon(D["train"].values())
R["truecase_lex_size"] = len(caser.lex); R["apos_lex_size"] = len(apos)
def restore_case_apos(words, labels):
    out = []
    start = True
    for w, l in zip(words, labels):
        t = apos.get(w, w)
        t = caser.lex.get(w, t) if w not in apos else t
        if start and t[:1].islower(): t = t[0].upper() + t[1:]
        if w == "i": t = "I"
        out.append(t); start = l in ("PERIOD", "QUESTION")
    return out
def ref_text(d): return " ".join(attach_punct([w for w, _, _ in d], [l for _, l, _ in d]))
def metrics(hyps, refs, rouge=False):
    r = {"strictWER": 100 * jiwer.wer(refs, hyps)}
    strip = lambda s: " ".join("".join(c for c in s.lower() if c.isalnum() or c in " '-").split()).replace("'", "")
    r["contentWER"] = 100 * jiwer.wer([strip(x) for x in refs], [strip(x) for x in hyps])
    r["BLEU"] = sacrebleu.corpus_bleu(hyps, [refs]).score
    r["chrF"] = sacrebleu.corpus_chrf(hyps, [refs]).score
    if rouge:
        sc = rouge_scorer.RougeScorer(["rouge1", "rougeL"]); r1, rl = [], []
        for h, f in zip(hyps, refs):
            hw, fw = h.split(), f.split()
            for i in range(0, len(fw), 100):
                s = sc.score(" ".join(fw[i:i + 100]), " ".join(hw[i:i + 100])); r1.append(s["rouge1"].fmeasure); rl.append(s["rougeL"].fmeasure)
        r["ROUGE1"] = 100 * np.mean(r1); r["ROUGEL"] = 100 * np.mean(rl)
    return r
test_docs = list(D["test"].values())
refs = [ref_text(d) for d in test_docs]
clean_in = [asr_style([w for w, _, _ in d]) for d in test_docs]
noisy_in = [inject_disfluency(w, seed=i) for i, w in enumerate(clean_in)]
R["noise_stats"] = {"clean_tokens": sum(map(len, clean_in)), "noisy_tokens": sum(map(len, noisy_in))}
stage = {}
stage["Raw ASR-style text (no processing)"] = metrics([" ".join(w) for w in clean_in], refs, True)
hy = [];
for ws in clean_in:
    labs = flat(crf.predict(chunk(stream_feats(ws, bw)))); hy.append(" ".join(attach_punct(ws, labs)))
stage["+ punctuation restoration (CRF)"] = metrics(hy, refs, True)
hy = []
for ws in clean_in:
    labs = flat(crf.predict(chunk(stream_feats(ws, bw)))); hy.append(" ".join(attach_punct(caser.apply(ws, labs), labs)))
stage["+ truecasing"] = metrics(hy, refs, True)
hy, hyp_tokens, hyp_labels = [], [], []
for ws in clean_in:
    labs = flat(crf.predict(chunk(stream_feats(ws, bw)))); toks = restore_case_apos(ws, labs)
    disp = attach_punct(toks, labs); hy.append(" ".join(disp)); hyp_tokens.append(disp); hyp_labels.append(labs)
stage["+ apostrophe restoration (full pipeline)"] = metrics(hy, refs, True)
R["E6_stage_clean"] = stage
hy_clean_full = hy
# oracle punctuation
hy = []
for d, ws in zip(test_docs, clean_in):
    labs = [l for _, l, _ in d]; hy.append(" ".join(attach_punct(restore_case_apos(ws, labs), labs)))
R["E6_oracle_punct"] = metrics(hy, refs, True)
# casing accuracy: predicted vs oracle boundaries
def case_acc(pred_toks_docs, docs):
    ok = n = 0
    for toks, d in zip(pred_toks_docs, docs):
        for t, (_, _, cs) in zip(toks, d):
            t = t.rstrip(".,?")
            pc = "UPPER" if (t.isupper() and len(t) > 1 and t.isalpha()) else "CAP" if t[:1].isupper() else "LOWER"
            ok += pc == cs; n += 1
    return 100 * ok / n
R["case_acc_pred"] = case_acc(hyp_tokens, test_docs)
orc = []
for d, ws in zip(test_docs, clean_in):
    labs = [l for _, l, _ in d]; orc.append(attach_punct(restore_case_apos(ws, labs), labs))
R["case_acc_oracle"] = case_acc(orc, test_docs)
R["case_acc_lowercase_only"] = case_acc([[w for w in ws] for ws in clean_in], test_docs)
# noisy condition
noisy = {}
noisy["Noisy ASR text (no processing)"] = metrics([" ".join(w) for w in noisy_in], refs)
def full_pipe(ws, rm):
    ws2 = remove_disfluencies(ws) if rm else ws
    labs = flat(crf.predict(chunk(stream_feats(ws2, bw)))); return attach_punct(restore_case_apos(ws2, labs), labs)
noisy["Pipeline without disfluency removal"] = metrics([" ".join(full_pipe(w, False)) for w in noisy_in], refs)
noisy_out = [full_pipe(w, True) for w in noisy_in]
noisy["Full pipeline (with disfluency removal)"] = metrics([" ".join(x) for x in noisy_out], refs)
R["E6_noisy"] = noisy
rm_clean = [remove_disfluencies(w) for w in clean_in]
R["disfl_collateral_tokens_removed_clean"] = sum(map(len, clean_in)) - sum(map(len, rm_clean))
R["disfl_tokens_removed_noisy"] = sum(map(len, noisy_in)) - sum(map(len, [remove_disfluencies(w) for w in noisy_in]))
R["disfl_exact_recovery_pct"] = 100 * np.mean([remove_disfluencies(n) == c for n, c in zip(noisy_in, clean_in)])
inj = sum(map(len, noisy_in)) - sum(map(len, clean_in)); R["disfl_injected"] = inj
# word-level disfluency P/R: tokens removed vs tokens injected
R["disfl_residual_extra_tokens"] = sum(len(remove_disfluencies(n)) - len(c) for n, c in zip(noisy_in, clean_in))
# ---- segmentation
seg = {"proposed": collections.Counter(), "baseline": collections.Counter()}
agg = {"proposed": [], "baseline": []}
pb = bb = 0; p_end_ok = b_end_ok = 0
for d, toks, ws in zip(test_docs, hyp_tokens, clean_in):
    ref_l = [l for _, l, _ in d]
    pblocks = segment_captions(toks); bblocks = baseline_segments(ws)
    for name, bl in (("proposed", pblocks), ("baseline", bblocks)):
        agg[name].append(bl)
    for name, bl in (("proposed", pblocks), ("baseline", bblocks)):
        idx = 0
        for b in bl:
            idx += len(" ".join(b).split())
            if 0 < idx <= len(ref_l) and ref_l[idx - 1] != "O":
                if name == "proposed": p_end_ok += 1
                else: b_end_ok += 1
S = {}
for name in ("proposed", "baseline"):
    allb = [b for bl in agg[name] for b in bl]; st = block_stats(allb); st["ends_at_ref_boundary"] = (p_end_ok if name == "proposed" else b_end_ok) / len(allb)
    S[name] = st
R["E6_segmentation"] = S
sample_blocks = segment_captions(hyp_tokens[0])[:14]
open("results/sample.srt", "w").write(to_srt(sample_blocks)); open("results/sample.vtt", "w").write(to_srt(sample_blocks, vtt=True))
open("results/full_test_speech0.srt", "w").write(to_srt(segment_captions(hyp_tokens[0])))
R["sample"] = {"id": list(D["test"])[0], "ref": " ".join(refs[0].split()[:45]), "asr_input": " ".join(clean_in[0][:45]),
               "output": " ".join(hy_clean_full[0].split()[:45]), "noisy_input": " ".join(noisy_in[0][:50]), "noisy_output": " ".join(noisy_out[0][:45])}
# timing
t = time.time(); _ = [full_pipe(w, True) for w in noisy_in]; el = time.time() - t
R["speed_words_per_sec"] = sum(map(len, noisy_in)) / el
R["total_runtime_s"] = time.time() - T0
json.dump(R, open("results/results.json", "w"), indent=1, default=float)

# ------------------------------------------------------------------- figures
fig, ax = plt.subplots(figsize=(6.4, 3.4))
names = list(E2); vals = [E2[n]["macroF1_CP"] for n in names]
ax.barh(names[::-1], vals[::-1], color="#3b6ea5"); ax.set_xlabel("Macro-F1 over comma and period classes (validation set)");
for i, v in enumerate(vals[::-1]): ax.text(v + 0.005, i, "%.3f" % v, va="center", fontsize=8)
plt.tight_layout(); plt.savefig("figs/exp_features.pdf"); plt.close()
c1s = [0.01, 0.1, 0.5]; c2s = [0.01, 0.1]
M = np.array([[grid["%s|%s" % (a, b)] for b in c2s] for a in c1s])
fig, ax = plt.subplots(figsize=(4.0, 3.2)); im = ax.imshow(M, cmap="Blues")
ax.set_xticks(range(2)); ax.set_xticklabels(c2s); ax.set_yticks(range(3)); ax.set_yticklabels(c1s); ax.set_xlabel("c2 (L2)"); ax.set_ylabel("c1 (L1)")
for i in range(3):
    for j in range(2): ax.text(j, i, "%.3f" % M[i, j], ha="center", va="center", fontsize=8, color="white" if M[i, j] > M.mean() else "black")
plt.colorbar(im); plt.tight_layout(); plt.savefig("figs/exp_grid.pdf"); plt.close()
fig, ax = plt.subplots(figsize=(4.4, 3.8)); cmn = cm / cm.sum(1, keepdims=True)
ax.imshow(cmn, cmap="Blues"); ax.set_xticks(range(4)); ax.set_yticks(range(4)); ax.set_xticklabels(["O", "Comma", "Period", "Quest."]); ax.set_yticklabels(["O", "Comma", "Period", "Quest."])
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
for i in range(4):
    for j in range(4): ax.text(j, i, "%.2f" % cmn[i, j], ha="center", va="center", color="white" if cmn[i, j] > .5 else "black", fontsize=8)
plt.tight_layout(); plt.savefig("figs/exp_confusion.pdf"); plt.close()
fig, ax = plt.subplots(figsize=(4.8, 3.2)); xs = [lc[f]["tokens"] for f in lc]; ys = [lc[f]["macroF1"] for f in lc]
ax.plot(xs, ys, "o-", color="#3b6ea5"); ax.set_xlabel("Training tokens"); ax.set_ylabel("Test macro-F1 (comma, period)"); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig("figs/exp_learning.pdf"); plt.close()
fig, ax = plt.subplots(figsize=(6.4, 3.2)); x = np.arange(3); w = 0.2
for i, (k, lab) in enumerate([("E4_test_majority", "Majority"), ("E4_test_rule", "Rule"), ("E4_test_LR", "Logistic reg."), ("E4_test", "CRF")]):
    ax.bar(x + (i - 1.5) * w, [R[k][c]["F1"] for c in P3], w, label=lab)
ax.set_xticks(x); ax.set_xticklabels(["Comma", "Period", "Question mark"]); ax.set_ylabel("F1 (test)"); ax.legend(fontsize=8); plt.tight_layout(); plt.savefig("figs/exp_models.pdf"); plt.close()
log("done")
