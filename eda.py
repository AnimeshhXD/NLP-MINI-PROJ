import pickle, json, collections, numpy as np, nltk
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.stem import WordNetLemmatizer
from capnlp import *
plt.rcParams.update({"font.size": 10, "font.family": "serif"})
D = pickle.load(open("data.pkl", "rb"))
tr = [t for d in D["train"].values() for t in d]
alltok = {s: [t for d in D[s].values() for t in d] for s in D}
R = {}
R["speeches"] = {s: len(D[s]) for s in D}
R["tokens"] = {s: len(alltok[s]) for s in D}
# missing / empty
R["empty_tokens"] = sum(1 for t in tr if not t[0])
# vocabulary
vocab = collections.Counter(w.lower() for w, _, _ in tr)
R["vocab_train"] = len(vocab)
lem = WordNetLemmatizer()
R["vocab_lemma"] = len({lem.lemmatize(w) for w in vocab})
te_words = [w.lower() for w, _, _ in alltok["test"]]
R["oov_test_pct"] = 100 * sum(w not in vocab for w in te_words) / len(te_words)
R["hapax_pct"] = 100 * sum(1 for c in vocab.values() if c == 1) / len(vocab)
# sentence lengths (train)
lens, cur = [], 0
for d in D["train"].values():
    cur = 0
    for w, l, _ in d:
        cur += 1
        if l in ("PERIOD", "QUESTION"):
            lens.append(cur); cur = 0
lens = np.array(lens)
R["sent_count_train"] = int(len(lens)); R["sent_mean"] = float(lens.mean()); R["sent_median"] = float(np.median(lens))
R["sent_max"] = int(lens.max()); R["sent_min"] = int(lens.min()); R["sent_p95"] = float(np.percentile(lens, 95))
R["sent_gt40_pct"] = float(100 * (lens > 40).mean())
plt.figure(figsize=(6, 3.4)); plt.hist(np.clip(lens, 0, 100), bins=50, color="#3b6ea5", edgecolor="white")
plt.axvline(lens.mean(), color="crimson", ls="--", label="mean = %.1f" % lens.mean())
plt.xlabel("Words per sentence (clipped at 100)"); plt.ylabel("Sentences"); plt.legend(); plt.tight_layout()
plt.savefig("figs/eda_sentlen.pdf"); plt.close()
# punctuation distribution
cls = {s: collections.Counter(l for _, l, _ in alltok[s]) for s in D}
R["punct_counts"] = {s: dict(cls[s]) for s in D}
fig, ax = plt.subplots(figsize=(6, 3.2))
x = np.arange(3); wdt = 0.27
for i, s in enumerate(["train", "val", "test"]):
    tot = sum(cls[s].values())
    ax.bar(x + (i - 1) * wdt, [100 * cls[s][c] / tot for c in ["COMMA", "PERIOD", "QUESTION"]], wdt, label=s)
ax.set_xticks(x); ax.set_xticklabels(["Comma", "Period", "Question mark"]); ax.set_ylabel("% of tokens"); ax.legend()
plt.tight_layout(); plt.savefig("figs/eda_punct.pdf"); plt.close()
# most frequent tokens
top = vocab.most_common(20); R["top20"] = top
plt.figure(figsize=(6, 3.4)); plt.bar([w for w, _ in top], [c for _, c in top], color="#3b6ea5")
plt.xticks(rotation=60, ha="right"); plt.ylabel("Frequency"); plt.tight_layout(); plt.savefig("figs/eda_top.pdf"); plt.close()
R["top20_share_pct"] = 100 * sum(c for _, c in top) / sum(vocab.values())
# word cloud (content words)
sw = set("the of and to in a that is we our for it be as are have this will by with on has not which from their been more an its at i all but can or they us who was these new must so they there what would more than".split())
wc = WordCloud(width=900, height=450, background_color="white", stopwords=sw, colormap="Blues", random_state=1).generate(" ".join(w.lower() for w, _, _ in tr))
plt.figure(figsize=(6, 3.2)); plt.imshow(wc); plt.axis("off"); plt.tight_layout(pad=0); plt.savefig("figs/eda_wordcloud.pdf"); plt.close()
# n-grams: word before comma / period, bigram before comma
before = {"COMMA": collections.Counter(), "PERIOD": collections.Counter()}
bg = collections.Counter(); after_start = collections.Counter()
for d in D["train"].values():
    ws = [w.lower() for w, _, _ in d]
    for i, (w, l, _) in enumerate(d):
        if l in before: before[l][w.lower()] += 1
        if l == "COMMA" and i > 0: bg[(ws[i-1], ws[i])] += 1
        if i + 1 < len(d) and l == "COMMA": after_start[ws[i+1]] += 1
R["before_comma"] = before["COMMA"].most_common(10); R["before_period"] = before["PERIOD"].most_common(10)
R["bigram_before_comma"] = [(" ".join(k), v) for k, v in bg.most_common(10)]
R["after_comma"] = after_start.most_common(10)
fig, axs = plt.subplots(1, 2, figsize=(7, 3.2))
for ax, (k, t) in zip(axs, [("COMMA", before["COMMA"]), ("PERIOD", before["PERIOD"])]):
    it = t.most_common(10)[::-1]; ax.barh([w for w, _ in it], [c for _, c in it], color="#3b6ea5"); ax.set_title("Word before %s" % k.lower())
plt.tight_layout(); plt.savefig("figs/eda_ngram.pdf"); plt.close()
# how predictable is a comma from the NEXT word?  P(comma | next word is conj)
nxt = collections.Counter(); nxt_c = collections.Counter()
for d in D["train"].values():
    ws = [w.lower() for w, _, _ in d]
    for i in range(len(d) - 1):
        nxt[ws[i+1]] += 1
        if d[i][1] == "COMMA": nxt_c[ws[i+1]] += 1
R["p_comma_before"] = {w: round(nxt_c[w] / nxt[w], 3) for w in ["and", "but", "which", "that", "the", "of"]}
# POS of word before punctuation (sample of 60k tokens)
pos_cnt = {"O": collections.Counter(), "COMMA": collections.Counter(), "PERIOD": collections.Counter()}
sample = []
for d in list(D["train"].values())[:20]: sample += d
sent, sents = [], []
for w, l, c in sample:
    sent.append((w, l))
    if l in ("PERIOD", "QUESTION") or len(sent) >= 60:
        sents.append(sent); sent = []
for s in sents:
    tags = nltk.pos_tag([w for w, _ in s])
    for (w, l), (_, t) in zip(s, tags):
        if l in pos_cnt: pos_cnt[l][t] += 1
R["pos_sample_tokens"] = len(sample)
R["pos_top"] = {k: [(t, round(100 * v / sum(c.values()), 1)) for t, v in c.most_common(5)] for k, c in pos_cnt.items()}
# special characters in raw: counts of tokens with digits / hyphen / apostrophe
R["digit_tokens_pct"] = 100 * sum(any(ch.isdigit() for ch in w) for w, _, _ in tr) / len(tr)
R["hyphen_tokens_pct"] = 100 * sum("-" in w for w, _, _ in tr) / len(tr)
R["apos_tokens_pct"] = 100 * sum("'" in w for w, _, _ in tr) / len(tr)
R["upper_tokens_pct"] = 100 * sum(c == "UPPER" for _, _, c in tr) / len(tr)
R["cap_tokens_pct"] = 100 * sum(c == "CAP" for _, _, c in tr) / len(tr)
# fillers / repeats in clean reference (sanity for disfluency module)
ws = [w.lower() for w, _, _ in tr]
R["fillers_in_ref"] = sum(w in FILLERS for w in ws)
R["repeats_in_ref"] = sum(1 for a, b in zip(ws, ws[1:]) if a == b)
# duplicate speeches
R["dup_speeches"] = len(D["train"]) + len(D["val"]) + len(D["test"]) - len({" ".join(w for w, _, _ in d) for s in D for d in D[s].values()})
json.dump(R, open("results/eda.json", "w"), indent=1)
print(json.dumps(R, indent=1)[:3500])
