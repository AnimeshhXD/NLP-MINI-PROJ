import pickle, collections
from capnlp import *
raw = load_speeches()
docs = {}
for k, t in raw.items():
    toks = to_tokens(clean_raw(t))
    docs[k] = toks
tr, va, te = split_speeches(list(docs))
data = {"train": {k: docs[k] for k in tr}, "val": {k: docs[k] for k in va}, "test": {k: docs[k] for k in te}}
pickle.dump(data, open("data.pkl", "wb"))
for s, d in data.items():
    n = sum(len(v) for v in d.values())
    c = collections.Counter(l for v in d.values() for _, l, _ in v)
    print(s, len(d), "speeches", n, "tokens", dict(c))
k = tr[0]; print(k, docs[k][:25])
print(" ".join(attach_punct([w for w,_,_ in docs[k][:60]],[l for _,l,_ in docs[k][:60]])))
