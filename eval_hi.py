#!/usr/bin/env python3
"""
eval_hi.py - evaluate Hindi -> English captions against your own reference, and prepare the human rating sheet.

  python eval_hi.py                                   # BLEU/chrF + aligned sentence pairs + human_eval/hi_sakec_adequacy.csv
  python eval_hi.py --hi-hyp output_hi\\hi_sakec.txt   # also Hindi WER/CER (needs a Hindi transcript, see below)

Hindi transcript for WER (about 4 minutes on CPU):
  python make_cc.py --task transcribe --lang hi --model medium --output output_hi --force
Everything printed comes from the files; nothing is typed in. One clip = no confidence interval.
"""
import argparse, csv, os, re, sys, unicodedata

def read(path):
    return open(path, encoding="utf-8-sig").read()

def sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.?!])\s+", " ".join(text.split())) if s.strip()]

def align(refs, outs, gap=15.0):
    """Monotonic alignment of reference sentences to output sentences by sentence-level chrF (gaps allowed)."""
    import sacrebleu
    n, m = len(refs), len(outs)
    S = [[sacrebleu.sentence_chrf(outs[j], [refs[i]]).score for j in range(m)] for i in range(n)]
    D = [[0.0] * (m + 1) for _ in range(n + 1)]; B = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1): D[i][0], B[i][0] = -gap * i, "up"
    for j in range(1, m + 1): D[0][j], B[0][j] = -gap * j, "left"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            opts = [(D[i - 1][j - 1] + S[i - 1][j - 1], "diag"), (D[i - 1][j] - gap, "up"), (D[i][j - 1] - gap, "left")]
            D[i][j], B[i][j] = max(opts)
    pairs, i, j = [], n, m
    while i > 0 or j > 0:
        mv = B[i][j]
        if mv == "diag": pairs.append((i - 1, j - 1, S[i - 1][j - 1])); i, j = i - 1, j - 1
        elif mv == "up": pairs.append((i - 1, None, 0.0)); i -= 1
        else: pairs.append((None, j - 1, 0.0)); j -= 1
    return pairs[::-1]

def merge_unmatched(refs, outs, pairs):
    """Attach output sentences that have no reference partner (Whisper split one reference sentence into several)
    to the neighbouring reference sentence where they raise the chrF the most. Returns [{'i': ref index or None, 'js': [output indices]}]."""
    import sacrebleu
    chrf = lambda h, r: sacrebleu.sentence_chrf(h, [r]).score
    rows = [{"i": i, "js": [] if j is None else [j]} for i, j, s in pairs]
    out, k = [], 0
    while k < len(rows):
        if rows[k]["i"] is not None:
            out.append(rows[k]); k += 1; continue
        run = []
        while k < len(rows) and rows[k]["i"] is None:
            run += rows[k]["js"]; k += 1
        prev = out[-1] if out and out[-1]["i"] is not None else None
        nxt = rows[k] if k < len(rows) and rows[k]["i"] is not None else None
        def gain(row, before):
            if row is None: return -1e9
            cur = chrf(" ".join(outs[x] for x in row["js"]), refs[row["i"]]) if row["js"] else 0.0
            js = row["js"] + run if before else run + row["js"]
            return chrf(" ".join(outs[x] for x in js), refs[row["i"]]) - cur
        gp, gn = gain(prev, True), gain(nxt, False)
        if prev is not None and gp >= gn: prev["js"] += run
        elif nxt is not None: nxt["js"] = run + nxt["js"]
        else: out.append({"i": None, "js": run})
    return out

PUNCT = re.compile(r"[\u0964\u0965,.?!;:\"'()\[\]\-\u2013\u2014\u2026]")   # includes the danda; matras are kept
def norm_hi(s):
    s = unicodedata.normalize("NFC", s).replace("\u093c", "")                # drop nukta so ज़/ज and ड़/ड match
    return " ".join(PUNCT.sub(" ", s).split())

def hindi_wer(ref, hyp):
    import jiwer
    r, h = norm_hi(ref), norm_hi(hyp)
    out = jiwer.process_words(r, h)
    print("\nHindi ASR (Whisper transcribe vs your Hindi script): WER %.1f%%  CER %.1f%%  (%d ref words)" % (100 * out.wer, 100 * jiwer.cer(r, h), len(r.split())))
    rw, hw, shown = r.split(), h.split(), 0
    print("first errors (ref -> hyp):")
    for ch in out.alignments[0]:
        if ch.type != "equal" and shown < 10:
            print("  %-10s %r -> %r" % (ch.type, " ".join(rw[ch.ref_start_idx:ch.ref_end_idx]), " ".join(hw[ch.hyp_start_idx:ch.hyp_end_idx]))); shown += 1

def main():
    for s in (sys.stdout, sys.stderr):
        try: s.reconfigure(encoding="utf-8", errors="replace")
        except Exception: pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output/hi_sakec.txt"); ap.add_argument("--ref", default="refs/hi_sakec.en.txt")
    ap.add_argument("--hi-ref", default="refs/hi_sakec.txt"); ap.add_argument("--hi-hyp", default=None)
    ap.add_argument("--sheet", default="human_eval/hi_sakec_adequacy.csv")
    a = ap.parse_args()
    import sacrebleu
    out_t, ref_t = read(a.out), read(a.ref)
    h, r = " ".join(out_t.split()), " ".join(ref_t.split())
    print("document-level, single reference: BLEU %.1f  chrF %.1f   (%d output words, %d reference words)"
          % (sacrebleu.corpus_bleu([h], [[r]]).score, sacrebleu.corpus_chrf([h], [[r]]).score, len(h.split()), len(r.split())))
    refs, outs = [l.strip() for l in ref_t.splitlines() if l.strip()], sentences(out_t)
    hi = [l.strip() for l in read(a.hi_ref).splitlines() if l.strip()] if os.path.exists(a.hi_ref) else []
    print("reference sentences: %d | output sentences: %d | Hindi script lines: %d" % (len(refs), len(outs), len(hi)))
    pairs = align(refs, outs)
    merged = merge_unmatched(refs, outs, pairs)
    print("\nOne row per reference sentence (output sentences that Whisper split off are attached to the sentence they belong to):")
    rows, scores = [], []
    for k, m in enumerate(merged, 1):
        o = " ".join(outs[x] for x in m["js"]); i = m["i"]
        s_ = sacrebleu.sentence_chrf(o, [refs[i]]).score if (i is not None and o) else 0.0
        if i is not None: scores.append(s_)
        print("\n[%d]  chrF %.0f%s" % (k, s_, "   (output was split into %d sentences)" % len(m["js"]) if len(m["js"]) > 1 else ""))
        print("  ref: %s" % (refs[i] if i is not None else "(no reference sentence)")); print("  out: %s" % (o or "(no output)"))
        rows.append({"id": k, "hindi_script": hi[i] if i is not None and i < len(hi) else "", "english_reference": refs[i] if i is not None else "", "english_output": o,
                     "sentence_chrF": round(s_, 1), "adequacy_1to5": "", "fluency_1to5": "", "notes": ""})
    print("\nmean sentence chrF over %d reference sentences: %.1f   (lowest: %s)" % (len(scores), sum(scores) / max(len(scores), 1), ", ".join("%.0f" % x for x in sorted(scores)[:4])))
    os.makedirs(os.path.dirname(a.sheet) or ".", exist_ok=True)
    with open(a.sheet, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("rating sheet written: %s  (a Hindi speaker fills adequacy and fluency, 1-5)" % a.sheet)
    if a.hi_hyp and os.path.exists(a.hi_hyp) and hi:
        hindi_wer(" ".join(hi), read(a.hi_hyp))

if __name__ == "__main__":
    main()