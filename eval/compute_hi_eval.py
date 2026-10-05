"""
eval/compute_hi_eval.py — Evaluate Hindi caption pipeline.

Sections:
  a) Hindi ASR quality (Method B only): WER/CER vs refs/hi_<stem>.txt
  b) Translation quality: sacrebleu BLEU + chrF vs refs/hi_<stem>.en.txt
  c) Caption format metrics from captions/hi_<stem>_<method>.srt
  d) Sync: compare caption start times with Whisper segment starts

Skips any metric whose required file is absent.

Usage:
  python eval/compute_hi_eval.py --stem hi_foo
"""
import argparse, json, pathlib, sys, re, unicodedata

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

REF_DIR = pathlib.Path("refs")
ASR_DIR = pathlib.Path("asr")
CAP_DIR = pathlib.Path("captions")
RES_DIR = pathlib.Path("results_new"); RES_DIR.mkdir(exist_ok=True)

MAX_CPS   = 17.0
MAX_CHARS = 42
MAX_LINES = 2


# ── normalisation ──────────────────────────────────────────────────────────────
def _norm_hi(text):
    """Normalise Hindi for WER: strip punctuation/danda, collapse whitespace."""
    # remove danda ।, punctuation, keep Devanagari/space
    text = re.sub(r"[।॥!?,.\-‍‌]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _norm_en(text):
    """Lower-case, strip non-word chars for word list."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


# ── a) Hindi WER/CER ──────────────────────────────────────────────────────────
def eval_hindi_asr(stem):
    ref_path  = REF_DIR / f"{stem}.txt"
    trans_path = ASR_DIR / f"{stem}.transcribe.json"
    if not ref_path.exists():
        print(f"[a] SKIP Hindi ASR WER/CER: refs/{stem}.txt missing")
        return None
    if not trans_path.exists():
        print(f"[a] SKIP Hindi ASR WER/CER: asr/{stem}.transcribe.json missing")
        return None

    from jiwer import wer as compute_wer, cer as compute_cer
    ref_text  = _norm_hi(ref_path.read_text(encoding="utf-8"))
    data      = json.loads(trans_path.read_text(encoding="utf-8"))
    hyp_text  = _norm_hi(" ".join(s["text"] for s in data["segments"]))

    wer_val = compute_wer(ref_text, hyp_text)
    cer_val = compute_cer(ref_text, hyp_text)

    print(f"\n[a] Hindi ASR — WER={wer_val*100:.2f}%  CER={cer_val*100:.2f}%")

    # print every substitution/deletion/insertion
    from jiwer import process_words
    out = process_words(ref_text, hyp_text)
    print(f"    Errors ({len(out.substitutions)} sub, {len(out.deletions)} del, {len(out.insertions)} ins):")
    for chunk in out.alignments[0]:
        if chunk.type != "equal":
            ref_chunk = " ".join(out.references[0][chunk.ref_start_idx:chunk.ref_end_idx])
            hyp_chunk = " ".join(out.hypotheses[0][chunk.hyp_start_idx:chunk.hyp_end_idx])
            print(f"      {chunk.type:5s}  ref={ref_chunk!r}  hyp={hyp_chunk!r}")

    return {
        "wer":           round(wer_val, 4),
        "cer":           round(cer_val, 4),
        "n_ref_words":   len(ref_text.split()),
        "n_hyp_words":   len(hyp_text.split()),
    }


# ── b) Translation BLEU / chrF ─────────────────────────────────────────────────
def eval_translation(stem):
    ref_path = REF_DIR / f"{stem}.en.txt"
    if not ref_path.exists():
        print(f"[b] SKIP translation eval: refs/{stem}.en.txt missing")
        return None

    results = {}
    for method in ("translate", "nllb"):
        asr_path = ASR_DIR / f"{stem}.{'translate' if method=='translate' else 'nllb'}.json"
        if not asr_path.exists():
            print(f"[b] SKIP {method}: {asr_path} missing")
            continue

        data   = json.loads(asr_path.read_text(encoding="utf-8"))
        hyp_en = " ".join(s.get("text", "") for s in data["segments"]).strip()
        ref_en = ref_path.read_text(encoding="utf-8").strip()

        from sacrebleu.metrics import BLEU, CHRF
        bleu = BLEU(effective_order=True).corpus_score([hyp_en], [[ref_en]])
        chrf = CHRF().corpus_score([hyp_en], [[ref_en]])

        print(f"\n[b] {method}: BLEU={bleu.score:.2f}  chrF={chrf.score:.2f}")
        print(f"    {len(hyp_en.split())} hyp words  {len(ref_en.split())} ref words")

        # print 10 aligned segment pairs
        ref_segs = [s.strip() for s in re.split(r"(?<=[.?!])\s+", ref_en) if s.strip()][:20]
        hyp_segs = [s.get("text","").strip() for s in data["segments"]]
        print(f"\n    10 aligned pairs (hypothesis | reference):")
        for i, (h, r) in enumerate(zip(hyp_segs[:10], ref_segs[:10])):
            print(f"    [{i+1}] HYP: {h}")
            print(f"         REF: {r}")

        results[method] = {
            "bleu":      round(bleu.score, 2),
            "chrf":      round(chrf.score, 2),
            "n_hyp_words": len(hyp_en.split()),
            "n_ref_words": len(ref_en.split()),
        }

    return results or None


# ── c) Caption format metrics ──────────────────────────────────────────────────
def eval_caption_format(stem):
    results = {}
    for method in ("translate", "nllb"):
        srt_path = CAP_DIR / f"{stem}_{method}.srt"
        if not srt_path.exists():
            print(f"[c] SKIP format metrics {method}: {srt_path} missing")
            continue

        from tests.test_hi_captions import parse_srt
        blocks = parse_srt(srt_path)
        n = len(blocks)
        from capnlp import FUNCTION_WORDS
        n_ok_len = n_ok_cps = n_fw = 0
        mid_word = 0
        durations = []
        cps_except = []
        for b in blocks:
            dur = b["end"] - b["start"]
            durations.append(dur)
            text = " ".join(b["lines"])
            n_ch = len(text)
            cps  = n_ch / max(dur, 0.01)
            if all(len(l) <= MAX_CHARS for l in b["lines"]) and len(b["lines"]) <= MAX_LINES:
                n_ok_len += 1
            if cps <= MAX_CPS:
                n_ok_cps += 1
            else:
                cps_except.append(f"{cps:.1f} cps: {text[:40]}")
            last_w = b["lines"][-1].rstrip(".,?!").split()[-1].lower() if b["lines"] else ""
            if last_w in FUNCTION_WORDS:
                n_fw += 1
            # mid-word break: check if any line boundary falls inside a word
            for j in range(len(b["lines"]) - 1):
                l1, l2 = b["lines"][j], b["lines"][j + 1]
                if l1 and l2:
                    if not l1[-1] == " " and not l2[0] == " " and l1[-1].isalpha() and l2[0].isalpha():
                        mid_word += 1

        import statistics
        m = {
            "n_blocks":           n,
            "pct_within_2x42":    round(100 * n_ok_len / max(n,1), 1),
            "pct_within_17cps":   round(100 * n_ok_cps / max(n,1), 1),
            "pct_ends_funcword":  round(100 * n_fw / max(n,1), 1),
            "mid_word_breaks":    mid_word,
            "mean_dur_s":         round(statistics.mean(durations), 2),
            "median_dur_s":       round(statistics.median(durations), 2),
            "min_dur_s":          round(min(durations), 2),
            "max_dur_s":          round(max(durations), 2),
        }
        print(f"\n[c] Format metrics ({method}):")
        for k, v in m.items():
            print(f"    {k}: {v}")
        if cps_except:
            print(f"    CPS exceptions: {cps_except[:5]}")
        results[method] = m

    return results or None


# ── d) Sync ────────────────────────────────────────────────────────────────────
def eval_sync(stem):
    results = {}
    for method in ("translate", "nllb"):
        srt_path = CAP_DIR / f"{stem}_{method}.srt"
        asr_key  = "translate" if method == "translate" else "nllb"
        asr_path = ASR_DIR / f"{stem}.{asr_key}.json"
        if not srt_path.exists() or not asr_path.exists():
            print(f"[d] SKIP sync {method}: files missing")
            continue

        from tests.test_hi_captions import parse_srt
        cap_blocks = parse_srt(srt_path)
        data = json.loads(asr_path.read_text(encoding="utf-8"))
        segs = data["segments"]

        # caption start times vs nearest segment start
        offsets = []
        for b in cap_blocks:
            t = b["start"]
            nearest = min(segs, key=lambda s: abs(s["start"] - t))
            offsets.append(abs(nearest["start"] - t))

        import statistics
        m = {
            "n_cap_blocks":     len(cap_blocks),
            "n_asr_segments":   len(segs),
            "mean_abs_offset_s": round(statistics.mean(offsets), 3),
            "median_abs_offset_s": round(statistics.median(offsets), 3),
            "resolution_note":  (
                "Caption start times are proportionally derived from Whisper segment "
                "boundaries; offset = |cap_start - nearest_seg_start|."
            ),
        }
        print(f"\n[d] Sync ({method}): mean offset={m['mean_abs_offset_s']}s  "
              f"median={m['median_abs_offset_s']}s  "
              f"({m['n_cap_blocks']} blocks vs {m['n_asr_segments']} segments)")
        results[method] = m

    return results or None


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", required=True, help="e.g. hi_foo")
    args = ap.parse_args()
    stem = args.stem

    print(f"\n{'='*60}")
    print(f"Evaluation: {stem}")
    print(f"{'='*60}")

    results = {
        "stem": stem,
        "hindi_asr": eval_hindi_asr(stem),
        "translation": eval_translation(stem),
        "caption_format": eval_caption_format(stem),
        "sync": eval_sync(stem),
    }

    out_path = RES_DIR / f"hi_eval_{stem}.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[eval] wrote {out_path}")


if __name__ == "__main__":
    main()
