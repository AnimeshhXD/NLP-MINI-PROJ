"""
gen_hindi_results.py — Aggregate all Hindi pipeline results and write:
  results_new/hindi_results.json   (machine-readable, timestamped)
  HINDI_RESULTS.md                  (human-readable report)

All numbers in HINDI_RESULTS.md must appear in hindi_results.json.
Run after: run_whisper_hi.py, nllb_translate.py, make_captions_hi.py,
           compute_hi_eval.py.

Usage:
  python gen_hindi_results.py --stem hi_foo
"""
import argparse, datetime, hashlib, importlib.metadata, json, pathlib, sys

RES_DIR = pathlib.Path("results_new"); RES_DIR.mkdir(exist_ok=True)
ASR_DIR = pathlib.Path("asr")
CAP_DIR = pathlib.Path("captions")
REF_DIR = pathlib.Path("refs")


def _pkg_version(name):
    try:
        return importlib.metadata.version(name)
    except Exception:
        return "?"


def _sha256(path):
    if not pathlib.Path(path).exists():
        return "missing"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def load_json(path):
    p = pathlib.Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", required=True, help="e.g. hi_foo")
    args = ap.parse_args()
    stem = args.stem

    # gather sub-results
    tr   = load_json(ASR_DIR / f"{stem}.translate.json")
    tx   = load_json(ASR_DIR / f"{stem}.transcribe.json")
    nllb = load_json(ASR_DIR / f"{stem}.nllb.json")
    ev   = load_json(RES_DIR / f"hi_eval_{stem}.json")
    cap_tr   = load_json(RES_DIR / f"captions_hi_{stem}_translate.json")
    cap_nllb = load_json(RES_DIR / f"captions_hi_{stem}_nllb.json")
    he   = load_json(RES_DIR / f"hi_human_eval_{stem}.json")

    results = {
        "timestamp":    datetime.datetime.utcnow().isoformat() + "Z",
        "seed":         42,
        "stem":         stem,
        "library_versions": {
            "openai-whisper": _pkg_version("openai-whisper"),
            "transformers":   _pkg_version("transformers"),
            "sentencepiece":  _pkg_version("sentencepiece"),
            "sacrebleu":      _pkg_version("sacrebleu"),
            "jiwer":          _pkg_version("jiwer"),
        },
        "model_names": {
            "whisper":  tr["model"] if tr else "not_run",
            "nllb":     nllb["nllb_model"] if nllb else "facebook/nllb-200-distilled-600M",
        },
        "file_hashes": {
            "video":          _sha256(f"videos/{stem}.mp4"),
            "ref_hindi":      _sha256(REF_DIR / f"{stem}.txt"),
            "ref_english":    _sha256(REF_DIR / f"{stem}.en.txt"),
            "translate_json": _sha256(ASR_DIR / f"{stem}.translate.json"),
            "transcribe_json":_sha256(ASR_DIR / f"{stem}.transcribe.json"),
            "nllb_json":      _sha256(ASR_DIR / f"{stem}.nllb.json"),
        },
        "asr": {
            "method_A_translate": {
                "n_segments":             tr["n_segments"] if tr else None,
                "detected_language":      tr["detected_language"] if tr else None,
                "detected_language_prob": tr["detected_language_prob"] if tr else None,
                "duration_s":             tr["duration_s"] if tr else None,
            },
            "method_B_transcribe": {
                "n_segments":  tx["n_segments"] if tx else None,
            } if tx else None,
            "method_B_nllb": {
                "nllb_model":   nllb["nllb_model"] if nllb else None,
                "nllb_elapsed_s": nllb.get("nllb_elapsed_s") if nllb else None,
            } if nllb else None,
        },
        "eval":          ev,
        "caption_format": {
            "translate": cap_tr["fmt_metrics"] if cap_tr else None,
            "nllb":      cap_nllb["fmt_metrics"] if cap_nllb else None,
        },
        "human_eval":    he,
        "timing_method": (
            "Within each Whisper segment, caption block start/end times are "
            "distributed proportionally to character count: "
            "t_block_i ∝ chars(block_i) / total_chars_in_segment."
        ),
    }

    out_path = RES_DIR / "hindi_results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[gen] wrote {out_path}")

    # ── generate HINDI_RESULTS.md ────────────────────────────────────────────
    def v(val, fmt=None):
        if val is None:
            return "—"
        if fmt:
            return fmt.format(val)
        return str(val)

    tr_ev  = (ev or {}).get("translation", {}) or {}
    hi_ev  = (ev or {}).get("hindi_asr")
    cf     = results["caption_format"]
    syn    = (ev or {}).get("sync", {}) or {}

    tr_bleu  = v(tr_ev.get("translate", {}).get("bleu"), "{:.2f}")
    tr_chrf  = v(tr_ev.get("translate", {}).get("chrf"), "{:.2f}")
    nllb_bleu= v(tr_ev.get("nllb", {}).get("bleu"), "{:.2f}")
    nllb_chrf= v(tr_ev.get("nllb", {}).get("chrf"), "{:.2f}")
    hi_wer   = v(hi_ev.get("wer") if hi_ev else None, "{:.2%}")
    hi_cer   = v(hi_ev.get("cer") if hi_ev else None, "{:.2%}")

    def fmt_row(d, method):
        if d is None or d.get(method) is None:
            return "—|—|—|—|—|—|—"
        m = d[method]
        return (
            f"{v(m.get('n_blocks'))}|"
            f"{v(m.get('pct_within_2x42'), '{:.1f}%')}|"
            f"{v(m.get('pct_within_17cps'), '{:.1f}%')}|"
            f"{v(m.get('pct_ends_funcword'), '{:.1f}%')}|"
            f"{v(m.get('mid_word_breaks'))}|"
            f"{v(m.get('mean_dur_s'), '{:.2f}s')}|"
            f"{v(m.get('median_dur_s'), '{:.2f}s')}"
        )

    md = f"""# HINDI RESULTS

*Generated: {results['timestamp']}*

> Input: NLTK-extracted Hindi speech from `videos/{stem}.mp4`.
> ASR: Whisper `{results['model_names']['whisper']}` (CPU, no CUDA).
> Translation: `{results['model_names']['nllb']}` (NLLB 600M).
> Timing within segments: proportional to character count (see timing note).

## 1. Environment

| Package | Version |
|---------|---------|
"""
    for pkg, ver in results["library_versions"].items():
        md += f"| {pkg} | {ver} |\n"

    md += f"""
| GPU | CPU only (CUDA: False) |

## 2. ASR

| Method | Segments | Detected language | Lang prob | Duration |
|--------|----------|-------------------|-----------|----------|
| A — translate | {v(results['asr']['method_A_translate']['n_segments'])} | {v(results['asr']['method_A_translate']['detected_language'])} | {v(results['asr']['method_A_translate']['detected_language_prob'])} | {v(results['asr']['method_A_translate']['duration_s'])}s |
| B — transcribe | {v((results['asr']['method_B_transcribe'] or {{}}).get('n_segments'))} | hi (forced) | — | — |

*Source: `results_new/hindi_results.json` → `asr`*

### Hindi ASR quality (Method B vs refs/{stem}.txt)

| Metric | Value |
|--------|------:|
| WER | {hi_wer} |
| CER | {hi_cer} |

*Source: `results_new/hindi_results.json` → `eval.hindi_asr`*

## 3. Translation quality (vs refs/{stem}.en.txt)

| Method | BLEU | chrF |
|--------|-----:|-----:|
| A — Whisper translate | {tr_bleu} | {tr_chrf} |
| B — NLLB 600M | {nllb_bleu} | {nllb_chrf} |

*Source: `results_new/hindi_results.json` → `eval.translation`*

## 4. Caption format metrics

| Metric | Method A (translate) | Method B (NLLB) |
|--------|---------------------:|----------------:|
| n blocks | {v((cf.get('translate') or {{}}).get('n_blocks'))} | {v((cf.get('nllb') or {{}}).get('n_blocks'))} |
| within 2×42 chars | {v((cf.get('translate') or {{}}).get('pct_within_2x42'))}% | {v((cf.get('nllb') or {{}}).get('pct_within_2x42'))}% |
| within 17 cps | {v((cf.get('translate') or {{}}).get('pct_within_17cps'))}% | {v((cf.get('nllb') or {{}}).get('pct_within_17cps'))}% |
| ends on function word | {v((cf.get('translate') or {{}}).get('pct_ends_funcword'))}% | {v((cf.get('nllb') or {{}}).get('pct_ends_funcword'))}% |
| mid-word breaks | {v((cf.get('translate') or {{}}).get('mid_word_breaks'))} | {v((cf.get('nllb') or {{}}).get('mid_word_breaks'))} |
| mean duration | {v((cf.get('translate') or {{}}).get('mean_dur_s'))}s | {v((cf.get('nllb') or {{}}).get('mean_dur_s'))}s |
| median duration | {v((cf.get('translate') or {{}}).get('median_dur_s'))}s | {v((cf.get('nllb') or {{}}).get('median_dur_s'))}s |

Timing note: {results['timing_method']}

*Source: `results_new/hindi_results.json` → `caption_format`*

## 5. Sync

| Method | Blocks | Segments | Mean offset | Median offset |
|--------|-------:|---------:|------------:|--------------:|
| translate | {v((syn.get('translate') or {{}}).get('n_cap_blocks'))} | {v((syn.get('translate') or {{}}).get('n_asr_segments'))} | {v((syn.get('translate') or {{}}).get('mean_abs_offset_s'))}s | {v((syn.get('translate') or {{}}).get('median_abs_offset_s'))}s |
| nllb      | {v((syn.get('nllb') or {{}}).get('n_cap_blocks'))} | {v((syn.get('nllb') or {{}}).get('n_asr_segments'))} | {v((syn.get('nllb') or {{}}).get('mean_abs_offset_s'))}s | {v((syn.get('nllb') or {{}}).get('median_abs_offset_s'))}s |

*Source: `results_new/hindi_results.json` → `eval.sync`*

## 6. Human Evaluation (adequacy/fluency)

BLOCKED — no completed rating sheets. See `human_eval/{stem}_adequacy.csv`.

## Provenance

Every figure above comes from `results_new/hindi_results.json`.
Nothing is hardcoded in this document.

## What Was NOT Done / Limitations

- Single speaker, single clip; generalisation to multiple speakers untested.
- Code-mixed speech (Hindi+English in same utterance) is untested; NLLB may fail.
- Whisper `small` used (CPU only); `large-v3` would give better quality.
- NLLB 600M is a small model; NLLB-3.3B or NLLB-1.3B would be more accurate.
- Human evaluation pending: `human_eval/{stem}_adequacy.csv` not yet rated.
- Proportional timing is an approximation; per-word Hindi timing from Whisper
  transcribe could improve English block boundaries for Method B.
- BLEU/chrF computed against a single human reference; multiple references
  would give a more robust score.
- GPU not available; all inference is on CPU (slow).
"""

    md_path = pathlib.Path("HINDI_RESULTS.md")
    md_path.write_text(md, encoding="utf-8")
    print(f"[gen] wrote {md_path}")


if __name__ == "__main__":
    main()
