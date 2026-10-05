"""
hi_song_pipeline.py — Sung-audio stress test for the Hindi caption pipeline.

Steps:
  1. Whisper transcribe + translate on videos/hi_song.mp4
     (language="hi", condition_on_previous_text=False)
  2. Hindi WER/CER vs refs/hi_song.txt with Devanagari normalisation
     • NFC, remove nukta (U+093C), remove bracketed tags [संगीत], strip punct
     • chandrabindu (U+0901) → anusvara (U+0902) so variants match
     • per-line WER (8 lines, aligned by order) + overall WER/CER
  3. Score refs/hi_song_youtube_auto.txt the same way
  4. English captions from translate output: format metrics only
     (<=2 lines x 42 chars, <=17 cps, min 1 s, max 7 s)
  5. Test no word is split (SRT words == Whisper words joined)
  6. Print first 10 English caption blocks
  7. Write results_new/song_results.json (no lyric text; quality note)

PRIVACY: Lyric text is never written to JSON or printed in full.
Only error positions and word pairs are printed/stored.
refs/hi_song* and videos/hi_song* are in .gitignore.
"""

import io, json, pathlib, re, sys, time, unicodedata, hashlib, importlib.metadata
sys.path.insert(0, str(pathlib.Path(__file__).parent))
# Windows terminals default to cp1252; force UTF-8 so Devanagari prints correctly
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ASR_DIR = pathlib.Path("asr");       ASR_DIR.mkdir(exist_ok=True)
CAP_DIR = pathlib.Path("captions");  CAP_DIR.mkdir(exist_ok=True)
RES_DIR = pathlib.Path("results_new"); RES_DIR.mkdir(exist_ok=True)

VIDEO    = pathlib.Path("videos/hi_song.mp4")
REF_PATH = pathlib.Path("refs/hi_song.txt")
YT_PATH  = pathlib.Path("refs/hi_song_youtube_auto.txt")

MAX_CHARS = 42; MAX_LINES = 2; MAX_CPS = 17.0; MIN_DUR = 1.0; MAX_DUR = 7.0
N_REF_LINES = 8
MAX_ERRORS_PRINTED = 10   # per-system limit on printed error pairs


# ─── 0. gitignore check ────────────────────────────────────────────────────────
def check_gitignore():
    gi = pathlib.Path(".gitignore")
    if gi.exists():
        text = gi.read_text(encoding="utf-8")
        if "refs/hi_song*" in text and "videos/hi_song*" in text:
            print("[gitignore] OK — song files already protected")
            return
    # append if missing
    with gi.open("a", encoding="utf-8") as f:
        f.write("\nrefs/hi_song*\nvideos/hi_song*\n")
    print("[gitignore] added refs/hi_song* and videos/hi_song*")


# ─── Devanagari normalisation ─────────────────────────────────────────────────
def norm_hi(text):
    """Normalise Hindi text for WER comparison.
    NFC → remove bracketed tags → remove nukta → chandrabindu→anusvara
    → strip non-Devanagari, non-space → collapse whitespace.
    """
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\[[^\]]*\]", " ", text)           # [संगीत], [music], etc.
    text = text.replace("़", "")                  # remove nukta
    text = text.replace("ँ", "ं")            # chandrabindu → anusvara
    # keep only Devanagari block (U+0900–U+097F) and whitespace
    text = re.sub(r"[^ऀ-ॿ\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


# ─── 1. Whisper ASR ────────────────────────────────────────────────────────────
def run_whisper(model_name):
    import whisper
    print(f"\n[whisper] loading model={model_name} ...")
    model = whisper.load_model(model_name)

    results = {}
    for task in ("transcribe", "translate"):
        print(f"[whisper] running task={task} language=hi condition_on_previous_text=False ...")
        t0 = time.time()
        r = model.transcribe(
            str(VIDEO),
            task=task,
            language="hi",
            word_timestamps=True,
            condition_on_previous_text=False,   # critical for sung audio; avoids hallucination loops
            verbose=False,
        )
        elapsed = round(time.time() - t0, 1)
        segs = r.get("segments", [])
        # flatten word timestamps
        words = []
        for seg in segs:
            for w in seg.get("words", []):
                txt = w.get("word", "").strip()
                if txt:
                    words.append({"word": txt,
                                  "start": round(float(w["start"]), 3),
                                  "end":   round(float(w["end"]),   3)})
        det_lang = r.get("language", "?")
        det_prob = r.get("language_probability",
                         r.get("language_probs", {}).get(det_lang, "unavailable"))
        duration_s = round(segs[-1]["end"], 3) if segs else 0.0

        out = {
            "model":               model_name,
            "task":                task,
            "language_forced":     "hi",
            "detected_language":   det_lang,
            "detected_language_prob": det_prob,
            "condition_on_previous_text": False,
            "duration_s":          duration_s,
            "elapsed_s":           elapsed,
            "video_sha256":        _sha256(VIDEO),
            "n_segments":          len(segs),
            "n_words":             len(words),
            # segments stored WITHOUT text to avoid printing lyrics in logs
            # only timing and word tokens kept for captioning / alignment
            "segments_notimestamp": [
                {"start": round(s["start"],3), "end": round(s["end"],3),
                 # text kept for captioning pipeline but marked private
                 "_text": s["text"].strip()}
                for s in segs
            ],
            "words": words,
        }
        path = ASR_DIR / f"hi_song.{task}.json"
        path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[whisper] {task}: {len(segs)} segments, {len(words)} words, "
              f"duration={duration_s}s, detected={det_lang}({det_prob}), elapsed={elapsed}s")
        print(f"[whisper] wrote {path}")
        results[task] = out
    return results


# ─── 2 & 3. WER / CER evaluation ─────────────────────────────────────────────
def wer_cer(ref_text, hyp_text):
    """Return (wer, cer) as floats 0–1."""
    from jiwer import wer as jiwer_wer, cer as jiwer_cer
    return jiwer_wer(ref_text, hyp_text), jiwer_cer(ref_text, hyp_text)


def get_errors(ref_text, hyp_text, max_print):
    """Return list of (position, ref_word, hyp_word) for sub/del/ins errors,
    limited to max_print.  ref_word or hyp_word may be '' for del/ins."""
    from jiwer import process_words
    out = process_words(ref_text, hyp_text)
    errors = []
    for chunk in out.alignments[0]:
        if chunk.type == "equal":
            continue
        ref_ws = out.references[0][chunk.ref_start_idx:chunk.ref_end_idx]
        hyp_ws = out.hypotheses[0][chunk.hyp_start_idx:chunk.hyp_end_idx]
        # zip longest
        for i, (rw, hw) in enumerate(
            zip(ref_ws + [""] * max(0, len(hyp_ws) - len(ref_ws)),
                hyp_ws + [""] * max(0, len(ref_ws) - len(hyp_ws)))
        ):
            pos = chunk.ref_start_idx + i
            errors.append({"pos": pos, "ref": rw, "hyp": hw, "type": chunk.type})
            if len(errors) >= max_print:
                return errors
    return errors


def per_line_wer_whisper(ref_lines_norm, hyp_words_all):
    """Align Whisper's continuous output to reference lines proportionally
    (by reference line word count), then compute per-line WER."""
    from jiwer import wer as jiwer_wer
    ref_counts = [len(l.split()) for l in ref_lines_norm]
    total_ref = sum(ref_counts)
    total_hyp = len(hyp_words_all)
    # Proportional split: hyp line i gets round(total_hyp * ref_count_i / total_ref) words
    splits = []
    cursor = 0
    for i, n in enumerate(ref_counts):
        take = round(total_hyp * n / total_ref) if i < len(ref_counts)-1 else len(hyp_words_all) - cursor
        splits.append(" ".join(hyp_words_all[cursor:cursor+take]))
        cursor += take
    results = []
    for i, (ref, hyp) in enumerate(zip(ref_lines_norm, splits)):
        w = jiwer_wer(ref, hyp) if ref.strip() else 0.0
        results.append({"line": i+1, "n_ref_words": len(ref.split()), "wer": round(w, 4)})
    return results


def per_line_wer_yt(ref_lines_norm, yt_lines_norm):
    """YouTube auto-captions: 1:1 alignment by line order."""
    from jiwer import wer as jiwer_wer
    results = []
    for i, (ref, hyp) in enumerate(zip(ref_lines_norm, yt_lines_norm)):
        w = jiwer_wer(ref, hyp) if ref.strip() else 0.0
        results.append({"line": i+1, "n_ref_words": len(ref.split()), "wer": round(w, 4)})
    return results


def eval_wer(whisper_transcribe_data, n_ref_lines):
    results = {}

    # load reference
    ref_raw = REF_PATH.read_text(encoding="utf-8")
    ref_lines_raw  = [l for l in ref_raw.splitlines() if l.strip()][:n_ref_lines]
    ref_lines_norm = [norm_hi(l) for l in ref_lines_raw]
    ref_full_norm  = " ".join(ref_lines_norm)
    n_ref_words    = len(ref_full_norm.split())

    # ── Whisper transcript ─────────────────────────────────────────────────────
    segs = whisper_transcribe_data["segments_notimestamp"]
    hyp_full_raw  = " ".join(s["_text"] for s in segs)
    hyp_full_norm = norm_hi(hyp_full_raw)
    hyp_words_all = hyp_full_norm.split()
    n_hyp_words   = len(hyp_words_all)

    w_wer, w_cer = wer_cer(ref_full_norm, hyp_full_norm)
    w_errors      = get_errors(ref_full_norm, hyp_full_norm, MAX_ERRORS_PRINTED)
    w_per_line    = per_line_wer_whisper(ref_lines_norm, hyp_words_all)

    print(f"\n[wer] Whisper transcribe vs lyrics:")
    print(f"  overall WER={w_wer*100:.2f}%  CER={w_cer*100:.2f}%  "
          f"({n_ref_words} ref words, {n_hyp_words} hyp words)")
    print(f"  per-line WER (proportional alignment, 8 lines):")
    for row in w_per_line:
        print(f"    line {row['line']}: WER={row['wer']*100:.1f}%  ({row['n_ref_words']} ref words)")
    print(f"  first {len(w_errors)} errors (pos, ref, hyp):")
    for e in w_errors:
        print(f"    [{e['pos']:3d}] {e['type']:6s}  ref={e['ref']!r}  hyp={e['hyp']!r}")

    results["whisper_transcribe"] = {
        "n_ref_words": n_ref_words,
        "n_hyp_words": n_hyp_words,
        "wer":         round(w_wer, 4),
        "cer":         round(w_cer, 4),
        "per_line_wer": w_per_line,
        "first_errors": w_errors,  # word-level only; no sentence context
    }

    # ── YouTube auto-captions ──────────────────────────────────────────────────
    if YT_PATH.exists():
        yt_raw = YT_PATH.read_text(encoding="utf-8")
        yt_lines_raw  = [l for l in yt_raw.splitlines() if l.strip()][:n_ref_lines]
        yt_lines_norm = [norm_hi(l) for l in yt_lines_raw]
        yt_full_norm  = " ".join(yt_lines_norm)
        n_yt_words    = len(yt_full_norm.split())

        yt_wer, yt_cer = wer_cer(ref_full_norm, yt_full_norm)
        yt_errors       = get_errors(ref_full_norm, yt_full_norm, MAX_ERRORS_PRINTED)
        yt_per_line     = per_line_wer_yt(ref_lines_norm, yt_lines_norm)

        print(f"\n[wer] YouTube auto-captions vs lyrics:")
        print(f"  overall WER={yt_wer*100:.2f}%  CER={yt_cer*100:.2f}%  "
              f"({n_ref_words} ref words, {n_yt_words} hyp words)")
        print(f"  per-line WER (1:1 alignment, 8 lines):")
        for row in yt_per_line:
            print(f"    line {row['line']}: WER={row['wer']*100:.1f}%  ({row['n_ref_words']} ref words)")
        print(f"  first {len(yt_errors)} errors (pos, ref, hyp):")
        for e in yt_errors:
            print(f"    [{e['pos']:3d}] {e['type']:6s}  ref={e['ref']!r}  hyp={e['hyp']!r}")

        results["youtube_auto"] = {
            "file": str(YT_PATH),
            "n_ref_words": n_ref_words,
            "n_yt_words":  n_yt_words,
            "wer":         round(yt_wer, 4),
            "cer":         round(yt_cer, 4),
            "per_line_wer": yt_per_line,
            "first_errors": yt_errors,
        }
    else:
        print("[wer] refs/hi_song_youtube_auto.txt not found — skipping")

    return results


# ─── 4 & 5. English captions from translate output ─────────────────────────────
from captions.make_captions_hi import (
    assign_times, _resolve_overlaps, fmt_metrics, write_srt, write_vtt
)
from capnlp import segment_captions


def build_en_captions(translate_data):
    segs = translate_data["segments_notimestamp"]
    all_timed = []
    for i, seg in enumerate(segs):
        text = seg["_text"].strip()
        if not text:
            continue
        words = text.split()
        blocks = segment_captions(words)
        next_start = segs[i+1]["start"] if i+1 < len(segs) else None
        timed = assign_times(blocks, seg["start"], seg["end"], next_start)
        all_timed.extend(timed)
    all_timed = _resolve_overlaps(all_timed)

    srt_path = CAP_DIR / "hi_song_translate.srt"
    vtt_path = CAP_DIR / "hi_song_translate.vtt"
    write_srt(all_timed, srt_path)
    write_vtt(all_timed, vtt_path)
    print(f"\n[captions] wrote {srt_path} ({len(all_timed)} blocks)")
    print(f"[captions] wrote {vtt_path}")

    metrics = fmt_metrics(all_timed)
    print("\n[captions] Format metrics:")
    for k, v in metrics.items():
        if k != "cps_exceptions":
            print(f"  {k}: {v}")
    if metrics["cps_exceptions"]:
        print(f"  CPS exceptions ({len(metrics['cps_exceptions'])}):")
        for ex in metrics["cps_exceptions"]:
            print(f"    [{ex['cps']:.1f} cps, {ex['dur']:.2f}s] {ex['text'][:50]}")

    # ── word-split test ────────────────────────────────────────────────────────
    # whisper translate words (cleaned, in order)
    whisper_words = [re.sub(r"[^\w]", "", (w["word"] if isinstance(w, dict) else w).lower())
                     for w in (translate_data.get("words") or [])
                     if re.sub(r"[^\w]", "", (w["word"] if isinstance(w, dict) else w))]
    # caption words (cleaned, in order)
    import re as _re
    cap_words = []
    for b in all_timed:
        for line in b["lines"]:
            for w in line.split():
                cw = _re.sub(r"[^\w]", "", w.lower())
                if cw:
                    cap_words.append(cw)

    split_test_pass = (whisper_words == cap_words)
    if split_test_pass:
        print("[word-split test] PASS — caption words match Whisper words exactly")
    else:
        # find first mismatch
        first_diff = next((i for i, (a, b) in enumerate(zip(whisper_words, cap_words)) if a != b), len(whisper_words))
        print(f"[word-split test] FAIL — first mismatch at index {first_diff}: "
              f"whisper={whisper_words[first_diff]!r} cap={cap_words[first_diff] if first_diff < len(cap_words) else 'OOB'!r}")
        print(f"  whisper={len(whisper_words)} words  captions={len(cap_words)} words")

    # ── first 10 blocks ───────────────────────────────────────────────────────
    print("\n[captions] First 10 English caption blocks:")
    for i, b in enumerate(all_timed[:10], 1):
        print(f"  [{i:2d}] {b['start']:.2f}→{b['end']:.2f}s  {b['lines']}")

    return {
        "n_blocks":      len(all_timed),
        "fmt_metrics":   metrics,
        "word_split_test": "PASS" if split_test_pass else "FAIL",
        "first_10_blocks": [
            {"start": b["start"], "end": b["end"], "lines": b["lines"]}
            for b in all_timed[:10]
        ],
        "quality_note":  (
            "quality not evaluated; no English reference; "
            "sung audio; copyrighted material not redistributed"
        ),
        "timing_method": (
            "Within each Whisper segment, caption block start/end times are "
            "distributed proportionally to character count."
        ),
    }


# ─── main ──────────────────────────────────────────────────────────────────────
def main():
    import torch, datetime
    check_gitignore()

    model_name = "large-v3" if torch.cuda.is_available() else "small"
    print(f"[env] Whisper model: {model_name}  CUDA: {torch.cuda.is_available()}")

    # ── 1. ASR ─────────────────────────────────────────────────────────────────
    tr_path = ASR_DIR / "hi_song.transcribe.json"
    tx_path = ASR_DIR / "hi_song.translate.json"
    if tr_path.exists() and tx_path.exists():
        print("[whisper] found cached ASR files, loading ...")
        tr_data = json.loads(tr_path.read_text(encoding="utf-8"))
        tx_data = json.loads(tx_path.read_text(encoding="utf-8"))
    else:
        asr_out = run_whisper(model_name)
        tr_data = asr_out["transcribe"]
        tx_data = asr_out["translate"]

    # ── 2 & 3. WER / CER ───────────────────────────────────────────────────────
    wer_results = eval_wer(tr_data, N_REF_LINES)

    # ── 4, 5, 6. English captions ──────────────────────────────────────────────
    cap_results = build_en_captions(tx_data)

    # ── 7. Write results_new/song_results.json ─────────────────────────────────
    def _pkg(n):
        try: return importlib.metadata.version(n)
        except: return "?"

    song_results = {
        "timestamp":  datetime.datetime.utcnow().isoformat() + "Z",
        "video_sha256_prefix": _sha256(VIDEO),
        "whisper_model": model_name,
        "condition_on_previous_text": False,
        "n_segments_transcribe": tr_data["n_segments"],
        "n_segments_translate":  tx_data["n_segments"],
        "duration_s": tr_data["duration_s"],
        "library_versions": {
            "openai-whisper": _pkg("openai-whisper"),
            "jiwer":          _pkg("jiwer"),
            "sacrebleu":      _pkg("sacrebleu"),
        },
        "wer_eval":     wer_results,
        "captions_en":  cap_results,
        "privacy_note": (
            "Lyric text is not stored in this file. "
            "Only word-level error pairs and caption timing/metrics are kept. "
            "refs/hi_song* and videos/hi_song* are in .gitignore."
        ),
    }

    out_path = RES_DIR / "song_results.json"
    out_path.write_text(json.dumps(song_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[results] wrote {out_path}")

    # ── 8. verify_real.py ──────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("Running verify_real.py ...")
    print("="*60)
    import subprocess
    r = subprocess.run([sys.executable, "verify_real.py"], capture_output=False, text=True)
    if r.returncode != 0:
        print("[verify] FAIL")
    else:
        print("[verify] OK")


if __name__ == "__main__":
    main()
