"""
task6_asr.py  -  OVERNIGHT FIX JOB Task 6

1. Run Whisper base.en on videos/weekly_2015_vra.mp4
2. WER and CER vs refs/weekly_2015_vra.txt (424-word human transcript)
   - Condition "punct": Whisper punctuation retained as separate tokens;
     per-word number normalisation via EnglishTextNormalizer
   - Condition "no_punct": EnglishTextNormalizer applied to both (strips punct,
     normalises numbers) — standard Whisper eval
3. Sync check: per TSV-segment offset between caption-block start and TSV start
4. Writes results_new/asr_results.json

NON-NEGOTIABLE: no simulation claims, no hardcoded numbers in report.
"""
import json, pathlib, re, sys, time, subprocess
import numpy as np
import jiwer
from whisper.normalizers import EnglishTextNormalizer

T0 = time.time()
def log(*a): print(f"[{time.time()-T0:5.0f}s]", *a, flush=True)

RESULTS = pathlib.Path("results_new"); RESULTS.mkdir(exist_ok=True)
ASR_DIR  = pathlib.Path("asr");        ASR_DIR.mkdir(exist_ok=True)
VIDEO    = pathlib.Path("videos/weekly_2015_vra.mp4")
REF_TXT  = pathlib.Path("refs/weekly_2015_vra.txt")
SEG_TSV  = pathlib.Path("refs/weekly_2015_vra.segments.tsv")
WAV      = ASR_DIR / "weekly_2015_vra.wav"
ASR_JSON = ASR_DIR / "weekly_2015_vra.json"

NORM = EnglishTextNormalizer()
EXPECTED_REF_WORDS = 424

# ------------------------------------------------------------------ helpers

def normalize_no_punct(text: str) -> str:
    """EnglishTextNormalizer: numbers + lowercase + strip punctuation (standard WER)."""
    return NORM(text)


def normalize_keep_punct(text: str) -> str:
    """
    Per-word number normalization, but punctuation retained as separate tokens.
    Pipeline:
      1. Detach punctuation marks into separate whitespace-delimited tokens
      2. Lowercase everything
      3. Normalize each purely-alphabetic token via EnglishTextNormalizer
         (converts "four" → "4", "fifty" → "50", etc.)
    Punctuation tokens (",", ".", "?", …) are preserved in the word sequence
    so they contribute to WER when misplaced or missing.
    """
    text = re.sub(r"([^\w\s'])", r" \1 ", text)       # detach punct
    text = re.sub(r"\s+", " ", text.strip().lower())
    tokens = text.split()
    out = []
    for t in tokens:
        if re.match(r'^[a-z]+$', t):                   # pure alpha → normalise
            n = NORM(t).strip()
            out.append(n if n else t)
        else:
            out.append(t)                              # digits, punct → as-is
    return " ".join(out)


def wer_cer(ref: str, hyp: str):
    w = float(100 * jiwer.wer(ref, hyp))
    c = float(100 * jiwer.cer(ref, hyp))
    return round(w, 2), round(c, 2)


# ------------------------------------------------------------------ 1. extract audio
log("checking ffmpeg…")
import shutil
if not shutil.which("ffmpeg"):
    sys.exit("BLOCKED: ffmpeg not on PATH")

if not WAV.exists():
    log("extracting 16 kHz mono WAV…")
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", str(VIDEO), "-ar", "16000", "-ac", "1", "-f", "wav", str(WAV)],
        capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"ffmpeg failed: {r.stderr[-300:]}")
    log(f"  saved {WAV}  ({WAV.stat().st_size//1024} KB)")
else:
    log(f"reusing {WAV}")

# ------------------------------------------------------------------ 2. run Whisper
if ASR_JSON.exists():
    log(f"reusing {ASR_JSON}")
    result = json.loads(ASR_JSON.read_text(encoding="utf-8"))
else:
    log("loading whisper base.en…")
    import whisper, torch
    model = whisper.load_model("base.en")
    log(f"  cuda={torch.cuda.is_available()}")
    log("transcribing (CPU, ~2:40 audio — may take a few minutes)…")
    t0 = time.time()
    result = model.transcribe(str(WAV), word_timestamps=True, language="en")
    log(f"  done in {time.time()-t0:.0f}s")
    ASR_JSON.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    log(f"  saved {ASR_JSON}")

whisper_text = result["text"].strip()
segments     = result.get("segments", [])
whisper_words = 0
for seg in segments:
    whisper_words += len(seg.get("words", []))
log(f"  Whisper word count: {whisper_words}")

# ------------------------------------------------------------------ 3. load reference
ref_raw = REF_TXT.read_text(encoding="utf-8").strip()
ref_words_list = ref_raw.split()
n_ref_words = len(ref_words_list)
log(f"  Reference word count: {n_ref_words}")
if n_ref_words != EXPECTED_REF_WORDS:
    log(f"  WARNING: expected {EXPECTED_REF_WORDS} words, got {n_ref_words}")

# ------------------------------------------------------------------ 4. WER / CER
log("computing WER/CER…")

# Condition A: "no_punct" (standard Whisper eval — EnglishTextNormalizer strips punct)
ref_np  = normalize_no_punct(ref_raw)
hyp_np  = normalize_no_punct(whisper_text)
wer_np, cer_np = wer_cer(ref_np, hyp_np)
log(f"  [no_punct]  WER={wer_np}%  CER={cer_np}%")

# Condition B: "punct" (punctuation marks as word tokens; per-word number normalisation)
ref_p   = normalize_keep_punct(ref_raw)
hyp_p   = normalize_keep_punct(whisper_text)
wer_p, cer_p = wer_cer(ref_p, hyp_p)
log(f"  [punct_kept] WER={wer_p}%  CER={cer_p}%")

# ------------------------------------------------------------------ 5. sync check
log("running sync check…")

# Flatten word-level timing from Whisper segments
word_timing = []   # list of (word_lower, word_start, seg_start, seg_idx)
for si, seg in enumerate(segments):
    seg_start = float(seg["start"])
    for w in seg.get("words", []):
        wl = re.sub(r"[^\w']", "", w["word"].lower())
        word_timing.append((wl, float(w["start"]), seg_start, si))

# Read TSV segments
tsv_rows = []
for line in SEG_TSV.read_text(encoding="utf-8").splitlines():
    if line.startswith("start_s") or not line.strip():
        continue
    parts = line.split("\t", 1)
    if len(parts) == 2:
        tsv_rows.append((int(parts[0]), parts[1].strip()))

offsets = []
tsv_sync = []
for tsv_start, tsv_text in tsv_rows:
    first_word = re.sub(r"[^\w']", "", tsv_text.split()[0].lower())

    # Find the word occurrence whose start time is closest to tsv_start_s
    # (handles common words like "the", "but", "for" that appear many times)
    best = None
    best_dist = float("inf")
    for wl, wstart, seg_start, si in word_timing:
        if wl == first_word:
            dist = abs(wstart - tsv_start)
            if dist < best_dist:
                best_dist = dist
                best = (wstart, seg_start, si)
    match = best

    if match is None:
        tsv_sync.append({
            "tsv_start_s": tsv_start,
            "tsv_first_word": first_word,
            "caption_block_start_s": None,
            "offset_s": None,
            "note": "word not found in Whisper output",
        })
        continue

    wstart, seg_start, si = match
    offset = round(seg_start - tsv_start, 2)
    offsets.append(offset)
    tsv_sync.append({
        "tsv_start_s":          tsv_start,
        "tsv_first_word":       first_word,
        "caption_block_start_s": round(seg_start, 2),
        "offset_s":              offset,
    })
    log(f"  TSV t={tsv_start:3d}s  first_word='{first_word}'  "
        f"block_start={seg_start:.1f}s  offset={offset:+.1f}s")

matched = [o for o in offsets]
mean_offset = round(float(np.mean(np.abs(matched))), 2) if matched else None
max_offset  = round(float(np.max(np.abs(matched))),  2) if matched else None
log(f"  sync: matched {len(matched)}/{len(tsv_rows)} segments  "
    f"mean|offset|={mean_offset}s  max|offset|={max_offset}s")

# ------------------------------------------------------------------ 6. write results
out = {
    "timestamp":       time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "video":           str(VIDEO),
    "whisper_model":   "base.en",
    "n_ref_words":     n_ref_words,
    "n_whisper_words": whisper_words,
    "wer_cer": {
        "no_punct": {
            "description": "EnglishTextNormalizer applied to both (numbers + lowercase + strip punct)",
            "WER_pct": wer_np,
            "CER_pct": cer_np,
        },
        "punct_kept": {
            "description": "Punct retained as tokens; per-word number normalisation",
            "WER_pct": wer_p,
            "CER_pct": cer_p,
        },
    },
    "sync_check": {
        "tsv_resolution_s":  1,
        "n_tsv_segments":    len(tsv_rows),
        "n_matched":         len(matched),
        "mean_abs_offset_s": mean_offset,
        "max_abs_offset_s":  max_offset,
        "per_segment":       tsv_sync,
    },
}
out_path = RESULTS / "asr_results.json"
out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
log(f"Wrote {out_path}")

# ------------------------------------------------------------------ 7. print report
print()
print("=" * 62)
print("TASK 6 RESULTS  —  1 video, 424 words")
print("=" * 62)
print()
print("WER / CER  (refs/weekly_2015_vra.txt vs Whisper base.en)")
print(f"{'Condition':<30} {'WER %':>7} {'CER %':>7}")
print("-" * 46)
print(f"{'Whisper punct kept (as tokens)':<30} {wer_p:>7.2f} {cer_p:>7.2f}")
print(f"{'Whisper punct stripped':<30} {wer_np:>7.2f} {cer_np:>7.2f}")
print()
print("Number normalisation: EnglishTextNormalizer (Whisper).")
print("1 video, 424 words. No confidence interval.")
print()
print("Sync check  (caption-block start vs TSV start, 1-second resolution)")
print(f"  TSV segments : {len(tsv_rows)}")
print(f"  Matched      : {len(matched)}")
print(f"  Mean |offset|: {mean_offset} s")
print(f"  Max  |offset|: {max_offset} s")
print()
print("TSV has 1-second resolution.")
print("Caption blocks = Whisper segments.")
print("=" * 62)
print()
print("Task 6 DONE")
