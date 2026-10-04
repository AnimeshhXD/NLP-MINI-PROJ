"""
asr/run_whisper.py  -  Step 2: real Whisper ASR on video/audio files.

Requires:
  - ffmpeg on PATH (for audio extraction)
  - video/audio files in videos/

With videos present:
  1. ffmpeg extracts 16 kHz mono WAV
  2. Whisper base with word_timestamps=True transcribes
  3. asr/<stem>.json  saved (full result incl. word timestamps)
  4. WER/CER vs hand-corrected refs/<stem>.txt (if present)
  5. results_new/asr_results.json written with provenance

If blocked: exits with non-zero code and writes BLOCKED.md entry.
"""
import sys, json, pathlib, time, subprocess, shutil, re, hashlib
import jiwer

T0 = time.time()
def log(*a): print(f"[{time.time()-T0:5.0f}s]", *a, flush=True)

VIDEOS_DIR  = pathlib.Path("videos")
ASR_DIR     = pathlib.Path("asr");        ASR_DIR.mkdir(exist_ok=True)
REFS_DIR    = pathlib.Path("refs");       REFS_DIR.mkdir(exist_ok=True)
RESULTS_DIR = pathlib.Path("results_new"); RESULTS_DIR.mkdir(exist_ok=True)

VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v",
              ".mp3", ".wav", ".flac"}

BLOCKED_MD = pathlib.Path("BLOCKED.md")


def _append_blocked(section_title: str, body: str):
    existing = BLOCKED_MD.read_text() if BLOCKED_MD.exists() else ""
    if section_title not in existing:
        with BLOCKED_MD.open("a") as f:
            f.write(f"\n---\n\n## {section_title}\n\n{body}\n")


# ---- check ffmpeg -------------------------------------------------------
if not shutil.which("ffmpeg"):
    msg = ("**Exact error:** `ffmpeg` not found on PATH.\n\n"
           "**Fix:**\n"
           "1. Download from https://www.gyan.dev/ffmpeg/builds/ "
           "(Windows: take the `full` release zip).\n"
           "2. Extract and add the `bin/` folder to your system PATH.\n"
           "3. Open a new terminal and verify: `ffmpeg -version`\n"
           "4. Re-run `python asr/run_whisper.py`\n")
    _append_blocked("Step 2 – ASR: ffmpeg not on PATH", msg)
    print("BLOCKED: ffmpeg not found. See BLOCKED.md.")
    sys.exit(1)

# ---- check video files --------------------------------------------------
video_files = sorted([f for f in VIDEOS_DIR.iterdir()
                       if f.suffix.lower() in VIDEO_EXTS]) \
              if VIDEOS_DIR.exists() else []

if not video_files:
    msg = ("**Exact error:** `videos/` directory is empty.\n\n"
           "**Fix:**\n"
           "1. Add 3–5 short (2–6 min) openly-licensed talks to `videos/`.\n"
           "   Example: download a Creative Commons TED talk with yt-dlp:\n"
           "   ```\n"
           "   pip install yt-dlp\n"
           "   yt-dlp -x --audio-format wav "
           "-o 'videos/%(title)s.%(ext)s' <CC YouTube URL>\n"
           "   ```\n"
           "2. Record the licence in `videos/SOURCES.md`.\n"
           "3. Re-run `python asr/run_whisper.py`\n")
    _append_blocked("Step 2 – ASR: no video files in videos/", msg)
    print("BLOCKED: no video files in videos/. See BLOCKED.md.")
    sys.exit(0)

# ---- load Whisper -------------------------------------------------------
log("loading whisper-base (CPU)")
import whisper
model = whisper.load_model("base")
import torch
log(f"whisper loaded, cuda={torch.cuda.is_available()}")

R = {
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "seed": 42,
    "whisper_model": "base",
    "cuda": torch.cuda.is_available(),
    "files": {},
}

# ---- process each video -------------------------------------------------
for vf in video_files:
    stem = vf.stem
    log(f"processing {vf.name}")

    # 1. extract 16 kHz mono WAV
    wav = ASR_DIR / (stem + ".wav")
    if not wav.exists():
        cmd = ["ffmpeg", "-y", "-i", str(vf),
               "-ar", "16000", "-ac", "1", "-f", "wav", str(wav)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            log(f"  ffmpeg failed: {result.stderr[-200:]}")
            R["files"][stem] = {"error": result.stderr[-200:]}
            continue
        log(f"  extracted audio -> {wav}")

    wav_hash = hashlib.md5(wav.read_bytes()).hexdigest()[:12]

    # 2. run Whisper with word timestamps
    log("  transcribing…")
    t_start = time.time()
    result = model.transcribe(str(wav), word_timestamps=True, language="en")
    t_elapsed = time.time() - t_start

    asr_out = ASR_DIR / (stem + ".json")
    json.dump(result, asr_out.open("w"), indent=2, default=str)
    log(f"  saved {asr_out} (took {t_elapsed:.0f}s)")

    whisper_text = result["text"].strip()
    word_count   = sum(len(seg.get("words", [])) for seg in result.get("segments", []))

    entry = {
        "wav_md5":          wav_hash,
        "whisper_text_head": whisper_text[:200],
        "word_count":        word_count,
        "duration_s":        round(t_elapsed, 1),
    }

    # 3. create empty ref file if missing
    ref_path = REFS_DIR / (stem + ".txt")
    if not ref_path.exists():
        ref_path.write_text(
            "# Replace this with a hand-corrected 2–3 min reference transcript.\n"
            "# One paragraph; no timestamps. Remove this comment line.\n")
        log(f"  created empty ref: {ref_path}")
        entry["wer"] = "BLOCKED – ref not yet written"
        R["files"][stem] = entry
        continue

    ref_text = ref_path.read_text().strip()
    if ref_text.startswith("#"):
        log("  ref is still the template – skipping WER")
        entry["wer"] = "BLOCKED – ref is still template"
        R["files"][stem] = entry
        continue

    # 4. compute WER/CER
    strip = lambda s: re.sub(r"[.,?!;:\-]", "", s.lower()).strip()
    wer  = jiwer.wer(ref_text, whisper_text)
    cer  = jiwer.cer(ref_text, whisper_text)
    wer_s = jiwer.wer(strip(ref_text), strip(whisper_text))

    entry.update({
        "wer_pct":         round(100 * wer, 2),
        "cer_pct":         round(100 * cer, 2),
        "wer_stripped_pct": round(100 * wer_s, 2),
    })
    log(f"  WER={100*wer:.1f}%  CER={100*cer:.1f}%  WER_stripped={100*wer_s:.1f}%")
    R["files"][stem] = entry

out_path = RESULTS_DIR / "asr_results.json"
out_path.write_text(json.dumps(R, indent=2))
log(f"Wrote {out_path}")
