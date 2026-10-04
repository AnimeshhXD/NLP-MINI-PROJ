"""
asr_whisper.py - Task 4: Run Whisper on real video/audio files in videos/.

If no videos are present, print instructions and write BLOCKED.md.
With videos present:
  1. Extract 16 kHz mono audio with ffmpeg
  2. Run Whisper (base without GPU) with word_timestamps=True
  3. Save transcripts to asr/<video>.json
  4. Report WER/CER vs hand-corrected references in refs/<video>.txt
     (skip WER if ref missing)
"""
import os, sys, json, pathlib, time, subprocess, shutil
import jiwer

T0 = time.time()
def log(*a): print("[%5.0fs]" % (time.time() - T0), *a, flush=True)

VIDEOS_DIR = pathlib.Path("videos")
ASR_DIR    = pathlib.Path("asr");    ASR_DIR.mkdir(exist_ok=True)
REFS_DIR   = pathlib.Path("refs");   REFS_DIR.mkdir(exist_ok=True)
LOG_FILE   = pathlib.Path("logs/asr_whisper.json")
LOG_FILE.parent.mkdir(exist_ok=True)

VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v", ".mp3", ".wav", ".flac"}

# ------------------------------------------------------------------ check for ffmpeg
if not shutil.which("ffmpeg"):
    print("ERROR: ffmpeg not found. Install ffmpeg and add to PATH.")
    # append to BLOCKED.md (don't overwrite the full version)
    blocked = pathlib.Path("BLOCKED.md")
    if blocked.exists() and "ffmpeg" in blocked.read_text():
        pass  # already documented
    else:
        blocked.write_text(
            "# BLOCKED: Task 4 – ASR\n\n"
            "ffmpeg is not on PATH. Install from https://ffmpeg.org/download.html "
            "and re-run asr_whisper.py.\n")
    sys.exit(1)

# ------------------------------------------------------------------ find video files
video_files = [f for f in VIDEOS_DIR.iterdir() if f.suffix.lower() in VIDEO_EXTS] \
              if VIDEOS_DIR.exists() else []

if not video_files:
    log("No video files found in videos/")
    msg = (
        "# BLOCKED: Task 4 – ASR\n\n"
        "No video or audio files found in `videos/`.\n\n"
        "## What to do\n"
        "1. Add 3–5 short (2–6 min) openly licensed talks/lectures to `videos/`.\n"
        "   Supported formats: mp4, mkv, webm, mov, avi, m4v, mp3, wav, flac.\n"
        "2. Record the licence/source in `videos/SOURCES.md`.\n"
        "3. Run `python asr_whisper.py` again.\n\n"
        "Alternatively, for a quick test, download a short Creative Commons talk "
        "from TED-Ed or Wikimedia Commons and place it in `videos/`.\n\n"
        "Example using yt-dlp:\n"
        "```bash\n"
        "pip install yt-dlp\n"
        "yt-dlp -x --audio-format wav -o 'videos/%(title)s.%(ext)s' "
        "<YouTube URL of CC-licensed video>\n"
        "```\n"
    )
    pathlib.Path("BLOCKED.md").write_text(msg)
    print(msg)
    sys.exit(0)

# ------------------------------------------------------------------ load Whisper
log("loading Whisper (base – CPU)")
import whisper
model = whisper.load_model("base")

# ------------------------------------------------------------------ process each video
R = {}
for vf in video_files:
    stem = vf.stem
    log("processing %s" % vf.name)

    # 1. extract 16 kHz mono WAV
    wav_path = ASR_DIR / (stem + ".wav")
    if not wav_path.exists():
        cmd = ["ffmpeg", "-y", "-i", str(vf), "-ar", "16000", "-ac", "1",
               "-f", "wav", str(wav_path)]
        subprocess.run(cmd, check=True, capture_output=True)
        log("  extracted audio -> %s" % wav_path)

    # 2. run Whisper with word timestamps
    log("  running Whisper…")
    result = model.transcribe(str(wav_path), word_timestamps=True, language="en")
    # save full result
    asr_out = ASR_DIR / (stem + ".json")
    json.dump(result, asr_out.open("w"), indent=2, default=str)
    log("  saved %s" % asr_out)

    # collect full text (Whisper's own punctuated output)
    whisper_text = result["text"].strip()

    # also build stripped (no punct) version
    import re
    stripped = re.sub(r"[.,?!;:]", "", whisper_text.lower()).strip()

    R[stem] = {
        "whisper_text_preview": whisper_text[:200],
        "word_count": sum(len(seg.get("words", [])) for seg in result.get("segments", [])),
    }

    # 3. create empty ref file if missing
    ref_path = REFS_DIR / (stem + ".txt")
    if not ref_path.exists():
        ref_path.write_text(
            "# Replace this comment with a hand-corrected 2–3 minute reference transcript.\n"
            "# One continuous paragraph; no timestamps.\n")
        log("  created empty ref: %s  (students must fill this in)" % ref_path)
        R[stem]["wer"] = "BLOCKED – no reference transcript yet"
        R[stem]["cer"] = "BLOCKED – no reference transcript yet"
        continue

    # 4. compute WER/CER vs reference
    ref_text = ref_path.read_text().strip()
    if ref_text.startswith("#"):
        log("  ref not filled in yet – skipping WER")
        R[stem]["wer"] = "BLOCKED – reference is still the template"
        continue

    wer = jiwer.wer(ref_text, whisper_text)
    cer = jiwer.cer(ref_text, whisper_text)
    # also WER with punctuation stripped
    strip = lambda s: re.sub(r"[.,?!;:]", "", s.lower())
    wer_stripped = jiwer.wer(strip(ref_text), strip(whisper_text))
    R[stem].update({
        "wer_with_punct":    round(100 * wer, 2),
        "cer_with_punct":    round(100 * cer, 2),
        "wer_stripped":      round(100 * wer_stripped, 2),
    })
    log("  WER=%.1f%%  CER=%.1f%%  WER_stripped=%.1f%%" % (
        100*wer, 100*cer, 100*wer_stripped))

LOG_FILE.write_text(json.dumps(R, indent=2))
log("done – wrote", LOG_FILE)
