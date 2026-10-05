"""
asr/run_whisper_hi.py — Whisper ASR on a Hindi video.

Method A (translate): task="translate", language="hi"  -> English text directly.
Method B (transcribe): task="transcribe", language="hi" -> Hindi text + word timestamps.

GPU present  → Whisper large-v3
No GPU       → Whisper small (medium is too slow on CPU for this domain)

Usage:
  python asr/run_whisper_hi.py --video videos/hi_foo.mp4
  # writes asr/hi_foo.translate.json and asr/hi_foo.transcribe.json

Outputs (asr/<stem>.translate.json):
  {model, task, language, detected_language, detected_language_prob,
   duration_s, segments: [{start,end,text}], words: [{word,start,end}]}

Outputs (asr/<stem>.transcribe.json):
  Same shape; task="transcribe"; text is Hindi (Devanagari).
"""
import argparse, json, pathlib, time, hashlib, sys

ASR_DIR  = pathlib.Path("asr");      ASR_DIR.mkdir(exist_ok=True)
RES_DIR  = pathlib.Path("results_new"); RES_DIR.mkdir(exist_ok=True)


def choose_model():
    try:
        import torch
        if torch.cuda.is_available():
            return "large-v3"
    except ImportError:
        pass
    return "small"


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(video_path: pathlib.Path, model_name: str, task: str, language: str) -> dict:
    import whisper
    print(f"[whisper] loading model={model_name} task={task} ...")
    t0 = time.time()
    model = whisper.load_model(model_name)
    print(f"[whisper] model loaded in {time.time()-t0:.1f}s")

    t1 = time.time()
    result = model.transcribe(
        str(video_path),
        task=task,
        language=language,
        word_timestamps=True,
        verbose=False,
    )
    elapsed = time.time() - t1
    print(f"[whisper] transcription done in {elapsed:.1f}s")

    # flatten word timestamps from segments
    words = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            text = w.get("word", "").strip()
            if text:
                words.append({
                    "word":  text,
                    "start": round(float(w["start"]), 3),
                    "end":   round(float(w["end"]),   3),
                })

    # detected language info is in result directly
    det_lang      = result.get("language", "?")
    det_lang_prob = result.get("language_probability",
                               result.get("language_probs", {}).get(det_lang, None))
    if det_lang_prob is None:
        det_lang_prob = "unavailable"

    # audio duration from last segment end
    segs = result.get("segments", [])
    duration_s = round(segs[-1]["end"], 3) if segs else 0.0

    out = {
        "model":                    model_name,
        "task":                     task,
        "language_forced":          language,
        "detected_language":        det_lang,
        "detected_language_prob":   det_lang_prob,
        "duration_s":               duration_s,
        "elapsed_s":                round(elapsed, 1),
        "video_sha256":             file_sha256(video_path),
        "segments": [
            {"start": round(s["start"],3), "end": round(s["end"],3), "text": s["text"].strip()}
            for s in segs
        ],
        "words": words,
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="path to hi_<name>.mp4")
    ap.add_argument("--model", default=None, help="override model name")
    args = ap.parse_args()

    video = pathlib.Path(args.video)
    if not video.exists():
        print(f"BLOCKED: video not found: {video}", file=sys.stderr)
        sys.exit(1)

    stem = video.stem  # e.g. hi_news
    model_name = args.model or choose_model()
    print(f"Using Whisper model: {model_name} (GPU={'yes' if model_name=='large-v3' else 'no'})")

    # Method A: translate
    print("\n=== Method A: translate ===")
    out_a = run(video, model_name, task="translate", language="hi")
    path_a = ASR_DIR / f"{stem}.translate.json"
    path_a.write_text(json.dumps(out_a, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[A] wrote {path_a}  ({len(out_a['segments'])} segments, {len(out_a['words'])} words)")

    # Method B step 1: transcribe (Hindi)
    print("\n=== Method B step 1: transcribe (Hindi) ===")
    out_b = run(video, model_name, task="transcribe", language="hi")
    path_b = ASR_DIR / f"{stem}.transcribe.json"
    path_b.write_text(json.dumps(out_b, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[B] wrote {path_b}  ({len(out_b['segments'])} segments, {len(out_b['words'])} words)")

    print("\nDone. Next: python asr/nllb_translate.py --stem", stem)


if __name__ == "__main__":
    main()
