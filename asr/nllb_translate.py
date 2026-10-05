"""
asr/nllb_translate.py — Translate Hindi Whisper segments to English with NLLB.

Input:  asr/<stem>.transcribe.json   (from run_whisper_hi.py Method B)
Output: asr/<stem>.nllb.json         (same segments, text replaced with English)

Model: facebook/nllb-200-distilled-600M
  src_lang = hin_Deva
  tgt_lang = eng_Latn

Each segment is translated independently (preserves Whisper timing).
~1.2 GB model; first run downloads it into the HuggingFace cache.

Usage:
  python asr/nllb_translate.py --stem hi_foo
"""
import argparse, json, pathlib, sys, time

ASR_DIR = pathlib.Path("asr")
MODEL_ID = "facebook/nllb-200-distilled-600M"
SRC_LANG = "hin_Deva"
TGT_LANG = "eng_Latn"
MAX_NEW_TOKENS = 256


def translate_segments(segments: list, model, tokenizer) -> list:
    """Translate each segment's text in-place; keep start/end times."""
    translated = []
    for i, seg in enumerate(segments):
        src_text = seg["text"].strip()
        if not src_text:
            translated.append({**seg, "text_hi": src_text, "text": ""})
            continue

        inputs = tokenizer(
            src_text,
            return_tensors="pt",
            src_lang=SRC_LANG,
        )
        out_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id[TGT_LANG],
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=4,
        )
        en_text = tokenizer.decode(out_tokens[0], skip_special_tokens=True).strip()
        translated.append({
            "start":   seg["start"],
            "end":     seg["end"],
            "text_hi": src_text,        # original Hindi kept
            "text":    en_text,         # English translation
        })
        if (i + 1) % 10 == 0 or (i + 1) == len(segments):
            print(f"  [{i+1}/{len(segments)}] {src_text[:40]!r} -> {en_text[:40]!r}")

    return translated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", required=True, help="e.g. hi_foo (no extension)")
    args = ap.parse_args()

    src_path = ASR_DIR / f"{args.stem}.transcribe.json"
    if not src_path.exists():
        print(f"BLOCKED: {src_path} not found. Run run_whisper_hi.py first.", file=sys.stderr)
        sys.exit(1)

    data = json.loads(src_path.read_text(encoding="utf-8"))
    segments = data["segments"]
    print(f"[nllb] loaded {len(segments)} segments from {src_path}")
    print(f"[nllb] loading model {MODEL_ID} ...")

    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import torch
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    print(f"[nllb] model loaded in {time.time()-t0:.1f}s  device={'cuda' if torch.cuda.is_available() else 'cpu'}")

    t1 = time.time()
    translated = translate_segments(segments, model, tokenizer)
    elapsed = round(time.time() - t1, 1)
    print(f"[nllb] translated {len(translated)} segments in {elapsed}s")

    out = {
        **{k: v for k, v in data.items() if k != "segments"},
        "nllb_model":      MODEL_ID,
        "nllb_src_lang":   SRC_LANG,
        "nllb_tgt_lang":   TGT_LANG,
        "nllb_elapsed_s":  elapsed,
        "segments":        translated,
    }
    out_path = ASR_DIR / f"{args.stem}.nllb.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[nllb] wrote {out_path}")
    print(f"\nDone. Next: python captions/make_captions_hi.py --video videos/{args.stem}.mp4 --method nllb --vtt")


if __name__ == "__main__":
    main()
