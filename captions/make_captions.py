"""
captions/make_captions.py  -  Step 5: timed SRT/VTT caption generation.

Input : asr/<stem>.json  (Whisper output with word_timestamps=True)
        --punct  crf|bert  (punctuation model to apply)
Output: captions/<stem>_<punct>.srt
        captions/<stem>_<punct>.vtt
        captions/baseline_<stem>.srt  (no-punct fixed-width baseline)
        results_new/caption_results.json

Caption rules (from spec):
  - ≤ 2 lines per block
  - ≤ 42 chars per line
  - ≤ 17 chars/s (reading speed)
  - 1.0 – 7.0 s per block
  - No word may be split across lines

Requires: asr/<stem>.json produced by  asr/run_whisper.py
BLOCKED  : until ffmpeg + videos/ are available (see BLOCKED.md)
"""
import argparse, json, pathlib, sys, time, re

OUT_DIR = pathlib.Path("captions"); OUT_DIR.mkdir(exist_ok=True)
RES_DIR = pathlib.Path("results_new"); RES_DIR.mkdir(exist_ok=True)

MAX_CHARS   = 42
MAX_LINES   = 2
MAX_CPS     = 17.0   # chars per second
MIN_DUR     = 1.0    # seconds
MAX_DUR     = 7.0    # seconds
PAD_S       = 0.05   # gap between blocks


def fmt_time_srt(s: float) -> str:
    h, rem = divmod(s, 3600)
    m, rem = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(rem):02d},{int((rem%1)*1000):03d}"


def fmt_time_vtt(s: float) -> str:
    h, rem = divmod(s, 3600)
    m, rem = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(rem):02d}.{int((rem%1)*1000):03d}"


def extract_words(whisper_json: dict) -> list[dict]:
    """Flatten Whisper segments into [{word, start, end}]."""
    words = []
    for seg in whisper_json.get("segments", []):
        for w in seg.get("words", []):
            text = w.get("word", "").strip()
            if text:
                words.append({"word": text,
                               "start": float(w["start"]),
                               "end":   float(w["end"])})
    return words


def split_into_lines(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    """Split text into ≤ max_chars lines without breaking words."""
    words  = text.split()
    lines  = []
    cur    = ""
    for w in words:
        candidate = (cur + " " + w).strip()
        if len(candidate) <= max_chars:
            cur = candidate
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def make_timed_blocks(words: list[dict], punct_fn=None) -> list[dict]:
    """
    Group Whisper words into caption blocks obeying timing and length rules.

    punct_fn(word_str) -> "O" | "COMMA" | "PERIOD" | "QUESTION"
    If None, no punctuation is added.
    """
    if not words:
        return []

    blocks  = []
    buf_words, buf_starts, buf_ends = [], [], []

    def flush():
        if not buf_words:
            return
        text      = " ".join(buf_words)
        dur       = buf_ends[-1] - buf_starts[0]
        n_chars   = len(text.replace(" ", ""))
        min_dur_r = n_chars / MAX_CPS   # minimum time to read at max speed

        t_start = buf_starts[0]
        t_end   = max(buf_ends[-1], t_start + max(MIN_DUR, min_dur_r))
        t_end   = min(t_end, t_start + MAX_DUR)

        lines = split_into_lines(text)[:MAX_LINES]
        blocks.append({"start": round(t_start, 3),
                        "end":   round(t_end,   3),
                        "text":  lines})

    i = 0
    while i < len(words):
        w      = words[i]
        raw    = w["word"].strip()
        label  = punct_fn(raw) if punct_fn else "O"

        # add punctuation
        display = raw
        if label == "COMMA":    display = raw + ","
        elif label == "PERIOD":   display = raw + "."
        elif label == "QUESTION": display = raw + "?"

        buf_words.append(display)
        buf_starts.append(w["start"])
        buf_ends.append(w["end"])

        total_chars = len(" ".join(buf_words).replace(" ", ""))
        dur         = buf_ends[-1] - buf_starts[0]

        # flush if: block would exceed limits, or punctuation ends a sentence
        should_flush = (
            total_chars > MAX_CHARS * MAX_LINES
            or dur >= MAX_DUR
            or (label in ("PERIOD", "QUESTION") and len(buf_words) >= 4)
        )

        if should_flush:
            flush()
            buf_words, buf_starts, buf_ends = [], [], []

        i += 1

    flush()

    # enforce no-overlap
    for j in range(1, len(blocks)):
        if blocks[j]["start"] < blocks[j-1]["end"] + PAD_S:
            blocks[j]["start"] = round(blocks[j-1]["end"] + PAD_S, 3)
        if blocks[j]["end"] <= blocks[j]["start"]:
            blocks[j]["end"] = round(blocks[j]["start"] + MIN_DUR, 3)

    return blocks


def write_srt(blocks: list[dict], path: pathlib.Path):
    lines = []
    for idx, b in enumerate(blocks, 1):
        lines.append(str(idx))
        lines.append(f"{fmt_time_srt(b['start'])} --> {fmt_time_srt(b['end'])}")
        lines.extend(b["text"])
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_vtt(blocks: list[dict], path: pathlib.Path):
    lines = ["WEBVTT", ""]
    for b in blocks:
        lines.append(f"{fmt_time_vtt(b['start'])} --> {fmt_time_vtt(b['end'])}")
        lines.extend(b["text"])
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def fixed_width_baseline(words: list[dict]) -> list[dict]:
    """Baseline: no punctuation, fixed max-chars blocks."""
    return make_timed_blocks(words, punct_fn=None)


# ---- CRF punctuation ------------------------------------------------
def load_crf(data_pkl: str = "data/talks.pkl"):
    """Train a quick CRF on train split to use for caption punctuation."""
    import pickle
    import sklearn_crfsuite

    talks = pickle.load(open(data_pkl, "rb"))  # our own data, safe

    def word_features(words, i):
        w = words[i]; l = w.lower()
        f = {"w": l, "suf2": l[-2:], "upper": w[0].isupper()}
        if i > 0:   f["pw"] = words[i-1].lower()
        else:        f["BOS"] = True
        if i < len(words)-1: f["nw"] = words[i+1].lower()
        else:        f["EOS"] = True
        return f

    X, y = [], []
    for t in talks:
        if t["split"] != "train": continue
        words  = [w for w,_ in t["pairs"]]
        X.append([word_features(words, i) for i in range(len(words))])
        y.append([l for _,l in t["pairs"]])

    crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=0.1,
                                max_iterations=100, all_possible_transitions=True)
    crf.fit(X, y)

    # build per-word cache from test set words for fast lookup
    # in production: just predict on the buffer
    def predict(word_buf: list[str]) -> list[str]:
        feats = [word_features(word_buf, i) for i in range(len(word_buf))]
        return crf.predict([feats])[0]

    return predict


# ---- BERT punctuation -----------------------------------------------
def load_bert(model_dir: str = "models/bert_punct_v2"):
    """Load saved BERT model for caption punctuation."""
    import torch
    from transformers import DistilBertForTokenClassification, DistilBertTokenizerFast
    import numpy as np

    mp = pathlib.Path(model_dir)
    if not mp.exists():
        print(f"BERT model not found at {model_dir}. Run punct/train_bert.py first.")
        return None

    LABELS = ["O", "COMMA", "PERIOD", "QUESTION"]
    tok    = DistilBertTokenizerFast.from_pretrained(model_dir)
    model  = DistilBertForTokenClassification.from_pretrained(model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = model.to(device).eval()

    @torch.no_grad()
    def predict(word_buf: list[str]) -> list[str]:
        if not word_buf:
            return []
        enc  = tok(word_buf, is_split_into_words=True, max_length=128,
                   truncation=True, padding="max_length", return_tensors="pt")
        out  = model(input_ids=enc["input_ids"].to(device),
                     attention_mask=enc["attention_mask"].to(device))
        wids = enc.word_ids(batch_index=0)
        preds = ["O"] * len(word_buf)
        seen  = set()
        for pos, wid in enumerate(wids):
            if wid is not None and wid not in seen and wid < len(word_buf):
                seen.add(wid)
                preds[wid] = LABELS[out.logits[0, pos].argmax().item()]
        return preds

    return predict


# ---- main -----------------------------------------------------------
def process_file(asr_json: pathlib.Path, punct_mode: str) -> dict:
    data  = json.loads(asr_json.read_bytes().decode("utf-8-sig"))
    words = extract_words(data)

    if not words:
        print(f"No word timestamps in {asr_json}.")
        return {}

    print(f"  {len(words)} words from {asr_json.name}")

    # baseline (no punct)
    baseline_blocks = fixed_width_baseline(words)
    srt_base = OUT_DIR / f"baseline_{asr_json.stem}.srt"
    vtt_base = OUT_DIR / f"baseline_{asr_json.stem}.vtt"
    write_srt(baseline_blocks, srt_base)
    write_vtt(baseline_blocks, vtt_base)
    print(f"  baseline: {len(baseline_blocks)} blocks -> {srt_base}")

    # punctuated
    if punct_mode == "crf":
        if not pathlib.Path("data/talks.pkl").exists():
            print("  data/talks.pkl not found; run data/prepare.py first")
            return {"baseline_blocks": len(baseline_blocks)}
        predict_fn_raw = load_crf()
        # wrap: predict_fn for make_timed_blocks takes single word
        # we pre-predict whole word list
        all_words_str = [w["word"].strip() for w in words]
        preds_list    = predict_fn_raw(all_words_str)
        pred_dict     = {i: p for i, p in enumerate(preds_list)}
        punct_fn      = lambda w, idx=None: pred_dict.get(all_words_str.index(w) if idx is None else idx, "O")
        # simpler: per-index wrapper using list position
        punct_list    = preds_list
        def punct_fn_indexed(word_str): return "O"  # placeholder
        # use indexed version via closure
        idx_counter = [0]
        def punct_fn(word_str):
            p = punct_list[idx_counter[0]] if idx_counter[0] < len(punct_list) else "O"
            idx_counter[0] += 1
            return p

    elif punct_mode == "bert":
        predict_raw = load_bert()
        if predict_raw is None:
            return {"baseline_blocks": len(baseline_blocks)}
        all_words_str = [w["word"].strip() for w in words]
        preds_list    = predict_raw(all_words_str)
        idx_counter   = [0]
        def punct_fn(word_str):
            p = preds_list[idx_counter[0]] if idx_counter[0] < len(preds_list) else "O"
            idx_counter[0] += 1
            return p
    else:
        punct_fn = None

    punct_blocks = make_timed_blocks(words, punct_fn=punct_fn)
    srt_p = OUT_DIR / f"{asr_json.stem}_{punct_mode}.srt"
    vtt_p = OUT_DIR / f"{asr_json.stem}_{punct_mode}.vtt"
    write_srt(punct_blocks, srt_p)
    write_vtt(punct_blocks, vtt_p)
    print(f"  {punct_mode}: {len(punct_blocks)} blocks -> {srt_p}")

    return {
        "baseline_blocks": len(baseline_blocks),
        "punct_blocks":    len(punct_blocks),
        "srt":             str(srt_p),
        "vtt":             str(vtt_p),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", help="asr/*.json file (default: all in asr/)")
    ap.add_argument("--punct", default="crf", choices=["crf", "bert", "none"])
    ap.add_argument("--burn",  action="store_true",
                    help="burn captions into video with ffmpeg (requires videos/)")
    args = ap.parse_args()

    asr_files = (
        [pathlib.Path(args.input)] if args.input
        else sorted(pathlib.Path("asr").glob("*.json"))
    )

    if not asr_files:
        print("No ASR JSON files found. Run  asr/run_whisper.py  first.")
        print("(ASR requires ffmpeg + videos/ – see BLOCKED.md)")
        sys.exit(0)

    results = {}
    for f in asr_files:
        print(f"Processing {f} …")
        results[f.stem] = process_file(f, args.punct)

    (RES_DIR / "caption_results.json").write_text(
        json.dumps({"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "seed": 42, "punct_mode": args.punct, "files": results}, indent=2))
    print("Wrote results_new/caption_results.json")


if __name__ == "__main__":
    main()
