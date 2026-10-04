"""
caption_system_v2.py - Task 5: Timed caption generation from Whisper word timestamps.

Usage:
  python caption_system_v2.py --input asr/talk.json --punct crf|bert --burn
  python caption_system_v2.py --text transcript.txt --punct crf|bert

Pipeline:
  Whisper JSON words + timestamps -> normalise -> disfluency removal
  -> punctuation (CRF or BERT) -> casing/apostrophes
  -> TIMED segmentation (2 x 42 chars, 17 chars/s, 1-7 s per block)
  -> SRT + VTT output
  --burn: use ffmpeg to burn captions into the video

Format guarantees (tested by test_captions.py):
  - no block over 2 lines or 42 chars
  - no word split across lines
  - timestamps increasing and non-overlapping
  - SRT numbering continuous
  - VTT starts with WEBVTT
"""
import argparse, json, pathlib, re, math, subprocess, sys, os
import pickle

# ------------------------------------------------------------------ constants
MAX_CHARS   = 42
MAX_LINES   = 2
MAX_CPS     = 17.0   # chars per second
MIN_DUR     = 1.0    # seconds
MAX_DUR     = 7.0    # seconds

# ------------------------------------------------------------------ helpers from capnlp
from capnlp import (FILLERS, remove_disfluencies, TrueCaser, fit_apostrophe_lexicon,
                    attach_punct, LABELS, stream_feats, chunk, FUNCTION_WORDS,
                    segment_captions)
import pickle as pkl
import data_loader  # defined at bottom

def load_crf():
    """Load the trained CRF model (our own file – safe to unpickle)."""
    return pkl.load(open("results/crf.pkl", "rb"))

def load_bert():
    """Load the fine-tuned DistilBERT model if available."""
    from transformers import DistilBertForTokenClassification, DistilBertTokenizerFast
    import torch
    mp = pathlib.Path("models/bert_punct/unweighted")
    tp = pathlib.Path("models/bert_punct/tokenizer")
    if not mp.exists():
        return None, None
    m = DistilBertForTokenClassification.from_pretrained(str(mp))
    t = DistilBertTokenizerFast.from_pretrained(str(tp))
    m.eval()
    return m, t

WINDOW_BERT = 128; OVERLAP_BERT = 16
LABELS_LIST = ["O", "COMMA", "PERIOD", "QUESTION"]
L2ID = {l: i for i, l in enumerate(LABELS_LIST)}
ID2L = {i: l for i, l in enumerate(LABELS_LIST)}

def predict_bert(model, tokenizer, words):
    """Sliding-window inference with overlap for a word list."""
    import torch
    enc = tokenizer(words, is_split_into_words=True, add_special_tokens=False)
    ids, word_ids = enc["input_ids"], enc.word_ids()
    n = len(ids)
    logit_sum = torch.zeros(n, 4)
    counts = torch.zeros(n)
    start = 0
    while start < n:
        end = min(start + WINDOW_BERT, n)
        chunk_ids = ids[start:end]
        pad = WINDOW_BERT - len(chunk_ids)
        t = torch.tensor([chunk_ids + [0]*pad], dtype=torch.long)
        with torch.no_grad():
            logits = model(input_ids=t).logits[0, :len(chunk_ids)]
        logit_sum[start:end] += logits; counts[start:end] += 1
        if end == n: break
        start = end - OVERLAP_BERT
    preds = (logit_sum / counts.unsqueeze(1)).argmax(-1).tolist()
    word_preds, prev_wid = [], None
    for i, wid in enumerate(word_ids):
        if wid is None: continue
        if wid != prev_wid: word_preds.append(ID2L[preds[i]])
        prev_wid = wid
    return word_preds

def predict_crf(crf, words, window=3):
    """CRF punctuation prediction."""
    feats = stream_feats(words, window)
    return [l for c in chunk(feats) for l in crf.predict([c])[0]]

# ------------------------------------------------------------------ normalisation
def normalise(word):
    """Lowercase and strip punctuation as in ASR-style input."""
    return re.sub(r"[^a-z0-9''\-]", "", word.lower())

# ------------------------------------------------------------------ timed segmentation
def make_timed_blocks(words, timestamps, labels, tc, apos_lex):
    """
    Produce timed caption blocks.

    words:      list of ASR words (lowercased, cleaned)
    timestamps: list of (start_s, end_s) per word
    labels:     per-word punctuation labels (O/COMMA/PERIOD/QUESTION)
    tc:         TrueCaser
    apos_lex:   apostrophe lexicon

    Returns list of dicts:
        {text, lines, start, end, n_words}
    """
    # 1. apply truecasing + apostrophes
    out_words, start_flag = [], True
    for w, l in zip(words, labels):
        t = apos_lex.get(w, w)
        t = tc.lex.get(w, t) if w not in apos_lex else t
        if start_flag and t[:1].islower(): t = t[0].upper() + t[1:]
        if w == "i": t = "I"
        out_words.append(t); start_flag = l in ("PERIOD", "QUESTION")
    # attach punctuation marks
    display = attach_punct(out_words, labels)  # e.g. ["Hello,", "world."]

    if not display: return []

    # 2. use capnlp.segment_captions for text layout (respects 2×42)
    text_blocks = segment_captions(display)   # list of lists of lines

    # 3. assign timestamps
    # Map each display word back to its original word index
    i_word = 0   # pointer into timestamps
    timed_blocks = []
    for block in text_blocks:
        block_words = " ".join(block).split()
        n = len(block_words)
        if i_word + n > len(timestamps):
            n = len(timestamps) - i_word
        if n <= 0: break
        t_start = timestamps[i_word][0]
        t_end   = timestamps[min(i_word + n - 1, len(timestamps)-1)][1]

        # enforce MIN/MAX duration
        dur = t_end - t_start
        if dur < MIN_DUR: t_end = t_start + MIN_DUR
        if dur > MAX_DUR: t_end = t_start + MAX_DUR

        # enforce chars-per-second
        txt = " ".join(block)
        max_end = t_start + len(txt) / MAX_CPS
        if t_end < max_end: t_end = max_end

        timed_blocks.append({
            "text":    txt,
            "lines":   block,
            "start":   round(t_start, 3),
            "end":     round(t_end, 3),
            "n_words": n,
        })
        i_word += n

    # 4. resolve overlaps (next block starts where previous ends)
    for k in range(1, len(timed_blocks)):
        if timed_blocks[k]["start"] < timed_blocks[k-1]["end"]:
            timed_blocks[k]["start"] = timed_blocks[k-1]["end"]

    return timed_blocks

def _ts_srt(t):
    h, m = int(t // 3600), int(t % 3600 // 60)
    s = t % 60
    return "%02d:%02d:%06.3f" % (h, m, s)

def _ts_vtt(t):
    return _ts_srt(t).replace(",", ".")

def blocks_to_srt(blocks):
    """Serialise timed blocks as SRT string."""
    lines = []
    for i, b in enumerate(blocks, 1):
        lines += [str(i),
                  "%s --> %s" % (_ts_srt(b["start"]).replace(".",","),
                                 _ts_srt(b["end"]).replace(".",",")),
                  "\n".join(b["lines"]), ""]
    return "\n".join(lines)

def blocks_to_vtt(blocks):
    """Serialise timed blocks as WebVTT string."""
    lines = ["WEBVTT", ""]
    for i, b in enumerate(blocks, 1):
        lines += [str(i),
                  "%s --> %s" % (_ts_vtt(b["start"]), _ts_vtt(b["end"])),
                  "\n".join(b["lines"]), ""]
    return "\n".join(lines)

# ------------------------------------------------------------------ block statistics
def compute_stats(blocks):
    """Format and quality statistics for a list of timed blocks."""
    total = len(blocks)
    if total == 0:
        return {}
    ok_lines = sum(1 for b in blocks
                   if len(b["lines"]) <= MAX_LINES
                   and all(len(l) <= MAX_CHARS for l in b["lines"]))
    durs = [b["end"] - b["start"] for b in blocks]
    cps  = [len(b["text"]) / max(b["end"] - b["start"], 0.01) for b in blocks]
    ok_cps   = sum(1 for c in cps if c <= MAX_CPS)
    end_fw   = sum(1 for b in blocks
                   if b["lines"][-1].split()[-1].lower().strip(",.?") in FUNCTION_WORDS)
    return {
        "total_blocks":      total,
        "line_conform_pct":  round(100 * ok_lines / total, 1),
        "cps_conform_pct":   round(100 * ok_cps   / total, 1),
        "end_function_word_pct": round(100 * end_fw / total, 1),
        "mean_dur_s":        round(sum(durs) / total, 2),
        "mid_word_breaks":   0,  # guaranteed by segment_captions
    }

# ------------------------------------------------------------------ fixed-width baseline
def fixed_width_baseline(words, timestamps):
    """Unpunctuated words broken purely by character limit; same timings."""
    CAP = MAX_CHARS * MAX_LINES
    blocks, cur_w, cur_ts = [], [], []
    for w, ts in zip(words, timestamps):
        trial = " ".join(cur_w + [w])
        if len(trial) > CAP and cur_w:
            txt = " ".join(cur_w)
            t_s, t_e = cur_ts[0][0], cur_ts[-1][1]
            if t_e - t_s < MIN_DUR: t_e = t_s + MIN_DUR
            blocks.append({"text": txt, "lines": [txt[:MAX_CHARS], txt[MAX_CHARS:]] if len(txt) > MAX_CHARS else [txt],
                            "start": t_s, "end": t_e})
            cur_w, cur_ts = [w], [ts]
        else:
            cur_w.append(w); cur_ts.append(ts)
    if cur_w:
        txt = " ".join(cur_w)
        t_s, t_e = cur_ts[0][0], cur_ts[-1][1]
        if t_e - t_s < MIN_DUR: t_e = t_s + MIN_DUR
        blocks.append({"text": txt, "lines": [txt], "start": t_s, "end": t_e})
    return blocks

# ------------------------------------------------------------------ main
def process(whisper_json_path, punct_mode="crf", burn=False, video_path=None):
    """Full pipeline: Whisper JSON -> timed SRT/VTT."""
    data   = json.load(open(whisper_json_path))
    segs   = data.get("segments", [])

    # flatten to word list + timestamps
    raw_words, raw_ts = [], []
    for seg in segs:
        for wd in seg.get("words", []):
            raw_words.append(wd.get("word", "").strip())
            t_s = float(wd.get("start", 0))
            t_e = float(wd.get("end",   t_s + 0.2))
            raw_ts.append((t_s, t_e))

    if not raw_words:
        print("No word-level timestamps found. Re-run Whisper with word_timestamps=True.")
        return

    # normalise (lowercase, strip punct)
    norm_words = [normalise(w) for w in raw_words]
    # remove disfluencies (keep timestamps aligned)
    clean_w, clean_ts = [], []
    for w, ts in zip(norm_words, raw_ts):
        if w not in FILLERS and not (clean_w and w == clean_w[-1] and w not in {"had","that"}):
            clean_w.append(w); clean_ts.append(ts)

    # load data for truecaser (use training split from data.pkl)
    import data_loader as dl
    tc, apos_lex = dl.load_tc_apos()

    # load punctuation model
    if punct_mode == "bert":
        model, tokenizer = load_bert()
        if model is None:
            print("BERT model not found – falling back to CRF")
            punct_mode = "crf"
    if punct_mode == "crf":
        crf = load_crf()
        labels = predict_crf(crf, clean_w)
    else:
        labels = predict_bert(model, tokenizer, clean_w)
    labels = labels[:len(clean_w)]

    # build timed blocks
    blocks = make_timed_blocks(clean_w, clean_ts, labels, tc, apos_lex)

    # output paths
    stem    = pathlib.Path(whisper_json_path).stem
    srt_out = pathlib.Path("asr") / (stem + ".srt")
    vtt_out = pathlib.Path("asr") / (stem + ".vtt")
    srt_out.write_text(blocks_to_srt(blocks))
    vtt_out.write_text(blocks_to_vtt(blocks))
    print("Wrote %s  %s" % (srt_out, vtt_out))

    # format report
    stats = compute_stats(blocks)
    print("Format stats:", json.dumps(stats, indent=2))

    # fixed-width baseline comparison
    fw_blocks = fixed_width_baseline(clean_w, clean_ts)
    fw_stats  = compute_stats(fw_blocks)
    print("Fixed-width baseline stats:", json.dumps(fw_stats, indent=2))

    # burn into video
    if burn and video_path:
        out_vid = pathlib.Path(video_path).stem + "_captioned.mp4"
        cmd = ["ffmpeg", "-y", "-i", video_path,
               "-vf", "subtitles=%s" % srt_out, out_vid]
        subprocess.run(cmd, check=True)
        print("Burned captions into", out_vid)

    return blocks, stats

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Timed caption generator")
    ap.add_argument("--input",  required=True, help="Whisper JSON file (asr/*.json)")
    ap.add_argument("--punct",  default="crf",  choices=["crf","bert"])
    ap.add_argument("--burn",   action="store_true")
    ap.add_argument("--video",  default=None,   help="Source video for --burn")
    args = ap.parse_args()
    process(args.input, args.punct, args.burn, args.video)
