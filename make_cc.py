#!/usr/bin/env python3
"""
make_cc.py - drop a video into  videos/  and get English closed captions in  output/.

  python make_cc.py                  caption every video in videos/ that has no captions yet
  python make_cc.py --watch          keep running; any new file dropped into videos/ is captioned automatically
  python make_cc.py --model medium   better accuracy (slower, bigger download); default is "small" (or "medium" with a GPU)
  python make_cc.py --burn           also write output/<name>_cc.mp4 with the captions burned into the picture
  python make_cc.py --lang hi        force the spoken language (default: detected automatically)
  python make_cc.py --reuse          rebuild captions from the saved Whisper result (seconds, no Whisper run) after a layout change
  python make_cc.py --selftest       test the caption layout code without Whisper or a video

Per video it writes to output/:  <name>.srt  <name>.vtt  <name>.txt (plain English text)  <name>_report.json (settings + format checks)

What it does: ffmpeg extracts the audio -> Whisper detects the language -> English speech is transcribed with word times;
any other language (Hindi etc.) is TRANSLATED to English by Whisper -> text is cut into captions of at most 2 lines x 42
characters, at most 17 characters/second, 1-7 seconds each, breaking at commas/sentence ends and never after words like
"the/of/and", never inside a word -> SRT and WebVTT files.

Timing: English speech uses Whisper's real word times. For translated speech Whisper only gives segment times, so words are
spread inside each segment in proportion to their length (an estimate, not measured word timing).
Needs: pip install openai-whisper imageio-ffmpeg   (ffmpeg on PATH also works)
"""
import argparse, hashlib, json, os, re, shutil, statistics, subprocess, sys, time
from pathlib import Path

MAX_CHARS, MAX_LINES, MAX_CPS = 42, 2, 17.0
MIN_DUR, MAX_DUR, GAP = 1.0, 7.0, 0.04
VIDEO_EXT = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".flv", ".wmv", ".mpg", ".mpeg"}
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"}
FUNC = set("the a an of to and but in on with for that which is are was be by as at or from".split())
CONJ = {"and", "but", "which", "so", "because", "while", "although", "however", "where", "when", "or", "that"}
ABBREV = {"mr.", "mrs.", "ms.", "dr.", "st.", "jr.", "sr.", "vs.", "etc.", "e.g.", "i.e."}

# ------------------------------------------------------------------ audio / ffmpeg
def ffmpeg_exe():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        sys.exit("ffmpeg not found. Run: pip install imageio-ffmpeg   (or: winget install Gyan.FFmpeg)")

def load_audio(path):
    """Decode any video/audio file to 16 kHz mono float samples (no dependency on ffmpeg being on PATH)."""
    import numpy as np
    cmd = [ffmpeg_exe(), "-nostdin", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"]
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode != 0:
        raise RuntimeError("ffmpeg could not read the file: " + p.stderr.decode(errors="replace")[:300])
    audio = np.frombuffer(p.stdout, np.int16).astype(np.float32) / 32768.0
    if audio.size == 0:
        raise RuntimeError("no audio found in the file")
    return audio

# ------------------------------------------------------------------ Whisper
def get_model(name):
    try:
        import whisper, torch
    except ImportError:
        sys.exit("Whisper is not installed. Run: pip install openai-whisper")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("loading Whisper '%s' on %s (first run downloads the model) ..." % (name, dev), flush=True)
    return whisper.load_model(name, device=dev)

def detect_language(model, audio):
    """Average language probabilities over three 30-second windows (start, 25%, 50% of the file)."""
    import whisper
    probs_sum = {}
    n = len(audio)
    for frac in (0.0, 0.25, 0.5):
        start = int(n * frac)
        chunk = whisper.pad_or_trim(audio[start:start + 16000 * 30])
        mel = whisper.log_mel_spectrogram(chunk, n_mels=model.dims.n_mels).to(model.device)
        _, probs = model.detect_language(mel)
        for k, v in probs.items():
            probs_sum[k] = probs_sum.get(k, 0.0) + v / 3
    lang = max(probs_sum, key=probs_sum.get)
    return lang, probs_sum[lang]

def spread(text, start, end):
    """Estimate word times inside a segment: time is shared out in proportion to word length."""
    toks = text.split()
    if not toks:
        return []
    wts = [len(t) + 1 for t in toks]; tot = sum(wts); t = start; out = []
    for tok, wt in zip(toks, wts):
        d = (end - start) * wt / tot
        out.append({"w": tok, "s": t, "e": t + d}); t += d
    return out

def words_from_result(res, use_word_times):
    """Turn a Whisper result into [{'w','s','e'}]; returns (words, skipped_segments, duplicate_segments)."""
    out, skipped, dups, prev_text = [], 0, 0, None
    for seg in res["segments"]:
        text = seg["text"].strip()
        if not text or (seg.get("no_speech_prob", 0) > 0.6 and seg.get("avg_logprob", 0) < -1.0):
            skipped += 1; continue
        if text == prev_text:
            dups += 1
        prev_text = text
        ws = seg.get("words") if use_word_times else None
        if ws:
            out += [{"w": x["word"].strip(), "s": float(x["start"]), "e": float(x["end"])} for x in ws if x["word"].strip()]
        else:
            out += spread(text, float(seg["start"]), float(seg["end"]))
    last_e = 0.0                                          # make times monotonic
    for w in out:
        w["s"] = max(w["s"], last_e); w["e"] = max(w["e"], w["s"] + 0.01); last_e = w["e"]
    return out, skipped, dups

# ------------------------------------------------------------------ caption layout
def clean(w):
    return re.sub(r"[^\w']", "", w).lower()

def fits_two_lines(texts):
    lines, cur = 1, 0
    for t in texts:
        L = len(t)
        if cur == 0:
            cur = L
        elif cur + 1 + L <= MAX_CHARS:
            cur += 1 + L
        else:
            lines += 1; cur = L
        if lines > MAX_LINES:
            return False
    return True

def split_sentences(W):
    sents, cur = [], []
    for k, w in enumerate(W):
        cur.append(k)
        if re.search(r"[.?!\u2026]['\")\]]*$", w["w"]) and w["w"].lower() not in ABBREV:
            sents.append(cur); cur = []
    if cur:
        sents.append(cur)
    return sents

def block_cost(W, sent, i, j):
    idx = sent[i:j]; texts = [W[k]["w"] for k in idx]
    chars = len(" ".join(texts)); first, last = W[idx[0]], W[idx[-1]]
    nxt = W[idx[-1] + 1] if idx[-1] + 1 < len(W) else None
    cost = 3.0                                                        # each block costs 3: split only when needed
    if last["w"][-1:] in ",;:":
        cost -= 1                                                      # break after a comma: good
    elif last["w"][-1:] not in ".?!" and clean(last["w"]) in FUNC:
        cost += 4                                                      # never end on the/of/and ...
    if j < len(sent) and last["w"][-1:] not in ",;:.?!" and clean(W[sent[j]]["w"]) in CONJ:
        cost -= 0.5                                                    # break before and/but/which
    avail = max((nxt["s"] - GAP if nxt else last["e"]) - first["s"], last["e"] - first["s"], 0.05)
    need = chars / MAX_CPS
    if need > avail:
        cost += 6 * (need / avail - 1)                                 # too fast to read
    if len(idx) == 1:
        cost += 1.5
    elif chars < 14 and len(sent) > len(idx):
        cost += 1.0
    return cost

def split_lines(texts):
    full = " ".join(texts)
    if len(full) <= MAX_CHARS:
        return [full]
    best = None
    for k in range(1, len(texts)):
        a, b = " ".join(texts[:k]), " ".join(texts[k:])
        if len(a) > MAX_CHARS or len(b) > MAX_CHARS:
            continue
        s = (3 if a[-1:] in ",;:.?!" else 0) + (2 if clean(texts[k]) in CONJ else 0) - (3 if clean(texts[k - 1]) in FUNC else 0) - abs(k - len(texts) / 2) * 0.1
        if best is None or s > best[0]:
            best = (s, a, b)
    if best:
        return [best[1], best[2]]
    lines, cur = [], ""                                                # only when one word is longer than a line
    for t in texts:
        trial = (cur + " " + t) if cur else t
        if len(trial) <= MAX_CHARS or not cur:
            cur = trial
        else:
            lines.append(cur); cur = t
    lines.append(cur)
    return lines

def build_blocks(W):
    """Optimal cutting of the word list into caption blocks (dynamic programming per sentence). Words are never split."""
    spans = []
    for sent in split_sentences(W):
        n, INF = len(sent), 1e18
        best, back = [INF] * (n + 1), [0] * (n + 1); best[0] = 0.0
        for i in range(n):
            if best[i] >= INF:
                continue
            for j in range(i + 1, n + 1):
                idx = sent[i:j]
                if j > i + 1 and (not fits_two_lines([W[k]["w"] for k in idx]) or W[idx[-1]]["e"] - W[idx[0]]["s"] > MAX_DUR):
                    break
                c = best[i] + block_cost(W, sent, i, j)
                if c < best[j]:
                    best[j], back[j] = c, i
        j, found = n, []
        while j > 0:
            i = back[j]; found.append((sent[i], sent[j - 1])); j = i
        spans += found[::-1]
    blocks = [{"i": a, "j": b, "start": W[a]["s"], "end": W[b]["e"],
               "lines": split_lines([W[k]["w"] for k in range(a, b + 1)])} for a, b in spans]
    for n, b in enumerate(blocks):                                      # display times: min 1 s, 17 chars/s, no overlap
        chars = len(" ".join(b["lines"])); need = max(MIN_DUR, chars / MAX_CPS)
        limit = blocks[n + 1]["start"] - GAP if n + 1 < len(blocks) else b["start"] + MAX_DUR
        b["end"] = max(b["end"], min(b["start"] + need, limit, b["start"] + MAX_DUR))
        if n + 1 < len(blocks) and b["end"] > blocks[n + 1]["start"] - GAP:
            b["end"] = max(b["start"] + 0.2, blocks[n + 1]["start"] - GAP)
    return blocks

def word_split_test(W, blocks):
    """PASS only if the words in the captions are exactly the words of the transcript (no word cut, none lost)."""
    return [w["w"] for w in W] == " ".join(" ".join(b["lines"]) for b in blocks).split()

def format_metrics(W, blocks):
    n = max(len(blocks), 1)
    chars = [len(" ".join(b["lines"])) for b in blocks]
    durs = [b["end"] - b["start"] for b in blocks]
    ok_2x42 = sum(len(b["lines"]) <= MAX_LINES and all(len(l) <= MAX_CHARS for l in b["lines"]) for b in blocks)
    ok_cps = sum(c / max(d, 1e-6) <= MAX_CPS + 1e-9 for c, d in zip(chars, durs))
    fw = sum(clean(" ".join(b["lines"]).split()[-1]) in FUNC and not " ".join(b["lines"])[-1:] in ".?!" for b in blocks)
    overlaps = sum(1 for a, b in zip(blocks, blocks[1:]) if b["start"] < a["end"] - 1e-9)
    return {"n_blocks": len(blocks), "pct_within_2x42": round(100 * ok_2x42 / n, 1), "pct_within_17cps": round(100 * ok_cps / n, 1),
            "pct_ends_on_function_word": round(100 * fw / n, 1), "overlapping_blocks": overlaps,
            "duration_mean_s": round(statistics.mean(durs), 2) if durs else 0, "duration_median_s": round(statistics.median(durs), 2) if durs else 0,
            "duration_min_s": round(min(durs), 2) if durs else 0, "duration_max_s": round(max(durs), 2) if durs else 0,
            "word_split_test": "PASS" if word_split_test(W, blocks) else "FAIL"}

# ------------------------------------------------------------------ writers
def ts(t, vtt=False):
    ms = int(round(t * 1000)); h, ms = divmod(ms, 3600000); m, ms = divmod(ms, 60000); s, ms = divmod(ms, 1000)
    return "%02d:%02d:%02d%s%03d" % (h, m, s, "." if vtt else ",", ms)

def write_captions(blocks, base):

    srt_lines, vtt_lines = [], ["WEBVTT", ""]
    for n, b in enumerate(blocks, 1):
        srt_lines += [str(n), "%s --> %s" % (ts(b["start"]), ts(b["end"])), "\n".join(b["lines"]), ""]
        vtt_lines += [str(n), "%s --> %s" % (ts(b["start"], True), ts(b["end"], True)), "\n".join(b["lines"]), ""]
    Path(str(base) + ".srt").write_text("\n".join(srt_lines), encoding="utf-8")
    Path(str(base) + ".vtt").write_text("\n".join(vtt_lines), encoding="utf-8")
    Path(str(base) + ".txt").write_text("\n".join(" ".join(b["lines"]) for b in blocks) + "\n", encoding="utf-8")

def burn(video, srt, out_mp4):
    tmp = Path("_burn_tmp.srt"); shutil.copy(srt, tmp)                  # simple name in the current folder avoids Windows path escaping
    vf = "subtitles=_burn_tmp.srt:force_style='FontSize=22,Outline=2,MarginV=30'"
    cmd = [ffmpeg_exe(), "-y", "-nostdin", "-v", "error", "-i", str(video), "-vf", vf, "-c:a", "copy", str(out_mp4)]
    try:
        p = subprocess.run(cmd, capture_output=True)
    finally:
        tmp.unlink(missing_ok=True)
    if p.returncode != 0:
        print("   burn failed: " + p.stderr.decode(errors="replace")[:300])
        return False
    return True

# ------------------------------------------------------------------ one video
def process_video(path, args, state):
    path = Path(path); out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / path.stem
    cache = Path(str(base) + "_whisper.json")                         # raw Whisper result, so layout changes need no re-run
    t0 = time.time()
    print("\n== %s" % path.name, flush=True)
    audio = load_audio(path); duration = len(audio) / 16000
    if args.reuse and cache.exists():
        saved = json.loads(cache.read_text(encoding="utf-8"))
        lang, prob, task, res, used_model = saved["language"], saved.get("language_confidence", 1.0), saved["task"], saved["result"], saved["model"]
        print("   using the saved Whisper result (%s, language %s, task %s) - Whisper not run again" % (used_model, lang, task), flush=True)
    else:
        if state.get("model") is None:
            state["model"] = get_model(args.model)
        model = state["model"]; used_model = args.model
        lang, prob = (args.lang, 1.0) if args.lang else detect_language(model, audio)
        if args.model.endswith(".en") and lang != "en":
            print("   skipped: '%s' is an English-only model but the speech is '%s'. Use --model small or medium." % (args.model, lang)); return None
        task = args.task if args.task != "auto" else ("transcribe" if lang == "en" else "translate")
        print("   language: %s (confidence %.2f) -> task: %s%s" % (lang, prob, task, "  [low confidence: consider --lang]" if prob < 0.5 else ""), flush=True)
        kw = dict(language=lang, task=task, word_timestamps=(task == "transcribe"), condition_on_previous_text=False, verbose=False)
        try:
            import torch
            kw["fp16"] = bool(torch.cuda.is_available())
        except Exception:
            kw["fp16"] = False
        if args.beam:
            kw["beam_size"] = args.beam
        raw = model.transcribe(audio, **kw)
        res = {"segments": [{"start": float(g["start"]), "end": float(g["end"]), "text": g["text"], "no_speech_prob": float(g.get("no_speech_prob", 0)),
                             "avg_logprob": float(g.get("avg_logprob", 0)),
                             "words": [{"word": w["word"], "start": float(w["start"]), "end": float(w["end"])} for w in g["words"]] if g.get("words") else None}
                            for g in raw["segments"]]}
        cache.write_text(json.dumps({"language": lang, "language_confidence": float(prob), "task": task, "model": used_model, "result": res}, ensure_ascii=False), encoding="utf-8")
    use_word_times = (task == "transcribe")
    W, skipped, dups = words_from_result(res, use_word_times)
    if not W:
        print("   no speech found - nothing written"); return None
    blocks = build_blocks(W)
    write_captions(blocks, base)
    metrics = format_metrics(W, blocks)
    burned = False
    if args.burn and path.suffix.lower() in VIDEO_EXT:
        burned = burn(path, str(base) + ".srt", out_dir / (path.stem + "_cc.mp4"))
    report = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "video": path.name, "video_sha256_prefix": hashlib.sha256(path.read_bytes()[:50_000_000]).hexdigest()[:16],
              "audio_duration_s": round(duration, 2), "whisper_model": used_model, "language_detected": lang, "language_confidence": round(prob, 3),
              "task": task, "word_times": "whisper word timestamps" if use_word_times else "estimated: spread inside each Whisper segment by word length",
              "condition_on_previous_text": False, "beam_size": args.beam or "default", "whisper_result_reused": bool(args.reuse and cache.exists()),
              "n_words": len(W), "segments_skipped_as_non_speech": skipped, "consecutive_duplicate_segments": dups,
              "layout_version": "2 (block cost 3, comma bonus 1, conjunction bonus 0.5)",
              "format_metrics": metrics, "burned_in_video": burned, "processing_time_s": round(time.time() - t0, 1),
              "note": "Caption quality is not evaluated by this script. Check the .txt against what is said."}
    Path(str(base) + "_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("   %d captions | within 2x42: %s%% | within 17 cps: %s%% | ends on function word: %s%% | word-split test: %s | %.0fs"
          % (metrics["n_blocks"], metrics["pct_within_2x42"], metrics["pct_within_17cps"], metrics["pct_ends_on_function_word"], metrics["word_split_test"], time.time() - t0))
    if dups:
        print("   warning: %d consecutive identical segments (Whisper may be repeating itself - check the .txt)" % dups)
    print("   -> %s.srt  %s.vtt  %s.txt  %s_report.json" % ((base,) * 4))
    return report

def todo(args):
    vd, od = Path(args.videos), Path(args.output)
    files = sorted(p for p in vd.glob("*") if p.suffix.lower() in VIDEO_EXT | AUDIO_EXT and not p.name.startswith("."))
    return [p for p in files if args.force or not (od / (p.stem + ".srt")).exists() or (od / (p.stem + ".srt")).stat().st_mtime < p.stat().st_mtime]

# ------------------------------------------------------------------ self-test (no Whisper, no video)
def selftest():
    import random
    rng = random.Random(1); words, t = [], 0.0
    vocab = "the of and but which we our people country government important system because however very long internationalization".split()
    for k in range(400):
        w = rng.choice(vocab)
        if k == 77: w = "x" * 45                                       # a word longer than a whole line
        if rng.random() < 0.12: w += ","
        if rng.random() < 0.08: w += "."
        d = 0.15 + 0.07 * len(w) / 5 + rng.random() * 0.1
        t += rng.choice([0, 0, 0, 0.3, 1.2]); words.append({"w": w, "s": t, "e": t + d}); t += d
    blocks = build_blocks(words); m = format_metrics(words, blocks)
    print("selftest layout:", m)
    assert m["word_split_test"] == "PASS" and m["overlapping_blocks"] == 0
    assert all(len(b["lines"]) <= 2 for b in blocks)
    assert all(len(l) <= 42 for b in blocks for l in b["lines"] if len(l) != 45)
    assert all(a["start"] < a["end"] <= b["start"] for a, b in zip(blocks, blocks[1:]))
    assert all(b["end"] - b["start"] <= MAX_DUR + 1e-9 for b in blocks)
    broken = [dict(b) for b in blocks]; broken[3]["lines"] = [broken[3]["lines"][0][:-2] + " " + broken[3]["lines"][0][-2:]] + broken[3]["lines"][1:]
    assert not word_split_test(words, broken), "the word-split test must FAIL when a word is cut"
    print("selftest PASS (limits respected, no word split, and the split-word check fails when it should)")

def main():
    for s in (sys.stdout, sys.stderr):
        try: s.reconfigure(encoding="utf-8", errors="replace")
        except Exception: pass
    ap = argparse.ArgumentParser(description="videos/ -> output/ English closed captions", formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--videos", default="videos"); ap.add_argument("--output", default="output")
    ap.add_argument("--model", default=None, help="tiny/base/small/medium/large-v3 (never *.en for non-English speech)")
    ap.add_argument("--lang", default=None, help="spoken language code, e.g. hi, en (default: detect)")
    ap.add_argument("--task", choices=["auto", "transcribe", "translate"], default="auto")
    ap.add_argument("--beam", type=int, default=0, help="beam size, e.g. 5 (better, slower)")
    ap.add_argument("--reuse", action="store_true", help="rebuild captions from the saved Whisper result (output/<name>_whisper.json) without running Whisper again")
    ap.add_argument("--burn", action="store_true"); ap.add_argument("--force", action="store_true", help="redo videos that already have captions")
    ap.add_argument("--watch", action="store_true"); ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if args.model is None:
        try:
            import torch; args.model = "medium" if torch.cuda.is_available() else "small"
        except Exception:
            args.model = "small"
    Path(args.videos).mkdir(exist_ok=True); Path(args.output).mkdir(exist_ok=True)
    state = {"model": None}
    if not args.watch:
        files = todo(args)
        if not files:
            print("nothing to do: put video files into '%s/' (already captioned files are skipped; use --force to redo)." % args.videos); return
        for p in files:
            try: process_video(p, args, state)
            except Exception as e: print("   FAILED: %s" % e)
        return
    print("watching '%s/' - drop a video in and captions appear in '%s/'. Ctrl+C to stop." % (args.videos, args.output))
    seen = {}
    while True:
        for p in todo(args):
            size = p.stat().st_size
            if seen.get(p.name) == size and size > 0:                 # size unchanged between two checks = copy finished
                try: process_video(p, args, state)
                except Exception as e: print("   FAILED: %s" % e)
                seen.pop(p.name, None)
            else:
                seen[p.name] = size
        time.sleep(5)

if __name__ == "__main__":
    main()