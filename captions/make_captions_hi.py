"""
captions/make_captions_hi.py — English captions from a Hindi video.

Builds timed SRT/VTT from Whisper translate or NLLB output using the
clause-aware segmenter (capnlp.segment_captions).

Timing within a Whisper segment:
  Whisper's "translate" mode gives segment-level English (no per-word times).
  We distribute a segment's [start, end] interval across its caption blocks
  in proportion to character count: t_block_i ∝ chars(block_i) / total_chars.
  This is an approximation; actual reading speed may vary.
  (For "transcribe + NLLB", the same proportional rule applies because NLLB
  outputs English sentences, not word-aligned tokens.)

Usage:
  python captions/make_captions_hi.py \\
      --video  videos/hi_foo.mp4 \\
      --method translate|nllb \\
      --out    captions/hi_foo_translate.srt \\
      [--vtt] [--burn] [--bilingual]

Options:
  --vtt        also write a .vtt file (same stem as --out)
  --burn       burn captions into output_burned.mp4 using ffmpeg
  --bilingual  write a second SRT with Hindi source text (method nllb only)
"""
import argparse, json, pathlib, re, sys, subprocess

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from capnlp import segment_captions

CAP_DIR = pathlib.Path("captions"); CAP_DIR.mkdir(exist_ok=True)
ASR_DIR = pathlib.Path("asr")
RES_DIR = pathlib.Path("results_new"); RES_DIR.mkdir(exist_ok=True)

MAX_CHARS = 42
MAX_LINES = 2
MAX_CPS   = 17.0
MIN_DUR   = 1.0
MAX_DUR   = 7.0


# ── time formatting ────────────────────────────────────────────────────────────
def _srt_time(s):
    h, r = divmod(s, 3600); m, r = divmod(r, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(r):02d},{int(r%1*1000):03d}"


def _vtt_time(s):
    h, r = divmod(s, 3600); m, r = divmod(r, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(r):02d}.{int(r%1*1000):03d}"


# ── proportional time distribution ────────────────────────────────────────────
def assign_times(blocks, seg_start, seg_end, next_seg_start=None):
    """Assign [start, end] to each caption block proportional to char count.

    Proportional timing: within the Whisper segment [seg_start, seg_end],
    block i gets (seg_end - seg_start) * chars(i) / total_chars seconds.
    After assignment, CPS is enforced by extending end to next_block_start
    or to next_seg_start for the last block in the segment.
    """
    if not blocks:
        return []

    # character count per block (all lines joined with space)
    char_counts = [sum(len(l) for l in b) + max(0, len(b) - 1) for b in blocks]
    total_chars = max(sum(char_counts), 1)
    seg_dur = seg_end - seg_start

    results = []
    t_cursor = seg_start
    for i, (block, n_chars) in enumerate(zip(blocks, char_counts)):
        raw_dur = seg_dur * (n_chars / total_chars)
        t_start = t_cursor
        t_end   = t_cursor + raw_dur
        t_cursor = t_end

        # duration bounds
        t_end = max(t_end, t_start + MIN_DUR)
        t_end = min(t_end, t_start + MAX_DUR)

        # CPS enforcement: extend end toward next block or next segment
        text_len = n_chars
        min_cps_dur = text_len / MAX_CPS
        if t_end - t_start < min_cps_dur:
            # try extending to start of next block
            if i + 1 < len(blocks):
                next_start = t_cursor          # where next block starts
                t_end = max(t_end, min(next_start, t_start + min_cps_dur))
            elif next_seg_start is not None:
                t_end = max(t_end, min(next_seg_start, t_start + min_cps_dur))

        results.append({
            "start": round(t_start, 3),
            "end":   round(t_end,   3),
            "lines": block,
            "text":  " ".join(block),
        })

    return results


def _resolve_overlaps(timed_blocks):
    """Trim end[i] so it does not overlap start[i+1]."""
    for i in range(len(timed_blocks) - 1):
        gap = timed_blocks[i + 1]["start"] - 0.04   # 40 ms gap
        if timed_blocks[i]["end"] > gap:
            timed_blocks[i]["end"] = max(timed_blocks[i]["start"] + 0.1, gap)
    return timed_blocks


# ── segment → timed blocks ─────────────────────────────────────────────────────
def segments_to_timed_blocks(segments):
    """Convert ASR segment list to timed caption blocks."""
    all_blocks = []
    for i, seg in enumerate(segments):
        text = seg.get("text", "").strip()
        if not text:
            continue
        words = text.split()
        if not words:
            continue
        blocks = segment_captions(words)
        next_start = segments[i + 1]["start"] if i + 1 < len(segments) else None
        timed = assign_times(blocks, seg["start"], seg["end"], next_start)
        all_blocks.extend(timed)
    return _resolve_overlaps(all_blocks)


# ── SRT / VTT writers ──────────────────────────────────────────────────────────
def write_srt(timed_blocks, path, hi_segments=None):
    """Write SRT.  If hi_segments is given, append Hindi text as a 3rd line."""
    lines = []
    for idx, b in enumerate(timed_blocks, 1):
        t = f"{_srt_time(b['start'])} --> {_srt_time(b['end'])}"
        block_text = "\n".join(b["lines"])
        if hi_segments:
            # best-effort: find the Hindi segment whose range contains this block
            hi = _find_hi_seg(hi_segments, b["start"])
            block_text += f"\n{hi}"
        lines.append(f"{idx}\n{t}\n{block_text}\n")
    pathlib.Path(path).write_text("\n".join(lines), encoding="utf-8")


def write_vtt(timed_blocks, path):
    lines = ["WEBVTT", ""]
    for idx, b in enumerate(timed_blocks, 1):
        t = f"{_vtt_time(b['start'])} --> {_vtt_time(b['end'])}"
        lines.append(f"{idx}")
        lines.append(t)
        lines.extend(b["lines"])
        lines.append("")
    pathlib.Path(path).write_text("\n".join(lines), encoding="utf-8")


def _find_hi_seg(hi_segments, t_start):
    """Return Hindi text for the segment whose range contains t_start."""
    for s in hi_segments:
        if s["start"] <= t_start <= s["end"]:
            return s.get("text_hi", s.get("text", ""))
    return ""


# ── ffmpeg burn ────────────────────────────────────────────────────────────────
def burn_captions(video_path, srt_path, out_path):
    """Burn English SRT into video; extract frame at 5s to verify."""
    srt_escaped = str(srt_path).replace("\\", "/").replace(":", "\\:")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", f"subtitles={srt_escaped}",
        "-c:a", "copy",
        str(out_path),
    ]
    print("[ffmpeg] burning captions:", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("[ffmpeg] STDERR:", r.stderr[-500:], file=sys.stderr)
        return False

    # extract a frame at 5 seconds as a verification screenshot
    frame_path = out_path.with_suffix(".frame5s.jpg")
    cmd2 = ["ffmpeg", "-y", "-ss", "5", "-i", str(out_path),
            "-frames:v", "1", str(frame_path)]
    subprocess.run(cmd2, capture_output=True)
    print(f"[ffmpeg] burned → {out_path}")
    print(f"[ffmpeg] verification frame → {frame_path}")
    return True


# ── format metrics ─────────────────────────────────────────────────────────────
def fmt_metrics(timed_blocks):
    n = len(timed_blocks)
    from capnlp import FUNCTION_WORDS
    n_ok_len = n_ok_cps = n_fw = 0
    cps_exceptions = []
    durations = []
    for b in timed_blocks:
        dur = b["end"] - b["start"]
        durations.append(dur)
        text = b["text"]
        n_chars = len(text)
        cps = n_chars / max(dur, 0.01)

        if all(len(l) <= MAX_CHARS for l in b["lines"]) and len(b["lines"]) <= MAX_LINES:
            n_ok_len += 1
        if cps <= MAX_CPS:
            n_ok_cps += 1
        else:
            cps_exceptions.append({"text": text, "cps": round(cps, 2), "dur": round(dur, 2)})
        last_w = text.rstrip(".,?!").split()[-1].lower() if text.split() else ""
        if last_w in FUNCTION_WORDS:
            n_fw += 1

    import statistics
    return {
        "n_blocks":          n,
        "pct_within_2x42":   round(100 * n_ok_len / max(n, 1), 1),
        "pct_within_17cps":  round(100 * n_ok_cps / max(n, 1), 1),
        "pct_ends_funcword": round(100 * n_fw / max(n, 1), 1),
        "mean_dur_s":        round(statistics.mean(durations), 2) if durations else 0,
        "median_dur_s":      round(statistics.median(durations), 2) if durations else 0,
        "cps_exceptions":    cps_exceptions[:10],   # up to 10 printed
    }


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video",  required=True)
    ap.add_argument("--method", required=True, choices=["translate", "nllb"])
    ap.add_argument("--out",    required=True, help="output .srt path")
    ap.add_argument("--vtt",       action="store_true")
    ap.add_argument("--burn",      action="store_true")
    ap.add_argument("--bilingual", action="store_true",
                    help="write second SRT with Hindi source (nllb only)")
    args = ap.parse_args()

    video_path = pathlib.Path(args.video)
    stem = video_path.stem
    out_srt = pathlib.Path(args.out)

    # load appropriate ASR JSON
    if args.method == "translate":
        asr_path = ASR_DIR / f"{stem}.translate.json"
    else:
        asr_path = ASR_DIR / f"{stem}.nllb.json"

    if not asr_path.exists():
        print(f"BLOCKED: {asr_path} not found. Run run_whisper_hi.py first.", file=sys.stderr)
        sys.exit(1)

    data = json.loads(asr_path.read_text(encoding="utf-8"))
    segments = data["segments"]
    print(f"[captions] loaded {len(segments)} segments from {asr_path}")

    # build timed blocks
    timed_blocks = segments_to_timed_blocks(segments)
    print(f"[captions] {len(timed_blocks)} caption blocks from {len(segments)} segments")

    # SRT
    write_srt(timed_blocks, out_srt)
    print(f"[captions] wrote {out_srt}")

    # VTT
    if args.vtt:
        vtt_path = out_srt.with_suffix(".vtt")
        write_vtt(timed_blocks, vtt_path)
        print(f"[captions] wrote {vtt_path}")

    # bilingual SRT
    if args.bilingual and args.method == "nllb":
        bi_path = out_srt.with_stem(out_srt.stem + "_bilingual")
        write_srt(timed_blocks, bi_path, hi_segments=segments)
        print(f"[captions] wrote bilingual {bi_path}")
    elif args.bilingual and args.method == "translate":
        print("[captions] --bilingual requires --method nllb (no Hindi source in translate mode)")

    # burn
    if args.burn:
        burned_path = out_srt.with_stem(out_srt.stem + "_burned").with_suffix(".mp4")
        if not video_path.exists():
            print(f"BLOCKED: video not found for --burn: {video_path}", file=sys.stderr)
        else:
            burn_captions(video_path, out_srt, burned_path)

    # format metrics (to stdout + JSON)
    metrics = fmt_metrics(timed_blocks)
    print("\n--- Format metrics ---")
    for k, v in metrics.items():
        if k != "cps_exceptions":
            print(f"  {k}: {v}")
    if metrics["cps_exceptions"]:
        print(f"  CPS exceptions ({len(metrics['cps_exceptions'])}):")
        for ex in metrics["cps_exceptions"]:
            print(f"    [{ex['cps']:.1f} cps, {ex['dur']:.2f}s] {ex['text'][:60]}")

    # save results
    out_json = {
        "method":      args.method,
        "stem":        stem,
        "n_segments":  len(segments),
        "n_blocks":    len(timed_blocks),
        "timing_note": (
            "Within each Whisper segment, caption block start/end times are "
            "distributed proportionally to character count: "
            "t_block_i ∝ chars(block_i) / total_chars_in_segment."
        ),
        "fmt_metrics": metrics,
    }
    res_path = RES_DIR / f"captions_hi_{stem}_{args.method}.json"
    res_path.write_text(json.dumps(out_json, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[captions] results written to {res_path}")


if __name__ == "__main__":
    main()
