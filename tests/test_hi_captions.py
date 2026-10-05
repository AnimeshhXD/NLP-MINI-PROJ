"""
tests/test_hi_captions.py — Format and integrity tests for Hindi caption files.

Run on the FULL SRT/VTT output (not a sample).
Usage:
  pytest tests/test_hi_captions.py -v --stem hi_foo --method translate
  (or set HI_STEM and HI_METHOD env vars)

Tests:
  1. No block exceeds MAX_LINES=2 or MAX_CHARS=42 per line
  2. Words in SRT (joined) match words from source English text exactly
     (a mid-word split must FAIL this test)
  3. Timestamps strictly increasing, no overlap, inside audio length
  4. SRT numbering is continuous (1, 2, 3, ...)
  5. VTT starts with 'WEBVTT'
  6. CPS <= 17 for >=95% of blocks (print % and exceptions)
"""
import os, re, pathlib, json, sys
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

# ── fixtures: find the right files ─────────────────────────────────────────────
STEM   = os.environ.get("HI_STEM",   None)
METHOD = os.environ.get("HI_METHOD", "translate")
CAP_DIR = pathlib.Path("captions")
ASR_DIR = pathlib.Path("asr")

MAX_CHARS = 42
MAX_LINES = 2
MAX_CPS   = 17.0
CPS_PASS_RATE = 0.95


def _srt_path():
    if STEM is None:
        # auto-discover any hi_* srt matching method
        for p in sorted(CAP_DIR.glob(f"hi_*_{METHOD}.srt")):
            return p
    return CAP_DIR / f"{STEM}_{METHOD}.srt"


def _vtt_path():
    p = _srt_path()
    return p.with_suffix(".vtt") if p else None


def _asr_path():
    if STEM is None:
        srt = _srt_path()
        if srt:
            stem = srt.stem.replace(f"_{METHOD}", "")
            return ASR_DIR / f"{stem}.{'nllb' if METHOD=='nllb' else 'translate'}.json"
    return ASR_DIR / f"{STEM}.{'nllb' if METHOD=='nllb' else 'translate'}.json"


# ── parse SRT ──────────────────────────────────────────────────────────────────
def parse_srt(path):
    """Parse an SRT file; return list of {idx, start, end, lines}."""
    blocks = []
    text = pathlib.Path(path).read_text(encoding="utf-8")
    for raw in re.split(r"\n\n+", text.strip()):
        parts = raw.strip().split("\n")
        if len(parts) < 2:
            continue
        try:
            idx = int(parts[0].strip())
        except ValueError:
            continue
        m = re.match(
            r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})",
            parts[1],
        )
        if not m:
            continue
        g = m.groups()
        start = int(g[0])*3600 + int(g[1])*60 + int(g[2]) + int(g[3])/1000
        end   = int(g[4])*3600 + int(g[5])*60 + int(g[6]) + int(g[7])/1000
        lines = parts[2:]
        blocks.append({"idx": idx, "start": start, "end": end, "lines": lines})
    return blocks


# ── helpers ────────────────────────────────────────────────────────────────────
def srt_words(blocks):
    """All word tokens from all caption blocks, in order."""
    words = []
    for b in blocks:
        for line in b["lines"]:
            words.extend(re.findall(r"\S+", line))
    return words


def source_en_words(asr_path):
    """English words from the ASR JSON (ground truth for word-order check)."""
    if not asr_path or not pathlib.Path(asr_path).exists():
        return None
    data = json.loads(pathlib.Path(asr_path).read_text(encoding="utf-8"))
    words = []
    for seg in data.get("segments", []):
        text = seg.get("text", "").strip()
        words.extend(re.findall(r"\S+", text))
    return words


# ── skip if no file ────────────────────────────────────────────────────────────
def _require_srt():
    p = _srt_path()
    if not p or not p.exists():
        pytest.skip(f"SRT not found: {p} — run make_captions_hi.py first")
    return p


# ──────────────────────────────────────────────────────────────────────────────
# Test 1: line length and count
# ──────────────────────────────────────────────────────────────────────────────
def test_line_length_and_count():
    srt_p = _require_srt()
    blocks = parse_srt(srt_p)
    assert blocks, "SRT has no blocks"
    violations = []
    for b in blocks:
        if len(b["lines"]) > MAX_LINES:
            violations.append(f"block {b['idx']}: {len(b['lines'])} lines (max {MAX_LINES})")
        for line in b["lines"]:
            if len(line) > MAX_CHARS:
                violations.append(f"block {b['idx']}: line too long ({len(line)} chars): {line!r}")
    assert not violations, "\n".join(violations)


# ──────────────────────────────────────────────────────────────────────────────
# Test 2: words in SRT == words in source (no mid-word split)
# ──────────────────────────────────────────────────────────────────────────────
def test_word_sequence_matches_source():
    srt_p  = _require_srt()
    asr_p  = _asr_path()
    if not asr_p or not pathlib.Path(asr_p).exists():
        pytest.skip(f"ASR JSON not found: {asr_p}")

    src_words = source_en_words(asr_p)
    cap_words = srt_words(parse_srt(srt_p))

    # strip punctuation for comparison (caption text may have punctuation attached)
    def norm(ws):
        return [re.sub(r"[^\w]", "", w).lower() for w in ws if re.sub(r"[^\w]", "", w)]

    src_n = norm(src_words)
    cap_n = norm(cap_words)

    assert src_n == cap_n, (
        f"Word mismatch: {len(src_n)} source vs {len(cap_n)} caption words.\n"
        f"First diff at index: "
        f"{next((i for i,(a,b) in enumerate(zip(src_n,cap_n)) if a!=b), 'end')}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test 3: timestamps strictly increasing, no overlap, inside audio length
# ──────────────────────────────────────────────────────────────────────────────
def test_timestamps():
    srt_p = _require_srt()
    blocks = parse_srt(srt_p)

    # get audio duration
    asr_p = _asr_path()
    audio_dur = None
    if asr_p and pathlib.Path(asr_p).exists():
        data = json.loads(pathlib.Path(asr_p).read_text(encoding="utf-8"))
        audio_dur = data.get("duration_s")

    errors = []
    for i, b in enumerate(blocks):
        if b["end"] <= b["start"]:
            errors.append(f"block {b['idx']}: end ({b['end']}) <= start ({b['start']})")
        if i > 0:
            prev = blocks[i - 1]
            if b["start"] < prev["end"] - 0.001:   # 1ms tolerance
                errors.append(
                    f"overlap: block {prev['idx']} ends {prev['end']:.3f}, "
                    f"block {b['idx']} starts {b['start']:.3f}"
                )
        if audio_dur and b["end"] > audio_dur + 0.5:
            errors.append(f"block {b['idx']} ends at {b['end']:.3f}s > audio {audio_dur}s")

    assert not errors, "\n".join(errors)


# ──────────────────────────────────────────────────────────────────────────────
# Test 4: SRT numbering continuous
# ──────────────────────────────────────────────────────────────────────────────
def test_srt_numbering():
    srt_p = _require_srt()
    blocks = parse_srt(srt_p)
    for expected, b in enumerate(blocks, 1):
        assert b["idx"] == expected, (
            f"Numbering gap: expected {expected}, found {b['idx']}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Test 5: VTT starts with WEBVTT
# ──────────────────────────────────────────────────────────────────────────────
def test_vtt_header():
    vtt_p = _vtt_path()
    if not vtt_p or not pathlib.Path(vtt_p).exists():
        pytest.skip(f"VTT not found: {vtt_p}")
    content = pathlib.Path(vtt_p).read_text(encoding="utf-8")
    assert content.startswith("WEBVTT"), f"VTT does not start with WEBVTT: {content[:40]!r}"


# ──────────────────────────────────────────────────────────────────────────────
# Test 6: CPS <= 17 for >= 95% of blocks
# ──────────────────────────────────────────────────────────────────────────────
def test_cps():
    srt_p = _require_srt()
    blocks = parse_srt(srt_p)
    n = len(blocks)
    exceptions = []
    for b in blocks:
        text = " ".join(b["lines"])
        dur = b["end"] - b["start"]
        cps = len(text) / max(dur, 0.001)
        if cps > MAX_CPS:
            exceptions.append({"idx": b["idx"], "cps": round(cps,2), "dur": round(dur,2), "text": text[:60]})

    pass_rate = 1.0 - len(exceptions) / max(n, 1)
    print(f"\nCPS pass rate: {pass_rate*100:.1f}% ({n-len(exceptions)}/{n} blocks within {MAX_CPS} cps)")
    if exceptions:
        print(f"CPS exceptions ({len(exceptions)}):")
        for ex in exceptions:
            print(f"  block {ex['idx']}: {ex['cps']:.1f} cps, {ex['dur']:.2f}s — {ex['text']!r}")

    assert pass_rate >= CPS_PASS_RATE, (
        f"CPS pass rate {pass_rate*100:.1f}% < required {CPS_PASS_RATE*100:.0f}%"
    )
