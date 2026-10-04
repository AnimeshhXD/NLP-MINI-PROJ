"""
tests/test_captions.py  -  Step 5: validate caption file format.

Checks every *.srt and *.vtt in captions/ and results/ against the spec:
  - ≤ 2 lines of text per block
  - ≤ 42 chars per line
  - no word split at line boundary (no hyphenation)
  - timestamps strictly increasing
  - no block overlap
  - SRT block numbers are consecutive integers starting at 1
  - VTT files begin with 'WEBVTT'
  - every block has at least one text line

Run:  pytest tests/test_captions.py -v
"""
import pathlib, re
import pytest

MAX_LINES = 2
MAX_CHARS = 42


# ---- helpers ---------------------------------------------------------------

def parse_srt(path: pathlib.Path):
    """Yield (index, start_ms, end_ms, lines[]) for every SRT block."""
    text   = path.read_text(encoding="utf-8", errors="replace")
    blocks = re.split(r"\n\n+", text.strip())
    for blk in blocks:
        rows = blk.strip().splitlines()
        if len(rows) < 2:
            continue
        idx = rows[0].strip()
        ts  = rows[1].strip()
        m   = re.match(
            r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*"
            r"(\d{2}):(\d{2}):(\d{2}),(\d{3})", ts)
        if not m:
            continue
        def to_ms(h, mi, s, ms): return (int(h)*3600+int(mi)*60+int(s))*1000+int(ms)
        start = to_ms(*m.group(1,2,3,4))
        end   = to_ms(*m.group(5,6,7,8))
        lines = [r.strip() for r in rows[2:] if r.strip()]
        yield {"idx": idx, "start": start, "end": end, "lines": lines}


def parse_vtt(path: pathlib.Path):
    """Yield (start_ms, end_ms, lines[]) for every VTT block."""
    text   = path.read_text(encoding="utf-8", errors="replace")
    blocks = re.split(r"\n\n+", text.strip())
    for blk in blocks:
        rows = blk.strip().splitlines()
        ts_idx = None
        for i, r in enumerate(rows):
            if "-->" in r:
                ts_idx = i; break
        if ts_idx is None:
            continue
        ts = rows[ts_idx].strip()
        m  = re.match(
            r"(\d{2}):(\d{2}):(\d{2})\.(\d{3})\s*-->\s*"
            r"(\d{2}):(\d{2}):(\d{2})\.(\d{3})", ts)
        if not m:
            continue
        def to_ms(h, mi, s, ms): return (int(h)*3600+int(mi)*60+int(s))*1000+int(ms)
        start = to_ms(*m.group(1,2,3,4))
        end   = to_ms(*m.group(5,6,7,8))
        lines = [r.strip() for r in rows[ts_idx+1:] if r.strip()]
        yield {"start": start, "end": end, "lines": lines}


def collect_srt_files():
    srts = list(pathlib.Path("captions").glob("*.srt"))
    srts += list(pathlib.Path("results").glob("*.srt"))
    return srts


def collect_vtt_files():
    vtts = list(pathlib.Path("captions").glob("*.vtt"))
    vtts += list(pathlib.Path("results").glob("*.vtt"))
    return vtts


# ---- SRT tests -------------------------------------------------------------

@pytest.mark.parametrize("srt_path", collect_srt_files())
def test_srt_max_lines(srt_path):
    for b in parse_srt(srt_path):
        assert len(b["lines"]) <= MAX_LINES, \
            f"{srt_path}  block {b['idx']}  has {len(b['lines'])} lines (max {MAX_LINES})"


@pytest.mark.parametrize("srt_path", collect_srt_files())
def test_srt_max_chars(srt_path):
    for b in parse_srt(srt_path):
        for line in b["lines"]:
            assert len(line) <= MAX_CHARS, \
                f"{srt_path}  block {b['idx']}  line '{line}' is {len(line)} chars (max {MAX_CHARS})"


@pytest.mark.parametrize("srt_path", collect_srt_files())
def test_srt_no_mid_word_break(srt_path):
    for b in parse_srt(srt_path):
        lines = b["lines"]
        for i in range(len(lines) - 1):
            assert not lines[i].endswith("-"), \
                f"{srt_path}  block {b['idx']}  line ends with hyphen (mid-word break)"


@pytest.mark.parametrize("srt_path", collect_srt_files())
def test_srt_timestamps_increasing(srt_path):
    prev_end = -1
    for b in parse_srt(srt_path):
        assert b["start"] >= prev_end, \
            f"{srt_path}  block {b['idx']}  starts before previous block ends"
        assert b["end"] > b["start"], \
            f"{srt_path}  block {b['idx']}  end <= start"
        prev_end = b["end"]


@pytest.mark.parametrize("srt_path", collect_srt_files())
def test_srt_no_overlap(srt_path):
    blocks = list(parse_srt(srt_path))
    for i in range(1, len(blocks)):
        assert blocks[i]["start"] >= blocks[i-1]["end"], \
            f"{srt_path}  blocks {blocks[i-1]['idx']} and {blocks[i]['idx']} overlap"


@pytest.mark.parametrize("srt_path", collect_srt_files())
def test_srt_consecutive_numbers(srt_path):
    for expected, b in enumerate(parse_srt(srt_path), 1):
        assert b["idx"] == str(expected), \
            f"{srt_path}  expected block {expected}, found {b['idx']}"


@pytest.mark.parametrize("srt_path", collect_srt_files())
def test_srt_has_text(srt_path):
    for b in parse_srt(srt_path):
        assert len(b["lines"]) >= 1, \
            f"{srt_path}  block {b['idx']}  has no text"


# ---- VTT tests -------------------------------------------------------------

@pytest.mark.parametrize("vtt_path", collect_vtt_files())
def test_vtt_webvtt_header(vtt_path):
    first_line = vtt_path.read_text(encoding="utf-8", errors="replace").lstrip()[:8]
    assert first_line.startswith("WEBVTT"), \
        f"{vtt_path}  does not start with 'WEBVTT'"


@pytest.mark.parametrize("vtt_path", collect_vtt_files())
def test_vtt_max_lines(vtt_path):
    for b in parse_vtt(vtt_path):
        assert len(b["lines"]) <= MAX_LINES, \
            f"{vtt_path}  block has {len(b['lines'])} lines"


@pytest.mark.parametrize("vtt_path", collect_vtt_files())
def test_vtt_max_chars(vtt_path):
    for b in parse_vtt(vtt_path):
        for line in b["lines"]:
            assert len(line) <= MAX_CHARS, \
                f"{vtt_path}  line '{line}' is {len(line)} chars"


@pytest.mark.parametrize("vtt_path", collect_vtt_files())
def test_vtt_timestamps_increasing(vtt_path):
    prev_end = -1
    for b in parse_vtt(vtt_path):
        assert b["start"] >= prev_end, "timestamps not increasing"
        assert b["end"] > b["start"], "end <= start"
        prev_end = b["end"]


@pytest.mark.parametrize("vtt_path", collect_vtt_files())
def test_vtt_no_overlap(vtt_path):
    blocks = list(parse_vtt(vtt_path))
    for i in range(1, len(blocks)):
        assert blocks[i]["start"] >= blocks[i-1]["end"], "overlapping blocks"
