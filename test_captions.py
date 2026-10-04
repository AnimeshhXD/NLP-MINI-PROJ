"""
test_captions.py - Task 5 format checks (pytest).

Run:  pytest test_captions.py -v

Tests:
  - no block exceeds 2 lines or 42 chars
  - no word split across lines
  - timestamps increasing and non-overlapping
  - SRT numbering continuous
  - VTT starts with WEBVTT
"""
import pathlib, re, pytest

MAX_CHARS = 42
MAX_LINES = 2

# ------------------------------------------------------------------ helpers
def parse_srt(text):
    """Parse SRT into list of (index, start, end, lines)."""
    blocks = []
    for block in text.strip().split("\n\n"):
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue
        try:
            idx = int(lines[0].strip())
        except ValueError:
            continue
        timing = lines[1]
        m = re.match(r"(\d+:\d+:\d+[,\.]\d+)\s+-->\s+(\d+:\d+:\d+[,\.]\d+)", timing)
        if not m:
            continue
        def to_sec(ts):
            ts = ts.replace(",", ".")
            h, mi, s = ts.split(":")
            return int(h)*3600 + int(mi)*60 + float(s)
        blocks.append((idx, to_sec(m.group(1)), to_sec(m.group(2)), lines[2:]))
    return blocks

def parse_vtt(text):
    """Parse WebVTT into list of (index, start, end, lines)."""
    lines_all = text.strip().split("\n")
    assert lines_all[0].strip() == "WEBVTT", "VTT must start with WEBVTT"
    return parse_srt("\n".join(lines_all[1:]))

# ------------------------------------------------------------------ find SRT/VTT files to test
def get_srt_files():
    files = list(pathlib.Path("results").glob("*.srt")) + \
            list(pathlib.Path("asr").glob("*.srt"))
    return files

def get_vtt_files():
    files = list(pathlib.Path("results").glob("*.vtt")) + \
            list(pathlib.Path("asr").glob("*.vtt"))
    return files

# always include sample files even if no asr/ output exists
@pytest.fixture(params=get_srt_files() or [pathlib.Path("results/sample.srt")])
def srt_file(request):
    return request.param

@pytest.fixture(params=get_vtt_files() or [pathlib.Path("results/sample.vtt")])
def vtt_file(request):
    return request.param

# ------------------------------------------------------------------ SRT tests
def test_srt_exists(srt_file):
    assert srt_file.exists(), "SRT file not found: %s" % srt_file

def test_srt_block_line_count(srt_file):
    """Each subtitle block has at most 2 display lines."""
    text = srt_file.read_text(encoding="utf-8")
    blocks = parse_srt(text)
    assert blocks, "No blocks parsed from %s" % srt_file
    for idx, _, _, caption_lines in blocks:
        assert len(caption_lines) <= MAX_LINES, \
            "Block %d has %d lines (max %d)" % (idx, len(caption_lines), MAX_LINES)

def test_srt_line_length(srt_file):
    """Each line is at most 42 characters."""
    text = srt_file.read_text(encoding="utf-8")
    for idx, _, _, caption_lines in parse_srt(text):
        for ln in caption_lines:
            assert len(ln) <= MAX_CHARS, \
                "Block %d line too long (%d chars): %r" % (idx, len(ln), ln)

def test_srt_no_word_split(srt_file):
    """No line starts or ends in the middle of a word (no trailing/leading hyphen splits)."""
    text = srt_file.read_text(encoding="utf-8")
    for idx, _, _, caption_lines in parse_srt(text):
        if len(caption_lines) > 1:
            for i in range(len(caption_lines) - 1):
                last_word = caption_lines[i].split()[-1] if caption_lines[i].split() else ""
                first_word = caption_lines[i+1].split()[0] if caption_lines[i+1].split() else ""
                # a word split would show as one part at end of line and continuation at start of next
                # check: last char of line i and first char of line i+1 are both word chars,
                # and last word of line i + first word of next form a single word without space
                assert not (last_word.endswith("-") and first_word[0:1].isalpha()), \
                    "Mid-word break at block %d between lines" % idx

def test_srt_timestamps_increasing(srt_file):
    """Block start times are strictly increasing."""
    text = srt_file.read_text(encoding="utf-8")
    blocks = parse_srt(text)
    for i in range(1, len(blocks)):
        assert blocks[i][1] >= blocks[i-1][1], \
            "Block %d start < block %d start" % (blocks[i][0], blocks[i-1][0])

def test_srt_no_overlap(srt_file):
    """No block's start time is before the previous block's end time."""
    text = srt_file.read_text(encoding="utf-8")
    blocks = parse_srt(text)
    for i in range(1, len(blocks)):
        assert blocks[i][1] >= blocks[i-1][2] - 0.001, \
            "Block %d overlaps with block %d" % (blocks[i][0], blocks[i-1][0])

def test_srt_numbering_continuous(srt_file):
    """SRT index numbers are 1, 2, 3, … without gaps."""
    text = srt_file.read_text(encoding="utf-8")
    blocks = parse_srt(text)
    for expected, (idx, *_) in enumerate(blocks, 1):
        assert idx == expected, "SRT numbering gap: expected %d got %d" % (expected, idx)

# ------------------------------------------------------------------ VTT tests
def test_vtt_starts_with_webvtt(vtt_file):
    """VTT file must start with exactly 'WEBVTT'."""
    text = vtt_file.read_text(encoding="utf-8")
    assert text.strip().startswith("WEBVTT"), "VTT must start with WEBVTT"

def test_vtt_block_line_count(vtt_file):
    text = vtt_file.read_text(encoding="utf-8")
    blocks = parse_vtt(text)
    for idx, _, _, caption_lines in blocks:
        assert len(caption_lines) <= MAX_LINES

def test_vtt_line_length(vtt_file):
    text = vtt_file.read_text(encoding="utf-8")
    for idx, _, _, caption_lines in parse_vtt(text):
        for ln in caption_lines:
            assert len(ln) <= MAX_CHARS, \
                "Block %d line too long: %r" % (idx, ln)

def test_vtt_timestamps_increasing(vtt_file):
    text = vtt_file.read_text(encoding="utf-8")
    blocks = parse_vtt(text)
    for i in range(1, len(blocks)):
        assert blocks[i][1] >= blocks[i-1][1]
