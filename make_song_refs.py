"""
make_song_refs.py - build the PRIVATE reference files for the sung-audio test, locally.

Run (use your real file names; quote paths that contain spaces):
  python make_song_refs.py --lyrics refs\\lyrics_full.txt --captions refs\\yt_captions_raw.txt.txt --first-lines 8 --start 0 --end 65

Writes:
  refs/hi_song.txt                 first N lyric lines (the Hindi reference for the excerpt)
  refs/hi_song_youtube_auto.txt    auto-caption lines that start inside [start, end) seconds, music tags removed
Prints only counts and layout, never the text. Adds the private files to .gitignore.
Handles caption dumps where the time, the "N seconds" label and the text are on separate lines,
on one line, or separated by spaces; and files saved as UTF-8, UTF-8 with BOM, or UTF-16.
"""
import argparse, os, re

TAG = re.compile(r"^\s*[\[\(][^\]\)]*[\]\)]\s*$")                      # [संगीत] / [Music]
TIME_ONLY = re.compile(r"^(\d+):(\d{2})$")                              # 1:05
CONCAT = re.compile(r"^(\d+):(\d{2})\s*(\d+\s+minutes?(?:,\s*\d+\s+seconds?)?|\d+\s+seconds?)\s*(.*)$", re.I)  # 0:099 secondsText
LABEL = re.compile(r"^\d+\s+(minutes?|seconds?)(,\s*\d+\s+seconds?)?$", re.I)  # "1 minute, 5 seconds"

def read_text(path):
    raw = open(path, "rb").read()
    for enc in ("utf-8-sig", "utf-16"):
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            pass
    return raw.decode("utf-8", errors="replace"), "utf-8 (with replacement)"

def read_lines(path):
    text, _ = read_text(path)
    return [l.strip() for l in text.splitlines()]

def parse_captions(path):
    """Return [(start_seconds, text)] from the pasted caption dump."""
    out, cur_t, cur_txt = [], None, []
    def flush():
        if cur_t is not None and cur_txt:
            out.append((cur_t, " ".join(cur_txt)))
    for ln in read_lines(path):
        if not ln:
            continue
        m = TIME_ONLY.match(ln)
        c = CONCAT.match(ln)
        if m:                                           # "0:02" on its own line
            flush(); cur_t, cur_txt = int(m[1]) * 60 + int(m[2]), []
        elif c:                                         # time + label (+ text) on one line
            flush(); cur_t, cur_txt = int(c[1]) * 60 + int(c[2]), []
            if c[4].strip():
                cur_txt.append(c[4].strip())
        elif cur_t is not None and not LABEL.match(ln):  # text line
            cur_txt.append(ln)
    flush()
    return out

def layout(path, n=8):
    """Show the first lines with every non-ASCII character replaced by X, so the layout can be seen without printing text."""
    text, enc = read_text(path)
    lines = [l for l in text.splitlines() if l.strip()]
    print("   file encoding read as: %s | non-empty lines: %d" % (enc, len(lines)))
    for l in lines[:n]:
        print("   | " + "".join(ch if ord(ch) < 128 else "X" for ch in l.strip())[:70])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lyrics", default="private/lyrics_full.txt")
    ap.add_argument("--captions", default="private/yt_captions_raw.txt")
    ap.add_argument("--first-lines", type=int, default=8)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=65.0)
    a = ap.parse_args()
    os.makedirs("refs", exist_ok=True)

    lyrics = [l for l in read_lines(a.lyrics) if l]
    open("refs/hi_song.txt", "w", encoding="utf-8").write("\n".join(lyrics[:a.first_lines]) + "\n")
    print("lyrics: %d lines in file, %d written to refs/hi_song.txt" % (len(lyrics), min(a.first_lines, len(lyrics))))

    allcaps = parse_captions(a.captions)
    caps = [(t, x) for t, x in allcaps if a.start <= t < a.end and not TAG.match(x)]
    open("refs/hi_song_youtube_auto.txt", "w", encoding="utf-8").write("\n".join(x for _, x in caps) + "\n")
    print("auto-captions: parsed %d caption entries in total (first start %s s, last start %s s); %d inside [%.0f, %.0f) written to refs/hi_song_youtube_auto.txt"
          % (len(allcaps), allcaps[0][0] if allcaps else "-", allcaps[-1][0] if allcaps else "-", len(caps), a.start, a.end))
    if not allcaps:
        print("!! nothing could be parsed. Layout of the captions file (non-ASCII shown as X):")
        layout(a.captions)

    want = ["private/", "refs/hi_song*", "videos/hi_song*", "videos/hindi.mp4"]
    have = open(".gitignore", encoding="utf-8").read().splitlines() if os.path.exists(".gitignore") else []
    add = [w for w in want if w not in have]
    if add:
        with open(".gitignore", "a", encoding="utf-8") as f:
            f.write("\n# copyrighted song material - never commit\n" + "\n".join(add) + "\n")
    print(".gitignore: added %d pattern(s)" % len(add))

if __name__ == "__main__":
    main()