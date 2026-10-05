"""
human_eval/gen_hi_adequacy.py — Generate adequacy rating sheet for Hindi captions.

Picks 20 random segments per clip.  For each segment writes:
  - Hindi source text (from Whisper transcribe or refs/hi_<stem>.txt)
  - English from Method A (translate)
  - English from Method B (NLLB)
  - Two blank columns: adequacy_1to5, fluency_1to5

To hide which system is which, the two English columns are shuffled randomly
per segment; a separate key file records the order.

Usage:
  python human_eval/gen_hi_adequacy.py --stem hi_foo [--n 20] [--seed 42]

Outputs:
  human_eval/hi_<stem>_adequacy.csv       (to fill in)
  human_eval/hi_<stem>_adequacy_key.csv   (system identity key)
"""
import argparse, csv, json, pathlib, random, sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

HE_DIR  = pathlib.Path("human_eval");  HE_DIR.mkdir(exist_ok=True)
ASR_DIR = pathlib.Path("asr")
REF_DIR = pathlib.Path("refs")


def load_hi_segments(stem):
    p = ASR_DIR / f"{stem}.transcribe.json"
    if p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))
        return [(s["start"], s["end"], s["text"].strip()) for s in data["segments"]]
    # fallback: split refs/hi_<stem>.txt by danda
    ref = REF_DIR / f"{stem}.txt"
    if ref.exists():
        import re
        sents = [s.strip() for s in re.split(r"[।\n]+", ref.read_text(encoding="utf-8")) if s.strip()]
        return [(0.0, 0.0, s) for s in sents]
    return []


def load_en_segments(stem, method):
    key = "translate" if method == "translate" else "nllb"
    p = ASR_DIR / f"{stem}.{key}.json"
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    # map by (start, end) rounded to 1dp for matching
    return {(round(s["start"],1), round(s["end"],1)): s.get("text","").strip()
            for s in data["segments"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", required=True)
    ap.add_argument("--n",    type=int, default=20, help="segments to sample")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    stem = args.stem

    hi_segs = load_hi_segments(stem)
    if not hi_segs:
        print(f"BLOCKED: no Hindi segments for {stem} (run run_whisper_hi.py first or provide refs/{stem}.txt)")
        sys.exit(1)

    en_a = load_en_segments(stem, "translate")
    en_b = load_en_segments(stem, "nllb")

    # sample n segments that have at least one English translation
    pool = []
    for t_start, t_end, hi_text in hi_segs:
        key = (round(t_start, 1), round(t_end, 1))
        txt_a = en_a.get(key, "")
        txt_b = en_b.get(key, "")
        if hi_text:
            pool.append((t_start, t_end, hi_text, txt_a, txt_b))

    n = min(args.n, len(pool))
    sampled = rng.sample(pool, n)
    sampled.sort(key=lambda x: x[0])   # sort by time

    # shuffle A/B order per segment; record key separately
    out_rows  = []
    key_rows  = []
    for i, (t_s, t_e, hi, txt_a, txt_b) in enumerate(sampled, 1):
        order = rng.choice(["AB", "BA"])
        if order == "AB":
            col1, col2 = txt_a, txt_b
            sys1, sys2 = "A_translate", "B_nllb"
        else:
            col1, col2 = txt_b, txt_a
            sys1, sys2 = "B_nllb", "A_translate"
        out_rows.append({
            "segment_id":    i,
            "time_start":    round(t_s, 2),
            "time_end":      round(t_e, 2),
            "hindi_source":  hi,
            "english_1":     col1,
            "english_2":     col2,
            "adequacy_1to5": "",
            "fluency_1to5":  "",
        })
        key_rows.append({
            "segment_id":  i,
            "english_1_is": sys1,
            "english_2_is": sys2,
        })

    csv_path = HE_DIR / f"{stem}_adequacy.csv"
    key_path = HE_DIR / f"{stem}_adequacy_key.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_rows[0].keys())
        w.writeheader(); w.writerows(out_rows)

    with open(key_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=key_rows[0].keys())
        w.writeheader(); w.writerows(key_rows)

    print(f"[gen_adequacy] wrote {csv_path}  ({n} segments)")
    print(f"[gen_adequacy] wrote {key_path}  (key — share only AFTER ratings collected)")


if __name__ == "__main__":
    main()
