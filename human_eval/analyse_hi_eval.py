"""
human_eval/analyse_hi_eval.py — Analyse completed Hindi adequacy ratings.

Reads:
  human_eval/<stem>_adequacy.csv      (with adequacy_1to5 / fluency_1to5 filled in)
  human_eval/<stem>_adequacy_key.csv  (system identity key)

Computes mean ±SD adequacy and fluency per method (A/translate vs B/nllb).
Prints results; writes results_new/hi_human_eval_<stem>.json.

Usage:
  python human_eval/analyse_hi_eval.py --stem hi_foo
"""
import argparse, csv, json, pathlib, statistics, sys

HE_DIR  = pathlib.Path("human_eval")
RES_DIR = pathlib.Path("results_new"); RES_DIR.mkdir(exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", required=True)
    args = ap.parse_args()
    stem = args.stem

    csv_path = HE_DIR / f"{stem}_adequacy.csv"
    key_path = HE_DIR / f"{stem}_adequacy_key.csv"

    if not csv_path.exists() or not key_path.exists():
        print(f"BLOCKED: rating files not found for {stem}")
        print(f"  expected: {csv_path}")
        print(f"  expected: {key_path}")
        sys.exit(1)

    with open(csv_path, encoding="utf-8") as f:
        ratings = list(csv.DictReader(f))
    with open(key_path, encoding="utf-8") as f:
        keys = {int(r["segment_id"]): r for r in csv.DictReader(f)}

    # check ratings are filled
    incomplete = [r["segment_id"] for r in ratings
                  if not r.get("adequacy_1to5","").strip() or not r.get("fluency_1to5","").strip()]
    if incomplete:
        print(f"BLOCKED: {len(incomplete)} segments have no ratings yet "
              f"(segment_ids: {incomplete[:5]} ...)")
        sys.exit(1)

    # map ratings to system
    scores = {"A_translate": {"adequacy": [], "fluency": []},
              "B_nllb":      {"adequacy": [], "fluency": []}}
    for r in ratings:
        sid = int(r["segment_id"])
        k = keys[sid]
        for col, sys_key in (("english_1", "english_1_is"), ("english_2", "english_2_is")):
            method = k[sys_key]
            try:
                adq = float(r["adequacy_1to5"])
                flu = float(r["fluency_1to5"])
            except ValueError:
                continue
            scores[method]["adequacy"].append(adq)
            scores[method]["fluency"].append(flu)

    results = {}
    for method, vals in scores.items():
        if not vals["adequacy"]:
            continue
        n = len(vals["adequacy"])
        results[method] = {
            "n": n,
            "adequacy_mean": round(statistics.mean(vals["adequacy"]), 2),
            "adequacy_sd":   round(statistics.stdev(vals["adequacy"]), 2) if n > 1 else 0.0,
            "fluency_mean":  round(statistics.mean(vals["fluency"]),   2),
            "fluency_sd":    round(statistics.stdev(vals["fluency"]),  2) if n > 1 else 0.0,
        }
        print(f"{method}: adequacy={results[method]['adequacy_mean']} ±{results[method]['adequacy_sd']}  "
              f"fluency={results[method]['fluency_mean']} ±{results[method]['fluency_sd']}  (n={n})")

    out_path = RES_DIR / f"hi_human_eval_{stem}.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[analyse] wrote {out_path}")


if __name__ == "__main__":
    main()
