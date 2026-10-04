"""
human_eval/analyse_human_eval.py  -  Step 6: human evaluation analysis.

Expects CSV files in  human_eval/  with columns:
  item_id, system, readability, accuracy, naturalness, overall
  (Likert scale 1–5 per criterion)

If no CSVs are found, prints instructions and exits 0.
If CSVs are found: computes mean ± SD per system, inter-rater agreement (Krippendorff alpha),
and writes results_new/human_eval_results.json.
"""
import pathlib, sys, json, time

HE_DIR  = pathlib.Path("human_eval")
RES_DIR = pathlib.Path("results_new"); RES_DIR.mkdir(exist_ok=True)

CRITERIA = ["readability", "accuracy", "naturalness", "overall"]

csv_files = sorted(HE_DIR.glob("*.csv")) if HE_DIR.exists() else []
csv_files  = [f for f in csv_files if not f.name.startswith("KEY")]

if not csv_files:
    print("""
BLOCKED – Step 6: Human Evaluation
====================================
No completed rating sheets found in  human_eval/*.csv

What to do:
1. Prepare caption samples:
   - Select 20–30 blocks from at least 3 systems
     (e.g. raw ASR, CRF-punct, BERT-punct, T5-corrected)
   - Assign each block an  item_id

2. Create a rating CSV template:
   item_id,system,readability,accuracy,naturalness,overall
   (Likert 1–5 for each criterion)
   Name it  human_eval/rater_<name>.csv

3. Have ≥ 2 raters fill it in independently without seeing the system label.
   (Blind evaluation: remove the "system" column from sheets shown to raters.)

4. Keep  human_eval/KEY_do_not_show_raters.csv  with the ground-truth mapping.

5. Re-run:  python human_eval/analyse_human_eval.py

Notes:
- Use ≥ 2 raters to compute inter-rater agreement.
- Bootstrap 95% CIs will be reported over item means.
- Example pilot sheet is in  human_eval/TEMPLATE.csv (if it exists).
""")
    # write a template if none exists
    tmpl = HE_DIR / "TEMPLATE.csv"
    if HE_DIR.exists() and not tmpl.exists():
        tmpl.write_text("item_id,system,readability,accuracy,naturalness,overall\n"
                        "item_001,raw_asr,,,\n"
                        "item_001,crf_punct,,,\n"
                        "item_001,bert_punct,,,\n")
        print(f"Created {tmpl}  as a starting template.")
    sys.exit(0)

# ---- analysis ----
import csv, numpy as np
from collections import defaultdict

print(f"Found {len(csv_files)} rating sheet(s): {[f.name for f in csv_files]}")

# load all sheets
all_rows = []
for f in csv_files:
    with f.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            row["rater"] = f.stem
            all_rows.append(row)

print(f"Loaded {len(all_rows)} ratings")

# mean ± SD per system × criterion
systems = sorted({r["system"] for r in all_rows})
stats   = {}
for sys_name in systems:
    rows = [r for r in all_rows if r["system"] == sys_name]
    stats[sys_name] = {}
    for crit in CRITERIA:
        vals = []
        for r in rows:
            try:
                vals.append(float(r[crit]))
            except (KeyError, ValueError):
                pass
        if vals:
            stats[sys_name][crit] = {
                "mean": round(np.mean(vals), 3),
                "sd":   round(np.std(vals, ddof=1), 3) if len(vals)>1 else 0.0,
                "n":    len(vals),
            }

print("\nSystem means (criterion: mean ± SD):")
for sn, sc in stats.items():
    row_str = "  ".join(f"{c}: {sc[c]['mean']:.2f}±{sc[c]['sd']:.2f}" for c in CRITERIA if c in sc)
    print(f"  {sn:20s}  {row_str}")

# bootstrap 95% CI on overall per system
rng = np.random.default_rng(42)
ci  = {}
for sn in systems:
    vals = []
    for r in all_rows:
        if r["system"] == sn:
            try:
                vals.append(float(r["overall"]))
            except (KeyError, ValueError):
                pass
    if len(vals) > 1:
        samples = rng.choice(vals, size=(2000, len(vals)), replace=True).mean(axis=1)
        ci[sn] = [round(float(np.percentile(samples, 2.5)), 3),
                   round(float(np.percentile(samples, 97.5)), 3)]

# Krippendorff's alpha (ordinal) – only if ≥ 2 raters
alpha_results = {}
if len(csv_files) >= 2:
    try:
        import krippendorff
        for crit in CRITERIA:
            # matrix: raters × items
            items  = sorted({r["item_id"] for r in all_rows})
            raters = sorted({r["rater"]   for r in all_rows})
            matrix = []
            for rater in raters:
                row_vals = []
                rater_data = {(r["item_id"], r["system"]): r
                               for r in all_rows if r["rater"] == rater}
                for item in items:
                    for sn in systems:
                        k = (item, sn)
                        try:
                            row_vals.append(float(rater_data[k][crit]))
                        except (KeyError, ValueError):
                            row_vals.append(np.nan)
                matrix.append(row_vals)
            alpha_results[crit] = round(
                krippendorff.alpha(reliability_data=matrix, level_of_measurement="ordinal"), 3)
        print("\nKrippendorff alpha (ordinal):", alpha_results)
    except ImportError:
        alpha_results = {"note": "pip install krippendorff to compute alpha"}
        print("Install krippendorff package for inter-rater agreement:")
        print("  pip install krippendorff")

R = {
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "n_raters":  len(csv_files),
    "n_items":   len({r["item_id"] for r in all_rows}),
    "systems":   systems,
    "stats":     stats,
    "boot95_overall_CI": ci,
    "krippendorff_alpha": alpha_results,
}
out = RES_DIR / "human_eval_results.json"
out.write_text(json.dumps(R, indent=2))
print(f"\nWrote {out}")
