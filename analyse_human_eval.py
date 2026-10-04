"""
analyse_human_eval.py - Task 6: Analyse human evaluation rating sheets.

Expected input:
  human_eval_returned/*.csv  -- one CSV per rater, columns:
      item_id, system, readability, accuracy, naturalness, overall
  human_eval_KEY_do_not_show_raters.csv -- same columns, plus correct answers
      (item_id, correct_text)

Outputs: logs/human_eval.json with mean ± SD per system and criterion,
         paired bootstrap differences, and mean pairwise Spearman agreement.

If no sheets exist, prints instructions and exits cleanly.
"""
import json, pathlib, sys
import numpy as np
import scipy.stats as stats
from itertools import combinations

LOG_FILE = pathlib.Path("logs/human_eval.json")
LOG_FILE.parent.mkdir(exist_ok=True)

RETURNED_DIR = pathlib.Path("human_eval_returned")
KEY_FILE     = pathlib.Path("human_eval_KEY_do_not_show_raters.csv")
CRITERIA     = ["readability", "accuracy", "naturalness", "overall"]

# ------------------------------------------------------------------ check for data
sheets = list(RETURNED_DIR.glob("*.csv")) if RETURNED_DIR.exists() else []
if not sheets:
    msg = (
        "No rating sheets found in human_eval_returned/.\n\n"
        "To run human evaluation:\n"
        "1. Create rating sheets with columns:\n"
        "   item_id, system, readability, accuracy, naturalness, overall\n"
        "   (scale 1-5 per criterion)\n"
        "2. Have raters fill them in independently.\n"
        "3. Place completed CSVs in human_eval_returned/.\n"
        "4. Re-run: python analyse_human_eval.py\n"
    )
    print(msg)
    LOG_FILE.write_text(json.dumps({"status": "no_sheets", "message": msg}))
    sys.exit(0)

# ------------------------------------------------------------------ load sheets
import csv

def load_csv(path):
    """Load a CSV file into a list of dicts."""
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

rater_data = {}   # {rater_id: [{item_id, system, readability, …}]}
for s in sheets:
    rater_id = s.stem
    rows = load_csv(s)
    rater_data[rater_id] = rows

# ------------------------------------------------------------------ merge all ratings
all_rows = []
for rater_id, rows in rater_data.items():
    for row in rows:
        row["rater"] = rater_id
        # convert ratings to float
        for c in CRITERIA:
            try: row[c] = float(row[c])
            except: row[c] = None
        all_rows.append(row)

systems = sorted(set(r["system"] for r in all_rows if r.get("system")))

# ------------------------------------------------------------------ mean ± SD per system
def summarise(rows, system, crit):
    """Return (mean, sd, n) for a criterion/system subset."""
    vals = [r[crit] for r in rows if r.get("system") == system and r.get(crit) is not None]
    if not vals: return None, None, 0
    return float(np.mean(vals)), float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0), len(vals)

results = {"by_system": {}}
for sys_name in systems:
    results["by_system"][sys_name] = {}
    for c in CRITERIA:
        mn, sd, n = summarise(all_rows, sys_name, c)
        results["by_system"][sys_name][c] = {"mean": mn, "sd": sd, "n": n}

# ------------------------------------------------------------------ paired bootstrap differences
def paired_bootstrap_rows(rows_a, rows_b, crit, n=1000):
    """Bootstrap p-value: is system A (rows_a) better than B on crit?"""
    # match by item_id
    a_by_item = {r["item_id"]: r[crit] for r in rows_a if r.get(crit) is not None}
    b_by_item = {r["item_id"]: r[crit] for r in rows_b if r.get(crit) is not None}
    common = sorted(set(a_by_item) & set(b_by_item))
    if len(common) < 2: return None, None, 0
    diffs = np.array([a_by_item[k] - b_by_item[k] for k in common])
    obs = float(np.mean(diffs))
    bs  = [float(np.mean(np.random.choice(diffs, len(diffs), replace=True)))
           for _ in range(n)]
    p = float(np.mean(np.array(bs) < 0))
    return obs, p, len(common)

pair_results = {}
for (s1, s2) in combinations(systems, 2):
    r1 = [r for r in all_rows if r["system"] == s1]
    r2 = [r for r in all_rows if r["system"] == s2]
    pair_results["%s_vs_%s" % (s1, s2)] = {}
    for c in CRITERIA:
        obs, p, n = paired_bootstrap_rows(r1, r2, c)
        pair_results["%s_vs_%s" % (s1, s2)][c] = {"delta": obs, "p_A_worse": p, "n": n}
results["pairwise_bootstrap"] = pair_results

# ------------------------------------------------------------------ Spearman inter-rater
def mean_pairwise_spearman(crit):
    """Mean pairwise Spearman correlation across raters for a criterion."""
    rater_ids = list(rater_data.keys())
    if len(rater_ids) < 2: return None
    # build rating vectors aligned by item_id+system
    def vec(rater_id):
        return {(r["item_id"], r["system"]): r[crit]
                for r in rater_data[rater_id] if r.get(crit) is not None}
    corrs = []
    for r1, r2 in combinations(rater_ids, 2):
        v1, v2 = vec(r1), vec(r2)
        keys = sorted(set(v1) & set(v2))
        if len(keys) < 3: continue
        a, b = [v1[k] for k in keys], [v2[k] for k in keys]
        rho, _ = stats.spearmanr(a, b)
        corrs.append(float(rho))
    return float(np.mean(corrs)) if corrs else None

results["inter_rater_spearman"] = {c: mean_pairwise_spearman(c) for c in CRITERIA}

# ------------------------------------------------------------------ print summary
print("\n=== Human Evaluation Summary ===")
for sys_name in systems:
    print("\n%s:" % sys_name)
    for c in CRITERIA:
        mn, sd, n = (results["by_system"][sys_name][c]["mean"],
                     results["by_system"][sys_name][c]["sd"],
                     results["by_system"][sys_name][c]["n"])
        if mn is not None:
            print("  %-15s %.2f ± %.2f  (n=%d)" % (c, mn, sd or 0, n))

print("\nInter-rater Spearman:")
for c in CRITERIA:
    r = results["inter_rater_spearman"][c]
    print("  %-15s %.3f" % (c, r if r is not None else float("nan")))

LOG_FILE.write_text(json.dumps(results, indent=2))
print("\nWrote", LOG_FILE)
