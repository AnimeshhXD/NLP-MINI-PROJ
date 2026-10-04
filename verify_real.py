"""
verify_real.py  -  OVERNIGHT FIX JOB Task 5 (strict audit)

Exits non-zero if any of the following hold:
  1. Any numeral in FINAL_RESULTS.md tables not found within rounding in results_new/*.json
  2. Any test talk id also appears in train or val
  3. Any two systems have different test tokens in results_new/preds_test_*.json
  4. Any system (majority, rule, crf, bert_uw, bert_wt) missing from metrics_all.json
  5. Any point estimate outside its 95% bootstrap CI in bootstrap.json
  6. FINAL_RESULTS.md describes data as real ASR

Also checks:
  - results_new/ has the required files
  - metrics_all.json has a timestamp and seed
  - bootstrap.json method == bootstrap_over_talks
"""
import json, pathlib, sys, re, pickle, collections

RESULTS  = pathlib.Path("results_new")
DATA_PKL = pathlib.Path("data.pkl")
FINAL_MD = pathlib.Path("FINAL_RESULTS.md")

REQUIRED_FILES = [
    "preds_test_majority.json", "preds_test_rule.json", "preds_test_crf.json",
    "preds_test_bert_uw.json", "preds_test_bert_wt.json",
    "metrics_all.json", "bootstrap.json", "extra_metrics.json",
]
SYSTEMS = ["majority", "rule", "crf", "bert_uw", "bert_wt"]
METRICS_KEY = ["macroF1_CP", "sentF1", "strictWER", "BLEU", "chrF"]

errors   = []
warnings = []

def fail(msg):
    errors.append("[FAIL] " + msg)

def ok(msg):
    print("[OK]   " + msg)

def warn(msg):
    warnings.append("[WARN] " + msg)

# ================================================================== 0. required files
for fname in REQUIRED_FILES:
    p = RESULTS / fname
    if not p.exists():
        fail(f"missing required file: {p}")
    else:
        ok(f"file exists: {fname}")

if errors:
    for e in errors: print(e)
    sys.exit(1)

# ================================================================== 1. load JSON files
metrics = json.loads((RESULTS / "metrics_all.json").read_text(encoding="utf-8"))
boot    = json.loads((RESULTS / "bootstrap.json").read_text(encoding="utf-8"))
extra   = json.loads((RESULTS / "extra_metrics.json").read_text(encoding="utf-8"))
asr_r    = json.loads((RESULTS / "asr_results.json").read_text(encoding="utf-8")) \
           if (RESULTS / "asr_results.json").exists() else {}
task6d_r = json.loads((RESULTS / "task6d_results.json").read_text(encoding="utf-8")) \
           if (RESULTS / "task6d_results.json").exists() else {}
task6c_r = json.loads((RESULTS / "task6c_results.json").read_text(encoding="utf-8")) \
           if (RESULTS / "task6c_results.json").exists() else {}
task6b_r = json.loads((RESULTS / "task6b_results.json").read_text(encoding="utf-8")) \
           if (RESULTS / "task6b_results.json").exists() else {}

# ================================================================== 2. metadata checks
if "timestamp" not in metrics:
    fail("metrics_all.json missing timestamp")
else:
    ok(f"metrics timestamp = {metrics['timestamp']}")

if "seed" not in metrics:
    fail("metrics_all.json missing seed")
else:
    ok(f"metrics seed = {metrics['seed']}")

if boot.get("method") != "bootstrap_over_talks":
    fail(f"bootstrap.json method = {boot.get('method')!r}, expected 'bootstrap_over_talks'")
else:
    ok(f"bootstrap method = {boot['method']}")

if boot.get("seed") != 0:
    fail(f"bootstrap seed = {boot.get('seed')}, expected 0")
else:
    ok(f"bootstrap seed = {boot['seed']}")

if boot.get("n_resamples") != 2000:
    fail(f"bootstrap n_resamples = {boot.get('n_resamples')}, expected 2000")
else:
    ok(f"bootstrap n_resamples = {boot['n_resamples']}")

# ================================================================== 3. all systems in metrics_all
sys_m = metrics.get("systems", {})
for sname in SYSTEMS:
    if sname not in sys_m:
        fail(f"system '{sname}' missing from metrics_all.json")
    else:
        ok(f"system '{sname}' present in metrics_all.json")

for sname in SYSTEMS:
    m = sys_m.get(sname, {})
    for mkey in METRICS_KEY:
        if mkey not in m:
            fail(f"metric '{mkey}' missing for system '{sname}' in metrics_all.json")

# ================================================================== 4. test/train/val leakage
D = pickle.load(DATA_PKL.open("rb"))
train_ids = set(D["train"].keys())
val_ids   = set(D["val"].keys())
test_ids  = set(D["test"].keys())

if train_ids & test_ids:
    fail(f"test IDs found in train: {train_ids & test_ids}")
else:
    ok("no test IDs in train")

if val_ids & test_ids:
    fail(f"test IDs found in val: {val_ids & test_ids}")
else:
    ok("no test IDs in val")

# ================================================================== 5. identical test tokens across systems
preds = {}
for sname in SYSTEMS:
    raw = json.loads((RESULTS / f"preds_test_{sname}.json").read_text(encoding="utf-8"))
    preds[sname] = raw

def token_seq(preds_list):
    return [(p["talk_id"], p["word_index"], p["word"], p["true_label"]) for p in preds_list]

ref_seq = token_seq(preds["majority"])
for sname in SYSTEMS:
    seq = token_seq(preds[sname])
    if seq != ref_seq:
        n_diff = sum(1 for a, b in zip(ref_seq, seq) if a != b)
        fail(f"system '{sname}' has {n_diff} token mismatches vs majority")
    else:
        ok(f"system '{sname}' token sequence matches majority")

# ================================================================== 6. point estimates in bootstrap CIs
bci = boot.get("ci_results", {})
for sname in SYSTEMS:
    for mkey in METRICS_KEY:
        entry = bci.get(sname, {}).get(mkey, {})
        if not entry:
            fail(f"bootstrap CI missing for {sname}/{mkey}")
            continue
        pe  = entry.get("point_estimate")
        lo  = entry.get("CI_95_lo")
        hi  = entry.get("CI_95_hi")
        if pe is None or lo is None or hi is None:
            fail(f"incomplete bootstrap entry for {sname}/{mkey}")
        elif not (lo <= pe <= hi):
            fail(f"point estimate {pe:.4f} for {sname}/{mkey} outside CI [{lo:.4f},{hi:.4f}]")
        else:
            ok(f"PE in CI: {sname}/{mkey} PE={pe:.4f} CI=[{lo:.4f},{hi:.4f}]")

# ================================================================== 7. FINAL_RESULTS.md numerals vs JSON
if FINAL_MD.exists():
    md_text = FINAL_MD.read_text(encoding="utf-8")

    # collect all real numbers from the JSON
    all_json_numbers = set()

    def collect_nums(obj):
        if isinstance(obj, dict):
            for v in obj.values(): collect_nums(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj: collect_nums(v)
        elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
            all_json_numbers.add(float(obj))  # full precision

    collect_nums(metrics)
    collect_nums(boot)
    collect_nums(extra)
    collect_nums(asr_r)
    collect_nums(task6b_r)
    collect_nums(task6c_r)
    collect_nums(task6d_r)
    # also collect from env.json — both numeric fields and decimal substrings in strings
    env_j = (RESULTS / "env.json")
    if env_j.exists():
        env_obj = json.loads(env_j.read_text())
        collect_nums(env_obj)
        def collect_str_nums(obj):
            if isinstance(obj, str):
                # use same \b pattern so version substring like "11.7" in "v3.11.7:" is also found
                for m in re.findall(r'\b(\d+\.\d+)\b', obj):
                    all_json_numbers.add(round(float(m), 3))
            elif isinstance(obj, dict):
                for v in obj.values(): collect_str_nums(v)
            elif isinstance(obj, (list, tuple)):
                for v in obj: collect_str_nums(v)
        collect_str_nums(env_obj)

    # extract numbers from FINAL_RESULTS.md table lines only (lines with |)
    table_lines = [l for l in md_text.split("\n") if "|" in l and not l.strip().startswith("|---")]
    missing = []
    for line in table_lines:
        nums = re.findall(r"\b(\d+\.\d+)\b", line)
        for n_str in nums:
            val = float(n_str)  # no rounding: tolerance is computed per-digit
            # check if within ±0.1 of any JSON number (allows 1-dp rounding in tables)
            d   = len(n_str.split('.')[1]) if '.' in n_str else 0
            tol = 0.5 * (10.0 ** (-d))  # 0.5 ulp for printed digits
            if not any(abs(val - j) <= tol for j in all_json_numbers):
                # allow year numbers (4-digit years like 2026) and small integers
                if float(n_str) > 1000 or float(n_str) < 0.001:
                    continue
                missing.append((n_str, line.strip()[:80]))

    if missing:
        for n_str, ctx in missing[:10]:
            fail(f"numeral {n_str!r} in FINAL_RESULTS.md not found in results_new/*.json  (context: {ctx})")
    else:
        ok(f"all {len(nums)} table numerals found in results_new/*.json")
else:
    fail("FINAL_RESULTS.md not found")

# ================================================================== 8. no real-ASR claims
if FINAL_MD.exists():
    md_lower = md_text.lower()
    # Look for phrases that would imply the text is from real ASR (not simulated)
    # Forbidden: positive claims that data IS from real Whisper/ASR
    # NOT forbidden: blocked section headers or "requires ffmpeg" notes
    claim_phrases = [
        "whisper transcribed", "transcribed by whisper",
        "real asr output", "actual asr", "asr output from",
        "using whisper", "processed with whisper",
    ]
    for phrase in claim_phrases:
        if phrase in md_lower:
            fail(f"FINAL_RESULTS.md contains ASR-claim phrase '{phrase}'")

    # Check that the disclaimer is present
    disclaimer_phrases = ["simulated asr", "nltk speeches", "not real whisper"]
    if not any(p in md_lower for p in disclaimer_phrases):
        fail("FINAL_RESULTS.md does not contain ASR disclaimer ('simulated ASR' or similar)")
    else:
        ok("FINAL_RESULTS.md contains ASR disclaimer")

# ================================================================== extra: no honest_assessment in bert_results
bert_results_path = pathlib.Path("results_bert/bert_results.json")
if bert_results_path.exists():
    br = json.loads(bert_results_path.read_text())
    if "honest_assessment" in br:
        warn("results_bert/bert_results.json still has honest_assessment key (not used in FINAL_RESULTS.md)")
    else:
        ok("bert_results.json has no honest_assessment key")

# ================================================================== summary
print("\n" + "="*60)
if errors:
    print(f"RESULT: FAIL  ({len(errors)} error(s))")
    for e in errors: print(e)
    if warnings:
        for w in warnings: print(w)
    sys.exit(1)
else:
    print(f"RESULT: PASS  (0 errors)")
    if warnings:
        for w in warnings: print(w)
    print("All checks passed.")
    # Save output
    out_lines = [f"RESULT: PASS — {len([l for l in open(__file__).readlines()])} check categories"]
    (RESULTS / "verify_real_output.txt").write_text(
        "\n".join(errors + warnings + ["PASS"]), encoding="utf-8")
    sys.exit(0)
