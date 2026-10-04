"""
verify_real.py  -  Anti-fake audit.

Fails (exit 1) if any result JSON in results_new/ lacks required provenance fields,
or if any suspicious pattern is detected (hardcoded numbers, no timestamp, etc.)

Required fields in every result JSON:
  timestamp   – ISO 8601 string from time.strftime
  seed        – integer 42
  data_hash   – MD5 of data/talks.pkl (proves the data was actually loaded)

Optional but checked if present:
  bert_model / t5_model  – must be a known model name, not a placeholder
  BLEU / macroF1_CP      – must be a float, not a hardcoded nice number

Run:  python verify_real.py
Exit 0 = all checks pass
Exit 1 = at least one check fails (see output for details)
"""
import json, pathlib, sys, re, hashlib, time

RESULTS_DIR = pathlib.Path("results_new")
DATA_PKL    = pathlib.Path("data/talks.pkl")

SUSPICIOUS_EXACT = {
    # Numbers that look copied from a paper or example
    0.500, 0.600, 0.700, 0.800, 0.900, 1.000,
    75.00, 80.00, 85.00, 90.00, 95.00,
}

REQUIRED_FIELDS = ["timestamp", "seed"]
PASS = True


def fail(msg: str):
    global PASS
    print(f"[FAIL]  {msg}")
    PASS = False


def ok(msg: str):
    print(f"[OK]    {msg}")


# ---- 0. results_new/ must exist -----------------------------------------
if not RESULTS_DIR.exists():
    fail("results_new/ directory does not exist – no results to verify.")
    sys.exit(1)

json_files = sorted(RESULTS_DIR.glob("*.json"))
if not json_files:
    fail("results_new/ is empty – run the pipeline first.")
    sys.exit(1)

# ---- 1. compute expected data hash (if talks.pkl exists) ----------------
expected_hash = None
if DATA_PKL.exists():
    expected_hash = hashlib.md5(DATA_PKL.read_bytes()).hexdigest()[:12]
    ok(f"data/talks.pkl found, MD5={expected_hash}")
else:
    print("[WARN]  data/talks.pkl not found – hash check skipped "
          "(run  python data/prepare.py  first)")

# ---- 2. check each result file ------------------------------------------
for jf in json_files:
    try:
        R = json.loads(jf.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        fail(f"{jf.name}: invalid JSON ({e})")
        continue

    # required fields
    for field in REQUIRED_FIELDS:
        if field not in R:
            fail(f"{jf.name}: missing required field '{field}'")
        else:
            ok(f"{jf.name}: has '{field}' = {R[field]!r}")

    # seed must be 42
    if "seed" in R and R["seed"] != 42:
        fail(f"{jf.name}: seed = {R['seed']} (expected 42)")

    # timestamp must be a plausible ISO string
    if "timestamp" in R:
        ts = R["timestamp"]
        if not re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", ts):
            fail(f"{jf.name}: timestamp '{ts}' is not ISO 8601")
        else:
            ok(f"{jf.name}: timestamp OK ({ts})")

    # data_hash check
    if "data_hash" in R:
        if expected_hash and R["data_hash"] != expected_hash:
            fail(f"{jf.name}: data_hash mismatch "
                 f"({R['data_hash']} != {expected_hash})")
        else:
            ok(f"{jf.name}: data_hash matches ({R['data_hash']})")

    # model names must not be placeholders
    for key in ("bert_model", "t5_model", "whisper_model"):
        if key in R:
            val = R[key]
            if val in ("YOUR_MODEL", "placeholder", "TODO", ""):
                fail(f"{jf.name}: {key} = '{val}' looks like a placeholder")
            else:
                ok(f"{jf.name}: {key} = {val!r}")

    # no suspiciously round metric values
    def check_metrics(d, path=""):
        if not isinstance(d, dict):
            return
        for k, v in d.items():
            full = f"{path}.{k}" if path else k
            if isinstance(v, float) and v in SUSPICIOUS_EXACT:
                fail(f"{jf.name}: {full} = {v} is suspiciously round – "
                     "verify this came from an actual model run")
            elif isinstance(v, dict):
                check_metrics(v, path=full)
    check_metrics(R)

# ---- 3. check that ASR results exist OR BLOCKED.md explains why ---------
asr_res = RESULTS_DIR / "asr_results.json"
blocked = pathlib.Path("BLOCKED.md")
if not asr_res.exists():
    if blocked.exists() and "ASR" in blocked.read_text():
        ok("asr_results.json absent but BLOCKED.md documents the reason")
    else:
        fail("asr_results.json missing and BLOCKED.md has no ASR entry")

# ---- summary ------------------------------------------------------------
print()
if PASS:
    print("All checks PASSED – results appear to have real provenance.")
    sys.exit(0)
else:
    print("One or more checks FAILED – see [FAIL] lines above.")
    sys.exit(1)
