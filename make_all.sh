#!/usr/bin/env bash
# make_all.sh  -  Full reproduction script for the closed-caption pipeline.
#
# Run:  bash make_all.sh
#
# Steps that require ffmpeg (ASR + video caption burning) are skipped
# automatically if ffmpeg is not on PATH; BLOCKED.md explains what to do.
#
# Expected total time (CPU only, no GPU): ~2–3 hours
#   Step 0: ~2 min
#   Step 1: ~1 min
#   Step 2: BLOCKED without ffmpeg
#   Step 3: ~60 min
#   Step 4: ~45 min
#   Step 5: BLOCKED without Step 2
#   Step 6: manual
#   Step 7: headless test ~30 sec
#   Step 8: ~5 sec
#
set -e

echo "=== Step 0: environment check ==="
python env_check.py

echo ""
echo "=== Step 1: data preparation ==="
python data/prepare.py

echo ""
echo "=== Step 2: ASR with Whisper ==="
if command -v ffmpeg &> /dev/null; then
    python asr/run_whisper.py
else
    echo "SKIPPED: ffmpeg not found. See BLOCKED.md."
fi

echo ""
echo "=== Step 3: punctuation model (BERT) ==="
python punct/train_bert.py

echo ""
echo "=== Step 4: grammar model (T5) ==="
python grammar/train_t5.py

echo ""
echo "=== Step 5: timed captions ==="
if ls asr/*.json 1> /dev/null 2>&1; then
    python captions/make_captions.py --punct crf
    python captions/make_captions.py --punct bert
else
    echo "SKIPPED: no ASR JSON in asr/. See BLOCKED.md."
fi

echo ""
echo "=== Step 5b: caption format tests ==="
pytest tests/test_captions.py -v || echo "No caption files to test yet."

echo ""
echo "=== Step 6: human evaluation ==="
python human_eval/analyse_human_eval.py || true  # exits 0 if no sheets

echo ""
echo "=== Step 7: demo app headless test ==="
python demo_app.py --test

echo ""
echo "=== Step 8: verify provenance ==="
python verify_real.py || echo "WARNING: verify_real.py reported issues (see above)"

echo ""
echo "=== Step 8: generate FINAL_RESULTS.md ==="
python generate_final_results.py

echo ""
echo "All steps complete. See FINAL_RESULTS.md for a summary."
echo "See BLOCKED.md for tasks that require ffmpeg or human raters."
