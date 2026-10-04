Closed-caption post-processing pipeline
========================================

QUICK REPRODUCTION (run make_all.sh or follow the steps below)
---------------------------------------------------------------
  bash make_all.sh          # Linux/Mac/Git-Bash (skips blocked steps)
  OR follow steps 0-8 below manually.

STEP 0  –  Environment check
  python env_check.py       # checks GPU/RAM/ffmpeg, installs missing packages
  Output: results_new/env.json

STEP 1  –  Data preparation
  python data/prepare.py    # parses NLTK speeches into talk-level (word,label) data
  Output: data/talks.pkl, data/stats.json, data/SOURCES.md
  (125 talks, 489k words; split by TALK 70/15/15; seed=42)

STEP 2  –  ASR with Whisper  *** BLOCKED – see BLOCKED.md ***
  python asr/run_whisper.py # requires ffmpeg + videos/ directory
  (without ffmpeg: all ASR and caption-timing steps are blocked)

STEP 3  –  Punctuation restoration
  python punct/train_bert.py
  # Trains: no-punct baseline, rule baseline, CRF, BERT (caption + recall modes)
  # Bootstrap 95% CIs over talks, paired bootstrap test
  # Output: results_new/bert_results.json, models/bert_punct_v2/
  # ~60 min on CPU

STEP 4  –  Grammar correction
  python grammar/train_t5.py
  # Trains T5-small seq2seq on CRF-punctuated → reference pairs
  # Output: results_new/t5_results.json, models/t5_grammar_v2/
  # ~45 min on CPU

STEP 5  –  Timed captions  *** BLOCKED until Step 2 produces asr/*.json ***
  python captions/make_captions.py --punct crf
  pytest tests/test_captions.py -v     # 43 format tests

STEP 6  –  Human evaluation  *** BLOCKED until rating sheets are completed ***
  python human_eval/analyse_human_eval.py
  (see BLOCKED.md for instructions on creating rating sheets)

STEP 7  –  Demo app
  streamlit run demo_app.py            # interactive Streamlit UI
  python demo_app.py --test            # headless verification (passes)

STEP 8  –  Verify provenance and generate report
  python verify_real.py                # anti-fake audit (exits 0 if all OK)
  python generate_final_results.py     # writes FINAL_RESULTS.md
  python update_final_results.py       # prints results table from old bert_punct.py

----

LEGACY BASELINE (original Stage 2)
------------------------------------
  python prep.py              # data.pkl (70/15/15 split, CRF labels)
  python eda.py               # EDA
  python run_experiments.py   # E1-E7: CRF experiments, results/results.json
  python bert_punct.py        # old BERT training -> results_bert/bert_results.json
  python t5_grammar.py        # old T5 training  -> results_bert/t5_results.json
  python extra_metrics.py     # METEOR + BERTScore
  python make_final_figures.py

Key results: CRF test macroF1(C,P)=0.425, BERT ep1 val=0.5382 (still training)

----

See BLOCKED.md for tasks requiring ffmpeg or human raters.
See FINAL_RESULTS.md for a summary of all numbers.
