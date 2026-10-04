# CHANGES.md

Changes to existing files (rule: only touch old files to fix real bugs).

---

## No changes to baseline files

None of the following files have been modified:
- `capnlp.py`
- `run_experiments.py`
- `prep.py`
- `eda.py`
- `results/results.json`

All new work is in separate files: `bert_punct.py`, `t5_grammar.py`,
`asr_whisper.py`, `caption_system_v2.py`, `test_captions.py`,
`analyse_human_eval.py`, `extra_metrics.py`, `demo_app.py`, `data_loader.py`,
`env_check.py`.

---

## New files added

| File | Purpose |
|------|---------|
| `env_check.py` | Task 1 – environment check |
| `bert_punct.py` | Task 2 – DistilBERT punctuation classifier |
| `t5_grammar.py` | Task 3 – T5-small grammar correction |
| `asr_whisper.py` | Task 4 – Whisper ASR on real video |
| `caption_system_v2.py` | Task 5 – timed caption pipeline |
| `data_loader.py` | Shared helper: load data.pkl + TrueCaser/apos |
| `test_captions.py` | Task 5 – pytest caption format tests |
| `analyse_human_eval.py` | Task 6 – human evaluation analysis |
| `extra_metrics.py` | Task 7 – METEOR and BERTScore |
| `demo_app.py` | Task 8 – Gradio demo app |
| `BLOCKED.md` | Blocked tasks with student instructions |
| `CHANGES.md` | This file |
| `FINAL_RESULTS.md` | Final reporting table |
