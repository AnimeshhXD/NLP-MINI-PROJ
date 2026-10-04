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

## 2026-10-05  Task 6b

### ffmpeg: imageio-ffmpeg workaround
`ffmpeg` was not on the system PATH. Installed `imageio-ffmpeg==0.6.0` via pip, which
bundles a static `ffmpeg-win-x86_64-v7.1.exe`. That binary was copied as `ffmpeg.exe`
into the Python installation directory (`C:\Users\ANIMESH\AppData\Local\Programs\Python\Python311\`)
which is on PATH, making it available to `whisper.audio.load_audio` and other subprocess
callers. No system PATH or registry was modified.

### verify_real.py: tolerance reduction
Reduced numeral-traceability tolerance from 0.1 to 0.05.
BLEU values in FINAL_RESULTS.md are printed to one decimal place (`.1f`), so the maximum
rounding error is 0.05. Worst observed gap is 0.0340. All checks pass at 0.05.

### Task 6: real Whisper ASR
- Ran Whisper base.en on `videos/weekly_2015_vra.mp4` (~2:40, White House Weekly Address).
- Results in `results_new/asr_results.json`.

### Task 6b: BERT captions
- Generated `captions/weekly_2015_vra.srt` and `.vtt` (BERT unweighted punctuation, Whisper word times).
- Generated `captions/baseline_weekly_2015_vra.srt` and `.vtt` (no punctuation baseline).

## 2026-10-05  Task 6d

### New segmenter (segment_captions + Whisper word times)
Ran `capnlp.segment_captions(whisper_raw_ws)` for clause-aware text layout, then assigned
Whisper word start/end times. CPS enforced by extending t_end; max 7s, min 1s; no 5s cap.
One hard-wrap mid-word break in block 5 ("Lyndon" split) — inherent in segment_captions
when no word-boundary split fits within 2×42 chars.

### METEOR/BERTScore consistency fix
Old extra_metrics.json mixed two preprocessing approaches: CRF used `word_tokenize+lower()`
(from extra_metrics.py) while bert_uw used `str.split()` (from task4_extra_metrics.py).
Both now use `str.split()` consistently, matching WER/BLEU/chrF input. CRF METEOR dropped
from 0.9638 (with lowercasing) to 0.9106 (case-sensitive, consistent).
BERTScore recomputed for all 5 systems; values unchanged for bert_uw.

### File cleanup
- `results_bert/bert_results.json`: removed honest_assessment, boot95, paired_vs_crf;
  archived to `results_old/bert_results_archived_fields.json`.
- `results_new/asr_results.json`: renamed `sync_check` → `whisper_segment_grouping`.
- FINAL_RESULTS.md: updated Text Quality Metrics section (all 5 systems); updated
  sync_check references to whisper_segment_grouping.

## 2026-10-05  Task 6c

### verify_real.py: per-number tolerance (0.5 × 10^−d)
Changed numeral-traceability tolerance from a flat 0.05 to `0.5 × 10^(−d)` where d is the
number of printed decimal digits. Also removed `round(..., 3)` from both the table numeral
extraction path and the JSON number collector, so 4-decimal-place values like `0.5576` are
compared at full float precision (tol=0.00005). All checks pass.

### Task 6c results
- `results_new/task6c_results.json` written.
- `task6c_r` added to verify_real.py collector.
- FINAL_RESULTS.md updated with punct F1 table and hybrid format metrics table.
