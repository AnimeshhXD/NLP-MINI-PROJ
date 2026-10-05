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

## 2026-10-05  Sung-audio stress test (hi_song)

### New files
| File | Purpose |
|------|---------|
| `hi_song_pipeline.py` | End-to-end: Whisper → WER → captions → verify |
| `asr/hi_song.transcribe.json` | Whisper small, task=transcribe, lang=hi, condition_on_previous_text=False |
| `asr/hi_song.translate.json`  | Whisper small, task=translate, same flags |
| `captions/hi_song_translate.srt` / `.vtt` | English captions from translate output |
| `results_new/song_results.json` | All metrics; no lyric text; quality note |

### Privacy
`refs/hi_song*` and `videos/hi_song*` already in `.gitignore`.
Lyric text never written to JSON or printed in full; only error word pairs stored.

### Results (all from results_new/song_results.json)
- Whisper transcribe vs lyrics: WER=75.36%, CER=27.15%  (expected for sung audio)
- YouTube auto-captions vs lyrics: WER=20.29%, CER=7.90%
- English captions: 12 blocks, 100% within 2×42, 91.7% within 17 cps, word-split PASS
- verify_real.py: PASS (0 errors)

### Normalisation (Devanagari)
NFC → remove bracketed tags [संगीत] → remove nukta (U+093C) →
chandrabindu (U+0901) → anusvara (U+0902) → keep only U+0900–U+097F + space.

---

## 2026-10-05  Hindi caption pipeline (Tasks H1–H9)

### New files
| File | Purpose |
|------|---------|
| `asr/run_whisper_hi.py` | Whisper Method A (translate) + B (transcribe) on hi_*.mp4 |
| `asr/nllb_translate.py` | NLLB 600M translation of Whisper transcribe output |
| `captions/make_captions_hi.py` | Hindi→English caption SRT/VTT with proportional timing |
| `tests/test_hi_captions.py` | pytest suite: line length, word order, timestamps, CPS |
| `eval/compute_hi_eval.py` | WER/CER (Hindi ASR), BLEU/chrF (translation), format metrics, sync |
| `human_eval/gen_hi_adequacy.py` | Generates 20-segment adequacy CSV (randomised A/B order) |
| `human_eval/analyse_hi_eval.py` | Analyses completed rating sheets → mean±SD per method |
| `gen_hindi_results.py` | Aggregates all results → hindi_results.json + HINDI_RESULTS.md |
| `results_new/hindi_env.json` | Environment check (written this session) |
| `DEMO_GUIDE.md` | How to demonstrate the pipeline |

### Environment
- Python 3.11.7, PyTorch 2.14.1+cpu, no CUDA
- Whisper model: `small` (no GPU; large-v3/medium too slow on CPU)
- NLLB: `facebook/nllb-200-distilled-600M` (~1.2 GB, downloads on first run)
- ffmpeg: OK (imageio-ffmpeg 0.6.0 bundle + system PATH)

### Status
All scripts written and smoke-tested (imports pass, capnlp.segment_captions integrates).
BLOCKED on `videos/hi_<name>.mp4` — see BLOCKED.md for run instructions.

### Timing method (documented in code + results)
Within each Whisper segment, caption block times are distributed proportionally
to character count: t_block_i ∝ chars(block_i) / total_chars_in_segment.

---

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
