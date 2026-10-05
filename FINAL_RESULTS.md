# FINAL RESULTS

*Generated: 2026-10-04T20:05:59Z*

> Results are for NLTK speeches with simulated ASR input. Causes of differences
> from published results were not tested. DistilBERT may have seen these public
> speeches during pretraining; this was not measured.

## 0. Environment

| Item | Value |
|------|-------|
| Python | 3.11.7 (tags/v3.11.7:fa7a6f2, Dec  4 2023, 19:24:49) [MSC v.1937 64 bit (AMD64)] |
| PyTorch | 2.14.1+cpu |
| CUDA | False |
| GPU | None |
| ffmpeg | OK (imageio-ffmpeg 0.6.0 bundle) |
| RAM (total) | 16.8 GB |

## 1. Data

*Source: data.pkl (old pipeline) — used for all model training and evaluation.*

| Split | Talks | Words |
|-------|-------|-------|
| train | 87 | – |
| val   | 18 | – |
| test  | 20 | 85,917 |
| **total** | **125** | **488,798** |

Test label distribution: {'O': 77027, 'COMMA': 5180, 'PERIOD': 3667, 'QUESTION': 43}

*Total counts from `data.pkl` (old pipeline, used for all models).*
*data/talks.pkl uses different tokenisation and a different test split —*
*see `data_reconciliation.md` for the full explanation.*

## 2. ASR (Whisper base.en)

**1 video, 424 words** — `videos/weekly_2015_vra.mp4` (White House Weekly Address, ~2:40).
Reference: `refs/weekly_2015_vra.txt` (human transcript).
Number normalisation: Whisper `EnglishTextNormalizer`. No confidence interval.

| Condition | WER % | CER % |
|-----------|------:|------:|
| Whisper punct kept (as tokens) | 5.99 | 2.36 |
| Whisper punct stripped | 1.84 | 1.07 |

*"Punct kept": punctuation marks detached as separate word tokens; per-token number normalisation.*
*"Punct stripped": `EnglishTextNormalizer` applied to both (normalises numbers, lowercases, strips punct).*
*Source: `results_new/asr_results.json` → `wer_cer.punct_kept.*`, `wer_cer.no_punct.*`*

### Sync check — Whisper segment grouping (nearest-by-time)

TSV has 1-second resolution. Caption blocks = Whisper segments.
22 TSV segments; all 22 matched.

| Statistic | Value |
|-----------|------:|
| Mean \|offset\| | 0.98 s |
| Max \|offset\|  | 4.56 s |

*Source: `results_new/asr_results.json` → `whisper_segment_grouping.mean_abs_offset_s`, `whisper_segment_grouping.max_abs_offset_s`*

### Three-way WER on caption output (BERT vs Whisper vs no punct)

1 video, 424 words. WER_strict: case-sensitive, punctuation attached (jiwer default).
WER_norm: EnglishTextNormalizer applied to both (numbers + lowercase + strip punct).

| System | WER_strict % | WER_norm % |
|--------|-------------:|-----------:|
| Whisper own punct | 9.43 | 1.84 |
| BERT punct (unweighted) | 19.10 | 3.22 |
| No punct | 28.54 | 7.36 |

*Source: `results_new/task6b_results.json` → `task3_wer`*

### Sync check — sequence alignment (Task 6b)

22/22 TSV segments matched by sequence-aligning Whisper words to reference.

| Statistic | Value |
|-----------|------:|
| Mean offset (signed) | 0.43 s |
| Median offset | 0.59 s |
| Max \|offset\| | 1.88 s |

*Source: `results_new/task6b_results.json` → `task4_sync`*

### Punctuation-only F1 on this video (Task 6c)

Aligned Whisper words to reference (SequenceMatcher). Whisper's own punctuation from raw word
text; BERT labels run on the same Whisper word sequence. 431 evaluation points.
Counts — reference: 28 commas, 24 periods; Whisper predicts: 28 commas, 26 periods; BERT predicts: 24 commas, 23 periods.

| System | Mark | P | R | F1 | TP/FP/FN |
|--------|------|--:|--:|---:|----------|
| Whisper | COMMA | 0.7857 | 0.7857 | 0.7857 | 22/6/6 |
| Whisper | PERIOD | 0.8077 | 0.8750 | 0.8400 | 21/5/3 |
| BERT | COMMA | 0.5000 | 0.4286 | 0.4615 | 12/12/16 |
| BERT | PERIOD | 0.6957 | 0.6667 | 0.6809 | 16/7/8 |

*Source: `results_new/task6c_results.json` → `task1_punct_f1`*

### Hybrid system format metrics (Task 6c)

Hybrid = Whisper's own punctuation+casing fed to the segmenter with real Whisper word times.

| Metric | Hybrid (Whisper→segmenter) | Whisper raw segments | Fixed-width baseline |
|--------|---------------------------:|---------------------:|---------------------:|
| n blocks | 31 | 39 | 30 |
| within 2×42 chars | 100.0% | 59.0% | 100.0% |
| within 17 cps | 87.1% | 25.6% | 86.7% |
| ends on function word | 54.8% | 23.1% | 36.7% |
| mid-word breaks | 0 | 0 | 0 |
| mean duration | 4.93 s | 3.33 s | 5.07 s |
| median duration | 4.90 s | 4.16 s | 5.18 s |

*Source: `results_new/task6c_results.json` → `task2_format`*

## 3. Punctuation Restoration

| System | Macro-F1(C,P) | Comma F1 | Period F1 | Sent-F1 | WER% | BLEU | 95% CI macro-F1 |
|--------|--------------|----------|-----------|---------|------|------|----------------|
| Majority (no marks) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 15.60 | 70.0 | [0.000, 0.000] |
| Rule baseline | 0.1173 | 0.1971 | 0.0375 | 0.0379 | 21.43 | 66.4 | [0.102, 0.133] |
| CRF | 0.4248 | 0.3312 | 0.5184 | 0.5208 | 13.30 | 78.0 | [0.402, 0.443] |
| BERT unweighted | 0.6390 | 0.5576 | 0.7204 | 0.7224 | 9.62 | 84.4 | [0.614, 0.657] |
| BERT weighted | 0.5241 | 0.4604 | 0.5877 | 0.5744 | 18.20 | 70.0 | [0.498, 0.545] |

Paired bootstrap (BERT-uw vs CRF): Δ=0.2143, p=<0.0005
Paired bootstrap (BERT-wt vs CRF): Δ=0.0993, p=<0.0005
Paired bootstrap (BERT-uw vs BERT-wt): Δ=0.1150, p=<0.0005

*Bootstrap CIs: 2000 resamples over 20 test talks (seed 0), pooled confusion counts.*
*Source: `results_new/metrics_all.json` + `results_new/bootstrap.json`*

### Text Quality Metrics (METEOR, BERTScore)

| System | METEOR | BERTScore F1 |
|--------|--------|-------------|
| Majority (no marks) | 0.8759 | 0.9273 |
| Rule baseline | 0.7528 | 0.9461 |
| CRF | 0.9106 | 0.9681 |
| BERT unweighted | 0.9385 | 0.9818 |
| BERT weighted | 0.8453 | 0.9471 |

> distilbert-base-uncased lowercases all input; BERTScore therefore
> cannot distinguish capitalisation. Scores are NOT baseline-rescaled.
> BERTScore model: `distilbert-base-uncased`
> METEOR tokenization: str.split() without lowercasing (consistent with WER/BLEU/chrF).
*Source: `results_new/extra_metrics.json`*
## 4. Grammar Correction (T5-small)

*Run `python grammar/train_t5.py` to populate.*

## 5. Caption Format Tests

Caption files: `captions/weekly_2015_vra.srt`, `captions/weekly_2015_vra.vtt` (BERT punct),
`captions/baseline_weekly_2015_vra.srt`, `captions/baseline_weekly_2015_vra.vtt` (no punct).
Segmentation: ≤2 lines × 42 chars, ≤17 chars/s, min 1 s, max 7 s, no word splits.
Whisper base.en word timestamps used for block boundaries.

| Result | Value |
|--------|------:|
| Tests passed | 67 |
| Tests failed | 0 |
| Pass rate | 100.0 % |

*Source: `results_new/task6b_results.json` → `task2`*

## 6. Human Evaluation

**BLOCKED** – no completed rating sheets.  See `BLOCKED.md`.

## Provenance

Every figure in every table above comes from a `results_new/*.json` file written by
a script in this session.  Nothing is hardcoded in this document.

| Table | JSON file | Key |
|-------|-----------|-----|
| Section 1 (Data) | `results_new/preds_test_majority.json` | `true_label` counts |
| Section 2 (WER/CER) | `results_new/asr_results.json` | `wer_cer.*` |
| Section 2 (Sync, segment grouping) | `results_new/asr_results.json` | `whisper_segment_grouping.*` |
| Section 2 (Three-way WER) | `results_new/task6b_results.json` | `task3_wer` |
| Section 2 (Sync, seq-align) | `results_new/task6b_results.json` | `task4_sync` |
| Section 5 (Caption tests) | `results_new/task6b_results.json` | `task2` |
| Section 2 (Punct F1, this video) | `results_new/task6c_results.json` | `task1_punct_f1` |
| Section 2 (Hybrid format metrics) | `results_new/task6c_results.json` | `task2_format` |
| Section 3 (Metrics) | `results_new/metrics_all.json` | `systems.<name>` |
| Section 3 (95% CI) | `results_new/bootstrap.json` | `ci_results.<name>.macroF1_CP` |
| Section 3 (Paired) | `results_new/bootstrap.json` | `paired_diffs.*` |
| Section 3 (METEOR/BERTScore) | `results_new/extra_metrics.json` | `METEOR`, `BERTScore_F1` |
| Section 0 (Environment) | `results_new/env.json` | all fields |

## What Was NOT Done

- ffmpeg installed via imageio-ffmpeg; Task 6 (real audio) now complete.
- T5 grammar correction test evaluation: see Section 4.
- Human evaluation: no completed rating sheets.
- Epoch grid search beyond 1–2 epochs: resource-limited (CPU only).
- Tuned decoding modes for BERT (beam search, etc.): default greedy only.
- BERTScore baseline rescaling: not applied.
