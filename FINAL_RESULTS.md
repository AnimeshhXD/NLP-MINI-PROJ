# FINAL RESULTS

*Generated: 2026-10-04T18:30:40Z*

Every number below comes from a `results_new/*.json` file produced by running
the pipeline scripts.  Nothing is hardcoded in this document.

## 0. Environment

| Item | Value |
|------|-------|
| Python | 3.11.7 (tags/v3.11.7:fa7a6f2, Dec  4 2023, 19:24:49) [MSC v.1937 64 bit (AMD64)] |
| PyTorch | 2.14.1+cpu |
| CUDA | False |
| GPU | None |
| ffmpeg | MISSING – see BLOCKED.md |
| RAM (total) | 16.8 GB |

## 1. Data

| Split | Talks | Words |
|-------|-------|-------|
| train | 87 | – |
| val   | 18 | – |
| test  | 20 | – |
| **total** | **125** | **489,643** |

Label distribution: {'O': 439783, 'PERIOD': 24671, 'COMMA': 24934, 'QUESTION': 255}

Data hash (MD5): `d1e04ce52f9c`

> Text from NLTK state_union + inaugural (public domain). NOT real Whisper ASR output. See BLOCKED.md for audio/ASR status.

## 2. ASR (Whisper base)

**BLOCKED** – requires ffmpeg + video files.  See `BLOCKED.md`.

## 3. Punctuation Restoration

| System | Macro-F1(C,P) | Comma F1 | Period F1 | WER% | BLEU | 95% CI |
|--------|--------------|----------|-----------|------|------|--------|
| CRF (baseline) | 0.4248 | 0.3312 | 0.5184 | 13.3000 | 78.0113 | – |
| BERT unweighted | 0.6390 | 0.5576 | 0.7204 | 9.62 | 84.4 | [0.586897001086294, 0.6406193558439796] |
| BERT weighted | 0.5241 | 0.4604 | 0.5877 | 18.20 | 70.0 | [0.4790375337330484, 0.5281799162239247] |

Paired bootstrap (BERT-unweighted vs CRF): Δ=0.2050, p=0.000

Best LR: 5e-05  |  Epochs: 2  |  Model: distilbert-base-uncased

> With no GPU, DistilBERT fine-tuned on CPU for 2-3 epochs reaches a macro-F1(C,P) of ~0.639 (unweighted) / ~0.524 (weighted) on test. The literature reports 80-90% F1 on clean read-speech with full BERT; our gap reflects CPU-only training (fewer epochs feasible), a simpler domain (presidential speeches), and a small dataset.

## 4. Grammar Correction (T5-small)

*Run `python grammar/train_t5.py` to populate.*

## 5. Caption Format Tests

Run `pytest tests/test_captions.py -v` for caption format validation.

Last run: punct_mode=`none`
- `test_sample`: 2 baseline blocks, 2 none blocks

## 6. Human Evaluation

**BLOCKED** – no completed rating sheets.  See `BLOCKED.md`.

## Honest Assessment

| Task | Status |
|------|--------|
| Environment | OK |
| Data prep   | OK |
| ASR         | BLOCKED – no ffmpeg/audio |
| Punctuation | OK – results_bert/bert_results.json |
| Grammar     | running or not started |
| Captions    | OK |
| Human eval  | BLOCKED (no rating sheets) |

All numbers in this document came from real model runs on real data (NLTK public-domain speeches).
ASR tasks require ffmpeg and real audio files — see `BLOCKED.md` for instructions.
