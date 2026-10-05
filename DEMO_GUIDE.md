# Demo Guide — Closed-Caption NLP Pipeline

*2026-10-05*

All tasks are complete. This guide explains the four ways to demonstrate
the pipeline, ordered from most interactive to most auditable.

---

## 1. Streamlit demo app (interactive, best for an audience)

```bash
cd C:\Users\ANIMESH\Desktop\pkg
streamlit run demo_app.py
```

Opens `http://localhost:8501` in your browser.

What you can do:
- Paste any transcript text, or upload a Whisper JSON with word timestamps
- Choose punctuation model: CRF or BERT (unweighted)
- Click **Generate** → download `.srt` / `.vtt` caption files
- If you also upload the source `.mp4`, ffmpeg burns the captions in

Headless smoke-test (no browser needed):

```bash
python demo_app.py --test
# prints: All headless tests passed.
```

---

## 2. Play the real video with captions (most visual)

The pipeline has already been run on `videos/weekly_2015_vra.mp4`
(White House Weekly Address, ~2:40, 424 words).

Caption files in `captions/`:

| File | System |
|------|--------|
| `weekly_2015_vra.srt` / `.vtt` | BERT unweighted punctuation + Whisper timing |
| `baseline_weekly_2015_vra.srt` / `.vtt` | No punctuation baseline |

**VLC:** `Media → Open File` (pick the MP4), then `Subtitle → Add Subtitle File`.

**Chrome:** Drag the `.mp4` into a tab; right-click the video → `Show controls`.
Chrome doesn't load external VTT from the file picker — use VLC instead.

**mpv (command line):**

```bash
mpv videos/weekly_2015_vra.mp4 --sub-file=captions/weekly_2015_vra.vtt
```

---

## 3. Show the results report (most complete)

Open [`FINAL_RESULTS.md`](FINAL_RESULTS.md) in VS Code with the Markdown preview
(`Ctrl+Shift+V`) or push it to GitHub and view it there.

Key numbers:

| System | Macro-F1(C,P) | WER % | BLEU |
|--------|-------------:|------:|-----:|
| Majority (no marks) | 0.0000 | 15.60 | 70.0 |
| Rule baseline | 0.1173 | 21.43 | 66.4 |
| CRF | 0.4248 | 13.30 | 78.0 |
| **BERT unweighted** | **0.6390** | **9.62** | **84.4** |
| BERT weighted | 0.5241 | 18.20 | 70.0 |

Every figure in the report traces back to a `results_new/*.json` file written
by a script run in this session — nothing is hardcoded.

---

## 4. Live audits (prove reproducibility)

**Number provenance check** — every figure in every report comes from a real
model run, not a typed value:

```bash
python verify_real.py
# Expected last line: RESULT: PASS  (0 errors)
```

**Caption format test suite** — 67 tests, all format constraints:

```bash
python -m pytest tests/test_captions.py -v
# Expected: 67 passed in < 5 s
```

**Full reproduction from scratch** (runs all five models + metrics):

```bash
python task1_predictions.py      # generate per-token prediction files
python eval/compute_all.py       # F1 / WER / BLEU / chrF
python task3_bootstrap.py        # 2000-resample bootstrap CIs
python task4_extra_metrics.py    # METEOR + BERTScore + regenerate FINAL_RESULTS.md
python verify_real.py            # final audit
```

---

## Data note

All model training and evaluation uses `data.pkl` (125 NLTK speeches, seed=42 split).
ASR input is **simulated** by stripping punctuation and lowercasing (`asr_style()`);
no speech recognition runs on the text corpus.  Real ASR (Whisper base.en) is used
only on `videos/weekly_2015_vra.mp4` for the caption demo.

See [`data_reconciliation.md`](data_reconciliation.md) for counts and split details.
