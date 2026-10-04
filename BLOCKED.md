# BLOCKED Items

This file lists tasks that cannot run automatically and what you must do to unblock them.
Items are ordered by impact.

---

## Step 2 – ASR: ffmpeg not on PATH

**Exact error:** `shutil.which("ffmpeg")` returns `None`.

**Root cause:** ffmpeg is not installed or not on the system PATH.

**Fix:**
1. Download ffmpeg from https://www.gyan.dev/ffmpeg/builds/
   (Windows: take the `ffmpeg-release-full.7z` zip)
2. Extract and copy the `bin/` folder contents to `C:\ffmpeg\bin\` (or wherever you like)
3. Add that folder to your system PATH:
   - Settings → System → About → Advanced System Settings → Environment Variables
   - Edit the `Path` variable → Add `C:\ffmpeg\bin`
4. Open a **new** terminal and verify: `ffmpeg -version`
5. Re-run `python asr/run_whisper.py`

**Consequence:** All audio extraction, Whisper ASR, and captioning from real video
are blocked until ffmpeg is available.

---

## Task 6 – Real audio: sample media is not bundled

Aggregate ASR and caption-evaluation metrics are included in `results_new/`.
The local MP4/WAV, reference transcript, full Whisper transcript, and text captions
are not included because their redistribution terms were not documented.

To reproduce Task 6, obtain media and a reference transcript that you may
redistribute, place them in `videos/` and `refs/`, install ffmpeg, then run:

```bash
python asr/run_whisper.py
python captions/make_captions.py --punct bert
```

---

## Step 2 – ASR: reference transcripts not yet written

**Exact error:** `refs/<video>.txt` contains the template comment.

**Root cause:** After Whisper runs, it creates empty `refs/<stem>.txt` files.
WER/CER cannot be computed without a hand-corrected reference.

**Fix:**
1. Open each `refs/<stem>.txt` file.
2. Replace the comment with a hand-corrected transcript of a 2–3 minute segment.
   Write it as one continuous paragraph; no timestamps.
3. Re-run `python asr/run_whisper.py`

---

## Step 5 – Captions: no Whisper JSON available

**Exact error:** `asr/*.json` not found.

**Root cause:** Captions from real timestamps require Whisper output.
`captions/make_captions.py` needs the word-level timing from `asr/run_whisper.py`.

**Fix:** Unblock Steps 2 (ffmpeg + videos), then run:
```
python asr/run_whisper.py
python captions/make_captions.py --punct crf
```

**Workaround for testing format compliance:**
You can generate synthetic caption files with fake timestamps to test the format:
```
python captions/make_captions.py --input asr/test_fake.json --punct none
```
Create `asr/test_fake.json` with:
```json
{"segments":[{"words":[
  {"word":"hello","start":0.0,"end":0.4},
  {"word":"world","start":0.5,"end":1.0}
]}]}
```
Then run `pytest tests/test_captions.py -v`.

---

## Step 6 – Human Evaluation: no rating sheets

**Exact error:** `human_eval/*.csv` is empty.

**Root cause:** Human evaluation requires human raters.

**Fix:**
1. Prepare 20–30 caption blocks from ≥ 3 systems (raw ASR, CRF, BERT, T5).
2. Create a CSV template:
   ```
   item_id,system,readability,accuracy,naturalness,overall
   ```
   (Likert 1–5 per criterion; each item_id appears once per system)
3. Have ≥ 2 raters fill it in **independently** with the system column hidden.
4. Save as `human_eval/rater_<name>.csv` for each rater.
5. Run `python human_eval/analyse_human_eval.py`

---

## Notes

- All punctuation model training (Steps 1, 3) uses NLTK public-domain speech text.
  These steps are **not blocked** and run without ffmpeg or audio.
- BERT training: `python punct/train_bert.py`  (~60 min on CPU)
- T5 training:   `python grammar/train_t5.py`  (~45 min on CPU)
- Caption format tests run without real video: copy a small Whisper JSON into asr/
  and run `pytest tests/test_captions.py -v`
