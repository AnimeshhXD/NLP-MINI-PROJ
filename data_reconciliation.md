# Data Reconciliation

## Summary

FINAL_RESULTS.md (Section 1) shows totals from **data/talks.pkl** (the new pipeline),
but all model results (BERT, CRF) were produced from **data.pkl** (the old pipeline).
The two files differ in tokenisation method, preprocessing, and test-split assignment,
so their label counts and talk IDs do not match.

## data.pkl (old pipeline — used by BERT and CRF)

Created by `prep.py` using `capnlp.clean_raw()` + `capnlp.to_tokens()`.
- Removes UPPER-CASE title lines
- Converts dashes to commas
- Removes abbreviation periods (U.S. → US)
- IDs have no `.txt` extension (e.g. `su_2001-GWBush`)
- Stores `(word, label, case_label)` triples

| Split | Talks | Words | O | COMMA | PERIOD | QUESTION |
|-------|-------|-------|---|-------|--------|---------|
| train | 87 | 336,238 | 300,910 | 19,685 | 15,452 | 191 |
| val | 18 | 66,643 | 59,452 | 3,880 | 3,292 | 19 |
| test | 20 | 85,917 | 77,027 | 5,180 | 3,667 | 43 |
| **total** | **125** | **488,798** | 437,389 | 28,745 | 22,411 | 253 |

Test talks: `in_1801-Jefferson, in_1805-Jefferson, in_1833-Jackson, in_1841-Harrison, in_1845-Polk, in_1857-Buchanan, in_1897-McKinley, in_1901-McKinley, in_1913-Wilson, in_1929-Hoover, in_2005-Bush, su_1955-Eisenhower, su_1961-Kennedy, su_1965-Johnson-2, su_1970-Nixon, su_1978-Carter, su_1994-Clinton, su_1999-Clinton, su_2003-GWBush, su_2004-GWBush`

## data/talks.pkl (new pipeline — used for FINAL_RESULTS.md Section 1)

Created by `data/prepare.py` using `re.findall(r"[A-Za-z']+|[.,?!;]", text)`.
- No title-line removal, no dash conversion
- IDs include `.txt` extension (e.g. `su_2001-GWBush.txt`)
- Stores `(word, label)` pairs (no case_label)
- Produces a different test set (different seed-42 shuffle due to different ID set)
- Total: 125 talks, 489,643 words
- Label distribution: {'O': 439783, 'PERIOD': 24671, 'COMMA': 24934, 'QUESTION': 255}

## Why the counts differ

1. **Tokenisation**: `to_tokens()` uses a more conservative regex (`TOKEN_RE`) that
   excludes standalone punctuation differently; `prepare.py` keeps more tokens.
2. **Preprocessing**: `clean_raw()` removes all-caps title lines and converts
   dashes to commas, changing the label distribution.
3. **Different test sets**: Both pipelines shuffle with `seed=42` but use
   different ID sets (with vs without `.txt` suffix), producing different splits.

## Impact on overnight tasks

All overnight prediction and evaluation tasks use **data.pkl** as ground truth,
because BERT was trained on data.pkl and the spec says do not retrain BERT.
FINAL_RESULTS.md Section 1 will be updated to reflect data.pkl counts.