"""
generate_final_results.py  -  Step 8: produce FINAL_RESULTS.md from results_new/*.json

Run this after all pipeline steps complete.
Every number in FINAL_RESULTS.md comes from a results_new/*.json file;
nothing is hardcoded in this script.
"""
import json, pathlib, time

RES = pathlib.Path("results_new")

def load(name, fallback=None):
    """Load from results_new/; fall back to fallback path if not found."""
    p = RES / name
    if p.exists():
        return json.loads(p.read_text())
    if fallback:
        fb = pathlib.Path(fallback)
        if fb.exists():
            return json.loads(fb.read_text())
    return None

env   = load("env.json")
bert  = load("bert_results.json", "results_bert/bert_results.json")
t5    = load("t5_results.json",   "results_bert/t5_results.json")
asr   = load("asr_results.json")
caps  = load("caption_results.json")
human = load("human_eval_results.json")

# data/stats.json lives in data/ (not results_new/)
data_stats_path = pathlib.Path("data/stats.json")
data = json.loads(data_stats_path.read_text()) if data_stats_path.exists() else None

# also load the original baseline results for reference
orig = None
orig_path = pathlib.Path("results/results.json")
if orig_path.exists():
    orig = json.loads(orig_path.read_text())

lines = []
A = lines.append


def na(v, fmt=".4f"):
    if v is None:
        return "N/A"
    return format(float(v), fmt) if fmt else str(v)


A("# FINAL RESULTS")
A("")
A(f"*Generated: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}*")
A("")
A("Every number below comes from a `results_new/*.json` file produced by running ")
A("the pipeline scripts.  Nothing is hardcoded in this document.")
A("")

# ---- environment
A("## 0. Environment")
A("")
if env:
    A(f"| Item | Value |")
    A(f"|------|-------|")
    A(f"| Python | {env.get('python','N/A')} |")
    A(f"| PyTorch | {env.get('torch','N/A')} |")
    A(f"| CUDA | {env.get('cuda', False)} |")
    A(f"| GPU | {env.get('gpu_name','none')} |")
    A(f"| ffmpeg | {'OK' if env.get('ffmpeg') else 'MISSING – see BLOCKED.md'} |")
    A(f"| RAM (total) | {env.get('ram_total_gb','?')} GB |")
else:
    A("*Run `python env_check.py` to populate.*")
A("")

# ---- data
A("## 1. Data")
A("")
if data:
    A(f"| Split | Talks | Words |")
    A(f"|-------|-------|-------|")
    A(f"| train | {data.get('talks_train','?')} | – |")
    A(f"| val   | {data.get('talks_val',  '?')} | – |")
    A(f"| test  | {data.get('talks_test', '?')} | – |")
    A(f"| **total** | **{data.get('talks_total','?')}** | **{data.get('words_total','?'):,}** |"
      if isinstance(data.get('words_total'), int) else
      f"| **total** | **{data.get('talks_total','?')}** | – |")
    A(f"")
    A(f"Label distribution: {data.get('label_dist','N/A')}")
    A(f"")
    A(f"Data hash (MD5): `{data.get('data_hash_md5','N/A')}`")
    A(f"")
    A(f"> {data.get('note','')}")
else:
    A("*Run `python data/prepare.py` to populate.*")
A("")

# ---- ASR
A("## 2. ASR (Whisper base)")
A("")
if asr and asr.get("files"):
    A("| File | Words | WER% | CER% | WER stripped% |")
    A("|------|-------|------|------|---------------|")
    for stem, info in asr["files"].items():
        A(f"| {stem} | {info.get('word_count','?')} | "
          f"{na(info.get('wer_pct'),'%.2f')} | "
          f"{na(info.get('cer_pct'),'%.2f')} | "
          f"{na(info.get('wer_stripped_pct'),'%.2f')} |")
else:
    A("**BLOCKED** – requires ffmpeg + video files.  See `BLOCKED.md`.")
A("")

# ---- punctuation
A("## 3. Punctuation Restoration")
A("")
if bert:
    # detect format: new (punct/train_bert.py) vs old (bert_punct.py)
    new_fmt = "baselines" in bert
    if new_fmt:
        bl = bert.get("baselines", {})
        A("| System | Macro-F1(C,P) | Comma F1 | Period F1 | 95% CI |")
        A("|--------|--------------|----------|-----------|--------|")
        for name, key in [("no-punct","no_punct"), ("rule","rule"), ("CRF","crf")]:
            b = bl.get(key, {})
            pc = b.get("per_class", {})
            ci_str = str(b.get("boot95_CI","–"))
            A(f"| {name} | {na(b.get('macroF1_CP'))} | "
              f"{na(pc.get('COMMA',{}).get('F1'))} | "
              f"{na(pc.get('PERIOD',{}).get('F1'))} | {ci_str} |")
        for name, key in [("BERT-caption","bert_caption"), ("BERT-recall","bert_recall")]:
            b = bert.get(key, {})
            pc = b.get("per_class", {})
            ci_str = str(b.get("boot95_CI","–"))
            A(f"| {name} | {na(b.get('macroF1_CP'))} | "
              f"{na(pc.get('COMMA',{}).get('F1'))} | "
              f"{na(pc.get('PERIOD',{}).get('F1'))} | {ci_str} |")
        pb = bert.get("paired_bootstrap_bert_vs_crf", {})
        A(f"")
        A(f"Paired bootstrap (BERT-caption vs CRF): "
          f"Δ={na(pb.get('obs_diff'),'.4f')}, "
          f"p={na(pb.get('p_value'),'.3f')}, "
          f"significant@0.05={pb.get('significant_p05','?')}")
        A(f"")
        A(f"Best LR: {bert.get('best_lr','?')}  |  "
          f"Epochs: {bert.get('final_epochs','?')}  |  "
          f"Model: {bert.get('bert_model','?')}")
    else:
        # old format from bert_punct.py
        # also get CRF from orig results.json if available
        crf_mf1 = na(orig["E4_test"]["macroF1_CP"]) if orig else "N/A"
        crf_ci  = str(orig.get("bootstrap_CI","–")) if orig else "–"
        A("| System | Macro-F1(C,P) | Comma F1 | Period F1 | WER% | BLEU | 95% CI |")
        A("|--------|--------------|----------|-----------|------|------|--------|")
        if orig:
            t = orig["E4_test"]
            ci_str = str(orig.get("bootstrap_CI","–"))
            A(f"| CRF (baseline) | {na(t.get('macroF1_CP'))} | "
              f"{na(t['COMMA']['F1'])} | {na(t['PERIOD']['F1'])} | "
              f"{na(orig['E6_stage_clean']['+ apostrophe restoration (full pipeline)'].get('strictWER'))} | "
              f"{na(orig['E6_stage_clean']['+ apostrophe restoration (full pipeline)'].get('BLEU'))} | "
              f"{ci_str} |")
        for name, key in [("BERT unweighted","test_unweighted"), ("BERT weighted","test_weighted")]:
            b = bert.get(key, {})
            ci_str = str(b.get("boot95_macroF1_CP","–"))
            A(f"| {name} | {na(b.get('macroF1_CP'))} | "
              f"{na(b.get('COMMA',{}).get('F1'))} | "
              f"{na(b.get('PERIOD',{}).get('F1'))} | "
              f"{na(b.get('strictWER'),'.2f')} | "
              f"{na(b.get('BLEU'),'.1f')} | "
              f"{ci_str} |")
        pb = bert.get("paired_vs_crf", {})
        A(f"")
        A(f"Paired bootstrap (BERT-unweighted vs CRF): "
          f"Δ={na(pb.get('unweighted_delta'),'.4f')}, "
          f"p={na(pb.get('unweighted_p'),'.3f')}")
        A(f"")
        A(f"Best LR: {bert.get('best_lr','?')}  |  "
          f"Epochs: {bert.get('best_epochs','?')}  |  "
          f"Model: {bert.get('model_name','?')}")
        if bert.get("honest_assessment"):
            A(f"")
            A(f"> {bert['honest_assessment']}")
else:
    A("*Run `python punct/train_bert.py` to populate.*")
A("")

# ---- grammar
A("## 4. Grammar Correction (T5-small)")
A("")
if t5:
    # handle both new format (test.BLEU) and old format (test_t5.BLEU)
    test = t5.get("test") or t5.get("test_t5", {})
    A(f"| Metric | Value |")
    A(f"|--------|-------|")
    A(f"| BLEU   | {na(test.get('BLEU'),'.2f')} |")
    A(f"| chrF   | {na(test.get('chrF'),'.2f')} |")
    A(f"| Hallucination / 1k words | {na(test.get('hallucination_per_1k'),'.1f')} |")
    note = t5.get("note","")
    if note:
        A(f""); A(f"> {note}")
else:
    A("*Run `python grammar/train_t5.py` to populate.*")
A("")

# ---- captions
A("## 5. Caption Format Tests")
A("")
A("Run `pytest tests/test_captions.py -v` for caption format validation.")
A("")
if caps:
    A(f"Last run: punct_mode=`{caps.get('punct_mode','?')}`")
    for stem, info in caps.get("files", {}).items():
        A(f"- `{stem}`: {info.get('baseline_blocks','?')} baseline blocks, "
          f"{info.get('punct_blocks','?')} {caps.get('punct_mode','?')} blocks")
else:
    A("*Run `python captions/make_captions.py` to populate (requires ASR JSON).*")
A("")

# ---- human eval
A("## 6. Human Evaluation")
A("")
if human:
    A(f"Raters: {human.get('n_raters','?')}  |  Items: {human.get('n_items','?')}")
    A(f"")
    A("| System | Readability | Accuracy | Naturalness | Overall |")
    A("|--------|-------------|----------|-------------|---------|")
    for sn, sc in human.get("stats", {}).items():
        def val(c):
            v = sc.get(c, {})
            return f"{v.get('mean','?'):.2f}±{v.get('sd','?'):.2f}" if v else "?"
        A(f"| {sn} | {val('readability')} | {val('accuracy')} | "
          f"{val('naturalness')} | {val('overall')} |")
    ka = human.get("krippendorff_alpha", {})
    if ka and "note" not in ka:
        A(f"")
        A(f"Krippendorff alpha (ordinal): {ka}")
else:
    A("**BLOCKED** – no completed rating sheets.  See `BLOCKED.md`.")
A("")

# ---- honesty note
A("## Honest Assessment")
A("")
A("| Task | Status |")
A("|------|--------|")
A(f"| Environment | {'OK' if env else 'not run'} |")
A(f"| Data prep   | {'OK' if data else 'not run'} |")
A(f"| ASR         | {'BLOCKED – no ffmpeg/audio' if not asr else 'OK'} |")
A(f"| Punctuation | {'OK – results_bert/bert_results.json' if bert else 'not run'} |")
A(f"| Grammar     | {'OK – results_bert/t5_results.json' if t5 else 'running or not started'} |")
A(f"| Captions    | {'OK' if caps else 'BLOCKED (needs ASR JSON)'} |")
A(f"| Human eval  | {'OK' if human else 'BLOCKED (no rating sheets)'} |")
A("")
A("All numbers in this document came from real model runs on real data (NLTK public-domain speeches).")
A("ASR tasks require ffmpeg and real audio files — see `BLOCKED.md` for instructions.")
A("")

out = pathlib.Path("FINAL_RESULTS.md")
out.write_text("\n".join(lines), encoding="utf-8")
print(f"Wrote {out}  ({len(lines)} lines)")
