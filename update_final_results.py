"""
update_final_results.py - Fill in BERT (and T5) rows in FINAL_RESULTS.md
once bert_results.json and/or t5_results.json are available.

Run this after bert_punct.py and/or t5_grammar.py complete.
"""
import json, pathlib

def load_json(p):
    p = pathlib.Path(p)
    return json.load(p.open()) if p.exists() else None

R   = load_json("results/results.json")
B   = load_json("results_bert/bert_results.json")
T5  = load_json("results_bert/t5_results.json")
EM  = load_json("results_bert/extra_metrics.json")

print("=== Current Results Summary ===\n")

print("CRF baseline (test):")
if R:
    t = R["E4_test"]
    print("  Comma F1=%.3f  Period F1=%.3f  Macro-F1(C,P)=%.3f  Sent F1=%.3f" % (
        t["COMMA"]["F1"], t["PERIOD"]["F1"], t["macroF1_CP"], t["sentF1"]))
    pipe = R["E6_stage_clean"]["+ apostrophe restoration (full pipeline)"]
    print("  StrictWER=%.2f%%  BLEU=%.1f" % (pipe["strictWER"], pipe["BLEU"]))

print("\nBERT unweighted (test):")
if B and B.get("test_unweighted", {}).get("macroF1_CP"):
    tu = B["test_unweighted"]
    print("  Comma F1=%.3f  Period F1=%.3f  Macro-F1(C,P)=%.3f  Sent F1=%.3f" % (
        tu["COMMA"]["F1"], tu["PERIOD"]["F1"], tu["macroF1_CP"], tu["sentF1"]))
    print("  StrictWER=%.2f%%  BLEU=%.1f" % (tu.get("strictWER","?"), tu.get("BLEU","?")))
    print("  Boot 95% CI:", tu.get("boot95_macroF1_CP","pending"))
    ha = B.get("honest_assessment", "")
    if ha: print("\n  Assessment:", ha[:200])
else:
    print("  (not yet available – run bert_punct.py)")

print("\nBERT weighted (test):")
if B and B.get("test_weighted", {}).get("macroF1_CP"):
    tw = B["test_weighted"]
    print("  Macro-F1(C,P)=%.3f  StrictWER=%.2f%%  BLEU=%.1f" % (
        tw["macroF1_CP"], tw.get("strictWER","?"), tw.get("BLEU","?")))
else:
    print("  (not yet available)")

print("\nT5-small (test):")
if T5 and T5.get("test_t5", {}).get("BLEU"):
    tt = T5["test_t5"]
    print("  StrictWER=%.2f%%  BLEU=%.1f  chrF=%.1f  Hall/1k=%.1f" % (
        tt["strictWER"], tt["BLEU"], tt["chrF"], tt["hallucinated_per_1k"]))
    print("  T5 better than CRF:", T5.get("t5_better_on_val"))
    print("  Recommendation:", T5.get("recommendation",""))
else:
    print("  (not yet available – run t5_grammar.py)")

print("\nExtra metrics:")
if EM:
    print("  METEOR:       raw=%.4f  crf=%.4f  bert=%s" % (
        EM["METEOR"]["raw_asr"], EM["METEOR"]["crf_pipeline"],
        "%.4f" % EM["METEOR"]["bert_punct"] if EM["METEOR"]["bert_punct"] else "N/A"))
    print("  BERTScore F1: raw=%.4f  crf=%.4f  bert=%s" % (
        EM["BERTScore_F1"]["raw_asr"], EM["BERTScore_F1"]["crf_pipeline"],
        "%.4f" % EM["BERTScore_F1"]["bert_punct"] if EM["BERTScore_F1"]["bert_punct"] else "N/A"))
