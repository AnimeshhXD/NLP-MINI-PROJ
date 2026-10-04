"""
demo_app.py  -  Step 7: Streamlit demo for the caption pipeline.

Usage:
  streamlit run demo_app.py

Test mode (headless, no browser):
  python demo_app.py --test

Features:
  - Upload a transcript (plain text) or a Whisper JSON file
  - Choose punctuation model: CRF or BERT
  - Download .srt / .vtt captions
  - If a video file is uploaded, burn captions with ffmpeg
"""
import sys, pathlib, argparse, json, re

# ---- headless test mode (no Streamlit) -----------------------------------
if "--test" in sys.argv:
    # verify all imports work and caption generation produces valid output
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from captions.make_captions import make_timed_blocks, write_srt, write_vtt, split_into_lines

    fake_words = [
        {"word": "hello", "start": 0.0, "end": 0.4},
        {"word": "world", "start": 0.5, "end": 0.9},
        {"word": "this",  "start": 1.0, "end": 1.3},
        {"word": "is",    "start": 1.4, "end": 1.5},
        {"word": "a",     "start": 1.6, "end": 1.7},
        {"word": "test",  "start": 1.8, "end": 2.2},
    ]
    blocks = make_timed_blocks(fake_words)
    assert len(blocks) > 0,  "no blocks generated"
    for b in blocks:
        for line in b["text"]:
            assert len(line) <= 42, f"line too long: {line}"
        assert b["end"] > b["start"], "end <= start"

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        srt_path = pathlib.Path(td) / "test.srt"
        vtt_path = pathlib.Path(td) / "test.vtt"
        write_srt(blocks, srt_path)
        write_vtt(blocks, vtt_path)
        assert srt_path.stat().st_size > 0, "SRT file is empty"
        assert "WEBVTT" in vtt_path.read_text(), "VTT missing WEBVTT header"

    print("All headless tests passed.")
    sys.exit(0)

# ---- Streamlit app -------------------------------------------------------
import streamlit as st

st.set_page_config(page_title="Caption Pipeline Demo", layout="wide")
st.title("Closed-Caption Pipeline Demo")
st.caption("Punctuation restoration → timed SRT/VTT captions")

# sidebar
with st.sidebar:
    st.header("Settings")
    punct_mode = st.radio("Punctuation model", ["CRF", "BERT", "None"],
                           help="CRF is fast; BERT is more accurate but slower (CPU)")
    st.markdown("---")
    st.info("**BLOCKED:** Audio/video upload requires ffmpeg on PATH.  "
            "Currently only transcript text and Whisper JSON are supported.")

# main area
tab1, tab2 = st.tabs(["Transcript text", "Whisper JSON"])

with tab1:
    st.subheader("Paste a plain-text transcript")
    text = st.text_area("Transcript (one paragraph or multiple lines)", height=200,
                         placeholder="Paste your speech text here…")

    if st.button("Generate captions from text", disabled=not text.strip()):
        from captions.make_captions import make_timed_blocks, write_srt, write_vtt

        # tokenize into fake timed words (assume 3 words/second, no real timestamps)
        words_raw = re.findall(r"[A-Za-z']+", text)
        if not words_raw:
            st.error("No words found in transcript.")
        else:
            SPEED = 3.0  # words per second (rough proxy)
            fake_words = [
                {"word": w, "start": i / SPEED, "end": (i + 0.8) / SPEED}
                for i, w in enumerate(words_raw)
            ]

            with st.spinner("Generating captions…"):
                # load punctuation model
                punct_fn = None
                if punct_mode == "CRF":
                    if pathlib.Path("data/talks.pkl").exists():
                        from captions.make_captions import load_crf
                        predict_raw = load_crf()
                        preds = predict_raw(words_raw)
                        idx = [0]
                        def punct_fn(w):
                            p = preds[idx[0]] if idx[0] < len(preds) else "O"
                            idx[0] += 1
                            return p
                    else:
                        st.warning("data/talks.pkl not found – run data/prepare.py. Using no-punct.")
                elif punct_mode == "BERT":
                    from captions.make_captions import load_bert
                    pred_fn = load_bert()
                    if pred_fn:
                        preds = pred_fn(words_raw)
                        idx = [0]
                        def punct_fn(w):
                            p = preds[idx[0]] if idx[0] < len(preds) else "O"
                            idx[0] += 1
                            return p
                    else:
                        st.warning("BERT model not found – run punct/train_bert.py. Using no-punct.")

                blocks = make_timed_blocks(fake_words, punct_fn=punct_fn)

            st.success(f"Generated {len(blocks)} caption blocks.")
            st.warning("⚠ Timestamps are estimated (no real audio). "
                       "Upload a Whisper JSON for real timestamps.")

            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".srt", delete=False) as tf:
                srt_p = pathlib.Path(tf.name)
            with tempfile.NamedTemporaryFile(suffix=".vtt", delete=False) as tf:
                vtt_p = pathlib.Path(tf.name)

            write_srt(blocks, srt_p)
            write_vtt(blocks, vtt_p)

            col1, col2 = st.columns(2)
            col1.download_button("Download .srt", srt_p.read_bytes(),
                                  "captions.srt", "text/plain")
            col2.download_button("Download .vtt", vtt_p.read_bytes(),
                                  "captions.vtt", "text/plain")

            with st.expander("Preview caption blocks"):
                for b in blocks[:10]:
                    st.write(f"**{b['start']:.1f}s – {b['end']:.1f}s**")
                    for line in b["text"]:
                        st.write(f"  {line}")

with tab2:
    st.subheader("Upload a Whisper JSON file")
    st.caption("Produced by  asr/run_whisper.py  (requires real audio + ffmpeg)")
    uploaded = st.file_uploader("Whisper JSON", type=["json"])

    if uploaded is not None:
        from captions.make_captions import (extract_words, make_timed_blocks,
                                             write_srt, write_vtt, load_crf, load_bert)
        with st.spinner("Processing…"):
            data  = json.load(uploaded)
            words = extract_words(data)

        if not words:
            st.error("No word timestamps found in this JSON. "
                     "Make sure Whisper was run with  word_timestamps=True.")
        else:
            st.success(f"Loaded {len(words)} words with real timestamps.")

            punct_fn = None
            if punct_mode == "CRF" and pathlib.Path("data/talks.pkl").exists():
                predict_raw = load_crf()
                preds = predict_raw([w["word"].strip() for w in words])
                idx = [0]
                def punct_fn(w):
                    p = preds[idx[0]] if idx[0] < len(preds) else "O"
                    idx[0] += 1
                    return p
            elif punct_mode == "BERT":
                pred_fn = load_bert()
                if pred_fn:
                    preds = pred_fn([w["word"].strip() for w in words])
                    idx = [0]
                    def punct_fn(w):
                        p = preds[idx[0]] if idx[0] < len(preds) else "O"
                        idx[0] += 1
                        return p

            blocks = make_timed_blocks(words, punct_fn=punct_fn)
            st.success(f"Generated {len(blocks)} caption blocks (real timestamps).")

            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".srt", delete=False) as tf:
                srt_p = pathlib.Path(tf.name)
            with tempfile.NamedTemporaryFile(suffix=".vtt", delete=False) as tf:
                vtt_p = pathlib.Path(tf.name)
            write_srt(blocks, srt_p)
            write_vtt(blocks, vtt_p)

            col1, col2 = st.columns(2)
            col1.download_button("Download .srt", srt_p.read_bytes(),
                                  "captions.srt", "text/plain")
            col2.download_button("Download .vtt", vtt_p.read_bytes(),
                                  "captions.vtt", "text/plain")
