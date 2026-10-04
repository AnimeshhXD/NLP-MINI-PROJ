"""
env_check.py  -  Step 0: environment verification.

Prints Python, GPU, RAM, disk, ffmpeg version.
Installs any missing packages.
Writes results_new/env.json  (timestamp, versions, flags).
"""
import sys, subprocess, shutil, json, pathlib, time, importlib

OUT = pathlib.Path("results_new"); OUT.mkdir(exist_ok=True)
LOG = OUT / "env.json"
R   = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": 42}

print("=" * 55)
print("ENVIRONMENT CHECK")
print("=" * 55)

# ---- Python
R["python"] = sys.version
print("Python :", sys.version.split()[0])

# ---- GPU
try:
    import torch
    R["torch"]    = torch.__version__
    R["cuda"]     = torch.cuda.is_available()
    R["gpu_name"] = torch.cuda.get_device_name(0) if R["cuda"] else None
    if R["cuda"]:
        gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print("GPU    : %s  (%.1f GB VRAM)" % (R["gpu_name"], gb))
    else:
        print("GPU    : none – CPU only")
        print("        Using whisper-base, distilbert-base-uncased, t5-small")
        R["model_plan"] = "distilbert-base-uncased | t5-small | whisper-base"
except ImportError:
    R["cuda"] = False; R["torch"] = "missing"
    print("torch  : NOT INSTALLED")

# ---- RAM / disk
import psutil
vm = psutil.virtual_memory()
du = psutil.disk_usage(".")
R["ram_total_gb"]  = round(vm.total / 1e9, 1)
R["ram_free_gb"]   = round(vm.available / 1e9, 1)
R["disk_free_gb"]  = round(du.free / 1e9, 1)
print("RAM    : %.1f GB total, %.1f GB free" % (R["ram_total_gb"], R["ram_free_gb"]))
print("Disk   : %.1f GB free" % R["disk_free_gb"])

# ---- ffmpeg
ff = shutil.which("ffmpeg")
if ff:
    out = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
    ver = out.stdout.split("\n")[0] if out.stdout else "unknown"
    R["ffmpeg"] = ver
    print("ffmpeg : %s" % ver[:60])
else:
    R["ffmpeg"] = None
    print("ffmpeg : NOT FOUND  ← BLOCKED: audio/video processing unavailable")
    print("         Install from https://ffmpeg.org/download.html")
    print("         (Windows: https://www.gyan.dev/ffmpeg/builds/)")

# ---- Python packages
PKGS = [
    ("torch", "torch"),
    ("transformers", "transformers"),
    ("datasets", "datasets"),
    ("accelerate", "accelerate"),
    ("evaluate", "evaluate"),
    ("jiwer", "jiwer"),
    ("sacrebleu", "sacrebleu"),
    ("bert_score", "bert-score"),
    ("nltk", "nltk"),
    ("whisper", "openai-whisper"),
    ("sklearn_crfsuite", "sklearn-crfsuite"),
    ("sklearn", "scikit-learn"),
    ("sentencepiece", "sentencepiece"),
    ("streamlit", "streamlit"),
]
missing = []
print("\nPackages:")
for mod, pkg in PKGS:
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, "__version__", "ok")
        print("  %-20s %s" % (mod, ver))
        R[mod] = str(ver)
    except ImportError:
        print("  %-20s MISSING" % mod)
        missing.append(pkg)

if missing:
    print("\nInstalling:", missing)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + missing)
    print("Done.")
    R["installed"] = missing

# ---- NLTK corpora
import nltk
for c in ["state_union", "inaugural", "wordnet", "omw-1.4",
          "averaged_perceptron_tagger_eng", "punkt_tab"]:
    try:
        nltk.data.find("corpora/" + c if c not in ("punkt_tab",) else "tokenizers/" + c)
    except LookupError:
        nltk.download(c, quiet=True)
R["nltk_corpora"] = "ok"

# ---- Summary
R["status"] = "ok" if R["ffmpeg"] else "ok_no_ffmpeg"
LOG.write_text(json.dumps(R, indent=2))
print("\nSummary: cuda=%s  ffmpeg=%s  packages=%s" % (
    R["cuda"],
    "ok" if R["ffmpeg"] else "MISSING",
    "ok" if not missing else "installed_now"))
print("Wrote", LOG)
if not R["ffmpeg"]:
    print("\n⚠  Audio/ASR tasks are BLOCKED until ffmpeg is on PATH.")
