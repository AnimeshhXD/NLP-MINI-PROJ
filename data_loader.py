"""
data_loader.py - shared helpers for loading data.pkl + derived resources.

Imported by caption_system_v2.py, extra_metrics.py, etc.
Not a standalone script.
"""
import pickle, pathlib
from capnlp import TrueCaser, fit_apostrophe_lexicon

_DATA = None
_TC   = None
_APOS = None

def load_data():
    """Load data.pkl once and cache it."""
    global _DATA
    if _DATA is None:
        _DATA = pickle.load(pathlib.Path("data.pkl").open("rb"))
    return _DATA

def load_tc_apos():
    """Return (TrueCaser, apostrophe_lexicon) fitted on the training split."""
    global _TC, _APOS
    if _TC is None:
        D = load_data()
        _TC   = TrueCaser().fit(D["train"].values())
        _APOS = fit_apostrophe_lexicon(D["train"].values())
    return _TC, _APOS
