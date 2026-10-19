"""Discrete-token vs flow-matching action heads on the catk SMART backbone.

catk's code is imported as the package ``src`` from third_party/catk (put first on sys.path).
TrajTok is appended only for its ``wosac_fast_eval_tool`` package; its own ``src`` is shadowed.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CATK = ROOT / "third_party" / "catk"
TRAJTOK = ROOT / "third_party" / "TrajTok"
CACHE = ROOT / "data" / "cache" / "SMART"
SPLITS = ROOT / "eval" / "splits"

if str(CATK) not in sys.path:
    sys.path.insert(0, str(CATK))
if str(TRAJTOK) not in sys.path:
    sys.path.append(str(TRAJTOK))
