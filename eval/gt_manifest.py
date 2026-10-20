"""SHA-256 of every ground-truth pickle the metric reads for the two splits -> eval/gt_manifest.sha256.

The pickles are derived data (not in git); scoring checks each one it loads against this manifest, so the
metric inputs are frozen with the code. Run once before freezing the eval.
"""
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wosac  # noqa: E402


def main():
    lines = []
    for split in ("val_frozen", "test_hidden"):
        for sid in wosac.read_split(split):
            p = wosac.GT_DIR / f"{sid}.pkl"
            lines.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {sid}")
    out = wosac.ROOT / "eval" / "gt_manifest.sha256"
    out.write_text("\n".join(lines) + "\n")
    print(f"{len(lines)} ground-truth pickles -> {out}")


if __name__ == "__main__":
    main()
