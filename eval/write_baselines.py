"""Write model-free baseline rollouts in the scoring format, so baselines get results.tsv rows and per-scenario
values for paired tests like any model.

Usage: python eval/write_baselines.py --baseline constant_velocity --run runs/eval_constant_velocity [--split val_frozen]
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wosac  # noqa: E402
from baselines import BASELINES  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", choices=list(BASELINES), required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--split", default="val_frozen")
    args = ap.parse_args()
    out = args.run / "rollouts"
    out.mkdir(parents=True, exist_ok=True)
    for sid in wosac.read_split(args.split):
        ro = BASELINES[args.baseline](wosac.load_gt(sid, "cpu"))
        torch.save({"agent_id": ro["agent_id"], "states": ro["states"]}, out / f"{sid}.pt")
    (args.run / "config.json").write_text(json.dumps({"head": args.baseline, "seed": ""}) + "\n")
    print(f"{args.baseline}: {len(wosac.read_split(args.split))} scenarios -> {out}")


if __name__ == "__main__":
    main()
