"""Closed-loop displacement growth of a run: ADE@8s / ADE@1s (added after the freeze; input to the pre-registered
check of compounding error between heads).

ADE@T = mean over the 32 rollouts, the evaluated agents (ego + tracks_to_predict, as in the WOSAC minADE) and their
logged-valid future steps t <= T of the 2D distance to the log. Pooled over the split's scenarios.
Writes runs/<id>/ade_growth.json with ADE@1..8 s and the ratio ADE@8s / ADE@1s.
Usage: python eval/ade_growth.py --run runs/<id> [--split val_frozen]
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wosac  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--split", default="val_frozen")
    args = ap.parse_args()
    sums, counts = torch.zeros(80, dtype=torch.float64), torch.zeros(80, dtype=torch.float64)
    for sid in wosac.read_split(args.split):
        gt = wosac.load_gt(sid, "cpu")
        ro = torch.load(args.run / "rollouts" / f"{sid}.pt")
        ev = gt["predict_agent_ids"].long()
        ids = gt["object_ids"].long()
        g_rows = (ev[:, None] == ids[None]).float().argmax(1)
        r_rows = (ev[:, None] == ro["agent_id"].long()[None]).float().argmax(1)
        cur = int(gt["current_time_index"])
        log = gt["tracks"][g_rows, cur + 1:cur + 81, :2].double()  # [E, 80, 2]
        valid = gt["track_masks"][g_rows, cur + 1:cur + 81]  # [E, 80]
        sim = ro["states"][:, r_rows, :, :2].double()  # [32, E, 80, 2]
        d = (sim - log[None]).norm(dim=-1) * valid[None]
        sums += d.sum((0, 1))
        counts += valid.sum(0).double() * sim.shape[0]
    cum = sums.cumsum(0) / counts.cumsum(0).clamp(min=1)
    ade = {f"{s}s": float(cum[10 * s - 1]) for s in range(1, 9)}
    out = {"split": args.split, "ade": ade, "growth_8s_over_1s": ade["8s"] / ade["1s"]}
    (args.run / "ade_growth.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
