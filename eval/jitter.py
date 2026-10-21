"""How jerky the simulated motion is, inside actions and at the joins between them (added after the freeze;
secondary, descriptive).

Both heads act every 0.5 s with a 0.5 s, 5-waypoint action, so a 10 Hz rollout is a chain of actions joined at
frames 15, 20, ..., 85 (the current frame is 10). For the evaluated agents (ego + tracks to predict), the
acceleration magnitude at frame k is the second difference |p(k+1) - 2 p(k) + p(k-1)| / 0.1 s^2 of the x, y
positions, the rollout with the logged history prepended. Frames k = 11 ... 89 are split into "inside" (within an
action) and "join" (k at an action boundary). Every triple of frames must be valid in the log (as the metric masks
its kinematic features), so the log is measured on the same frames. Reported per group: median, p90, p99 and the
share above 4 m/s^2, pooled over (rollout, agent, frame); also for vehicles alone. Positions in the log carry tracking
noise, so the log is a reference level, not zero.
Built-in check: runs/eval_log_replay reproduces the log's numbers.
Writes runs/<id>/jitter.json.
Usage: python eval/jitter.py --run runs/<id> [--split val_frozen]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wosac  # noqa: E402

EDGES = np.concatenate([np.arange(0.0, 50.0, 0.02), [np.inf]])  # m/s^2 histogram bins
HARSH = 4.0  # m/s^2
VEH = 1  # WOMD object type of vehicles


def accelerations(gt: dict, ro: dict = None):
    """Acceleration magnitudes [R, E, 79] at frames 11 ... 89, their validity [E, 79] and agent types [E]."""
    ids = gt["predict_agent_ids"]
    _, gi = torch.where(ids[:, None] == gt["object_ids"][None])
    log = gt["tracks"][gi][..., :2].double()  # [E, 91, 2]
    if ro is None:
        traj = log[None]
    else:
        _, pi = torch.where(ids[:, None] == ro["agent_id"][None])
        sim = ro["states"][:, pi, :, :2].double()
        traj = torch.cat([log[None, :, :11].expand(sim.shape[0], -1, -1, -1), sim], dim=-2)
    acc = (traj[..., 2:, :] - 2 * traj[..., 1:-1, :] + traj[..., :-2, :]).norm(dim=-1) / 0.01  # centred at 1 ... 89
    m = gt["track_masks"][gi].bool()
    valid = m[:, 2:] & m[:, 1:-1] & m[:, :-2]
    return acc[..., 10:], valid[:, 10:], gt["object_types"][gi].long()


def summary(hist: np.ndarray) -> dict:
    n = hist.sum()
    if n == 0:
        return {"n": 0, "median": None, "p90": None, "p99": None, "share_above_4": None}
    cdf = np.cumsum(hist) / n
    q = lambda p: float(EDGES[min(np.searchsorted(cdf, p) + 1, len(EDGES) - 2)])  # upper edge of the bin
    return {"n": int(n), "median": q(0.5), "p90": q(0.9), "p99": q(0.99),
            "share_above_4": float(hist[EDGES[:-1] >= HARSH].sum() / n)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--split", default="val_frozen")
    args = ap.parse_args()
    frames = torch.arange(11, 90)
    join = ((frames - 10) % 5 == 0)
    groups = {f"{src}_{part}": np.zeros(len(EDGES) - 1) for src in ("sim", "log") for part in
              ("inside", "join", "inside_veh", "join_veh")}
    with torch.no_grad():
        for sid in wosac.read_split(args.split):
            gt = wosac.load_gt(sid, "cpu")
            log_acc, valid, types = accelerations(gt)
            sim_acc = accelerations(gt, torch.load(args.run / "rollouts" / f"{sid}.pt"))[0]
            for src, a in (("sim", sim_acc), ("log", log_acc)):
                for part, sel in (("inside", ~join), ("join", join)):
                    for suffix, agents in (("", torch.ones_like(types, dtype=torch.bool)), ("_veh", types == VEH)):
                        m = (valid & sel[None] & agents[:, None])[None].expand_as(a)
                        groups[f"{src}_{part}{suffix}"] += np.histogram(a[m].numpy(), EDGES)[0]
    out = {"split": args.split, "harsh_threshold_mps2": HARSH, **{k: summary(v) for k, v in groups.items()}}
    (args.run / "jitter.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: (v["median"], v["p90"], v["share_above_4"]) for k, v in out.items() if isinstance(v, dict)}))


if __name__ == "__main__":
    main()
