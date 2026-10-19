"""Per-type mean/std of the continuous 0.5 s action (5 waypoints x (x, y, yaw)) on training scenarios.

Writes src/tokens_vs_flows/action_stats.json, used to normalise the flow head's targets.
Usage: python -m tokens_vs_flows.action_stats [--n 3000]
"""
import argparse
import json
import random

import torch

from tokens_vs_flows import CACHE
from tokens_vs_flows.actions import clean_trajectories, segment_features
from tokens_vs_flows.data import ScenarioDataset
from tokens_vs_flows.model import ACTION_STATS

# catk's fixed (width, length) per type, TokenProcessor._get_agent_shape_and_token_traj
SHAPES = {0: (2.0, 4.8), 1: (1.0, 1.0), 2: (1.0, 2.0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    args = ap.parse_args()
    ds = ScenarioDataset(CACHE / "training", train=True)
    idx = random.Random(0).sample(range(len(ds)), min(args.n, len(ds)))
    acc = {t: [] for t in SHAPES}
    for i in idx:
        data = ds[i]
        agent = data["agent"]
        traj = clean_trajectories(agent)
        shape = torch.tensor([SHAPES[int(t)] for t in agent["type"]])
        seg = segment_features(traj, shape)
        mask = seg["in_valid"][:, 1:-1] & seg["target_valid"][:, 1:-1] & agent["train_mask"][:, None]
        tgt = seg["target"][:, 1:-1]
        for t in SHAPES:
            m = mask & (agent["type"] == t)[:, None]
            acc[t].append(tgt[m])
    out = {}
    for t, name in zip(SHAPES, ("veh", "ped", "cyc")):
        x = torch.cat(acc[t])
        out[name] = {"n": int(x.shape[0]), "mean": x.mean(0).tolist(), "std": x.std(0).clamp(min=1e-3).tolist()}
        print(name, x.shape[0], "mean", [round(v, 3) for v in out[name]["mean"][-3:]],
              "std", [round(v, 3) for v in out[name]["std"][-3:]])
    out["n_scenarios"] = len(idx)
    ACTION_STATS.write_text(json.dumps(out, indent=1))
    print("wrote", ACTION_STATS)


if __name__ == "__main__":
    main()
