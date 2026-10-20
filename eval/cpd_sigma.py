"""Per-type CPD scale sigma_c, fixed on the training split (Flow-ERD leaves the estimator open).

sigma_c = RMS displacement of logged agents of type c from their current position (t = 1.0 s) over the 8 s
future, sum over valid future steps, from 3000 training scenarios (seed 0). Writes eval/cpd_sigma.json.
"""
import json
import pickle
import random
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "cache" / "SMART" / "training"
CUR = 10


def main(n: int = 3000):
    paths = sorted(TRAIN.glob("*.pkl"))
    pick = random.Random(0).sample(paths, min(n, len(paths)))
    sq = {0: 0.0, 1: 0.0, 2: 0.0}
    cnt = {0: 0, 1: 0, 2: 0}
    for p in pick:
        with open(p, "rb") as f:
            agent = pickle.load(f)["agent"]
        pos = agent["position"][..., :2].double()
        valid = agent["valid_mask"]
        disp2 = (pos[:, CUR + 1:] - pos[:, CUR:CUR + 1]).pow(2).sum(-1)
        m = valid[:, CUR + 1:] & valid[:, CUR:CUR + 1]
        for t in sq:
            mt = m & (agent["type"] == t)[:, None]
            sq[t] += float(disp2[mt].sum())
            cnt[t] += int(mt.sum())
    sigma = {name: (sq[t] / cnt[t]) ** 0.5 for t, name in zip(sq, ("veh", "ped", "cyc"))}
    out = {"sigma": sigma, "counts": {name: cnt[t] for t, name in zip(sq, ("veh", "ped", "cyc"))},
           "n_scenarios": len(pick), "definition": __doc__.strip()}
    (ROOT / "eval" / "cpd_sigma.json").write_text(json.dumps(out, indent=1) + "\n")
    print(out["sigma"], out["counts"])


if __name__ == "__main__":
    main()
