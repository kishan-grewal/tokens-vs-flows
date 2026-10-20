"""When collisions and off-road events start in closed-loop rollouts (added after the freeze; secondary, descriptive).

The per-step collision and off-road indicators of the evaluated agents come from the fast WOSAC feature code the
metric itself uses (metric_features.compute_scenario_rollouts_features), masked by the logged validity exactly as the
metric masks them before taking "any over time". Per scenario: the share of (rollout, evaluated agent) pairs whose
first event happened at or before each step t = 0.1 ... 8.0 s; then the mean over scenarios. At 8 s these are the
simulated collision and off-road rates that eval/score.py reports (a built-in check).
Writes runs/<id>/event_timing.json.
Usage: python eval/event_timing.py --run runs/<id> [--split val_frozen] [--device cpu]
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wosac  # noqa: E402
from wosac_fast_eval_tool.fast_sim_agents_metrics import metric_features  # noqa: E402


def first_event_share(per_step: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """per_step [R, E, T] bool, valid [E, T] bool -> share of (r, e) pairs with an event at or before each step [T]."""
    ev = (per_step & valid[None]).int()
    return ev.cummax(dim=-1).values.double().mean((0, 1))


def scenario_curves(sid: str, run: Path, device: str) -> dict:
    gt = wosac.load_gt(sid, device)
    ro = torch.load(run / "rollouts" / f"{sid}.pt")
    bundle = {"agent_id": ro["agent_id"].to(device), "simulated_states": ro["states"].to(device)}
    _, sim, valid = metric_features.compute_scenario_rollouts_features(gt, bundle, wosac.VERSION)
    valid = valid.to(device)
    return {"collision": first_event_share(sim["collision_per_step"], valid),
            "offroad": first_event_share(sim["offroad_per_step"], valid)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--split", default="val_frozen")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    sums = {"collision": torch.zeros(80, dtype=torch.float64), "offroad": torch.zeros(80, dtype=torch.float64)}
    sids = wosac.read_split(args.split)
    with torch.no_grad():
        for sid in sids:
            for k, v in scenario_curves(sid, args.run, args.device).items():
                sums[k] += v.cpu()
    curves = {k: (v / len(sids)).tolist() for k, v in sums.items()}
    out = {"split": args.split, "n_scenarios": len(sids), "t_seconds": [round(0.1 * (i + 1), 1) for i in range(80)],
           "share_with_event_by_t": curves,
           "at_1s": {k: v[9] for k, v in curves.items()}, "at_4s": {k: v[39] for k, v in curves.items()},
           "at_8s": {k: v[79] for k, v in curves.items()}}
    (args.run / "event_timing.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("at_1s", "at_4s", "at_8s")}))


if __name__ == "__main__":
    main()
