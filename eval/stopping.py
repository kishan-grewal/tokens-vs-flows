"""Whether agents come to a stop in closed-loop rollouts (added after the freeze; secondary, descriptive).

Speeds of the evaluated agents (ego + tracks to predict) come from the fast WOSAC kinematic code the metric uses
(trajectory_features.compute_kinematic_features on the rollout with the logged history prepended, central
differences, as in metric_features.compute_metric_features) and are masked by the log's speed validity. An agent is
stopped at the end if its mean speed over the last ten steps with a defined speed (t = 7.0 ... 7.9 s) is below
0.5 m/s; only agents whose logged speed is valid over that whole window count. Pooled over the split:
  stopped_end_share  share of (rollout, agent) pairs stopped at the end (for the log: share of agents),
  stop_recall        over agents stopped at the end in the log, the share of their rollouts that end stopped too,
  false_stop         over agents still moving at the end in the log (mean speed > 2 m/s), the share of rollouts
                     that end stopped,
  stopped_by_t       share of valid (rollout, agent, step) triples with speed < 0.5 m/s at each step t = 0.1 ... 7.9 s.
The end-of-rollout numbers are also given per agent type (veh, ped, cyc).
Built-in check: on runs/eval_log_replay, stop_recall = 1 and false_stop = 0.
Writes runs/<id>/stopping.json.
Usage: python eval/stopping.py --run runs/<id> [--split val_frozen]
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wosac  # noqa: E402
from wosac_fast_eval_tool.fast_sim_agents_metrics import trajectory_features  # noqa: E402

STOP, MOVING = 0.5, 2.0  # m/s
END = slice(69, 79)  # future steps t = 7.0 ... 7.9 s (the last step has no central-difference speed)


def scenario_speeds(gt: dict, ro: dict = None):
    """Speeds [R, E, 80] of the evaluated agents (the log itself if ro is None, R = 1), the log's speed validity
    [E, 80] and the agents' WOMD types [E]."""
    ids = gt["predict_agent_ids"]
    _, gi = torch.where(ids[:, None] == gt["object_ids"][None])
    log = gt["tracks"][gi][..., [0, 1, 2, 6]]  # [E, 91, (x, y, z, heading)]
    if ro is None:
        traj = log[None]
    else:
        _, pi = torch.where(ids[:, None] == ro["agent_id"][None])
        sim = ro["states"][:, pi].to(log.dtype)
        traj = torch.cat([log[None, :, :11].expand(sim.shape[0], -1, -1, -1), sim], dim=-2)
    speed = trajectory_features.compute_kinematic_features(traj, seconds_per_step=0.1)[0][..., 11:]
    valid = trajectory_features.compute_kinematic_validity(gt["track_masks"][gi][:, 11:])[0]
    return speed, valid, gt["object_types"][gi].long()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--split", default="val_frozen")
    args = ap.parse_args()
    sids = wosac.read_split(args.split)
    groups = {"all": None, **{name: code for code, name in wosac.TYPE_NAMES.items()}}
    n = {g: {k: 0.0 for k in ("pairs", "stopped", "log_stopped_agents", "recall_hits", "recall_pairs", "moving_pairs",
                              "false_stops", "agents")} for g in groups}
    by_t_hits, by_t_valid = torch.zeros(80, dtype=torch.float64), torch.zeros(80, dtype=torch.float64)
    with torch.no_grad():
        for sid in sids:
            gt = wosac.load_gt(sid, "cpu")
            log_speed, valid, types = scenario_speeds(gt)
            speed = scenario_speeds(gt, torch.load(args.run / "rollouts" / f"{sid}.pt"))[0]
            ok = valid[:, END].all(-1)  # agents with a defined logged speed over the whole end window
            stopped_all = speed[..., END].mean(-1) < STOP  # [R, E]
            log_end = log_speed[0, :, END].mean(-1)
            for g, code in groups.items():
                m = ok if code is None else ok & (types == code)
                stopped, ls, lm = stopped_all[:, m], log_end[m] < STOP, log_end[m] > MOVING
                a = n[g]
                a["agents"] += int(m.sum())
                a["pairs"] += stopped.numel()
                a["stopped"] += float(stopped.sum())
                a["log_stopped_agents"] += int(ls.sum())
                a["recall_pairs"] += float(stopped[:, ls].numel())
                a["recall_hits"] += float(stopped[:, ls].sum())
                a["moving_pairs"] += float(stopped[:, lm].numel())
                a["false_stops"] += float(stopped[:, lm].sum())
            v = valid[None].expand_as(speed)
            by_t_hits += ((speed < STOP) & v).sum((0, 1)).double()
            by_t_valid += v.sum((0, 1)).double()
    by_t = (by_t_hits / by_t_valid.clamp(min=1))[:79]
    def summary(a):
        return {"n_agents": int(a["agents"]), "stopped_end_share": a["stopped"] / max(a["pairs"], 1),
                "log_stopped_end_share": a["log_stopped_agents"] / max(a["agents"], 1),
                "stop_recall": a["recall_hits"] / a["recall_pairs"] if a["recall_pairs"] else None,
                "false_stop": a["false_stops"] / a["moving_pairs"] if a["moving_pairs"] else None}
    out = {"split": args.split, "n_scenarios": len(sids), "stop_threshold_mps": STOP, "moving_threshold_mps": MOVING,
           "end_window_s": [7.0, 7.9], **summary(n["all"]),
           "by_type": {g: summary(n[g]) for g in groups if g != "all"},
           "t_seconds": [round(0.1 * (i + 1), 1) for i in range(79)], "stopped_by_t": by_t.tolist()}
    (args.run / "stopping.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k not in ("t_seconds", "stopped_by_t")}))


if __name__ == "__main__":
    main()
