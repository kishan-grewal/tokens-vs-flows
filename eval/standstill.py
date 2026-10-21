"""Whether agents that wait in the log keep still in closed-loop rollouts, and whether moving agents brake to a stop
(a follow-up to eval/stopping.py; secondary, descriptive).

Speeds as in eval/stopping.py (the fast WOSAC kinematic code on the rollout with the logged history prepended, masked
by the log's speed validity), over the future steps t = 0.2 ... 7.9 s (the first and last future steps have no
central-difference validity); only agents whose logged speed is valid over that whole window count. Two groups of evaluated agents (ego + tracks to predict), pooled over the split:
  waiting   below 0.5 m/s at every step of the log's 8 s (parked, or held at a light the whole time):
              held_share         share of their rollouts that keep them below 0.5 m/s at every step,
              end_stopped_share  share that end stopped (mean speed over t = 7.0 ... 7.9 s below 0.5 m/s),
              median_speed, p90_speed  of their rollout speeds over all steps and rollouts (m/s),
              first_move_s_median  over rollouts that move, the median time of the first step at or above 0.5 m/s,
              speed_by_t, moving_by_t  per step, the median rollout speed and the share at or above 0.5 m/s;
  braking   moving at the start in the log (mean speed over t = 0.2 ... 1.1 s above 2 m/s) and stopped at the end:
              end_stopped_share  share of their rollouts that end stopped too.
Together the two split eval/stopping.py's stop recall into "stays put" and "comes to a stop".
Built-in check: on runs/eval_log_replay every share is 1.
Writes runs/<id>/standstill.json.
Usage: python eval/standstill.py --run runs/<id> [--split val_frozen]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wosac  # noqa: E402
from stopping import END, MOVING, STOP, scenario_speeds  # noqa: E402

FUT = slice(1, 79)  # t = 0.2 ... 7.9 s (the first and last future steps have no central-difference validity)
START = slice(1, 11)  # t = 0.2 ... 1.1 s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--split", default="val_frozen")
    args = ap.parse_args()
    sids = wosac.read_split(args.split)
    w = {"agents": 0, "pairs": 0, "held": 0.0, "ended": 0.0}
    b = {"agents": 0, "pairs": 0, "ended": 0.0}
    speeds, first_move = [], []
    with torch.no_grad():
        for sid in sids:
            gt = wosac.load_gt(sid, "cpu")
            log_speed, valid, _ = scenario_speeds(gt)
            speed = scenario_speeds(gt, torch.load(args.run / "rollouts" / f"{sid}.pt"))[0]  # [R, E, 80]
            log = log_speed[0]
            ok = valid[:, FUT].all(-1)
            waiting = ok & (log[:, FUT] < STOP).all(-1)
            braking = ok & (log[:, START].mean(-1) > MOVING) & (log[:, END].mean(-1) < STOP)
            ended = speed[..., END].mean(-1) < STOP  # [R, E]
            if waiting.any():
                s = speed[:, waiting][..., FUT]  # [R, W, 79]
                moving = s >= STOP
                moved = moving.any(-1)
                w["agents"] += int(waiting.sum())
                w["pairs"] += moved.numel()
                w["held"] += float((~moved).sum())
                w["ended"] += float(ended[:, waiting].sum())
                speeds.append(s.reshape(-1, s.shape[-1]).numpy())
                if moved.any():
                    first_move.append((moving.float().argmax(-1)[moved] * 0.1 + 0.2).numpy())
            if braking.any():
                b["agents"] += int(braking.sum())
                b["pairs"] += ended[:, braking].numel()
                b["ended"] += float(ended[:, braking].sum())
    sp = np.concatenate(speeds) if speeds else np.zeros((1, FUT.stop - FUT.start))  # [pairs, steps]
    fm = np.concatenate(first_move) if first_move else np.zeros(0)
    out = {"split": args.split, "n_scenarios": len(sids), "stop_threshold_mps": STOP, "moving_threshold_mps": MOVING,
           "waiting": {"n_agents": w["agents"], "held_share": w["held"] / max(w["pairs"], 1),
                       "end_stopped_share": w["ended"] / max(w["pairs"], 1),
                       "median_speed": float(np.median(sp)), "p90_speed": float(np.percentile(sp, 90)),
                       "first_move_s_median": float(np.median(fm)) if len(fm) else None,
                       "t_seconds": [round(0.1 * (i + 1), 1) for i in range(FUT.start, FUT.stop)],
                       "speed_by_t": [round(float(v), 4) for v in np.median(sp, 0)],
                       "moving_by_t": [round(float(v), 4) for v in (sp >= STOP).mean(0)]},
           "braking": {"n_agents": b["agents"], "end_stopped_share": b["ended"] / max(b["pairs"], 1)}}
    (args.run / "standstill.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: ({kk: vv for kk, vv in v.items() if not kk.endswith("_t") and kk != "t_seconds"}
                          if isinstance(v, dict) else v) for k, v in out.items()}))


if __name__ == "__main__":
    main()
