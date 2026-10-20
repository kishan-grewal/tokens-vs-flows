"""Score a run's closed-loop rollouts on a fixed split.

Usage:
  python eval/score.py --run runs/<id> --split val_frozen      # -> runs/<id>/{metrics.json, metrics_detail.json, per_scenario.csv}
  python eval/score.py --run runs/<id> --split test_hidden --final   # held-out split: once per run, logged
  python eval/score.py --sanity --split val_frozen             # eval sanity checks (see eval/sanity.py)
  python eval/score.py --gt-ub --split test_hidden             # log-replay ceiling of a split (see eval/sanity.py)

Expects runs/<id>/rollouts/<scenario_id>.pt = {"agent_id": [A], "states": [32, A, 80, 4]} for every scenario of
the split, and runs/<id>/config.json with "head" and "seed". metrics.json holds exactly the results.tsv fields.
Smoke settings (--max-scenarios, --n-rollouts) write metrics_smoke.json instead, so they never reach results.tsv.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wosac  # noqa: E402

RESULT_FIELDS = ["head", "seed", "split", "realism", "kinematic", "interactive", "map", "collision_rate",
                 "offroad_rate", "minADE", "CPD", "intent_entropy", "mode_mixture", "GT_UB"]


def score_rollouts(rollout_fn, ids, device, sigma, config, n_rollouts=wosac.N_ROLLOUTS, log_every=100):
    """Per-scenario rows for rollouts given by rollout_fn(scenario_id, gt) -> rollout dict."""
    rows = []
    t0 = time.time()
    for i, sid in enumerate(ids):
        gt = wosac.load_gt(sid, device)
        ro = rollout_fn(sid, gt)
        if ro["states"].shape[0] != n_rollouts or ro["states"].shape[2] != 80:
            raise ValueError(f"{sid}: rollouts of shape {tuple(ro['states'].shape)}, expected [{n_rollouts}, A, 80, 4]")
        sim = gt["sim_agent_ids"].long().cpu()
        if set(ro["agent_id"].long().tolist()) != set(sim.tolist()):
            raise ValueError(f"{sid}: rollout agents differ from the WOSAC sim agents")
        row = {"scenario_id": sid, **wosac.realism(config, gt, ro, device)}
        types = wosac.sim_agent_types(gt, ro["agent_id"])
        row["cpd"] = wosac.cpd(ro["states"].to(device), types, sigma)
        row["intent_entropy"] = wosac.ego_intent_entropy(gt, ro)
        rows.append(row)
        if log_every and (i + 1) % log_every == 0:
            print(f"scored {i + 1}/{len(ids)} ({(time.time() - t0) / (i + 1):.2f} s/scenario)", flush=True)
    return rows


def aggregate(rows, config):
    means = {k: float(np.mean([r[k] for r in rows])) for k in wosac.SCENARIO_FIELDS + ["cpd", "intent_entropy"]}
    return means, wosac.buckets(config, means)


def write_csv(path: Path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path)
    ap.add_argument("--split", default="val_frozen")
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--gt-ub", action="store_true")
    ap.add_argument("--final", action="store_true", help="required to score the held-out test_hidden split")
    ap.add_argument("--max-scenarios", type=int, default=0, help="smoke tests only; never for reported numbers")
    ap.add_argument("--n-rollouts", type=int, default=wosac.N_ROLLOUTS, help="smoke tests only")
    args = ap.parse_args()
    if args.sanity or args.gt_ub:
        import sanity
        return sanity.main(args.split) if args.sanity else sanity.write_gt_ub(args.split)

    wosac.check_trajtok_commit()
    smoke = bool(args.max_scenarios) or args.n_rollouts != wosac.N_ROLLOUTS
    ledger = wosac.ROOT / "runs" / "test_hidden_ledger.tsv"
    if args.split == "test_hidden" and not smoke:
        if not args.final:
            raise SystemExit("test_hidden is the held-out split: pass --final (scored once per run, at the end)")
        done = ledger.read_text().split() if ledger.exists() else []
        if str(args.run) in done:
            raise SystemExit(f"{args.run} was already scored on test_hidden ({ledger})")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config, sigma = wosac.load_config(), wosac.load_sigma()
    ids = wosac.read_split(args.split)
    if args.max_scenarios:
        ids = ids[:args.max_scenarios]
    rdir = args.run / "rollouts"
    rows = score_rollouts(lambda sid, gt: torch.load(rdir / f"{sid}.pt"), ids, device, sigma, config, args.n_rollouts)
    means, b = aggregate(rows, config)
    cfg = json.loads((args.run / "config.json").read_text())
    gt_ub_file = wosac.ROOT / "eval" / f"gt_ub_{args.split}.json"
    gt_ub = json.loads(gt_ub_file.read_text())["realism"] if gt_ub_file.exists() and not smoke else ""
    metrics = {
        "head": cfg.get("head", ""), "seed": cfg.get("seed", ""), "split": args.split,
        "realism": round(means["metametric"], 6), "kinematic": round(b["kinematic"], 6),
        "interactive": round(b["interactive"], 6), "map": round(b["map"], 6),
        "collision_rate": round(means["simulated_collision_rate"], 6),
        "offroad_rate": round(means["simulated_offroad_rate"], 6),
        "minADE": round(means["min_average_displacement_error"], 4), "CPD": round(means["cpd"], 6),
        "intent_entropy": round(means["intent_entropy"], 6), "mode_mixture": "", "GT_UB": gt_ub,
    }
    assert list(metrics) == RESULT_FIELDS
    detail = {**metrics, "n_scenarios": len(rows), "n_rollouts": args.n_rollouts, "means": means, "buckets": b,
              "metric": f"WOSAC {wosac.VERSION} realism via TrajTok fast metric @ {wosac.TRAJTOK_COMMIT[:7]}"}
    tag = "_smoke" if smoke else ""
    (args.run / f"metrics{tag}.json").write_text(json.dumps(metrics, indent=1) + "\n")
    (args.run / f"metrics_detail{tag}.json").write_text(json.dumps(detail, indent=1) + "\n")
    write_csv(args.run / f"per_scenario{tag}.csv", rows)
    if args.split == "test_hidden" and not smoke:
        with open(ledger, "a") as f:
            f.write(f"{args.run}\t{time.strftime('%Y-%m-%d %H:%M')}\n")
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
