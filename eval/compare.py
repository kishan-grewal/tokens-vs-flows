"""Paired per-scenario comparison of two arms (each a set of seed runs scored on the same split).

Usage: python eval/compare.py --a runs/bc_token_s0 runs/bc_token_s1 --b runs/bc_flow_s0 runs/bc_flow_s1 [--out f.json]

For each metric: per-scenario values are averaged over each arm's seeds, then
  delta = mean over scenarios of (b - a), with a 95% bootstrap CI over scenarios (10,000 resamples, seed 0),
  paired t-test and Wilcoxon signed-rank p-values over scenarios,
  seed spread, two definitions reported side by side (needs >= 2 seeds in each arm, else undefined):
    spread_std   = max over the two arms of the std (ddof=1) of the per-seed split means,
    spread_range = max over the two arms of (max - min) of the per-seed split means.
Reporting rule: a difference is called real only if |delta| > 2 x seed spread AND p_t < 0.05 ("real_std" and
"real_range" for the two spread definitions); otherwise it is a null result at this power. Only scenarios scored
in every run are compared; the count is reported and a warning printed if any run is missing scenarios.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wosac  # noqa: E402

METRICS = {"realism": "metametric", "kinematic": None, "interactive": None, "map": None,
           "collision_rate": "simulated_collision_rate", "offroad_rate": "simulated_offroad_rate",
           "minADE": "min_average_displacement_error", "CPD": "cpd", "intent_entropy": "intent_entropy",
           **{k: k for k in wosac.LIKELIHOODS}}


def load_run(run: Path, config) -> dict:
    with open(run / "per_scenario.csv") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for r in rows:
        v = {k: float(r[col]) for k, col in METRICS.items() if col}
        for b, fields in wosac.BUCKETS.items():
            w = [getattr(config, x).metametric_weight for x in fields]
            v[b] = sum(wi * float(r[x + "_likelihood"]) for wi, x in zip(w, fields)) / sum(w)
        out[r["scenario_id"]] = v
    return out


def arm_matrix(runs, ids, metric):
    return np.array([[run[s][metric] for s in ids] for run in runs])  # [n_seed, n_scenario]


def _spread(m: np.ndarray, kind: str) -> float:
    """m: [n_seed, n_scenario] -> spread of the per-seed means, nan with fewer than 2 seeds."""
    if len(m) < 2:
        return float("nan")
    seed_means = m.mean(1)
    return float(seed_means.std(ddof=1) if kind == "std" else seed_means.max() - seed_means.min())


def compare(a_runs, b_runs, n_boot=10000, real_factor=2.0, p_max=0.05) -> dict:
    sets = [set(r) for r in a_runs + b_runs]
    ids = sorted(set.intersection(*sets))
    if len(ids) < max(len(x) for x in sets):
        print(f"WARNING: only {len(ids)} scenarios are scored in every run (max {max(len(x) for x in sets)})")
    rng = np.random.default_rng(0)
    boot_idx = rng.integers(0, len(ids), size=(n_boot, len(ids)))
    res = {"n_scenarios": len(ids), "n_seeds_a": len(a_runs), "n_seeds_b": len(b_runs), "metrics": {}}
    for m in METRICS:
        A, B = arm_matrix(a_runs, ids, m), arm_matrix(b_runs, ids, m)
        d = B.mean(0) - A.mean(0)
        boot = d[boot_idx].mean(1)
        sp = {k: float(np.max([_spread(A, k), _spread(B, k)])) for k in ("std", "range")}  # nan if undefined
        p_t = float(stats.ttest_rel(B.mean(0), A.mean(0)).pvalue) if np.any(d != 0) else 1.0
        p_w = float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
        real = {k: bool(np.isfinite(v) and abs(d.mean()) > real_factor * v and p_t < p_max) for k, v in sp.items()}
        res["metrics"][m] = {"a": float(A.mean()), "b": float(B.mean()), "delta": float(d.mean()),
                             "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
                             "p_ttest": p_t, "p_wilcoxon": p_w, "spread_std": sp["std"], "spread_range": sp["range"],
                             "seed_means_a": A.mean(1).tolist(), "seed_means_b": B.mean(1).tolist(),
                             "real_std": real["std"], "real_range": real["range"]}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", nargs="+", type=Path, required=True)
    ap.add_argument("--b", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    config = wosac.load_config()
    res = compare([load_run(r, config) for r in args.a], [load_run(r, config) for r in args.b])
    res["a"], res["b"] = [str(r) for r in args.a], [str(r) for r in args.b]
    print(f"{res['n_scenarios']} scenarios; seeds a {res['n_seeds_a']}, b {res['n_seeds_b']}")
    print(f"{'metric':38s} {'a':>9s} {'b':>9s} {'b-a':>9s} {'95% CI':>21s} {'p_t':>8s} {'sd':>7s} {'range':>7s} real(sd/range)")
    for m, v in res["metrics"].items():
        print(f"{m:38s} {v['a']:9.4f} {v['b']:9.4f} {v['delta']:+9.4f} [{v['ci95'][0]:+8.4f},{v['ci95'][1]:+8.4f}] "
              f"{v['p_ttest']:8.1e} {v['spread_std']:7.4f} {v['spread_range']:7.4f} {v['real_std']}/{v['real_range']}")
    if args.out:
        args.out.write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
