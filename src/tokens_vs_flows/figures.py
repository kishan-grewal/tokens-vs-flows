"""Result figures from scored runs (runs/<id>/metrics_detail.json).

Colours follow the arm everywhere: token = blue, flow = orange (categorical slots 1 and 2 of the
validated default palette); references (log replay, constant velocity) are neutral grey.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ARM_COLORS = {"token": "#2a78d6", "flow": "#eb6834"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

PANELS = [  # (label, key in metrics_detail, higher is better)
    ("realism (meta-metric)", "realism", True),
    ("kinematic", "kinematic", True),
    ("interactive", "interactive", True),
    ("map", "map", True),
    ("collision rate", "collision_rate", False),
    ("off-road rate", "offroad_rate", False),
    ("minADE (m)", "minADE", False),
    ("CPD diversity", "CPD", None),
]


def load_detail(run: Path) -> dict:
    return json.loads((Path(run) / "metrics_detail.json").read_text())


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def head_comparison(arms: Dict[str, List[Path]], out: Path, refs: Optional[Dict[str, dict]] = None,
                    title: str = "") -> None:
    """One panel per metric: each seed as a dot and the arm mean as a tick, on an axis zoomed to the arms
    (differences between heads are small next to the baselines). Reference values (log replay, constant
    velocity) are printed under each panel title instead of drawn, so they do not flatten the comparison.

    arms: {"token": [run dirs], "flow": [run dirs]}; refs: {"log replay": {...}, "const. velocity": {...}} with the
    keys of metrics_detail.json.
    """
    details = {a: [load_detail(r) for r in runs] for a, runs in arms.items()}
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.2))
    for ax, (label, key, better) in zip(axes.flat, PANELS):
        _style(ax)
        for i, (arm, ds) in enumerate(details.items()):
            vals = np.array([d[key] for d in ds], dtype=float)
            color = ARM_COLORS.get(arm, MUTED)
            x = np.full(len(vals), i) + (np.linspace(-0.08, 0.08, len(vals)) if len(vals) > 1 else 0)
            ax.scatter(x, vals, s=26, color=color, zorder=3, linewidths=0.8, edgecolors="white")
            ax.hlines(vals.mean(), i - 0.22, i + 0.22, color=color, lw=2.2, zorder=2)
            fmt = "{:.2f}" if key == "minADE" else "{:.4f}"
            ax.text(i + 0.26, vals.mean(), fmt.format(vals.mean()), va="center", ha="left", fontsize=8, color=INK)
        ax.set_xticks(range(len(details)), list(details), fontsize=9)
        ax.set_xlim(-0.5, len(details) - 0.2)
        arrow = "" if better is None else (" ↑" if better else " ↓")
        ref_txt = "  ".join(f"{n} {float(r[key]):.3f}" for n, r in (refs or {}).items()
                            if r.get(key) not in ("", None))
        ax.set_title(label + arrow + (f"\n{ref_txt}" if ref_txt else ""), fontsize=9, color=INK)
        all_vals = np.concatenate([[d[key] for d in ds] for ds in details.values()]).astype(float)
        lo, hi = np.nanmin(all_vals), np.nanmax(all_vals)
        pad = 0.25 * (hi - lo) if hi > lo else 0.05 * max(abs(hi), 1e-3)
        ax.set_ylim(lo - pad, hi + pad)
    n = {a: len(r) for a, r in arms.items()}
    fig.suptitle(title or f"Token vs flow head on SMART-nano: val_frozen, 32 rollouts, seeds {n} "
                          "(dot = seed, tick = mean; ↑/↓ = higher/lower is better)", fontsize=10.5, color=INK)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def baseline_bars(runs: Dict[str, Path], out: Path, title: str = "") -> None:
    """Model-free baselines on one split: realism and its parts as horizontal bars from zero (one neutral colour;
    the bars are named on the axis). runs: {"log replay": run dir, ...} with metrics_detail.json."""
    details = {n: load_detail(r) for n, r in runs.items()}
    names = list(details)[::-1]
    fig, axes = plt.subplots(1, 6, figsize=(15, 0.55 * len(names) + 1.6), sharey=True)
    for ax, (label, key) in zip(axes, [("realism", "realism"), ("kinematic", "kinematic"),
                                       ("interactive", "interactive"), ("map", "map"),
                                       ("collision rate", "collision_rate"), ("off-road rate", "offroad_rate")]):
        _style(ax)
        ax.yaxis.grid(False)
        ax.xaxis.grid(True, color=GRID, lw=0.8)
        vals = [details[n][key] for n in names]
        ax.barh(range(len(names)), vals, height=0.6, color="#86b6ef", edgecolor="white", linewidth=2)
        for i, v in enumerate(vals):
            ax.text(v, i, f" {v:.3f}", va="center", ha="left", fontsize=8, color=INK)
        ax.set_xlim(0, max(1.0, max(vals) * 1.25))
        ax.set_title(label, fontsize=9, color=INK)
    axes[0].set_yticks(range(len(names)), names, fontsize=9)
    fig.suptitle(title, fontsize=10.5, color=INK)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def operating_points(sweeps: Dict[str, List[Tuple[str, Path]]], out: Path, defaults: Dict[str, str],
                     refs: Optional[Dict[str, Path]] = None, ceilings: Optional[Dict[str, Path]] = None,
                     title: str = "") -> None:
    """Realism and collision rate against rollout diversity (CPD) along each head's sampling knob.

    sweeps: {"token": [("k=16", run dir), ...], "flow": [...]} in knob order; defaults: the configured point's
    label per arm (ringed); refs: model-free baselines drawn as grey points in the overview; ceilings: replays drawn
    as dashed horizontal lines. Panels: overview, zoom on the models' best region, collision rate.
    """
    pts = {a: [(lab, load_detail(r)) for lab, r in s] for a, s in sweeps.items()}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.9), gridspec_kw={"width_ratios": [1.15, 1, 1]})
    best = max(d["realism"] for s in pts.values() for _, d in s)
    zoom = [(a, lab, d) for a, s in pts.items() for lab, d in s if d["realism"] >= best - 0.06]
    for k, (ax, key) in enumerate(zip(axes, ["realism", "realism", "collision_rate"])):
        _style(ax)
        ax.xaxis.grid(True, color=GRID, lw=0.8)
        for arm, s in pts.items():
            c = ARM_COLORS.get(arm, MUTED)
            x = [d["CPD"] for _, d in s]
            y = [d[key] for _, d in s]
            ax.plot(x, y, color=c, lw=2, zorder=2, label=f"{arm} head")
            ax.scatter(x, y, s=42, color=c, zorder=3, edgecolors="white", linewidths=1.5)
            for lab, d in s:
                if lab == defaults.get(arm):
                    ax.scatter(d["CPD"], d[key], s=150, facecolors="none", edgecolors=INK, linewidths=1.2, zorder=4)
                if k == 1 or lab == s[-1][0]:
                    ax.annotate(lab, (d["CPD"], d[key]), xytext=(6, 4), textcoords="offset points", fontsize=7.5,
                                color=MUTED)
        if k == 0:
            for name, r in (refs or {}).items():
                d = load_detail(r)
                ax.scatter(d["CPD"], d["realism"], s=30, color="#a3a29e", zorder=3, edgecolors="white", linewidths=1)
                ax.annotate(name, (d["CPD"], d["realism"]), xytext=(6, 4), textcoords="offset points", fontsize=7.5,
                            color=MUTED)
            for name, r in (ceilings or {}).items():
                v = load_detail(r)["realism"]
                ax.axhline(v, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
                ax.text(ax.get_xlim()[1], v, f"{name} {v:.3f}", fontsize=7.5, color=MUTED, ha="right", va="bottom")
            ax.set_title("realism vs rollout diversity", fontsize=9.5, color=INK)
            ax.legend(fontsize=8, frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.86))
        elif k == 1:
            xs = [d["CPD"] for _, _, d in zoom]
            ys = [d["realism"] for _, _, d in zoom]
            ax.set_xlim(min(xs) - 0.05, max(xs) + 0.12)
            ax.set_ylim(min(ys) - 0.006, max(ys) + 0.006)
            ax.set_title("zoom: the best operating points", fontsize=9.5, color=INK)
        else:
            ax.set_title("collision rate vs rollout diversity (lower is better)", fontsize=9.5, color=INK)
        ax.set_xlabel("CPD diversity of the 32 rollouts (higher = more spread)", fontsize=8, color=MUTED)
        ax.set_ylabel("collision rate" if key == "collision_rate" else "realism meta-metric", fontsize=8,
                      color=MUTED)
    fig.suptitle(title, fontsize=10.5, color=INK)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def realism_ladder(entries: List[Tuple[str, List[Path], str]], out: Path, title: str = "") -> None:
    """Realism of every policy on one axis from zero: models between the model-free floor and the replay ceilings.

    entries: [(label, [run dirs, one per seed], kind)] top to bottom, kind = "token" | "flow" | "ref". A model's bar
    is its seed mean, with one dot per seed; references are grey.
    """
    fig, ax = plt.subplots(figsize=(8.2, 0.42 * len(entries) + 1.3))
    _style(ax)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    for i, (label, runs, kind) in enumerate(entries):
        vals = np.array([load_detail(r)["realism"] for r in runs], dtype=float)
        c = ARM_COLORS.get(kind, "#c9c8c4")
        ax.barh(i, vals.mean(), height=0.62, color=c, edgecolor="white", linewidth=2)
        if len(vals) > 1:
            ax.scatter(vals, np.full(len(vals), i), s=14, color=INK, zorder=3)
        ax.text(vals.max() + 0.008, i, f"{vals.mean():.3f}" + (f"  ({len(vals)} seeds)" if len(vals) > 1 else ""),
                va="center", ha="left", fontsize=8.5, color=INK)
    ax.set_yticks(range(len(entries)), [e[0] for e in entries], fontsize=9, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("realism meta-metric (WOSAC 2025, higher is better)", fontsize=8.5, color=MUTED)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)


FIELDS = [("kinematic", ["linear_speed", "linear_acceleration", "angular_speed", "angular_acceleration"]),
          ("interactive", ["distance_to_nearest_object", "collision_indication", "time_to_collision"]),
          ("map", ["distance_to_road_edge", "offroad_indication", "traffic_light_violation"])]


def per_scenario(runs: List[Path]) -> Dict[str, np.ndarray]:
    """Per-scenario values averaged over seeds (runs/<id>/per_scenario.csv), rows sorted by scenario id."""
    import csv
    tables = []
    for r in runs:
        with open(Path(r) / "per_scenario.csv") as f:
            rows = sorted(csv.DictReader(f), key=lambda x: x["scenario_id"])
        tables.append({k: np.array([float(x[k]) for x in rows]) for k in rows[0] if k != "scenario_id"})
    return {k: np.mean([t[k] for t in tables], 0) for k in tables[0]}


def field_gaps(a_runs: List[Path], b_runs: List[Path], out: Path, labels=("token", "flow"), title: str = "",
               n_boot: int = 10000) -> None:
    """Where the realism gap between two arms comes from: per WOSAC field, the paired per-scenario difference
    b - a of the likelihood (left) and its weighted contribution to the meta-metric difference (right), with 95%
    bootstrap intervals over scenarios. Dots take the colour of the arm that is better on that field."""
    import sys
    from tokens_vs_flows import ROOT
    sys.path.insert(0, str(ROOT / "eval"))
    import wosac
    cfg = wosac.load_config()
    a, b = per_scenario(a_runs), per_scenario(b_runs)
    rng = np.random.default_rng(0)
    n = len(a["metametric"])
    idx = rng.integers(0, n, size=(n_boot, n))
    rows = [("realism (meta-metric)", "metametric", 1.0, None)]
    for bucket, fields in FIELDS:
        rows += [(f"{f.replace('_', ' ')}", f + "_likelihood", getattr(cfg, f).metametric_weight, bucket)
                 for f in fields]
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.42 * len(rows) + 1.6), sharey=True)
    for k, ax in enumerate(axes):
        _style(ax)
        ax.yaxis.grid(False)
        ax.xaxis.grid(True, color=GRID, lw=0.8)
        ax.axvline(0, color=MUTED, lw=1)
        for i, (lab, key, w, bucket) in enumerate(rows):
            d = (b[key] - a[key]) * (w if k == 1 else 1.0)
            boot = d[idx].mean(1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            c = ARM_COLORS.get(labels[1] if d.mean() > 0 else labels[0], MUTED)
            ax.plot([lo, hi], [i, i], color=c, lw=2, solid_capstyle="round")
            ax.scatter(d.mean(), i, s=46 if i else 70, color=c, zorder=3, edgecolors="white", linewidths=1.5)
            ax.annotate(f"{d.mean():+.4f}", (hi, i), xytext=(6, 0), textcoords="offset points", va="center",
                        fontsize=7.5, color=INK)
        ax.set_title(("likelihood difference" if k == 0 else "weighted contribution to the realism difference")
                     + f" ({labels[1]} − {labels[0]})", fontsize=9.5, color=INK)
        xl = ax.get_xlim()
        ax.set_xlim(xl[0], xl[1] + 0.25 * (xl[1] - xl[0]))
    for arm in labels:
        axes[1].scatter([], [], s=46, color=ARM_COLORS.get(arm, MUTED), label=f"{arm} better")
    axes[1].legend(fontsize=8, frameon=False, loc="upper right")
    names = [r[0] + (f"  [{r[3]}, w {r[2]:.2f}]" if r[3] else "") for r in rows]
    axes[0].set_yticks(range(len(rows)), names, fontsize=8.5, color=INK)
    axes[0].invert_yaxis()
    fig.suptitle(title or f"Per-field realism gap, paired over {n} val_frozen scenarios (seeds: {labels[0]} "
                          f"{len(a_runs)}, {labels[1]} {len(b_runs)}); colour = the better arm on that field",
                 fontsize=10, color=INK)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def rl_rounds(chains: Dict[str, List[Path]], out: Path, title: str = "") -> None:
    """Metrics against fine-tuning round for each head; round 0 is the behaviour-cloned start.

    chains: {"token": [bc run, round-1 run, ...], "flow": [...]} (one seed, or the seed mean if lists of lists).
    """
    keys = [("realism", "realism", True), ("kinematic", "kinematic", True), ("interactive", "interactive", True),
            ("map", "map", True), ("collision rate", "collision_rate", False), ("off-road rate", "offroad_rate", False),
            ("minADE (m)", "minADE", False), ("CPD diversity", "CPD", None)]
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.8))
    for ax, (label, key, better) in zip(axes.flat, keys):
        _style(ax)
        for arm, runs in chains.items():
            seeds = runs if runs and isinstance(runs[0], (list, tuple)) else [runs]
            n = min(len(s) for s in seeds)
            vals = np.array([[load_detail(r)[key] for r in s[:n]] for s in seeds], dtype=float)
            x = np.arange(n)
            ax.plot(x, vals.mean(0), color=ARM_COLORS.get(arm, MUTED), lw=2, marker="o", ms=5, label=arm)
            if len(seeds) > 1:
                ax.fill_between(x, vals.min(0), vals.max(0), color=ARM_COLORS.get(arm, MUTED), alpha=0.15, lw=0)
            ax.text(x[-1] + 0.1, vals.mean(0)[-1], f"{vals.mean(0)[-1]:.4f}" if key != "minADE" else
                    f"{vals.mean(0)[-1]:.2f}", fontsize=7.5, va="center", color=INK)
        arrow = "" if better is None else (" ↑" if better else " ↓")
        ax.set_title(label + arrow, fontsize=9, color=INK)
        ax.set_xticks(range(n))
        ax.set_xlim(-0.2, n - 0.4)
        ax.set_xlabel("round (0 = BC)", fontsize=8, color=MUTED)
    axes.flat[0].legend(fontsize=8, frameon=False)
    fig.suptitle(title, fontsize=10.5, color=INK)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main():
    """Rebuild the summary figures from the scored runs in runs/ (only runs that exist are used).
    Usage: python -m tokens_vs_flows.figures [--sweep-seed 0]"""
    import argparse
    from tokens_vs_flows import ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-seed", type=int, default=0)
    args = ap.parse_args()
    runs, figs = ROOT / "runs", ROOT / "figures"
    seeds = {a: [runs / f"bc_{a}_s{s}" for s in (0, 1, 2) if (runs / f"bc_{a}_s{s}" / "metrics_detail.json").exists()]
             for a in ("token", "flow")}
    n = max(len(v) for v in seeds.values())
    realism_ladder([("log replay (ceiling)", [runs / "eval_log_replay"], "ref"),
                    ("token replay (vocabulary ceiling)", [runs / "eval_token_replay"], "ref"),
                    ("token head", seeds["token"], "token"), ("flow head", seeds["flow"], "flow"),
                    ("noisy constant velocity", [runs / "eval_noisy_actions"], "ref"),
                    ("constant velocity", [runs / "eval_constant_velocity"], "ref"),
                    ("random agent", [runs / "eval_random_agent"], "ref")],
                   figs / "realism_ladder.png",
                   title="Realism on val_frozen (1000 scenarios, 32 rollouts each)"
                         + (" — heads: seed 0" if n == 1 else ""))
    s, b = args.sweep_seed, lambda name: runs / name
    sweeps = {"token": [("k=16", b(f"bc_token_s{s}_topk16")), ("k=48", b(f"bc_token_s{s}")),
                        ("k=2048", b(f"bc_token_s{s}_topk2048"))],
              "flow": [("noise 0.8", b(f"bc_flow_s{s}_noise0.8")), ("noise 1.0", b(f"bc_flow_s{s}")),
                       ("noise 1.25", b(f"bc_flow_s{s}_noise1.25")), ("noise 1.5", b(f"bc_flow_s{s}_noise1.5"))]}
    sweeps = {a: [(lab, r) for lab, r in v if (r / "metrics_detail.json").exists()] for a, v in sweeps.items()}
    operating_points(sweeps, figs / f"sampling_sweeps_s{s}.png", defaults={"token": "k=48", "flow": "noise 1.0"},
                     refs={"constant velocity": runs / "eval_constant_velocity", "noisy CV": runs / "eval_noisy_actions"},
                     ceilings={"log replay": runs / "eval_log_replay", "token replay": runs / "eval_token_replay"},
                     title=f"Sampling knobs trade diversity for realism (seed {s}, val_frozen; ringed = configured point;"
                           " token: top-k, flow: initial-noise scale)")
    field_gaps(seeds["token"], seeds["flow"], figs / "field_gaps_token_vs_flow.png")
    print("wrote", figs / "realism_ladder.png", figs / f"sampling_sweeps_s{s}.png", figs / "field_gaps_token_vs_flow.png")


if __name__ == "__main__":
    main()
