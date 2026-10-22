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
                     title: str = "", extra: Optional[Dict[str, List[Tuple[str, Path]]]] = None,
                     extra_label: str = "other policies") -> None:
    """Realism and collision rate against rollout diversity (CPD) along each head's sampling knob.

    sweeps: {"token": [("k=16", run dir), ...], "flow": [...]} in knob order; defaults: the configured point's
    label per arm (ringed); refs: model-free baselines drawn as grey points in the overview; ceilings: replays drawn
    as dashed horizontal lines; extra: other policies of each arm (e.g. fine-tuned ones) drawn as diamonds, so a
    point above its arm's curve beats every sampling setting of that diversity. Panels: overview, zoom on the
    models' best region, collision rate.
    """
    pts = {a: [(lab, load_detail(r)) for lab, r in s] for a, s in sweeps.items()}
    ext = {a: [(lab, load_detail(r)) for lab, r in s] for a, s in (extra or {}).items()}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.9), gridspec_kw={"width_ratios": [1.15, 1, 1]})
    best = max(d["realism"] for s in pts.values() for _, d in s)
    zoom = [(a, lab, d) for src in (pts, ext) for a, s in src.items() for lab, d in s if d["realism"] >= best - 0.06]
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
        for arm, s in ext.items():
            for lab, d in s:
                ax.scatter(d["CPD"], d[key], s=58, marker="D", color=ARM_COLORS.get(arm, MUTED), zorder=5,
                           edgecolors=INK, linewidths=0.8)
                if k > 0:
                    ax.annotate(lab, (d["CPD"], d[key]), xytext=(7, -11), textcoords="offset points", fontsize=7.5,
                                color=INK)
        if k == 0 and ext:
            ax.scatter([], [], s=58, marker="D", color="white", edgecolors=INK, linewidths=0.8, label=extra_label)
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
               n_boot: int = 10000, colors: Dict[str, str] = None) -> None:
    """Where the realism gap between two arms comes from: per WOSAC field, the paired per-scenario difference
    b - a of the likelihood (left) and its weighted contribution to the meta-metric difference (right), with 95%
    bootstrap intervals over scenarios. Dots take the colour of the arm that is better on that field (colors maps
    each label to a colour; default ARM_COLORS)."""
    colors = colors or ARM_COLORS
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
            c = colors.get(labels[1] if d.mean() > 0 else labels[0], MUTED)
            ax.plot([lo, hi], [i, i], color=c, lw=2, solid_capstyle="round")
            ax.scatter(d.mean(), i, s=46 if i else 70, color=c, zorder=3, edgecolors="white", linewidths=1.5)
            ax.annotate(f"{d.mean():+.4f}", (hi, i), xytext=(6, 0), textcoords="offset points", va="center",
                        fontsize=7.5, color=INK)
        ax.set_title("likelihood difference" if k == 0 else "weighted contribution to the realism difference",
                     fontsize=9.5, color=INK)
        xl = ax.get_xlim()
        ax.set_xlim(xl[0], xl[1] + 0.25 * (xl[1] - xl[0]))
    handles = [axes[1].scatter([], [], s=46, color=colors.get(arm, MUTED), label=f"{arm} better") for arm in labels]
    fig.legend(handles=handles, fontsize=8.5, frameon=False, loc="lower center", ncol=2)
    names = [r[0] + (f"  [{r[3]}, w {r[2]:.2f}]" if r[3] else "") for r in rows]
    axes[0].set_yticks(range(len(rows)), names, fontsize=8.5, color=INK)
    axes[0].invert_yaxis()
    fig.suptitle((title or f"Per-field realism gap, paired over {n} val_frozen scenarios (seeds: {labels[0]} "
                           f"{len(a_runs)}, {labels[1]} {len(b_runs)})")
                 + f"\ndifferences are {labels[1]} − {labels[0]}; colour = the better one on that field",
                 fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def event_timing(runs: Dict[str, Tuple[List[Path], str]], out: Path, title: str = "") -> None:
    """Share of (rollout, evaluated agent) pairs whose first collision / off-road event happened by time t
    (runs/<id>/event_timing.json from eval/event_timing.py), one line per policy, seed mean if several runs.

    runs: {"token head": ([run dirs], "token"), "log replay": ([run dir], "ref"), ...}; kind "token" | "flow" takes
    the arm colour, a variant such as "flow_noise" the arm colour dashed, references are grey with a dash pattern each.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    dashes = iter([(0, (5, 3)), (0, (1, 2)), (0, (6, 2, 1, 2))])
    styles = {}
    for name, (dirs, kind) in runs.items():  # kind "flow_noise": a variant of an arm, its colour dashed
        if kind in ARM_COLORS:
            styles[name] = (ARM_COLORS[kind], "-", 2.0)
        elif kind.split("_")[0] in ARM_COLORS:
            styles[name] = (ARM_COLORS[kind.split("_")[0]], (0, (5, 2)), 2.0)
        else:
            styles[name] = ("#8d8c88", next(dashes), 1.5)
    for ax, key, label in zip(axes, ["collision", "offroad"], ["collision", "off-road"]):
        _style(ax)
        ax.xaxis.grid(True, color=GRID, lw=0.8)
        placed = []
        for name, (dirs, kind) in runs.items():
            ds = [json.loads((Path(r) / "event_timing.json").read_text()) for r in dirs]
            t = np.array(ds[0]["t_seconds"])
            y = np.mean([d["share_with_event_by_t"][key] for d in ds], 0)
            c, ls, lw = styles[name]
            ax.plot(t, y, color=c, ls=ls, lw=lw, label=name)
            if all(abs(y[-1] - q) > 0.004 for q in placed):  # skip a label that would overprint another
                ax.annotate(f"{y[-1]:.3f}", (t[-1], y[-1]), xytext=(4, 0), textcoords="offset points", va="center",
                            fontsize=7.5, color=INK)
                placed.append(y[-1])
        ax.set_xlim(0, 8.9)
        ax.set_ylim(0, None)
        ax.set_xlabel("time since the end of the logged history (s)", fontsize=8, color=MUTED)
        ax.set_ylabel(f"share of agents with a {label} so far", fontsize=8, color=MUTED)
        ax.set_title(f"first {label} by time t (32 rollouts x evaluated agents, val_frozen)", fontsize=9.5,
                     color=INK)
    axes[0].legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle(title, fontsize=10.5, color=INK)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def stopping(runs: Dict[str, Tuple[List[Path], str]], out: Path, title: str = "") -> None:
    """Agents at rest over the rollout and stop recall (runs/<id>/stopping.json from eval/stopping.py), seed mean.

    runs: {"token head": ([run dirs], "token"), "log replay": ([run dir], "ref"), ...}, styled as in event_timing.
    Left: share of evaluated agents with speed < 0.5 m/s at each step (from t = 0.2 s: the metric defines no speed at
    the first simulated step). Right: of the agents the log shows stopped at the end, the share of rollouts that end
    stopped (stop recall), and of those still moving in the log, the share that end stopped (false stops).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1.3, 1]})
    dashes = iter([(0, (5, 3)), (0, (1, 2)), (0, (6, 2, 1, 2))])
    ax = axes[0]
    _style(ax)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    bars, placed = [], []
    for name, (dirs, kind) in runs.items():
        ds = [json.loads((Path(r) / "stopping.json").read_text()) for r in dirs]
        if kind in ARM_COLORS:
            c, ls = ARM_COLORS[kind], "-"
        elif kind.split("_")[0] in ARM_COLORS:
            c, ls = ARM_COLORS[kind.split("_")[0]], (0, (5, 2))
        else:
            c, ls = "#8d8c88", next(dashes)
        t = np.array(ds[0]["t_seconds"])[1:]
        y = np.mean([d["stopped_by_t"] for d in ds], 0)[1:]
        ax.plot(t, y, color=c, ls=ls, lw=2.0 if kind != "ref" else 1.5, label=name)
        if all(abs(y[-1] - q) > 0.004 for q in placed):  # skip a label that would overprint another
            ax.annotate(f"{y[-1]:.3f}", (t[-1], y[-1]), xytext=(4, 0), textcoords="offset points", va="center",
                        fontsize=7.5, color=INK)
            placed.append(y[-1])
        bars.append((name, c, [d["stop_recall"] for d in ds], [d["false_stop"] for d in ds]))
    ax.set_xlim(0, 8.6)
    ax.set_ylim(0, None)
    ax.set_xlabel("time since the end of the logged history (s)", fontsize=8, color=MUTED)
    ax.set_ylabel("share of evaluated agents at rest (< 0.5 m/s)", fontsize=8, color=MUTED)
    ax.set_title("agents at rest over the rollout", fontsize=9.5, color=INK)
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    ax = axes[1]
    _style(ax)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    for i, (name, c, rec, fs) in enumerate(bars):
        for j, (vals, alpha, lab) in enumerate([(rec, 1.0, "stop recall"), (fs, 0.45, "false stops")]):
            y = i + (j - 0.5) * 0.36
            ax.barh(y, np.mean(vals), height=0.34, color=c, alpha=alpha, edgecolor="white", linewidth=1.5,
                    label=lab if i == 0 else None)
            if len(vals) > 1:
                ax.scatter(vals, np.full(len(vals), y), s=10, color=INK, zorder=3)
            ax.text(np.max(vals) + 0.01, y, f"{np.mean(vals):.2f}", va="center", fontsize=7.5, color=INK)
    ax.set_yticks(range(len(bars)), [b[0] for b in bars], fontsize=8.5, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("share of rollouts that end stopped (t = 7.0-7.9 s)", fontsize=8, color=MUTED)
    ax.set_title("stop recall (solid) and false stops (light)", fontsize=9.5, color=INK)
    fig.suptitle(title, fontsize=10.5, color=INK)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def standstill(runs: Dict[str, Tuple[List[Path], str]], out: Path, title: str = "") -> None:
    """Agents that wait in the log, and agents that brake to a stop (runs/<id>/standstill.json from
    eval/standstill.py), seed mean; runs styled as in event_timing.
    Left: of the agents below 0.5 m/s for the log's whole 8 s, the share of rollouts moving (>= 0.5 m/s) at each step;
    the dotted line marks 1.1 s, after which the policy's 11-frame history holds only its own simulated states.
    Right: the share of rollouts that keep those agents still for all 8 s (solid), and of the agents moving at the
    start that the log brings to a stop, the share of rollouts that end stopped (light)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1.3, 1]})
    dashes = iter([(0, (5, 3)), (0, (1, 2)), (0, (6, 2, 1, 2))])
    ax = axes[0]
    _style(ax)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    bars, placed = [], []
    for name, (dirs, kind) in runs.items():
        ds = [json.loads((Path(r) / "standstill.json").read_text()) for r in dirs]
        if kind in ARM_COLORS:
            c, ls = ARM_COLORS[kind], "-"
        elif kind.split("_")[0] in ARM_COLORS:
            c, ls = ARM_COLORS[kind.split("_")[0]], (0, (5, 2))
        else:
            c, ls = "#8d8c88", next(dashes)
        t = np.array(ds[0]["waiting"]["t_seconds"])
        y = np.mean([d["waiting"]["moving_by_t"] for d in ds], 0)
        ax.plot(t, y, color=c, ls=ls, lw=2.0 if kind != "ref" else 1.5, label=name)
        if all(abs(y[-1] - q) > 0.03 for q in placed):  # skip a label that would overprint another
            ax.annotate(f"{y[-1]:.2f}", (t[-1], y[-1]), xytext=(4, 0), textcoords="offset points", va="center",
                        fontsize=7.5, color=INK)
            placed.append(y[-1])
        bars.append((name, c, [d["waiting"]["held_share"] for d in ds], [d["braking"]["end_stopped_share"] for d in ds]))
    ax.axvline(1.1, color=MUTED, lw=1, ls=":")
    ax.annotate("history now all simulated", (1.1, 1.0), xytext=(4, -2), textcoords="offset points", va="top",
                fontsize=7.5, color=MUTED)
    ax.set_xlim(0, 8.6)
    ax.set_ylim(0, 1.04)
    ax.set_xlabel("time since the end of the logged history (s)", fontsize=8, color=MUTED)
    ax.set_ylabel("share of rollouts moving (>= 0.5 m/s)", fontsize=8, color=MUTED)
    n_wait = json.loads((Path(next(iter(runs.values()))[0][0]) / "standstill.json").read_text())["waiting"]["n_agents"]
    ax.set_title(f"agents that wait for all 8 s in the log ({n_wait})", fontsize=9.5, color=INK)
    ax.legend(fontsize=8, frameon=False, loc="center right")
    ax = axes[1]
    _style(ax)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    for i, (name, c, held, braked) in enumerate(bars):
        for j, (vals, alpha, lab) in enumerate([(held, 1.0, "waiting agents kept still for 8 s"),
                                                 (braked, 0.45, "moving agents brought to a stop")]):
            y = i + (j - 0.5) * 0.36
            ax.barh(y, np.mean(vals), height=0.34, color=c, alpha=alpha, edgecolor="white", linewidth=1.5,
                    label=lab if i == 0 else None)
            if len(vals) > 1:
                ax.scatter(vals, np.full(len(vals), y), s=10, color=INK, zorder=3)
            ax.text(np.max(vals) + 0.01, y, f"{np.mean(vals):.2f}", va="center", fontsize=7.5, color=INK)
    ax.set_yticks(range(len(bars)), [b[0] for b in bars], fontsize=8.5, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("share of rollouts", fontsize=8, color=MUTED)
    ax.set_title("kept still (solid) and brought to a stop (light)", fontsize=9.5, color=INK)
    fig.suptitle(title, fontsize=10.5, color=INK)
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


def rl_behaviour(rows: List[Tuple[str, Path, str]], out: Path, title: str = "") -> None:
    """What reward fine-tuning changed in the simulated behaviour, one bar per policy and the log as reference.

    rows: [(label, run dir, kind)], kind "ref" (log replay, grey), an arm ("token" / "flow": behaviour-cloned start,
    light), "<arm>_rl" (fine-tuned, full colour) or "<arm>_ctrl" (shuffled-advantage control, hatched). Panels: realism (metrics.json), collision rate, false stops
    (agents moving at > 2 m/s at the end of the log whose rollout ends stopped; runs/<id>/stopping.json) and jerky
    motion (share of 10 Hz frames inside actions with acceleration > 4 m/s^2; runs/<id>/jitter.json). Dotted lines
    mark the log's own level.
    """
    panels = [("realism (val, 1000 scenarios) ↑", lambda r: json.loads((r / "metrics.json").read_text())["realism"]),
              ("collision rate ↓", lambda r: json.loads((r / "metrics.json").read_text())["collision_rate"]),
              ("false stops: moving in the log,\nstopped at the end of the rollout",
               lambda r: json.loads((r / "stopping.json").read_text())["false_stop"]),
              ("jerky motion: frames inside actions\nwith acceleration > 4 m/s²",
               lambda r: json.loads((r / "jitter.json").read_text())["sim_inside"]["share_above_4"])]
    fig, axes = plt.subplots(1, len(panels), figsize=(14, 0.42 * len(rows) + 1.6), sharey=True)
    for ax, (name, get) in zip(axes, panels):
        _style(ax)
        ax.xaxis.grid(True, color=GRID, lw=0.8)
        vals = []
        for i, (label, run, kind) in enumerate(rows):
            arm = kind.split("_")[0]
            v = get(Path(run))
            vals.append(v)
            if kind == "ref":
                ax.axvline(v, color="#8d8c88", ls=(0, (1, 2)), lw=1.2, zorder=1)
            c = ARM_COLORS.get(arm, "#8d8c88")
            if kind.endswith("_ctrl"):
                ax.barh(i, v, height=0.62, color="white", hatch="////", edgecolor=c, linewidth=1.0, zorder=2)
            else:
                ax.barh(i, v, height=0.62, color=c, alpha=1.0 if kind.endswith("_rl") or kind == "ref" else 0.45,
                        edgecolor="white", linewidth=1.5, zorder=2)
            ax.text(v, i, f" {v:.3f}", va="center", fontsize=7.5, color=INK, zorder=3)
        lo = min(vals) - 0.15 * (max(vals) - min(vals)) if "realism" in name else 0.0
        ax.set_xlim(lo, max(vals) + 0.22 * (max(vals) - lo))
        ax.set_title(name, fontsize=9, color=INK)
    axes[0].set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=8.5, color=INK)
    axes[0].invert_yaxis()
    fig.suptitle(title, fontsize=10.5, color=INK)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def flow_straightness(rows: List[Tuple[str, Path, str]], out: Path, title: str = "") -> None:
    """Where along the sampling path each flow model departs from straight paths (runs/<id>/flow_straightness.json
    from `python -m tokens_vs_flows.flow_straightness`): mean ||v(x_t, t) - (x_1 - x_0)||^2 per normalised dim over
    a 200-step solve, in 20 bins of t (0 everywhere = straight flow, one Euler step exact), log scale.

    rows: [(label, run dir, style)], style "bc" (behaviour cloning, light), "rl" (fine-tuned, full colour), "rl2"
    (a second fine-tuned run, dash-dot), "ctrl" (shuffled-advantage control, dashed) or "fix" (fine-tuned with the
    velocity anchor, dark). The legend gives each model's mean over t.
    """
    base = np.array([int(ARM_COLORS["flow"][i:i + 2], 16) for i in (1, 3, 5)], dtype=float)
    hexc = lambda rgb: "#" + "".join(f"{int(round(v)):02x}" for v in rgb)
    styles = {"bc": (hexc(base + (255 - base) * 0.5), "-", 3.0), "rl": (ARM_COLORS["flow"], "-", 1.8),
              "rl2": (ARM_COLORS["flow"], (0, (6, 2, 1, 2)), 1.8), "ctrl": (ARM_COLORS["flow"], (0, (4, 2)), 1.8),
              "fix": (hexc(base * 0.55), "-", 1.6)}
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    _style(ax)
    for label, run, style in rows:
        d = json.loads((Path(run) / "flow_straightness.json").read_text())
        color, ls, lw = styles[style]
        by_t = np.array(d["straightness_by_t"])
        ax.plot((np.arange(len(by_t)) + 0.5) / len(by_t), by_t, color=color, ls=ls, lw=lw,
                label=f"{label} (mean {d['straightness']:.3f})")
    ax.set_yscale("log")
    ax.set_xlim(0, 1)
    ax.set_xlabel("flow time t (0 = noise, 1 = sample)", fontsize=8.5, color=MUTED)
    ax.set_ylabel("departure from a straight path\n(per normalised action dim)", fontsize=8.5, color=MUTED)
    ax.legend(fontsize=7.5, frameon=False, loc="upper center")
    ax.set_title(title, fontsize=9.5, color=INK, loc="left")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def anchor_study(columns: List[Tuple[str, Optional[Path], List[Path], Optional[Path]]], base: Path, out: Path,
                 refs: Optional[Dict[str, Tuple[Path, Path]]] = None, log: Optional[Path] = None,
                 title: str = "") -> None:
    """Flow fine-tuning with and without anchors, one column per method: realism change vs the behaviour-cloning start
    (paired mean over scenarios, bootstrap 95% CI), false stops (runs/<id>/stopping.json), jerky frames inside actions
    (runs/<id>/jitter.json) and flow straightness (runs/<id>/flow_straightness.json).

    columns: [(label, 30-iteration run or None, [100-iteration runs], shuffled-advantage control at 100 iterations or
    None)]; open circle = 30 iterations, filled = 100, cross = the control. Dotted lines: the start's level; log: the
    log replay run, whose jerkiness is drawn dashed grey; refs: {label: (fine-tuned run, its start)} drawn as dashed
    lines in the realism panel.
    """
    b = per_scenario([base])["metametric"]
    rng = np.random.default_rng(0)

    def delta(run):
        d = per_scenario([run])["metametric"] - b
        boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
        return d.mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5)

    load = lambda run, f: json.loads((Path(run) / f).read_text())
    panels = [("realism change vs behaviour cloning ↑", None),
              ("false stops ↓", lambda r: load(r, "stopping.json")["false_stop"]),
              ("jerky frames inside actions\n(acceleration > 4 m/s²)", lambda r: load(r, "jitter.json")["sim_inside"]["share_above_4"]),
              ("flow straightness ↓ (0 = straight)", lambda r: load(r, "flow_straightness.json")["straightness"])]
    c, x = ARM_COLORS["flow"], np.arange(len(columns))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 4.2))
    for ax, (name, get) in zip(axes, panels):
        _style(ax)
        for i, (label, r30, r100, ctrl) in enumerate(columns):
            for run, kind in [(r30, "30")] + [(r, "100") for r in r100] + [(ctrl, "ctrl")]:
                if run is None or not (Path(run) / "metrics.json").exists():
                    continue
                dx = {"30": -0.15, "100": 0.0, "ctrl": 0.15}[kind]
                if get is None:
                    v, lo, hi = delta(run)
                    ax.plot([i + dx, i + dx], [lo, hi], color=MUTED if kind == "ctrl" else c, lw=1.2, zorder=2)
                else:
                    try:
                        v = get(run)
                    except FileNotFoundError:
                        continue
                style = {"30": dict(marker="o", facecolors="white", edgecolors=c, s=46, linewidths=1.6),
                         "100": dict(marker="o", color=c, edgecolors="white", s=54, linewidths=1.0),
                         "ctrl": dict(marker="X", color=MUTED, edgecolors="white", s=60, linewidths=0.8)}[kind]
                ax.scatter(i + dx, v, zorder=3, **style)
        if get is None:
            ax.axhline(0, color=MUTED, lw=1, ls=(0, (1, 2)))
            for lab, (run, start) in (refs or {}).items():
                d = (per_scenario([run])["metametric"] - per_scenario([start])["metametric"]).mean()
                ax.axhline(d, color=ARM_COLORS["token"], lw=1.2, ls=(0, (4, 3)), label=f"{lab} ({d:+.4f})")
        else:
            ax.axhline(get(base), color=MUTED, lw=1, ls=(0, (1, 2)))
            if "jerky" in name and log is not None:
                ax.axhline(get(log), color="#8d8c88", lw=1.2, ls=(0, (4, 3)), label="the log")
        ax.set_xticks(x, [col[0] for col in columns], fontsize=7.5, rotation=30, ha="right")
        ax.set_title(name, fontsize=9, color=INK)
    axes[0].scatter([], [], marker="o", facecolors="white", edgecolors=c, s=46, linewidths=1.6, label="30 iterations")
    axes[0].scatter([], [], marker="o", color=c, s=54, label="100 iterations")
    axes[0].scatter([], [], marker="X", color=MUTED, s=60, label="100 iterations, shuffled-advantage control (no reward)")
    axes[0].plot([], [], color=MUTED, lw=1, ls=(0, (1, 2)), label="behaviour-cloning start")
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles += [hh for hh, ll in zip(h, l) if ll not in labels]
        labels += [ll for ll in l if ll not in labels]
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=8, frameon=False)
    fig.suptitle(title, fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def _ramp(hex_color: str, n: int) -> List[str]:
    """n steps of one hue from light to full (blended with white), for an ordered knob such as the clip range."""
    rgb = np.array([int(hex_color[i:i + 2], 16) for i in (1, 3, 5)], dtype=float)
    return ["#" + "".join(f"{int(round(v)):02x}" for v in rgb + (255 - rgb) * m)
            for m in (np.linspace(0.6, 0.0, n) if n > 1 else [0.0])]


def rl_curves(rows: Dict[str, Tuple[str, List[Tuple[str, Path]]]], out: Path,
              emphasis: Optional[Dict[str, str]] = None, smooth: int = 5, title: str = "") -> None:
    """On-policy fine-tuning diagnostics against iteration (runs/<id>/iters.csv), one row per arm.

    rows: {"token head, PPO": ("token", [(label, run dir), ...]), ...} with the runs in knob order (lighter to darker
    in the arm's hue); a label containing "non-negative" or "shuffled" is dashed; emphasis: the label drawn thick in
    each row. Columns: realism of the training rollouts (rolling mean over `smooth` iterations; the scenes change
    every iteration but are the same for every run of a seed), the anchor's KL estimate to the BC model (log scale)
    and the fraction of clipped ratios. The legend gives each run's val_frozen realism change vs its BC start.
    """
    fig, axes = plt.subplots(len(rows), 3, figsize=(14, 3.7 * len(rows)), squeeze=False)
    for i, (ax_row, (row_title, (arm, entries))) in enumerate(zip(axes, rows.items())):
        solid = [lab for lab, _ in entries if not any(w in lab for w in ("non-negative", "shuffled"))]
        colors = dict(zip(solid, _ramp(ARM_COLORS.get(arm, MUTED), len(solid))))
        for lab, run in entries:
            it = np.genfromtxt(Path(run) / "iters.csv", delimiter=",", names=True)
            init = Path(run).resolve().parents[1] / json.loads((Path(run) / "config.json").read_text())["init"]
            delta = load_detail(run)["realism"] - load_detail(init)["realism"]
            style = dict(color=colors.get(lab, ARM_COLORS.get(arm, MUTED)), ls="-" if lab in colors else (0, (4, 2)),
                         lw=2.6 if (emphasis or {}).get(row_title) == lab else 1.4)
            x = it["iter"]
            ax_row[0].plot(x[smooth - 1:], np.convolve(it["train_realism_k"], np.ones(smooth) / smooth, "valid"),
                           label=f"{lab}  (val {delta:+.4f})", **style)
            ax_row[1].plot(x, it["k3"], **style)
            ax_row[2].plot(x, it["clip_frac"], **style)
        for j, ax in enumerate(ax_row):
            _style(ax)
            if i == 0:
                ax.set_title([f"realism of the training rollouts ({smooth}-iteration mean)",
                              "KL estimate to the BC model (k3, log scale)", "fraction of clipped ratios"][j],
                             fontsize=9.5, color=INK)
            if i == len(rows) - 1:
                ax.set_xlabel("iteration", fontsize=8, color=MUTED)
        ax_row[1].set_yscale("log")
        ax_row[0].set_ylabel(row_title, fontsize=9, color=INK)
        ax_row[0].legend(fontsize=7.5, frameon=False, loc="lower left")
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
    sweeps = {"token": [(f"k={k}", b(f"bc_token_s{s}" if k == 48 else f"bc_token_s{s}_topk{k}"))
                        for k in (8, 16, 24, 32, 48, 2048)],
              "flow": [(f"noise {v}", b(f"bc_flow_s{s}" if v == "1.0" else f"bc_flow_s{s}_noise{v}"))
                       for v in ("0.8", "0.9", "1.0", "1.25", "1.5")]}
    sweeps = {a: [(lab, r) for lab, r in v if (r / "metrics_detail.json").exists()] for a, v in sweeps.items()}
    operating_points(sweeps, figs / f"sampling_sweeps_s{s}.png", defaults={"token": "k=48", "flow": "noise 1.0"},
                     refs={"constant velocity": runs / "eval_constant_velocity", "noisy CV": runs / "eval_noisy_actions"},
                     ceilings={"log replay": runs / "eval_log_replay", "token replay": runs / "eval_token_replay"},
                     title=f"Sampling knobs trade diversity for realism (seed {s}, val_frozen; ringed = configured point;"
                           " token: top-k, flow: initial-noise scale)")
    field_gaps(seeds["token"], seeds["flow"], figs / "field_gaps_token_vs_flow.png")
    noise_seeds = [runs / f"bc_flow_noise_s{k}" for k in (0, 1, 2) if (runs / f"bc_flow_noise_s{k}").exists()]
    still = {"log replay": ([runs / "eval_log_replay"], "ref"), "token replay": ([runs / "eval_token_replay"], "ref"),
             "token head": (seeds["token"], "token"), "flow head": (seeds["flow"], "flow"),
             "flow head + history noise": (noise_seeds, "flow_noise")}
    still = {k: v for k, v in still.items() if v[0] and all((r / "standstill.json").exists() for r in v[0])}
    if len(still) == 5:
        standstill(still, figs / "standstill.png",
                   title="Waiting and braking (val_frozen, 32 rollouts per scenario; heads: mean over seeds, "
                         "dots = single seeds)")
    frozen = runs / f"fpo_clip0.01_flow_noise_s{s}"
    if (frozen / "per_scenario.csv").exists():
        field_gaps([runs / f"bc_flow_noise_s{s}"], [frozen], figs / f"field_gaps_freezing_s{s}.png",
                   labels=("behaviour cloning", "after 30 FPO iterations"),
                   colors={"behaviour cloning": MUTED, "after 30 FPO iterations": ARM_COLORS["flow"]},
                   title=f"Where the traffic-freezing run's realism gain comes from (flow head with history noise, "
                         f"seed {s}, paired over 1000 val_frozen scenarios)")
    behaviour = [("log replay", runs / "eval_log_replay", "ref"),
                 ("token: behaviour cloning", runs / f"bc_token_s{s}", "token"),
                 ("token: PPO, 100 iterations", runs / f"ppo_token_s{s}", "token_rl"),
                 ("token: PPO control (shuffled advantages)", runs / f"ppo_shuffled_token_s{s}", "token_ctrl"),
                 ("flow (history noise): behaviour cloning", runs / f"bc_flow_noise_s{s}", "flow"),
                 ("flow: FPO run 1, 100 iterations", runs / f"fpo_flow_noise_s{s}", "flow_rl"),
                 ("flow: FPO run 2, 30 iterations", runs / f"fpo_clip0.01_flow_noise_s{s}", "flow_rl"),
                 ("flow: FPO run 2, 100 iterations", runs / f"fpo_rerun_flow_noise_s{s}", "flow_rl"),
                 ("flow: FPO control (shuffled advantages)", runs / f"fpo_shuffled_flow_noise_s{s}", "flow_ctrl"),
                 ("flow: FPO + velocity anchor, 30 iterations", runs / f"fpo_vanchor10_flow_noise_s{s}", "flow_rl"),
                 ("flow: FPO + velocity anchor, 100 iterations", runs / f"fpo_anchored_flow_noise_s{s}", "flow_rl"),
                 ("flow: anchored control (shuffled)", runs / f"fpo_anchored_shuffled_flow_noise_s{s}", "flow_ctrl")]
    behaviour = [r for r in behaviour if all((r[1] / f).exists() for f in ("metrics.json", "stopping.json", "jitter.json"))]
    rl_behaviour(behaviour, figs / f"rl_behaviour_s{s}.png",
                 title=f"What reward fine-tuning changed (seed {s}, val_frozen; both FPO runs use identical settings)")
    anchored = runs / f"fpo_anchored_flow_noise_s{s}"
    rows = [("behaviour cloning", runs / f"bc_flow_noise_s{s}", "bc"),
            ("FPO run 1, 100 iterations", runs / f"fpo_flow_noise_s{s}", "rl"),
            ("FPO run 2, 100 iterations", runs / f"fpo_rerun_flow_noise_s{s}", "rl2"),
            ("FPO control (shuffled advantages)", runs / f"fpo_shuffled_flow_noise_s{s}", "ctrl"),
            ("FPO + velocity anchor, " + ("100" if (anchored / "flow_straightness.json").exists() else "30")
             + " iterations", anchored if (anchored / "flow_straightness.json").exists()
             else runs / f"fpo_vanchor10_flow_noise_s{s}", "fix")]
    noise = lambda name: runs / f"{name}_flow_noise_s{s}"
    anchor_study([("no anchor", noise("fpo_clip0.01"), [noise("fpo"), noise("fpo_rerun")], noise("fpo_shuffled")),
                  ("uniform anchor, w = 0.01", noise("fpo_vanchor0.01"), [noise("fpo_anchor0.01")],
                   noise("fpo_anchor0.01_shuffled")),
                  ("uniform, w = 0.1", noise("fpo_vanchor0.1"), [], None),
                  ("uniform, w = 1", noise("fpo_vanchor1"), [], None),
                  ("uniform, w = 10", noise("fpo_vanchor10"), [noise("fpo_anchored")], noise("fpo_anchored_shuffled")),
                  ("clean-sample anchor, w = 10", noise("fpo_x1anchor10"), [noise("fpo_anchorx1")],
                   noise("fpo_anchorx1_shuffled"))],
                 runs / f"bc_flow_noise_s{s}", figs / f"anchor_study_s{s}.png",
                 refs={"PPO on the token head, 100 it.": (runs / f"ppo_token_s{s}", runs / f"bc_token_s{s}")},
                 log=runs / "eval_log_replay",
                 title=f"Anchoring FPO on the flow head (history noise, seed {s}): realism and behaviour by anchor")
    flow_straightness([r for r in rows if (r[1] / "flow_straightness.json").exists()],
                      figs / f"flow_straightness_s{s}.png",
                      title=f"FPO bends the flow head's sampling paths, with or without reward;\nthe velocity anchor "
                            f"keeps them as straight as behaviour cloning's (seed {s})")
    print("wrote", figs / "realism_ladder.png", figs / f"sampling_sweeps_s{s}.png", figs / "field_gaps_token_vs_flow.png",
          figs / f"field_gaps_freezing_s{s}.png", figs / "standstill.png",
          figs / f"rl_behaviour_s{s}.png", figs / f"flow_straightness_s{s}.png", figs / f"anchor_study_s{s}.png")


if __name__ == "__main__":
    main()
