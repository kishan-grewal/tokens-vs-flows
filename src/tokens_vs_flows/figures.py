"""Result figures from scored runs (runs/<id>/metrics_detail.json).

Colours follow the arm everywhere: token = blue, flow = orange (categorical slots 1 and 2 of the
validated default palette); references (log replay, constant velocity) are neutral grey.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional

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
