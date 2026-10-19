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
    """One panel per metric: arm mean (bar) and each seed (dot); grey reference lines if given.

    arms: {"token": [run dirs], "flow": [run dirs]}; refs: {"log replay": {...}, "constant velocity": {...}}
    holding the same keys as metrics_detail.json.
    """
    details = {a: [load_detail(r) for r in runs] for a, runs in arms.items()}
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.6))
    for ax, (label, key, better) in zip(axes.flat, PANELS):
        _style(ax)
        for i, (arm, ds) in enumerate(details.items()):
            vals = np.array([d[key] for d in ds], dtype=float)
            ax.bar(i, vals.mean(), width=0.55, color=ARM_COLORS.get(arm, MUTED), alpha=0.85, edgecolor="white",
                   linewidth=2, zorder=2)
            ax.scatter(np.full(len(vals), i) + np.linspace(-0.12, 0.12, len(vals)), vals, s=18, color=INK,
                       zorder=3, linewidths=0)
            ax.text(i, vals.mean(), f"{vals.mean():.3f}" if key != "minADE" else f"{vals.mean():.2f}",
                    ha="center", va="bottom", fontsize=8, color=INK)
        for j, (name, r) in enumerate((refs or {}).items()):
            if key in r and r[key] not in ("", None):
                ax.axhline(float(r[key]), color=MUTED, lw=1, ls=(0, (4, 2)) if j == 0 else (0, (1, 2)), zorder=1)
                ax.text(len(details) - 0.45, float(r[key]), name, fontsize=7, color=MUTED, va="bottom", ha="right")
        ax.set_xticks(range(len(details)), list(details), fontsize=9)
        arrow = "" if better is None else (" (higher better)" if better else " (lower better)")
        ax.set_title(label + arrow, fontsize=9, color=INK)
        lo = min(min(d[key] for d in ds) for ds in details.values())
        hi = max(max(d[key] for d in ds) for ds in details.values())
        if key in ("realism", "kinematic", "interactive", "map"):
            ref_vals = [float(r[key]) for r in (refs or {}).values() if r.get(key) not in ("", None)]
            lo = min([lo] + ref_vals)
            hi = max([hi] + ref_vals)
            pad = 0.15 * (hi - lo + 1e-3)
            ax.set_ylim(max(0, lo - pad), hi + pad)
    n = {a: len(r) for a, r in arms.items()}
    fig.suptitle(title or f"Token vs flow head on SMART-nano, val_frozen, 32 rollouts ({n} seeds)", fontsize=11,
                 color=INK)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
