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
