"""Animated bird's-eye view of closed-loop rollouts: the logged future next to each run's rollout of the same scenario.

Agents are boxes with their logged length and width. The 1.1 s history (frames 0-10) is the log in every panel; after
it each panel plays its own 8 s. One rollout index is shown (rollout 0 of 32 unless --rollout), never the best one.
The ego vehicle is green; boxes that overlap another box are red (a collision).
Usage: python -m tokens_vs_flows.animate --runs runs/bc_token_s0 runs/bc_flow_s0 --labels "token head" "flow head" \
           --scenario <id> [--out figures/rollouts_<id>.gif]
       python -m tokens_vs_flows.animate --runs ... --labels ... --pick 3   # the 3 busiest val_frozen scenarios
       python -m tokens_vs_flows.animate --runs ... --labels ... --pick 1 --by turning   # the most turning agents
"""
import argparse
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.collections import LineCollection, PolyCollection  # noqa: E402

from tokens_vs_flows import CACHE, ROOT  # noqa: E402
from tokens_vs_flows.data import read_split  # noqa: E402

EGO, AGENT, EDGE, LANE, HIT = "#1baf7a", "#8d8c88", "#52514e", "#dcdbd7", "#d62f2f"


def load_gt(scenario_id: str) -> dict:
    with open(CACHE / "validation_gt" / f"{scenario_id}.pkl", "rb") as f:
        return pickle.load(f)


def boxes(xyh: np.ndarray, lw: np.ndarray) -> np.ndarray:
    """Corners [A, 4, 2] of boxes centred at xyh [A, 3] (x, y, heading) with sizes lw [A, 2]."""
    c, s = np.cos(xyh[:, 2]), np.sin(xyh[:, 2])
    half = lw / 2
    local = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]], dtype=float)
    dx = local[None, :, 0] * half[:, None, 0]
    dy = local[None, :, 1] * half[:, None, 1]
    return np.stack([xyh[:, None, 0] + c[:, None] * dx - s[:, None] * dy,
                     xyh[:, None, 1] + s[:, None] * dx + c[:, None] * dy], -1)


def overlapping(corners: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Which boxes overlap another valid box (separating-axis test on the 4 edge normals of each pair)."""
    idx = np.flatnonzero(valid)
    hit = np.zeros(len(corners), bool)
    cen = corners.mean(1)
    for n, i in enumerate(idx):
        for j in idx[n + 1:]:
            if np.linalg.norm(cen[i] - cen[j]) > 12:
                continue
            sep = False
            for poly in (corners[i], corners[j]):
                for k in range(2):
                    e = poly[k + 1] - poly[k]
                    ax_ = np.array([-e[1], e[0]])
                    pi, pj = corners[i] @ ax_, corners[j] @ ax_
                    if pi.max() < pj.min() or pj.max() < pi.min():
                        sep = True
                        break
                if sep:
                    break
            if not sep:
                hit[i] = hit[j] = True
    return hit


def panel_states(gt: dict, rows: np.ndarray, rollout: np.ndarray = None):
    """Poses [A, 91, 3] and validity [A, 91]: logged history, then the log (rollout None) or the rollout [A, 80, 4]."""
    tr = gt["tracks"].numpy()
    xyh = tr[rows][:, :, [0, 1, 6]].copy()
    valid = gt["track_masks"].numpy()[rows].copy()
    if rollout is not None:
        xyh[:, 11:] = rollout[:, :, [0, 1, 3]]
        valid[:, 11:] = True
    return xyh, valid


def animate(scenario_id: str, runs, labels, out: Path, rollout_idx: int = 0, stride: int = 2, dpi: int = 85):
    """Writes a GIF that plays in real time with one frame every `stride` steps of 0.1 s."""
    gt = load_gt(scenario_id)
    ro = [torch.load(Path(r) / "rollouts" / f"{scenario_id}.pt") for r in runs]
    agent_id = ro[0]["agent_id"].numpy()
    ids = gt["object_ids"].numpy()
    rows = np.array([np.flatnonzero(ids == a)[0] for a in agent_id])
    lw = gt["tracks"].numpy()[rows][:, 10, [3, 4]]
    ego = rows == gt["sdc_track_index"]
    panels = [("logged future", *panel_states(gt, rows))]
    for lab, r in zip(labels, ro):
        order = np.array([np.flatnonzero(r["agent_id"].numpy() == a)[0] for a in agent_id])
        panels.append((f"{lab} (rollout {rollout_idx})", *panel_states(gt, rows, r["states"][rollout_idx, order].numpy())))
    # one square viewport for all panels: the ego's paths plus a margin
    ego_xy = np.concatenate([p[1][ego][p[2][ego]][:, :2] for p in panels])
    lo, hi = ego_xy.min(0), ego_xy.max(0)
    half = max((hi - lo).max() / 2 + 18, 35)
    mid = (lo + hi) / 2
    edges = [np.asarray(e)[:, :2] for e in gt["road_edges"]]
    with open(CACHE / "validation" / f"{scenario_id}.pkl", "rb") as f:
        lanes = list(pickle.load(f)["map_save"]["traj_pos"].numpy())  # all map segments (lanes, lines, crosswalks)

    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.6))
    arts = []
    for ax, (title, xyh, valid) in zip(axes, panels):
        ax.add_collection(LineCollection(lanes, colors=LANE, linewidths=0.7, zorder=0))
        ax.add_collection(LineCollection(edges, colors=EDGE, linewidths=1.0, zorder=1))
        trail = LineCollection([], colors=AGENT, linewidths=1.0, alpha=0.6, zorder=2)
        polys = PolyCollection([], zorder=3, linewidths=1.2)
        ax.add_collection(trail)
        ax.add_collection(polys)
        ax.set_xlim(mid[0] - half, mid[0] + half)
        ax.set_ylim(mid[1] - half, mid[1] + half)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(LANE)
        ax.set_title(title, fontsize=10, color="#0b0b0b")
        arts.append((trail, polys, xyh, valid))
    clock = fig.text(0.5, 0.03, "", ha="center", fontsize=10, color="#52514e")
    fig.suptitle(f"WOMD scenario {scenario_id}: same 1.1 s history, then 8 s closed loop", fontsize=10.5,
                 color="#0b0b0b")
    fig.tight_layout(rect=(0, 0.06, 1, 1))

    def frame(t):
        for trail, polys, xyh, valid in arts:
            v = valid[:, t]
            corners = boxes(xyh[:, t], lw)
            hit = overlapping(corners, v)
            polys.set_verts(list(corners[v]))
            polys.set_facecolors([HIT if h else (EGO if e else AGENT) for e, h in zip(ego[v], hit[v])])
            polys.set_edgecolors("white")
            t0 = max(0, t - 10)
            segs = [xyh[a, t0:t + 1, :2][valid[a, t0:t + 1]] for a in np.flatnonzero(v)]
            trail.set_segments([s for s in segs if len(s) > 1])
        phase = "history (log)" if t <= 10 else "closed-loop rollout"
        clock.set_text(f"t = {(t - 10) / 10:+.1f} s   {phase}      green: ego vehicle   red: in collision")
        return []

    anim = FuncAnimation(fig, frame, frames=range(0, 91, stride))
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out, writer=PillowWriter(fps=10 / stride), dpi=dpi)
    plt.close(fig)
    shrink_gif(out, 100 * stride)
    print("wrote", out)


def shrink_gif(path: Path, duration_ms: int, colors: int = 96) -> None:
    """Re-encode a GIF with one palette built from frames across the whole clip (so colours that appear late,
    like collision outlines, survive) and without dithering, which lets unchanged regions compress (~10x smaller)."""
    from PIL import Image, ImageSequence
    frames = [f.convert("RGB") for f in ImageSequence.Iterator(Image.open(path))]
    sample = frames[::max(1, len(frames) // 8)]
    keys = [EGO, HIT, AGENT, EDGE, LANE]  # large swatches keep the key colours exact in the palette
    h = sample[0].height
    montage = Image.new("RGB", (sample[0].width, h * len(sample) + 300), "white")
    for i, f in enumerate(sample):
        montage.paste(f, (0, i * h))
    for k, c in enumerate(keys):
        montage.paste(Image.new("RGB", (sample[0].width // len(keys), 300), c), (k * sample[0].width // len(keys), h * len(sample)))
    pal = montage.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    q[0].save(path, save_all=True, append_images=q[1:], duration=duration_ms, loop=0, optimize=True, disposal=1)


def busiest(split: str, n: int, by: str = "busy"):
    """The n scenarios of the split with the most interaction: simulated agents that turn (heading change > 0.8 rad
    while moving > 10 m in the log) count 3, other agents moving > 10 m count 1; by="turning" ranks by the number of
    turning agents first (ties by that score). Chosen from the log alone, never from any model's rollouts or scores."""
    score = []
    for sid in read_split(split):
        gt = load_gt(sid)
        rows = np.flatnonzero(np.isin(gt["object_ids"].numpy(), gt["sim_agent_ids"].numpy()))
        tr, m = gt["tracks"].numpy()[rows], gt["track_masks"].numpy()[rows]
        total = turning = 0
        for a in range(len(rows)):
            if m[a].sum() < 2:
                continue
            xy, h = tr[a, m[a], :2], tr[a, m[a], 6]
            moved = np.linalg.norm(xy[-1] - xy[0]) > 10
            turned = abs(np.angle(np.exp(1j * (h[-1] - h[0])))) > 0.8
            total += 3 if (moved and turned) else int(moved)
            turning += int(moved and turned)
        score.append(((turning, total) if by == "turning" else (total,), sid))
    return [sid for _, sid in sorted(score, reverse=True)[:n]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--scenario", default=None)
    ap.add_argument("--pick", type=int, default=0, help="animate the N busiest val_frozen scenarios")
    ap.add_argument("--by", default="busy", choices=["busy", "turning"], help="ranking used by --pick")
    ap.add_argument("--rollout", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    sids = [args.scenario] if args.scenario else busiest("val_frozen", args.pick, args.by)
    for sid in sids:
        animate(sid, args.runs, args.labels, args.out or ROOT / "figures" / f"rollouts_{sid}.gif", args.rollout)


if __name__ == "__main__":
    main()
