"""Bird's-eye-view plots of closed-loop rollouts against the logged trajectories."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402


def bev_rollouts(batch, pos: torch.Tensor, path: Path, n_scenarios: int = 3, n_show: int = 4, title: str = "",
                 radius: float = 70.0) -> None:
    """batch: collated scenarios; pos: [A, R, 80, 2] rollouts for the batch's agents (global frame)."""
    n = min(n_scenarios, int(batch.num_graphs))
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6), squeeze=False)
    agent_b = batch["agent"]["batch"].cpu()
    map_b = batch["pt_token"]["batch"].cpu()
    gt = batch["agent"]["position"][..., :2].cpu()
    valid = batch["agent"]["valid_mask"].cpu()
    pos = pos.cpu()
    for s in range(n):
        ax = axes[0, s]
        a = (agent_b == s).nonzero().squeeze(1)
        av = a[batch["agent"]["role"][a, 0].cpu()]
        center = gt[av[0] if len(av) else a[0], 10]
        mp = batch["map_save"]["traj_pos"][map_b == s].cpu()  # [n_pl, 3, 2]
        for seg in mp:
            ax.plot(seg[:, 0], seg[:, 1], color="0.82", lw=0.6, zorder=0)
        for k, i in enumerate(a.tolist()):
            g = gt[i][valid[i]]
            ax.plot(g[:, 0], g[:, 1], color="0.25", lw=1.0, zorder=2, label="logged" if k == 0 else None)
            for r in range(min(n_show, pos.shape[1])):
                ax.plot(pos[i, r, :, 0], pos[i, r, :, 1], color=f"C{r}", lw=0.8, alpha=0.8, zorder=1,
                        label=f"rollout {r}" if k == 0 else None)
            ax.scatter(gt[i, 10, 0], gt[i, 10, 1], s=8, color="k", zorder=3)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_aspect("equal")
        ax.set_title(f"{batch['scenario_id'][s]}", fontsize=9)
        if s == 0:
            ax.legend(fontsize=7, loc="upper left")
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
