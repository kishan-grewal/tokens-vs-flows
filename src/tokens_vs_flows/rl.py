"""Offline reward-weighted round (advantage-weighted regression on the policy's own closed-loop rollouts).

One round, for either head:
  1. K closed-loop rollouts of the current policy on N training scenes (the same scenes for every arm in a round);
  2. leave-one-out realism reward per rollout: r_k = M(all K) - M(all K but k), M = WOSAC 2025 meta-metric of the
     scene (fast metric); advantage A_k = r_k - mean_j r_j (group baseline);
  3. weights w_k = min(exp(A_k / beta), w_max);
  4. one epoch of weighted behaviour cloning on the K x N rollouts: each rollout becomes a scene whose future is the
     rollout itself, and every target of that scene is weighted by w_k. For tokens the targets re-tokenise to the
     sampled tokens; for flow they are the sampled waypoints.
Usage: python -m tokens_vs_flows.rl --init runs/bc_flow_s0 --round 1 [--n-scenes 500 --k 4 --beta ... --lr ...]
Writes <run>/ckpt/final.pt (the round's policy), <run>/round.json (reward statistics) and <run>/config.json.
"""
import argparse
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader

from tokens_vs_flows import CACHE, ROOT
from tokens_vs_flows.data import ScenarioDataset
from tokens_vs_flows.rollout import load_model
from tokens_vs_flows.train import lr_factor, save, set_seed, setup_cuda, train_step
from src.smart.datamodules.target_builder import WaymoTargetBuilderTrain

sys.path.insert(0, str(ROOT / "eval"))
import wosac  # noqa: E402  (the frozen metric; also puts TrajTok's fast metric on the path)
from wosac_fast_eval_tool.scenario_gt_converter import gt_scenario_to_device  # noqa: E402

TRAIN_GT = CACHE / "training_gt"
CUR = 10  # current frame


def round_scenes(round_idx: int, n: int) -> list:
    """N training scenes for a round: seeded by the round only, so every arm and seed sees the same scenes."""
    pool = sorted(l.split("\t")[0] for l in (CACHE / "training_gt_index.tsv").read_text().splitlines() if l.strip())
    return sorted(random.Random(10_000 + round_idx).sample(pool, n))


@torch.no_grad()
def collect(model, scene_ids, k: int, batch_size: int = 32, device: str = "cuda") -> dict:
    """{scenario_id: {"agent_id": [A], "states": [K, A, 80, 4]}} of K closed-loop rollouts per scene."""
    ds = ScenarioDataset(CACHE / "training", scenario_ids=scene_ids)
    out = {}
    for batch in DataLoader(ds, batch_size=batch_size, num_workers=8):
        batch = batch.to(device)
        ro = model.rollout(batch, k)
        states = torch.cat([ro["pos"], ro["z"][..., None], ro["head"][..., None]], dim=-1)  # [A, K, 80, 4]
        for s, sid in enumerate(batch["scenario_id"]):
            m = batch["agent"]["batch"] == s
            out[sid] = {"agent_id": batch["agent"]["id"][m].cpu(), "states": states[m].transpose(0, 1).float().cpu()}
    return out


def load_train_gt(sid: str, device: str) -> dict:
    with open(TRAIN_GT / f"{sid}.pkl", "rb") as f:
        return gt_scenario_to_device(pickle.load(f), device)


def mloo_rewards(rollouts: dict, config, device: str) -> dict:
    """Leave-one-out realism per rollout: r_k = M(all K) - M(all but k), per scene."""
    rewards = {}
    for sid, ro in rollouts.items():
        gt = load_train_gt(sid, device)
        k = ro["states"].shape[0]
        m_all = wosac.realism(config, gt, ro, device)["metametric"]
        r = []
        for i in range(k):
            keep = [j for j in range(k) if j != i]
            m_wo = wosac.realism(config, gt, {"agent_id": ro["agent_id"], "states": ro["states"][keep]}, device)
            r.append(m_all - m_wo["metametric"])
        rewards[sid] = {"m_all": m_all, "r": r}
    return rewards


def awr_weights(rewards: dict, beta: float, w_max: float) -> dict:
    """Group-baseline advantages A_k = r_k - mean_j r_j, weights min(exp(A_k / beta), w_max)."""
    out = {}
    for sid, rw in rewards.items():
        r = np.array(rw["r"])
        a = r - r.mean()
        out[sid] = np.minimum(np.exp(a / beta), w_max).tolist()
    return out


class RolloutScenes(Dataset):
    """Each item: a training scene whose future (frames 11..90) is replaced by one closed-loop rollout, with the
    rollout's weight as a graph attribute. Simulated agents exist at every future step."""

    def __init__(self, rollouts: dict, weights: dict) -> None:
        self.items = [(sid, k) for sid in sorted(rollouts) for k in range(rollouts[sid]["states"].shape[0])]
        self.rollouts, self.weights = rollouts, weights
        super().__init__(transform=WaymoTargetBuilderTrain(32))

    def len(self) -> int:
        return len(self.items)

    def get(self, idx: int):
        sid, k = self.items[idx]
        with open(CACHE / "training" / f"{sid}.pkl", "rb") as f:
            data = pickle.load(f)
        a, ro = data["agent"], self.rollouts[sid]
        assert torch.equal(a["id"], ro["agent_id"]), sid
        st = ro["states"][k]  # [A, 80, 4]
        a["position"][:, CUR + 1:, :2] = st[..., :2]
        a["position"][:, CUR + 1:, 2] = st[..., 2]
        a["heading"][:, CUR + 1:] = st[..., 3]
        a["valid_mask"][:, CUR + 1:] = True
        prev = torch.cat([a["position"][:, CUR:CUR + 1, :2], st[:, :-1, :2]], dim=1)
        a["velocity"][:, CUR + 1:] = (st[..., :2] - prev) / 0.1
        data["weight"] = torch.tensor([float(self.weights[sid][k])])
        return data


def calibrate(rewards: dict, args, run_dir: Path, seconds: dict) -> None:
    """Advantage and weight statistics for a range of temperatures (absolute and relative to the advantage std)."""
    r = np.array([v["r"] for v in rewards.values()])
    a = r - r.mean(1, keepdims=True)
    sd = float(a.std())
    rows = []
    for beta in sorted({*(np.array([0.25, 0.5, 1.0, 2.0]) * sd), 0.001, 0.003, 0.01, 0.03}):
        w = np.minimum(np.exp(a / beta), args.w_max)
        rows.append({"beta": float(beta), "beta_over_adv_std": float(beta / sd), "ess_frac": float(w.sum() ** 2 / (w ** 2).sum() / w.size),
                     "w_max_frac": float((w >= args.w_max).mean()), "w_p90": float(np.quantile(w, 0.9))})
    info = {"init": str(args.init), "n_scenes": len(rewards), "k": args.k, "m_all_mean": float(np.mean([v["m_all"] for v in rewards.values()])),
            "reward_mean": float(r.mean()), "reward_std": float(r.std()), "adv_std": sd,
            "adv_abs_p50_p90_p99": [float(np.quantile(np.abs(a), q)) for q in (0.5, 0.9, 0.99)], "betas": rows, "seconds": seconds}
    (run_dir / "calibration.json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps(info, indent=1), flush=True)


def finetune(model, rollouts: dict, weights: dict, args) -> list:
    """One epoch of weighted behaviour cloning over all rollouts (fixed seeded order), cosine lr."""
    ds = RolloutScenes(rollouts, weights)
    g = torch.Generator().manual_seed(args.seed * 100 + args.round)
    loader = DataLoader(ds, batch_size=args.batch_size, sampler=torch.randperm(len(ds), generator=g).tolist(),
                        num_workers=8, drop_last=False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    n_steps = len(loader)
    model.train()
    losses = []
    for i, batch in enumerate(loader):
        for grp in opt.param_groups:
            grp["lr"] = args.lr * lr_factor(i, n_steps, 0, args.lr_min_ratio)
        lv, _, _ = train_step(model, opt, batch, grad_clip=0.5, weighted=True)
        losses.append(lv)
    return losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", type=Path, required=True, help="run dir of the starting policy (ckpt/final.pt)")
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--n-scenes", type=int, default=500)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--beta", type=float, default=0.01)
    ap.add_argument("--w-max", type=float, default=20.0)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--lr-min-ratio", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--calibrate", action="store_true",
                    help="only collect and score (no fine-tune); report advantage and weight statistics per beta")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    run_dir = Path(args.run_dir or os.environ["RUN_DIR"])
    (run_dir / "ckpt").mkdir(parents=True, exist_ok=True)
    setup_cuda("fp32")
    set_seed(args.seed * 1000 + args.round)
    t0 = time.time()
    model = load_model(args.init, device=args.device)
    scenes = round_scenes(args.round, args.n_scenes)
    rollouts = collect(model.eval(), scenes, args.k, device=args.device)
    t_roll = time.time() - t0
    config = wosac.load_config()
    rewards = mloo_rewards(rollouts, config, args.device)
    t_score = time.time() - t0 - t_roll
    if args.calibrate:
        return calibrate(rewards, args, run_dir, {"rollouts": t_roll, "scoring": t_score})
    weights = awr_weights(rewards, args.beta, args.w_max)
    losses = finetune(model, rollouts, weights, args)
    save(run_dir / "ckpt" / "final.pt", model, None, {"round": args.round, "init": str(args.init)})
    r = np.array([v["r"] for v in rewards.values()])
    w = np.array([weights[s] for s in rewards])
    m_all = np.array([v["m_all"] for v in rewards.values()])
    info = {"round": args.round, "init": str(args.init), "n_scenes": len(scenes), "k": args.k, "beta": args.beta,
            "w_max": args.w_max, "lr": args.lr, "train_realism_k": float(m_all.mean()),
            "reward_mean": float(r.mean()), "reward_std": float(r.std()), "adv_std": float((r - r.mean(1, keepdims=True)).std()),
            "weight_mean": float(w.mean()), "weight_max": float(w.max()), "weight_frac_clipped": float((w >= args.w_max).mean()),
            "ess_frac": float(w.sum() ** 2 / (w ** 2).sum() / w.size),
            "finetune_steps": len(losses), "finetune_loss_first_last": [losses[0], losses[-1]] if losses else None,
            "seconds": {"rollouts": t_roll, "scoring": t_score, "total": time.time() - t0}}
    (run_dir / "round.json").write_text(json.dumps(info, indent=1) + "\n")
    cfg = {**model.cfg.to_dict(), **{k: str(v) for k, v in vars(args).items()}, "seed": args.seed}
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=1) + "\n")
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
