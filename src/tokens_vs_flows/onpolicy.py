"""On-policy reward fine-tuning: PPO-clip on the token head vs FPO on the flow head, same reward and budget.

Each iteration, from the current policy (the "old" policy of the iteration):
  1. K closed-loop rollouts on B training scenes; leave-one-out realism reward per rollout (RLFTSim Eq. 2, the
     centred MLOO), advantages normalised by their std over the batch (both arms);
  2. a no-grad pass over the rollouts with the old policy and the frozen behaviour-cloning (BC) reference stores,
     per (agent, 0.5 s step) action a:
       token: log pi(a) over the old policy's top-k support (the sampling distribution), for old and BC;
       flow:  N_mc stored (t, noise) draws and their per-draw losses l = ||eps_hat - eps||^2 (summed over the 15
              dims; eps-form, = t^2 ||v_hat - (x1 - eps)||^2), for old and BC; log pi(a) := -mean_draws l (FPO);
  3. E epochs of minibatch updates on  -min(r A, clip(r, 1 - c, 1 + c) A) + kl_coef * k3,
       r = exp(log pi_theta(a) - log pi_old(a))        (token: exact; flow: FPO ratio on the stored draws)
       k3 = rho - 1 - log rho, rho = pi_BC(a) / pi_theta(a)   (anchor to BC in each arm's own likelihood)
     with dropout off; kl_coef adapts to a common target (Ziegler-style, both arms).
Usage: python -m tokens_vs_flows.onpolicy --init runs/bc_flow_s0 --iters 100 [--scenes 32 --k 4 --epochs 4 ...]
Writes <run>/ckpt/final.pt, <run>/iters.csv (per-iteration statistics) and <run>/config.json.
"""
import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from tokens_vs_flows import CACHE
from tokens_vs_flows.rl import RolloutScenes, collect, mloo_rewards, wosac
from tokens_vs_flows.rollout import load_model
from tokens_vs_flows.train import save, set_seed, setup_cuda
from src.smart.datamodules.target_builder import WaymoTargetBuilderVal


def token_logp(logits, action, support):
    """log pi(action) under the softmax restricted to `support` (top-k indices); -inf if action is outside it."""
    sub = torch.log_softmax(logits.gather(-1, support), dim=-1)
    hit = support == action[:, None]
    return torch.where(hit.any(-1), (sub * hit).sum(-1), torch.full_like(sub[:, 0], -math.inf))


def flow_draw_losses(head, h, x1, t, eps):
    """Per-draw eps-form losses [N, M] for actions x1 [N, D] given h [N, H], draws t [N, M], eps [N, M, D]."""
    n, m, d = eps.shape
    with torch.autocast(device_type=h.device.type, enabled=False):
        x_t = (1 - t)[..., None] * eps + t[..., None] * x1[:, None]
        v = head.velocity(x_t.reshape(n * m, d), t.reshape(-1), h.float().repeat_interleave(m, 0)).view(n, m, d)
        return (t[..., None] ** 2 * (v - (x1[:, None] - eps)) ** 2).sum(-1)


def batches(n_items: int, size: int, gen: torch.Generator):
    perm = torch.randperm(n_items, generator=gen).tolist()
    return [perm[i:i + size] for i in range(0, n_items, size)]


def action_quantities(model, out, store=None, n_mc=8, gen=None):
    """log pi(a) per masked action for one teacher-forced pass `out` (policy_inputs). For the token arm the support
    comes from `store` (old policy) or is created here (top-k of these logits); for flow the draws likewise."""
    m = out["mask"]
    if model.cfg.head == "token":
        logits = out["logits"][m]
        a = out["action"][m]
        support = store["support"] if store is not None else logits.topk(model.cfg.token_topk, dim=-1).indices
        return {"logp": token_logp(logits, a, support), "support": support}
    h, x1 = out["h"][m], out["action"][m].float()
    if store is not None:
        t, eps = store["t"].to(h.device), store["eps"].to(h.device)
    else:
        t = torch.rand(x1.shape[0], n_mc, device=h.device, generator=gen)
        eps = torch.randn(x1.shape[0], n_mc, x1.shape[1], device=h.device, generator=gen)
    loss = flow_draw_losses(model.agent_encoder.flow_head, h, x1, t, eps)
    return {"logp": -loss.mean(-1), "t": t, "eps": eps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", type=Path, required=True, help="run dir of the BC policy (start and KL reference)")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--scenes", type=int, default=32, help="training scenes per iteration")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=16, help="rollout scenes per minibatch")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--clip", type=float, default=0.1)
    ap.add_argument("--n-mc", type=int, default=8)
    ap.add_argument("--kl-target", type=float, default=0.01)
    ap.add_argument("--kl-coef", type=float, default=0.1)
    ap.add_argument("--kl-horizon", type=float, default=5.0)
    ap.add_argument("--adv", choices=["mloo", "zero"], default="mloo", help="zero: control run without reward")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt-every", type=int, default=10)
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--device", default="cuda", help="cpu for tests only")
    args = ap.parse_args()
    dev = args.device
    run_dir = Path(args.run_dir or os.environ["RUN_DIR"])
    (run_dir / "ckpt").mkdir(parents=True, exist_ok=True)
    setup_cuda("fp32")
    set_seed(args.seed)
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    cpu_gen = torch.Generator().manual_seed(args.seed)

    policy = load_model(args.init, device=dev)
    ref = load_model(args.init, device=dev).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=0.0)
    config = wosac.load_config()
    pool = sorted(l.split("\t")[0] for l in (CACHE / "training_gt_index.tsv").read_text().splitlines() if l.strip())
    start = 0
    last = run_dir / "ckpt" / "last.pt"
    if last.exists():  # resume
        ck = torch.load(last, map_location=dev)
        policy.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start, args.kl_coef = ck["state"]["iter"], ck["state"]["kl_coef"]
    (run_dir / "config.json").write_text(json.dumps({**policy.cfg.to_dict(), **{k: str(v) for k, v in vars(args).items()},
                                                     "seed": args.seed}, indent=1) + "\n")
    log = open(run_dir / "iters.csv", "a")
    wr = csv.writer(log)
    if start == 0:
        wr.writerow(["iter", "train_realism_k", "adv_std_raw", "ratio_mean", "ratio_std", "clip_frac", "k3", "kl_coef",
                     "frac_outside_support", "n_actions", "seconds"])
    for it in range(start, args.iters):
        t0 = time.time()
        scenes = sorted(random.Random(1_000_000 + 1000 * args.seed + it).sample(pool, args.scenes))
        policy.eval()
        rollouts = collect(policy, scenes, args.k, batch_size=args.scenes, device=dev)
        rewards = mloo_rewards(rollouts, config, dev)
        r = np.array([rewards[s]["r"] for s in scenes])
        a = r - r.mean(1, keepdims=True)  # centred MLOO (RLFTSim Eq. 2)
        adv_std = float(a.std())
        a = a / (adv_std + 1e-8) if args.adv == "mloo" else np.zeros_like(a)
        adv = {s: a[i].tolist() for i, s in enumerate(scenes)}
        ds = RolloutScenes(rollouts, adv, transform=WaymoTargetBuilderVal())

        # old-policy and reference quantities, stored per rollout scene (item)
        store = {}
        with torch.no_grad():
            for idxs in batches(len(ds), args.minibatch, cpu_gen):
                b = next(iter(DataLoader(ds, batch_size=len(idxs), sampler=idxs))).to(dev)
                out_old = policy.policy_inputs(b)
                q_old = action_quantities(policy, out_old, n_mc=args.n_mc, gen=gen)
                q_ref = action_quantities(ref, ref.policy_inputs(b), store=q_old)
                per_item = (out_old["mask"].sum(-1)).split(torch.bincount(b["agent"]["batch"], minlength=len(idxs)).tolist())
                n_act = [int(x.sum()) for x in per_item]
                for j, i in enumerate(idxs):
                    sl = slice(sum(n_act[:j]), sum(n_act[:j + 1]))
                    store[i] = {k: v[sl].cpu() for k, v in q_old.items()}
                    store[i]["logp_ref"] = q_ref["logp"][sl].cpu()
                    store[i]["adv"] = float(b["weight"][j])
        n_actions = sum(len(v["logp"]) for v in store.values())
        out_frac = float(np.mean([float((~torch.isfinite(v["logp"])).float().mean()) for v in store.values()]))

        stats = {"ratio": [], "clip": [], "k3": []}
        for _ in range(args.epochs):
            for idxs in batches(len(ds), args.minibatch, cpu_gen):
                b = next(iter(DataLoader(ds, batch_size=len(idxs), sampler=idxs))).to(dev)
                st = {k: torch.cat([store[i][k] for i in idxs]).to(dev) for k in store[idxs[0]] if k != "adv"}
                advs = torch.cat([torch.full((len(store[i]["logp"]),), store[i]["adv"]) for i in idxs]).to(dev)
                q = action_quantities(policy, policy.policy_inputs(b), store=st)
                ok = torch.isfinite(st["logp"]) & torch.isfinite(q["logp"])
                log_r = (q["logp"] - st["logp"])[ok]
                ratio = log_r.exp()
                A = advs[ok]
                surr = -torch.min(ratio * A, ratio.clamp(1 - args.clip, 1 + args.clip) * A).mean()
                log_rho = (st["logp_ref"] - q["logp"])[ok]
                k3 = (log_rho.exp() - 1 - log_rho).mean()
                loss = surr + args.kl_coef * k3
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                opt.step()
                stats["ratio"].append(ratio.detach())
                stats["clip"].append(((ratio - 1).abs() > args.clip).float().mean().detach())
                stats["k3"].append(float(k3.detach()))
        # adaptive anchor (Ziegler et al.): move the coefficient toward the common KL target
        k3_now = float(np.mean(stats["k3"][-max(1, len(stats["k3"]) // args.epochs):]))
        args.kl_coef *= 1 + float(np.clip(k3_now / args.kl_target - 1, -0.2, 0.2)) / args.kl_horizon
        ratios = torch.cat(stats["ratio"])
        m_all = float(np.mean([rewards[s]["m_all"] for s in scenes]))
        wr.writerow([it + 1, f"{m_all:.5f}", f"{adv_std:.5f}", f"{ratios.mean():.4f}", f"{ratios.std():.4f}",
                     f"{float(torch.stack(stats['clip']).mean()):.4f}", f"{k3_now:.5f}", f"{args.kl_coef:.4f}",
                     f"{out_frac:.4f}", n_actions, f"{time.time() - t0:.1f}"])
        log.flush()
        if (it + 1) % args.ckpt_every == 0 or it + 1 == args.iters:
            save(last, policy, opt, {"iter": it + 1, "kl_coef": args.kl_coef})
    save(run_dir / "ckpt" / "final.pt", policy, None, {"iter": args.iters})
    print("done", flush=True)


if __name__ == "__main__":
    main()
