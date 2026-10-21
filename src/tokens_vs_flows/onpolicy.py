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
     Optional for flow (--v-anchor w > 0): + w * ||v_theta - v_BC||^2 at the stored draws (summed over dims, mean
     over draws), an anchor on the velocity field itself. Unlike the eps-form likelihood, which weights each draw by
     t^2, it is not down-weighted near t = 0, where sampling starts. With --v-anchor-form x1 it anchors the predicted
     clean sample x1_hat = x_t + (1 - t) v instead, i.e. weights each draw by (1 - t)^2: it holds the field where the
     eps-form likelihood is blind and leaves it free where the likelihood (and so the ratio) acts.
Usage: python -m tokens_vs_flows.onpolicy --init runs/bc_flow_s0 --iters 100 [--scenes 32 --k 4 --epochs 4 ...]
Writes <run>/ckpt/final.pt, <run>/iters.csv (per-iteration statistics) and <run>/config.json.
"""
import argparse
import csv
import gc
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


def flow_draw_losses(head, h, x1, t, eps, return_v: bool = False):
    """Per-draw eps-form losses [N, M] for actions x1 [N, D] given h [N, H], draws t [N, M], eps [N, M, D]
    (and the velocities [N, M, D] at the draws if return_v)."""
    n, m, d = eps.shape
    with torch.autocast(device_type=h.device.type, enabled=False):
        x_t = (1 - t)[..., None] * eps + t[..., None] * x1[:, None]
        v = head.velocity(x_t.reshape(n * m, d), t.reshape(-1), h.float().repeat_interleave(m, 0)).view(n, m, d)
        loss = (t[..., None] ** 2 * (v - (x1[:, None] - eps)) ** 2).sum(-1)
        return (loss, v) if return_v else loss


def velocity_anchor(v, v_ref, t=None):
    """Per-action anchor [N] on the velocity field: ||v - v_ref||^2 summed over dims, mean over the draws; with the
    draw times t [N, n_mc], each draw is weighted by (1 - t)^2 (= ||x1_hat - x1_hat_ref||^2, the x1-prediction form)."""
    d = (v - v_ref).pow(2).sum(-1)
    return (d if t is None else d * (1 - t).pow(2)).mean(-1)


def batches(n_items: int, size: int, gen: torch.Generator):
    perm = torch.randperm(n_items, generator=gen).tolist()
    return [perm[i:i + size] for i in range(0, n_items, size)]


def action_quantities(model, out, store=None, n_mc=8, gen=None, with_v=False):
    """log pi(a) per masked action for one teacher-forced pass `out` (policy_inputs). For the token arm the support
    comes from `store` (old policy) or is created here (top-k of these logits); for flow the draws likewise, and
    with_v adds the velocities at the draws."""
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
    loss, v = flow_draw_losses(model.agent_encoder.flow_head, h, x1, t, eps, return_v=True)
    return {"logp": -loss.mean(-1), "t": t, "eps": eps, **({"v": v} if with_v else {})}


def parts(idxs: list, n: int) -> list:
    return [idxs[j::n] for j in range(n) if idxs[j::n]]


def load(ds, idxs: list, dev):
    return next(iter(DataLoader(ds, batch_size=len(idxs), sampler=idxs))).to(dev)


def retry_in_parts(fn, idxs: list, on_retry=None):
    """fn(parts(idxs, n)) for n = 1, 2, 4, ...: scenes that do not fit in GPU memory together are processed in parts
    (as in train.train_step). The retry runs after the exception handler has exited, so the failed attempt's frame,
    with its activations and graph, is freed before the smaller attempt starts. Returns (fn's result, n)."""
    n = 1
    while True:
        try:
            return fn(parts(idxs, n)), n
        except torch.cuda.OutOfMemoryError:
            if 2 * n > len(idxs):
                raise
        if on_retry is not None:
            on_retry()
        gc.collect()
        torch.cuda.empty_cache()
        n *= 2


def store_pass(policy, ref, ds, idxs: list, args, gen) -> dict:
    """Old-policy and reference quantities for the rollout scenes `idxs`, per scene."""
    def run(chunks):
        out = {}
        with torch.no_grad():
            for part in chunks:
                b = load(ds, part, args.device)
                out_old = policy.policy_inputs(b)
                q_old = action_quantities(policy, out_old, n_mc=args.n_mc, gen=gen)
                q_ref = action_quantities(ref, ref.policy_inputs(b), store=q_old, with_v=args.v_anchor > 0)
                per_item = out_old["mask"].sum(-1).split(torch.bincount(b["agent"]["batch"], minlength=len(part)).tolist())
                n_act = [int(x.sum()) for x in per_item]
                for j, i in enumerate(part):
                    sl = slice(sum(n_act[:j]), sum(n_act[:j + 1]))
                    out[i] = {k: v[sl].cpu() for k, v in q_old.items()}
                    out[i]["logp_ref"] = q_ref["logp"][sl].cpu()
                    if "v" in q_ref:
                        out[i]["v_ref"] = q_ref["v"][sl].cpu()
                    out[i]["adv"] = float(b["weight"][j])
        return out
    return retry_in_parts(run, idxs)[0]


def update_step(policy, opt, ds, idxs: list, store: dict, args):
    """One optimiser step on the rollout scenes `idxs`: -min(r A, clip(r) A) + kl_coef * k3 (+ v_anchor * velocity
    anchor), averaged over the minibatch's valid actions. If the scenes do not fit in GPU memory together they are
    processed in 2, 4, ... parts with gradient accumulation (per-part sums over the minibatch's action count: the same
    gradient). Returns (ratios, mean k3, clip fraction, n parts, mean velocity anchor)."""
    dev = args.device
    n_ok = sum(int(torch.isfinite(store[i]["logp"]).sum()) for i in idxs)  # the support is fixed: known in advance

    def run(chunks):
        ratios, k3_sum, va_sum = [], 0.0, 0.0
        for part in chunks:
            b = load(ds, part, dev)
            st = {k: torch.cat([store[i][k] for i in part]).to(dev) for k in store[part[0]] if k != "adv"}
            advs = torch.cat([torch.full((len(store[i]["logp"]),), store[i]["adv"]) for i in part]).to(dev)
            q = action_quantities(policy, policy.policy_inputs(b), store=st, with_v="v_ref" in st)
            ok = torch.isfinite(st["logp"]) & torch.isfinite(q["logp"])
            ratio = (q["logp"] - st["logp"])[ok].exp()
            A = advs[ok]
            surr = -torch.min(ratio * A, ratio.clamp(1 - args.clip, 1 + args.clip) * A).sum()
            log_rho = (st["logp_ref"] - q["logp"])[ok]
            k3 = (log_rho.exp() - 1 - log_rho).sum()
            va = (velocity_anchor(q["v"], st["v_ref"], q["t"] if args.v_anchor_form == "x1" else None)[ok].sum()
                  if "v_ref" in st else torch.zeros((), device=dev))
            ((surr + args.kl_coef * k3 + args.v_anchor * va) / max(n_ok, 1)).backward()
            ratios.append(ratio.detach())
            k3_sum += float(k3.detach())
            va_sum += float(va.detach())
        return torch.cat(ratios), k3_sum, va_sum

    opt.zero_grad(set_to_none=True)
    (r, k3_sum, va_sum), n = retry_in_parts(run, idxs, on_retry=lambda: opt.zero_grad(set_to_none=True))
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    opt.step()
    return r, k3_sum / max(n_ok, 1), float(((r - 1).abs() > args.clip).float().mean()), n, va_sum / max(n_ok, 1)


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
    ap.add_argument("--adv", choices=["mloo", "positive", "shuffled", "zero"], default="mloo",
                    help="positive: max(A, 0) (only better-than-average rollouts are reinforced); shuffled: control "
                         "without reward signal (advantages permuted across the iteration's rollouts, same statistics); "
                         "zero: no update at all (the gradient is zero at the reference)")
    ap.add_argument("--v-anchor", type=float, default=0.0,
                    help="flow only: weight of the velocity-field anchor ||v_theta - v_BC||^2 at the stored draws")
    ap.add_argument("--v-anchor-form", default="velocity", choices=["velocity", "x1"],
                    help="velocity: every draw weighted equally; x1: weighted by (1 - t)^2 (anchors the predicted "
                         "clean sample, i.e. the field near the noise end that the eps-form likelihood ignores)")
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
    if args.v_anchor > 0 and policy.cfg.head != "flow":
        raise SystemExit("--v-anchor applies to the flow head only")
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
                     "frac_outside_support", "n_actions", "seconds", "split_steps", "v_anchor"])
    for it in range(start, args.iters):
        t0 = time.time()
        scenes = sorted(random.Random(1_000_000 + 1000 * args.seed + it).sample(pool, args.scenes))
        policy.eval()
        rollouts = collect(policy, scenes, args.k, batch_size=args.scenes, device=dev)
        rewards = mloo_rewards(rollouts, config, dev)
        r = np.array([rewards[s]["r"] for s in scenes])
        a = r - r.mean(1, keepdims=True)  # centred MLOO (RLFTSim Eq. 2)
        adv_std = float(a.std())
        a = a / (adv_std + 1e-8)
        if args.adv == "positive":
            a = np.maximum(a, 0.0)
        elif args.adv == "shuffled":
            a = np.random.default_rng(1_000_000 + 1000 * args.seed + it).permutation(a.ravel()).reshape(a.shape)
        elif args.adv == "zero":
            a = np.zeros_like(a)
        adv = {s: a[i].tolist() for i, s in enumerate(scenes)}
        ds = RolloutScenes(rollouts, adv, transform=WaymoTargetBuilderVal())

        # old-policy and reference quantities, stored per rollout scene (item)
        store = {}
        for idxs in batches(len(ds), args.minibatch, cpu_gen):
            store.update(store_pass(policy, ref, ds, idxs, args, gen))
        n_actions = sum(len(v["logp"]) for v in store.values())
        out_frac = float(np.mean([float((~torch.isfinite(v["logp"])).float().mean()) for v in store.values()]))

        stats = {"ratio": [], "clip": [], "k3": [], "parts": [], "va": []}
        for _ in range(args.epochs):
            for idxs in batches(len(ds), args.minibatch, cpu_gen):
                ratio, k3, clip_frac, n_parts, va = update_step(policy, opt, ds, idxs, store, args)
                stats["ratio"].append(ratio)
                stats["clip"].append(clip_frac)
                stats["k3"].append(k3)
                stats["parts"].append(n_parts)
                stats["va"].append(va)
        # adaptive anchor (Ziegler et al.): move the coefficient toward the common KL target
        k3_now = float(np.mean(stats["k3"][-max(1, len(stats["k3"]) // args.epochs):]))
        args.kl_coef *= 1 + float(np.clip(k3_now / args.kl_target - 1, -0.2, 0.2)) / args.kl_horizon
        ratios = torch.cat(stats["ratio"])
        m_all = float(np.mean([rewards[s]["m_all"] for s in scenes]))
        wr.writerow([it + 1, f"{m_all:.5f}", f"{adv_std:.5f}", f"{ratios.mean():.4f}", f"{ratios.std():.4f}",
                     f"{float(np.mean(stats['clip'])):.4f}", f"{k3_now:.5f}", f"{args.kl_coef:.4f}",
                     f"{out_frac:.4f}", n_actions, f"{time.time() - t0:.1f}", sum(n > 1 for n in stats["parts"]),
                     f"{float(np.mean(stats['va'][-max(1, len(stats['va']) // args.epochs):])):.5f}"])
        log.flush()
        if (it + 1) % args.ckpt_every == 0 or it + 1 == args.iters:
            save(last, policy, opt, {"iter": it + 1, "kl_coef": args.kl_coef})
    save(run_dir / "ckpt" / "final.pt", policy, None, {"iter": args.iters})
    print("done", flush=True)


if __name__ == "__main__":
    main()
