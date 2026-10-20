"""Overfit test for an action head: fit one fixed training batch, check the fit, plot closed-loop rollouts.

Usage: python -m tokens_vs_flows.overfit --head flow [--steps 3000] [--batch-size 4] [--lr 5e-4]
Pass criteria:
  token: eval-mode CE drops by >= 90%. Label smoothing is off for this test: with catk's 0.1 smoothing over 2048
         tokens the CE floor is ~1.09 nats against ~7.6 at init, so a 90% drop would be impossible.
  flow:  primary, teacher-forced 10-step Euler samples land within 0.2 m mean displacement (over the 5 waypoints)
         of the logged 0.5 s action; secondary, eval-mode CFM loss (fixed noise) drops by >= 80%.
  bins:  as token (eval-mode CE drop >= 90%); teacher-forced sample displacement is reported as well.
Also renders closed-loop rollouts of 3 of the batch's scenarios. Writes <run>/overfit.json and
figures/overfit_<head>.png. Run dir defaults to $RUN_DIR.
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch

from tokens_vs_flows import ROOT
from tokens_vs_flows.data import train_loader
from tokens_vs_flows.model import ModelConfig, SimAgent
from tokens_vs_flows.plots import bev_rollouts
from tokens_vs_flows.train import setup_cuda

@torch.no_grad()
def eval_loss(model: SimAgent, batch, seed: int = 123) -> float:
    model.eval()
    torch.manual_seed(seed)
    out = float(model.loss(batch)["loss"])
    model.train()
    return out


@torch.no_grad()
def flow_sample_error(model: SimAgent, batch, n_steps: int = 10, seed: int = 123) -> dict:
    """Teacher-forced: sample each step's action from the flow head given the logged history; compare to the log."""
    model.eval()
    torch.manual_seed(seed)
    tokenized_map, tokenized_agent, seg = model._tokenize(batch)
    pred = model.agent_encoder(tokenized_agent, model.map_encoder(tokenized_map))
    h = pred["next_token_logits"]
    mask = pred["next_token_valid"] & seg["target_valid"][:, 1:-1] & batch["agent"]["train_mask"][:, None]
    agent_type = tokenized_agent["type"].long()[:, None].expand_as(mask)[mask]
    enc = model.agent_encoder
    x = enc.flow_head.sample(h[mask], n_steps=n_steps) if model.cfg.head == "flow" else enc.bins_head.sample(h[mask])
    wp = enc.denormalize(x, agent_type).view(-1, 5, 3)
    gt = seg["target"][:, 1:-1][mask].view(-1, 5, 3)
    disp = (wp[..., :2] - gt[..., :2]).norm(dim=-1)  # [N, 5]
    model.train()
    return {"mean_disp_m": float(disp.mean()), "final_disp_m": float(disp[:, -1].mean()),
            "p95_disp_m": float(disp.mean(1).quantile(0.95)), "n": int(mask.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", choices=["token", "flow", "bins"], required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--flow-width", type=int, default=256)
    ap.add_argument("--token-min-drop", type=float, default=0.9)
    ap.add_argument("--flow-max-disp", type=float, default=0.2)
    ap.add_argument("--flow-min-drop", type=float, default=0.8)
    ap.add_argument("--run-dir", type=Path, default=None)
    args = ap.parse_args()
    run_dir = Path(args.run_dir or os.environ["RUN_DIR"])
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_cuda("fp32")
    torch.manual_seed(args.seed)
    cfg = ModelConfig(head=args.head, label_smoothing=0.0, flow_width=args.flow_width)
    model = SimAgent(cfg).cuda().train()
    batch = next(iter(train_loader(args.batch_size, args.seed, 0, 0, num_workers=2))).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    l0 = eval_loss(model, batch)
    s0 = flow_sample_error(model, batch) if args.head != "token" else None
    t0 = time.time()
    curve = []
    for step in range(1, args.steps + 1):
        loss = model.loss(batch)["loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        opt.step()
        if step % max(1, args.steps // 10) == 0:
            curve.append((step, eval_loss(model, batch)))
            print(f"step {step} train {loss.item():.4f} eval {curve[-1][1]:.4f}", flush=True)
    l1 = eval_loss(model, batch)
    res = {"head": args.head, "steps": args.steps, "batch_size": args.batch_size, "lr": args.lr,
           "n_agents": int(batch["agent"]["num_nodes"]), "loss_init": l0, "loss_final": l1,
           "loss_drop": 1 - l1 / l0, "curve": curve, "seconds": time.time() - t0}
    if args.head == "token":
        res["pass"] = res["loss_drop"] >= args.token_min_drop
    elif args.head == "bins":  # categorical head: the token criterion; sample displacement reported too
        res.update(sample_init=s0, sample_final=flow_sample_error(model, batch))
        res["pass"] = res["loss_drop"] >= args.token_min_drop
    else:
        s1 = flow_sample_error(model, batch)
        res.update(sample_init=s0, sample_final=s1, primary_pass=s1["mean_disp_m"] < args.flow_max_disp,
                   secondary_pass=res["loss_drop"] >= args.flow_min_drop)
        res["pass"] = res["primary_pass"] and res["secondary_pass"]
    model.eval()
    torch.manual_seed(0)
    ro = model.rollout(batch, n_rollouts=4)
    ade = (ro["pos"] - batch["agent"]["position"][:, None, 11:, :2]).norm(dim=-1)
    m = batch["agent"]["valid_mask"][:, None, 11:].expand_as(ade)
    res["closed_loop_ade_m"] = float(ade[m].mean())
    fig = ROOT / "figures" / f"overfit_{args.head}.png"
    bev_rollouts(batch, ro["pos"], fig, title=f"{args.head} head after overfitting one batch ({args.steps} steps): "
                 f"4 closed-loop rollouts (colour) vs log (black)")
    res["figure"] = str(fig.relative_to(ROOT))
    (run_dir / "overfit.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "curve"}), flush=True)


if __name__ == "__main__":
    main()
