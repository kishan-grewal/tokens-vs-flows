"""How straight a flow head's learned flow is, and how far few-step Euler samples land from a fine solve.

Contexts come from the model's own closed-loop rollouts on the first n validation scenarios (every agent at every
2 Hz step, 2 rollouts). For a random subset of them, the same initial noise x0 is integrated with k Euler steps and
with 200 steps (the reference x_ref). Per model:
  straightness  mean over the reference path of ||v(x_t, t) - (x_ref - x0)||^2 / D (Liu et al.'s straightness of a
                rectified flow: 0 when the flow is straight and one Euler step is exact), in normalised action units;
                also over t < 0.2 and t > 0.8,
  err           RMS difference of the k-step sample from the reference over the 15 normalised action dims,
  end_err_m     mean distance (m) between the k-step and the reference end waypoint (0.5 s ahead), denormalised,
  accel_err     RMS difference (m/s^2) of the accelerations inside the action (second differences of the 5
                waypoints at 10 Hz, from the agent's current position), k-step vs reference; accel_rms is the RMS of
                those accelerations in the reference samples (how jerky the flow's own actions are).
Writes runs/<id>/flow_straightness.json.
Usage: python -m tokens_vs_flows.flow_straightness --run runs/<id> [runs/<id> ...] [--n-scenes 8] [--device cpu]
"""
import argparse
import json
from pathlib import Path

import torch

from tokens_vs_flows.data import eval_loader, read_split
from tokens_vs_flows.model import FlowAgentDecoder
from tokens_vs_flows.rollout import load_model

STEPS = (1, 2, 5, 10, 20, 50)
N_REF = 200


def in_action_accel(action: torch.Tensor) -> torch.Tensor:
    """Accelerations [N, 4, 2] (m/s^2) from the 5 denormalised waypoints [N, 15] and the agent's position (origin)."""
    pos = torch.cat([action.new_zeros(action.shape[0], 1, 2), action.view(-1, 5, 3)[..., :2]], dim=1)
    vel = (pos[:, 1:] - pos[:, :-1]) / 0.1
    return (vel[:, 1:] - vel[:, :-1]) / 0.1


@torch.no_grad()
def contexts(model, n_scenes: int, device: str):
    """Decoder states h [N, H] and agent types [N] at every flow sampling call in 2 rollouts of each scene."""
    dec = next(m for m in model.modules() if isinstance(m, FlowAgentDecoder))
    hs, types, orig = [], [], dec.sample_action

    def record(h, agent_type):
        hs.append(h.float().cpu())
        types.append(agent_type.cpu())
        return orig(h, agent_type)

    dec.sample_action = record
    torch.manual_seed(0)
    try:
        for batch in eval_loader(read_split("val_frozen")[:n_scenes], 4, 0):
            model.rollout(batch.to(device), 2)
    finally:
        dec.sample_action = orig
    return dec, torch.cat(hs), torch.cat(types)


@torch.no_grad()
def measure(run: Path, n_scenes: int, n_ctx: int, device: str) -> dict:
    model = load_model(run, device=device)
    dec, h_all, type_all = contexts(model, n_scenes, device)
    head = dec.flow_head.cpu()
    g = torch.Generator().manual_seed(1)
    idx = torch.randperm(h_all.shape[0], generator=g)[:n_ctx]
    h, types = h_all[idx], type_all[idx]
    x0 = torch.randn(h.shape[0], head.action_dim, generator=g)
    x, vs, dt = x0.clone(), [], 1.0 / N_REF
    for i in range(N_REF):
        v = head.velocity(x, torch.full((h.shape[0],), i * dt), h)
        vs.append(v)
        x = x + dt * v
    x_ref = x
    dev = torch.stack([(v - (x_ref - x0)).pow(2).mean(-1).mean() for v in vs])  # [N_REF]
    mean_std = dec.act_mean.cpu()[types], dec.act_std.cpu()[types]
    act_ref = x_ref * mean_std[1] + mean_std[0]
    acc_ref = in_action_accel(act_ref)
    out = {"run": str(run), "n_contexts": int(h.shape[0]), "n_scenes": n_scenes,
           "straightness": float(dev.mean()), "straightness_t_lt_0.2": float(dev[:N_REF // 5].mean()),
           "straightness_t_gt_0.8": float(dev[-N_REF // 5:].mean()),
           "accel_rms": float(acc_ref.pow(2).sum(-1).mean().sqrt()), "err": {}, "end_err_m": {}, "accel_err": {}}
    for k in STEPS:
        xk = head.sample(h, n_steps=k, x0=x0)
        out["err"][k] = float((xk - x_ref).pow(2).mean(-1).sqrt().mean())
        act_k = xk * mean_std[1] + mean_std[0]
        out["end_err_m"][k] = float((act_k[:, 12:14] - act_ref[:, 12:14]).norm(dim=-1).mean())
        out["accel_err"][k] = float((in_action_accel(act_k) - acc_ref).pow(2).sum(-1).mean().sqrt())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, nargs="+", required=True)
    ap.add_argument("--n-scenes", type=int, default=8)
    ap.add_argument("--n-contexts", type=int, default=4000)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    for run in args.run:
        r = measure(run, args.n_scenes, args.n_contexts, args.device)
        (run / "flow_straightness.json").write_text(json.dumps(r, indent=1) + "\n")
        print(f"{run.name:28s} straightness {r['straightness']:.4f} (t<0.2 {r['straightness_t_lt_0.2']:.4f}, "
              f"t>0.8 {r['straightness_t_gt_0.8']:.4f}) | accel rms {r['accel_rms']:.2f} | vs {N_REF} steps: end (m) "
              + " ".join(f"{k}:{e:.3f}" for k, e in r["end_err_m"].items()) + " | accel (m/s^2) "
              + " ".join(f"{k}:{e:.2f}" for k, e in r["accel_err"].items()), flush=True)


if __name__ == "__main__":
    main()
