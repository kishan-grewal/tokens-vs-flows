"""Closed-loop rollouts of a trained SimAgent on a fixed split -> <run>/rollouts/<scenario_id>.pt

Usage: python -m tokens_vs_flows.rollout --run runs/<id> [--split val_frozen] [--n-rollouts 32] [--batch-size 64]
           [--ckpt final] [--out-name rollouts] [--token-topk 48] [--flow-noise-scale 1.0] [--flow-steps 10]
Each file holds {"agent_id": int64 [A], "states": float32 [R, A, 80, 4] = (x, y, z, heading)} in the global frame,
the format eval/score.py reads. Sampling knobs default to the checkpoint's config.
"""
import argparse
import json
import time
from pathlib import Path

import torch

from tokens_vs_flows.data import eval_loader, read_split
from tokens_vs_flows.model import ModelConfig, SimAgent
from tokens_vs_flows.train import setup_cuda


def load_model(run: Path, ckpt: str = "final", device: str = "cuda", **overrides) -> SimAgent:
    ck = torch.load(run / "ckpt" / f"{ckpt}.pt", map_location=device)
    cfg = ModelConfig(**{**ck["cfg"], **{k: v for k, v in overrides.items() if v is not None}})
    model = SimAgent(cfg).to(device)
    model.load_state_dict(ck["model"])
    return model.eval()


@torch.no_grad()
def write_rollouts(model: SimAgent, scenario_ids, out_dir: Path, n_rollouts: int, batch_size: int,
                   num_workers: int = 8, seed: int = 0, log_every: int = 10) -> float:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    t0 = time.time()
    loader = eval_loader(scenario_ids, batch_size, num_workers)
    done = 0
    for i, batch in enumerate(loader):
        batch = batch.cuda(non_blocking=True)
        ro = model.rollout(batch, n_rollouts)
        states = torch.cat([ro["pos"], ro["z"][..., None], ro["head"][..., None]], dim=-1)  # [A, R, 80, 4]
        agent_batch = batch["agent"]["batch"]
        for s, sid in enumerate(batch["scenario_id"]):
            m = agent_batch == s
            torch.save({"agent_id": batch["agent"]["id"][m].cpu(),
                        "states": states[m].transpose(0, 1).contiguous().float().cpu()}, out_dir / f"{sid}.pt")
        done += len(batch["scenario_id"])
        if log_every and (i + 1) % log_every == 0:
            print(f"rolled out {done}/{len(scenario_ids)} ({(time.time() - t0) / done:.2f} s/scenario)", flush=True)
    return time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--split", default="val_frozen")
    ap.add_argument("--ckpt", default="final")
    ap.add_argument("--n-rollouts", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--max-scenarios", type=int, default=0)
    ap.add_argument("--out-name", default="rollouts")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--token-topk", type=int, default=None)
    ap.add_argument("--flow-noise-scale", type=float, default=None)
    ap.add_argument("--flow-steps", type=int, default=None)
    args = ap.parse_args()
    setup_cuda("fp32")
    model = load_model(args.run, args.ckpt, token_topk=args.token_topk, flow_noise_scale=args.flow_noise_scale,
                       flow_steps=args.flow_steps)
    ids = read_split(args.split)
    if args.max_scenarios:
        ids = ids[:args.max_scenarios]
    secs = write_rollouts(model, ids, args.run / args.out_name, args.n_rollouts, args.batch_size, args.num_workers,
                          args.seed)
    info = {"split": args.split, "n_scenarios": len(ids), "n_rollouts": args.n_rollouts, "seconds": secs,
            "sampling": {"token_topk": model.cfg.token_topk, "token_temp": model.cfg.token_temp,
                         "flow_noise_scale": model.cfg.flow_noise_scale, "flow_steps": model.cfg.flow_steps}}
    (args.run / f"{args.out_name}.json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
