"""Behaviour-cloning training of a SimAgent (token or flow head): step-based, resumable.

Usage: python -m tokens_vs_flows.train --head flow --seed 0 --steps 20000 --batch-size 16 [--run-dir runs/x]
The run dir defaults to $RUN_DIR. Checkpoints to <run>/ckpt/last.pt every 30 min and on
SIGTERM; re-running the same command resumes from it. Writes config.json, loss.csv, summary.txt, ckpt/final.pt.
"""
import argparse
import json
import math
import os
import random
import signal
import time
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

from tokens_vs_flows.data import train_loader
from tokens_vs_flows.model import ModelConfig, SimAgent, count_params, head_params

_STOP = False


def _on_sigterm(signum, frame):
    global _STOP
    _STOP = True


def lr_factor(step: int, total: int, warmup: int, min_ratio: float) -> float:
    """Linear warmup, then cosine decay to min_ratio (catk's schedule, per optimiser step)."""
    if step < warmup:
        return min_ratio + (1 - min_ratio) * step / warmup
    p = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_ratio + 0.5 * (1 - min_ratio) * (1 + math.cos(math.pi * p))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_cuda(precision: str) -> torch.dtype:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    return {"fp32": torch.float32, "bf16": torch.bfloat16}[precision]


def add_model_args(ap: argparse.ArgumentParser) -> None:
    for f in fields(ModelConfig):
        if f.name != "head":
            ap.add_argument(f"--{f.name.replace('_', '-')}", type=type(f.default), default=f.default)


def model_config(args, head: str) -> ModelConfig:
    return ModelConfig(head=head, **{f.name: getattr(args, f.name) for f in fields(ModelConfig) if f.name != "head"})


def split_batch(batch, n: int) -> list:
    """Split a collated batch of scenarios into n smaller collated batches (same scenarios)."""
    from torch_geometric.data import Batch
    data = batch.to_data_list()
    return [Batch.from_data_list(data[i::n]) for i in range(n) if data[i::n]]


def train_step(model, opt, batch, grad_clip: float, amp_dtype=torch.float32):
    """One optimiser step on a CPU batch. If the batch does not fit in GPU memory it is processed in 2, 4, ...
    micro-batches with gradient accumulation; per-target losses are summed and divided by the batch's total
    target count, so the gradient is that of the full batch. Returns (mean loss, grad norm, n micro-batches)."""
    n = 1
    while True:
        try:
            opt.zero_grad(set_to_none=True)
            tot, cnt = 0.0, 0
            for part in (split_batch(batch, n) if n > 1 else [batch]):
                with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype != torch.float32):
                    out = model.loss(part.cuda(non_blocking=True))
                out["sum"].backward()
                tot += float(out["sum"].detach())
                cnt += int(out["count"])
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.div_(max(cnt, 1))
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            return tot / max(cnt, 1), float(gnorm), n
        except torch.cuda.OutOfMemoryError:
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            n *= 2
            if n > batch.num_graphs:
                raise


def save(path: Path, model, opt, state: dict) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict() if opt else None, "state": state,
                "cfg": model.cfg.to_dict(), "rng": {"torch": torch.get_rng_state(),
                                                    "cuda": torch.cuda.get_rng_state_all()}}, tmp)
    os.replace(tmp, path)


def gpu_util(run_dir: Path) -> str:
    f = run_dir / "gpu.csv"
    if not f.exists():
        return "n/a"
    vals = []
    for line in f.read_text().splitlines()[1:]:
        try:
            vals.append(float(line.split(",")[1].strip().split()[0]))
        except (IndexError, ValueError):
            pass
    return f"{np.mean(vals):.0f}%" if vals else "n/a"


def write_summary(run_dir: Path, args, cfg, curve, state, t_start, n_params, n_head) -> None:
    lines = [f"head {cfg.head}  seed {args.seed}  steps {state['step']}/{args.steps}  batch {args.batch_size}"
             f"  lr {args.lr}  precision {args.precision}",
             f"params total {n_params}  head {n_head}",
             f"wall {(time.time() - t_start) / 3600:.2f} h this session  steps/s {state.get('steps_per_s', 0):.2f}"
             f"  peak mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GB  mean GPU util {gpu_util(run_dir)}"
             f"  steps split for memory {state.get('split_steps', 0)}",
             "loss at each 10% of steps (step: loss, mean of last 50):"]
    marks = sorted({max(1, round(args.steps * k / 10)) for k in range(1, 11)})
    by_step = dict(curve)
    for m in marks:
        if m in by_step:
            lines.append(f"  {m}: {by_step[m]:.4f}")
    if curve:
        lines.append(f"  final ({curve[-1][0]}): {curve[-1][1]:.4f}")
    (run_dir / "summary.txt").write_text("\n".join(lines[:20]) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", choices=["token", "flow"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--lr-min-ratio", type=float, default=0.01)
    ap.add_argument("--warmup-steps", type=int, default=0)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--grad-clip", type=float, default=0.5)
    ap.add_argument("--precision", choices=["fp32"], default="fp32",
                    help="fp32 storage with TF32 matmuls; catk's embedding index_put fails under bf16 autocast")
    ap.add_argument("--num-workers", type=int, default=12)
    ap.add_argument("--ckpt-minutes", type=float, default=30.0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--run-dir", type=Path, default=None)
    add_model_args(ap)
    args = ap.parse_args()
    run_dir = Path(args.run_dir or os.environ["RUN_DIR"])
    (run_dir / "ckpt").mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, _on_sigterm)
    amp_dtype = setup_cuda(args.precision)
    set_seed(args.seed)

    cfg = model_config(args, args.head)
    model = SimAgent(cfg).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    state = {"step": 0, "epoch": 0, "sample": 0}
    curve = []
    last = run_dir / "ckpt" / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location="cuda")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        state = ck["state"]
        curve = state.get("curve", [])
        torch.set_rng_state(ck["rng"]["torch"])
        torch.cuda.set_rng_state_all(ck["rng"]["cuda"])
        print(f"resumed from step {state['step']}", flush=True)
    n_params, n_head = count_params(model), head_params(model)
    (run_dir / "config.json").write_text(json.dumps({**vars(args), "run_dir": str(run_dir), **cfg.to_dict(),
                                                     "n_params": n_params, "n_head_params": n_head},
                                                    default=str, indent=1) + "\n")
    print(f"params {n_params} (head {n_head})", flush=True)

    model.train()
    loss_log = open(run_dir / "loss.csv", "a")
    t_start = t_ckpt = t_log = time.time()
    recent = []
    step_at_log = state["step"]
    while state["step"] < args.steps and not _STOP:
        loader = train_loader(args.batch_size, args.seed, state["epoch"], state["sample"], args.num_workers)
        for batch in loader:
            for g in opt.param_groups:
                g["lr"] = args.lr * lr_factor(state["step"], args.steps, args.warmup_steps, args.lr_min_ratio)
            lv, gnorm, n_micro = train_step(model, opt, batch, args.grad_clip, amp_dtype)
            state["step"] += 1
            state["sample"] += args.batch_size
            state["split_steps"] = state.get("split_steps", 0) + (n_micro > 1)
            if not math.isfinite(lv):
                raise FloatingPointError(f"non-finite loss at step {state['step']}")
            recent = (recent + [lv])[-50:]
            if state["step"] % args.log_every == 0 or state["step"] == args.steps:
                now = time.time()
                state["steps_per_s"] = (state["step"] - step_at_log) / (now - t_log)
                t_log, step_at_log = now, state["step"]
                loss_log.write(f"{state['step']},{np.mean(recent):.5f},{float(gnorm):.4f},"
                               f"{opt.param_groups[0]['lr']:.3e},{now - t_start:.1f}\n")
                loss_log.flush()
                print(f"step {state['step']} loss {np.mean(recent):.4f} gnorm {float(gnorm):.3f} "
                      f"{state['steps_per_s']:.2f} it/s split_steps {state['split_steps']}", flush=True)
            marks = {max(1, round(args.steps * k / 10)) for k in range(1, 11)}
            if state["step"] in marks:
                curve.append((state["step"], float(np.mean(recent))))
            if state["step"] >= args.steps or _STOP or time.time() - t_ckpt > 60 * args.ckpt_minutes:
                state["curve"] = curve
                save(last, model, opt, state)
                t_ckpt = time.time()
            if state["step"] >= args.steps or _STOP:
                break
        else:
            state["epoch"] += 1
            state["sample"] = 0
    state["curve"] = curve
    save(last, model, opt, state)
    write_summary(run_dir, args, cfg, curve, state, t_start, n_params, n_head)
    if _STOP and state["step"] < args.steps:
        print(f"stopped by SIGTERM at step {state['step']}; checkpoint saved", flush=True)
        raise SystemExit(143)
    save(run_dir / "ckpt" / "final.pt", model, None, state)
    print("done", flush=True)


if __name__ == "__main__":
    main()
