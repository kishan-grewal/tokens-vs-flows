"""Timing probe on one GPU: max batch, s/step, the cost of one offline RL round, and the cost of one eval.

Usage: python -m tokens_vs_flows.probe [--run-dir runs/<id>]   (writes <run>/probe.json)
  max_batch  largest batch whose peak memory over 12 random training batches stays under the cap without a
             memory split, for both heads (the arms share one batch size). Rare heavier batches are handled by
             train_step's micro-batch fallback (same gradient).
  s_step     mean s/step over 200 steps at max_batch on random training batches (token head; flow too)
  t_roll     one offline RL round: K=4 rollouts of 500 scenes + leave-one-out scoring (K+1 metric calls per
             scene) + one weighted fine-tuning epoch over the 2000 rollouts at max_batch
  t_eval     1000 scenes x 32 rollouts + scoring
"""
import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from tokens_vs_flows import CACHE, ROOT
from tokens_vs_flows.data import ScenarioDataset, train_loader
from tokens_vs_flows.model import ModelConfig, SimAgent
from tokens_vs_flows.train import setup_cuda, train_step

sys.path.insert(0, str(ROOT / "eval"))
import wosac  # noqa: E402

CANDIDATES = [8, 12, 16, 20, 24, 32]


def random_batches(batch_size: int, n: int, seed: int = 7) -> list:
    loader = train_loader(batch_size, seed=seed, epoch=0, start_sample=0, num_workers=8)
    it = iter(loader)
    out = [next(it) for _ in range(n)]
    del it, loader
    return out


def memory_check(head: str, batches: list, cap_gb: float) -> dict:
    """Peak memory of single-shot steps (no micro-batch split) on the given batches."""
    torch.manual_seed(0)
    model = SimAgent(ModelConfig(head=head)).cuda().train()
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4)
    peaks, n_oom = [], 0
    for b in batches:
        torch.cuda.reset_peak_memory_stats()
        try:
            opt.zero_grad(set_to_none=True)
            model.loss(b.cuda(non_blocking=True))["loss"].backward()
            opt.step()
            torch.cuda.synchronize()
            peaks.append(torch.cuda.max_memory_allocated() / 2**30)
        except torch.cuda.OutOfMemoryError:
            n_oom += 1
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
    del model, opt
    gc.collect()
    torch.cuda.empty_cache()
    return {"max_gb": max(peaks) if peaks else float("inf"), "mean_gb": sum(peaks) / max(len(peaks), 1),
            "n_oom": n_oom, "ok": n_oom == 0 and bool(peaks) and max(peaks) < cap_gb}


def time_steps(head: str, batch_size: int, n: int, warmup: int = 10) -> dict:
    torch.manual_seed(0)
    model = SimAgent(ModelConfig(head=head)).cuda().train()
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4)
    loader = train_loader(batch_size, seed=1, epoch=0, start_sample=0, num_workers=12)
    it = iter(loader)
    t0, splits = None, 0
    for i in range(n + warmup):
        if i == warmup:
            torch.cuda.synchronize()
            t0 = time.time()
        _, _, n_micro = train_step(model, opt, next(it), grad_clip=0.5)
        splits += (n_micro > 1) and i >= warmup
    torch.cuda.synchronize()
    out = {"s_step": (time.time() - t0) / n, "n": n, "split_steps": int(splits),
           "peak_gb": torch.cuda.max_memory_allocated() / 2**30}
    del model, opt, loader, it
    gc.collect()
    torch.cuda.empty_cache()
    return out


@torch.no_grad()
def time_rollouts(head: str, ids, n_rollouts: int, batch_size: int) -> float:
    model = SimAgent(ModelConfig(head=head)).cuda().eval()
    ds = ScenarioDataset(CACHE / "validation", scenario_ids=ids)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=8)
    torch.cuda.synchronize()
    t0 = time.time()
    for b in loader:
        model.rollout(b.cuda(non_blocking=True), n_rollouts)
    torch.cuda.synchronize()
    return (time.time() - t0) / len(ids)


def time_scoring(ids, n_rollouts: int, mloo: bool) -> float:
    """s/scenario of the fast metric on constant-velocity-like rollouts (cost does not depend on the policy)."""
    from baselines import random_agent
    config = wosac.load_config()
    t0 = time.time()
    for sid in ids:
        gt = wosac.load_gt(sid, "cuda")
        ro = random_agent(gt, n_rollouts)
        wosac.realism(config, gt, ro, "cuda")
        if mloo:  # leave-one-out: K extra calls with K-1 rollouts
            for k in range(n_rollouts):
                keep = [j for j in range(n_rollouts) if j != k]
                wosac.realism(config, gt, {"agent_id": ro["agent_id"], "states": ro["states"][keep]}, "cuda")
    torch.cuda.synchronize()
    return (time.time() - t0) / len(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--cap-gb", type=float, default=21.0)
    ap.add_argument("--quick", action="store_true", help="tiny settings to test the probe itself")
    args = ap.parse_args()
    global CANDIDATES
    n_tok, n_flow, n_scenes, n_mem = 200, 100, 128, 12
    if args.quick:
        CANDIDATES, n_tok, n_flow, n_scenes, n_mem = [2, 4], 5, 5, 8, 2
    run_dir = Path(args.run_dir or os.environ["RUN_DIR"])
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_cuda("fp32")
    res = {"gpu": torch.cuda.get_device_name(0), "cap_gb": args.cap_gb}
    t_all = time.time()

    res["memory"] = {}
    max_batch = None
    for b in CANDIDATES:
        batches = random_batches(b, n_mem)
        ok_all = True
        for head in ("token", "flow"):
            m = memory_check(head, batches, args.cap_gb)
            res["memory"][f"{head}_b{b}"] = m
            print(f"batch {b} {head}: max {m['max_gb']:.1f} GB mean {m['mean_gb']:.1f} GB oom {m['n_oom']}", flush=True)
            ok_all &= m["ok"]
        del batches
        if not ok_all:
            break
        max_batch = b
    res["max_batch"] = max_batch
    print(f"max_batch {max_batch}", flush=True)
    dump = lambda: (run_dir / "probe.json").write_text(json.dumps(res, indent=1) + "\n")  # noqa: E731
    dump()

    res["token_fp32"] = time_steps("token", max_batch, n_tok)
    res["flow_fp32"] = time_steps("flow", max_batch, n_flow)
    res["s_step"] = res["token_fp32"]["s_step"]
    print(json.dumps({k: res[k] for k in ("token_fp32", "flow_fp32")}), flush=True)
    dump()

    pool = wosac.read_split("val_frozen")[:n_scenes]
    res["rollout_batch"] = 64
    res["rollout_s_per_scene_k4"] = {h: time_rollouts(h, pool, 4, 64) for h in ("token", "flow")}
    res["rollout_s_per_scene_k32"] = {h: time_rollouts(h, pool, 32, 64) for h in ("token", "flow")}
    res["score_s_per_scene_k32"] = time_scoring(pool[:32], 32, mloo=False)
    res["score_s_per_scene_mloo_k4"] = time_scoring(pool[:32], 4, mloo=True)
    print(json.dumps({k: res[k] for k in res if k.startswith(("rollout", "score"))}), flush=True)

    roll4 = max(res["rollout_s_per_scene_k4"].values())
    roll32 = max(res["rollout_s_per_scene_k32"].values())
    s_step_ft = max(res["token_fp32"]["s_step"], res["flow_fp32"]["s_step"])
    ft_steps = (500 * 4) / max_batch
    res["t_roll_parts_h"] = {"rollouts": 500 * roll4 / 3600, "mloo_scoring": 500 * res["score_s_per_scene_mloo_k4"] / 3600,
                             "finetune_epoch": ft_steps * s_step_ft / 3600}
    res["t_roll_hours"] = sum(res["t_roll_parts_h"].values())
    res["t_eval_parts_min"] = {"rollouts": 1000 * roll32 / 60, "scoring": 1000 * res["score_s_per_scene_k32"] / 60}
    res["t_eval_minutes"] = sum(res["t_eval_parts_min"].values())
    res["seconds"] = time.time() - t_all
    dump()
    print(json.dumps({k: res[k] for k in ("max_batch", "s_step", "t_roll_hours", "t_eval_minutes", "seconds")}),
          flush=True)


if __name__ == "__main__":
    main()
