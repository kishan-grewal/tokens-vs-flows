"""Token replay: the logged future after catk's rolling tokenisation, decoded exactly as the token head's rollouts
are (each 0.5 s token's 5 frames from its contour trajectory, in the frame of the previous tokenised pose).

It is the token arm's ceiling: the realism of following the log as closely as the vocabulary allows. Missing logged
steps are parked far outside the scene, as in the log replay. Writes <run>/rollouts in the scoring format (the
deterministic replay repeated 32 times) and <run>/config.json.
Usage: python -m tokens_vs_flows.token_replay --run runs/eval_token_replay [--split val_frozen]
"""
import argparse
import json
import pickle
from pathlib import Path

import torch
from omegaconf import OmegaConf

from tokens_vs_flows import CACHE
from tokens_vs_flows.data import eval_loader, read_split
from src.smart.tokens.token_processor import TokenProcessor
from src.smart.utils import transform_to_global

N_ROLLOUTS = 32


@torch.no_grad()
def decode(ta: dict) -> tuple:
    """Future 10 Hz poses [A, 80, 2], [A, 80] from the rolling-matched tokens of steps 2..17."""
    a = ta["gt_idx"].shape[0]
    idx = ta["gt_idx"][:, 2:18]  # tokens for the motion 10->15, ..., 85->90
    traj = ta["token_traj_all"][torch.arange(a)[:, None], idx]  # [A, 16, 6, 4, 2] start-frame contours
    pos0, head0 = ta["gt_pos"][:, 1:17], ta["gt_heading"][:, 1:17]  # tokenised poses at frames 10, 15, ..., 85
    glob = transform_to_global(traj[:, :, 1:].reshape(a * 16, 20, 2), None, pos0.reshape(-1, 2),
                               head0.reshape(-1))[0].view(a, 16, 5, 4, 2)
    pos = glob.mean(3).reshape(a, 80, 2)
    d = glob[:, :, :, 0] - glob[:, :, :, 3]
    head = torch.atan2(d[..., 1], d[..., 0]).reshape(a, 80)
    return pos, head


def logged_validity(scenario_id: str, agent_id: torch.Tensor) -> torch.Tensor:
    """Raw logged validity of the future frames [A, 80] (the ground-truth pickle the metric uses, no gap filling)."""
    with open(CACHE / "validation_gt" / f"{scenario_id}.pkl", "rb") as f:
        gt = pickle.load(f)
    rows = (agent_id[:, None] == gt["object_ids"].long()[None]).float().argmax(1)
    return gt["track_masks"][rows][:, 11:91]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--split", default="val_frozen")
    args = ap.parse_args()
    out = args.run / "rollouts"
    out.mkdir(parents=True, exist_ok=True)
    s = OmegaConf.create({"num_k": 1, "temp": 1.0})
    tp = TokenProcessor("map_traj_token5.pkl", "agent_vocab_555_s2.pkl", s, s).eval()
    n = 0
    for batch in eval_loader(read_split(args.split), batch_size=16, num_workers=4):
        z = batch["agent"]["position"][:, 10, 2].clone()
        _, ta = tp(batch)
        pos, head = decode(ta)
        states = torch.cat([pos, z[:, None, None].expand(-1, 80, 1), head[..., None]], dim=-1)  # [A, 80, 4]
        for i, sid in enumerate(batch["scenario_id"]):
            m = batch["agent"]["batch"] == i
            st, ids = states[m].clone(), batch["agent"]["id"][m]
            valid = logged_validity(sid, ids)
            far = torch.stack([1e5 + 100.0 * torch.arange(len(ids)), torch.full((len(ids),), 1e5)], -1)
            st[..., :2] = torch.where(valid[..., None], st[..., :2], far[:, None, :])
            torch.save({"agent_id": ids, "states": st[None].repeat(N_ROLLOUTS, 1, 1, 1).float()}, out / f"{sid}.pt")
            n += 1
    (args.run / "config.json").write_text(json.dumps({"head": "token_replay", "seed": ""}) + "\n")
    print(f"token replay: {n} scenarios -> {out}")


if __name__ == "__main__":
    main()
