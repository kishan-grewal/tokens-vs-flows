"""Tokenisation error of the token arm's teacher-forced inputs on training data, per dimension and agent type.

At every 2 Hz input step (frames 5, 10, ..., 90) with a valid token, compares catk's rolling-tokenised pose with the
raw logged pose: position error in the raw pose's frame (longitudinal x, lateral y), heading error, and the error of
the step speed |p_i - p_(i-1)| / 0.5 s. Also the lag-1 autocorrelation of each error along the steps, which sets
the correlation of the matched noise. Writes src/tokens_vs_flows/token_error.json.
Usage: python -m tokens_vs_flows.token_error [--n 3000]
       python -m tokens_vs_flows.token_error --check-noise [--n 500]   # same statistics for the matched AR(1) noise
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from tokens_vs_flows import CACHE
from tokens_vs_flows.actions import clean_trajectories, to_local
from tokens_vs_flows.data import ScenarioDataset
from src.smart.tokens.token_processor import TokenProcessor
from src.smart.utils import wrap_angle
from torch_geometric.loader import DataLoader

OUT = Path(__file__).with_name("token_error.json")
TYPES = {0: "veh", 1: "ped", 2: "cyc"}
SPEED_BINS = [0.0, 0.5, 3.0, 10.0, 1e9]  # m/s, raw speed over the 0.5 s step into the pose


def lag1(e: torch.Tensor, m: torch.Tensor) -> float:
    """Lag-1 autocorrelation along steps of error e [A, S] over pairs where both steps are valid."""
    pair = m[:, 1:] & m[:, :-1]
    a, b = e[:, 1:][pair].double(), e[:, :-1][pair].double()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1]) if len(a) > 2 else float("nan")


@torch.no_grad()
def check_noise(n: int) -> None:
    """Perturb raw training poses with the model's matched noise; compare robust std per type x speed bin x dim
    (lon, lat, heading, step speed) with the measured tokenisation error."""
    from tokens_vs_flows.actions import perturb_anchors
    from tokens_vs_flows.model import ModelConfig, SimAgent
    err = json.loads(OUT.read_text())
    enc = SimAgent(ModelConfig(head="flow", hist_noise=True)).agent_encoder
    ds = ScenarioDataset(CACHE / "training")
    idx = random.Random(1).sample(range(len(ds)), min(n, len(ds)))
    acc = {t: {k: [] for k in ("lon", "lat", "head", "speed", "m", "ms", "v")} for t in TYPES}
    g = torch.Generator().manual_seed(0)
    fr = torch.arange(0, 91, 5)
    for batch in DataLoader(torch.utils.data.Subset(ds, idx), batch_size=16, num_workers=8):
        raw = clean_trajectories(batch["agent"])
        typ = batch["agent"]["type"].long()
        ap_, ah_ = perturb_anchors(raw, typ, enc.noise_sigma, enc.noise_rho, enc.noise_speed_edges, generator=g)
        p_raw, h_raw = raw["pos"][:, fr[1:]], raw["head"][:, fr[1:]]
        m = raw["valid"][:, fr[:-1]] & raw["valid"][:, fr[1:]]
        d, dh = to_local(ap_[:, 1:], ah_[:, 1:], p_raw, h_raw)
        sp_n = (ap_[:, 2:] - ap_[:, 1:-1]).norm(dim=-1) / 0.5
        sp_r = (p_raw[:, 1:] - p_raw[:, :-1]).norm(dim=-1) / 0.5
        v_in = (raw["pos"][:, fr[1:]] - raw["pos"][:, fr[:-1]]).norm(dim=-1) / 0.5
        for t in TYPES:
            a = typ == t
            acc[t]["lon"].append(d[a, :, 0]); acc[t]["lat"].append(d[a, :, 1]); acc[t]["head"].append(wrap_angle(dh[a]))
            acc[t]["speed"].append((sp_n - sp_r)[a]); acc[t]["m"].append(m[a]); acc[t]["ms"].append((m[:, 1:] & m[:, :-1])[a])
            acc[t]["v"].append(v_in[a])
    for t, name in TYPES.items():
        c = {k: torch.cat(v) for k, v in acc[t].items()}
        for b in err[name]["by_speed"]:
            lo, hi = b["speed_bin"]
            inb = (c["v"] >= lo) & (c["v"] < hi)
            got = {}
            for k in ("lon", "lat", "head", "speed"):
                mk = (c["ms"] & inb[:, 1:]) if k == "speed" else (c["m"] & inb)
                vals = c[k][mk]
                if len(vals) >= 50 and k in b:
                    got[k] = (round(float(1.4826 * (vals - vals.median()).abs().median()), 4), round(b[k]["std_robust"], 4))
            print(f"{name} {lo}-{hi} m/s (noise, token):", got)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--check-noise", action="store_true")
    args = ap.parse_args()
    if args.check_noise:
        return check_noise(args.n)
    ds = ScenarioDataset(CACHE / "training")
    idx = random.Random(0).sample(range(len(ds)), min(args.n, len(ds)))
    s = OmegaConf.create({"num_k": 1, "temp": 1.0})
    tp = TokenProcessor("map_traj_token5.pkl", "agent_vocab_555_s2.pkl", s, s).eval()
    acc = {t: {k: [] for k in ("lon", "lat", "head", "speed", "m", "ms", "v")} for t in TYPES}
    for batch in DataLoader(torch.utils.data.Subset(ds, idx), batch_size=16, num_workers=8):
        raw = clean_trajectories(batch["agent"])
        fr = torch.arange(0, 91, 5)
        v_in = (raw["pos"][:, fr[1:]] - raw["pos"][:, fr[:-1]]).norm(dim=-1) / 0.5  # [A, 18] raw speed into step
        _, ta = tp(batch)
        frames = torch.arange(5, 91, 5)
        p_raw, h_raw = raw["pos"][:, frames], raw["head"][:, frames]
        m = ta["valid_mask"]  # [A, 18]
        d, dh = to_local(ta["gt_pos"], ta["gt_heading"], p_raw, h_raw)  # tokenised pose in the raw pose's frame
        sp_tok = (ta["gt_pos"][:, 1:] - ta["gt_pos"][:, :-1]).norm(dim=-1) / 0.5
        sp_raw = (p_raw[:, 1:] - p_raw[:, :-1]).norm(dim=-1) / 0.5
        ms = m[:, 1:] & m[:, :-1]
        for t in TYPES:
            a = batch["agent"]["type"] == t
            acc[t]["lon"].append(d[a, :, 0]); acc[t]["lat"].append(d[a, :, 1]); acc[t]["head"].append(wrap_angle(dh[a]))
            acc[t]["speed"].append((sp_tok - sp_raw)[a]); acc[t]["m"].append(m[a]); acc[t]["ms"].append(ms[a])
            acc[t]["v"].append(v_in[a])
    out = {}
    for t, name in TYPES.items():
        c = {k: torch.cat(v) for k, v in acc[t].items()}
        row = {"n_steps": int(c["m"].sum())}
        pos_err = torch.stack([c["lon"], c["lat"]], dim=-1).norm(dim=-1)[c["m"]]
        row["frac_steps_pos_err_over_1m"] = float((pos_err > 1.0).float().mean())  # tokenisation failures
        row["frac_steps_pos_err_over_0.1m"] = float((pos_err > 0.1).float().mean())
        for k in ("lon", "lat", "head", "speed"):
            mk = c["ms"] if k == "speed" else c["m"]
            v = c[k][mk]
            med = v.median()
            sd_r = float(1.4826 * (v - med).abs().median())  # robust std (MAD): the bulk, not the rare failures
            bulk = mk & ((c[k] - med).abs() <= 5 * sd_r)
            row[k] = {"mean": float(v.mean()), "std": float(v.std()), "p95_abs": float(v.abs().quantile(0.95)),
                      "std_robust": sd_r, "frac_beyond_5_robust_sd": float(1 - bulk.sum() / mk.sum())}
            if k != "speed":
                row[k]["lag1"] = lag1(c[k], mk)
                row[k]["lag1_bulk"] = lag1(c[k], bulk)
    # speed of the step into each pose, for the per-speed-bin statistics below
        by = []
        for lo, hi in zip(SPEED_BINS[:-1], SPEED_BINS[1:]):
            inb = (c["v"] >= lo) & (c["v"] < hi)
            b = {"speed_bin": [lo, hi], "n_steps": int((c["m"] & inb).sum())}
            for k in ("lon", "lat", "head", "speed"):
                mk = (c["ms"] if k == "speed" else c["m"]) & (inb[:, 1:] if k == "speed" else inb)
                v = c[k][mk]
                if len(v) < 50:
                    continue
                med = v.median()
                sd_r = float(1.4826 * (v - med).abs().median())
                b[k] = {"std": float(v.std()), "std_robust": sd_r, "p95_abs": float(v.abs().quantile(0.95))}
                if k != "speed":
                    bulk = mk & ((c[k] - med).abs() <= 5 * max(sd_r, 1e-6))
                    b[k]["lag1_bulk"] = lag1(c[k], bulk)
            by.append(b)
        row["by_speed"] = by
        out[name] = row
        print(name, json.dumps({k: v for k, v in row.items() if k != "by_speed"}))
        for b in by:
            print("   ", b["speed_bin"], b["n_steps"], {k: (round(b[k]["std_robust"], 4), round(b[k]["p95_abs"], 4), round(b[k]["std"], 4))
                                                   for k in ("lon", "lat", "head", "speed") if k in b})
    out["n_scenarios"] = len(idx)
    out["definition"] = __doc__.strip()
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
