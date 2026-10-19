"""Checks that the continuous-action pipeline matches catk's tokenizer frames. Needs the preprocessed val cache.

Run: PYTHONPATH=src .venv/bin/python -m pytest -q tests
"""
import copy

import pytest
import torch
from omegaconf import OmegaConf

from tokens_vs_flows import CACHE
from tokens_vs_flows.actions import (clean_trajectories, extrapolate_to_prev_token_step, segment_features,
                                     to_global, to_local, waypoint_contours)
from tokens_vs_flows.data import ScenarioDataset
from src.smart.tokens.token_processor import TokenProcessor
from torch_geometric.loader import DataLoader


def catk_extrapolate(valid, pos, heading, vel, shift=5):
    """catk's original per-agent loop (TokenProcessor._extrapolate_agent_to_prev_token_step)."""
    first_valid_step = torch.max(valid, dim=1).indices
    for i, t in enumerate(first_valid_step):
        n = t % shift
        if (t == 10) and (not valid[i, 10 - shift]):
            n = shift
        if n > 0:
            vel[i, t - n:t] = vel[i, t]
            valid[i, t - n:t] = True
            heading[i, t - n:t] = heading[i, t]
            for j in range(n):
                pos[i, t - j - 1] = pos[i, t - j] - vel[i, t] * 0.1
    return valid, pos, heading, vel


@pytest.fixture(scope="module")
def batch():
    if not (CACHE / "validation").exists():
        pytest.skip("no preprocessed validation cache")
    ds = ScenarioDataset(CACHE / "validation", train=True)
    return next(iter(DataLoader(ds, batch_size=8, shuffle=False)))


def test_extrapolation_matches_catk(batch):
    a = batch["agent"]
    args = [a["valid_mask"].clone(), a["position"][..., :2].clone(), a["heading"].clone(), a["velocity"].clone()]
    ref = catk_extrapolate(*[x.clone() for x in args])
    out = extrapolate_to_prev_token_step(*args)
    for r, o in zip(ref, out):
        assert torch.equal(r, o)


def test_segments_match_tokenizer(batch):
    s = OmegaConf.create({"num_k": 1, "temp": 1.0})
    tp = TokenProcessor("map_traj_token5.pkl", "agent_vocab_555_s2.pkl", s, s)
    data = copy.deepcopy(batch)
    traj = clean_trajectories(data["agent"])
    _, ta = tp(data)
    seg = segment_features(traj, ta["token_agent_shape"])
    assert torch.equal(seg["in_valid"], ta["valid_mask"])
    assert (seg["pos2"] - ta["gt_pos_raw"])[ta["valid_mask"]].abs().max() == 0
    # the 5th waypoint of the action out of step i is the pose at step i+1
    m = seg["target_valid"][:, :-1] & seg["in_valid"][:, :-1]
    lp, _ = to_local(seg["pos2"][:, 1:], seg["head2"][:, 1:], seg["pos2"][:, :-1], seg["head2"][:, :-1])
    assert (seg["target"][:, :-1, 12:14] - lp)[m].abs().max() < 1e-4


def test_waypoint_contours_roundtrip():
    torch.manual_seed(0)
    a = torch.randn(7, 15)
    a[:, 2::3] = a[:, 2::3].clamp(-3, 3)
    shape = torch.tensor([[2.0, 4.8]] * 7)
    c = waypoint_contours(a, shape)  # [7, 6, 4, 2]
    wp = a.view(7, 5, 3)
    assert torch.allclose(c[:, 1:].mean(2), wp[..., :2], atol=1e-5)
    d = c[:, 1:, 0] - c[:, 1:, 3]
    assert torch.allclose(torch.atan2(d[..., 1], d[..., 0]), wp[..., 2], atol=1e-5)
    assert torch.allclose(c[:, 0].mean(1), torch.zeros(7, 2), atol=1e-6)


def test_local_global_inverse():
    torch.manual_seed(0)
    p, h, p0, h0 = torch.randn(5, 2) * 10, torch.randn(5), torch.randn(5, 2) * 10, torch.randn(5)
    lp, lh = to_local(p, h, p0, h0)
    gp, gh = to_global(lp, lh, p0, h0)
    assert torch.allclose(gp, p, atol=1e-4)
    assert torch.allclose(torch.atan2(gh.sin(), gh.cos()), torch.atan2(h.sin(), h.cos()), atol=1e-5)
