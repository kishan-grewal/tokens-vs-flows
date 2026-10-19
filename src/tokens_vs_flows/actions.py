"""Continuous 0.5 s actions from the 10 Hz ground truth, in the frames catk's tokenizer uses.

2 Hz step i (i = 0..17) sits at 10 Hz frame 5(i+1). The action *into* step i is the pose at frame 5(i+1)
seen from the pose at frame 5i (what catk's token at step i encodes). The action *out of* step i is the
segment of frames 5(i+1)+1 .. 5(i+1)+5 seen from the pose at frame 5(i+1): 5 waypoints x (x, y, yaw).
"""
from typing import Dict

import torch
from torch import Tensor

from tokens_vs_flows import ROOT  # noqa: F401  (sets sys.path for catk)
from src.smart.tokens.token_processor import TokenProcessor
from src.smart.utils import cal_polygon_contour, wrap_angle

SHIFT = 5  # 10 Hz frames per 0.5 s step
N_WAYPOINT = 5
ACTION_DIM = 3 * N_WAYPOINT


@torch.no_grad()
def extrapolate_to_prev_token_step(valid: Tensor, pos: Tensor, heading: Tensor, vel: Tensor):
    """Vectorised, in-place equivalent of TokenProcessor._extrapolate_agent_to_prev_token_step.

    Extends each track backwards from its first valid frame to the previous 0.5 s boundary (a full 0.5 s if the
    track starts at the current frame), at constant velocity and heading.
    """
    first = torch.max(valid, dim=1).indices  # first valid frame
    n_fill = first % SHIFT
    n_fill = torch.where((first == 10) & ~valid[:, 10 - SHIFT], torch.full_like(n_fill, SHIFT), n_fill)
    k = torch.arange(1, SHIFT + 1, device=valid.device)
    fill = k[None] <= n_fill[:, None]  # [A, 5]
    a_idx = torch.arange(valid.shape[0], device=valid.device)[:, None].expand_as(fill)[fill]
    f_idx = (first[:, None] - k[None])[fill]
    k_f = k[None].expand_as(fill)[fill].to(pos.dtype)
    src = first[a_idx]
    vel[a_idx, f_idx] = vel[a_idx, src]
    valid[a_idx, f_idx] = True
    heading[a_idx, f_idx] = heading[a_idx, src]
    pos[a_idx, f_idx] = pos[a_idx, src] - vel[a_idx, src] * 0.1 * k_f[:, None]
    return valid, pos, heading, vel


def _extrapolate_method(self, valid, pos, heading, vel):
    return extrapolate_to_prev_token_step(valid, pos, heading, vel)


# Same result as catk's per-agent Python loop (checked in tests/test_actions.py), without a GPU sync per agent.
TokenProcessor._extrapolate_agent_to_prev_token_step = _extrapolate_method


@torch.no_grad()
def clean_trajectories(agent: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """10 Hz pos/heading/valid with catk's heading fix and history extrapolation, computed on copies.

    catk's tokenizer applies the same two steps but edits the batch in place.
    """
    valid = agent["valid_mask"].clone()
    head = TokenProcessor._clean_heading(valid, agent["heading"].clone())
    pos = agent["position"][..., :2].clone()
    valid, pos, head, _ = extrapolate_to_prev_token_step(valid, pos, head, agent["velocity"].clone())
    return {"pos": pos, "head": head, "valid": valid}


def to_local(pos: Tensor, head: Tensor, pos0: Tensor, head0: Tensor):
    """Express poses (pos [..., 2], head [...]) in the frame of (pos0 [..., 2], head0 [...])."""
    d = pos - pos0
    c, s = head0.cos(), head0.sin()
    x = c * d[..., 0] + s * d[..., 1]
    y = -s * d[..., 0] + c * d[..., 1]
    return torch.stack([x, y], dim=-1), wrap_angle(head - head0)


def to_global(pos_local: Tensor, head_local: Tensor, pos0: Tensor, head0: Tensor):
    c, s = head0.cos(), head0.sin()
    x = c * pos_local[..., 0] - s * pos_local[..., 1] + pos0[..., 0]
    y = s * pos_local[..., 0] + c * pos_local[..., 1] + pos0[..., 1]
    return torch.stack([x, y], dim=-1), head_local + head0


@torch.no_grad()
def segment_features(traj: Dict[str, Tensor], agent_shape: Tensor) -> Dict[str, Tensor]:
    """2 Hz poses, input-action contours and continuous action targets for every 2 Hz step.

    agent_shape: [A, 2] (width, length), catk's fixed per-type box used for token contours.
    Returns pos2 [A,18,2], head2 [A,18], in_valid [A,18], in_contour [A,18,8], target [A,18,15],
    target_valid [A,18].
    """
    pos, head, valid = traj["pos"], traj["head"], traj["valid"]
    dev = pos.device
    frames = torch.arange(SHIFT, pos.shape[1], SHIFT, device=dev)  # 5, 10, ..., 90
    pos2, head2 = pos[:, frames], head[:, frames]
    in_valid = valid[:, frames - SHIFT] & valid[:, frames]
    rel_pos, rel_head = to_local(pos2, head2, pos[:, frames - SHIFT], head[:, frames - SHIFT])
    in_contour = cal_polygon_contour(rel_pos, rel_head, agent_shape[:, None, :]).flatten(-2)  # [A,18,8]

    pad = lambda x: torch.cat([x, x[:, -1:].expand(-1, SHIFT, *x.shape[2:])], dim=1)  # noqa: E731
    pos_p, head_p = pad(pos), pad(head)
    valid_p = torch.cat([valid, valid.new_zeros(valid.shape[0], SHIFT)], dim=1)
    tgt = frames[:, None] + torch.arange(1, N_WAYPOINT + 1, device=dev)[None]  # [18, 5]
    wp_pos, wp_head = to_local(pos_p[:, tgt], head_p[:, tgt], pos2[:, :, None], head2[:, :, None])
    target = torch.cat([wp_pos, wp_head[..., None]], dim=-1).flatten(2)  # [A,18,15]
    target_valid = valid[:, frames] & valid_p[:, frames + SHIFT]
    zero = ~in_valid
    return {
        "pos2": pos2.masked_fill(zero[..., None], 0.0),
        "head2": head2.masked_fill(zero, 0.0),
        "in_valid": in_valid,
        "in_contour": in_contour,
        "target": target,
        "target_valid": target_valid,
    }


def waypoint_contours(action: Tensor, agent_shape: Tensor) -> Tensor:
    """Contours of the start pose and the 5 waypoints of a local action [A, 15] -> [A, 6, 4, 2].

    Same layout as a token's trajectory (token_traj_all), so catk's rollout code recovers the waypoint
    positions (corner mean) and headings (corner 0 - corner 3) exactly.
    """
    wp = action.view(-1, N_WAYPOINT, 3)
    start = wp.new_zeros(wp.shape[0], 1, 3)
    wp = torch.cat([start, wp], dim=1)  # [A, 6, 3]
    return cal_polygon_contour(wp[..., :2], wp[..., 2], agent_shape[:, None, :])
