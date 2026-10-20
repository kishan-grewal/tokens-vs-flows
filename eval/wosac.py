"""Scoring primitives: WOSAC 2025 realism (TrajTok fast metric), CPD diversity, ego intent entropy.

Frozen with the rest of eval/ once the sanity checks pass. Rollouts are dicts
{"agent_id": int64 [A], "states": float32 [R, A, 80, 4] = (x, y, z, heading), global frame, t = 1.1 .. 9.0 s}.
"""
import hashlib
import json
import math
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
TRAJTOK = ROOT / "third_party" / "TrajTok"
if str(TRAJTOK) not in sys.path:
    sys.path.insert(0, str(TRAJTOK))

from google.protobuf import text_format  # noqa: E402
from waymo_open_dataset.protos import sim_agents_metrics_pb2  # noqa: E402
from wosac_fast_eval_tool.fast_sim_agents_metrics import metrics as fast_metrics  # noqa: E402
from wosac_fast_eval_tool.scenario_gt_converter import gt_scenario_to_device  # noqa: E402

TRAJTOK_COMMIT = "7a9bb997d304d8fc604061c8137b1e109c618696"
GT_DIR = ROOT / "data" / "cache" / "SMART" / "validation_gt"
TFRECORD_DIR = ROOT / "data" / "cache" / "SMART" / "validation_tfrecords_splitted"
SPLITS = ROOT / "eval" / "splits"
VERSION = "2025"
N_ROLLOUTS = 32
DT = 0.1

LIKELIHOODS = [
    "linear_speed_likelihood", "linear_acceleration_likelihood", "angular_speed_likelihood",
    "angular_acceleration_likelihood", "distance_to_nearest_object_likelihood", "collision_indication_likelihood",
    "time_to_collision_likelihood", "distance_to_road_edge_likelihood", "offroad_indication_likelihood",
    "traffic_light_violation_likelihood",
]
SCENARIO_FIELDS = ["metametric", "average_displacement_error", "min_average_displacement_error", *LIKELIHOODS,
                   "simulated_collision_rate", "simulated_offroad_rate", "simulated_traffic_light_violation_rate"]
BUCKETS = {
    "kinematic": ["linear_speed", "linear_acceleration", "angular_speed", "angular_acceleration"],
    "interactive": ["distance_to_nearest_object", "collision_indication", "time_to_collision"],
    "map": ["distance_to_road_edge", "offroad_indication", "traffic_light_violation"],
}
# WOMD object types (scenario.proto): 1 vehicle, 2 pedestrian, 3 cyclist
TYPE_NAMES = {1: "veh", 2: "ped", 3: "cyc"}
INTENTS = ["stationary", "straight", "straight_left", "straight_right", "left_u_turn", "left_turn",
           "right_u_turn", "right_turn"]


def check_trajtok_commit() -> None:
    head = subprocess.run(["git", "-C", str(TRAJTOK), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if head != TRAJTOK_COMMIT:
        raise RuntimeError(f"third_party/TrajTok is at {head}, the frozen eval expects {TRAJTOK_COMMIT}")


def load_config() -> sim_agents_metrics_pb2.SimAgentMetricsConfig:
    path = TRAJTOK / "wosac_fast_eval_tool" / "fast_sim_agents_metrics" / "challenge_2025_sim_agents_config.textproto"
    config = sim_agents_metrics_pb2.SimAgentMetricsConfig()
    text_format.Parse(path.read_text(), config)
    return config


def read_split(name: str) -> List[str]:
    return [l.strip() for l in (SPLITS / f"{name}.txt").read_text().splitlines() if l.strip()]


_MANIFEST = None


def _manifest() -> dict:
    global _MANIFEST
    if _MANIFEST is None:
        f = ROOT / "eval" / "gt_manifest.sha256"
        _MANIFEST = dict(l.split()[::-1] for l in f.read_text().splitlines()) if f.exists() else {}
    return _MANIFEST


def load_gt(scenario_id: str, device: str = "cpu") -> dict:
    """Ground-truth pickle of a scenario, checked against eval/gt_manifest.sha256 when it is listed there."""
    raw = (GT_DIR / f"{scenario_id}.pkl").read_bytes()
    want = _manifest().get(scenario_id)
    if want is not None and hashlib.sha256(raw).hexdigest() != want:
        raise RuntimeError(f"ground truth of {scenario_id} differs from eval/gt_manifest.sha256")
    return gt_scenario_to_device(pickle.loads(raw), device)


def realism(config, gt: dict, rollout: dict, device: str) -> Dict[str, float]:
    """Per-scenario WOSAC 2025 fields from the fast metric (official-equivalent to < 1e-6 per its authors)."""
    predict = {"agent_id": rollout["agent_id"].to(device), "simulated_states": rollout["states"].to(device)}
    return fast_metrics.compute_scenario_metrics_for_bundle(config, gt, predict, VERSION)


def buckets(config, means: Dict[str, float]) -> Dict[str, float]:
    """Official aggregate_metrics_to_buckets: metametric-weighted average of each bucket's likelihoods."""
    out = {}
    for name, fields in BUCKETS.items():
        w = [getattr(config, f).metametric_weight for f in fields]
        out[name] = sum(wi * means[f + "_likelihood"] for wi, f in zip(w, fields)) / sum(w)
    return out


def sim_agent_types(gt: dict, agent_id: torch.Tensor) -> torch.Tensor:
    """WOMD object type of each simulated agent, matched by object id."""
    ids = gt["object_ids"].long().cpu()
    match = agent_id.long().cpu()[:, None] == ids[None]
    if not bool(match.any(1).all()):
        raise ValueError("rollout contains agents missing from the ground truth")
    return gt["object_types"].long().cpu()[match.float().argmax(1)]


def cpd(states: torch.Tensor, types: torch.Tensor, sigma: Dict[str, float]) -> float:
    """Flow-ERD Eqs. 20-21: mean over rollout pairs of the type-normalised RMS position distance.

    d(k, k')^2 = sum_c 1/sigma_c^2 * 1/(N_c T) * sum_{i in c, t} ||p^i_t(k) - p^i_t(k')||^2  over all simulated
    agents and all T = 80 simulated steps; types with N_c = 0 are skipped.
    """
    pos = states[..., :2].double()
    r = pos.shape[0]
    d2 = torch.zeros(r, r, dtype=torch.float64, device=pos.device)
    for code, name in TYPE_NAMES.items():
        m = (types == code).to(pos.device)
        if not bool(m.any()):
            continue
        x = pos[:, m].reshape(r, -1)
        n_ct = int(m.sum()) * pos.shape[2]
        d2 = d2 + torch.cdist(x, x).pow(2) / n_ct / sigma[name] ** 2
    iu = torch.triu_indices(r, r, offset=1)
    return float(d2[iu[0], iu[1]].clamp(min=0).sqrt().mean())


def classify_intent(start_xy, start_heading, start_vel, end_xy, end_heading, end_vel) -> int:
    """WOMD trajectory type rule (waymo-open-dataset motion_metrics_utils.cc, Ettinger et al. 2021)."""
    max_speed_stationary, max_disp_stationary = 2.0, 5.0
    max_lat_straight, min_lon_u_turn, max_heading_straight = 5.0, -5.0, math.pi / 6.0
    dx, dy = end_xy[0] - start_xy[0], end_xy[1] - start_xy[1]
    final_disp = math.hypot(dx, dy)
    heading_diff = math.atan2(math.sin(end_heading - start_heading), math.cos(end_heading - start_heading))
    c, s = math.cos(-start_heading), math.sin(-start_heading)
    lon, lat = c * dx - s * dy, s * dx + c * dy
    max_speed = max(math.hypot(*start_vel), math.hypot(*end_vel))
    if max_speed < max_speed_stationary and final_disp < max_disp_stationary:
        return INTENTS.index("stationary")
    if abs(heading_diff) < max_heading_straight:
        if abs(lat) < max_lat_straight:
            return INTENTS.index("straight")
        return INTENTS.index("straight_right" if lat < 0 else "straight_left")
    if heading_diff < -max_heading_straight and lat < 0:
        return INTENTS.index("right_u_turn" if lon < min_lon_u_turn else "right_turn")
    if lon < min_lon_u_turn:
        return INTENTS.index("left_u_turn")
    return INTENTS.index("left_turn")


def ego_intent_entropy(gt: dict, rollout: dict) -> float:
    """Shannon entropy (nats) of the ego's WOMD trajectory type over the rollouts of one scenario.

    Start = logged state at t = 1.0 s (current); end = last simulated step, end velocity from its last two steps.
    """
    sdc = int(gt["sdc_track_index"])
    ego_id = int(gt["object_ids"][sdc])
    k = int((rollout["agent_id"].long().cpu() == ego_id).float().argmax())
    cur = int(gt["current_time_index"])
    tr = gt["tracks"][sdc].cpu().double()
    start_xy, start_heading, start_vel = tr[cur, :2].tolist(), float(tr[cur, 6]), tr[cur, 7:9].tolist()
    st = rollout["states"][:, k].cpu().double()  # [R, 80, 4]
    counts = np.zeros(len(INTENTS))
    for r in range(st.shape[0]):
        end_vel = ((st[r, -1, :2] - st[r, -2, :2]) / DT).tolist()
        counts[classify_intent(start_xy, start_heading, start_vel, st[r, -1, :2].tolist(), float(st[r, -1, 3]),
                               end_vel)] += 1
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log(p)).sum())


def load_sigma() -> Dict[str, float]:
    return json.loads((ROOT / "eval" / "cpd_sigma.json").read_text())["sigma"]
