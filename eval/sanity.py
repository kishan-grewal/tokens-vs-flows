"""Eval sanity checks, run once before the eval is frozen.

1. Ground-truth replay (32 copies of the log) on the split: the realism ceiling, written to eval/gt_ub_<split>.json.
2. Constant velocity must beat the random agent on realism (both as defined in the WOSAC paper, Sec. 5.1).
   noisy_actions (constant velocity plus random actions) is scored too, as a diagnostic only.
3. Fast metric vs the official waymo-open-dataset metric (same 2025 config) on the first 20 scenarios of the
   split, for log-replay, constant-velocity and noisy-action rollouts: |fast - official| < 1e-3 on the
   meta-metric.
Writes eval/sanity_<split>.json and eval/gt_ub_<split>.json; once eval/ holds a sanity_<split>.json it writes to
runs/sanity_rerun/ instead, so a rerun can never overwrite the recorded results.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

import wosac
from baselines import BASELINES
from score import aggregate, score_rollouts

N_OFFICIAL = 20
TOL = 1e-3


def out_dir():
    if any((wosac.ROOT / "eval").glob("sanity_*.json")):
        d = wosac.ROOT / "runs" / "sanity_rerun"
        d.mkdir(parents=True, exist_ok=True)
        return d
    return wosac.ROOT / "eval"


def gt_ub_record(split: str, means: dict, b: dict) -> dict:
    return {"split": split, "realism": means["metametric"], **b, "means": means}


def write_gt_ub(split: str) -> int:
    """Log-replay ceiling of a split (no model involved) -> gt_ub_<split>.json."""
    wosac.check_trajtok_commit()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config, sigma = wosac.load_config(), wosac.load_sigma()
    rows = score_rollouts(lambda sid, gt: BASELINES["gt_replay"](gt), wosac.read_split(split), device, sigma, config,
                          log_every=250)
    means, b = aggregate(rows, config)
    (out_dir() / f"gt_ub_{split}.json").write_text(json.dumps(gt_ub_record(split, means, b), indent=1) + "\n")
    print(f"log replay on {split}: realism {means['metametric']:.4f}", {k: round(v, 4) for k, v in b.items()})
    return 0


def official_rollouts_proto(scenario_id: str, ro: dict):
    from waymo_open_dataset.protos import sim_agents_submission_pb2 as sub
    st = ro["states"].double().numpy()
    ids = ro["agent_id"].long().numpy()
    scenes = []
    for r in range(st.shape[0]):
        trajs = [sub.SimulatedTrajectory(center_x=st[r, a, :, 0], center_y=st[r, a, :, 1], center_z=st[r, a, :, 2],
                                         heading=st[r, a, :, 3], object_id=int(ids[a])) for a in range(len(ids))]
        scenes.append(sub.JointScene(simulated_trajectories=trajs))
    return sub.ScenarioRollouts(joint_scenes=scenes, scenario_id=scenario_id)


def official_metrics(scenario_id: str, ro: dict) -> dict:
    import tensorflow as tf
    from waymo_open_dataset.protos import scenario_pb2
    from waymo_open_dataset.wdl_limited.sim_agents_metrics import metrics as official
    from google.protobuf import text_format
    from waymo_open_dataset.protos import sim_agents_metrics_pb2
    tf.config.set_visible_devices([], "GPU")
    scenario = scenario_pb2.Scenario()
    for rec in tf.data.TFRecordDataset([str(wosac.TFRECORD_DIR / f"{scenario_id}.tfrecords")], compression_type=""):
        scenario.ParseFromString(bytes(rec.numpy()))
        break
    config = sim_agents_metrics_pb2.SimAgentMetricsConfig()
    text_format.Parse((Path(official.__file__).parent / "challenge_2025_sim_agents_config.textproto").read_text(),
                      config)
    m = official.compute_scenario_metrics_for_bundle(config, scenario, official_rollouts_proto(scenario_id, ro))
    return {f: float(getattr(m, f)) for f in wosac.SCENARIO_FIELDS}


def main(split: str = "val_frozen"):
    wosac.check_trajtok_commit()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config, sigma = wosac.load_config(), wosac.load_sigma()
    ids = wosac.read_split(split)
    out = {"split": split, "n_scenarios": len(ids), "baselines": {}}
    for name, fn in BASELINES.items():
        t0 = time.time()
        rows = score_rollouts(lambda sid, gt: fn(gt), ids, device, sigma, config, log_every=250)
        means, b = aggregate(rows, config)
        out["baselines"][name] = {"realism": means["metametric"], **b, "means": means,
                                  "per_scenario_realism": [r["metametric"] for r in rows],
                                  "s_per_scenario": (time.time() - t0) / len(ids)}
        print(f"{name}: realism {means['metametric']:.4f} kin {b['kinematic']:.4f} int {b['interactive']:.4f} "
              f"map {b['map']:.4f} coll {means['simulated_collision_rate']:.4f} "
              f"offroad {means['simulated_offroad_rate']:.4f} minADE {means['min_average_displacement_error']:.3f}",
              flush=True)
    bl = out["baselines"]
    gt_r, cv_r, rnd_r = (bl[k]["realism"] for k in ("gt_replay", "constant_velocity", "random_agent"))
    out["check_gt_is_ceiling"] = bool(gt_r > cv_r and gt_r > rnd_r)
    out["check_constvel_beats_random"] = bool(cv_r > rnd_r)

    diffs = []
    t0 = time.time()
    for sid in ids[:N_OFFICIAL]:
        gt = wosac.load_gt(sid, device)
        for name in ("gt_replay", "constant_velocity", "noisy_actions"):
            ro = BASELINES[name](gt)
            fast = wosac.realism(config, gt, ro, device)
            off = official_metrics(sid, ro)
            diffs.append({"scenario_id": sid, "rollouts": name,
                          **{f: abs(fast[f] - off[f]) for f in wosac.SCENARIO_FIELDS},
                          "fast_metametric": fast["metametric"], "official_metametric": off["metametric"]})
    max_diff = {f: max(d[f] for d in diffs) for f in wosac.SCENARIO_FIELDS}
    out["official_check"] = {"n_scenarios": N_OFFICIAL, "max_abs_diff": max_diff, "rows": diffs,
                             "s_total": time.time() - t0}
    out["check_fast_vs_official"] = bool(max_diff["metametric"] < TOL)
    out["pass"] = out["check_gt_is_ceiling"] and out["check_constvel_beats_random"] and out["check_fast_vs_official"]
    print("fast vs official max |diff|:", json.dumps({k: float(f"{v:.2e}") for k, v in max_diff.items()}))
    print("checks:", {k: out[k] for k in out if k.startswith("check_")}, "PASS" if out["pass"] else "FAIL")

    gt = bl["gt_replay"]
    (out_dir() / f"gt_ub_{split}.json").write_text(json.dumps(
        {"split": split, "realism": gt["realism"], "kinematic": gt["kinematic"], "interactive": gt["interactive"],
         "map": gt["map"], "means": gt["means"]}, indent=1) + "\n")
    (out_dir() / f"sanity_{split}.json").write_text(json.dumps(out, indent=1) + "\n")
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
