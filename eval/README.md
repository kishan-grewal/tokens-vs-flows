# eval/
Frozen once the sanity checks passed and not edited afterwards. Metrics added later are new files and secondary.

## Files
- `score.py`: scores `runs/<id>/rollouts/<scenario_id>.pt` (32 rollouts x simulated agents x 80 steps x (x, y, z, heading)) on a split. Writes `metrics.json` (the results.tsv fields), `metrics_detail.json` (all means and components) and `per_scenario.csv` (for paired tests).
- `wosac.py`: scoring primitives. Realism = WOSAC 2025 meta-metric from the TrajTok fast metric (pinned commit, 2025 config identical to waymo-open-dataset 1.6.7), kinematic / interactive / map buckets with the official bucket weighting, collision / off-road / traffic-light rates, ADE and minADE.
- `compare.py`: paired per-scenario comparison of two arms (seed-averaged per scenario): mean delta, 95% bootstrap CI over scenarios, paired t-test and Wilcoxon p, seed spread. A difference is called real only if it exceeds 2x the seed spread and p < 0.05.
- `baselines.py`: model-free rollouts: log replay, constant velocity and random agent (WOSAC paper definitions), and a noisy-action diagnostic.
- `sanity.py` (`score.py --sanity`): the checks run before freezing; results in `sanity_val_frozen.json`, the log-replay ceiling in `gt_ub_val_frozen.json`.
- `cpd_sigma.py`, `cpd_sigma.json`: per-type CPD scales.
- `splits/val_frozen.txt`: 1000 dev scenarios (validation shards 0-3). `splits/test_hidden.txt`: 1000 held-out scenarios (validation shards 4-7), scored once at the end.

## Definitions
- **Rollouts:** 32 per scenario, all WOSAC simulated agents (objects valid at the current step), 8 s at 10 Hz.
- **CPD** (Flow-ERD Eqs. 20-21): mean over the 496 rollout pairs of d(k, k') = sqrt(sum_c (1 / sigma_c^2) (1 / (N_c T)) sum_{i in c, t} ||p_t^i(k) - p_t^i(k')||^2), over all simulated agents and T = 80 steps. sigma_c = RMS 8 s displacement from the current position of logged agents of type c on 3000 training scenes: vehicle 22.06 m, pedestrian 4.65 m, cyclist 18.50 m. Not comparable to published CPD values.
- **Intent entropy:** the ego's trajectory type under the WOMD 8-class rule (start = logged current state, end = last simulated step, end velocity from its last two steps), Shannon entropy in nats over the 32 rollouts, averaged over scenarios.
- **Log replay (ceiling):** the logged future repeated 32 times; at steps where an object is missing from the log it is parked far outside the scene, so it interacts with nothing (simulated objects always count as present, while the logged features ignore missing objects).
- **Constant velocity:** last speed (from the last two history steps; zero if either is missing) along the last heading. **Random agent:** every step of every agent at x, y, heading ~ N(1.0, 0.1^2) in the AV frame (WOSAC paper Sec. 5.1). **Noisy actions** (diagnostic only): a unicycle from the current state with i.i.d. uniform acceleration and yaw-rate actions; it scores above constant velocity, because the likelihood metric rewards spread.

## Sanity checks (all passed before freezing)
1. Log replay scores above every baseline. 2. Constant velocity beats the random agent. 3. Fast metric vs the official waymo-open-dataset metric (2025 config) on the first 20 scenarios of val_frozen, for log-replay, constant-velocity and noisy-action rollouts: |difference| < 1e-3 on the meta-metric (observed max 6.7e-7; every field below 2e-5).

Results on val_frozen: log replay 0.8343, constant velocity 0.4095, random agent 0.2685, noisy actions 0.4622 (results.tsv rows eval_*). Log replay on test_hidden: 0.8312.
