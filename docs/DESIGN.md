# Design notes per arm

Backbone: catk SMART-nano (≈1.2M params, hidden 64), held identical across arms. The head lives at `SMARTAgentDecoder.token_predict_head` in `src/smart/modules/agent_decoder.py`. It is used in `forward()` (training) and `inference()`, which samples via `src/smart/utils/rollout.py::sample_next_token_traj`. The loss is `src/smart/metrics/cross_entropy.py::CrossEntropy`.

## Head interface
`ActionHead` with `loss(h, target)`, `sample(h, n, guidance=None)` and `logp_or_surrogate(h, a)`.
- `target` / `a` is the next 0.5 s motion segment in the agent frame. For token arms it's the token id; for continuous arms it's 5 waypoints × (x, y, yaw), plus optional speed.
- **Input embedding of the chosen action:** token arms keep catk's token embedding. Continuous arms use an MLP over the continuous segment. This difference is part of the arm.
- Sanity check for every head: overfit 1 batch, then a 3-scenario closed-loop rollout plot.

## Continuous flow head
- An MLP denoiser, 3 × 256, conditioned on the backbone hidden h plus a sinusoidal time embedding (pi0-style action expert, scaled down).
- Rectified-flow / conditional flow matching loss (Lipman et al.). Sampling: Euler, 10 steps by default; record the realism vs steps curve at 1, 2, 5, 10.
- Optional per-type kinematics (Flow-ERD): predict bicycle-model controls for vehicles and cyclists, displacements for pedestrians. Use it only if the plain version beats the constant-velocity baseline and compute remains.

## Uniform-bin head (factorisation study)
- Per-coordinate uniform bins (e.g. 15 values × 256 bins), each predicted INDEPENDENTLY given h. That's where within-token factorisation error comes from (Yi et al.).
- CFG: drop the conditioning (the map/agent context in h) with p = 0.1 during training. At sampling, use logits_g = (1 + w)·cond − w·uncond, with w ∈ {0, 1, 2}.
- Mode-mixture detector: Yi et al.'s definition, run on every head.

## Discrete flow over bins (factorisation study)
Discrete Flow Matching (Gat et al.) with a masked or uniform source over the same bins, about 10 steps.

## Reward: MLOO (RL study)
- Leave-one-out realism over a group of K = 4 rollouts of the same scenario: r_i = M(all K) − M(all K except i), where M is the WOSAC realism meta-metric from the fast metric.
- Definition and normalisation as in RLFTSim.
- Advantage: A_i = r_i − mean_j(r_j), the group baseline.
- Hacking watch: always log collision rate, off-road rate, minADE and CPD alongside the reward.

## RL methods (RL study)
- **Offline AWR rounds (both heads):** weights w_i = exp(A_i / β), clipped at 20, with β tuned once on the token head and then frozen. Weighted CE for tokens, weighted CFM loss for flow. Same rollouts budget per round for both heads.
- **On-policy:** PPO-clip on tokens with the exact log-ratio. FPO on flow, with the ratio exp(−(L_CFM,θ − L_CFM,old)) estimated using shared noise and time samples (see the FPO paper). Same reward, same K, same number of updates, same KL/anchor budget: the exact KL to the behaviour-cloning model for tokens, the CFM loss to the frozen behaviour-cloning model for flow.
- No RoaD/CAT-K stage by default: RL starts straight from the behaviour-cloning checkpoints, since a weaker base shows larger RL gains (RLFTSim). Add RoaD only if offline AWR fails to improve either head.

## Metrics (eval/score.py; frozen before any model comparison)
- Realism meta-metric and its components (TrajTok fast metric, 2025 version).
- Collision and off-road rates.
- minADE.
- CPD diversity (Flow-ERD definition) and intent entropy over the 8 WOMD trajectory types.
- GT_UB, the ground-truth replay score on the same scenarios.
- Paired per-scenario deltas with bootstrap 95% CIs.

## Pitfalls
- Collision plus off-road make up 50% of the realism score, and the ground truth itself contains about 4.2% false collisions and 16.3% false off-road events. A copy of the ground truth that never crashes only scores ≈0.72. Always report the parts.
- Expect realism gains of about 0.003. Lead with the paired stats and with metrics that swing more (CPD, mode-mixture rate).
