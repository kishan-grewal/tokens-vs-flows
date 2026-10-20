# Design notes per arm

Backbone: catk SMART-nano (≈1.2M params, hidden 64), held identical across arms. The head lives at `SMARTAgentDecoder.token_predict_head` in `src/smart/modules/agent_decoder.py`. It is used in `forward()` (training) and `inference()`, which samples via `src/smart/utils/rollout.py::sample_next_token_traj`. The loss is `src/smart/metrics/cross_entropy.py::CrossEntropy`.

## Head interface (as implemented)
Both arms run catk's map encoder, agent decoder and rollout loop unchanged. `patches/catk_action_hooks.patch` moves the two action-specific pieces of `SMARTAgentDecoder.inference()` into overridable methods: `sample_next_action()` (head + sampler) and `embed_next_action()` (input embedding of the chosen action). The token arm uses catk's defaults; the flow arm (`src/tokens_vs_flows/model.py::FlowAgentDecoder`) overrides both.
- **Action:** the next 0.5 s segment in the agent frame at the current 2 Hz step. Token arm: a token id (catk vocabulary `agent_vocab_555_s2.pkl`, 2048 per type). Flow arm: 5 waypoints × (x, y, yaw), normalised per agent type (`action_stats.json`, 3000 training scenes).
- **Input embedding of the chosen action:** catk embeds a token with a per-type MLP over the 8-d box contour of the token's end pose (fixed box per type). The flow arm uses the same MLPs on the contour of the continuous action's end pose: same function, continuous input.
- **Teacher-forced history:** token arm, catk's rolling-tokenised ground truth (2 cm mean, 4 cm p95 off the raw poses on validation); flow arm, raw ground-truth poses at 2 Hz. Neither arm adds input noise (catk's behaviour-cloning config has noised tokenisation off).
- **History-noise variant of the flow arm (`--hist-noise true`, runs `bc_flow_noise_s*`):** a test of whether the token arm's tokenised history acts as training-time noise that protects it against compounding error. The flow arm's 2 Hz input poses are perturbed during training with Gaussian AR(1) noise in each pose's frame (longitudinal, lateral, heading), matched per agent type and speed bin to the measured tokenisation error (robust std and lag-1 correlation; `python -m tokens_vs_flows.token_error`), and the targets become the raw waypoints seen from the noisy pose. Moving vehicles: about 2.3 cm per axis and 0.007 rad. Rare tokenisation failures over 1 m (0.14% of vehicle steps) are not modelled. The rule for when this variant replaces the plain flow arm in the headline comparison was fixed before any flow run was scored: the flow arm is below the token arm on the kinematic bucket by more than twice the seed spread with paired Wilcoxon p < 0.05, or its ADE growth from 1 s to 8 s is more than 1.5× the token arm's.
- **Targets:** the same (agent, step) pairs in both arms (valid start, end and train mask). Token: catk's CE target, the token closest to the raw next pose seen from the tokenised current pose, label smoothing 0.1. Flow: the raw logged segment.
- Sanity check for every head: overfit 1 batch, then a 3-scenario closed-loop rollout plot (`python -m tokens_vs_flows.overfit`).

## Continuous flow head (as implemented)
- A minimal MLP denoiser: [x_t, h, sinusoidal(t)] → 3 × (Linear 256 + SiLU) → Linear 15, last layer zero-initialised; 172k parameters (catk's token head: 137k; shared backbone 1.02M). A FiLM-residual variant (897k parameters) was not used, to keep the heads comparable in size. We call it a *minimal flow head*: Flow-ERD's AFM decoder (chunked transformer, per-type kinematics, receding horizon) is much stronger.
- Rectified-flow / conditional flow matching loss (Lipman et al.): x_t = (1 − t) x0 + t x1, x0 ~ N(0, I), t ~ U[0, 1], target x1 − x0; 4 noise draws per target, loss in fp32. Sampling: Euler, 10 steps by default; record the realism vs steps curve at 1, 2, 5, 10.
- Training (both arms): AdamW, lr 5e-4, cosine to 1%, weight decay 0.01, gradient clip 0.5, fp32 with TF32 matmuls (catk's embedding code fails under bf16 autocast). A batch too large for GPU memory is processed in micro-batches with gradient accumulation (same gradient).
- Optional per-type kinematics (Flow-ERD): predict bicycle-model controls for vehicles and cyclists, displacements for pedestrians. Use it only if the plain version beats the constant-velocity baseline and compute remains.

## Uniform-bin head (factorisation study)
- Per-coordinate uniform bins (e.g. 15 values × 256 bins), each predicted INDEPENDENTLY given h. That's where within-token factorisation error comes from (Yi et al.).
- CFG: drop the conditioning (the map/agent context in h) with p = 0.1 during training. At sampling, use logits_g = (1 + w)·cond − w·uncond, with w ∈ {0, 1, 2}.
- Mode-mixture detector: Yi et al.'s definition, run on every head.

## Discrete flow over bins (factorisation study)
Discrete Flow Matching (Gat et al.) with a masked or uniform source over the same bins, about 10 steps.

## Reward: MLOO (RL study)
- Leave-one-out realism over a group of K = 4 rollouts of the same scenario: r_i = M(all K) − M(all K except i), where M is the WOSAC realism meta-metric from the fast metric.
- This matches RLFTSim (Eq. 2): its MLOO reward equals the centred A_i below, scored on the ego plus up to 8 tracks to predict, as in our metric.
- Advantage: A_i = r_i − mean_j(r_j), the group baseline. Rewards are computed on training scenes only (ground truth extracted from the training shards), never on the evaluation splits.
- Hacking watch: always log collision rate, off-road rate, minADE and CPD alongside the reward.

## RL methods (RL study)
- **Offline AWR rounds (both heads):** weights w_i = exp(A_i / β), clipped at 20, with β = 0.02 set once from the advantage scale on the token head (std ≈ 0.023) and then frozen. Weighted CE for tokens, weighted CFM loss for flow. Each round: 2000 training scenes × K = 4 rollouts from the current policy, then one epoch of fine-tuning (lr 5e-5). Same rollout budget per round for both heads.
- **Data anchor:** a first token round fine-tuned on its own rollouts alone lowered realism by 0.0017 (paired 95% CI [−0.0029, −0.0005]): weighted self-imitation on rollouts that are worse than the log drifts away from the data. Every AWR round therefore mixes logged training scenes (outside the round's scenes) 1:1 with the rollouts at weight 1. With the anchor, round 1 is neutral (−0.0006, CI [−0.0017, +0.0005]).
- **On-policy:** PPO-clip on tokens with the exact log-ratio over the old policy's top-48 support (the sampling distribution). FPO on flow, with the ratio exp(−(L_θ − L_old)) over N = 8 stored (t, ε) draws per action, L the ε-form CFM loss (= t² × velocity error). Same reward, same K, same number of updates, same anchor: a k3 KL estimate to the frozen behaviour-cloning model in each head's own likelihood (exact for tokens, the −L surrogate FPO uses for flow), with an adaptive coefficient driving both arms to the same KL target (0.01). Each iteration: 32 training scenes × K = 4, advantages normalised by their batch std, 4 epochs of minibatch updates, dropout off. Controls: the same loop with zero advantages (drift from self-generated data alone).
- No RoaD/CAT-K stage by default: RL starts straight from the behaviour-cloning checkpoints, since a weaker base shows larger RL gains (RLFTSim). Add RoaD only if offline AWR fails to improve either head.

## Metrics (eval/, frozen before any model comparison; definitions in eval/README.md)
- Realism meta-metric (WOSAC 2025) and its kinematic / interactive / map buckets, via the TrajTok fast metric (meta-metric within 1e-6 of the official metric on our check, every component within 2e-5).
- Collision, off-road and traffic-light violation rates; ADE and minADE.
- CPD diversity (Flow-ERD Eqs. 20-21, our per-type scales) and ego intent entropy over the 8 WOMD trajectory types.
- GT_UB, the log-replay score on the same scenarios; the token replay (the log after catk's tokenisation, decoded like a rollout: the token arm's ceiling); constant-velocity, noisy constant-velocity and random-agent baselines.
- ADE growth from 1 s to 8 s of closed-loop simulation (`eval/ade_growth.py`, added after the freeze), an indicator of compounding error.
- Paired per-scenario deltas with bootstrap 95% CIs (eval/compare.py).

## Pitfalls
- Collision plus off-road make up 50% of the realism score, and the ground truth itself contains about 4.2% false collisions and 16.3% false off-road events. A copy of the ground truth that never crashes only scores ≈0.72. Always report the parts.
- Expect realism gains of about 0.003. Lead with the paired stats and with metrics that swing more (CPD, mode-mixture rate).
