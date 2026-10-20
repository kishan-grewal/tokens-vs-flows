"""SMART-nano sim agent with a swappable action head: catk's motion-token head or a flow-matching head.

Both arms share catk's map encoder and agent decoder code (src/smart/modules), unchanged except for the two
action hooks added by patches/catk_action_hooks.patch. The arms differ only in:
  - the head and its loss (token CE with label smoothing vs conditional flow matching);
  - the sampler (top-k over tokens vs Euler integration of the flow);
  - the chosen-action input embedding: the same per-type MLP over the action's end-pose box contour,
    evaluated on the token's (quantised) contour vs the continuous action's contour;
  - teacher-forced history: rolling-tokenised GT poses (token) vs raw GT poses (flow).
"""
import copy
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import Tensor

from tokens_vs_flows import ROOT
from tokens_vs_flows.actions import ACTION_DIM, clean_trajectories, segment_features, waypoint_contours
from tokens_vs_flows.heads import BinsHead, FlowHead
from src.smart.metrics.utils import get_euclidean_targets, get_prob_targets
from src.smart.modules.agent_decoder import SMARTAgentDecoder
from src.smart.modules.map_decoder import SMARTMapDecoder
from src.smart.tokens.token_processor import TokenProcessor

ACTION_STATS = Path(__file__).with_name("action_stats.json")


@dataclass
class ModelConfig:
    head: str = "token"  # token | flow
    # catk configs/model/smart.yaml + smart_nano_1M.yaml
    hidden_dim: int = 64
    num_freq_bands: int = 64
    num_heads: int = 4
    head_dim: int = 8
    dropout: float = 0.1
    hist_drop_prob: float = 0.1
    num_map_layers: int = 2
    num_agent_layers: int = 4
    pl2pl_radius: float = 10.0
    pl2a_radius: float = 20.0
    a2a_radius: float = 20.0
    time_span: int = 20
    num_historical_steps: int = 11
    num_future_steps: int = 80
    # token head training loss (catk smart.yaml training_loss) and sampling (catk wosac_sub.yaml)
    label_smoothing: float = 0.1
    token_topk: int = 48
    token_temp: float = 1.0
    # flow head
    flow_width: int = 256
    flow_depth: int = 3
    flow_noise_samples: int = 4
    flow_steps: int = 10
    flow_noise_scale: float = 1.0
    # uniform-bins head (factorisation study)
    bins_n: int = 256
    bins_range: float = 5.0
    bins_width: int = 64
    bins_p_uncond: float = 0.1
    bins_cfg: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class ContinuousAgentDecoder(SMARTAgentDecoder):
    """catk's agent decoder for heads over continuous 0.5 s actions (5 waypoints x (x, y, yaw), normalised per type).

    The token head is replaced by Identity, so forward() returns the hidden state as "next_token_logits"; the chosen
    action is embedded with catk's own per-type contour MLPs (embed_next_action). Subclasses add the head and
    implement sample_action(h, agent_type) -> normalised action [A, 15].
    """

    def __init__(self, cfg: "ModelConfig", **kwargs) -> None:
        super().__init__(**kwargs)  # builds (and initialises) the token head too: same RNG use as the token arm
        self.token_predict_head = nn.Identity()
        stats = json.loads(ACTION_STATS.read_text())
        self.register_buffer("act_mean", torch.tensor([stats[k]["mean"] for k in ("veh", "ped", "cyc")]))
        self.register_buffer("act_std", torch.tensor([stats[k]["std"] for k in ("veh", "ped", "cyc")]))

    def normalize(self, action: Tensor, agent_type: Tensor) -> Tensor:
        return (action - self.act_mean[agent_type]) / self.act_std[agent_type]

    def denormalize(self, x: Tensor, agent_type: Tensor) -> Tensor:
        return x * self.act_std[agent_type] + self.act_mean[agent_type]

    def sample_next_action(self, feat_a_now, tokenized_agent, sampling_scheme, pos_now, head_now, n_step):
        agent_type = tokenized_agent["type"].long()
        x = self.sample_action(feat_a_now, agent_type)
        action = self.denormalize(x, agent_type)
        traj_all = waypoint_contours(action, tokenized_agent["token_agent_shape"])
        idx = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        return x, idx, traj_all

    def embed_next_action(self, next_token_idx, next_token_traj_all, agent_token_emb_tables, type_masks):
        contour = next_token_traj_all[:, -1].flatten(1, 2)  # end-pose contour in the start frame, [A, 8]
        emb = contour.new_zeros(contour.shape[0], self.hidden_dim)
        for mlp, mask in zip((self.token_emb_veh, self.token_emb_ped, self.token_emb_cyc), type_masks):
            if mask.any():
                emb[mask] = mlp(contour[mask]).to(emb.dtype)
        return emb


class FlowAgentDecoder(ContinuousAgentDecoder):
    """Flow-matching head over the continuous action (Euler sampler)."""

    def __init__(self, cfg: "ModelConfig", **kwargs) -> None:
        super().__init__(cfg, **kwargs)
        self.flow_head = FlowHead(cond_dim=self.hidden_dim, action_dim=ACTION_DIM, width=cfg.flow_width,
                                  depth=cfg.flow_depth)
        self.flow_steps = cfg.flow_steps
        self.flow_noise_scale = cfg.flow_noise_scale

    @property
    def action_head(self):
        return self.flow_head

    def sample_action(self, h, agent_type):
        return self.flow_head.sample(h, n_steps=self.flow_steps, noise_scale=self.flow_noise_scale)


class BinsAgentDecoder(ContinuousAgentDecoder):
    """Per-coordinate uniform-bins head over the same continuous action (factorised categorical, optional CFG)."""

    def __init__(self, cfg: "ModelConfig", **kwargs) -> None:
        super().__init__(cfg, **kwargs)
        self.bins_head = BinsHead(cond_dim=self.hidden_dim, action_dim=ACTION_DIM, n_bins=cfg.bins_n,
                                  r=cfg.bins_range, width=cfg.bins_width, p_uncond=cfg.bins_p_uncond)
        self.bins_cfg = cfg.bins_cfg

    @property
    def action_head(self):
        return self.bins_head

    def sample_action(self, h, agent_type):
        return self.bins_head.sample(h, cfg_weight=self.bins_cfg)


class SimAgent(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        sampling = OmegaConf.create({"num_k": 1, "temp": 1.0})
        self.token_processor = TokenProcessor(
            map_token_file="map_traj_token5.pkl", agent_token_file="agent_vocab_555_s2.pkl",
            map_token_sampling=sampling, agent_token_sampling=sampling)
        self.map_encoder = SMARTMapDecoder(
            hidden_dim=cfg.hidden_dim, pl2pl_radius=cfg.pl2pl_radius, num_freq_bands=cfg.num_freq_bands,
            num_layers=cfg.num_map_layers, num_heads=cfg.num_heads, head_dim=cfg.head_dim, dropout=cfg.dropout)
        dec_kwargs = dict(
            hidden_dim=cfg.hidden_dim, num_historical_steps=cfg.num_historical_steps,
            num_future_steps=cfg.num_future_steps, time_span=cfg.time_span, pl2a_radius=cfg.pl2a_radius,
            a2a_radius=cfg.a2a_radius, num_freq_bands=cfg.num_freq_bands, num_layers=cfg.num_agent_layers,
            num_heads=cfg.num_heads, head_dim=cfg.head_dim, dropout=cfg.dropout,
            hist_drop_prob=cfg.hist_drop_prob, n_token_agent=self.token_processor.n_token_agent)
        if cfg.head == "token":
            self.agent_encoder = SMARTAgentDecoder(**dec_kwargs)
        elif cfg.head == "flow":
            self.agent_encoder = FlowAgentDecoder(cfg, **dec_kwargs)
        elif cfg.head == "bins":
            self.agent_encoder = BinsAgentDecoder(cfg, **dec_kwargs)
        else:
            raise ValueError(cfg.head)
        self.sampling = OmegaConf.create({"criterium": "topk_prob", "num_k": cfg.token_topk, "temp": cfg.token_temp})

    # ----- inputs -----
    def _tokenize(self, data):
        """catk tokenisation; for the flow arm also swap in raw GT poses and continuous action contours."""
        seg = None
        # catk's tokenizer edits valid_mask/heading/velocity in place: give it copies so a batch can be reused
        data = copy.copy(data)
        for k in ("valid_mask", "heading", "velocity"):
            data["agent"][k] = data["agent"][k].clone()
        if self.cfg.head != "token":
            traj = clean_trajectories(data["agent"])
        tokenized_map, tokenized_agent = self.token_processor(data)
        if self.cfg.head != "token":
            seg = segment_features(traj, tokenized_agent["token_agent_shape"])
            n_agent, n_step = seg["in_valid"].shape
            ta = dict(tokenized_agent)
            idx = torch.arange(n_agent * n_step, device=seg["pos2"].device).view(n_agent, n_step)
            for k in ("veh", "ped", "cyc"):
                ta[f"trajectory_token_{k}"] = seg["in_contour"].flatten(0, 1)  # one "token" per agent-step
            ta.update(sampled_idx=idx, gt_idx=idx, sampled_pos=seg["pos2"], gt_pos=seg["pos2"],
                      sampled_heading=seg["head2"], gt_heading=seg["head2"])
            tokenized_agent = ta
        return tokenized_map, tokenized_agent, seg

    # ----- training -----
    def loss(self, data, agent_weight: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """Behaviour-cloning loss. "sum" is the (optionally per-agent weighted) sum of per-target losses and "count"
        the number of targets; "loss" = sum / count."""
        tokenized_map, tokenized_agent, seg = self._tokenize(data)
        map_feature = self.map_encoder(tokenized_map)
        pred = self.agent_encoder(tokenized_agent, map_feature)
        train_mask = data["agent"]["train_mask"]
        if self.cfg.head == "token":
            # catk CrossEntropy (use_gt_raw=True, no distance filter): target = token closest to the raw GT
            # next pose, seen from the tokenised current pose.
            target, target_valid = get_euclidean_targets(
                pred_pos=pred["pred_pos"], pred_head=pred["pred_head"], pred_valid=pred["pred_valid"],
                gt_pos=pred["gt_pos_raw"], gt_head=pred["gt_head_raw"], gt_valid=pred["gt_valid_raw"])
            prob = get_prob_targets(target, tokenized_agent["token_agent_shape"], tokenized_agent["token_traj"])
            ce = F.cross_entropy(pred["next_token_logits"].transpose(1, 2), prob.transpose(1, 2),
                                 reduction="none", label_smoothing=self.cfg.label_smoothing)  # [A, 16]
            mask = pred["next_token_valid"] & target_valid & train_mask[:, None]
            w = mask.float() if agent_weight is None else mask.float() * agent_weight[:, None]
            total = (ce * w).sum()
            return {"loss": total / mask.sum().clamp(min=1), "sum": total, "count": mask.sum()}
        h = pred["next_token_logits"]  # [A, 16, hidden]: steps 1..16, actions (10->15) .. (85->90)
        mask = pred["next_token_valid"] & seg["target_valid"][:, 1:-1] & train_mask[:, None]
        agent_type = tokenized_agent["type"].long()[:, None].expand_as(mask)[mask]
        x1 = self.agent_encoder.normalize(seg["target"][:, 1:-1][mask], agent_type)
        per = self.agent_encoder.action_head.loss(h[mask], x1, n_noise=self.cfg.flow_noise_samples)
        if agent_weight is not None:
            per = per * agent_weight[:, None].expand_as(mask)[mask]
        return {"loss": per.sum() / mask.sum().clamp(min=1), "sum": per.sum(), "count": mask.sum()}

    # ----- policy-gradient inputs -----
    def policy_inputs(self, data) -> Dict[str, Tensor]:
        """Teacher-forced pass over (rollout) scenes for policy-gradient updates, all agents, steps 1..16.

        token: "logits" [A, 16, V] and "action" [A, 16] = the token the step's motion re-tokenises to (for a
        rollout, the sampled token); flow: "h" [A, 16, H] and "action" [A, 16, 15] = the normalised action.
        "mask" [A, 16] marks steps with a valid start and end. Call in eval mode to switch off dropout.
        """
        tokenized_map, tokenized_agent, seg = self._tokenize(data)
        pred = self.agent_encoder(tokenized_agent, self.map_encoder(tokenized_map))
        if self.cfg.head == "token":
            target, target_valid = get_euclidean_targets(
                pred_pos=pred["pred_pos"], pred_head=pred["pred_head"], pred_valid=pred["pred_valid"],
                gt_pos=pred["gt_pos_raw"], gt_head=pred["gt_head_raw"], gt_valid=pred["gt_valid_raw"])
            prob = get_prob_targets(target, tokenized_agent["token_agent_shape"], tokenized_agent["token_traj"])
            return {"logits": pred["next_token_logits"], "action": prob.argmax(-1),
                    "mask": pred["next_token_valid"] & target_valid}
        agent_type = tokenized_agent["type"].long()[:, None].expand(-1, 16)
        x = self.agent_encoder.normalize(seg["target"][:, 1:-1], agent_type)
        return {"h": pred["next_token_logits"], "action": x,
                "mask": pred["next_token_valid"] & seg["target_valid"][:, 1:-1]}

    # ----- closed-loop rollout -----
    @torch.no_grad()
    def rollout(self, data, n_rollouts: int = 1) -> Dict[str, Tensor]:
        """Closed-loop rollouts of all agents for a batch of scenarios (80 steps at 10 Hz).

        Tokenises and encodes the map once, then samples n_rollouts independent rollouts.
        Returns pos [A, R, 80, 2], head [A, R, 80], z [A, R, 80].
        """
        tokenized_map, tokenized_agent, _ = self._tokenize(data)
        map_feature = self.map_encoder(tokenized_map)
        outs = [self.agent_encoder.inference(tokenized_agent, map_feature, self.sampling) for _ in range(n_rollouts)]
        return {"pos": torch.stack([o["pred_traj_10hz"] for o in outs], dim=1),
                "head": torch.stack([o["pred_head_10hz"] for o in outs], dim=1),
                "z": torch.stack([o["pred_z_10hz"] for o in outs], dim=1)}


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def head_params(model: SimAgent) -> int:
    enc = model.agent_encoder
    return count_params(enc.token_predict_head) if model.cfg.head == "token" else count_params(enc.action_head)
