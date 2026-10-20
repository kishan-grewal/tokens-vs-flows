"""Model-free rollouts from the logged scenario: ground-truth replay, constant velocity, random actions."""
import torch

from wosac import DT, N_ROLLOUTS

N_FUTURE = 80


def sim_agents(gt: dict):
    """Logged tracks of the simulated agents (WOSAC sim agents, valid at the current step), in sim-id order."""
    ids = gt["object_ids"].long().cpu()
    sim = gt["sim_agent_ids"].long().cpu()
    idx = (sim[:, None] == ids[None]).float().argmax(1)
    return sim, gt["tracks"].cpu()[idx].double(), gt["track_masks"].cpu()[idx]


def gt_replay(gt: dict, n_rollouts: int = N_ROLLOUTS) -> dict:
    """The log itself, repeated. Simulated objects always count as present, so at steps where an object is
    missing from the log it is parked far outside the scene (each at its own spot): it then interacts with
    nothing, as in the logged features, which ignore invalid objects."""
    sim, tr, m = sim_agents(gt)
    cur = int(gt["current_time_index"])
    st = tr[..., [0, 1, 2, 6]].clone()
    far = torch.stack([1e5 + 100.0 * torch.arange(len(sim), dtype=st.dtype), torch.full((len(sim),), 1e5,
                                                                                         dtype=st.dtype)], dim=-1)
    st[..., :2] = torch.where(m[..., None], st[..., :2], far[:, None, :])
    fut = st[:, cur + 1:cur + 1 + N_FUTURE].float()
    return {"agent_id": sim, "states": fut[None].repeat(n_rollouts, 1, 1, 1)}


def constant_velocity(gt: dict, n_rollouts: int = N_ROLLOUTS) -> dict:
    """WOSAC constant-velocity agent: extrapolate at the last speed (from the last two history steps, zero if
    either is invalid) along the last heading (deterministic, repeated)."""
    sim, tr, m = sim_agents(gt)
    cur = int(gt["current_time_index"])
    c = tr[:, cur]
    speed = torch.where(m[:, cur] & m[:, cur - 1], (tr[:, cur, :2] - tr[:, cur - 1, :2]).norm(dim=-1) / DT,
                        torch.zeros_like(c[:, 0]))
    k = torch.arange(1, N_FUTURE + 1, dtype=torch.float64) * DT
    x = c[:, 0, None] + (speed * c[:, 6].cos())[:, None] * k
    y = c[:, 1, None] + (speed * c[:, 6].sin())[:, None] * k
    z = c[:, 2, None].expand_as(x)
    h = c[:, 6, None].expand_as(x)
    st = torch.stack([x, y, z, h], dim=-1).float()
    return {"agent_id": sim, "states": st[None].repeat(n_rollouts, 1, 1, 1)}


def random_agent(gt: dict, n_rollouts: int = N_ROLLOUTS, seed: int = 0) -> dict:
    """WOSAC random agent: every step of every agent at x, y, heading ~ N(1.0, 0.1^2) in the AV's frame at the
    current step (z kept at the agent's current height)."""
    sim, tr, _ = sim_agents(gt)
    cur = int(gt["current_time_index"])
    av = gt["tracks"][int(gt["sdc_track_index"]), cur].cpu().double()
    g = torch.Generator().manual_seed(seed)
    loc = 1.0 + 0.1 * torch.randn((n_rollouts, tr.shape[0], N_FUTURE, 3), generator=g, dtype=torch.float64)
    c, s = av[6].cos(), av[6].sin()
    x = av[0] + c * loc[..., 0] - s * loc[..., 1]
    y = av[1] + s * loc[..., 0] + c * loc[..., 1]
    h = av[6] + loc[..., 2]
    z = tr[:, cur, 2][None, :, None].expand_as(x)
    return {"agent_id": sim, "states": torch.stack([x, y, z, h], dim=-1).float()}


def noisy_actions(gt: dict, n_rollouts: int = N_ROLLOUTS, seed: int = 0) -> dict:
    """Noisy constant velocity (diagnostic baseline, not part of the sanity pass/fail): unicycle from the current
    state driven by i.i.d. uniform actions every 0.1 s, acceleration ~ U[-4, 4] m/s^2, yaw rate ~ U[-0.5, 0.5] rad/s,
    speed clipped at 0. It shows how much realism model-free spread alone earns; report it next to constant
    velocity."""
    sim, tr, _ = sim_agents(gt)
    c = tr[:, int(gt["current_time_index"])]
    g = torch.Generator().manual_seed(seed)
    shape = (n_rollouts, c.shape[0], N_FUTURE)
    acc = (torch.rand(shape, generator=g, dtype=torch.float64) * 2 - 1) * 4.0
    yaw_rate = (torch.rand(shape, generator=g, dtype=torch.float64) * 2 - 1) * 0.5
    speed = (torch.hypot(c[:, 7], c[:, 8])[None, :, None] + torch.cumsum(acc * DT, dim=-1)).clamp(min=0)
    h = c[:, 6][None, :, None] + torch.cumsum(yaw_rate * DT, dim=-1)
    x = c[:, 0][None, :, None] + torch.cumsum(speed * h.cos() * DT, dim=-1)
    y = c[:, 1][None, :, None] + torch.cumsum(speed * h.sin() * DT, dim=-1)
    z = c[:, 2][None, :, None].expand_as(x)
    return {"agent_id": sim, "states": torch.stack([x, y, z, h], dim=-1).float()}


BASELINES = {"gt_replay": gt_replay, "constant_velocity": constant_velocity, "random_agent": random_agent,
             "noisy_actions": noisy_actions}
