"""Unit tests for the in-action acceleration used by the flow straightness diagnostic (CPU)."""
import torch

from tokens_vs_flows.flow_straightness import in_action_accel


def _action(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """[N, 5] waypoint coordinates at t = 0.1 ... 0.5 s -> [N, 15] action (yaw 0)."""
    return torch.stack([x, y, torch.zeros_like(x)], dim=-1).flatten(1)


def test_constant_velocity_has_no_acceleration():
    t = torch.arange(1, 6) * 0.1
    a = in_action_accel(_action((8.0 * t)[None], (-3.0 * t)[None]))
    assert a.shape == (1, 4, 2)
    assert torch.allclose(a, torch.zeros_like(a), atol=1e-4)


def test_constant_acceleration_is_recovered_from_the_origin():
    t = torch.arange(1, 6, dtype=torch.float64) * 0.1
    a = in_action_accel(_action((5.0 * t + 0.5 * 2.0 * t ** 2)[None], (0.5 * -1.5 * t ** 2)[None]))
    assert torch.allclose(a, torch.tensor([2.0, -1.5], dtype=torch.float64).expand(1, 4, 2), atol=1e-9)
