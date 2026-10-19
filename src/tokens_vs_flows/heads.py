"""Continuous flow-matching action head (rectified-flow / conditional flow matching)."""
import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int, max_period: float = 1000.0) -> None:
        super().__init__()
        half = dim // 2
        self.register_buffer("freqs", torch.exp(-math.log(max_period) * torch.arange(half) / half), persistent=False)

    def forward(self, t: Tensor) -> Tensor:  # t: [N] in [0, 1]
        x = 1000.0 * t[:, None] * self.freqs[None]
        return torch.cat([x.cos(), x.sin()], dim=-1)


class FlowHead(nn.Module):
    """Velocity field v(x_t, t | h) over a normalised action x (dim 15 = 5 waypoints x (x, y, yaw)).

    MLP denoiser: [x_t, h, sinusoidal(t)] -> depth x (Linear(width) + SiLU) -> Linear(action_dim); with the defaults
    (width 256, depth 3, h of size 64) it has ~172k parameters, close to catk's token head (~137k).
    Convention (Lipman et al.): x_t = (1 - t) x0 + t x1, x0 ~ N(0, I) noise, x1 data, target v = x1 - x0.
    Sampling integrates dx/dt = v from t=0 to 1 with Euler steps. Runs in fp32.
    """

    def __init__(self, cond_dim: int, action_dim: int = 15, width: int = 256, depth: int = 3,
                 t_dim: int = 64) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.t_emb = SinusoidalEmbedding(t_dim)
        layers, d = [], action_dim + cond_dim + t_dim
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.SiLU()]
            d = width
        layers.append(nn.Linear(width, action_dim))
        self.mlp = nn.Sequential(*layers)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def velocity(self, x_t: Tensor, t: Tensor, h: Tensor) -> Tensor:
        return self.mlp(torch.cat([x_t, h, self.t_emb(t)], dim=-1))

    def loss(self, h: Tensor, x1: Tensor, n_noise: int = 1, generator: Optional[torch.Generator] = None) -> Tensor:
        """Per-sample CFM loss [N] (mean over action dims and noise draws)."""
        with torch.autocast(device_type=h.device.type, enabled=False):
            h, x1 = h.float(), x1.float()
            if n_noise > 1:
                h = h.repeat_interleave(n_noise, dim=0)
                x1 = x1.repeat_interleave(n_noise, dim=0)
            x0 = torch.randn(x1.shape, device=x1.device, generator=generator)
            t = torch.rand(x1.shape[0], device=x1.device, generator=generator)
            x_t = (1 - t[:, None]) * x0 + t[:, None] * x1
            err = (self.velocity(x_t, t, h) - (x1 - x0)).pow(2).mean(-1)
            return err.view(-1, n_noise).mean(-1)

    @torch.no_grad()
    def sample(self, h: Tensor, n_steps: int = 10, noise_scale: float = 1.0, x0: Optional[Tensor] = None,
               generator: Optional[torch.Generator] = None) -> Tensor:
        with torch.autocast(device_type=h.device.type, enabled=False):
            h = h.float()
            if x0 is None:
                x0 = torch.randn(h.shape[0], self.action_dim, device=h.device, generator=generator)
            x = x0.float() * noise_scale
            dt = 1.0 / n_steps
            for i in range(n_steps):
                t = torch.full((h.shape[0],), i * dt, device=h.device)
                x = x + dt * self.velocity(x, t, h)
            return x
