"""Unit tests for the flow-matching head (CPU, no data needed)."""
import torch

from tokens_vs_flows.heads import FlowHead


def test_shapes_and_finite_loss():
    torch.manual_seed(0)
    head = FlowHead(cond_dim=64)
    h, x1 = torch.randn(10, 64), torch.randn(10, 15)
    per = head.loss(h, x1, n_noise=4)
    assert per.shape == (10,) and torch.isfinite(per).all()
    x = head.sample(h, n_steps=10)
    assert x.shape == (10, 15) and torch.isfinite(x).all()


def test_deterministic_under_seed():
    head = FlowHead(cond_dim=64)
    h = torch.randn(5, 64)
    torch.manual_seed(1)
    a = head.sample(h)
    torch.manual_seed(1)
    b = head.sample(h)
    assert torch.equal(a, b)


def test_untrained_head_returns_noise():
    """The last layer is zero-initialised: before training, samples equal the (scaled) initial noise."""
    head = FlowHead(cond_dim=64)
    h, x0 = torch.randn(3, 64), torch.randn(3, 15)
    assert torch.allclose(head.sample(h, x0=x0, noise_scale=0.5), 0.5 * x0)


def test_fits_a_deterministic_map():
    """With a deterministic target per condition, the CFM loss can approach zero and samples land on target."""
    torch.manual_seed(0)
    head = FlowHead(cond_dim=8, action_dim=3, width=64, depth=3)
    h = torch.randn(16, 8)
    x1 = torch.randn(16, 3)
    opt = torch.optim.Adam(head.parameters(), lr=3e-3)
    for _ in range(1500):
        loss = head.loss(h, x1, n_noise=8).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    err = (head.sample(h, n_steps=20) - x1).norm(dim=-1).mean()
    assert err < 0.25, err


def test_bins_roundtrip_and_cfg():
    from tokens_vs_flows.heads import BinsHead
    torch.manual_seed(0)
    head = BinsHead(cond_dim=64, action_dim=15, n_bins=256, r=5.0)
    x = torch.empty(100, 15).uniform_(-4.9, 4.9)
    assert (head.centres(head.to_bins(x)) - x).abs().max() <= 5.0 / 256 + 1e-6  # within half a bin
    h = torch.randn(7, 64)
    assert head.loss(h, x[:7]).shape == (7,)
    assert head.sample(h, cfg_weight=2.0).shape == (7, 15)
    # cfg weight 0 equals the conditional logits
    assert torch.allclose(head.guided_logits(h, 0.0), head.logits(h))
