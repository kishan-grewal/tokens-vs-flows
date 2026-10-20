"""Unit tests for the on-policy likelihood pieces: the token log-ratio support and the FPO per-draw losses (CPU)."""
import torch

from tokens_vs_flows.heads import FlowHead
from tokens_vs_flows.onpolicy import flow_draw_losses, token_logp


def test_token_logp_is_the_softmax_over_the_support():
    torch.manual_seed(0)
    logits = torch.randn(5, 2048)
    support = logits.topk(48, dim=-1).indices
    lp = token_logp(logits, support[:, 3], support)
    assert torch.allclose(lp, torch.log_softmax(logits.gather(-1, support), -1)[:, 3])
    assert torch.isinf(token_logp(logits, logits.argmin(-1), support)).all()  # outside the support


def _trained_like_head():
    torch.manual_seed(0)
    head = FlowHead(cond_dim=16)
    for p in head.parameters():  # the last layer starts at zero; give it some weight
        torch.nn.init.normal_(p, std=0.05)
    return head


def test_flow_draw_losses_are_the_eps_form_loss():
    head = _trained_like_head()
    h, x1 = torch.randn(6, 16), torch.randn(6, 15)
    t, eps = torch.rand(6, 8), torch.randn(6, 8, 15)
    loss = flow_draw_losses(head, h, x1, t, eps)
    assert loss.shape == (6, 8)
    x_t = (1 - t)[..., None] * eps + t[..., None] * x1[:, None]
    v = head.velocity(x_t.reshape(48, 15), t.reshape(-1), h.repeat_interleave(8, 0)).view(6, 8, 15)
    eps_hat = x_t - t[..., None] * v  # the noise the velocity implies
    assert torch.allclose(loss, ((eps_hat - eps) ** 2).sum(-1), atol=1e-5)
    assert torch.equal(loss, flow_draw_losses(head, h, x1, t, eps))  # same draws, same parameters: ratio exactly 1


def test_fpo_ratio_rises_when_the_policy_moves_toward_the_action():
    head = _trained_like_head()
    h, x1 = torch.randn(4, 16), torch.randn(4, 15)
    t, eps = torch.rand(4, 8), torch.randn(4, 8, 15)
    logp_old = -flow_draw_losses(head, h, x1, t, eps).mean(-1).detach()
    opt = torch.optim.SGD(head.parameters(), lr=1e-2)
    (-(-flow_draw_losses(head, h, x1, t, eps).mean(-1)).sum()).backward()  # ascend log pi(a) = -mean loss
    opt.step()
    ratio = (-flow_draw_losses(head, h, x1, t, eps).mean(-1) - logp_old).exp()
    assert (ratio > 1).all()
