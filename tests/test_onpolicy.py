"""Unit tests for the on-policy likelihood pieces: the token log-ratio support and the FPO per-draw losses (CPU)."""
import weakref

import pytest
import torch

from tokens_vs_flows.heads import FlowHead
from tokens_vs_flows.onpolicy import flow_draw_losses, retry_in_parts, token_logp, velocity_anchor


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


def test_velocity_anchor_holds_the_field_near_t0_where_the_eps_loss_does_not():
    head, ref = _trained_like_head(), _trained_like_head()
    with torch.no_grad():
        for p in ref.parameters():
            p.add_(0.05 * torch.randn_like(p))  # a reference field that differs from the policy's
    h, x1 = torch.randn(6, 16), torch.randn(6, 15)
    t, eps = torch.full((6, 8), 1e-3), torch.randn(6, 8, 15)  # draws at the start of sampling
    loss, v = flow_draw_losses(head, h, x1, t, eps, return_v=True)
    assert torch.equal(loss, flow_draw_losses(head, h, x1, t, eps))
    v_ref = flow_draw_losses(ref, h, x1, t, eps, return_v=True)[1].detach()
    assert torch.equal(velocity_anchor(v_ref, v_ref), torch.zeros(6))
    grad = lambda f: torch.cat([g.flatten() for g in torch.autograd.grad(f, list(head.parameters()), retain_graph=True)])
    g_eps, g_anchor = grad(loss.mean(-1).sum()), grad(velocity_anchor(v, v_ref).sum())
    assert g_eps.norm() < 1e-4 * g_anchor.norm()  # t^2 weighting: the likelihood barely sees the field near t = 0


def test_x1_form_anchor_weights_draws_by_one_minus_t_squared():
    v, v_ref = torch.randn(4, 3, 15), torch.randn(4, 3, 15)
    t = torch.tensor([0.0, 0.5, 1.0]).expand(4, 3)
    d = (v - v_ref).pow(2).sum(-1)
    assert torch.allclose(velocity_anchor(v, v_ref, t), (d * torch.tensor([1.0, 0.25, 0.0])).mean(-1))
    assert torch.allclose(velocity_anchor(v, v_ref, torch.zeros(4, 3)), velocity_anchor(v, v_ref))  # = uniform at t = 0


def test_retry_in_parts_frees_the_failed_attempt_before_retrying():
    attempts = []

    def fn(chunks):
        activations = torch.zeros(1000)  # stands for the attempt's activations and graph
        attempts.append(weakref.ref(activations))
        if len(chunks) == 1:
            raise torch.cuda.OutOfMemoryError("forced")
        assert attempts[0]() is None  # released before the smaller attempt started
        return [len(c) for c in chunks]

    assert retry_in_parts(fn, list(range(4))) == ([2, 2], 2)


def test_retry_in_parts_gives_up_at_one_scene_per_part():
    calls = []

    def fn(chunks):
        calls.append(len(chunks))
        raise torch.cuda.OutOfMemoryError("forced")

    with pytest.raises(torch.cuda.OutOfMemoryError):
        retry_in_parts(fn, list(range(4)))
    assert calls == [1, 2, 4]
