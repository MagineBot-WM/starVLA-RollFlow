import pytest
import torch

from starVLA.model.modules.action_model.rolling_meanflow_matching_head.meanflow_sampler import sample_mean_flow


@pytest.mark.parametrize("steps", [1, 4, 8, 16])
def test_uniform_forward_time_and_exact_endpoint(steps):
    noise = torch.randn(2, 32, 7, requires_grad=True)
    original = noise.detach().clone()
    target = torch.randn_like(noise)
    calls = []

    def oracle(x, s, t, context, *, scale):
        assert not torch.is_grad_enabled()
        assert torch.all(s == s[:, :1]) and torch.all(t == t[:, :1])
        assert torch.all(t > s)
        calls.append((s.clone(), t.clone()))
        return scale * (context - x) / (1 - s)

    result = sample_mean_flow(oracle, noise, steps, context=target, scale=1)
    torch.testing.assert_close(result, target)
    torch.testing.assert_close(noise.detach(), original)
    assert len(calls) == steps
    assert calls[0][0].eq(0).all() and calls[-1][1].eq(1).all()
    assert not result.requires_grad


@pytest.mark.parametrize("steps", [0, -1, 1.5, True])
def test_invalid_step_count(steps):
    with pytest.raises(ValueError, match="positive integer"):
        sample_mean_flow(None, torch.zeros(1, 32, 7), steps)


def test_velocity_clip_and_dtype():
    noise = torch.zeros(1, 4, 2, dtype=torch.bfloat16)

    def model(x, s, t, context):
        assert s.dtype == t.dtype == torch.float32
        return torch.full_like(x, 100)

    result = sample_mean_flow(model, noise, 4, clip_velocity=2)
    assert result.dtype == noise.dtype
    torch.testing.assert_close(result, torch.full_like(noise, 2))


def test_euler_queries_diagonal_but_advances_time():
    calls = []

    def velocity(x, s, t, context):
        torch.testing.assert_close(s, t)
        calls.append(s[0, 0, 0].item())
        return torch.ones_like(x) * 2

    result = sample_mean_flow(velocity, torch.zeros(2, 32, 7), 4, instantaneous=True)
    assert calls == [0, 0.25, 0.5, 0.75]
    torch.testing.assert_close(result, torch.full_like(result, 2))
