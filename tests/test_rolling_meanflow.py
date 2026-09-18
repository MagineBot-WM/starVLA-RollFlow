import math

import pytest
import torch
from torch import nn

from starVLA.model.modules.action_model.rolling_meanflow_matching_head.rolling_meanflow import (
    RollFlow,
    RollFlowConfig,
    StaircaseTimeSampler,
    _validate_times,
    central_difference_lsd,
    masked_mse,
    ot_match,
)


def _config(**overrides):
    values = {
        "horizon": 8,
        "action_dim": 1,
        "chunk_size": 1,
        "finite_difference_delta": 0.01,
        "inference_steps": 4,
        "p_k1": 1.0,
        "p_fm": 0.0,
        "fm_only_steps": 0,
        "mf_kv": 1.0,
        "mf_weight": 0.1,
        "mf_loss_threshold": 0.5,
        "use_ot": False,
        "iterative_cold_start": True,
    }
    values.update(overrides)
    return RollFlowConfig(**values)


class _LinearPathOracle(nn.Module):
    """Exact interval and instantaneous velocity for the linear FM path."""

    def forward(self, z, source_time, target_time, context, **kwargs):
        del target_time, kwargs
        return (context - z) / (1.0 - source_time)


class _LearnableConstant(nn.Module):
    def __init__(self):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(0.0))
        self.batch_sizes = []
        self.grad_enabled = []

    def forward(self, z, source_time, target_time, context, **kwargs):
        del source_time, target_time, context, kwargs
        self.batch_sizes.append(z.shape[0])
        self.grad_enabled.append(torch.is_grad_enabled())
        return self.value.expand_as(z)


class _GapVelocity(nn.Module):
    """u(z,s,t)=value+slope*(t-s), useful for checking MeanFlow correction."""

    def __init__(self, value=0.0, slope=1.0):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(float(value)))
        self.slope = nn.Parameter(torch.tensor(float(slope)))

    def forward(self, z, source_time, target_time, context, **kwargs):
        del context, kwargs
        gap = target_time - source_time
        return (self.value + self.slope * gap).expand_as(z)


def test_action_loss_weights_are_mean_normalized_and_preserve_scale():
    cfg = _config(action_dim=3, action_loss_weights=[2.0, 1.0, 0.0])
    assert cfg.action_loss_weights == pytest.approx((2.0, 1.0, 0.0))
    error = torch.ones(2, 4, 3)
    weighted = masked_mse(error, action_loss_weights=torch.tensor(cfg.action_loss_weights))
    assert weighted == pytest.approx(1.0)


def test_action_loss_weights_validate_dimension_and_positive_sum():
    with pytest.raises(ValueError, match="one entry per action dimension"):
        _config(action_dim=3, action_loss_weights=[1.0, 1.0])
    with pytest.raises(ValueError, match="positive value"):
        _config(action_dim=3, action_loss_weights=[0.0, 0.0, 0.0])


def test_training_times_are_valid_for_every_divisor_k():
    sampler = StaircaseTimeSampler(_config())
    for k in (1, 2, 4, 8):
        times = sampler.sample_training(5, device="cpu", p_fm=0.0, num_time_groups=k)
        assert torch.all(times.s >= 0)
        assert torch.all(times.s <= times.t)
        assert torch.all(times.t <= 1)
        assert torch.all(times.t[times.active] + 0.01 <= 1.0 + 1e-6)
        assert times.num_time_groups == k
        assert times.block_size == 8 // k
        assert times.ratio.shape == (5, 1, 1)
        torch.testing.assert_close(times.s, times.ratio * times.t)


def test_sampled_active_times_pass_validation_at_float_boundaries():
    cfg = _config(horizon=32, chunk_size=8, p_k1=0.7)
    sampler = StaircaseTimeSampler(cfg)
    generator = torch.Generator().manual_seed(123)
    for _ in range(100):
        times = sampler.sample_training(32, device="cpu", generator=generator, p_fm=0.0)
        _validate_times(times.s, times.t, cfg.finite_difference_delta, times.active)


def test_k_sampling_probabilities_and_config_validation():
    generator = torch.Generator().manual_seed(0)
    sampler = StaircaseTimeSampler(_config(p_k1=0.7))
    counts = {1: 0, 2: 0, 4: 0, 8: 0}
    for _ in range(2000):
        counts[sampler._sample_k(torch.device("cpu"), generator)] += 1
    assert counts[1] / 2000 == pytest.approx(0.7, abs=0.04)
    for k in (2, 4, 8):
        assert counts[k] / 2000 == pytest.approx(0.1, abs=0.035)
    with pytest.raises(ValueError, match="p_fm"):
        _config(p_fm=1.1)
    with pytest.raises(ValueError, match="mf_kv"):
        _config(mf_kv=float("nan"))
    with pytest.raises(ValueError, match="mf_loss_threshold"):
        _config(mf_loss_threshold=-1.0)


def test_fm_only_warmup_then_fixed_mixture():
    torch.manual_seed(0)
    rollflow = RollFlow(_config(p_fm=0.7, fm_only_steps=10))
    model = _LearnableConstant()
    actions = torch.randn(32, 8, 1)

    _, start = rollflow.loss(model, actions, step=0)
    assert start["fm_only"] == 1.0
    assert start["fm_only_steps"] == 10
    assert start["p_fm"] == 1.0
    assert start["fm_only_frac"] == 1.0
    assert start["active_mf_frac"] == 0.0
    assert model.batch_sizes == [32]
    assert start["mf_loss"] == 0.0

    _, warmup_end = rollflow.loss(model, actions, step=9)
    assert warmup_end["p_fm"] == 1.0
    _, fixed = rollflow.loss(model, actions, step=10)
    assert fixed["fm_only"] == 0.0
    assert fixed["p_fm"] == 0.7


def test_fm_fast_path_retains_full_fm_supervision():
    model = _LearnableConstant()
    x0 = torch.zeros(2, 8, 1)
    x1 = torch.ones_like(x0)
    times = torch.full_like(x0, 0.5)
    loss, stats = central_difference_lsd(model, x0, x1, times, times, delta=0.01)
    assert loss.requires_grad
    assert loss.item() == pytest.approx(1.0)
    assert stats["mf_loss"] == 0.0
    loss.backward()
    assert model.batch_sizes == [2]
    assert model.value.grad.item() == pytest.approx(-2.0)


def test_zero_meanflow_weight_skips_interval_forwards():
    model = _LearnableConstant()
    x0 = torch.zeros(1, 4, 1)
    x1 = torch.ones_like(x0)
    s, t = torch.full_like(x0, 0.2), torch.full_like(x0, 0.6)
    loss, stats = central_difference_lsd(
        model, x0, x1, s, t, delta=0.01, mf_weight=0.0
    )
    assert loss.requires_grad
    assert stats["mf_loss"] == 0.0
    assert model.batch_sizes == [1]


def test_linear_path_oracle_has_zero_fm_and_meanflow_error():
    torch.manual_seed(0)
    cfg = _config()
    x0 = torch.randn(3, cfg.horizon, cfg.action_dim)
    x1 = torch.randn_like(x0)
    times = StaircaseTimeSampler(cfg).sample_training(
        x0.shape[0], device=x0.device, p_fm=0.0, num_time_groups=4
    )
    loss, stats = central_difference_lsd(
        _LinearPathOracle(),
        x0,
        x1,
        times.s,
        times.t,
        delta=cfg.finite_difference_delta,
        active=times.active,
        context=x1,
        mf_kv=cfg.mf_kv,
        mf_weight=cfg.mf_weight,
    )
    assert loss < 1e-8
    assert stats["fm_loss"] < 1e-10
    assert stats["mf_loss"] < 1e-7


def test_meanflow_derivative_is_clamped_and_loss_is_finite():
    model = _GapVelocity(value=0.0, slope=10.0)
    x0 = torch.zeros(1, 4, 1)
    x1 = torch.ones_like(x0)
    s, t = torch.full_like(x0, 0.2), torch.full_like(x0, 0.6)
    loss, stats = central_difference_lsd(
        model, x0, x1, s, t, delta=0.01, mf_kv=0.5, mf_weight=0.5
    )
    # du/dt=slope=10 is clipped to kv=0.5 before forming V_mf.
    assert torch.isfinite(loss)
    loss.backward()
    assert model.value.grad is not None and torch.isfinite(model.value.grad)
    assert model.slope.grad is not None and torch.isfinite(model.slope.grad)


def test_meanflow_gate_drops_abnormal_batch_and_keeps_fm_gradient():
    model = _GapVelocity(value=0.0, slope=10.0)
    x0 = torch.zeros(1, 4, 1)
    x1 = torch.ones_like(x0)
    s, t = torch.full_like(x0, 0.2), torch.full_like(x0, 0.6)
    loss, stats = central_difference_lsd(
        model,
        x0,
        x1,
        s,
        t,
        delta=0.01,
        mf_kv=0.5,
        mf_weight=0.5,
        mf_loss_threshold=0.5,
    )
    assert stats["mf_loss"] > 0.5
    assert stats["mf_loss_weighted"] == pytest.approx(0.0)
    assert stats["mf_gate_active"] == pytest.approx(0.0)
    loss.backward()
    # Diagonal FM depends on value, while the gated-off MF branch is the only
    # path that can update the interval slope.
    assert model.value.grad is not None and torch.isfinite(model.value.grad)
    assert model.slope.grad is not None and model.slope.grad.item() == pytest.approx(0.0)


def test_loss_keeps_interval_prediction_graph_and_detaches_correction():
    rollflow = RollFlow(_config(p_fm=0.0))
    model = _LearnableConstant()
    actions = torch.randn(3, 8, 1)
    loss, stats = rollflow.loss(model, actions)
    loss.backward()
    assert model.batch_sizes == [6, 6]
    assert model.grad_enabled == [True, False]
    assert model.value.grad is not None and torch.isfinite(model.value.grad)
    assert stats["active_mf_frac"] > 0


def test_padded_rows_are_excluded_from_interval_loss():
    model = _LearnableConstant()
    x0 = torch.randn(4, 8, 1)
    x1 = torch.randn_like(x0)
    t = torch.full((4, 8, 1), 0.6)
    s = t.clone()
    s[[1, 3]] = 0.2
    pad = torch.zeros(4, 8, dtype=torch.bool)
    pad[3] = True
    loss, stats = central_difference_lsd(model, x0, x1, s, t, delta=0.01, pad=pad)
    loss.backward()
    assert model.batch_sizes == [8, 8]
    assert stats["active_mf_frac"] == 0.25


def test_bfloat16_inference_preserves_float32_time_grid():
    class TimeRecorder(_LearnableConstant):
        def forward(self, z, s, t, *args, **kwargs):
            self.times = (s, t)
            return super().forward(z, s, t, *args, **kwargs)

    model = TimeRecorder()
    flow = RollFlow(_config(inference_steps=3, iterative_cold_start=False))
    output = flow.step(model, 1, device="cpu", dtype=torch.bfloat16)
    s, t = model.times
    assert output.dtype == torch.bfloat16
    assert s.dtype == t.dtype == torch.float32
    torch.testing.assert_close(t[0, :3, 0], torch.tensor([1.0, 2 / 3, 1 / 3]))


def test_inference_velocity_mode_switches_query_times():
    class TimeRecorder(_LearnableConstant):
        def forward(self, z, s, t, *args, **kwargs):
            self.times = (s, t)
            return super().forward(z, s, t, *args, **kwargs)

    cfg = _config(iterative_cold_start=False)
    average_model = TimeRecorder()
    RollFlow(cfg).step(average_model, 1, device="cpu", velocity_mode="average")
    average_s, average_t = average_model.times
    assert not torch.equal(average_s, average_t)

    instant_model = TimeRecorder()
    RollFlow(cfg).step(instant_model, 1, device="cpu", velocity_mode="instant")
    instant_s, instant_t = instant_model.times
    torch.testing.assert_close(instant_s, instant_t)


def test_training_input_shapes_and_parameters_are_checked():
    x0 = torch.randn(2, 8, 1)
    x1 = torch.randn_like(x0)
    times = torch.zeros(2, 8, 1)
    with pytest.raises(ValueError, match="x0 and x1"):
        central_difference_lsd(_LearnableConstant(), x0, x1[:, :-1], times, times, delta=0.01)
    with pytest.raises(ValueError, match="s and t"):
        central_difference_lsd(
            _LearnableConstant(), x0, x1, times.squeeze(-1), times.squeeze(-1), delta=0.01
        )
    with pytest.raises(ValueError, match="delta"):
        central_difference_lsd(_LearnableConstant(), x0, x1, times, times, delta=0.0)
    with pytest.raises(ValueError, match="mf_weight"):
        _config(mf_weight=-1.0)


def test_all_padding_returns_graph_connected_zero():
    rollflow = RollFlow(_config())
    model = _LearnableConstant()
    actions = torch.randn(2, 8, 1)
    loss, _ = rollflow.loss(model, actions, pad=torch.ones(2, 8, dtype=torch.bool))
    loss.backward()
    assert loss.item() == 0.0
    assert model.value.grad is not None and model.value.grad.item() == 0.0
    assert model.batch_sizes == [2]


def test_ot_matching_ignores_padded_target_values():
    x1 = torch.tensor([[[0.0], [1000.0]], [[10.0], [-1000.0]]])
    x0 = torch.tensor([[[0.0], [-1000.0]], [[10.0], [1000.0]]])
    pad = torch.tensor([[False, True], [False, True]])
    torch.testing.assert_close(ot_match(x1, x0, pad=pad), x0)


def test_oracle_rolls_sine_and_reset():
    cfg = _config()
    rollflow = RollFlow(cfg)
    oracle = _LinearPathOracle()
    phase_step = math.pi / 16.0
    phases = torch.linspace(-20.0 * math.pi, 20.0 * math.pi, 641)
    offsets = torch.arange(cfg.horizon) * phase_step
    predictions = []
    for phase in phases:
        target_chunk = torch.sin(phase + offsets)[None, :, None]
        predictions.append(rollflow.step(oracle, batch=1, context=target_chunk)[0, 0, 0])
    torch.testing.assert_close(torch.stack(predictions), torch.sin(phases), atol=2e-5, rtol=2e-5)
    assert rollflow.cache_info.shape == (1, 8, 1)
    assert rollflow.cache_info.refinement_steps == 4
    rollflow.reset()
    assert rollflow.cache_info is None
