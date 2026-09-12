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


def test_action_loss_weights_are_mean_normalized_and_preserve_scale():
    cfg = _config(action_dim=3, action_loss_weights=[2.0, 1.0, 0.0])
    assert cfg.action_loss_weights == pytest.approx((2.0, 1.0, 0.0))
    error = torch.ones(2, 4, 3)
    weighted = masked_mse(
        error,
        action_loss_weights=torch.tensor(cfg.action_loss_weights),
    )
    assert weighted == pytest.approx(1.0)


def test_action_loss_weights_validate_dimension_and_positive_sum():
    with pytest.raises(ValueError, match="one entry per action dimension"):
        _config(action_dim=3, action_loss_weights=[1.0, 1.0])
    with pytest.raises(ValueError, match="positive value"):
        _config(action_dim=3, action_loss_weights=[0.0, 0.0, 0.0])


class _LinearPathOracle(nn.Module):
    """Exact average velocity from an arbitrary source point to context=x1."""

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


class _HighTangentResidual(nn.Module):
    """Produces a deliberately noisy tangent target for the LSD gate test."""

    def __init__(self):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(1.0))

    def forward(self, z, source_time, target_time, context, **kwargs):
        del z, context, kwargs
        gap = target_time - source_time
        return self.value * (1.0 + 100.0 * gap)


class _NonFiniteTangent(nn.Module):
    """Emits NaNs only on the packed tangent rows."""

    def __init__(self):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(0.0))

    def forward(self, z, source_time, target_time, context, **kwargs):
        del source_time, target_time, context, kwargs
        output = self.value.expand_as(z)
        if z.shape[0] == 3:  # B=1: [tangent-, tangent+, local-FM]
            output = output.clone()
            output[:2] = float("nan")
        return output


def _config(**overrides):
    values = {
        "horizon": 8,
        "action_dim": 1,
        "chunk_size": 1,
        "finite_difference_delta": 0.01,
        "inference_steps": 4,
        "p_k1": 1.0,
        "p_fm": 0.0,
        "fm_curriculum_steps": 0,
        "w_fm": 1.0,
        "w_lsd": 0.1,
        "use_ot": False,
        "iterative_cold_start": True,
    }
    values.update(overrides)
    return RollFlowConfig(**values)


def test_training_times_are_valid_for_every_divisor_k():
    sampler = StaircaseTimeSampler(_config())

    for k in (1, 2, 4, 8):
        times = sampler.sample_training(
            5,
            device="cpu",
            p_fm=0.0,
            num_time_groups=k,
        )
        assert torch.all(times.s >= 0)
        assert torch.all(times.s <= times.t)
        assert torch.all(times.t <= 1)
        assert torch.all(times.t[times.active] + 0.01 <= 1.0 + 1e-6)
        assert times.num_time_groups == k
        assert times.block_size == 8 // k
        assert times.ratio.shape == (5, 1, 1)
        group_width = times.block_size
        grouped_t = times.t[:, ::group_width, 0]
        assert torch.all(grouped_t[:, :-1] >= grouped_t[:, 1:])
        torch.testing.assert_close(times.s, times.ratio * times.t)


def test_sampled_active_times_pass_validation_at_float_boundaries():
    """The sampler and validator must use an identical LSD boundary test."""
    cfg = _config(horizon=32, chunk_size=8, p_k1=0.7)
    sampler = StaircaseTimeSampler(cfg)
    generator = torch.Generator().manual_seed(123)

    for _ in range(500):
        times = sampler.sample_training(
            32,
            device="cpu",
            generator=generator,
            p_fm=0.0,
        )
        _validate_times(
            times.s,
            times.t,
            cfg.finite_difference_delta,
            times.active,
        )


def test_grouped_training_times_match_32_by_8_design():
    cfg = RollFlowConfig(
        horizon=32,
        action_dim=1,
        chunk_size=8,
        finite_difference_delta=0.01,
        inference_steps=4,
    )
    sampler = StaircaseTimeSampler(cfg)
    for k, block_size in ((1, 4), (2, 2), (4, 1)):
        sampled = sampler.sample_training(
            3, device="cpu", p_fm=1.0, num_time_groups=k
        )
        assert sampled.block_size == block_size
        assert sampled.num_time_groups == k
        assert not sampled.active.any()
        torch.testing.assert_close(sampled.s, sampled.t)


def test_k_sampling_probabilities_and_config_validation():
    generator = torch.Generator().manual_seed(0)
    sampler = StaircaseTimeSampler(_config(p_k1=0.7))
    counts = {1: 0, 2: 0, 4: 0, 8: 0}
    for _ in range(4000):
        counts[sampler._sample_k(torch.device("cpu"), generator)] += 1
    assert counts[1] / 4000 == pytest.approx(0.7, abs=0.03)
    for k in (2, 4, 8):
        assert counts[k] / 4000 == pytest.approx(0.1, abs=0.025)
    with pytest.raises(ValueError, match="p_fm"):
        _config(p_fm=1.1)


def test_fm_curriculum_starts_diagonal_and_reaches_target_probability():
    torch.manual_seed(0)
    rollflow = RollFlow(_config(p_fm=0.2, fm_curriculum_steps=100))
    model = _LearnableConstant()
    actions = torch.randn(32, 8, 1)

    _, start = rollflow.loss(model, actions, step=0)
    assert start["p_fm"] == 1.0
    assert start["fm_only_frac"] == 1.0
    assert start["active_lsd_frac"] == 0.0
    assert model.batch_sizes == [32, 32, 96]
    assert start["lsd_loss"] == 0.0

    _, middle = rollflow.loss(model, actions, step=50)
    _, end = rollflow.loss(model, actions, step=100)
    assert middle["p_fm"] == pytest.approx(0.6)
    assert end["p_fm"] == pytest.approx(0.2)


@pytest.mark.parametrize("teacher_clip,scale", [(0.0, 1.0), (2.0, 0.1)])
def test_linear_path_oracle_has_zero_fm_and_lsd_error(teacher_clip, scale):
    torch.manual_seed(0)
    cfg = _config()
    x0 = scale * torch.randn(3, cfg.horizon, cfg.action_dim)
    x1 = scale * torch.randn_like(x0)
    times = StaircaseTimeSampler(cfg).sample_training(
        x0.shape[0],
        device=x0.device,
        p_fm=0.0,
        num_time_groups=4,
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
        w_fm=cfg.w_fm,
        # Keep the oracle test focused on the unscaled teacher identity.  The
        # production default intentionally uses a conservative target scale.
        w_lsd=1.0,
        teacher_clip=teacher_clip,
    )

    assert loss < 1e-8
    assert stats["fm_loss"] < 1e-10
    assert stats["lsd_loss"] < 1e-7


def test_clipped_teacher_has_expected_bias_and_no_teacher_gradient():
    model = _LearnableConstant()
    model.value.data.fill_(3.0)
    x = torch.zeros(1, 2, 1)
    s, t = torch.full_like(x, 0.2), torch.full_like(x, 0.6)
    loss, stats = central_difference_lsd(model, x, x, s, t, delta=0.01, w_fm=0, w_lsd=1, teacher_clip=2)
    assert stats["teacher_clip_frac"] == 1
    assert stats["v_teacher_abs"] == 2
    assert stats["lsd_loss_metric"] == pytest.approx(1, abs=1e-5)
    assert stats["lsd_loss_raw"] == pytest.approx(0.02, abs=1e-5)
    assert loss.item() == pytest.approx(0.02, abs=1e-4)
    loss.backward()
    assert model.value.grad.item() == pytest.approx(0.04, abs=1e-3)
    assert model.grad_enabled == [False, False, True]


def test_lsd_scaling_can_be_disabled_for_historical_ablation():
    model = _LearnableConstant()
    model.value.data.fill_(3.0)
    x = torch.zeros(1, 2, 1)
    s, t = torch.full_like(x, 0.2), torch.full_like(x, 0.6)
    loss, stats = central_difference_lsd(
        model,
        x,
        x,
        s,
        t,
        delta=0.01,
        w_fm=0,
        w_lsd=1,
        use_lsd_scaling=False,
        use_lsd_gate=False,
        teacher_clip=2,
    )

    assert stats["lsd_scaling_enabled"] == 0.0
    assert stats["lsd_loss_metric"] == pytest.approx(1.0, abs=1e-5)
    assert stats["lsd_loss_raw"] == pytest.approx(1.0, abs=1e-5)
    assert loss.item() == pytest.approx(1.0, abs=1e-5)
    loss.backward()
    assert model.value.grad.item() == pytest.approx(2.0, abs=1e-3)


def test_lsd_scalar_is_hard_masked_by_detached_fm_budget():
    model = _HighTangentResidual()
    x0 = torch.zeros(1, 2, 1)
    x1 = torch.zeros_like(x0)
    s = torch.full_like(x0, 0.2)
    t = torch.full_like(x0, 0.6)

    loss, stats = central_difference_lsd(
        model,
        x0,
        x1,
        s,
        t,
        delta=0.01,
        w_fm=1.0,
        w_lsd=1.0,
        teacher_clip=0.0,
    )

    assert stats["lsd_loss_raw"] > stats["fm_loss"]
    assert stats["lsd_gate_active"] == 1.0
    assert stats["lsd_loss"] == 0.0
    loss.backward()
    # The over-budget LSD branch is hard-masked; only FM contributes.
    assert model.value.grad.item() == pytest.approx(2.0, abs=1e-4)


def test_lsd_gate_can_be_disabled_without_changing_fm_supervision():
    model = _HighTangentResidual()
    x0 = torch.zeros(1, 2, 1)
    x1 = torch.zeros_like(x0)
    s = torch.full_like(x0, 0.2)
    t = torch.full_like(x0, 0.6)

    loss, stats = central_difference_lsd(
        model,
        x0,
        x1,
        s,
        t,
        delta=0.01,
        w_fm=1.0,
        w_lsd=1.0,
        use_lsd_gate=False,
        teacher_clip=0.0,
    )

    assert stats["lsd_gate_enabled"] == 0.0
    assert stats["lsd_gate_active"] == 0.0
    assert stats["lsd_keep_frac"] == 1.0
    assert stats["lsd_loss"] == stats["lsd_loss_raw"]
    assert loss > stats["fm_loss"]


def test_nonfinite_lsd_is_dropped_without_poisoning_fm():
    model = _NonFiniteTangent()
    x0 = torch.zeros(1, 2, 1)
    x1 = torch.zeros_like(x0)
    s = torch.full_like(x0, 0.2)
    t = torch.full_like(x0, 0.6)

    loss, stats = central_difference_lsd(
        model,
        x0,
        x1,
        s,
        t,
        delta=0.01,
        w_fm=1.0,
        w_lsd=1.0,
        teacher_clip=0.0,
    )

    assert torch.isnan(stats["lsd_loss_raw"])
    assert stats["lsd_gate_active"] == 1.0
    assert loss.item() == 0.0
    loss.backward()
    assert model.value.grad is not None
    assert torch.isfinite(model.value.grad)
    assert model.value.grad.item() == pytest.approx(0.0, abs=1e-6)


def test_loss_retains_gradients_only_for_tangent_and_local_paths():
    torch.manual_seed(0)
    rollflow = RollFlow(_config())
    model = _LearnableConstant()
    actions = torch.randn(3, 8, 1)

    loss, _ = rollflow.loss(model, actions)
    loss.backward()

    assert model.batch_sizes == [3, 3, 9]
    assert model.grad_enabled == [False, False, True]
    assert model.value.grad is not None
    assert torch.isfinite(model.value.grad)


def test_mixed_fm_batch_runs_all_rows_with_masked_lsd():
    model = _LearnableConstant()
    x0 = torch.randn(4, 8, 1)
    x1 = torch.randn_like(x0)
    t = torch.full((4, 8, 1), 0.6)
    s = t.clone()
    s[[1, 3]] = 0.2

    loss, stats = central_difference_lsd(
        model,
        x0,
        x1,
        s,
        t,
        delta=0.01,
    )
    loss.backward()

    assert model.batch_sizes == [4, 4, 12]
    assert model.grad_enabled == [False, False, True]
    assert stats["active_lsd_frac"] == 0.5


def test_padded_rows_run_forwards_but_are_excluded_from_lsd_loss():
    model = _LearnableConstant()
    x0 = torch.randn(4, 8, 1)
    x1 = torch.randn_like(x0)
    t = torch.full((4, 8, 1), 0.6)
    s = t.clone()
    s[[1, 3]] = 0.2
    pad = torch.zeros(4, 8, dtype=torch.bool)
    pad[3] = True

    loss, stats = central_difference_lsd(
        model,
        x0,
        x1,
        s,
        t,
        delta=0.01,
        pad=pad,
    )
    loss.backward()

    assert model.batch_sizes == [4, 4, 12]
    assert stats["active_lsd_frac"] == 0.25


def test_training_input_shapes_are_checked():
    x0 = torch.randn(2, 8, 1)
    x1 = torch.randn_like(x0)
    times = torch.zeros(2, 8, 1)
    with pytest.raises(ValueError, match="x0 and x1"):
        central_difference_lsd(
            _LearnableConstant(), x0, x1[:, :-1], times, times, delta=0.01
        )
    with pytest.raises(ValueError, match="s and t"):
        central_difference_lsd(
            _LearnableConstant(), x0, x1, times.squeeze(-1), times.squeeze(-1), delta=0.01
        )


def test_all_padding_returns_graph_connected_zero():
    torch.manual_seed(0)
    rollflow = RollFlow(_config())
    model = _LearnableConstant()
    actions = torch.randn(2, 8, 1)
    padding = torch.ones(2, 8, dtype=torch.bool)

    loss, _ = rollflow.loss(model, actions, pad=padding)
    loss.backward()

    assert loss.item() == 0.0
    assert model.value.grad is not None
    assert model.value.grad.item() == 0.0
    assert model.batch_sizes == [2, 2, 6]


def test_ot_matching_ignores_padded_target_values():
    x1 = torch.tensor([[[0.0], [1000.0]], [[10.0], [-1000.0]]])
    x0 = torch.tensor([[[0.0], [-1000.0]], [[10.0], [1000.0]]])
    pad = torch.tensor([[False, True], [False, True]])

    matched = ot_match(x1, x0, pad=pad)

    torch.testing.assert_close(matched, x0)


def test_oracle_rolls_sine_from_minus_20pi_to_plus_20pi():
    torch.manual_seed(0)
    cfg = _config()
    rollflow = RollFlow(cfg)
    oracle = _LinearPathOracle()
    phase_step = math.pi / 16.0
    phases = torch.linspace(-20.0 * math.pi, 20.0 * math.pi, 641)
    offsets = torch.arange(cfg.horizon) * phase_step
    predictions = []

    for phase in phases:
        target_chunk = torch.sin(phase + offsets)[None, :, None]
        action = rollflow.step(oracle, batch=1, context=target_chunk)
        predictions.append(action[0, 0, 0])

    torch.testing.assert_close(torch.stack(predictions), torch.sin(phases), atol=2e-5, rtol=2e-5)
    assert rollflow.cache_info is not None
    assert rollflow.cache_info.shape == (1, 8, 1)
    assert rollflow.cache_info.refinement_steps == 4

    rollflow.reset()
    assert rollflow.cache_info is None
