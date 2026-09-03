import math

import pytest
import torch
from torch import nn

from starVLA.model.modules.action_model.rolling_meanflow_matching_head.rolling_meanflow import (
    RollFlow,
    RollFlowConfig,
    StaircaseTimeSampler,
    central_difference_lsd,
)


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


def _config(**overrides):
    values = {
        "horizon": 8,
        "action_dim": 1,
        "chunk_size": 1,
        "finite_difference_delta": 0.01,
        "train_steps": [2, 4],
        "inference_steps": 4,
        "w_fm": 1.0,
        "w_lsd": 0.25,
        "use_ot": False,
        "iterative_cold_start": True,
    }
    values.update(overrides)
    return RollFlowConfig(**values)


def test_staircase_times_are_valid_and_adjacent_for_every_configured_k():
    sampler = StaircaseTimeSampler(_config(train_steps=[1, 2, 4, 8]))

    for refinement_steps in (1, 2, 4, 8):
        times = sampler.sample_training(
            5,
            device="cpu",
            refinement_steps=refinement_steps,
        )
        assert torch.all(times.s >= 0)
        assert torch.all(times.s <= times.t)
        assert torch.all(times.t <= 1)
        assert torch.all(times.t[times.active] + 0.01 <= 1.0 + 1e-6)
        if refinement_steps > 1:
            torch.testing.assert_close(
                times.t[:, 1:refinement_steps],
                times.s[:, : refinement_steps - 1],
            )
        assert int(times.active[:, :, 0].sum(dim=1).min()) == refinement_steps


def test_grouped_training_times_match_32_by_8_design():
    cfg = RollFlowConfig(
        horizon=32,
        action_dim=1,
        chunk_size=8,
        finite_difference_delta=0.01,
        train_block_sizes=[1, 2, 4],
        inference_steps=4,
    )
    sampler = StaircaseTimeSampler(cfg)
    lower = torch.zeros(1)
    upper = torch.ones(1)
    expected = {
        1: ([0.75, 0.50, 0.25, 0.00], [1.00, 0.75, 0.50, 0.25]),
        2: ([0.50, 0.50, 0.00, 0.00], [1.00, 1.00, 0.50, 0.50]),
        4: ([0.00, 0.00, 0.00, 0.00], [1.00, 1.00, 1.00, 1.00]),
    }

    for block_size, (expected_s, expected_t) in expected.items():
        s, t, active = sampler._build_grouped(lower, upper, block_size)
        torch.testing.assert_close(s[0, ::8, 0], torch.tensor(expected_s))
        torch.testing.assert_close(t[0, ::8, 0], torch.tensor(expected_t))
        assert active.all()

        sampled = sampler.sample_training(3, device="cpu", block_size=block_size)
        assert sampled.block_size == block_size
        assert sampled.refinement_steps == cfg.num_action_chunks // block_size
        assert sampled.active.all()


def test_train_block_size_and_legacy_train_steps_are_unambiguous():
    with pytest.raises(ValueError, match="only one"):
        RollFlowConfig(
            horizon=32,
            action_dim=1,
            chunk_size=8,
            train_steps=[4],
            train_block_sizes=[1, 2, 4],
        )


def test_linear_path_oracle_has_zero_fm_and_lsd_error():
    torch.manual_seed(0)
    cfg = _config(train_steps=[4])
    x0 = torch.randn(3, cfg.horizon, cfg.action_dim)
    x1 = torch.randn_like(x0)
    times = StaircaseTimeSampler(cfg).sample_training(
        x0.shape[0],
        device=x0.device,
        refinement_steps=4,
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
        w_lsd=cfg.w_lsd,
    )

    assert loss < 1e-8
    assert stats["fm_loss"] < 1e-10
    assert stats["lsd_loss"] < 1e-7


def test_loss_retains_gradients_only_for_tangent_and_local_paths():
    torch.manual_seed(0)
    rollflow = RollFlow(_config(train_steps=[2]))
    model = _LearnableConstant()
    actions = torch.randn(3, 8, 1)

    loss, _ = rollflow.loss(model, actions)
    loss.backward()

    assert model.batch_sizes == [6, 3, 3, 3]
    assert model.grad_enabled == [True, False, False, True]
    assert model.value.grad is not None
    assert torch.isfinite(model.value.grad)


def test_all_padding_returns_graph_connected_zero():
    torch.manual_seed(0)
    rollflow = RollFlow(_config(train_steps=[2]))
    model = _LearnableConstant()
    actions = torch.randn(2, 8, 1)
    padding = torch.ones(2, 8, dtype=torch.bool)

    loss, _ = rollflow.loss(model, actions, pad=padding)
    loss.backward()

    assert loss.item() == 0.0
    assert model.value.grad is not None
    assert model.value.grad.item() == 0.0


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
