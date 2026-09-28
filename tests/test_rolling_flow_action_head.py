from types import SimpleNamespace

import pytest
import torch

from starVLA.model.modules.action_model.GR00T_RollFlowMatching_ActionHeader import (
    PaddedGrootActionHead,
    RollFlowMatchingActionHead,
)
from starVLA.model.modules.action_model.GR00T_ActionHeader import (
    DiTConfig,
    FlowmatchingActionHead,
)


def _config(**overrides):
    values = {
        "action_model_type": "DiT-B",
        "action_horizon": 4,
        "execution_horizon": 2,
        "action_dim": 3,
        "state_dim": 4,
        "hidden_size": 16,
        "add_pos_embed": True,
        "max_seq_len": 8,
        "num_target_vision_tokens": 2,
        "noise_beta_alpha": 1.5,
        "noise_beta_beta": 1.0,
        "noise_s": 0.999,
        "num_inference_timesteps": 2,
        "num_timestep_buckets": 1000,
        "p_k1": 0.0,
        "use_ot": False,
        "inference_steps": 2,
        "iterative_cold_start": True,
        "diffusion_model_cfg": {
            "num_attention_heads": 2,
            "attention_head_dim": 8,
            "output_dim": 16,
            "num_layers": 2,
            "cross_attention_dim": 12,
            "dropout": 0.0,
            "final_dropout": False,
            "positional_embeddings": None,
            "interleave_self_attention": True,
            "norm_type": "ada_norm",
        },
    }
    values.update(overrides)
    return SimpleNamespace(
        framework=SimpleNamespace(action_model=SimpleNamespace(**values)),
        trainer=SimpleNamespace(max_train_steps=10),
    )


@pytest.fixture(autouse=True)
def _use_tiny_matching_dit_dimensions(monkeypatch):
    monkeypatch.setitem(
        DiTConfig,
        "DiT-B",
        {"input_embedding_dim": 16, "attention_head_dim": 8, "num_attention_heads": 2},
    )


def test_new_head_keeps_standard_groot_state_dict_layout():
    standard = FlowmatchingActionHead(_config())
    rolling = RollFlowMatchingActionHead(_config())

    assert standard.state_dict().keys() == rolling.state_dict().keys()
    assert {
        key: value.shape for key, value in standard.state_dict().items()
    } == {key: value.shape for key, value in rolling.state_dict().items()}
    assert all("interval_timestep" not in key for key in rolling.state_dict())
    assert isinstance(rolling, FlowmatchingActionHead)
    assert type(rolling.action_encoder) is type(standard.action_encoder)
    assert type(rolling.model) is type(standard.model)


def test_parallel_groot_matches_official_scalar_time_path(monkeypatch):
    # Tiny DiT keeps this a fast CPU test while exercising the original head
    # and the parallel wrapper with the same module dimensions and weights.
    monkeypatch.setitem(
        DiTConfig,
        "DiT-B",
        {"input_embedding_dim": 16, "attention_head_dim": 8, "num_attention_heads": 2},
    )
    config = _config()
    official = FlowmatchingActionHead(config).eval()
    parallel = PaddedGrootActionHead(config).eval()
    assert official.state_dict().keys() == parallel.state_dict().keys()
    parallel.load_state_dict(official.state_dict(), strict=True)

    context = torch.randn(2, 5, 12)
    state = torch.randn(2, 1, 4)
    actions = torch.randn(2, 4, 3)
    torch.manual_seed(29)
    official_loss = official(context, actions, state)
    torch.manual_seed(29)
    parallel_loss = parallel(context, actions, state)
    torch.testing.assert_close(parallel_loss, official_loss, rtol=1e-5, atol=1e-6)

    official.eval()
    parallel.eval()
    torch.manual_seed(31)
    official_actions = official.predict_action(context, state)
    torch.manual_seed(31)
    parallel_actions = parallel.predict_action(context, state)
    torch.testing.assert_close(parallel_actions, official_actions, rtol=1e-5, atol=1e-6)


def test_single_group_training_matches_standard_groot_flow_matching(monkeypatch):
    monkeypatch.setitem(
        DiTConfig,
        "DiT-B",
        {"input_embedding_dim": 16, "attention_head_dim": 8, "num_attention_heads": 2},
    )
    config = _config(p_k1=1.0, use_ot=False)
    official = FlowmatchingActionHead(config)
    rolling = RollFlowMatchingActionHead(config)
    rolling.load_state_dict(official.state_dict(), strict=True)

    context = torch.randn(2, 5, 12)
    state = torch.randn(2, 1, 4)
    actions = torch.randn(2, 4, 3)
    torch.manual_seed(103)
    official_loss = official(context, actions, state)
    torch.manual_seed(103)
    rolling_loss = rolling(context, actions, state)

    torch.testing.assert_close(rolling_loss, official_loss, rtol=1e-5, atol=1e-6)


def test_optimal_transport_is_an_explicit_configuration(monkeypatch):
    import starVLA.model.modules.action_model.rolling_flow_matching_head.rolling_flow as flow

    calls = []
    original = flow._ot_match

    def record_ot(target, noise, pad):
        calls.append((target.shape, pad is not None))
        return original(target, noise, pad)

    monkeypatch.setattr(flow, "_ot_match", record_ot)
    model = RollFlowMatchingActionHead(_config(p_k1=1.0, use_ot=True)).train()
    loss = model(torch.randn(2, 5, 12), torch.randn(2, 4, 3))

    assert torch.isfinite(loss)
    assert calls == [((2, 4, 3), False)]


def test_parallel_groot_fm_path_masks_padded_action_keys_and_loss():
    torch.manual_seed(3)
    model = PaddedGrootActionHead(_config()).train()
    context = torch.randn(2, 5, 12)
    context_mask = torch.ones(2, 5, dtype=torch.bool)
    actions = torch.randn(2, 4, 3)
    state = torch.randn(2, 1, 4)
    padding = torch.tensor([[False, False, False, True], [False, False, False, False]])

    loss = model(
        context,
        actions,
        state,
        encoder_attention_mask=context_mask,
        action_padding_mask=padding,
    )

    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert model.model.proj_out_2.weight.grad is not None
    assert torch.isfinite(model.model.proj_out_2.weight.grad).all()


def test_grouped_instantaneous_fm_has_no_distillation_and_backpropagates(monkeypatch):
    torch.manual_seed(0)
    model = RollFlowMatchingActionHead(_config()).train()
    context = torch.randn(2, 5, 12)
    context_mask = torch.tensor([[True, True, True, True, False]] * 2)
    actions = torch.randn(2, 4, 3)
    state = torch.randn(2, 1, 4)
    padding = torch.tensor([[False, False, False, True], [False, False, False, False]])
    calls = []
    predict_grouped = model._predict_grouped

    def record_call(actions, action_times, context, **kwargs):
        calls.append(action_times.detach().clone())
        return predict_grouped(actions, action_times, context, **kwargs)

    monkeypatch.setattr(model, "_predict_grouped", record_call)

    loss = model(
        context,
        actions,
        state,
        encoder_attention_mask=context_mask,
        action_padding_mask=padding,
        training_step=3,
    )

    assert loss.ndim == 0 and torch.isfinite(loss)
    assert model.last_loss_stats["num_time_groups"] == 2
    assert model.last_loss_stats["train_block_size"] == 1
    assert "p_fm" not in model.last_loss_stats
    assert "lsd_loss" not in model.last_loss_stats
    assert model.last_loss_stats["use_ot"] is False
    assert len(calls) == 1  # One packed batch; every row follows scalar-time Groot.
    torch.testing.assert_close(calls[0], calls[0][:, :1].expand_as(calls[0]))
    loss.backward()
    assert model.action_encoder.layer1.weight.grad is not None
    assert torch.isfinite(model.action_encoder.layer1.weight.grad).all()


@pytest.mark.parametrize("head_type", [PaddedGrootActionHead, RollFlowMatchingActionHead])
def test_padding_values_do_not_change_loss_or_receive_gradients(head_type):
    model = head_type(_config()).train()
    context = torch.randn(2, 5, 12)
    state = torch.randn(2, 1, 4)
    actions = torch.randn(2, 4, 3, requires_grad=True)
    padding = torch.tensor([[False, False, False, True], [False, False, False, False]])
    changed = actions.detach().clone()
    changed[0, -1] = 10_000

    torch.manual_seed(47)
    loss = model(context, actions, state, action_padding_mask=padding)
    loss.backward()
    assert torch.count_nonzero(actions.grad[padding]) == 0

    torch.manual_seed(47)
    changed_loss = model(context, changed, state, action_padding_mask=padding)
    torch.testing.assert_close(changed_loss, loss.detach(), rtol=1e-5, atol=1e-6)


def test_rolling_head_uses_original_scalar_timestep_for_each_time_plateau():
    model = RollFlowMatchingActionHead(_config()).eval()
    seen_action_times = []
    seen_dit_times = []
    model.action_encoder.register_forward_pre_hook(
        lambda _module, args: seen_action_times.append(args[1].detach().clone())
    )
    model.model.register_forward_pre_hook(
        lambda _module, _args, kwargs: seen_dit_times.append(
            kwargs["timestep"].detach().clone()
        ),
        with_kwargs=True,
    )

    actions = torch.randn(2, 4, 3)
    context = torch.randn(2, 5, 12)
    source_time = torch.tensor([[0.1239, 0.1239, 0.7812, 0.7812]]).expand(2, -1)
    output = model._predict_grouped(actions, source_time, context)

    assert output.shape == actions.shape
    # Time plateaus share one vectorized Groot forward, in batch-major order.
    assert len(seen_action_times) == 1
    assert len(seen_dit_times) == 1
    expected = torch.tensor([123, 781, 123, 781])
    torch.testing.assert_close(seen_action_times[0], expected)
    torch.testing.assert_close(seen_dit_times[0], expected)


def test_inference_is_instantaneous_and_emits_one_execution_chunk():
    torch.manual_seed(1)
    model = RollFlowMatchingActionHead(_config()).eval()
    context = torch.randn(2, 5, 12)
    state = torch.randn(2, 1, 4)
    context_mask = torch.ones(2, 5, dtype=torch.bool)

    actions = model.predict_action(
        context,
        state,
        encoder_attention_mask=context_mask,
        refinement_steps=2,
    )

    assert actions.shape == (2, 2, 3)
    assert model.cache_info.shape == (2, 4, 3)
    assert model.cache_info.refinement_steps == 2
    next_actions = model.predict_action(context, state, encoder_attention_mask=context_mask)
    assert next_actions.shape == (2, 2, 3)
    assert model.cache_info.shape == (2, 4, 3)
    with pytest.raises(ValueError, match="instantaneous-velocity inference only"):
        model.predict_action(context, velocity_mode="average")


def test_h64_c16_four_step_inference_emits_sixteen_actions():
    model = RollFlowMatchingActionHead(
        _config(
            action_horizon=64,
            execution_horizon=16,
            max_seq_len=64,
            inference_steps=4,
            num_inference_timesteps=4,
        )
    ).eval()
    context = torch.randn(1, 5, 12)
    state = torch.randn(1, 1, 4)

    actions = model.predict_action(context, state, refinement_steps=4)

    assert actions.shape == (1, 16, 3)
    assert model.cache_info.shape == (1, 64, 3)
    assert model.cache_info.refinement_steps == 4
