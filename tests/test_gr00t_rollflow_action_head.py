from types import SimpleNamespace

import pytest
import torch

from starVLA.model.modules.action_model.GR00T_ActionHeader import (
    FlowmatchingActionHead as StandardFlowmatchingActionHead,
)
from starVLA.model.modules.action_model.GR00T_RollFlow_ActionHeader import (
    RollFlowActionHead,
    RollFlowActionHeadConfig,
)
from starVLA.model.modules.action_model.rolling_meanflow_matching_head.action_encoder import (
    ActionEncoder,
)
from starVLA.model.modules.action_model.rolling_meanflow_matching_head.cross_attention_dit import DiT


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
        "finite_difference_delta": 0.01,
        "p_k1": 0.0,
        "p_fm": 0.0,
        "fm_curriculum_steps": 0,
        "inference_steps": 2,
        "w_fm": 1.0,
        "w_lsd": 0.5,
        "use_ot": False,
        "clip_velocity": 0.0,
        "iterative_cold_start": False,
        "reset_cache_each_step": False,
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
    return SimpleNamespace(framework=SimpleNamespace(action_model=SimpleNamespace(**values)))


def test_native_adapters_share_trunk_and_isolate_cache():
    cfg = _config(embodiments={"agibot-g1": {"state_dim": 22, "action_dim": 22}})
    head = RollFlowActionHead(cfg)
    context = torch.randn(2, 5, 12)
    loss = head(context, torch.randn(2, 4, 3), torch.randn(2, 1, 4), embodiment="franka")
    loss = loss + head(context, torch.randn(2, 4, 22), torch.randn(2, 1, 22), embodiment="agibot-g1")
    loss.backward()
    for module in (head.model, head.action_encoder, head.adapters["agibot-g1"]):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
        assert all(torch.isfinite(p.grad).all() for p in module.parameters() if p.grad is not None)
    restored = RollFlowActionHead(cfg)
    restored.load_state_dict(head.state_dict(), strict=True)
    assert not any("adapters.agibot-g1.model." in k for k in head.state_dict())
    assert head.predict_action(context, torch.randn(2, 1, 4), embodiment="franka").shape == (2, 2, 3)
    old_cache = head.rollflow._cache.clone()
    assert head.predict_action(context, torch.randn(2, 1, 22), embodiment="agibot-g1").shape == (2, 2, 22)
    torch.testing.assert_close(head.rollflow._cache, old_cache)
    head.reset()
    assert all(flow.cache_info is None for flow in head.rollflows.values())


def test_action_encoder_accepts_tokenwise_source_time():
    torch.manual_seed(0)
    encoder = ActionEncoder(action_dim=3, hidden_size=16)
    actions = torch.zeros(2, 4, 3)
    source_time = torch.tensor([[[0.0], [0.0], [500.0], [500.0]]] * 2)

    encoded = encoder(actions, source_time)

    assert encoded.shape == (2, 4, 16)
    assert not torch.allclose(encoded[:, 0], encoded[:, 2])
    with pytest.raises(ValueError, match="timesteps must have shape"):
        encoder(actions, torch.zeros(2, 3))


def test_dit_supports_endpoint_and_interval_per_token():
    torch.manual_seed(0)
    model = DiT(
        num_attention_heads=2,
        attention_head_dim=8,
        output_dim=16,
        num_layers=1,
        cross_attention_dim=12,
        dropout=0.0,
        final_dropout=False,
        positional_embeddings=None,
        norm_type="ada_norm",
    ).eval()
    hidden = torch.randn(2, 6, 16)
    context = torch.randn(2, 5, 12)
    target = torch.tensor([[1000.0] * 3 + [500.0] * 3] * 2)
    source = torch.tensor([[500.0] * 3 + [0.0] * 3] * 2)

    output = model(hidden, context, timestep=target, start_timestep=source)
    different_interval = model(hidden, context, timestep=target, start_timestep=target)
    legacy_output = model(hidden, context, timestep=torch.tensor([1000.0, 500.0]))

    assert output.shape == hidden.shape
    assert legacy_output.shape == hidden.shape
    assert not torch.allclose(output, different_interval)

    checkpoint_calls = []

    def checkpoint(module, *args):
        checkpoint_calls.append(module)
        return module(*args)

    model.enable_gradient_checkpointing(checkpoint)
    model.train()
    model(hidden, context, timestep=target, start_timestep=source)
    assert len(checkpoint_calls) == len(model.transformer_blocks)


def test_rollflow_head_loss_backward_and_rolling_inference():
    torch.manual_seed(0)
    model = RollFlowActionHead(_config())
    context = torch.randn(2, 5, 12)
    actions = torch.randn(2, 4, 3)
    state = torch.randn(2, 1, 4)
    context_mask = torch.tensor([[True, True, True, True, False]] * 2)

    times = model.rollflow.time.deployment(
        2,
        refinement_steps=2,
        cold=False,
        device="cpu",
    )
    velocity = model._predict_velocity(
        actions,
        times.s,
        times.t,
        context,
        state_features=model._encode_state(state),
        encoder_attention_mask=context_mask,
    )
    assert velocity.shape == actions.shape

    loss = model(context, actions, state, encoder_attention_mask=context_mask)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()

    endpoint_grad = model.model.timestep_encoder.timestep_embedder.linear_1.weight.grad
    interval_grad = model.model.interval_timestep_encoder.timestep_embedder.linear_1.weight.grad
    assert endpoint_grad is not None and torch.isfinite(endpoint_grad).all()
    assert interval_grad is not None and torch.isfinite(interval_grad).all()
    assert endpoint_grad.abs().sum() > 0
    assert interval_grad.abs().sum() > 0
    assert model.last_loss_stats["num_time_groups"] == 2

    model.eval()
    cold_chunk = model.predict_action(context, state, encoder_attention_mask=context_mask)
    warm_chunk = model.predict_action(context, state, encoder_attention_mask=context_mask)
    assert cold_chunk.shape == warm_chunk.shape == (2, 2, 3)
    assert model.cache_info.shape == (2, 4, 3)

    model.reset_cache()
    assert model.cache_info is None

    reloaded = RollFlowActionHead(_config())
    reloaded.load_state_dict(model.state_dict(), strict=True)
    assert "model.interval_timestep_encoder.timestep_embedder.linear_1.weight" in model.state_dict()


def test_rollflow_head_disables_stochastic_finite_differences():
    diffusion_cfg = _config().framework.action_model.diffusion_model_cfg.copy()
    diffusion_cfg["dropout"] = 0.1
    model = RollFlowActionHead(_config(diffusion_model_cfg=diffusion_cfg))
    assert model.model.config.dropout == 0.0
    assert model.model.config.final_dropout is False


def test_config_defaults_and_legacy_gr00t_checkpoint_loading():
    defaults = RollFlowActionHeadConfig()
    assert defaults.diffusion_model_cfg == {}
    assert defaults.to_dict()["finite_difference_delta"] == 0.01
    assert defaults.to_dict()["use_ot"] is True
    assert defaults.to_dict()["p_k1"] == 0.7
    assert defaults.to_dict()["p_fm"] == 0.3
    assert defaults.to_dict()["fm_curriculum_steps"] == 5000
    assert defaults.to_dict()["w_lsd"] == 0.1

    diffusion_cfg = _config().framework.action_model.diffusion_model_cfg.copy()
    diffusion_cfg.update(num_attention_heads=12, attention_head_dim=64, num_layers=1)
    config = _config(diffusion_model_cfg=diffusion_cfg)
    standard = StandardFlowmatchingActionHead(config)
    rollflow = RollFlowActionHead(config)
    rollflow.load_state_dict(standard.state_dict(), strict=True)

    endpoint = rollflow.model.timestep_encoder.state_dict()
    interval = rollflow.model.interval_timestep_encoder.state_dict()
    assert endpoint.keys() == interval.keys()
    for name in endpoint:
        torch.testing.assert_close(endpoint[name], interval[name])
