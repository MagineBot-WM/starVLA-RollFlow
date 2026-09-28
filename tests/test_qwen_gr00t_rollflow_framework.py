from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch import nn

import starVLA.model.framework.VLM4A.QwenGR00TRollFlow as rollflow_framework
import starVLA.model.framework.VLM4A.QwenGR00TRollingFlow as rolling_flow_framework
from starVLA.model.modules.action_model.GR00T_ActionHeader import DiTConfig


class _TinyBackbone(nn.Module):
    def __init__(self, hidden_size=12):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.token = nn.Parameter(torch.randn(1, 1, hidden_size))

    def forward(self, input_ids, **kwargs):
        del kwargs
        hidden = self.token.expand(input_ids.shape[0], input_ids.shape[1], -1)
        return SimpleNamespace(hidden_states=(hidden,))


class _TinyVLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _TinyBackbone()

    def build_qwenvl_inputs(self, images, instructions):
        batch = len(images)
        assert batch == len(instructions)
        device = self.model.token.device
        return {
            "input_ids": torch.ones(batch, 5, dtype=torch.long, device=device),
            "attention_mask": torch.ones(batch, 5, dtype=torch.bool, device=device),
        }

    def forward(self, **kwargs):
        return self.model(**kwargs)


def _tiny_framework_config():
    return OmegaConf.create(
        {
            "framework": {
                "name": "QwenGR00TRollFlow",
                "action_model": {
                    "action_horizon": 4,
                    "execution_horizon": 2,
                    "action_dim": 3,
                    "state_dim": 2,
                    "hidden_size": 16,
                    "max_seq_len": 8,
                    "num_target_vision_tokens": 2,
                    "p_k1": 0.0,
                    "p_fm": 0.0,
                    "fm_curriculum_steps": 0,
                    "inference_steps": 2,
                    "num_inference_timesteps": 2,
                    "diffusion_model_cfg": {
                        "num_attention_heads": 2,
                        "attention_head_dim": 8,
                        "output_dim": 16,
                        "num_layers": 2,
                    },
                },
            },
            "datasets": {"vla_data": {}},
        }
    )


def test_qwen_rollflow_defaults_are_explicit_and_reproducible():
    defaults = rollflow_framework.QwenGR00TRollFlowDefaultConfig()
    assert defaults.qwenvl["base_vlm"].endswith("Qwen3.5-0.8B")
    assert defaults.action_model["action_horizon"] == 32
    assert defaults.action_model["execution_horizon"] == 8
    assert defaults.action_model["repeated_diffusion_steps"] == 4
    assert defaults.action_model["finite_difference_delta"] == 0.01
    assert defaults.action_model["p_k1"] == 0.7
    assert defaults.action_model["p_fm"] == 0.3
    assert defaults.action_model["fm_curriculum_steps"] == 5000
    assert defaults.action_model["inference_steps"] == 4
    assert defaults.action_model["use_ot"] is True
    assert defaults.action_model["diffusion_model_cfg"]["dropout"] == 0.0


def test_qwen_rollflow_full_policy_with_tiny_vlm(monkeypatch):
    torch.manual_seed(0)
    monkeypatch.setattr(rollflow_framework, "get_vlm_model", lambda config: _TinyVLM())
    model = rollflow_framework.Qwen_GR00T_RollFlow(_tiny_framework_config())
    assert model.config.framework.action_model.repeated_diffusion_steps == 4

    image = Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8))
    example = {
        "action": np.zeros((4, 3), dtype=np.float32),
        "image": [image],
        "lang": "test instruction",
        "state": np.zeros((1, 2), dtype=np.float32),
    }

    output = model([example])
    loss = output["action_loss"]
    assert {"fm_loss", "lsd_loss", "p_fm", "fm_only_frac", "active_lsd_frac"}.issubset(
        output["rollflow_metrics"]
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert model.qwen_vl_interface.model.token.grad.abs().sum() > 0
    assert model.action_model.model.interval_timestep_encoder.timestep_embedder.linear_1.weight.grad is not None

    model.eval()
    cold = model.predict_action([example], refinement_steps=2)["normalized_actions"]
    warm = model.predict_action([example], refinement_steps=2)["normalized_actions"]
    assert cold.shape == warm.shape == (1, 2, 3)
    assert np.isfinite(cold).all() and np.isfinite(warm).all()
    assert model.action_model.cache_info.refinement_steps == 2

    model.reset()
    assert model.action_model.cache_info is None


def test_qwen_rollflow_forwards_action_padding_mask(monkeypatch):
    torch.manual_seed(0)
    monkeypatch.setattr(rollflow_framework, "get_vlm_model", lambda config: _TinyVLM())
    model = rollflow_framework.Qwen_GR00T_RollFlow(_tiny_framework_config())
    image = Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8))
    example = {
        "action": np.zeros((4, 3), dtype=np.float32),
        "action_padding_mask": np.ones(4, dtype=bool),
        "image": [image],
        "lang": "fully padded test",
        "state": np.zeros((1, 2), dtype=np.float32),
    }

    loss = model([example])["action_loss"]
    loss.backward()

    assert loss.item() == 0.0
    assert model.action_model.action_decoder.layer2.weight.grad is not None


def test_qwen_rollflow_converts_bfloat16_predictions_to_numpy(monkeypatch):
    torch.manual_seed(0)
    monkeypatch.setattr(rollflow_framework, "get_vlm_model", lambda config: _TinyVLM())
    model = rollflow_framework.Qwen_GR00T_RollFlow(_tiny_framework_config())
    image = Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8))
    example = {
        "action": np.zeros((4, 3), dtype=np.float32),
        "image": [image],
        "lang": "bfloat16 inference",
        "state": np.zeros((1, 2), dtype=np.float32),
    }
    monkeypatch.setattr(
        model.action_model,
        "predict_action",
        lambda *args, **kwargs: torch.ones(1, 2, 3, dtype=torch.bfloat16),
    )

    prediction = model.predict_action([example])["normalized_actions"]

    assert prediction.dtype == np.float32
    np.testing.assert_array_equal(prediction, np.ones((1, 2, 3), dtype=np.float32))


def test_qwen_rolling_flow_uses_groot_and_has_no_distillation_config(monkeypatch):
    defaults = rolling_flow_framework.QwenGR00TRollingFlowDefaultConfig()
    assert defaults.name == "QwenGR00TRollingFlow"
    assert issubclass(rolling_flow_framework.Qwen_GR00T_RollingFlow, rolling_flow_framework.Qwen_GR00T)
    assert not issubclass(
        rolling_flow_framework.Qwen_GR00T_RollingFlow,
        rollflow_framework.Qwen_GR00T_RollFlow,
    )
    assert "p_fm" not in defaults.action_model
    assert "w_lsd" not in defaults.action_model
    assert "fm_curriculum_steps" not in defaults.action_model
    assert defaults.action_model["use_ot"] is True

    monkeypatch.setitem(
        DiTConfig,
        "DiT-B",
        {"input_embedding_dim": 16, "attention_head_dim": 8, "num_attention_heads": 2},
    )
    monkeypatch.setattr(rolling_flow_framework, "get_vlm_model", lambda config: _TinyVLM())
    config = _tiny_framework_config()
    config.framework.name = "QwenGR00TRollingFlow"
    config.framework.action_model.p_k1 = 1.0
    config.framework.action_model.use_ot = False
    model = rolling_flow_framework.Qwen_GR00T_RollingFlow(config)

    image = Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8))
    example = {
        "action": np.zeros((4, 3), dtype=np.float32),
        "image": [image],
        "lang": "rolling flow test",
        "state": np.zeros((1, 2), dtype=np.float32),
    }
    loss = model([example])["action_loss"]
    assert torch.isfinite(loss)
    assert "lsd_loss" not in model.action_model.last_loss_stats
    loss.backward()

    model.eval()
    prediction = model.predict_action([example], refinement_steps=2)["normalized_actions"]
    assert prediction.shape == (1, 2, 3)
    assert np.isfinite(prediction).all()
