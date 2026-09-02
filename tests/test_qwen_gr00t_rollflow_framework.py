from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch import nn

import starVLA.model.framework.VLM4A.QwenGR00TRollFlow as rollflow_framework


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
                    "action_dim": 3,
                    "state_dim": 2,
                    "hidden_size": 16,
                    "max_seq_len": 8,
                    "num_target_vision_tokens": 2,
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


def test_qwen_rollflow_defaults_keep_noise_repetition():
    defaults = rollflow_framework.QwenGR00TRollFlowDefaultConfig()
    assert defaults.qwenvl["base_vlm"].endswith("Qwen3.5-0.8B")
    assert defaults.action_model["repeated_diffusion_steps"] == 8


def test_qwen_rollflow_full_policy_with_tiny_vlm(monkeypatch):
    torch.manual_seed(0)
    monkeypatch.setattr(rollflow_framework, "get_vlm_model", lambda config: _TinyVLM())
    model = rollflow_framework.Qwen_GR00T_RollFlow(_tiny_framework_config())
    assert model.config.framework.action_model.repeated_diffusion_steps == 8

    image = Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8))
    example = {
        "action": np.zeros((4, 3), dtype=np.float32),
        "image": [image],
        "lang": "test instruction",
        "state": np.zeros((1, 2), dtype=np.float32),
    }

    loss = model([example])["action_loss"]
    assert torch.isfinite(loss)
    loss.backward()
    assert model.qwen_vl_interface.model.token.grad.abs().sum() > 0
    assert model.action_model.model.interval_timestep_encoder.timestep_embedder.linear_1.weight.grad is not None

    model.eval()
    cold = model.predict_action([example])["normalized_actions"]
    warm = model.predict_action([example])["normalized_actions"]
    assert cold.shape == warm.shape == (1, 1, 3)
    assert np.isfinite(cold).all() and np.isfinite(warm).all()

    model.action_model.reset_cache()
    assert model.action_model.cache_info is None
