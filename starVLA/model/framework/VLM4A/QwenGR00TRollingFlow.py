"""Parallel standard-Groot framework for pure rolling Flow Matching."""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.VLM4A.QwenGR00T import (
    Qwen_GR00T,
    QwenGR00TDefaultConfig,
)
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.GR00T_RollFlowMatching_ActionHeader import (
    PaddedGrootActionHead,
    get_action_model as get_rolling_action_model,
)
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


def _pure_flow_action_defaults():
    """Standard Groot settings plus rolling-flow controls, without LSD knobs."""
    defaults = dict(QwenGR00TDefaultConfig().action_model)
    defaults.update(
        {
            "execution_horizon": 1,
            "p_k1": 0.7,
            "inference_steps": defaults.get("num_inference_timesteps", 4),
            "use_ot": True,
            "clip_velocity": 0.0,
            "iterative_cold_start": True,
            "reset_cache_each_step": False,
        }
    )
    return defaults


@dataclass
class QwenGR00TRollingFlowDefaultConfig(QwenGR00TDefaultConfig):
    """Groot defaults with staircase-time training and rolling inference."""

    name: str = "QwenGR00TRollingFlow"
    action_model: dict = field(default_factory=_pure_flow_action_defaults)


@dataclass
class QwenGR00TPaddedCompareDefaultConfig(QwenGR00TRollingFlowDefaultConfig):
    """Padding-aware standard Groot baseline config for controlled comparisons."""

    name: str = "QwenGR00TPaddedCompare"


class _QwenGR00TRollingFramework(Qwen_GR00T):
    """Share standard Groot's VLM/input contract, with a selectable action head."""

    _truncate_prediction = False

    def _initialize_parallel_policy(self, config, default_config, action_head_factory):
        # Build the same VLM + action conditioning contract as Qwen_GR00T while
        # selecting the parallel action head before constructing any Groot head.
        baseframework.__init__(self)
        self.config = merge_framework_config(default_config, config)
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = (
            self.qwen_vl_interface.model.config.hidden_size
        )
        self.action_model = action_head_factory(self.config)
        self.action_horizon = int(self.config.framework.action_model.action_horizon)
        self.execution_horizon = int(
            self.config.framework.action_model.execution_horizon
        )

    def forward(
        self,
        examples: Optional[List[dict]] = None,
        **kwargs,
    ) -> dict[str, object]:
        images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]
        states = [example["state"] for example in examples] if "state" in examples[0] else None

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=images, instructions=instructions
        )
        backbone_attention_mask = qwen_inputs.get("attention_mask")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = outputs.hidden_states[-1]

        with torch.autocast("cuda", enabled=False):
            actions = torch.as_tensor(
                np.asarray(actions), device=last_hidden.device, dtype=torch.float32
            )
            actions_target = actions[:, -self.action_horizon :]
            repeats = int(self.config.framework.action_model.get("repeated_diffusion_steps", 1))
            actions_target = actions_target.repeat(repeats, 1, 1)
            last_hidden = last_hidden.repeat(repeats, 1, 1)
            if backbone_attention_mask is not None:
                backbone_attention_mask = backbone_attention_mask.repeat(repeats, 1).bool()

            state = None
            if states is not None:
                state = torch.as_tensor(
                    np.asarray(states), device=last_hidden.device, dtype=torch.float32
                ).repeat(repeats, 1, 1)

            action_padding_mask = None
            if "action_padding_mask" in examples[0]:
                if not all("action_padding_mask" in example for example in examples):
                    raise ValueError(
                        "action_padding_mask must be present for every example or none"
                    )
                action_padding_mask = torch.as_tensor(
                    np.asarray([example["action_padding_mask"] for example in examples]),
                    device=last_hidden.device,
                    dtype=torch.bool,
                )[:, -self.action_horizon :].repeat(repeats, 1)

            action_loss = self.action_model(
                last_hidden,
                actions_target,
                state,
                encoder_attention_mask=backbone_attention_mask,
                action_padding_mask=action_padding_mask,
                training_step=int(kwargs.get("training_step", 0)),
            )

        return {
            "action_loss": action_loss,
            "action_metrics": self.action_model.last_loss_stats or {},
        }

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict],
        **kwargs,
    ) -> dict[str, np.ndarray]:
        if type(examples) is not list:
            examples = [examples]
        images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]
        states = [example["state"] for example in examples] if "state" in examples[0] else None

        datasets = getattr(self.config, "datasets", None)
        vla_data = getattr(datasets, "vla_data", None)
        train_image_size = getattr(vla_data, "obs_image_size", None)
        if train_image_size:
            images = resize_images(images, target_size=train_image_size)

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=images, instructions=instructions
        )
        backbone_attention_mask = qwen_inputs.get("attention_mask")
        if backbone_attention_mask is not None:
            backbone_attention_mask = backbone_attention_mask.bool()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = outputs.hidden_states[-1]

        state = (
            torch.as_tensor(np.asarray(states), device=last_hidden.device, dtype=last_hidden.dtype)
            if states is not None
            else None
        )
        with torch.autocast("cuda", enabled=False):
            actions = self.action_model.predict_action(
                last_hidden,
                state,
                encoder_attention_mask=backbone_attention_mask,
                refinement_steps=kwargs.get("refinement_steps"),
                velocity_mode=kwargs.get("velocity_mode"),
            )
        if self._truncate_prediction:
            actions = actions[:, : self.execution_horizon]
        return {"normalized_actions": actions.detach().float().cpu().numpy()}

    def reset(self) -> None:
        reset = getattr(self.action_model, "reset_cache", self.action_model.reset)
        reset()

    reset_cache = reset


@FRAMEWORK_REGISTRY.register("QwenGR00TRollingFlow")
class Qwen_GR00T_RollingFlow(_QwenGR00TRollingFramework):
    """Standard Groot modules trained with pure staircase-time Flow Matching."""

    def __init__(self, config=None, **kwargs) -> None:
        self._initialize_parallel_policy(
            config,
            QwenGR00TRollingFlowDefaultConfig,
            lambda resolved: get_rolling_action_model(config=resolved),
        )


@FRAMEWORK_REGISTRY.register("QwenGR00TPaddedCompare")
class Qwen_GR00T_PaddedCompare(_QwenGR00TRollingFramework):
    """Padding-aware standard-Groot FM baseline for controlled comparisons."""

    _truncate_prediction = True

    def __init__(self, config=None, **kwargs) -> None:
        self._initialize_parallel_policy(
            config,
            QwenGR00TPaddedCompareDefaultConfig,
            lambda resolved: PaddedGrootActionHead(full_config=resolved),
        )
