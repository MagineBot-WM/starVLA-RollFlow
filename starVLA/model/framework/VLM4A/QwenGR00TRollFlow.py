# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# ruff: noqa: E402
# Implemented by [Junqiu YU / Fudan University] in [2025].
# Design and Merged by [Jinhui YE / HKUST University] in [2025].
"""
Qwen-GR00T Framework
A lightweight implementation that Qwen-VL + Flow-matching head to directly predict continuous actions
Flow-matching header is copyright from GR00T N1.5,
"""

import sys
from pathlib import Path

# Add workspace root to Python path if not already there
_workspace_root = Path(__file__).parent.parent.parent.parent.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
from PIL import Image

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.GR00T_RollFlow_ActionHeader import (
    RollFlowActionHead,
    get_action_model,
)
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


# ──────────────────────────────────────────────────────────────────────
#  Default Config for QwenGR00T
#  - Documents every framework-level parameter with type + description
#  - YAML values override these defaults; extra YAML keys are preserved
# ──────────────────────────────────────────────────────────────────────
@dataclass
class QwenGR00TRollFlowDefaultConfig:
    """QwenGR00TRollFlow framework default parameters.

    All fields can be overridden by the corresponding key in the YAML
    ``framework:`` section.  Extra YAML keys not listed here are kept
    as-is (Config-as-API flexibility).
    """

    # --- Registry identifier ---
    name: str = "QwenGR00TRollFlow"

    # === VLM backbone (Qwen2.5-VL / Qwen3-VL) ===
    qwenvl: dict = field(
        default_factory=lambda: {
            # Path to base VLM checkpoint (local or HF hub id)
            "base_vlm": "./playground/Pretrained_models/Qwen3.5-0.8B",
            # Attention implementation: "flash_attention_2" | "eager" | "sdpa"
            "attn_implementation": "flash_attention_2",
            # VLM hidden dimension (used for cross-attention alignment)
            "vl_hidden_dim": 2048,
        }
    )

    # DINO is not used in this QwenGR00T version; it can be added later for
    # optional multi-view spatial tokens.
    # dino: dict = field(default_factory=lambda: {
    #     # DINO backbone variant: "dinov2_vits14" | "dinov2_vitb14" | ...
    #     "dino_backbone": "dinov2_vits14",
    # })

    # === Action head (Flow-matching / DiT diffusion) ===
    action_model: dict = field(
        default_factory=lambda: {
            # DiT model size: "DiT-B" | "DiT-L" | "DiT-XL"
            "action_model_type": "DiT-B",
            # Hidden dim for action model (auto-aligned at runtime)
            "action_hidden_dim": 1024,
            "hidden_size": 1024,
            # Whether to add positional embeddings in the action head
            "add_pos_embed": True,
            "max_seq_len": 1024,
            # Dimensionality of each action vector (e.g., 7 for 6-DoF + gripper)
            "action_dim": 7,
            # State dimension (proprioception input)
            "state_dim": 7,
            # Canonical chunk length (number of action steps the head predicts).
            # Legacy YAMLs may use future_action_window_size = action_horizon - 1;
            # apply_config_compat normalises both directions.
            "action_horizon": 32,
            # Number of actions returned and shifted from the rolling cache.
            "execution_horizon": 8,
            # Four independent noise/time samples per raw condition.
            "repeated_diffusion_steps": 4,
            "num_timestep_buckets": 1000,
            # RollFlow objective and randomized grouped-time defaults.
            "finite_difference_delta": 0.01,
            "p_k1": 0.7,
            "p_fm": 0.3,
            "fm_curriculum_steps": 5000,
            "inference_steps": 4,
            "w_fm": 1.0,
            "w_lsd": 0.1,
            "use_ot": True,
            "use_lsd_gate": True,
            "clip_velocity": 0.0,
            "iterative_cold_start": True,
            "reset_cache_each_step": False,
            # Number of vision tokens fed to action head
            "num_target_vision_tokens": 32,
            # === DiT Transformer sub-config ===
            "diffusion_model_cfg": {
                # Cross-attention dim (aligned to VLM hidden_size at runtime)
                "cross_attention_dim": 2048,
                # Independent masks would contaminate the finite difference.
                "dropout": 0.0,
                "final_dropout": False,
                "interleave_self_attention": True,
                "norm_type": "ada_norm",
                "num_layers": 16,
                "output_dim": 1024,
                "positional_embeddings": None,
            },
        }
    )

    # # === Training precision flag === This is unnecessary, unused parameter
    # reduce_in_full_precision: bool = True


@FRAMEWORK_REGISTRY.register("QwenGR00TRollFlow")
class Qwen_GR00T_RollFlow(baseframework):
    """
    Multimodal vision-language-action model with rolling MeanFlow inference.

    Components:
      - Qwen2.5-VL / Qwen3-VL backbone for fused language/vision token embeddings
      - RollFlow-conditioned DiT head for continuous action sequence modeling

    Focus: Predict future continuous actions conditioned on images + instruction.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Construct all submodules and cache key configuration values.

        Args:
            config: Hierarchical configuration (OmegaConf/dict) containing framework + trainer sections.
            **kwargs: Reserved for future overrides (unused).
        """
        super().__init__()
        # Merge framework defaults with YAML config (YAML wins on conflicts)
        self.config = merge_framework_config(QwenGR00TRollFlowDefaultConfig, config)
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        # align dims --> we should put them to config or no?
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = (
            self.qwen_vl_interface.model.config.hidden_size
        )

        self.action_model: RollFlowActionHead = get_action_model(config=self.config)

        # `action_horizon` is the single source of truth for chunk length.
        # Legacy aliases (`future_action_window_size`, `past_action_window_size`)
        # are normalised upstream by `share_tools.apply_config_compat`, so we
        # only ever read `action_horizon` here.
        self.action_horizon = int(self.config.framework.action_model.action_horizon)

    def forward(
        self,
        examples: Optional[List[dict]] = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """ """
        batch_images = [example["image"] for example in examples]  #  [B, [PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]

        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        backbone_attention_mask = qwen_inputs.get("attention_mask", None)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            # last_hidden_state: [B, seq_len, H]
            last_hidden = qwenvl_outputs.hidden_states[-1]  # [B, L, H]

        # Native action widths differ, but all views share one VLM forward.
        groups = {}
        for i, example in enumerate(examples):
            tag = example.get("robot_tag", self.action_model.default_embodiment)
            # Single-embodiment legacy heads do not route by dataset tag.
            if not self.action_model.adapters:
                tag = self.action_model.default_embodiment
            groups.setdefault(tag, []).append(i)
        metrics = {}
        action_loss = last_hidden.new_zeros((), dtype=torch.float32)
        with torch.autocast("cuda", enabled=False):
            for tag, indices in groups.items():
                rows = torch.tensor(indices, device=last_hidden.device)
                group = [examples[i] for i in indices]
                loss = self._native_loss(
                    group, last_hidden[rows],
                    None if backbone_attention_mask is None else backbone_attention_mask[rows],
                    tag, int(kwargs.get("training_step", 0)),
                )
                fraction = len(group) / len(examples)
                action_loss = action_loss + fraction * loss
                for key, value in (self.action_model.last_loss_stats or {}).items():
                    metrics[f"rollflow/{tag}/{key}"] = value
        return {"action_loss": action_loss, **metrics}

    def _native_loss(self, examples, hidden, attention_mask, tag, step):
        repeats = int(self.config.framework.action_model.get("repeated_diffusion_steps", 4))
        def tensor(key):
            return torch.as_tensor(np.stack([e[key] for e in examples]),
                                   device=hidden.device, dtype=torch.float32)
        actions = tensor("action")[:, -self.action_horizon:].repeat(repeats, 1, 1)
        encoder = self.action_model._native_module("state_encoder", tag)
        state = tensor("state").repeat(repeats, 1, 1) if encoder is not None else None
        pad = None
        if any("action_padding_mask" in e for e in examples):
            pad = tensor("action_padding_mask")[:, -self.action_horizon:].bool().repeat(repeats, 1)
        return self.action_model(
            hidden.repeat(repeats, 1, 1), actions, state,
            encoder_attention_mask=None if attention_mask is None else attention_mask.bool().repeat(repeats, 1),
            action_padding_mask=pad, training_step=step, embodiment=tag,
        )

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict],
        **kwargs,
    ) -> dict[str, np.ndarray]:
        """
        Steps:
          1. Resize images to training resolution (if specified)
          2. Encode with QwenVL (hidden states retained)
          6. Return normalized action trajectory
        Returns:
            dict:
                normalized_actions (np.ndarray): Shape [B, T, action_dim], diffusion-sampled normalized actions.
        """
        if type(examples) is not list:
            examples = [examples]
        tag = kwargs.get("embodiment") or examples[0].get("robot_tag", self.action_model.default_embodiment)
        if not self.action_model.adapters:
            tag = self.action_model.default_embodiment
        if any(e.get("robot_tag", tag) != tag for e in examples):
            raise ValueError("Each inference request must use a single embodiment")
        batch_images = [to_pil_preserve(example["image"]) for example in examples]  #  [B, [PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]

        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]
        if self.action_model._native_module("state_encoder", tag) is None:
            state = None

        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        backbone_attention_mask = qwen_inputs.get("attention_mask", None)
        if backbone_attention_mask is not None:
            backbone_attention_mask = backbone_attention_mask.to(dtype=torch.bool)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )

            # last_hidden_state: [B, seq_len, H]
            last_hidden = qwenvl_outputs.hidden_states[-1]  # [B, L, H]

        state = (
            torch.from_numpy(np.array(state)).to(last_hidden.device, dtype=last_hidden.dtype)
            if state is not None
            else None
        )

        # Step 4: Action Expert Forward
        with torch.autocast("cuda", enabled=False):
            pred_actions = self.action_model.predict_action(
                last_hidden,
                state,
                encoder_attention_mask=backbone_attention_mask,
                refinement_steps=kwargs.get("refinement_steps"),
                embodiment=tag,
            )  # (B, chunk_len, action_dim)

        # NumPy has no bfloat16 dtype. DeepSpeed/bf16 evaluation can propagate
        # the backbone dtype through the action head, so normalize the public
        # API to float32 before crossing the Torch/NumPy boundary.
        normalized_actions = pred_actions.detach().float().cpu().numpy()
        return {"normalized_actions": normalized_actions}

    def reset(self) -> None:
        """Clear rolling state at every episode or task boundary."""
        self.action_model.reset_cache()

    reset_cache = reset


if __name__ == "__main__":
    import argparse
    import os

    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/simBenchmarks/LIBERO/train_files/starvla_cotrain_libero.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    if os.getenv("DEBUGPY_ENABLE", "0") == "1":
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)

    model: Qwen_GR00T_RollFlow = Qwen_GR00T_RollFlow(cfg)
    print(model)

    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16),
        "image": [image],
        "lang": "This is a fake instruction for testing.",
    }
    sample2 = sample.copy()
    sample2["lang"] = "Another fake instruction for testing."

    batch = [sample, sample2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    action_loss = forward_output["action_loss"]
    print(f"Action Loss: {action_loss.item()}")

    predict_output = model.predict_action(examples=[sample])
    normalized_actions = predict_output["normalized_actions"]
    print(f"Unnormalized Action: {normalized_actions}")

    print("Finished")
