from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# G1 right-arm experiment note:
# The runnable training tree is /data/tzq/starVLA-RollFlow, whose ACT.py is
# intentionally kept byte-identical to this workspace copy.  This workspace
# snapshot contains the experiment-specific ACT implementation but not the
# complete training package (for example, starVLA/training/train_starvla.py),
# so launches use the full tree.  Mirror any future ACT changes to that tree
# before restarting a run, then verify the two files have matching hashes.

# LeRobot is an optional dependency used only by the ACT framework. Guard the import so
# `build_framework()`'s auto-discovery pass (which imports every framework module) still
# succeeds in environments without LeRobot; instantiating ACT raises a clear error instead.
try:
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.constants import ACTION, OBS_STATE
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy

    _LEROBOT_IMPORT_ERROR: ImportError | None = None
except ImportError as _exc:  # pragma: no cover - exercised only when lerobot is absent
    FeatureType = PolicyFeature = ACTConfig = ACTPolicy = None
    ACTION, OBS_STATE = "action", "observation.state"
    _LEROBOT_IMPORT_ERROR = _exc

from starVLA.model.framework.base_framework import FRAMEWORK_REGISTRY, baseframework
from starVLA.model.framework.share_tools import merge_framework_config


@dataclass
class ACTDefaultConfig:
    name: str = "ACT"
    chunk_size: int = 16
    action_dim: int = 8  # 8D joints: 7 delta joints + 1 gripper abs
    state_dim: int = 8
    n_obs_steps: int = 1
    image_keys: tuple[str, ...] = ("cam0_rgb", "cam1_rgb")
    # Align with train_realman_pi.yaml obs_image_size / QwenPI; deployment must resize to this too.
    image_size: tuple[int, int] = (224, 224)
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    action_loss_weights: tuple[float, ...] | None = None
    action_loss_mask: tuple[float, ...] | None = None


def _image_feature_name(image_key: str) -> str:
    if image_key.startswith("observation.images."):
        return image_key
    return f"observation.images.{image_key}"


def _get_image_size(config) -> tuple[int, int]:
    image_size = tuple(config.get("image_size", ACTDefaultConfig.image_size))
    return int(image_size[0]), int(image_size[1])


def _build_input_features(config) -> dict[str, PolicyFeature]:
    image_keys = config.get("image_keys", ACTDefaultConfig.image_keys)
    image_height, image_width = _get_image_size(config)
    features = {
        _image_feature_name(image_key): PolicyFeature(
            type=FeatureType.VISUAL, shape=(3, image_height, image_width)
        )
        for image_key in image_keys
    }
    features[OBS_STATE] = PolicyFeature(
        type=FeatureType.STATE, shape=(int(config.state_dim),)
    )
    return features


def _build_output_features(config) -> dict[str, PolicyFeature]:
    return {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(int(config.action_dim),))
    }


@FRAMEWORK_REGISTRY.register("ACT")
class ACT(baseframework):
    def __init__(self, config, **kwargs):
        if _LEROBOT_IMPORT_ERROR is not None:
            raise ImportError(
                "The ACT framework requires the optional dependency `lerobot` "
                "(e.g. `pip install lerobot`). Framework discovery works without it, "
                "but ACT cannot be instantiated."
            ) from _LEROBOT_IMPORT_ERROR
        super().__init__()
        self.config = merge_framework_config(ACTDefaultConfig, config)
        framework_config = self.config.framework
        act_config = ACTConfig(
            n_obs_steps=int(framework_config.n_obs_steps),
            chunk_size=int(framework_config.chunk_size),
            n_action_steps=int(framework_config.chunk_size),
            input_features=_build_input_features(framework_config),
            output_features=_build_output_features(framework_config),
            pretrained_backbone_weights=framework_config.get(
                "pretrained_backbone_weights", None
            ),
        )
        self.action_model = ACTPolicy(act_config)
        self._set_identity_normalizers()

    def _set_identity_normalizers(self) -> None:
        # starVLA StateActionTransform/PolicyNormProcessor own normalization.
        # LeRobot ACTPolicy normalizers must be identity to avoid double normalization.
        with torch.no_grad():
            normalizers = (
                self.action_model.normalize_inputs,
                self.action_model.normalize_targets,
                self.action_model.unnormalize_outputs,
            )
            for normalizer in normalizers:
                tensors = list(normalizer.named_buffers()) + list(
                    normalizer.named_parameters()
                )
                for name, tensor in tensors:
                    if name.endswith(("mean", "min")):
                        tensor.zero_()
                    elif name.endswith(("std", "max")):
                        tensor.fill_(1.0)

    def _examples_to_act_batch(
        self, examples: List[dict], require_action: bool = True
    ) -> dict:
        framework_config = self.config.framework
        image_keys = list(
            framework_config.get("image_keys", ACTDefaultConfig.image_keys)
        )
        chunk_size = int(framework_config.chunk_size)
        action_dim = int(framework_config.action_dim)
        state_dim = int(framework_config.state_dim)
        image_height, image_width = _get_image_size(framework_config)
        target_image_size = (image_width, image_height)
        model_param = next(self.action_model.parameters())
        device = model_param.device
        model_dtype = model_param.dtype

        batch = {}

        # Convention: example["image"][i] is paired with framework.image_keys[i].
        for view_idx, image_key in enumerate(image_keys):
            # Fast path: batch uint8 HWC worker output for one H2D transfer and GPU normalization
            # (only when every frame already matches the configured input size); otherwise fall
            # back to the PIL path below, which also resizes ndarray frames of any resolution.
            view_frames = [example["image"][view_idx] for example in examples]
            if isinstance(view_frames[0], np.ndarray) and all(
                isinstance(f, np.ndarray) and f.shape[:2] == (image_height, image_width)
                for f in view_frames
            ):
                arr = np.stack(view_frames, axis=0)
                t = torch.from_numpy(arr).to(device, non_blocking=True)
                t = t.permute(0, 3, 1, 2).to(torch.float32).div_(255.0).to(model_dtype)
                batch[_image_feature_name(image_key)] = t
                continue
            images = []
            for example in examples:
                image = example["image"][view_idx]
                if isinstance(image, np.ndarray):
                    image = Image.fromarray(image)
                image = image.convert("RGB")
                if image.size != target_image_size:
                    image = image.resize(target_image_size)
                image_array = np.asarray(image, dtype=np.float32) / 255.0
                image_tensor = (
                    torch.from_numpy(image_array)
                    .permute(2, 0, 1)
                    .to(device=device, dtype=model_dtype)
                )
                images.append(image_tensor)
            batch[_image_feature_name(image_key)] = torch.stack(images, dim=0)

        states = []
        for example in examples:
            state = example.get("state", np.zeros(state_dim, dtype=np.float32))
            state_tensor = torch.as_tensor(
                state, dtype=model_dtype, device=device
            ).reshape(-1)
            if state_tensor.numel() != state_dim:
                raise ValueError(
                    f"ACT state must have {state_dim} values, got {state_tensor.numel()}"
                )
            states.append(state_tensor)
        batch[OBS_STATE] = torch.stack(states, dim=0)

        if not require_action:
            return batch

        actions = []
        for example in examples:
            action_tensor = torch.as_tensor(
                example["action"], dtype=model_dtype, device=device
            )
            if action_tensor.ndim != 2 or action_tensor.shape[-1] != action_dim:
                raise ValueError(
                    f"ACT action must have shape [T, {action_dim}], got {tuple(action_tensor.shape)}"
                )
            if action_tensor.shape[0] < chunk_size:
                raise ValueError(
                    f"ACT action horizon must be at least {chunk_size}, got {action_tensor.shape[0]}"
                )
            # By default preserve the legacy tail-window behavior. New
            # aligned runs can set action_start_index=0 so observation t is
            # supervised against actions t..t+chunk_size-1.
            action_start = self.config.framework.get("action_start_index", None)
            if action_start is None:
                actions.append(action_tensor[-chunk_size:])
            else:
                action_start = int(action_start)
                action_end = action_start + chunk_size
                if action_start < 0 or action_end > action_tensor.shape[0]:
                    raise ValueError(
                        f"ACT action_start_index={action_start} with chunk_size={chunk_size} "
                        f"does not fit action window of length {action_tensor.shape[0]}"
                    )
                actions.append(action_tensor[action_start:action_end])
        batch[ACTION] = torch.stack(actions, dim=0)
        batch["action_is_pad"] = torch.zeros(
            len(examples), chunk_size, dtype=torch.bool, device=device
        )

        return batch

    def forward(self, examples: List[dict], **kwargs) -> dict:
        batch = self._examples_to_act_batch(examples)
        # Use ACTPolicy internals to get predictions, then compute weighted loss
        # (bypasses ACTPolicy.forward's unweighted L1 loss)
        batch_norm = self.action_model.normalize_inputs(dict(batch))
        if self.action_model.config.image_features:
            from lerobot.constants import OBS_IMAGES as _OBS
            batch_norm[_OBS] = [batch_norm[k] for k in self.action_model.config.image_features]
        batch_norm = self.action_model.normalize_targets(batch_norm)
        actions_hat, (mu, log_sig2) = self.action_model.model(batch_norm)

        configured_weights = self.config.framework.get("action_loss_mask", None)
        if configured_weights is None:
            configured_weights = self.config.framework.get("action_loss_weights", None)
        if configured_weights is None:
            w = torch.ones(actions_hat.shape[-1], device=actions_hat.device, dtype=actions_hat.dtype)
        else:
            if len(configured_weights) != actions_hat.shape[-1]:
                raise ValueError(
                    f"ACT action_loss_weights must have {actions_hat.shape[-1]} values, "
                    f"got {len(configured_weights)}"
                )
            w = torch.as_tensor(configured_weights, device=actions_hat.device, dtype=actions_hat.dtype)

        pad_mask = ~batch_norm["action_is_pad"].unsqueeze(-1)  # (B, T, 1)
        l1_raw = F.l1_loss(batch_norm["action"], actions_hat, reduction="none")  # (B, T, D)
        l1_loss = (l1_raw * pad_mask * w).sum() / (pad_mask * w).sum().clamp(min=1)

        loss = l1_loss
        if self.action_model.config.use_vae:
            kld = (-0.5 * (1 + log_sig2 - mu.pow(2) - log_sig2.exp())).sum(-1).mean()
            loss = l1_loss + kld * self.action_model.config.kl_weight
        return {"action_loss": loss}

    @torch.inference_mode()
    def predict_action(self, examples: List[dict], **kwargs) -> dict:
        batch = self._examples_to_act_batch(examples, require_action=False)
        was_training = self.action_model.training
        model_param = next(self.action_model.parameters())
        autocast_enabled = model_param.dtype in (
            torch.float16,
            torch.bfloat16,
        ) and model_param.device.type in (
            "cuda",
            "cpu",
        )
        try:
            if autocast_enabled:
                with torch.autocast(model_param.device.type, dtype=model_param.dtype):
                    action_chunk = self.action_model.predict_action_chunk(batch)
            else:
                action_chunk = self.action_model.predict_action_chunk(batch)
        finally:
            self.action_model.train(was_training)
        normalized_actions = action_chunk.detach().float().cpu().numpy()
        return {"normalized_actions": normalized_actions}
