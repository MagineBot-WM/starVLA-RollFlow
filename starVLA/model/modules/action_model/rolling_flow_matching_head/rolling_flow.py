"""Grouped-time pure flow matching and rolling cache utilities.

This module is intentionally separate from ``rolling_meanflow_matching_head``.
It contains no interval sampling, ``p_fm`` curriculum, or self-distillation.
The caller supplies Groot's ordinary scalar-time prediction function, so the
policy parameterization and DiT forward path stay unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch
from torch import Tensor


@dataclass
class RollingFlowConfig:
    horizon: int
    action_dim: int
    chunk_size: int
    p_k1: float = 0.7
    inference_steps: Optional[int] = None
    use_ot: bool = True
    clip_velocity: float = 0.0
    iterative_cold_start: bool = True
    reset_cache_each_step: bool = False

    def __post_init__(self) -> None:
        if self.horizon <= 0 or self.action_dim <= 0 or self.chunk_size <= 0:
            raise ValueError("horizon, action_dim and chunk_size must be positive")
        if self.horizon % self.chunk_size:
            raise ValueError("horizon must be divisible by chunk_size")
        if not 0.0 <= self.p_k1 <= 1.0:
            raise ValueError("p_k1 must lie in [0, 1]")
        if self.inference_steps is not None and not self.valid_steps(self.inference_steps):
            raise ValueError(
                f"inference_steps must lie in [1,{self.num_action_chunks}]"
            )

    @property
    def num_action_chunks(self) -> int:
        return self.horizon // self.chunk_size

    @property
    def resolved_inference_steps(self) -> int:
        return self.num_action_chunks if self.inference_steps is None else int(self.inference_steps)

    @property
    def training_group_choices(self) -> tuple[int, ...]:
        chunks = self.num_action_chunks
        return tuple(k for k in range(1, chunks + 1) if chunks % k == 0)

    def valid_steps(self, steps: int) -> bool:
        return 1 <= int(steps) <= self.num_action_chunks


@dataclass(frozen=True)
class FlowCacheInfo:
    shape: tuple[int, int, int]
    refinement_steps: int
    device: torch.device
    dtype: torch.dtype


def _repeat_batch(value: Any, repeats: int, batch: int) -> Any:
    """Repeat batch-major tensors in ``[B,K,...]`` order for grouped passes."""
    if isinstance(value, torch.Tensor):
        if value.ndim and value.shape[0] == batch:
            return value[:, None].expand(batch, repeats, *value.shape[1:]).reshape(
                batch * repeats, *value.shape[1:]
            )
        return value
    if isinstance(value, list):
        return [_repeat_batch(item, repeats, batch) for item in value]
    if isinstance(value, tuple):
        return tuple(_repeat_batch(item, repeats, batch) for item in value)
    if isinstance(value, dict):
        return {key: _repeat_batch(item, repeats, batch) for key, item in value.items()}
    return value


@torch.no_grad()
def _ot_match(target: Tensor, noise: Tensor, pad: Optional[Tensor]) -> Tensor:
    """Pair each clean trajectory with a batch noise sample using OT."""
    if target.shape[0] <= 1:
        return noise
    from scipy.optimize import linear_sum_assignment

    target_flat = target.float().flatten(1)
    noise_flat = noise.float().flatten(1)
    if pad is None or not bool(pad.any()):
        cost = torch.cdist(target_flat, noise_flat)
    else:
        valid = (~pad.bool()).unsqueeze(-1).expand_as(target).float().flatten(1)
        denom = valid.sum(1, keepdim=True).clamp_min(1.0)
        target_sq = (valid * target_flat.square()).sum(1, keepdim=True)
        noise_sq = valid @ noise_flat.square().T
        cross = (valid * target_flat) @ noise_flat.T
        cost = (target_sq + noise_sq - 2 * cross).clamp_min_(0) / denom
    rows, cols = linear_sum_assignment(cost.cpu().numpy())
    rows = torch.as_tensor(rows, device=noise.device)
    cols = torch.as_tensor(cols, device=noise.device)
    matched = torch.empty_like(noise)
    matched[rows] = noise[cols]
    return matched


class RollingFlow:
    """Pure FM objective and inference cache; the caller owns Groot's model."""

    def __init__(self, config: RollingFlowConfig):
        self.config = config
        self._cache: Optional[Tensor] = None
        self._cache_steps: Optional[int] = None

    def sample_training_groups(self, device: torch.device) -> int:
        choices = self.config.training_group_choices
        if len(choices) == 1:
            return 1
        if self.config.p_k1 == 1.0:
            return 1
        probabilities = torch.full(
            (len(choices),),
            (1.0 - self.config.p_k1) / (len(choices) - 1),
            device=device,
        )
        probabilities[choices.index(1)] = self.config.p_k1
        index = int(torch.multinomial(probabilities, 1).item())
        return choices[index]

    def training_loss(
        self,
        predict: Callable[..., Tensor],
        actions: Tensor,
        *,
        sample_time: Callable[[int, torch.device, torch.dtype], Tensor],
        context=None,
        pad: Optional[Tensor] = None,
        predict_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[Tensor, dict[str, Any]]:
        """Grouped staircase-time flow matching; every prediction uses scalar t."""
        cfg = self.config
        if actions.ndim != 3 or tuple(actions.shape[1:]) != (cfg.horizon, cfg.action_dim):
            raise ValueError(
                f"actions must have shape [B,{cfg.horizon},{cfg.action_dim}]"
            )
        batch = actions.shape[0]
        if pad is not None:
            if pad.shape != (batch, cfg.horizon):
                raise ValueError(f"pad must have shape [B,{cfg.horizon}]")
            pad = pad.to(device=actions.device, dtype=torch.bool)
            clean = actions.masked_fill(pad.unsqueeze(-1), 0)
        else:
            clean = actions

        noise = torch.randn_like(clean)
        if pad is not None:
            noise = noise.masked_fill(pad.unsqueeze(-1), 0)
        if cfg.use_ot:
            noise = _ot_match(clean, noise, pad)
        if pad is not None:
            noise = noise.masked_fill(pad.unsqueeze(-1), 0)

        groups = self.sample_training_groups(actions.device)
        time_groups = sample_time(batch * groups, actions.device, torch.float32).reshape(
            batch, groups
        )
        time_groups = torch.sort(time_groups, dim=1, descending=True).values
        chunks_per_group = cfg.num_action_chunks // groups
        actions_per_group = chunks_per_group * cfg.chunk_size

        time_for_batch = time_groups.reshape(batch * groups, 1, 1)
        grouped_noise = noise[:, None].expand(-1, groups, -1, -1).reshape(
            batch * groups, cfg.horizon, cfg.action_dim
        )
        grouped_clean = clean[:, None].expand(-1, groups, -1, -1).reshape(
            batch * groups, cfg.horizon, cfg.action_dim
        )
        noisy = (1.0 - time_for_batch) * grouped_noise + time_for_batch * grouped_clean
        expanded_times = time_for_batch.expand(-1, cfg.horizon, 1)
        kwargs = {} if predict_kwargs is None else {
            key: _repeat_batch(value, groups, batch)
            for key, value in predict_kwargs.items()
        }
        expanded_context = _repeat_batch(context, groups, batch)
        prediction = predict(noisy, expanded_times, expanded_context, **kwargs)
        expected_shape = (batch * groups, cfg.horizon, cfg.action_dim)
        if prediction.shape != expected_shape:
            raise ValueError(f"predict returned {tuple(prediction.shape)}, expected {expected_shape}")
        prediction = prediction.reshape(batch, groups, cfg.horizon, cfg.action_dim)
        target_velocity = (clean - noise).float()[:, None].expand_as(prediction)

        group_mask = torch.zeros(
            groups, cfg.horizon, device=actions.device, dtype=torch.bool
        )
        for group in range(groups):
            start = group * actions_per_group
            group_mask[group, start : start + actions_per_group] = True
        valid = group_mask[None, :, :, None].expand(batch, -1, -1, -1)
        if pad is not None:
            valid = valid & ~pad[:, None, :, None]
        error = prediction.float() - target_velocity
        safe_error = torch.where(valid, error, torch.zeros((), device=error.device))
        denominator = valid.sum().to(torch.float32).clamp_min(1.0) * cfg.action_dim
        loss = safe_error.square().sum() / denominator

        return loss, {
            "fm_loss": float(loss.detach().cpu()),
            "num_time_groups": groups,
            "train_block_size": chunks_per_group,
            "t_min": float(time_groups.min().detach().cpu()),
            "t_max": float(time_groups.max().detach().cpu()),
            "t_mean": float(time_groups.mean().detach().cpu()),
            "pad_frac": 0.0 if pad is None else float(pad.float().mean().detach().cpu()),
            "use_ot": bool(cfg.use_ot),
        }

    @property
    def cache_info(self) -> Optional[FlowCacheInfo]:
        if self._cache is None or self._cache_steps is None:
            return None
        return FlowCacheInfo(
            shape=tuple(self._cache.shape),
            refinement_steps=self._cache_steps,
            device=self._cache.device,
            dtype=self._cache.dtype,
        )

    def reset(self) -> None:
        self._cache = None
        self._cache_steps = None

    @torch.no_grad()
    def step(
        self,
        predict: Callable[..., Tensor],
        batch: int,
        *,
        context=None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        refinement_steps: Optional[int] = None,
        predict_kwargs: Optional[dict[str, Any]] = None,
    ) -> Tensor:
        cfg = self.config
        steps = cfg.resolved_inference_steps if refinement_steps is None else int(refinement_steps)
        if not cfg.valid_steps(steps):
            raise ValueError(f"refinement_steps must lie in [1,{cfg.num_action_chunks}]")
        if device is None:
            device = context.device if isinstance(context, Tensor) else self._cache.device
        device = torch.device(device)
        if dtype is None:
            dtype = self._cache.dtype if self._cache is not None else torch.float32

        if cfg.reset_cache_each_step:
            self.reset()
        cache_valid = (
            self._cache is not None
            and self._cache_steps == steps
            and self._cache.shape == (batch, cfg.horizon, cfg.action_dim)
            and self._cache.device == device
            and self._cache.dtype == dtype
        )
        if cache_valid:
            latent = torch.cat(
                (
                    self._cache[:, cfg.chunk_size :],
                    torch.randn(batch, cfg.chunk_size, cfg.action_dim, device=device, dtype=dtype),
                ),
                dim=1,
            )
        else:
            latent = torch.randn(batch, cfg.horizon, cfg.action_dim, device=device, dtype=dtype)

        source, target = self._deployment_times(batch, steps, device)
        if not cache_valid and cfg.iterative_cold_start:
            current = torch.zeros_like(target)
            for level in range(1, steps + 1):
                next_time = torch.minimum(target, torch.full_like(target, level / steps))
                velocity = predict(latent, current, context, **(predict_kwargs or {}))
                velocity = self._clip(velocity)
                latent = (
                    latent.float()
                    + (next_time - current).float() * velocity.float()
                ).to(dtype)
                current = next_time
        else:
            if not cache_valid:
                source = torch.zeros_like(source)
            velocity = predict(latent, source, context, **(predict_kwargs or {}))
            velocity = self._clip(velocity)
            latent = (
                latent.float() + (target - source).float() * velocity.float()
            ).to(dtype)

        self._cache = latent.detach()
        self._cache_steps = steps
        return self._cache[:, : cfg.chunk_size].clone()

    def _clip(self, velocity: Tensor) -> Tensor:
        clip = self.config.clip_velocity
        return velocity if clip <= 0 else clip * torch.tanh(velocity / clip)

    def _deployment_times(self, batch: int, steps: int, device) -> tuple[Tensor, Tensor]:
        cfg = self.config
        chunks = torch.arange(cfg.num_action_chunks, device=device, dtype=torch.float32)
        upper = torch.ones(batch, 1, device=device, dtype=torch.float32)
        lower = torch.zeros_like(upper)
        gap = upper - lower
        t_chunk = upper - torch.clamp(chunks / steps, max=1.0)[None] * gap
        s_chunk = upper - torch.clamp((chunks + 1.0) / steps, max=1.0)[None] * gap
        t = t_chunk.repeat_interleave(cfg.chunk_size, dim=1).unsqueeze(-1)
        s = s_chunk.repeat_interleave(cfg.chunk_size, dim=1).unsqueeze(-1)
        return s, t
