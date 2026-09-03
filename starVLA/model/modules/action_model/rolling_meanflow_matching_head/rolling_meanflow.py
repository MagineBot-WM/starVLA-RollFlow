"""RollFlow: Flow Matching + Central-Difference Lagrangian Self-Distillation.

Core training algorithm
-----------------------
    s, t = sample_staircase_times(delta)
    x_s = (1-s) * x0 + s * x1
    x_t = (1-t) * x0 + t * x1
    v_gt = x1 - x0

    # Build the complete stop-gradient teacher path first.
    with no_grad:
        V_st = model(x_s, s, t)
        X_t_hat = x_s + (t-s) * V_st
        v_teacher = model(X_t_hat, t, t)

    # Pack every gradient-bearing prediction into one forward.
    V_minus, V_plus, v_local = model(
        cat(x_s, x_s, x_t),
        cat(s, s, t),
        cat(t-delta, t+delta, t),
    ).chunk(3)
    X_minus = x_s + (t-delta-s) * V_minus
    X_plus  = x_s + (t+delta-s) * V_plus
    v_tangent = (X_plus - X_minus) / (2*delta)

    loss_fm  = MSE(v_local, v_gt)
    loss_lsd = MSE(v_tangent, stopgrad(v_teacher))
    loss = w_fm * loss_fm + w_lsd * loss_lsd

Time convention
---------------
    x0 = Gaussian prior at flow time 0
    x1 = clean action trajectory at flow time 1
    X_{s,t}(z) = z + (t-s) * u_theta(z,s,t),  s <= t

Training uses a random RollFlow staircase.  With horizon H and execution chunk
size C, M = H/C action chunks exist.  ``train_block_sizes`` controls how many
adjacent action chunks share one flow interval.  A block size of M gives the
ordinary FM time layout (one source/target time for the whole horizon), while a
block size of 1 gives the full M-level rolling staircase.  The legacy
``train_steps`` refinement-depth sampler remains available for old experiments.

There is no JVP, CSF, Split loss, persistent training cache, or adjacent-window
training sampler.  Rolling state exists only at inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import torch
from torch import Tensor, nn

# =============================================================================
# 1. Configuration
# =============================================================================


@dataclass
class RollFlowConfig:
    horizon: int
    action_dim: int
    chunk_size: int

    finite_difference_delta: float = 0.01
    train_steps: Optional[Sequence[int]] = None
    train_block_sizes: Optional[Sequence[int]] = None
    inference_steps: Optional[int] = None

    w_fm: float = 1.0
    w_lsd: float = 0.5
    use_ot: bool = True

    clip_velocity: float = 0.0
    iterative_cold_start: bool = False
    reset_cache_each_step: bool = False

    def __post_init__(self) -> None:
        if self.horizon <= 0 or self.action_dim <= 0:
            raise ValueError("horizon/action_dim must be positive")
        if self.chunk_size <= 0 or self.horizon % self.chunk_size:
            raise ValueError("chunk_size must be positive and divide horizon")
        if not 0.0 < self.finite_difference_delta < 0.5:
            raise ValueError("finite_difference_delta must be in (0, 0.5)")
        if self.w_fm < 0 or self.w_lsd < 0:
            raise ValueError("loss weights must be non-negative")
        if self.clip_velocity < 0:
            raise ValueError("clip_velocity must be non-negative")

        if self.train_steps is not None and self.train_block_sizes is not None:
            raise ValueError("set only one of train_steps and train_block_sizes")

        block_sizes = self.resolved_train_block_sizes
        if block_sizes is None:
            steps = self.resolved_train_steps
            if not steps or any(not self.valid_steps(k) for k in steps):
                raise ValueError(f"train_steps must lie in [1,{self.num_action_chunks}]")
            if len(set(steps)) != len(steps):
                raise ValueError("train_steps must not contain duplicates")
            max_levels = max(steps)
            interval_name = "train_steps"
        else:
            if not block_sizes or any(
                size <= 0 or self.num_action_chunks % size for size in block_sizes
            ):
                raise ValueError(
                    "train_block_sizes must be positive divisors of "
                    f"num_action_chunks={self.num_action_chunks}"
                )
            if len(set(block_sizes)) != len(block_sizes):
                raise ValueError("train_block_sizes must not contain duplicates")
            max_levels = max(self.num_action_chunks // size for size in block_sizes)
            interval_name = "train_block_sizes"

        if not self.valid_steps(self.resolved_inference_steps):
            raise ValueError(
                f"inference_steps must lie in [1,{self.num_action_chunks}]"
            )

        # A K-level central-difference staircase needs K intervals, each wider
        # than delta, and delta room above the largest terminal time.  Keep this
        # check identical to the bounds used by ``sample_training`` so an
        # accepted config cannot fail only when the first batch is sampled.
        eps = max(1e-6, self.finite_difference_delta * 1e-3)
        if max_levels * (self.finite_difference_delta + eps) >= 1.0 - self.finite_difference_delta:
            raise ValueError(
                "finite_difference_delta is too large for the requested "
                f"{interval_name}"
            )

    @property
    def num_action_chunks(self) -> int:
        return self.horizon // self.chunk_size

    @property
    def resolved_train_steps(self) -> tuple[int, ...]:
        if self.train_steps is None:
            return tuple(range(1, self.num_action_chunks + 1))
        return tuple(int(k) for k in self.train_steps)

    @property
    def resolved_train_block_sizes(self) -> Optional[tuple[int, ...]]:
        if self.train_block_sizes is None:
            return None
        return tuple(int(size) for size in self.train_block_sizes)

    @property
    def resolved_inference_steps(self) -> int:
        return (
            self.num_action_chunks
            if self.inference_steps is None
            else int(self.inference_steps)
        )

    def valid_steps(self, k: int) -> bool:
        return 1 <= int(k) <= self.num_action_chunks


@dataclass(frozen=True)
class StaircaseTimes:
    s: Tensor
    t: Tensor
    active: Tensor
    refinement_steps: int
    lower: Tensor
    upper: Tensor
    block_size: Optional[int] = None


@dataclass(frozen=True)
class RollFlowCacheInfo:
    shape: tuple[int, int, int]
    refinement_steps: int
    device: torch.device
    dtype: torch.dtype


# =============================================================================
# 2. Core algorithm: exactly FM + central-difference LSD
# =============================================================================


def lerp_path(x0: Tensor, x1: Tensor, time: Tensor) -> Tensor:
    return (1.0 - time.float()) * x0.float() + time.float() * x1.float()


def flow_map(x_s: Tensor, s: Tensor, t: Tensor, velocity: Tensor) -> Tensor:
    return x_s.float() + (t - s).float() * velocity.float()


def central_difference_lsd(
    model: nn.Module,
    x0: Tensor,
    x1: Tensor,
    s: Tensor,
    t: Tensor,
    *,
    delta: float,
    active: Optional[Tensor] = None,
    context: Optional[Tensor] = None,
    pad: Optional[Tensor] = None,
    w_fm: float = 1.0,
    w_lsd: float = 0.5,
    model_kwargs: Optional[dict[str, Any]] = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Memory-efficient forwards implementing the paper-style objective.

    ``active`` marks tokens that carry a finite off-diagonal interval.  Diagonal
    staircase tokens still receive FM supervision but are excluded from LSD.
    Teacher-only branches run without autograd; this changes memory use, not the
    predicted values or objective.
    """
    kwargs = {} if model_kwargs is None else model_kwargs
    delta = float(delta)
    active = _lsd_mask(s, t, delta) if active is None else active.bool()
    _validate_times(s, t, delta, active)

    x_s = lerp_path(x0, x1, s)
    x_t = lerp_path(x0, x1, t)
    v_gt = x1.float() - x0.float()

    # Diagonal tokens are not used by LSD. Keep their queries diagonal instead
    # of creating invalid backward maps with t-delta < s.
    t_minus = torch.where(active, t - delta, t)
    t_plus = torch.where(active, t + delta, t)

    # -------------------------------------------------------------------------
    # 1. Inference-only self-teacher path
    # -------------------------------------------------------------------------
    # Compute this path before constructing the student graph so its temporary
    # activations are released immediately.
    with torch.no_grad():
        V_st = _predict(model, x_s, s, t, context, kwargs)
        X_t_hat = flow_map(x_s, s, t, V_st)
        v_teacher = _predict(model, X_t_hat, t, t, context, kwargs)

    # -------------------------------------------------------------------------
    # 2. Gradient-bearing tangent endpoints + local FM prediction
    # -------------------------------------------------------------------------
    # Packing all three student predictions keeps the retained graph at 3B
    # while reducing the total number of model calls from four to three.
    V_minus, V_plus, v_local = _parallel_student_predictions(
        model, x_s, x_t, s, t, t_minus, t_plus, context, kwargs
    )

    X_minus = flow_map(x_s, s, t_minus, V_minus)
    X_plus = flow_map(x_s, s, t_plus, V_plus)
    v_tangent = (X_plus - X_minus) / (2.0 * delta)

    loss_fm = masked_mse(v_local - v_gt, pad=pad)
    loss_lsd = masked_mse(v_tangent - v_teacher, pad=pad, valid=active)
    loss = w_fm * loss_fm + w_lsd * loss_lsd

    return loss, {
        "fm_loss": loss_fm.detach(),
        "lsd_loss": loss_lsd.detach(),
        "active_lsd_frac": active.float().mean().detach(),
        "v_local_abs": v_local.detach().abs().mean(),
        "v_tangent_abs": v_tangent.detach().abs().mean(),
        "v_teacher_abs": v_teacher.abs().mean(),
    }


# =============================================================================
# 3. Staircase time sampling
# =============================================================================


class StaircaseTimeSampler:
    """Grouped random training times + exact deployment staircase.

    During training, a block size G makes each group of G action chunks share
    one interval.  For L=M/G levels, chunk j uses ``level=floor(j/G)`` and

        t_j = T - level/L     * (T-R)
        s_j = T - (level+1)/L * (T-R)

    Thus G=M is the ordinary whole-horizon FM time layout, and G=1 is the full
    rolling staircase.

    Deployment retains the refinement-depth K layout required for exact cache
    alignment after shifting one execution chunk:

    For a sampled interval [R,T] and K refinement levels, chunk j uses

        t_j = T - min(j,   K)/K * (T-R)
        s_j = T - min(j+1, K)/K * (T-R)

    so for j < K:

        t_{j+1} = s_j,

    while j >= K is diagonal at R.  The front chunk is closest to the clean
    endpoint; future chunks are progressively noisier.
    """

    def __init__(self, cfg: RollFlowConfig):
        self.cfg = cfg
        self._steps = torch.tensor(cfg.resolved_train_steps, dtype=torch.long)
        block_sizes = cfg.resolved_train_block_sizes
        self._block_sizes = (
            None if block_sizes is None else torch.tensor(block_sizes, dtype=torch.long)
        )

    def sample_training(
        self,
        batch: int,
        *,
        device,
        dtype=torch.float32,
        generator=None,
        refinement_steps: Optional[int] = None,
        block_size: Optional[int] = None,
    ) -> StaircaseTimes:
        if batch <= 0:
            raise ValueError("batch must be positive")
        if refinement_steps is not None and block_size is not None:
            raise ValueError("set only one of refinement_steps and block_size")
        if self._block_sizes is not None and refinement_steps is not None:
            raise ValueError("grouped training uses block_size, not refinement_steps")
        device = torch.device(device)

        grouped = block_size is not None or self._block_sizes is not None
        if grouped:
            G = (
                self._sample_value(self._block_sizes, device, generator)
                if block_size is None
                else int(block_size)
            )
            if G <= 0 or self.cfg.num_action_chunks % G:
                raise ValueError(
                    "block_size must be a positive divisor of "
                    f"num_action_chunks={self.cfg.num_action_chunks}"
                )
            levels = self.cfg.num_action_chunks // G
        else:
            G = None
            levels = (
                self._sample_value(self._steps, device, generator)
                if refinement_steps is None
                else int(refinement_steps)
            )
            if not self.cfg.valid_steps(levels):
                raise ValueError(
                    f"refinement_steps must lie in [1,{self.cfg.num_action_chunks}]"
                )

        delta = self.cfg.finite_difference_delta
        eps = max(1e-6, delta * 1e-3)
        min_gap = levels * (delta + eps)
        max_gap = 1.0 - delta
        if min_gap >= max_gap:
            raise ValueError("no valid central-difference interval for these training times")

        # Random global interval [R,T], with enough width for every adjacent,
        # central-difference-valid subinterval and delta room above T.
        gap = torch.empty(batch, device=device, dtype=dtype).uniform_(
            min_gap, max_gap, generator=generator
        )
        lower = torch.rand(batch, device=device, dtype=dtype, generator=generator)
        lower = lower * (1.0 - delta - gap)
        upper = lower + gap

        if G is None:
            s, t, active = self._build(lower, upper, levels)
        else:
            s, t, active = self._build_grouped(lower, upper, G)
        return StaircaseTimes(s, t, active, levels, lower, upper, G)

    def deployment(
        self,
        batch: int,
        *,
        refinement_steps: int,
        cold: bool,
        device,
        dtype=torch.float32,
    ) -> StaircaseTimes:
        K = int(refinement_steps)
        if not self.cfg.valid_steps(K):
            raise ValueError(f"refinement_steps must lie in [1,{self.cfg.num_action_chunks}]")

        lower = torch.zeros(batch, device=device, dtype=dtype)
        upper = torch.ones(batch, device=device, dtype=dtype)
        s, t, active = self._build(lower, upper, K)
        if cold:
            # One-pass cold start: every token begins at the prior time 0 and
            # jumps directly to its staircase target t.
            s = torch.zeros_like(s)
            active = t > 0
        return StaircaseTimes(s, t, active, K, lower, upper)

    @staticmethod
    def _sample_value(values, device, generator) -> int:
        if values is None:
            raise ValueError("no configured training values to sample")
        idx = int(
            torch.randint(
                len(values), (), device=device, generator=generator
            ).item()
        )
        return int(values[idx])

    def _build(
        self, lower: Tensor, upper: Tensor, K: int
    ) -> tuple[Tensor, Tensor, Tensor]:
        cfg = self.cfg
        M = cfg.num_action_chunks
        j = torch.arange(M, device=lower.device, dtype=lower.dtype)
        gap = (upper - lower)[:, None]

        alpha_t = torch.clamp(j / K, max=1.0)[None, :]
        alpha_s = torch.clamp((j + 1.0) / K, max=1.0)[None, :]
        t_chunk = upper[:, None] - gap * alpha_t
        s_chunk = upper[:, None] - gap * alpha_s

        t = t_chunk.repeat_interleave(cfg.chunk_size, dim=1).unsqueeze(-1)
        s = s_chunk.repeat_interleave(cfg.chunk_size, dim=1).unsqueeze(-1)
        active = (t - s) > self.cfg.finite_difference_delta
        return s, t, active

    def _build_grouped(
        self, lower: Tensor, upper: Tensor, block_size: int
    ) -> tuple[Tensor, Tensor, Tensor]:
        cfg = self.cfg
        levels = cfg.num_action_chunks // block_size
        chunk = torch.arange(
            cfg.num_action_chunks, device=lower.device, dtype=lower.dtype
        )
        level = torch.floor(chunk / block_size)
        gap = (upper - lower)[:, None]

        t_chunk = upper[:, None] - gap * (level / levels)[None, :]
        s_chunk = upper[:, None] - gap * ((level + 1.0) / levels)[None, :]
        t = t_chunk.repeat_interleave(cfg.chunk_size, dim=1).unsqueeze(-1)
        s = s_chunk.repeat_interleave(cfg.chunk_size, dim=1).unsqueeze(-1)
        active = (t - s) > cfg.finite_difference_delta
        return s, t, active


# =============================================================================
# 4. Engineering helpers
# =============================================================================


def _lsd_mask(s: Tensor, t: Tensor, delta: float) -> Tensor:
    return (t - s) > float(delta)


def _validate_times(s: Tensor, t: Tensor, delta: float, active: Tensor) -> None:
    tol = 1e-6
    if s.shape != t.shape:
        raise ValueError("s and t must have identical shapes")
    if bool((s < -tol).any()) or bool((t > 1.0 + tol).any()) or bool((s > t + tol).any()):
        raise ValueError("require 0 <= s <= t <= 1")
    if bool((active & (s >= t - delta - tol)).any()):
        raise ValueError("active LSD tokens require s < t-delta")
    if bool((active & (t + delta > 1.0 + tol)).any()):
        raise ValueError("active LSD tokens require t+delta <= 1")


def _predict(
    model: nn.Module,
    z: Tensor,
    s: Tensor,
    t: Tensor,
    context: Optional[Tensor],
    kwargs: dict[str, Any],
) -> Tensor:
    out = model(z, s, t, context, **kwargs)
    if out.shape != z.shape:
        raise ValueError(f"model output {tuple(out.shape)} must match z {tuple(z.shape)}")
    return out


def _parallel_student_predictions(
    model,
    x_s,
    x_t,
    s,
    t,
    t_minus,
    t_plus,
    context,
    kwargs,
):
    B = x_s.shape[0]
    out = _predict(
        model,
        torch.cat((x_s, x_s, x_t), dim=0),
        torch.cat((s, s, t), dim=0),
        torch.cat((t_minus, t_plus, t), dim=0),
        repeat_batch(context, 3, B),
        repeat_batch(kwargs, 3, B),
    )
    return out.chunk(3, dim=0)


def repeat_batch(value: Any, repeats: int, batch: int) -> Any:
    """Repeat batch-aligned tensors recursively; leave constants unchanged."""
    if value is None:
        return None
    if torch.is_tensor(value):
        return torch.cat([value] * repeats, dim=0) if value.ndim and value.shape[0] == batch else value
    if isinstance(value, dict):
        return {k: repeat_batch(v, repeats, batch) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(repeat_batch(v, repeats, batch) for v in value)
    if isinstance(value, list):
        return [repeat_batch(v, repeats, batch) for v in value]
    return value


def masked_mse(
    error: Tensor,
    *,
    pad: Optional[Tensor] = None,
    valid: Optional[Tensor] = None,
) -> Tensor:
    loss = error.float().pow(2).mean(dim=-1)
    mask = torch.ones_like(loss, dtype=torch.bool)
    if pad is not None:
        mask &= ~pad.bool()
    if valid is not None:
        v = valid.squeeze(-1) if valid.ndim == 3 else valid
        mask &= v.bool()
    if not bool(mask.any()):
        # Summing an empty differentiable view returns a graph-connected zero,
        # allowing distributed/all-padding batches to participate in backward.
        return loss[mask].sum()
    return loss[mask].mean()


@torch.no_grad()
def ot_match(x1: Tensor, x0: Tensor) -> Tensor:
    if x1.shape[0] <= 1:
        return x0
    from scipy.optimize import linear_sum_assignment

    cost = torch.cdist(x1.float().flatten(1), x0.float().flatten(1)).cpu()
    row, col = linear_sum_assignment(cost.numpy())
    row = torch.as_tensor(row, device=x0.device)
    col = torch.as_tensor(col, device=x0.device)
    matched = torch.empty_like(x0)
    matched[row] = x0[col]
    return matched


def infer_device_dtype(model, context=None, device=None, dtype=None):
    if context is not None:
        return (
            context.device if device is None else torch.device(device),
            context.dtype if dtype is None else dtype,
        )
    try:
        p = next(model.parameters())
        return (
            p.device if device is None else torch.device(device),
            p.dtype if dtype is None else dtype,
        )
    except StopIteration:
        return (
            torch.device("cpu") if device is None else torch.device(device),
            torch.float32 if dtype is None else dtype,
        )


# =============================================================================
# 5. Public wrapper: ordinary training samples + rolling inference
# =============================================================================


class RollFlow:
    def __init__(self, cfg: RollFlowConfig):
        self.cfg = cfg
        self.time = StaircaseTimeSampler(cfg)
        self._cache: Optional[Tensor] = None
        self._cache_steps: Optional[int] = None

    def loss(
        self,
        model: nn.Module,
        x: Tensor,
        step: int = 0,
        context: Optional[Tensor] = None,
        pad: Optional[Tensor] = None,
        **model_kwargs,
    ) -> tuple[Tensor, dict[str, Any]]:
        del step  # API compatibility only; training is stationary.
        self._check_actions(x)
        cfg = self.cfg
        B = x.shape[0]
        if pad is not None and pad.shape != (B, cfg.horizon):
            raise ValueError(f"pad must be [B,{cfg.horizon}]")

        x1 = x
        x0 = torch.randn_like(x1)
        if cfg.use_ot:
            x0 = ot_match(x1, x0)

        times = self.time.sample_training(B, device=x.device)
        loss, stats = central_difference_lsd(
            model,
            x0,
            x1,
            times.s,
            times.t,
            delta=cfg.finite_difference_delta,
            active=times.active,
            context=context,
            pad=pad,
            w_fm=cfg.w_fm,
            w_lsd=cfg.w_lsd,
            model_kwargs=model_kwargs,
        )

        return loss, {
            "loss": float(loss.detach().cpu()),
            "fm_loss": float(stats["fm_loss"].cpu()),
            "lsd_loss": float(stats["lsd_loss"].cpu()),
            "active_lsd_frac": float(stats["active_lsd_frac"].cpu()),
            "refinement_steps": times.refinement_steps,
            "num_time_levels": times.refinement_steps,
            "train_block_size": times.block_size,
            "interval_lower": float(times.lower.mean().cpu()),
            "interval_upper": float(times.upper.mean().cpu()),
            "s_min": float(times.s.min().cpu()),
            "s_max": float(times.s.max().cpu()),
            "t_min": float(times.t.min().cpu()),
            "t_max": float(times.t.max().cpu()),
            "use_ot": bool(cfg.use_ot),
        }

    def reset(self) -> None:
        self._cache = None
        self._cache_steps = None

    @property
    def cache_info(self) -> Optional[RollFlowCacheInfo]:
        if self._cache is None or self._cache_steps is None:
            return None
        return RollFlowCacheInfo(
            tuple(self._cache.shape),
            self._cache_steps,
            self._cache.device,
            self._cache.dtype,
        )

    @torch.no_grad()
    def step(
        self,
        model: nn.Module,
        batch: int,
        context: Optional[Tensor] = None,
        *,
        device=None,
        dtype=None,
        refinement_steps: Optional[int] = None,
        **model_kwargs,
    ) -> Tensor:
        cfg = self.cfg
        K = cfg.resolved_inference_steps if refinement_steps is None else int(refinement_steps)
        if not cfg.valid_steps(K):
            raise ValueError(f"refinement_steps must lie in [1,{cfg.num_action_chunks}]")

        device, dtype = infer_device_dtype(model, context, device, dtype)
        if cfg.reset_cache_each_step:
            self.reset()

        z, cold = self._rolling_input(batch, device, dtype, K)
        times = self.time.deployment(
            batch,
            refinement_steps=K,
            cold=cold,
            device=device,
        )

        if cold and cfg.iterative_cold_start:
            cache = self._cold_iterative(
                model, z, times.t, K, context, model_kwargs, dtype
            )
        else:
            velocity = self._predict_inference(
                model, z, times.s, times.t, context, model_kwargs
            )
            cache = flow_map(z, times.s, times.t, velocity).to(dtype)

        self._cache = cache.detach()
        self._cache_steps = K
        return self._cache[:, : cfg.chunk_size].clone()

    def _rolling_input(self, batch, device, dtype, K):
        cfg = self.cfg
        valid = (
            self._cache is not None
            and self._cache_steps == K
            and self._cache.shape == (batch, cfg.horizon, cfg.action_dim)
            and self._cache.device == device
            and self._cache.dtype == dtype
        )
        if not valid:
            return torch.randn(batch, cfg.horizon, cfg.action_dim, device=device, dtype=dtype), True

        keep = self._cache[:, cfg.chunk_size :]
        tail = torch.randn(batch, cfg.chunk_size, cfg.action_dim, device=device, dtype=dtype)
        return torch.cat((keep, tail), dim=1), False

    def _cold_iterative(self, model, z, final_t, K, context, kwargs, dtype):
        current_s = torch.zeros_like(final_t)
        for level in range(1, K + 1):
            next_t = torch.minimum(final_t, torch.full_like(final_t, level / K))
            v = self._predict_inference(model, z, current_s, next_t, context, kwargs)
            z = flow_map(z, current_s, next_t, v).to(dtype)
            current_s = next_t
        return z

    def _predict_inference(self, model, z, s, t, context, kwargs):
        v = _predict(model, z, s, t, context, kwargs)
        clip = self.cfg.clip_velocity
        return v if clip <= 0 else clip * torch.tanh(v / clip)

    def _check_actions(self, x: Tensor) -> None:
        expected = (self.cfg.horizon, self.cfg.action_dim)
        if x.ndim != 3 or tuple(x.shape[1:]) != expected:
            raise ValueError(f"x must be [B,{expected[0]},{expected[1]}]")
