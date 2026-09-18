"""RollFlow: Flow Matching + detached MeanFlow interval regularization.
This is the core implementation of the rolling meanflow matching algorithm.
Do not delete this file; it is the core of the rolling inference and training algorithm.

Core training algorithm
-----------------------
    # 1. Sample a batch of source/target pairs (x0, x1) from the action trajectory.
    s, t = sample_training_times()
    x_s = (1-s) * x0 + s * x1
    x_t = (1-t) * x0 + t * x1
    v_gt = x1 - x0


    # 2. Predict interval and instantaneous velocities with one graph-bearing
    # forward.  ``u_st`` is the average velocity on [s, t], while v_instant is
    # the diagonal instantaneous prediction at x_t.
    u_st, v_instant = model(
        cat(x_s, x_t),
        cat(s, t),
        cat(t, t),
    ).chunk(2)

    # 3. Build a detached MeanFlow target for the interval velocity.  The
    # finite difference estimates the local change of u_st with terminal time;
    # ``mf_kv`` bounds the finite-difference derivative component-wise.
    with no_grad:
        u_minus, u_plus = model(
            cat(x_s, x_s),
            cat(s, s),
            cat(t-delta, t+delta),
        ).chunk(2)
        du_dt = (u_plus - u_minus) / (2.0 * delta)

    V_mf = u_st + (t - s) * du_dt.clamp(-mf_kv, mf_kv)

    # 4. FM is always present; MeanFlow is a weak interval regularizer.
    loss_fm = MSE(v_instant, v_gt)
    loss_mf = MSE(V_mf, v_gt)
    use_mf = isfinite(loss_mf) and (
        mf_loss_threshold is None or loss_mf <= mf_loss_threshold
    )
    loss = loss_fm + mf_weight * (loss_mf if use_mf else 0)

``mf_loss_threshold`` is an optional batch-level safety gate.  If the raw
interval loss exceeds the threshold (or is non-finite), the MeanFlow term is
set to zero for that update; FM remains unchanged and always trains.  The
default threshold is ``0.5``; set it to ``None`` to disable the gate.



Time convention
---------------
    x0 = Gaussian prior at flow time 0
    x1 = clean action trajectory at flow time 1
    s = source time
    t = terminal time
    X_{s,t}(z) = z + (t-s) * u_theta(z,s,t),  s <= t
Time sampling
-------------
    Input: horizon H, chunk size C, probabilities p_k1, p_fm

    1. Temporal refinement:
        M = H / C, K ∈ Div(M)
        P(K=1)=p_k1, P(K>1)=(1-p_k1)/(#K-1)
        Sample K

    2. Ordered time sampling:
        Sample: t_i ~ U(0,1)
        Sort: t_1 > t_2 > ... > t_K

    3. Interval sampling:
        Sample: ratio ~ p_fm·δ(1)+(1-p_fm)·U(0,1)

    4. Construct:
        s_i = ratio * t_i

Training samples K from the divisors of M=H/C.  K=1 is favoured, while the
remaining choices share the residual probability.  K independently sorted
target times are expanded over equal action blocks, and one source ratio per
sample gives s=ratio*t.  The first ``fm_only_steps`` (default 30,000) are
diagonal instantaneous FM; afterwards the sampler switches directly to the
fixed ``p_fm`` mixture.  Deployment still uses the exact rolling staircase
needed for cache alignment.
When a batch is entirely diagonal (including FM warmup), the interval forward
is skipped; one local FM forward is sufficient and the interval regularizer is
zero; FM retains the training graph.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import torch
from torch import Tensor, nn


_VELOCITY_MODES = ("average", "instant")


@dataclass
class RollFlowConfig:
    horizon: int
    action_dim: int
    chunk_size: int
    finite_difference_delta: float = 0.01
    inference_steps: Optional[int] = None
    p_k1: float = 0.7
    p_fm: float = 0.7
    fm_only_steps: int = 30_000
    # Component-wise clamp for the detached terminal-time derivative.
    mf_kv: float = 1.0
    mf_weight: float = 0.1
    # Disable the interval regularizer for abnormal batches.  ``None`` keeps
    # the historical ungated objective.
    mf_loss_threshold: Optional[float] = 0.5
    use_ot: bool = True
    action_loss_weights: Optional[Sequence[float]] = None
    clip_velocity: float = 0.0
    iterative_cold_start: bool = False
    reset_cache_each_step: bool = False

    def __post_init__(self):
        if self.horizon <= 0 or self.action_dim <= 0 or self.chunk_size <= 0:
            raise ValueError("horizon/action_dim/chunk_size must be positive")
        if self.horizon % self.chunk_size:
            raise ValueError("chunk_size must divide horizon")
        if not 0 < self.finite_difference_delta < 0.5:
            raise ValueError("finite_difference_delta must lie in (0, 0.5)")
        if not 0 <= self.p_k1 <= 1 or not 0 <= self.p_fm <= 1:
            raise ValueError("p_k1 and p_fm must lie in [0, 1]")
        numeric = (self.mf_kv, self.mf_weight, self.clip_velocity, self.fm_only_steps)
        if any(not math.isfinite(value) or value < 0 for value in numeric):
            raise ValueError(
                "mf_kv/mf_weight/clip_velocity/fm_only_steps must be finite "
                "and non-negative"
            )
        _validate_mf_loss_threshold(self.mf_loss_threshold)
        if not self.valid_steps(self.steps):
            raise ValueError(f"inference_steps must lie in [1,{self.num_chunks}]")
        if self.action_loss_weights is not None:
            w = torch.as_tensor(self.action_loss_weights, dtype=torch.float32)
            if w.shape != (self.action_dim,):
                raise ValueError("action_loss_weights must have one entry per action dimension")
            if bool((w < 0).any()) or not bool(torch.isfinite(w).all()):
                raise ValueError("action_loss_weights must be finite and non-negative")
            if w.sum() <= 0:
                raise ValueError("action_loss_weights must contain a positive value")
            self.action_loss_weights = tuple((w / w.mean()).tolist())

    @property
    def num_chunks(self) -> int:
        return self.horizon // self.chunk_size

    @property
    def num_action_chunks(self) -> int:  # compatibility alias
        return self.num_chunks

    @property
    def steps(self) -> int:
        return self.num_chunks if self.inference_steps is None else int(self.inference_steps)

    @property
    def resolved_inference_steps(self) -> int:  # compatibility alias
        return self.steps

    def valid_steps(self, k: int) -> bool:
        return 1 <= int(k) <= self.num_chunks


@dataclass(frozen=True)
class TrainingTimes:
    s: Tensor
    t: Tensor
    active: Tensor
    num_time_groups: int
    block_size: int
    ratio: Tensor
    fm_mask: Tensor


@dataclass(frozen=True)
class StaircaseTimes:
    s: Tensor
    t: Tensor
    active: Tensor
    refinement_steps: int
    lower: Tensor
    upper: Tensor


@dataclass(frozen=True)
class RollFlowCacheInfo:
    shape: tuple[int, int, int]
    refinement_steps: int
    device: torch.device
    dtype: torch.dtype


# --- Core math ----------------------------------------------------------------


def lerp_path(x0: Tensor, x1: Tensor, t: Tensor) -> Tensor:
    return (1 - t.float()) * x0.float() + t.float() * x1.float()


def flow_map(x: Tensor, s: Tensor, t: Tensor, v: Tensor) -> Tensor:
    return x.float() + (t - s).float() * v.float()


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
    mf_kv: float = 1.0,
    mf_weight: float = 0.1,
    mf_loss_threshold: Optional[float] = 0.5,
    action_loss_weights: Optional[Tensor] = None,
    model_kwargs: Optional[dict[str, Any]] = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Train instantaneous FM and the detached MeanFlow interval objective."""
    kwargs = model_kwargs or {}
    _validate_inputs(
        x0,
        x1,
        s,
        t,
        active,
        pad,
        delta,
        mf_kv,
        mf_weight,
        mf_loss_threshold,
    )
    valid = torch.ones_like(s[..., 0], dtype=torch.bool) if pad is None else ~pad.bool()
    active = _lsd_mask(s, t, delta) if active is None else active.bool()
    _validate_times(s, t, delta, active)
    active = active & valid.unsqueeze(-1)

    x_t = lerp_path(x0, x1, t)
    v_gt = x1.float() - x0.float()
    _require_finite(v_gt, "RollFlow ground truth")

    # FM-only batch (warmup, sampled diagonal, or disabled MeanFlow).
    if mf_weight == 0 or not bool(active.any()):
        v = _predict(model, x_t, t, t, context, kwargs)
        loss_fm = masked_mse(
            v - v_gt, valid=valid, action_loss_weights=action_loss_weights
        )
        return loss_fm, _stats(
            loss_fm,
            loss_fm.new_zeros(()),
            loss_fm.new_zeros(()),
            active,
            mf_weight,
            False,
        )

    x_s = lerp_path(x0, x1, s)

    # Graph-bearing interval prediction and diagonal instantaneous FM.
    B = x0.shape[0]
    u_st, v_instant = _predict(
        model,
        torch.cat((x_s, x_t)),
        torch.cat((s, t)),
        torch.cat((t, t)),
        repeat_batch(context, 2, B),
        repeat_batch(kwargs, 2, B),
    ).chunk(2)

    # Detached finite-difference derivative.  Only u_st in V_mf carries the
    # training graph; endpoint predictions are used as a stop-gradient
    # correction estimate exactly as in the MeanFlow identity.
    t_minus = torch.where(active, t - delta, t)
    t_plus = torch.where(active, t + delta, t)
    with torch.no_grad():
        u_minus, u_plus = _predict(
            model,
            torch.cat((x_s, x_s)),
            torch.cat((s, s)),
            torch.cat((t_minus, t_plus)),
            repeat_batch(context, 2, B),
            repeat_batch(kwargs, 2, B),
        ).chunk(2)
        du_dt = (u_plus - u_minus) / (2.0 * delta)
        du_dt = du_dt.clamp(min=-mf_kv, max=mf_kv)

    # Keep this expression outside no_grad: the MF loss must update u_st.
    v_mf = u_st + (t - s) * du_dt

    # One direct mean over the selected action elements.
    loss_fm = masked_mse(
        v_instant - v_gt, valid=valid, action_loss_weights=action_loss_weights
    )
    loss_mf = masked_mse(
        v_mf - v_gt,
        valid=active.squeeze(-1),
        action_loss_weights=action_loss_weights,
    )

    # Gate the raw batch MeanFlow scalar before applying its weight.  The
    # decision is detached, so an abnormal interval objective contributes
    # exactly zero MF gradient while the FM anchor remains untouched.
    mf_gate_active = torch.isfinite(loss_mf.detach())
    if mf_loss_threshold is not None:
        mf_gate_active &= loss_mf.detach() <= float(mf_loss_threshold)
    mf_loss_used = torch.nan_to_num(loss_mf, nan=0.0, posinf=0.0, neginf=0.0)
    mf_loss_used = mf_loss_used * mf_gate_active.to(mf_loss_used.dtype)
    loss = loss_fm + mf_weight * mf_loss_used

    return loss, _stats(
        loss_fm,
        loss_mf,
        mf_loss_used,
        active,
        mf_weight,
        bool(mf_gate_active),
    )


def _stats(
    fm: Tensor,
    mf: Tensor,
    mf_used: Tensor,
    active: Tensor,
    weight: float,
    gate_active: bool,
):
    """Build the few metrics needed to monitor the two loss terms."""
    return {
        "fm_loss": fm.detach(),
        "mf_loss": mf.detach(),
        "mf_loss_weighted": (weight * mf_used).detach(),
        "active_mf_frac": active.float().mean().detach(),
        "mf_gate_active": fm.new_tensor(float(gate_active)),
    }


# --- Time sampling ------------------------------------------------------------


class StaircaseTimeSampler:
    def __init__(self, cfg: RollFlowConfig):
        self.cfg = cfg
        self.ks = tuple(k for k in range(1, cfg.num_chunks + 1) if cfg.num_chunks % k == 0)

    def sample_training(
        self,
        batch: int,
        *,
        device,
        dtype=torch.float32,
        generator=None,
        p_fm: Optional[float] = None,
        num_time_groups: Optional[int] = None,
    ) -> TrainingTimes:
        if batch <= 0:
            raise ValueError("batch must be positive")
        p_fm = self.cfg.p_fm if p_fm is None else float(p_fm)
        if not 0 <= p_fm <= 1:
            raise ValueError("p_fm must lie in [0, 1]")
        K = self._sample_k(device, generator) if num_time_groups is None else int(num_time_groups)
        if K not in self.ks:
            raise ValueError(f"num_time_groups must divide {self.cfg.num_chunks}")

        t_group = torch.rand(batch, K, 1, device=device, dtype=dtype, generator=generator)
        t_group = t_group.sort(1, descending=True).values
        ratio = torch.rand(batch, 1, 1, device=device, dtype=dtype, generator=generator)
        fm_mask = torch.rand(batch, 1, 1, device=device, generator=generator) < p_fm
        ratio = torch.where(fm_mask, torch.ones_like(ratio), ratio)

        repeat = (self.cfg.num_chunks // K) * self.cfg.chunk_size
        t = t_group.repeat_interleave(repeat, 1)
        s = (ratio * t_group).repeat_interleave(repeat, 1)
        return TrainingTimes(
            s,
            t,
            _lsd_mask(s, t, self.cfg.finite_difference_delta),
            K,
            self.cfg.num_chunks // K,
            ratio,
            fm_mask,
        )

    def deployment(
        self, batch: int, *, refinement_steps: int, cold: bool, device, dtype=torch.float32
    ) -> StaircaseTimes:
        K = int(refinement_steps)
        if not self.cfg.valid_steps(K):
            raise ValueError(f"refinement_steps must lie in [1,{self.cfg.num_chunks}]")

        lower = torch.zeros(batch, device=device, dtype=dtype)
        upper = torch.ones(batch, device=device, dtype=dtype)
        j = torch.arange(self.cfg.num_chunks, device=device, dtype=dtype)
        gap = (upper - lower)[:, None]
        t = upper[:, None] - gap * (j / K).clamp(max=1)[None]
        s = upper[:, None] - gap * ((j + 1) / K).clamp(max=1)[None]

        def expand(z):
            return z.repeat_interleave(self.cfg.chunk_size, 1).unsqueeze(-1)

        s, t = expand(s), expand(t)
        if cold:
            s = torch.zeros_like(s)
        return StaircaseTimes(s, t, t > s, K, lower, upper)

    def _sample_k(self, device, generator) -> int:
        if len(self.ks) == 1:
            return 1
        p = torch.full((len(self.ks),), (1 - self.cfg.p_k1) / (len(self.ks) - 1), device=device)
        p[0] = self.cfg.p_k1
        return self.ks[torch.multinomial(p, 1, generator=generator).item()]


# --- Training + rolling inference --------------------------------------------


class RollFlow:
    def __init__(self, cfg: RollFlowConfig):
        self.cfg = cfg
        self.time = StaircaseTimeSampler(cfg)
        self._cache: Optional[Tensor] = None
        self._cache_steps: Optional[int] = None
        self._cache_mode: Optional[str] = None

    def loss(self, model, x, step=0, context=None, pad=None, **model_kwargs):
        self._check_actions(x)
        if pad is not None and pad.shape != x.shape[:2]:
            raise ValueError(f"pad must be [B,{self.cfg.horizon}]")

        x1 = x
        x0 = torch.randn_like(x1)
        if self.cfg.use_ot:
            x0 = ot_match(x1, x0, pad)

        p_fm = 1.0 if step < self.cfg.fm_only_steps else self.cfg.p_fm
        times = self.time.sample_training(x.shape[0], device=x.device, p_fm=p_fm)
        weights = (
            None
            if self.cfg.action_loss_weights is None
            else torch.as_tensor(self.cfg.action_loss_weights, device=x.device, dtype=torch.float32)
        )
        loss, stats = central_difference_lsd(
            model,
            x0,
            x1,
            times.s,
            times.t,
            delta=self.cfg.finite_difference_delta,
            active=times.active,
            context=context,
            pad=pad,
            mf_kv=self.cfg.mf_kv,
            mf_weight=self.cfg.mf_weight,
            mf_loss_threshold=self.cfg.mf_loss_threshold,
            action_loss_weights=weights,
            model_kwargs=model_kwargs,
        )
        scalars = {k: v for k, v in stats.items() if v.ndim == 0}
        stats = dict(zip(scalars, torch.stack(list(scalars.values())).cpu().tolist(), strict=True))
        stats.update(
            {
                "loss": float(loss.detach().cpu()),
                "fm_only": float(step < self.cfg.fm_only_steps),
                "fm_only_steps": self.cfg.fm_only_steps,
                "p_fm": p_fm,
                "fm_only_frac": float(times.fm_mask.float().mean().cpu()),
                "source_ratio": float(times.ratio.mean().cpu()),
                "num_time_groups": times.num_time_groups,
                "train_block_size": times.block_size,
            }
        )
        return loss, stats

    @torch.no_grad()
    def step(
        self,
        model,
        batch,
        context=None,
        *,
        device=None,
        dtype=None,
        refinement_steps=None,
        velocity_mode="average",
        **model_kwargs,
    ):
        """Advance the rolling cache using interval or instantaneous velocity.

        ``average`` preserves the original query ``model(z, s, t)``.  In
        ``instant`` mode the model is queried at the current source time,
        ``model(z, s, s)``, while the same interval ``t-s`` is still used for
        the Euler update.
        """
        if velocity_mode not in _VELOCITY_MODES:
            raise ValueError("velocity_mode must be 'average' or 'instant'")
        K = self.cfg.steps if refinement_steps is None else int(refinement_steps)
        if not self.cfg.valid_steps(K):
            raise ValueError(f"refinement_steps must lie in [1,{self.cfg.num_chunks}]")

        device, dtype = infer_device_dtype(model, context, device, dtype)
        if self.cfg.reset_cache_each_step:
            self.reset()
        elif self._cache is not None and self._cache_mode != velocity_mode:
            # Do not mix trajectories generated with different velocity
            # parameterizations when the caller switches modes.
            self.reset()

        z, cold = self._rolling_input(batch, device, dtype, K)
        times = self.time.deployment(batch, refinement_steps=K, cold=cold, device=device)

        if cold and self.cfg.iterative_cold_start:
            cache = self._iterative_cold_start(
                model, z, times.t, K, context, model_kwargs, dtype, velocity_mode
            )
        else:
            query_t = _velocity_query_time(times.s, times.t, velocity_mode)
            v = self._velocity(model, z, times.s, query_t, context, model_kwargs)
            cache = flow_map(z, times.s, times.t, v).to(dtype)

        self._cache, self._cache_steps, self._cache_mode = cache.detach(), K, velocity_mode
        return self._cache[:, : self.cfg.chunk_size].clone()

    def reset(self):
        self._cache = self._cache_steps = None
        self._cache_mode = None

    @property
    def cache_info(self):
        if self._cache is None:
            return None
        return RollFlowCacheInfo(
            tuple(self._cache.shape),
            self._cache_steps,
            self._cache.device,
            self._cache.dtype,
        )

    def _rolling_input(self, batch, device, dtype, K):
        shape = (batch, self.cfg.horizon, self.cfg.action_dim)
        ok = (
            self._cache is not None
            and self._cache_steps == K
            and self._cache.shape == shape
            and self._cache.device == device
            and self._cache.dtype == dtype
        )
        if not ok:
            return torch.randn(shape, device=device, dtype=dtype), True
        tail = torch.randn(
            batch, self.cfg.chunk_size, self.cfg.action_dim, device=device, dtype=dtype
        )
        return torch.cat((self._cache[:, self.cfg.chunk_size :], tail), 1), False

    def _velocity(self, model, z, s, t, context, kwargs):
        v = _predict(model, z, s, t, context, kwargs)
        c = self.cfg.clip_velocity
        return v if c <= 0 else c * torch.tanh(v / c)

    def _iterative_cold_start(
        self, model, z, final_t, K, context, kwargs, dtype, velocity_mode
    ):
        s = torch.zeros_like(final_t)
        for i in range(1, K + 1):
            t = torch.minimum(final_t, torch.full_like(final_t, i / K))
            query_t = _velocity_query_time(s, t, velocity_mode)
            z = flow_map(
                z, s, t, self._velocity(model, z, s, query_t, context, kwargs)
            ).to(dtype)
            s = t
        return z

    def _check_actions(self, x):
        if x.ndim != 3 or tuple(x.shape[1:]) != (self.cfg.horizon, self.cfg.action_dim):
            raise ValueError(f"x must be [B,{self.cfg.horizon},{self.cfg.action_dim}]")


# --- Utilities ----------------------------------------------------------------


def _require_finite(value: Tensor, name: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"Nonfinite {name}")


def _validate_inputs(
    x0,
    x1,
    s,
    t,
    active,
    pad,
    delta,
    mf_kv,
    mf_weight,
    mf_loss_threshold,
):
    if not 0 < delta < 0.5:
        raise ValueError("delta must lie in (0, 0.5)")
    if any(not math.isfinite(v) or v < 0 for v in (mf_kv, mf_weight)):
        raise ValueError("mf_kv and mf_weight must be finite and non-negative")
    _validate_mf_loss_threshold(mf_loss_threshold)
    if x0.ndim != 3 or x0.shape != x1.shape:
        raise ValueError("x0 and x1 must have identical [B,H,A] shapes")
    shape = (*x0.shape[:2], 1)
    if s.shape != shape or t.shape != shape:
        raise ValueError(f"s and t must have shape {shape}")
    if active is not None and active.shape != shape:
        raise ValueError(f"active must have shape {shape}")
    if pad is not None and pad.shape != x0.shape[:2]:
        raise ValueError("pad must have shape [B,H]")


def _validate_mf_loss_threshold(threshold: Optional[float]) -> None:
    if threshold is not None and (not math.isfinite(threshold) or threshold < 0):
        raise ValueError("mf_loss_threshold must be finite and non-negative or None")


def _validate_times(s: Tensor, t: Tensor, delta: float, active: Tensor) -> None:
    _require_finite(s, "source times")
    _require_finite(t, "target times")
    if s.shape != t.shape or bool(((s < -1e-6) | (t > 1 + 1e-6) | (s > t + 1e-6)).any()):
        raise ValueError("require matching s and t with 0 <= s <= t <= 1")
    if bool((active & ~_lsd_mask(s, t, delta)).any()):
        raise ValueError("active interval tokens require s < t-delta and t+delta <= 1")


def masked_mse(error: Tensor, *, pad=None, valid=None, action_loss_weights=None) -> Tensor:
    mask = torch.ones_like(error[..., 0], dtype=torch.bool)
    if pad is not None:
        mask = mask & ~pad.bool()
    if valid is not None:
        mask = mask & (valid.squeeze(-1) if valid.ndim == 3 else valid).bool()
    sq = error.float().square()
    if not bool(mask.any()):
        return sq.sum() * 0.0
    if action_loss_weights is None:
        return sq[mask].mean()
    weights = torch.as_tensor(
        action_loss_weights, device=error.device, dtype=sq.dtype
    )
    if weights.shape != (error.shape[-1],):
        raise ValueError("action_loss_weights must have shape [action_dim]")
    denominator = (mask.sum() * weights.sum()).clamp_min(1e-12)
    return (sq[mask] * weights).sum() / denominator


def _lsd_mask(s: Tensor, t: Tensor, delta: float) -> Tensor:
    return ((t - s) > delta + 1e-6) & (t + delta <= 1 - 1e-6)


def _predict(model, z, s, t, context, kwargs):
    out = model(z, s, t, context, **kwargs)
    if out.shape != z.shape:
        raise ValueError(f"model output {tuple(out.shape)} must match z {tuple(z.shape)}")
    return out


def _velocity_query_time(source_time: Tensor, target_time: Tensor, mode: str) -> Tensor:
    """Select the time argument for the inference velocity query."""
    return target_time if mode == "average" else source_time


def repeat_batch(value: Any, repeats: int, batch: int) -> Any:
    if torch.is_tensor(value):
        return torch.cat([value] * repeats) if value.ndim and value.shape[0] == batch else value
    if isinstance(value, dict):
        return {k: repeat_batch(v, repeats, batch) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(repeat_batch(v, repeats, batch) for v in value)
    if isinstance(value, list):
        return [repeat_batch(v, repeats, batch) for v in value]
    return value


@torch.no_grad()
def ot_match(x1: Tensor, x0: Tensor, pad: Optional[Tensor] = None) -> Tensor:
    if x1.shape[0] <= 1:
        return x0
    from scipy.optimize import linear_sum_assignment

    if pad is None or not bool(pad.any()):
        cost = torch.cdist(x1.float().flatten(1), x0.float().flatten(1))
    else:
        valid = (~pad.bool()).float().unsqueeze(-1)
        sq = (x1.float()[:, None] - x0.float()[None]).square()
        dims = valid.sum((1, 2)).clamp_min(1) * x1.shape[-1]
        cost = (sq * valid[:, None]).sum((-2, -1)) / dims[:, None]

    row, col = linear_sum_assignment(cost.cpu().numpy())
    row, col = torch.as_tensor(row, device=x0.device), torch.as_tensor(col, device=x0.device)
    out = torch.empty_like(x0)
    out[row] = x0[col]
    return out


def infer_device_dtype(model, context=None, device=None, dtype=None):
    if context is not None:
        return (context.device if device is None else torch.device(device), context.dtype if dtype is None else dtype)
    try:
        p = next(model.parameters())
        return (p.device if device is None else torch.device(device), p.dtype if dtype is None else dtype)
    except StopIteration:
        return (
            torch.device("cpu") if device is None else torch.device(device),
            torch.float32 if dtype is None else dtype,
        )
