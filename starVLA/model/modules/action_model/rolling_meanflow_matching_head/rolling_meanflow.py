"""RollFlow: Flow Matching + Central-Difference Lagrangian Self-Distillation.

Core training algorithm
-----------------------
    # 1. Sample a batch of source/target pairs (x0, x1) from the action trajectory.
    s, t = sample_training_times()
    x_s = (1-s) * x0 + s * x1
    x_t = (1-t) * x0 + t * x1
    v_gt = x1 - x0

    # 2. Build the stop-gradient teacher path for the full batch.
    with no_grad:
        u_st = model(x_s, s, t)
        X_t_hat = x_s + (t-s) * u_st
        v_teacher = model(X_t_hat, t, t)

    # 3. Pack tangent endpoints and local FM for the full batch.
    u_minus, u_plus, v_instant = model(
        cat(x_s, x_s, x_t),
        cat(s, s, t),
        cat(t-delta, t+delta, t),
    ).chunk(3)
    X_minus = x_s + (t-delta-s) * u_minus
    X_plus  = x_s + (t+delta-s) * u_plus
    v_tangent = (X_plus - X_minus) / (2.0*delta)

    # 4. Compute per-sample losses and apply the optional LSD gate.
    loss_fm         = MSE(v_instant, v_gt)
    loss_fm_active  = MSE(v_instant, v_gt, active rows)
    loss_lsd_metric = MSE(v_tangent, stopgrad(v_teacher), active rows)
    loss_lsd_raw    = 2*delta * loss_lsd_metric  # optional legacy-safe scaling
    lsd_budget[b] = w_lsd * stopgrad(loss_fm_active[b])
    mask[b] = isfinite(loss_lsd_raw[b]) ∧ (loss_lsd_raw[b] ≤ lsd_budget[b])
    # Accepted sample losses are active-token weighted before the batch mean:
    #   loss_lsd = sum_b n_b * mask[b] * loss_lsd_raw[b] / sum_b n_b.
    # ``w_lsd`` controls the gate budget. FM always has coefficient 1.
    loss_lsd = weighted_mean_by_active_tokens(mask, loss_lsd_raw)
    loss = loss_fm + loss_lsd

With ``use_lsd_gate=False``, the finite LSD term is added directly and the
detached FM budget is not used; ``loss_fm_active`` remains a diagnostic only.

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
sample gives s=ratio*t.  The first ``fm_only_steps`` (default 10,000) are
diagonal instantaneous FM; afterwards the sampler switches directly to the
fixed ``p_fm`` mixture.  Deployment still uses the exact rolling staircase
needed for cache alignment.
When a batch is entirely diagonal (including FM warmup), the teacher and
tangent forwards are skipped; one local FM forward is sufficient and LSD is a
zero; FM retains the training graph.

The LSD estimator is the packed central difference above.  The optional gate is
sample-wise: each sample first aggregates its own active tokens, then compares
its finite LSD value against its own detached FM budget.  Samples with no
active tokens, non-finite LSD, or an over-budget LSD contribute no LSD
gradient; accepted samples are combined using their active-token counts so
that the all-accepted case has exactly the same scale as the global MSE.
There is no CSF,
split loss, persistent training cache, or adjacent-window training sampler.
Rolling state exists only at inference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import torch
from torch import Tensor, nn


@dataclass
class RollFlowConfig:
    horizon: int
    action_dim: int
    chunk_size: int
    finite_difference_delta: float = 0.01
    inference_steps: Optional[int] = None
    p_k1: float = 0.7
    p_fm: float = 0.5
    fm_only_steps: int = 10_000
    w_lsd: float = 0.1
    use_ot: bool = True
    use_lsd_scaling: bool = True
    use_lsd_gate: bool = True
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
        if any(not math.isfinite(v) or v < 0 for v in (self.w_lsd, self.clip_velocity, self.fm_only_steps)):
            raise ValueError("w_lsd/clip_velocity/fm_only_steps must be finite and non-negative")
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
    w_lsd: float = 0.1,
    use_lsd_scaling: bool = True,
    use_lsd_gate: bool = True,
    action_loss_weights: Optional[Tensor] = None,
    model_kwargs: Optional[dict[str, Any]] = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """FM everywhere; LSD only on valid off-diagonal intervals."""
    kwargs = model_kwargs or {}
    _validate_inputs(x0, x1, s, t, active, pad, delta, w_lsd)
    valid = torch.ones_like(s[..., 0], dtype=torch.bool) if pad is None else ~pad.bool()
    active = _lsd_mask(s, t, delta) if active is None else active.bool()
    _validate_times(s, t, delta, active)
    active = active & valid.unsqueeze(-1)

    x_t = lerp_path(x0, x1, t)
    v_gt = x1.float() - x0.float()
    _require_finite(v_gt, "RollFlow ground truth")

    # FM-only batch (warmup or sampled diagonal).
    if not bool(active.any()):
        v = _predict(model, x_t, t, t, context, kwargs)
        loss_fm = _masked_token_mean(_token_mse(v - v_gt, action_loss_weights), valid)
        zero_tokens = torch.zeros_like(valid, dtype=loss_fm.dtype)
        loss_lsd, stats = _gated_lsd(
            zero_tokens,
            zero_tokens,
            active.squeeze(-1),
            scale=2 * delta if use_lsd_scaling else 1.0,
            budget_ratio=w_lsd,
            use_gate=use_lsd_gate,
        )
        return loss_fm, _stats(loss_fm, loss_lsd, active, stats, use_lsd_scaling)

    x_s = lerp_path(x0, x1, s)

    # Stop-gradient teacher at the predicted terminal state.
    with torch.no_grad():
        u = _predict(model, x_s, s, t, context, kwargs)
        x_hat_t = flow_map(x_s, s, t, u)
        v_teacher = _predict(model, x_hat_t, t, t, context, kwargs)
        _require_finite(v_teacher, "RollFlow teacher prediction")
    del u, x_hat_t

    # Left / right finite-difference endpoints + local FM in one packed forward.
    t_minus = torch.where(active, t - delta, t)
    t_plus = torch.where(active, t + delta, t)
    B = x0.shape[0]
    u_minus, u_plus, v_local = _predict(
        model,
        torch.cat((x_s, x_s, x_t)),
        torch.cat((s, s, t)),
        torch.cat((t_minus, t_plus, t)),
        repeat_batch(context, 3, B),
        repeat_batch(kwargs, 3, B),
    ).chunk(3)

    x_minus = flow_map(x_s, s, t_minus, u_minus)
    x_plus = flow_map(x_s, s, t_plus, u_plus)
    v_tangent = (x_plus - x_minus) / (2 * delta)

    fm_tokens = _token_mse(v_local - v_gt, action_loss_weights)
    lsd_tokens = _token_mse(v_tangent - v_teacher.detach(), action_loss_weights)
    loss_fm = _masked_token_mean(fm_tokens, valid)
    loss_lsd, stats = _gated_lsd(
        fm_tokens,
        lsd_tokens,
        active.squeeze(-1),
        scale=2 * delta if use_lsd_scaling else 1.0,
        budget_ratio=w_lsd,
        use_gate=use_lsd_gate,
    )
    return loss_fm + loss_lsd, _stats(loss_fm, loss_lsd, active, stats, use_lsd_scaling)


def _gated_lsd(fm_tokens, lsd_tokens, active, *, scale, budget_ratio, use_gate):
    counts = active.sum(1).to(fm_tokens.dtype)
    total = counts.sum().clamp_min(1)

    fm = torch.where(active, fm_tokens, 0).sum(1) / counts.clamp_min(1)
    loss_lsd_metric = torch.where(active, lsd_tokens, 0).sum(1) / counts.clamp_min(1)
    loss_lsd_raw = scale * loss_lsd_metric
    budget = budget_ratio * fm.detach()

    has_active = counts > 0
    finite = torch.isfinite(loss_lsd_raw) & has_active
    keep = finite & (loss_lsd_raw <= budget) if use_gate else finite
    accepted = torch.where(keep, torch.nan_to_num(loss_lsd_raw, nan=0.0, posinf=0.0, neginf=0.0), 0)
    loss_lsd = (accepted * counts).sum() / total
    with torch.no_grad():
        sample_count = has_active.sum().clamp_min(1)
        stats = {
            "fm_loss_active": (fm * counts).sum() / total,
            "lsd_loss_metric": (loss_lsd_metric * counts).sum() / total,
            "lsd_loss_raw": (loss_lsd_raw * counts).sum() / total,
            "lsd_budget": (budget * counts).sum() / total,
            "lsd_gate_enabled": fm.new_tensor(float(use_gate)),
            "lsd_gate_active": (has_active & ~keep).sum() / sample_count,
            "lsd_finite_frac": finite.sum() / sample_count,
            "lsd_keep_frac": keep.sum() / sample_count,
            "lsd_keep_token_frac": (keep * counts).sum() / total,
            "lsd_active_sample_frac": has_active.float().mean(),
            "fm_loss_active_per_sample": fm,
            "lsd_loss_raw_per_sample": loss_lsd_raw,
            "lsd_loss_per_sample": accepted,
            "lsd_budget_per_sample": budget,
            "lsd_active_per_sample": has_active,
            "lsd_finite_per_sample": finite,
            "lsd_keep_per_sample": keep,
        }
    return loss_lsd, {k: v.detach() for k, v in stats.items()}


def _stats(fm: Tensor, lsd: Tensor, active: Tensor, gate_stats: dict, scaling: bool):
    return {
        **gate_stats,
        "fm_loss": fm.detach(),
        "lsd_loss": lsd.detach(),
        "active_lsd_frac": active.float().mean().detach(),
        "lsd_scaling_enabled": fm.new_tensor(float(scaling)),
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
            s, t, _lsd_mask(s, t, self.cfg.finite_difference_delta), K, self.cfg.num_chunks // K, ratio, fm_mask
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
            w_lsd=self.cfg.w_lsd,
            use_lsd_scaling=self.cfg.use_lsd_scaling,
            use_lsd_gate=self.cfg.use_lsd_gate,
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
    def step(self, model, batch, context=None, *, device=None, dtype=None, refinement_steps=None, **model_kwargs):
        K = self.cfg.steps if refinement_steps is None else int(refinement_steps)
        if not self.cfg.valid_steps(K):
            raise ValueError(f"refinement_steps must lie in [1,{self.cfg.num_chunks}]")

        device, dtype = infer_device_dtype(model, context, device, dtype)
        if self.cfg.reset_cache_each_step:
            self.reset()

        z, cold = self._rolling_input(batch, device, dtype, K)
        times = self.time.deployment(batch, refinement_steps=K, cold=cold, device=device)

        if cold and self.cfg.iterative_cold_start:
            cache = self._iterative_cold_start(model, z, times.t, K, context, model_kwargs, dtype)
        else:
            v = self._velocity(model, z, times.s, times.t, context, model_kwargs)
            cache = flow_map(z, times.s, times.t, v).to(dtype)

        self._cache, self._cache_steps = cache.detach(), K
        return self._cache[:, : self.cfg.chunk_size].clone()

    def reset(self):
        self._cache = self._cache_steps = None

    @property
    def cache_info(self):
        if self._cache is None:
            return None
        return RollFlowCacheInfo(tuple(self._cache.shape), self._cache_steps, self._cache.device, self._cache.dtype)

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
        tail = torch.randn(batch, self.cfg.chunk_size, self.cfg.action_dim, device=device, dtype=dtype)
        return torch.cat((self._cache[:, self.cfg.chunk_size :], tail), 1), False

    def _velocity(self, model, z, s, t, context, kwargs):
        v = _predict(model, z, s, t, context, kwargs)
        c = self.cfg.clip_velocity
        return v if c <= 0 else c * torch.tanh(v / c)

    def _iterative_cold_start(self, model, z, final_t, K, context, kwargs, dtype):
        s = torch.zeros_like(final_t)
        for i in range(1, K + 1):
            t = torch.minimum(final_t, torch.full_like(final_t, i / K))
            z = flow_map(z, s, t, self._velocity(model, z, s, t, context, kwargs)).to(dtype)
            s = t
        return z

    def _check_actions(self, x):
        if x.ndim != 3 or tuple(x.shape[1:]) != (self.cfg.horizon, self.cfg.action_dim):
            raise ValueError(f"x must be [B,{self.cfg.horizon},{self.cfg.action_dim}]")


# --- Utilities ----------------------------------------------------------------


def _require_finite(value: Tensor, name: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"Nonfinite {name}")


def _validate_inputs(x0, x1, s, t, active, pad, delta, w_lsd):
    if not 0 < delta < 0.5:
        raise ValueError("delta must lie in (0, 0.5)")
    if not math.isfinite(w_lsd) or w_lsd < 0:
        raise ValueError("w_lsd must be finite and non-negative")
    if x0.ndim != 3 or x0.shape != x1.shape:
        raise ValueError("x0 and x1 must have identical [B,H,A] shapes")
    shape = (*x0.shape[:2], 1)
    if s.shape != shape or t.shape != shape:
        raise ValueError(f"s and t must have shape {shape}")
    if active is not None and active.shape != shape:
        raise ValueError(f"active must have shape {shape}")
    if pad is not None and pad.shape != x0.shape[:2]:
        raise ValueError("pad must have shape [B,H]")


def _validate_times(s: Tensor, t: Tensor, delta: float, active: Tensor) -> None:
    _require_finite(s, "source times")
    _require_finite(t, "target times")
    if s.shape != t.shape or bool(((s < -1e-6) | (t > 1 + 1e-6) | (s > t + 1e-6)).any()):
        raise ValueError("require matching s and t with 0 <= s <= t <= 1")
    if bool((active & ~_lsd_mask(s, t, delta)).any()):
        raise ValueError("active LSD tokens require s < t-delta and t+delta <= 1")


def masked_mse(error: Tensor, *, pad=None, valid=None, action_loss_weights=None) -> Tensor:
    mask = torch.ones_like(error[..., 0], dtype=torch.bool)
    if pad is not None:
        mask = mask & ~pad.bool()
    if valid is not None:
        mask = mask & (valid.squeeze(-1) if valid.ndim == 3 else valid).bool()
    return _masked_token_mean(_token_mse(error, action_loss_weights), mask)


def _lsd_mask(s: Tensor, t: Tensor, delta: float) -> Tensor:
    return ((t - s) > delta + 1e-6) & (t + delta <= 1 - 1e-6)


def _token_mse(error: Tensor, weights: Optional[Tensor]) -> Tensor:
    sq = error.float().square()
    if weights is None:
        return sq.mean(-1)
    w = weights.to(error.device, sq.dtype)
    if w.shape != (error.shape[-1],):
        raise ValueError("action_loss_weights must have shape [action_dim]")
    return (sq * w).sum(-1) / w.sum().clamp_min(1e-12)


def _masked_token_mean(values: Tensor, mask: Tensor) -> Tensor:
    return torch.where(mask, values, 0).sum() / mask.sum().clamp_min(1)


def _predict(model, z, s, t, context, kwargs):
    out = model(z, s, t, context, **kwargs)
    if out.shape != z.shape:
        raise ValueError(f"model output {tuple(out.shape)} must match z {tuple(z.shape)}")
    return out


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
