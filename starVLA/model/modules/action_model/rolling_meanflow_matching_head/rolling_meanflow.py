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
    # Accepted sample losses are active-token weighted before the batch mean.
    loss = loss_fm + weighted_mean_b(where(mask[b], loss_lsd_raw[b], 0))

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
sample gives s=ratio*t.  An FM-only curriculum starts with s=t and anneals to
the configured mixture probability.  Deployment still uses the exact rolling
staircase needed for cache alignment.

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

# =============================================================================
# 1. Configuration
# =============================================================================


@dataclass
class RollFlowConfig:
    horizon: int
    action_dim: int
    chunk_size: int

    finite_difference_delta: float = 0.01
    inference_steps: Optional[int] = None

    p_k1: float = 0.7
    p_fm: float = 0.3
    fm_curriculum_steps: int = 5000

    w_fm: float = 1.0
    w_lsd: float = 0.1
    use_ot: bool = True
    # Remove the explicit central-difference gradient amplification.  Keep
    # this configurable so the unscaled historical objective can be measured
    # as a controlled ablation.
    use_lsd_scaling: bool = True
    # Separately control the finite, detached FM-budget suppression for LSD.
    use_lsd_gate: bool = True
    action_loss_weights: Optional[Sequence[float]] = None

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
        if self.action_loss_weights is not None:
            weights = tuple(float(weight) for weight in self.action_loss_weights)
            if len(weights) != self.action_dim:
                raise ValueError(
                    "action_loss_weights must have one entry per action dimension"
                )
            if any(not math.isfinite(weight) or weight < 0 for weight in weights):
                raise ValueError("action_loss_weights must be finite and non-negative")
            mean = sum(weights) / len(weights)
            if mean <= 0:
                raise ValueError("action_loss_weights must contain a positive value")
            # Keep the overall loss scale unchanged when reweighting groups.
            self.action_loss_weights = tuple(weight / mean for weight in weights)
        if not 0.0 <= self.p_k1 <= 1.0:
            raise ValueError("p_k1 must lie in [0, 1]")
        if not 0.0 <= self.p_fm <= 1.0:
            raise ValueError("p_fm must lie in [0, 1]")
        if self.fm_curriculum_steps < 0:
            raise ValueError("fm_curriculum_steps must be non-negative")

        if not self.valid_steps(self.resolved_inference_steps):
            raise ValueError(
                f"inference_steps must lie in [1,{self.num_action_chunks}]"
            )

    @property
    def num_action_chunks(self) -> int:
        return self.horizon // self.chunk_size

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
    w_lsd: float = 0.1,
    use_lsd_scaling: bool = True,
    use_lsd_gate: bool = True,
    action_loss_weights: Optional[Tensor] = None,
    model_kwargs: Optional[dict[str, Any]] = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Memory-efficient forwards implementing the paper-style objective.

    ``active`` marks tokens that carry a finite off-diagonal interval.  Diagonal
    staircase tokens still receive FM supervision but are excluded from LSD.
    Teacher-only branches run without autograd; this changes memory use, not the
    predicted values or objective.  The LSD
    metric is scaled by ``2*delta`` (when ``use_lsd_scaling`` is true) to remove
    the explicit central-difference gradient amplification.  For each sample,
    the active rows are reduced to one LSD value and one detached FM budget;
    ``use_lsd_gate`` then makes an independent finite/budget decision for that
    sample.  Accepted samples are combined with active-token weighting, so the
    all-accepted case has the same scale as the original global MSE.  Samples
    with no active rows are excluded from LSD.  With the gate disabled, finite
    sample losses are added directly and non-finite samples are safely dropped.
    """
    kwargs = {} if model_kwargs is None else model_kwargs
    delta = float(delta)
    if w_fm < 0 or w_lsd < 0:
        raise ValueError("w_fm and w_lsd must be non-negative")
    active = _lsd_mask(s, t, delta) if active is None else active.bool()
    _validate_training_inputs(x0, x1, s, t, active, pad)
    _validate_times(s, t, delta, active)

    # Padding and interval validity only mask the loss; all rows run all branches.
    loss_active = active
    if pad is not None:
        loss_active = active & ~pad.bool().unsqueeze(-1)

    x_s = lerp_path(x0, x1, s)
    x_t = lerp_path(x0, x1, t)
    v_gt = x1.float() - x0.float()

    # -------------------------------------------------------------------------
    # 1. Inference-only self-teacher path
    # -------------------------------------------------------------------------
    # Compute this path before constructing the student graph so its temporary
    # activations are released immediately.
    with torch.no_grad():
        V_st = _predict(model, x_s, s, t, context, kwargs)
        X_t_hat = flow_map(x_s, s, t, V_st)
        v_teacher = _predict(model, X_t_hat, t, t, context, kwargs)
        if not bool(torch.isfinite(v_teacher).all()):
            raise FloatingPointError("Nonfinite RollFlow teacher prediction")
        if not bool(torch.isfinite(v_gt).all()):
            raise FloatingPointError("Nonfinite RollFlow ground truth")

    # -------------------------------------------------------------------------
    # 2. Gradient-bearing tangent + local FM prediction
    # -------------------------------------------------------------------------
    # Leave inactive target times unperturbed to avoid t-delta < s or
    # t+delta > 1.  One packed 3B student forward covers all branches.
    t_minus = torch.where(loss_active, t - delta, t)
    t_plus = torch.where(loss_active, t + delta, t)
    V_minus, V_plus, v_local = _parallel_student_predictions(
        model, x_s, x_t, s, t, t_minus, t_plus, context, kwargs
    )
    X_minus = flow_map(x_s, s, t_minus, V_minus)
    X_plus = flow_map(x_s, s, t_plus, V_plus)
    v_tangent = (X_plus - X_minus) / (2.0 * delta)

    fm_error = v_local - v_gt
    _, fm_sums, fm_counts = _masked_mse_components(
        fm_error,
        pad=pad,
        action_loss_weights=action_loss_weights,
    )
    loss_fm = fm_sums.sum() / fm_counts.sum().clamp_min(1.0)
    fm_active_per_sample, fm_active_sums, active_counts = _masked_mse_components(
        fm_error,
        pad=pad,
        valid=loss_active,
        action_loss_weights=action_loss_weights,
    )
    loss_fm_active = fm_active_sums.sum() / active_counts.sum().clamp_min(1.0)
    lsd_metric_per_sample, lsd_sums, _ = _masked_mse_components(
        v_tangent - v_teacher.detach(),
        pad=pad,
        valid=loss_active,
        action_loss_weights=action_loss_weights,
    )
    loss_lsd_metric = lsd_sums.sum() / active_counts.sum().clamp_min(1.0)
    lsd_scale = 2.0 * delta if use_lsd_scaling else 1.0
    loss_lsd_raw_per_sample = lsd_scale * lsd_metric_per_sample
    loss_lsd_raw = lsd_scale * loss_lsd_metric
    # ``w_lsd`` is the legacy config name for this budget ratio.  Keep the
    # semantic name local so the comparison is explicit and self-documenting.
    lsd_budget_ratio = float(w_lsd)
    lsd_budget_per_sample = lsd_budget_ratio * fm_active_per_sample.detach()
    has_active_sample = active_counts > 0
    lsd_finite_per_sample = (
        torch.isfinite(loss_lsd_raw_per_sample) & has_active_sample
    )
    lsd_keep_per_sample = (
        lsd_finite_per_sample
        & (loss_lsd_raw_per_sample <= lsd_budget_per_sample)
        if use_lsd_gate
        else lsd_finite_per_sample
    )
    # ``where`` is essential here: NaN/Inf multiplied by a zero mask remains
    # NaN/Inf.  The false branch is a graph-connected zero and contributes no
    # loss or gradient for non-finite or over-budget LSD values.  Weighting by
    # active token count preserves the global masked-MSE scale when every
    # sample is accepted, while allowing each sample to be rejected alone.
    safe_lsd_raw_per_sample = torch.nan_to_num(
        loss_lsd_raw_per_sample,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    accepted_sums = torch.where(
        lsd_keep_per_sample,
        safe_lsd_raw_per_sample * active_counts,
        torch.zeros_like(safe_lsd_raw_per_sample),
    )
    loss_lsd = accepted_sums.sum() / active_counts.sum().clamp_min(1.0)
    active_sample_count = has_active_sample.to(loss_lsd_raw.dtype).sum()
    active_token_count = active_counts.sum()
    lsd_finite_frac = (
        lsd_finite_per_sample.to(loss_lsd_raw.dtype).sum()
        / active_sample_count.clamp_min(1.0)
    )
    lsd_keep_frac = (
        lsd_keep_per_sample.to(loss_lsd_raw.dtype).sum()
        / active_sample_count.clamp_min(1.0)
    )
    lsd_gate_active = (
        (has_active_sample & ~lsd_keep_per_sample)
        .to(loss_lsd_raw.dtype)
        .sum()
        / active_sample_count.clamp_min(1.0)
    )
    lsd_keep_token_frac = (
        torch.where(lsd_keep_per_sample, active_counts, torch.zeros_like(active_counts)).sum()
        / active_token_count.clamp_min(1.0)
    )
    lsd_budget = lsd_budget_ratio * loss_fm_active.detach()
    loss = w_fm * loss_fm + loss_lsd

    return loss, {
        "fm_loss": loss_fm.detach(),
        "fm_loss_active": loss_fm_active.detach(),
        "lsd_loss": loss_lsd.detach(),
        "lsd_loss_raw": loss_lsd_raw.detach(),
        "lsd_loss_metric": loss_lsd_metric.detach(),
        "lsd_scaling_enabled": loss_lsd_raw.new_tensor(float(use_lsd_scaling)).detach(),
        "lsd_gate_active": lsd_gate_active.detach(),
        "lsd_gate_enabled": loss_lsd_raw.new_tensor(float(use_lsd_gate)).detach(),
        "lsd_finite_frac": lsd_finite_frac.detach(),
        "lsd_keep_frac": lsd_keep_frac.detach(),
        "lsd_budget": lsd_budget.detach(),
        "active_lsd_frac": loss_active.float().mean().detach(),
        "lsd_active_sample_frac": has_active_sample.float().mean().detach(),
        "lsd_keep_token_frac": lsd_keep_token_frac.detach(),
        "fm_loss_active_per_sample": fm_active_per_sample.detach(),
        "lsd_loss_raw_per_sample": loss_lsd_raw_per_sample.detach(),
        "lsd_loss_per_sample": torch.where(
            lsd_keep_per_sample,
            safe_lsd_raw_per_sample,
            torch.zeros_like(safe_lsd_raw_per_sample),
        ).detach(),
        "lsd_budget_per_sample": lsd_budget_per_sample.detach(),
        "lsd_active_per_sample": has_active_sample.detach(),
        "lsd_finite_per_sample": lsd_finite_per_sample.detach(),
        "lsd_keep_per_sample": lsd_keep_per_sample.detach(),
        "v_local_abs": v_local.detach().abs().mean(),
        "v_tangent_abs": masked_mean(v_tangent.detach().abs(), loss_active),
        "v_teacher_abs": masked_mean(v_teacher.abs(), loss_active),
    }


# =============================================================================
# 3. Staircase time sampling
# =============================================================================


class StaircaseTimeSampler:
    """Random grouped training times + exact deployment staircase.

    Training samples K from the divisors of M=H/C.  K=1 has probability
    ``p_k1`` and every other divisor shares the remaining mass.  The K sorted
    target times are expanded over equal contiguous action blocks.  Each sample
    uses one ratio for all groups, s=ratio*t; FM-only samples use ratio=1.

    Deployment retains the K-level layout required for cache alignment:

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
        self._train_ks = torch.tensor(
            [k for k in range(1, cfg.num_action_chunks + 1) if cfg.num_action_chunks % k == 0],
            dtype=torch.long,
        )

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
        device = torch.device(device)
        p_fm = self.cfg.p_fm if p_fm is None else float(p_fm)
        if not 0.0 <= p_fm <= 1.0:
            raise ValueError("p_fm must lie in [0, 1]")

        if num_time_groups is None:
            K = self._sample_k(device, generator)
        else:
            K = int(num_time_groups)
            if K not in self._train_ks.tolist():
                raise ValueError(f"num_time_groups must divide {self.cfg.num_action_chunks}")

        t_group = torch.sort(
            torch.rand(batch, K, 1, device=device, dtype=dtype, generator=generator),
            dim=1,
            descending=True,
        ).values
        ratio = torch.rand(batch, 1, 1, device=device, dtype=dtype, generator=generator)
        fm_mask = torch.rand(batch, 1, 1, device=device, generator=generator) < p_fm
        ratio = torch.where(fm_mask, torch.ones_like(ratio), ratio)
        s_group = ratio * t_group

        repeats = (self.cfg.num_action_chunks // K) * self.cfg.chunk_size
        s = s_group.repeat_interleave(repeats, dim=1)
        t = t_group.repeat_interleave(repeats, dim=1)
        active = _lsd_mask(s, t, self.cfg.finite_difference_delta)
        return TrainingTimes(
            s,
            t,
            active,
            K,
            self.cfg.num_action_chunks // K,
            ratio,
            fm_mask,
        )

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

    def _sample_k(self, device, generator) -> int:
        ks = self._train_ks.to(device=device)
        if len(ks) == 1:
            return 1
        probs = torch.full(
            (len(ks),),
            (1.0 - self.cfg.p_k1) / (len(ks) - 1),
            device=device,
        )
        probs[0] = self.cfg.p_k1
        index = torch.multinomial(probs, 1, generator=generator)
        return int(ks[index].item())

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

# =============================================================================
# 4. Engineering helpers
# =============================================================================


def _lsd_mask(s: Tensor, t: Tensor, delta: float) -> Tensor:
    delta = float(delta)
    # Reuse the same gap expression as ``_validate_times``.  Computing the
    # equivalent ``s >= t - delta`` predicate separately can disagree with
    # this comparison for a float32 sample lying right on the threshold.
    gap = t - s
    tol = 1e-6
    return (gap > delta + tol) & (t + delta <= 1.0 - tol)


def _validate_times(s: Tensor, t: Tensor, delta: float, active: Tensor) -> None:
    tol = 1e-6
    if s.shape != t.shape:
        raise ValueError("s and t must have identical shapes")
    if bool((s < -tol).any()) or bool((t > 1.0 + tol).any()) or bool((s > t + tol).any()):
        raise ValueError("require 0 <= s <= t <= 1")
    gap = t - s
    if bool((active & (gap <= float(delta) + tol)).any()):
        raise ValueError("active LSD tokens require s < t-delta")
    if bool((active & (t + float(delta) > 1.0 - tol)).any()):
        raise ValueError("active LSD tokens require t+delta <= 1")


def _validate_training_inputs(
    x0: Tensor,
    x1: Tensor,
    s: Tensor,
    t: Tensor,
    active: Tensor,
    pad: Optional[Tensor],
) -> None:
    if x0.shape != x1.shape or x0.ndim != 3:
        raise ValueError("x0 and x1 must have identical [B,H,A] shapes")
    expected_times = (*x0.shape[:2], 1)
    if s.shape != expected_times or t.shape != expected_times:
        raise ValueError(f"s and t must have shape {expected_times}")
    if active.shape != expected_times:
        raise ValueError(f"active must have shape {expected_times}")
    if pad is not None and pad.shape != x0.shape[:2]:
        raise ValueError(f"pad must have shape {tuple(x0.shape[:2])}")


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
    if torch.is_tensor(value):
        return torch.cat([value] * repeats, dim=0) if value.ndim and value.shape[0] == batch else value
    if isinstance(value, dict):
        return {key: repeat_batch(item, repeats, batch) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(repeat_batch(item, repeats, batch) for item in value)
    if isinstance(value, list):
        return [repeat_batch(item, repeats, batch) for item in value]
    return value


def masked_mse(
    error: Tensor,
    *,
    pad: Optional[Tensor] = None,
    valid: Optional[Tensor] = None,
    action_loss_weights: Optional[Tensor] = None,
) -> Tensor:
    _, sums, counts = _masked_mse_components(
        error,
        pad=pad,
        valid=valid,
        action_loss_weights=action_loss_weights,
    )
    return sums.sum() / counts.sum().clamp_min(1.0)


def _masked_mse_components(
    error: Tensor,
    *,
    pad: Optional[Tensor] = None,
    valid: Optional[Tensor] = None,
    action_loss_weights: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return per-sample MSE, masked sums, and masked token counts.

    The public ``masked_mse`` reduces the same components globally.  Keeping
    the sums/counts lets the RollFlow gate make a decision per sample without
    changing the scale of the original token-weighted batch objective.
    """
    squared_error = error.float().pow(2)
    if action_loss_weights is None:
        token_loss = squared_error.mean(dim=-1)
    else:
        weights = action_loss_weights.to(device=error.device, dtype=squared_error.dtype)
        if weights.ndim != 1 or weights.shape[0] != error.shape[-1]:
            raise ValueError(
                "action_loss_weights must have shape [action_dim] matching error"
            )
        token_loss = (squared_error * weights).sum(dim=-1) / weights.sum().clamp_min(1e-12)
    mask = torch.ones_like(token_loss, dtype=torch.bool)
    if pad is not None:
        mask &= ~pad.bool()
    if valid is not None:
        v = valid.squeeze(-1) if valid.ndim == 3 else valid
        mask &= v.bool()
    # ``where`` prevents NaNs in inactive/padded tokens from contaminating the
    # per-sample sums, while retaining NaNs in selected active tokens so the
    # sample-wise finite check can reject that sample.
    masked_token_loss = torch.where(mask, token_loss, torch.zeros_like(token_loss))
    sums = masked_token_loss.sum(dim=1)
    counts = mask.sum(dim=1).to(dtype=sums.dtype)
    per_sample = sums / counts.clamp_min(1.0)
    return per_sample, sums, counts


def masked_mean(value: Tensor, valid: Tensor) -> Tensor:
    """Mean over valid action tokens and all feature dimensions."""
    mask = valid.expand_as(value).bool()
    return value[mask].mean() if bool(mask.any()) else value.new_zeros(())


@torch.no_grad()
def ot_match(x1: Tensor, x0: Tensor, pad: Optional[Tensor] = None) -> Tensor:
    """Match noise to targets, excluding padded target timesteps from cost."""
    if x1.shape != x0.shape or x1.ndim != 3:
        raise ValueError("x1 and x0 must have identical [B,H,A] shapes")
    if pad is not None and pad.shape != x1.shape[:2]:
        raise ValueError(f"pad must have shape {tuple(x1.shape[:2])}")
    if x1.shape[0] <= 1:
        return x0
    from scipy.optimize import linear_sum_assignment

    if pad is None or not bool(pad.any()):
        cost = torch.cdist(x1.float().flatten(1), x0.float().flatten(1))
    else:
        valid = (~pad.bool()).float().unsqueeze(-1)
        squared_error = (x1.float()[:, None] - x0.float()[None, :]).square()
        valid_dims = valid.sum(dim=(1, 2)).clamp_min(1) * x1.shape[-1]
        cost = (squared_error * valid[:, None]).sum(dim=(-2, -1)) / valid_dims[:, None]
    cost = cost.cpu()
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
        self._check_actions(x)
        cfg = self.cfg
        B = x.shape[0]
        if pad is not None and pad.shape != (B, cfg.horizon):
            raise ValueError(f"pad must be [B,{cfg.horizon}]")

        x1 = x
        x0 = torch.randn_like(x1)
        if cfg.use_ot:
            x0 = ot_match(x1, x0, pad=pad)

        curriculum = cfg.fm_curriculum_steps
        progress = 1.0 if curriculum == 0 else min(max(step, 0) / curriculum, 1.0)
        p_fm = 1.0 - progress * (1.0 - cfg.p_fm)
        times = self.time.sample_training(B, device=x.device, p_fm=p_fm)
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
            use_lsd_scaling=cfg.use_lsd_scaling,
            use_lsd_gate=cfg.use_lsd_gate,
            action_loss_weights=(
                None
                if cfg.action_loss_weights is None
                else torch.as_tensor(
                    cfg.action_loss_weights,
                    device=x.device,
                    dtype=torch.float32,
                )
            ),
            model_kwargs=model_kwargs,
        )

        return loss, {
            "loss": float(loss.detach().cpu()),
            "fm_loss": float(stats["fm_loss"].cpu()),
            "fm_loss_active": float(stats["fm_loss_active"].cpu()),
            "lsd_loss": float(stats["lsd_loss"].cpu()),
            "lsd_loss_raw": float(stats["lsd_loss_raw"].cpu()),
            "lsd_loss_metric": float(stats["lsd_loss_metric"].cpu()),
            "lsd_scaling_enabled": float(stats["lsd_scaling_enabled"].cpu()),
            "lsd_gate_active": float(stats["lsd_gate_active"].cpu()),
            "lsd_gate_enabled": float(stats["lsd_gate_enabled"].cpu()),
            "lsd_finite_frac": float(stats["lsd_finite_frac"].cpu()),
            "lsd_keep_frac": float(stats["lsd_keep_frac"].cpu()),
            # Sample-wise gate summaries.  Keep these scalar so framework
            # logging remains compatible with trackers expecting one number.
            "lsd_active_sample_frac": float(stats["lsd_active_sample_frac"].cpu()),
            "lsd_finite_sample_frac": float(stats["lsd_finite_frac"].cpu()),
            "lsd_keep_sample_frac": float(stats["lsd_keep_frac"].cpu()),
            "lsd_reject_sample_frac": float(stats["lsd_gate_active"].cpu()),
            "lsd_keep_token_frac": float(stats["lsd_keep_token_frac"].cpu()),
            "lsd_budget": float(stats["lsd_budget"].cpu()),
            "active_lsd_frac": float(stats["active_lsd_frac"].cpu()),
            "num_time_groups": times.num_time_groups,
            "train_block_size": times.block_size,
            "p_fm": p_fm,
            "fm_only_frac": float(times.fm_mask.float().mean().cpu()),
            "source_ratio": float(times.ratio.mean().cpu()),
            "s_min": float(times.s.min().cpu()),
            "s_max": float(times.s.max().cpu()),
            "s_mean": float(times.s.mean().cpu()),
            "t_min": float(times.t.min().cpu()),
            "t_max": float(times.t.max().cpu()),
            "t_mean": float(times.t.mean().cpu()),
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
