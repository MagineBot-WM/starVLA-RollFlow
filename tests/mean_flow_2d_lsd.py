"""Toy experiment for the current RollFlow FM + finite hard-gate LSD loss.

The toy problem uses the same convention as the production action head:

    x0 ~ N(0, I)          (flow time 0)
    x1 ~ spiral            (flow time 1)
    x_s = (1-s)x0 + s x1
    x_t = (1-t)x0 + t x1

The script deliberately mirrors ``central_difference_lsd`` rather than using
JVPs or the old reverse-time mean-flow test.  It trains one small network and
checks whether the learned field can generate the spiral in 1, 2, 4, 8 and 16
uniform average-velocity Euler steps.  In particular, one step evaluates

    x <- x + model(x, 0, 1),

which is the whole-horizon mean-flow query corresponding to RollFlow's cold
start, not an instantaneous ``model(x, 0, 0)`` query.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import Tensor, nn


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "figures"


class SimpleNet(nn.Module):
    """Small time-conditioned velocity field for the 2-D sanity check."""

    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, z: Tensor, s: Tensor, t: Tensor) -> Tensor:
        if z.ndim != 2 or z.shape[-1] != 2:
            raise ValueError("z must have shape [B, 2]")
        if s.shape != t.shape or s.shape != (z.shape[0], 1):
            raise ValueError("s and t must have shape [B, 1]")
        features = torch.cat((z.float(), t.float(), (t - s).float()), dim=-1)
        return self.net(features).to(dtype=z.dtype)


def sample_prior(n_samples: int, device: torch.device) -> Tensor:
    return torch.randn(n_samples, 2, device=device)


def sample_spiral_2d(
    n_samples: int, *, noise: float = 0.08, device: torch.device
) -> Tensor:
    """Sample points from a noisy, expanding two-dimensional spiral."""
    theta = torch.rand(n_samples, device=device) * (4.0 * torch.pi)
    radius = theta / (2.0 * torch.pi)
    points = torch.stack((radius * torch.cos(theta), radius * torch.sin(theta)), dim=-1)
    return points + noise * torch.randn_like(points)


def lerp_path(x0: Tensor, x1: Tensor, time_value: Tensor) -> Tensor:
    return (1.0 - time_value.float()) * x0.float() + time_value.float() * x1.float()


def masked_mse(error: Tensor, valid: Tensor | None = None) -> Tensor:
    values = error.float().square().mean(dim=-1)
    if valid is None:
        return values.mean()
    valid = valid.bool().reshape(-1)
    if valid.numel() != values.shape[0]:
        raise ValueError("valid must have one entry per sample")
    return values[valid].mean() if bool(valid.any()) else values[valid].sum()


def sample_times(
    batch_size: int,
    *,
    delta: float,
    p_fm: float,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Sample terminal times and the FM/interval mixture used by RollFlow."""
    t = delta + (1.0 - 2.0 * delta) * torch.rand(batch_size, 1, device=device)
    ratio = torch.rand(batch_size, 1, device=device)
    fm_mask = torch.rand(batch_size, 1, device=device) < p_fm
    ratio = torch.where(fm_mask, torch.ones_like(ratio), ratio)
    s = ratio * t
    active = (t - s > delta + 1e-6) & (t + delta <= 1.0 - 1e-6)
    return s, t, active, fm_mask


def rollflow_lsd_loss(
    model: nn.Module,
    x0: Tensor,
    x1: Tensor,
    s: Tensor,
    t: Tensor,
    active: Tensor,
    *,
    delta: float,
    w_lsd: float,
    teacher_clip: float,
) -> tuple[Tensor, dict[str, float]]:
    """Compute the production finite-difference LSD objective on toy data."""
    x_s = lerp_path(x0, x1, s)
    x_t = lerp_path(x0, x1, t)
    v_gt = x1.float() - x0.float()

    t_minus = torch.where(active, t - delta, t)
    t_plus = torch.where(active, t + delta, t)

    with torch.no_grad():
        u_st = model(x_s, s, t)
        x_t_hat = x_s + (t - s) * u_st
        v_teacher = model(x_t_hat, t, t)
        if not bool(torch.isfinite(v_teacher).all()):
            raise FloatingPointError("non-finite toy teacher prediction")
        clip_fraction = (
            (v_teacher.abs() > teacher_clip).float().mean()
            if teacher_clip > 0
            else v_teacher.new_zeros(())
        )
        if teacher_clip > 0:
            v_teacher = v_teacher.clamp(-teacher_clip, teacher_clip)

    student_input = torch.cat((x_s, x_s, x_t), dim=0)
    student_s = torch.cat((s, s, t), dim=0)
    student_t = torch.cat((t_minus, t_plus, t), dim=0)
    u_minus, u_plus, v_instant = model(student_input, student_s, student_t).chunk(3)

    x_minus = x_s + (t_minus - s) * u_minus
    x_plus = x_s + (t_plus - s) * u_plus
    v_tangent = (x_plus - x_minus) / (2.0 * delta)

    loss_fm = masked_mse(v_instant - v_gt)
    loss_fm_active = masked_mse(v_instant - v_gt, active)
    loss_lsd_metric = masked_mse(v_tangent - v_teacher.detach(), active)
    loss_lsd_raw = (2.0 * delta) * loss_lsd_metric

    budget = float(w_lsd) * loss_fm_active.detach()
    keep = torch.isfinite(loss_lsd_raw) & (loss_lsd_raw <= budget)
    safe_raw = torch.nan_to_num(loss_lsd_raw, nan=0.0, posinf=0.0, neginf=0.0)
    loss_lsd = torch.where(keep, safe_raw, torch.zeros_like(safe_raw))
    loss = loss_fm + loss_lsd

    stats = {
        "loss": float(loss.detach()),
        "fm_loss": float(loss_fm.detach()),
        "fm_loss_active": float(loss_fm_active.detach()),
        "lsd_loss_metric": float(loss_lsd_metric.detach()),
        "lsd_loss_raw": float(loss_lsd_raw.detach()),
        "lsd_loss": float(loss_lsd.detach()),
        # 0 means LSD was kept; 1 means non-finite or over-budget LSD was masked.
        "lsd_gate_active": float((~keep).detach()),
        "active_frac": float(active.float().mean()),
        "teacher_clip_frac": float(clip_fraction.detach()),
    }
    return loss, stats


def train(
    *,
    steps: int,
    batch_size: int,
    lr: float,
    delta: float,
    w_lsd: float,
    teacher_clip: float,
    curriculum_steps: int,
    target_p_fm: float,
    device: torch.device,
    log_interval: int,
) -> tuple[SimpleNet, list[dict[str, float]]]:
    model = SimpleNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[dict[str, float]] = []
    started = time.perf_counter()

    for step in range(1, steps + 1):
        progress = 1.0 if curriculum_steps == 0 else min(step / curriculum_steps, 1.0)
        p_fm = 1.0 - progress * (1.0 - target_p_fm)
        x0 = sample_prior(batch_size, device)
        x1 = sample_spiral_2d(batch_size, device=device)
        s, t, active, _ = sample_times(
            batch_size, delta=delta, p_fm=p_fm, device=device
        )

        optimizer.zero_grad(set_to_none=True)
        loss, stats = rollflow_lsd_loss(
            model,
            x0,
            x1,
            s,
            t,
            active,
            delta=delta,
            w_lsd=w_lsd,
            teacher_clip=teacher_clip,
        )
        if not math.isfinite(stats["loss"]):
            raise FloatingPointError(f"non-finite toy loss at step {step}: {stats}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        stats["step"] = float(step)
        stats["p_fm"] = float(p_fm)
        history.append(stats)

        if step == 1 or step % log_interval == 0 or step == steps:
            elapsed = time.perf_counter() - started
            print(
                f"step={step:5d} loss={stats['loss']:.5f} "
                f"fm={stats['fm_loss']:.5f} "
                f"lsd={stats['lsd_loss']:.5f} "
                f"raw={stats['lsd_loss_raw']:.5f} "
                f"gate={int(stats['lsd_gate_active'])} "
                f"p_fm={p_fm:.3f} ({elapsed:.1f}s)"
            )
    return model, history


@torch.no_grad()
def sample_mean_flow(model: nn.Module, noise: Tensor, n_steps: int) -> Tensor:
    """Forward average-velocity Euler decode from time 0 to time 1."""
    if not isinstance(n_steps, int) or isinstance(n_steps, bool) or n_steps <= 0:
        raise ValueError("n_steps must be a positive integer")
    x = noise.clone()
    for index in range(n_steps):
        s = torch.full(
            (x.shape[0], 1), index / n_steps, device=x.device, dtype=torch.float32
        )
        t = torch.full_like(s, (index + 1) / n_steps)
        velocity = model(x, s, t)
        x = (x.float() + (t - s) * velocity.float()).to(dtype=noise.dtype)
    return x


def symmetric_chamfer(predicted: Tensor, target: Tensor) -> float:
    distances = torch.cdist(predicted.float(), target.float())
    return float(
        0.5
        * (distances.min(dim=1).values.mean() + distances.min(dim=0).values.mean())
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    *,
    device: torch.device,
    eval_samples: int,
    step_counts: list[int],
) -> dict[str, float]:
    noise = sample_prior(eval_samples, device)
    target = sample_spiral_2d(eval_samples, device=device)
    metrics: dict[str, float] = {}
    for n_steps in step_counts:
        generated = sample_mean_flow(model, noise, n_steps)
        metrics[f"{n_steps}_step_chamfer"] = symmetric_chamfer(generated, target)
        metrics[f"{n_steps}_step_mean_radius"] = float(generated.norm(dim=-1).mean())
    return metrics


def save_plots(
    model: nn.Module,
    history: list[dict[str, float]],
    *,
    device: torch.device,
    eval_samples: int,
    step_counts: list[int],
) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    noise = sample_prior(eval_samples, device)
    target = sample_spiral_2d(eval_samples, device=device)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes[0, 0].scatter(noise[:, 0].cpu(), noise[:, 1].cpu(), s=4, alpha=0.35)
    axes[0, 0].set_title("Prior x0")
    axes[0, 1].scatter(
        target[:, 0].cpu(), target[:, 1].cpu(), s=4, alpha=0.35, c="tab:red"
    )
    axes[0, 1].set_title("Target x1: spiral")
    generated_1 = sample_mean_flow(model, noise, step_counts[0])
    axes[0, 3].scatter(
        generated_1[:, 0].cpu(), generated_1[:, 1].cpu(), s=4, alpha=0.35
    )
    axes[0, 3].set_title(f"{step_counts[0]} step")
    for axis, n_steps in zip(axes[1], step_counts[1:]):
        generated = sample_mean_flow(model, noise, n_steps)
        axis.scatter(generated[:, 0].cpu(), generated[:, 1].cpu(), s=4, alpha=0.35)
        axis.set_title(f"{n_steps} steps")
    axes[0, 2].plot(
        [row["step"] for row in history],
        [row["fm_loss"] for row in history],
        label="FM",
    )
    axes[0, 2].plot(
        [row["step"] for row in history],
        [row["lsd_loss"] for row in history],
        label="LSD kept",
    )
    axes[0, 2].set_yscale("log")
    axes[0, 2].set_title("Training losses")
    axes[0, 2].legend()
    for axis in axes.flat:
        if axis is not axes[0, 2]:
            axis.set_aspect("equal")
            axis.set_xlim(-3.0, 3.0)
            axis.set_ylim(-3.0, 3.0)
            axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "mean_flow_2d_lsd_hard_gate.png", dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-samples", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--w-lsd", type=float, default=0.1)
    parser.add_argument("--teacher-clip", type=float, default=2.0)
    parser.add_argument("--curriculum-steps", type=int, default=500)
    parser.add_argument("--p-fm", type=float, default=0.3)
    parser.add_argument("--log-interval", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps <= 0 or args.batch_size <= 0 or args.eval_samples <= 0:
        raise ValueError("steps, batch-size and eval-samples must be positive")
    if args.w_lsd < 0 or not 0.0 <= args.p_fm <= 1.0:
        raise ValueError("w-lsd must be non-negative and p-fm must lie in [0, 1]")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    print("Training RollFlow finite hard-gate LSD: Gaussian -> spiral")
    print(
        f"device={device} delta={args.delta} w_lsd={args.w_lsd} "
        f"teacher_clip={args.teacher_clip}"
    )
    model, history = train(
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        delta=args.delta,
        w_lsd=args.w_lsd,
        teacher_clip=args.teacher_clip,
        curriculum_steps=args.curriculum_steps,
        target_p_fm=args.p_fm,
        device=device,
        log_interval=args.log_interval,
    )

    step_counts = [1, 2, 4, 8, 16]
    metrics = evaluate(
        model,
        device=device,
        eval_samples=args.eval_samples,
        step_counts=step_counts,
    )
    print("Decode metrics (symmetric Chamfer; lower is better):")
    for steps in step_counts:
        print(
            f"  {steps:2d} step: chamfer={metrics[f'{steps}_step_chamfer']:.5f}, "
            f"mean_radius={metrics[f'{steps}_step_mean_radius']:.5f}"
        )

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), FIGURE_DIR / "mean_flow_2d_lsd_hard_gate.pt")
    (FIGURE_DIR / "mean_flow_2d_lsd_hard_gate_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    (FIGURE_DIR / "mean_flow_2d_lsd_hard_gate_history.json").write_text(
        json.dumps(history) + "\n"
    )
    save_plots(
        model,
        history,
        device=device,
        eval_samples=args.eval_samples,
        step_counts=step_counts,
    )
    print("Saved model, metrics, history and plot under figures/mean_flow_2d_lsd_hard_gate.*")


if __name__ == "__main__":
    main()
