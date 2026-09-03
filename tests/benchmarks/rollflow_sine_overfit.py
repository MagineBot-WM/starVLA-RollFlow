"""Tiny end-to-end learning check for the standalone RollFlow action head.

The task is deterministic: a phase observation conditions an ``H``-step sine
chunk, and inference executes ``C`` samples at a time while rolling the cached
tail forward.  It intentionally avoids a real VLM so failures are localized to
the action head and RollFlow algorithm.

Run from the repository root, for example::

    python tests/benchmarks/rollflow_sine_overfit.py --steps 1000
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from starVLA.model.modules.action_model.GR00T_RollFlow_ActionHeader import RollFlowActionHead  # noqa: E402


def make_config() -> SimpleNamespace:
    action_model = SimpleNamespace(
        action_model_type="DiT-B",
        action_horizon=32,
        execution_horizon=8,
        action_dim=1,
        state_dim=0,
        hidden_size=32,
        add_pos_embed=True,
        max_seq_len=32,
        num_target_vision_tokens=2,
        num_timestep_buckets=1000,
        finite_difference_delta=0.01,
        train_block_sizes=[1, 2, 4],
        inference_steps=4,
        w_fm=1.0,
        w_lsd=0.25,
        use_ot=True,
        clip_velocity=0.0,
        iterative_cold_start=True,
        reset_cache_each_step=False,
        diffusion_model_cfg={
            "num_attention_heads": 2,
            "attention_head_dim": 16,
            "output_dim": 32,
            "num_layers": 2,
            "cross_attention_dim": 8,
            "dropout": 0.0,
            "final_dropout": False,
            "positional_embeddings": None,
            "interleave_self_attention": True,
            "norm_type": "ada_norm",
        },
    )
    return SimpleNamespace(framework=SimpleNamespace(action_model=action_model))


def phase_batch(
    phase: torch.Tensor,
    *,
    horizon: int,
    phase_step: float,
    context_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return fixed phase features and the target sine chunk."""
    normalized = phase / (20.0 * math.pi)
    features = torch.stack(
        (phase.sin(), phase.cos(), normalized, normalized.square(), torch.ones_like(phase)),
        dim=-1,
    )
    context = torch.zeros(phase.shape[0], 1, context_dim, device=phase.device, dtype=phase.dtype)
    context[..., : features.shape[-1]] = features[:, None]
    offsets = torch.arange(horizon, device=phase.device, dtype=phase.dtype) * phase_step
    actions = torch.sin(phase[:, None] + offsets[None, :]).unsqueeze(-1)
    return context, actions


@torch.inference_mode()
def rollout(
    model: RollFlowActionHead,
    phases: torch.Tensor,
    *,
    phase_step: float,
    reset_each_step: bool,
) -> tuple[np.ndarray, float]:
    model.eval()
    model.reset_cache()
    predictions = []
    start = time.perf_counter()
    chunk_size = model.execution_horizon
    for index in range(0, phases.numel(), chunk_size):
        if reset_each_step:
            model.reset_cache()
        phase = phases[index]
        context, _ = phase_batch(
            phase[None],
            horizon=model.action_horizon,
            phase_step=phase_step,
            context_dim=model.model.config.cross_attention_dim,
        )
        predictions.extend(model.predict_action(context)[0, :, 0].float().cpu().tolist())
    elapsed = time.perf_counter() - start
    return np.asarray(predictions[: phases.numel()]), elapsed


@torch.inference_mode()
def first_action_predictions(
    model: RollFlowActionHead,
    phases: torch.Tensor,
    *,
    phase_step: float,
) -> np.ndarray:
    """Cold-start every phase so odd phase anchors are a true held-out set."""
    model.eval()
    predictions = []
    for phase in phases:
        model.reset_cache()
        context, _ = phase_batch(
            phase[None],
            horizon=model.action_horizon,
            phase_step=phase_step,
            context_dim=model.model.config.cross_attention_dim,
        )
        predictions.append(float(model.predict_action(context)[0, 0, 0].cpu()))
    return np.asarray(predictions)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps <= 0 or args.batch_size <= 0 or args.repeats <= 0:
        raise ValueError("steps, batch-size, and repeats must be positive")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    model = RollFlowActionHead(make_config()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-6)

    phase_step = math.pi / 16.0
    rollout_phases = torch.linspace(-20.0 * math.pi, 20.0 * math.pi, 641, device=device)
    train_phases = rollout_phases[::2]
    truth = torch.sin(rollout_phases).cpu().numpy()

    torch.manual_seed(args.seed + 1)
    initial_warm, _ = rollout(model, rollout_phases, phase_step=phase_step, reset_each_step=False)
    initial_rmse = float(np.sqrt(np.mean((initial_warm - truth) ** 2)))

    model.train()
    started = time.perf_counter()
    losses = []
    for step in range(1, args.steps + 1):
        indices = torch.randint(train_phases.numel(), (args.batch_size,), device=device)
        phase = train_phases[indices]
        context, actions = phase_batch(
            phase,
            horizon=model.action_horizon,
            phase_step=phase_step,
            context_dim=model.model.config.cross_attention_dim,
        )
        context = context.repeat(args.repeats, 1, 1)
        actions = actions.repeat(args.repeats, 1, 1)

        optimizer.zero_grad(set_to_none=True)
        loss = model(context, actions)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}: {loss.item()}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        last_loss = float(loss.detach().cpu())
        losses.append(last_loss)
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            stats = model.last_loss_stats
            print(
                f"step={step:04d} loss={last_loss:.6f} "
                f"fm={stats['fm_loss']:.6f} lsd={stats['lsd_loss']:.6f} "
                f"G={stats['train_block_size']} levels={stats['num_time_levels']}"
            )

    train_seconds = time.perf_counter() - started
    torch.manual_seed(args.seed + 2)
    warm, warm_seconds = rollout(model, rollout_phases, phase_step=phase_step, reset_each_step=False)
    torch.manual_seed(args.seed + 2)
    cold, cold_seconds = rollout(model, rollout_phases, phase_step=phase_step, reset_each_step=True)

    warm_rmse = float(np.sqrt(np.mean((warm - truth) ** 2)))
    cold_rmse = float(np.sqrt(np.mean((cold - truth) ** 2)))
    torch.manual_seed(args.seed + 2)
    phase_predictions = first_action_predictions(model, rollout_phases, phase_step=phase_step)
    train_anchor_rmse = float(np.sqrt(np.mean((phase_predictions[::2] - truth[::2]) ** 2)))
    heldout_phase_rmse = float(np.sqrt(np.mean((phase_predictions[1::2] - truth[1::2]) ** 2)))
    max_abs_error = float(np.max(np.abs(warm - truth)))
    delta_rmse = float(np.sqrt(np.mean((np.diff(warm) - np.diff(truth)) ** 2)))

    # Verify the exact rolling call contract: K calls to initialize, then one
    # warm call, and K calls again after an episode reset.
    reference_predict = model._predict_velocity
    call_count = 0

    def counted_predict(*call_args, **call_kwargs):
        nonlocal call_count
        call_count += 1
        return reference_predict(*call_args, **call_kwargs)

    model._predict_velocity = counted_predict
    probe_context, _ = phase_batch(
        rollout_phases[:1],
        horizon=model.action_horizon,
        phase_step=phase_step,
        context_dim=model.model.config.cross_attention_dim,
    )
    model.reset_cache()
    torch.manual_seed(args.seed + 3)
    first_cold = model.predict_action(probe_context)
    cold_calls = call_count
    model.predict_action(probe_context)
    warm_calls = call_count - cold_calls
    model.reset_cache()
    before_reset_cold = call_count
    torch.manual_seed(args.seed + 3)
    repeated_cold = model.predict_action(probe_context)
    reset_cold_calls = call_count - before_reset_cold
    cold_reproducible = bool(torch.allclose(first_cold, repeated_cold))
    cache_info = model.cache_info
    model._predict_velocity = reference_predict

    model.reset_cache()
    reset_ok = model.cache_info is None
    window = min(50, len(losses))
    initial_loss_mean = float(np.mean(losses[:window]))
    final_loss_mean = float(np.mean(losses[-window:]))
    passed = bool(
        final_loss_mean < initial_loss_mean * 0.25
        and warm_rmse < 0.15
        and heldout_phase_rmse < 0.15
        and delta_rmse < 0.15
        and cold_calls == 4
        and warm_calls == 1
        and reset_cold_calls == 4
        and cold_reproducible
        and cache_info is not None
        and cache_info.shape == (1, 32, 1)
        and cache_info.refinement_steps == 4
        and reset_ok
    )
    result = {
        "device": str(device),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "steps": args.steps,
        "raw_batch_size": args.batch_size,
        "repeated_diffusion_steps": args.repeats,
        "initial_warm_rmse": initial_rmse,
        "initial_loss_mean": initial_loss_mean,
        "final_loss_mean": final_loss_mean,
        "warm_rollout_rmse": warm_rmse,
        "cold_rollout_rmse": cold_rmse,
        "train_anchor_rmse": train_anchor_rmse,
        "heldout_phase_rmse": heldout_phase_rmse,
        "warm_max_abs_error": max_abs_error,
        "warm_delta_rmse": delta_rmse,
        "train_seconds": train_seconds,
        "warm_rollout_seconds": warm_seconds,
        "cold_rollout_seconds": cold_seconds,
        "rollout_points": int(rollout_phases.numel()),
        "cold_calls": cold_calls,
        "warm_calls": warm_calls,
        "reset_cold_calls": reset_cold_calls,
        "cold_reproducible_after_reset": cold_reproducible,
        "cache_reset_ok": reset_ok,
        "passed": passed,
    }
    print(json.dumps(result, indent=2))
    if not passed:
        raise RuntimeError("RollFlow sine overfit gates failed")


if __name__ == "__main__":
    main()
