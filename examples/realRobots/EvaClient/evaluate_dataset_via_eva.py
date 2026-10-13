"""Replay LeRobot episodes through EVA's native dataset and policy/action pipeline.

This evaluator never calls ``publish_action``. For the G2 held-out set it samples
one observation per 16-action rolling chunk and compares EVA-processed predictions
with the recorded actions.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EVA_ROOT = ROOT / "EVA-CLIENT"
DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "g2_replay.py"
DEFAULT_EPISODES = (0, 13, 27, 40, 54, 67, 81, 94, 108)


def _bootstrap(eva_root: Path):
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(eva_root / "src"))
    import policy_client  # noqa: F401
    import robots  # noqa: F401
    import transport  # noqa: F401
    from examples.realRobots.EvaClient import eva_extension  # noqa: F401
    from core.app.handlers.io import fetch_action_chunk
    from core.app.handlers.recording import build_policy_observation
    from core.app.state import RuntimeState, SessionState
    from core.config import load_config
    from core.registry import POLICY_REGISTRY, ROBOT_REGISTRY, TRANSPORT_REGISTRY, STRATEGY_REGISTRY
    from policy_client.base import PolicyBuildContext

    return (
        fetch_action_chunk,
        build_policy_observation,
        RuntimeState,
        SessionState,
        load_config,
        POLICY_REGISTRY,
        ROBOT_REGISTRY,
        TRANSPORT_REGISTRY,
        PolicyBuildContext,
        STRATEGY_REGISTRY,
    )


def _metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    error = predicted.astype(np.float64) - target.astype(np.float64)
    p = predicted.reshape(-1, predicted.shape[-1]).astype(np.float64)
    t = target.reshape(-1, target.shape[-1]).astype(np.float64)
    denom = np.linalg.norm(p, axis=1) * np.linalg.norm(t, axis=1)
    cosine = (p * t).sum(axis=1) / np.maximum(denom, 1e-12)
    return {
        "actions_scored": int(predicted.shape[0]),
        "mse": float(np.square(error).mean()),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "max_abs_error": float(np.abs(error).max()),
        "mean_action_vector_cosine": float(cosine.mean()),
        "per_dimension_rmse": np.sqrt(np.square(error).mean(axis=0)).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eva-root", type=Path, default=DEFAULT_EVA_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--episodes", default=",".join(map(str, DEFAULT_EPISODES)))
    parser.add_argument("--output", type=Path, default=ROOT / "work_dirs/eva_validation/g2_replay.json")
    parser.add_argument("--max-chunks", type=int, default=0)
    args = parser.parse_args()
    (
        fetch_action_chunk,
        build_policy_observation,
        RuntimeState,
        SessionState,
        load_config,
        policy_registry,
        robot_registry,
        transport_registry,
        PolicyBuildContext,
        strategy_registry,
    ) = _bootstrap(args.eva_root)

    config = load_config(args.config)
    if args.dataset is not None:
        config.transport.dataset_dir = str(args.dataset)
    episodes = [int(item) for item in args.episodes.split(",") if item.strip()]
    if not episodes:
        raise ValueError("pass at least one episode id")
    robot = robot_registry.build(config.robot.type)
    policy = policy_registry.build(
        config.policy.type,
        config.policy,
        PolicyBuildContext(action_dim=robot.total_action_dim),
    )
    strategy_spec = config.inference_strategies.rolling
    strategy = strategy_registry.build(strategy_spec.type, **strategy_spec.args)
    all_predicted: list[np.ndarray] = []
    all_target: list[np.ndarray] = []
    trace_arrays: dict[str, np.ndarray] = {}
    chunk_rows: list[dict[str, Any]] = []
    episode_metrics: dict[str, Any] = {}
    start_time = time.perf_counter()

    for episode_id in episodes:
        config.transport.episode_id = episode_id
        transport = transport_registry.build(config.transport.type, config, robot)
        try:
            target = np.asarray(transport.get_action_trajectory(), dtype=np.float32)
            chunk_size = int(policy.metadata["action_chunk_size"])
            expected_dim = int(policy.metadata["action_dim"])
            if target.ndim != 2 or target.shape[1] != expected_dim:
                raise ValueError(f"episode {episode_id} target has invalid shape {target.shape}")
            runtime = RuntimeState(robot=robot, transport=transport, policy=policy)
            prompt = str(transport.current_task)
            session = SessionState(selected_task=prompt)
            policy.reset()
            strategy.reset()
            predictions: list[np.ndarray] = []
            targets: list[np.ndarray] = []
            for frame_index in range(0, len(target), chunk_size):
                if args.max_chunks and len(chunk_rows) >= args.max_chunks:
                    break
                transport.seek(frame_index)
                frame = transport.get_frame()
                if frame is None:
                    break
                observation = build_policy_observation(frame, prompt, config, runtime)
                infer_start = time.perf_counter()
                predicted_chunk = strategy.take_or_fetch_chunk(
                    prompt, lambda task, _obs: fetch_action_chunk(
                        config, runtime, task, session, prebuilt_observation=observation)
                )
                latency_ms = (time.perf_counter() - infer_start) * 1000.0
                valid = min(chunk_size, len(target) - frame_index)
                predicted = predicted_chunk[:valid]
                truth = target[frame_index : frame_index + valid]
                predictions.append(predicted)
                targets.append(truth)
                all_predicted.append(predicted)
                all_target.append(truth)
                chunk_rows.append({
                    "episode_id": episode_id,
                    "frame_index": frame_index,
                    "valid_actions": valid,
                    "latency_ms": latency_ms,
                    "rmse_after_eva_action_processing": float(np.sqrt(np.square(predicted - truth).mean())),
                })
            if predictions:
                trace_arrays[f"episode_{episode_id}_prediction"] = np.concatenate(predictions)
                trace_arrays[f"episode_{episode_id}_target"] = np.concatenate(targets)
                episode_metrics[str(episode_id)] = _metrics(
                    np.concatenate(predictions), np.concatenate(targets)
                )
                print(f"episode={episode_id} chunks={len(predictions)} "
                      f"rmse={episode_metrics[str(episode_id)]['rmse']:.6f}", flush=True)
        finally:
            transport.close()

    result = {
        "config": str(args.config.resolve()),
        "dataset": str(Path(config.transport.dataset_dir).resolve()),
        "server": policy.metadata,
        "episodes_requested": episodes,
        "episode_metrics": episode_metrics,
        "chunks": chunk_rows,
        "overall_metrics": _metrics(np.concatenate(all_predicted), np.concatenate(all_target)) if all_predicted else {},
        "elapsed_seconds": time.perf_counter() - start_time,
        "action_publishing": False,
        "inference_path": "EVA DatasetTransport -> build_policy_observation -> StarVlaRollingStrategy -> PolicyClient -> fetch_action_chunk/action processing",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    trace_path = args.output.with_suffix(".npz")
    np.savez_compressed(trace_path, **trace_arrays)
    result["trajectory_artifact"] = str(trace_path)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved EVA replay report: {args.output}")
    print(json.dumps(result["overall_metrics"], indent=2))
    policy.close()


if __name__ == "__main__":
    main()
