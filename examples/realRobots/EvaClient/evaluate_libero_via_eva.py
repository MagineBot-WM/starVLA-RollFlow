"""Closed-loop LIBERO rollouts through EVA's policy, processing and transport APIs."""

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
DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "libero_sim_verified.py"


def _bootstrap(eva_root: Path):
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(eva_root / "src"))
    import policy_client  # noqa: F401
    import robots  # noqa: F401
    import transport  # noqa: F401
    from examples.realRobots.EvaClient import eva_extension  # noqa: F401
    from core.app.handlers.io import fetch_action_chunk
    from core.app.state import RuntimeState, SessionState
    from core.config import load_config
    from core.registry import POLICY_REGISTRY, ROBOT_REGISTRY, TRANSPORT_REGISTRY, STRATEGY_REGISTRY
    from policy_client.base import PolicyBuildContext

    return fetch_action_chunk, RuntimeState, SessionState, load_config, POLICY_REGISTRY, ROBOT_REGISTRY, TRANSPORT_REGISTRY, PolicyBuildContext, STRATEGY_REGISTRY


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eva-root", type=Path, default=DEFAULT_EVA_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--suite", default="libero_goal")
    parser.add_argument("--task-ids", default="0")
    parser.add_argument("--episodes-per-task", type=int, default=1)
    parser.add_argument("--init-state-id", type=int, default=0)
    parser.add_argument("--max-chunks", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ROOT / "work_dirs/eva_validation/libero_rollout.json")
    parser.add_argument("--video-dir", type=Path)
    parser.add_argument("--policy-seed", type=int,
                        help="Reset model RNG to this seed at each episode for paired comparisons")
    args = parser.parse_args()
    (
        fetch_action_chunk,
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
    config.transport.libero.suite = args.suite
    robot = robot_registry.build(config.robot.type)
    transport = transport_registry.build(config.transport.type, config, robot)
    policy = policy_registry.build(
        config.policy.type,
        config.policy,
        PolicyBuildContext(action_dim=robot.total_action_dim),
    )
    runtime = RuntimeState(robot=robot, transport=transport, policy=policy)
    strategy_spec = config.inference_strategies.rolling
    strategy = strategy_registry.build(strategy_spec.type, **strategy_spec.args)
    task_ids = [int(item) for item in args.task_ids.split(",") if item.strip()]
    chunk_size = int(policy.metadata["action_chunk_size"])
    episodes: list[dict[str, Any]] = []
    all_latencies: list[float] = []
    traces: dict[str, np.ndarray] = {}
    writer = None

    try:
        for task_id in task_ids:
            for episode_index in range(args.episodes_per_task):
                transport.reset_episode(task_id, args.init_state_id + episode_index)
                prompt = transport.current_task
                session = SessionState(selected_task=prompt)
                policy.reset(seed=args.policy_seed)
                strategy.reset()
                started = time.perf_counter()
                chunks = 0
                infer_latencies: list[float] = []
                actions_executed: list[np.ndarray] = []
                writer = None
                video_path = None
                if args.video_dir:
                    import imageio.v2 as imageio
                    args.video_dir.mkdir(parents=True, exist_ok=True)
                    video_path = args.video_dir / f"task_{task_id}_episode_{episode_index}.mp4"
                    writer = imageio.get_writer(video_path, fps=20, codec="libx264")
                while not transport.done:
                    if args.max_chunks and chunks >= args.max_chunks:
                        break
                    tic = time.perf_counter()
                    action_chunk = strategy.take_or_fetch_chunk(
                        prompt, lambda task, _obs: fetch_action_chunk(config, runtime, task, session))
                    latency_ms = (time.perf_counter() - tic) * 1000.0
                    infer_latencies.append(latency_ms)
                    all_latencies.append(latency_ms)
                    chunks += 1
                    for action in action_chunk[:chunk_size]:
                        transport.publish_action(action, target="sim")
                        actions_executed.append(np.asarray(action).copy())
                        if writer is not None:
                            views = transport._observation_images()
                            writer.append_data(np.concatenate([
                                views["agentview"][::-1, ::-1], views["wrist"][::-1, ::-1]
                            ], axis=1))
                        session.step_index += 1
                        if transport.done:
                            break
                if writer is not None:
                    writer.close()
                traces[f"task_{task_id}_episode_{episode_index}_actions"] = np.asarray(actions_executed)
                episodes.append({
                    "task_id": task_id,
                    "episode_index": episode_index,
                    "init_state_id": args.init_state_id + episode_index,
                    "task": prompt,
                    "success": bool(transport.success),
                    "steps": int(transport.step_count),
                    "chunks": chunks,
                    "mean_inference_ms": float(np.mean(infer_latencies)) if infer_latencies else None,
                    "elapsed_seconds": time.perf_counter() - started,
                    "complete": bool(transport.done),
                    "video": str(video_path) if video_path else None,
                })
                print(
                    f"task={task_id} episode={episode_index} success={transport.success} "
                    f"steps={transport.step_count} chunks={chunks}"
                )
    finally:
        if writer is not None:
            writer.close()
        transport.close()
        policy.close()

    result = {
        "config": str(args.config.resolve()),
        "server": policy.metadata,
        "suite": args.suite,
        "task_ids": task_ids,
        "policy_seed_per_episode": args.policy_seed,
        "episodes": episodes,
        "success_rate": float(np.mean([item["success"] for item in episodes])) if episodes else None,
        "mean_inference_ms": float(np.mean(all_latencies)) if all_latencies else None,
        "rollout_path": "EVA StarVlaRollingStrategy -> fetch_action_chunk -> gripper/action normalization -> LiberoSimTransport.publish_action -> closed-loop next observation",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    trace_path = args.output.with_suffix(".npz")
    np.savez_compressed(trace_path, **traces)
    result["trajectory_artifact"] = str(trace_path)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved LIBERO EVA rollout report: {args.output}")


if __name__ == "__main__":
    main()
