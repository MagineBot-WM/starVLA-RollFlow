"""Replay held-out G2 episodes through EVA DatasetTransport and the EVA policy API.

This evaluates the real request/response and rolling-cache path, but never calls
``publish_action`` and never starts the G2 hardware node. Recorded observations are
sampled once per 16-action policy chunk, as in rolling inference.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from .client import EvaPolicyClient
from .contract import ACTION_CHUNK_SIZE, CAMERA_ORDER, HORIZON, REFINEMENT_STEPS
from .replay_transport import LeRobotReplayTransport

DEFAULT_DATASET = Path(os.environ.get("STARVLA_G2_DATASET", str(
    Path(__file__).resolve().parents[4] / "datasets/AgiBotG2/white_carton_to_left_tan_carton_20261009")))
DEFAULT_HELDOUT_EPISODES = (0, 13, 27, 40, 54, 67, 81, 94, 108)


def _dataset_transport(dataset_dir: Path, episode_id: int):
    """Build the read-only LeRobot source implementing EVA's Observation surface."""
    return LeRobotReplayTransport(dataset_dir=dataset_dir, episode_id=episode_id)


def _episode_metrics(predicted: list[np.ndarray], target: list[np.ndarray]) -> dict[str, Any]:
    pred = np.concatenate(predicted, axis=0).astype(np.float64)
    truth = np.concatenate(target, axis=0).astype(np.float64)
    error = pred - truth
    pred_flat, truth_flat = pred.reshape(-1, pred.shape[-1]), truth.reshape(-1, truth.shape[-1])
    denom = np.linalg.norm(pred_flat, axis=1) * np.linalg.norm(truth_flat, axis=1)
    cosine = np.sum(pred_flat * truth_flat, axis=1) / np.maximum(denom, 1e-12)
    return {
        "frames_scored": int(pred.shape[0]),
        "chunks_inferred": len(predicted),
        "mae_rad": float(np.abs(error).mean()),
        "rmse_rad": float(np.sqrt(np.square(error).mean())),
        "max_abs_error_rad": float(np.abs(error).max()),
        "mean_joint_vector_cosine": float(cosine.mean()),
        "per_joint_rmse_rad": np.sqrt(np.square(error).mean(axis=0)).tolist(),
    }


def _write_json(path: Path, content: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(content, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10094)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--episodes", default=",".join(map(str, DEFAULT_HELDOUT_EPISODES)))
    parser.add_argument("--output", type=Path, default=Path("work_dirs/eva_validation/g2_legacy_replay.json"))
    parser.add_argument("--expected-checkpoint")
    parser.add_argument("--max-chunks", type=int, default=0, help="0 evaluates all selected episodes")
    args = parser.parse_args()

    episodes = [int(value.strip()) for value in args.episodes.split(",") if value.strip()]
    if not episodes:
        raise ValueError("at least one episode id is required")
    if not args.dataset.is_dir():
        raise FileNotFoundError(args.dataset)

    result: dict[str, Any] = {
        "dataset": str(args.dataset.resolve()),
        "episodes_requested": episodes,
        "inference_contract": {
            "transport": "EVA-compatible LeRobot replay source (raw BGR HWC + raw 24-D qpos)",
            "camera_order": list(CAMERA_ORDER),
            "horizon": HORIZON,
            "output_chunk": ACTION_CHUNK_SIZE,
            "refinement_steps": REFINEMENT_STEPS,
            "rolling_cache": "reset once per episode; preserved between 16-step requests",
            "action_publishing": False,
            "scope": "recorded-observation replay; not closed-loop dynamics or robot success",
        },
        "server": {},
        "episode_metrics": {},
        "chunks": [],
        "complete": False,
    }

    with EvaPolicyClient(
        args.host,
        args.port,
        expected_checkpoint=args.expected_checkpoint,
    ) as client:
        result["server"] = client.metadata
        all_predicted: list[np.ndarray] = []
        all_target: list[np.ndarray] = []
        chunks_seen = 0

        for episode_id in episodes:
            transport = _dataset_transport(args.dataset, episode_id)
            try:
                missing_cameras = sorted(set(CAMERA_ORDER) - set(transport.available_camera_keys()))
                if missing_cameras:
                    raise RuntimeError(
                        f"episode {episode_id} lacks EVA camera streams: {missing_cameras}"
                    )
                actions = transport.get_action_trajectory()
                if actions.ndim != 2 or actions.shape[1] != 24:
                    raise RuntimeError(f"episode {episode_id} action table shape is {actions.shape}")
                episode_predicted: list[np.ndarray] = []
                episode_target: list[np.ndarray] = []
                chunk_records = []
                client.reset()

                for frame_index in range(0, transport.n_steps, ACTION_CHUNK_SIZE):
                    if args.max_chunks and chunks_seen >= args.max_chunks:
                        break
                    transport.seek(frame_index)
                    observation = transport.get_frame()
                    if observation is None:
                        raise RuntimeError(f"EVA transport returned no frame at {frame_index}")
                    started = time.perf_counter()
                    prediction = client.infer(
                        observation.images,
                        observation.state_qpos,
                        transport.current_task,
                    )
                    latency_ms = (time.perf_counter() - started) * 1000
                    valid = min(ACTION_CHUNK_SIZE, actions.shape[0] - frame_index)
                    if valid <= 0:
                        continue
                    pred_valid = prediction[:valid]
                    target_valid = actions[frame_index : frame_index + valid]
                    episode_predicted.append(pred_valid)
                    episode_target.append(target_valid)
                    all_predicted.append(pred_valid)
                    all_target.append(target_valid)
                    chunk_records.append(
                        {
                            "episode_id": episode_id,
                            "frame_index": frame_index,
                            "valid_actions": valid,
                            "latency_ms": latency_ms,
                            "rmse_rad": float(np.sqrt(np.square(pred_valid - target_valid).mean())),
                            "first_action_rmse_rad": float(
                                np.sqrt(np.square(pred_valid[0] - target_valid[0]).mean())
                            ),
                        }
                    )
                    chunks_seen += 1

                result["chunks"].extend(chunk_records)
                if episode_predicted:
                    result["episode_metrics"][str(episode_id)] = _episode_metrics(
                        episode_predicted, episode_target
                    )
                _write_json(args.output, result)
                if args.max_chunks and chunks_seen >= args.max_chunks:
                    break
            finally:
                transport.close()

        if all_predicted:
            result["overall_metrics"] = _episode_metrics(all_predicted, all_target)
            result["latency_ms"] = {
                "mean": float(np.mean([row["latency_ms"] for row in result["chunks"]])),
                "p50": float(np.percentile([row["latency_ms"] for row in result["chunks"]], 50)),
                "p95": float(np.percentile([row["latency_ms"] for row in result["chunks"]], 95)),
            }
        result["complete"] = len(result["episode_metrics"]) == len(episodes) or bool(args.max_chunks)
        _write_json(args.output, result)

    print(f"Saved EVA replay report: {args.output}")
    print(json.dumps(result.get("overall_metrics", {}), indent=2))


if __name__ == "__main__":
    main()
