"""Compare rolling EVA requests with the stock LIBERO ModelClient using one seed.

EVA records observations while executing a closed-loop simulation. The stock
client replays those exact observations in order against a freshly reset model
cache; both raw chunks and LIBERO gripper processing must match.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image


class _RecordingPolicy:
    def __init__(self, policy):
        self.policy = policy
        self.chunks = []

    @property
    def metadata(self):
        return self.policy.metadata

    def infer(self, observation):
        result = self.policy.infer(observation)
        self.chunks.append(np.asarray(result["actions"]).copy())
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-python", default=os.environ.get("STARVLA_MODEL_PYTHON"),
                        help="StarVLA environment's Python executable (or STARVLA_MODEL_PYTHON)")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[3] / "work_dirs/eva_validation/libero_reference_parity.json")
    parser.add_argument("--reference-worker", type=Path)
    parser.add_argument("--config", type=Path)
    from .runtime import DEFAULT_EVA_ROOT
    parser.add_argument("--eva-root", type=Path, default=DEFAULT_EVA_ROOT)
    parser.add_argument("--chunks", type=int, default=1)
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--init-state-id", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10093)
    args = parser.parse_args()
    if args.chunks < 1:
        parser.error("--chunks must be positive")
    if not args.reference_worker and not args.model_python:
        parser.error("pass --model-python or set STARVLA_MODEL_PYTHON")
    if args.reference_worker:
        from .adapters.reference import ReferenceTransport
        from examples.simBenchmarks.LIBERO.eval_files.model2libero_interface import ModelClient
        arrays = np.load(args.reference_worker)
        client = ModelClient(host=args.host, port=args.port, unnorm_key="franka")
        client.client = ReferenceTransport(client.client)
        try:
            client.client.request("reset", {"seed": 7})
            images = {key: arrays[key] for key in ("agentview", "wrist")}
            if images["agentview"].ndim == 3:
                images = {key: value[None] for key, value in images.items()}
            chunks = []
            for index in range(len(images["agentview"])):
                client.step({"image": [images["agentview"][index], images["wrist"][index]],
                             "lang": str(arrays["prompt"])}, step=index * client.action_chunk_size)
                chunks.append(client.raw_actions.copy())
            raw_actions = np.stack(chunks)
            processed = raw_actions.copy()
            # Same mapping as eval_libero._binarize_gripper_open:
            # model > 0.5 means open; LIBERO -1=open, +1=close.
            processed[..., 6] = 1.0 - 2.0 * (raw_actions[..., 6] > 0.5)
            np.savez(args.reference_worker.with_suffix(".reference.npz"),
                     actions=raw_actions, processed_actions=processed)
        finally:
            client.client.close()
        return

    from .evaluate_libero_via_eva import _bootstrap, DEFAULT_EVA_ROOT, DEFAULT_CONFIG
    b = _bootstrap(args.eva_root)
    cfg = b[3](args.config or DEFAULT_CONFIG, args.eva_root)
    cfg.policy.host, cfg.policy.port = args.host, args.port
    cfg.transport.libero.task_id = args.task_id
    cfg.transport.libero.init_state_id = args.init_state_id
    robot = b[5].build(cfg.robot.type)
    transport = b[6].build(cfg.transport.type, cfg, robot)
    from core.app.handlers.recording import build_policy_observation
    from policy_client.base import PolicyBuildContext
    policy = b[4].build(cfg.policy.type, cfg.policy, PolicyBuildContext(action_dim=7))
    recording_policy = _RecordingPolicy(policy)
    try:
        prompt = transport.current_task
        runtime = b[1](robot=robot, transport=transport, policy=recording_policy)
        session = b[2](selected_task=prompt)
        spec = cfg.inference_strategies.rolling
        strategy = b[8].build(spec.type, **spec.args)
        policy.reset(seed=7)
        pixel_max_error = 0
        frame_trace = {"agentview": [], "wrist": []}
        processed_chunks = []
        for _ in range(args.chunks):
            if transport.done:
                break
            frame = transport.get_frame()
            obs = build_policy_observation(frame, prompt, cfg, runtime)
            images = policy._ordered_images(obs["images"])
            rotated = {k: np.ascontiguousarray(v[::-1, ::-1]) for k, v in frame.images.items()}
            pixel_max_error = max(pixel_max_error, *[
                int(np.abs(image.astype(int) - np.asarray(
                    Image.fromarray(rotated[key]).resize((224, 224), Image.Resampling.BILINEAR)
                ).astype(int)).max())
                for key, image in zip(("agentview", "wrist"), images)
            ])
            for key in frame_trace:
                frame_trace[key].append(rotated[key])
            chunk = strategy.take_or_fetch_chunk(prompt, lambda task, _obs:
                b[0](cfg, runtime, task, session, prebuilt_observation=obs))
            processed_chunks.append(chunk.copy())
            for action in chunk:
                transport.publish_action(action, target="sim")
                session.step_index += 1
                if transport.done:
                    break
        eva_actions = np.stack(recording_policy.chunks)
        eva_processed = np.stack(processed_chunks)
        success, steps = transport.success, transport.step_count
        args.output.parent.mkdir(parents=True, exist_ok=True)
        input_path = args.output.with_suffix(".input.npz")
        np.savez_compressed(input_path, **{k: np.stack(v) for k, v in frame_trace.items()},
                            prompt=np.asarray(prompt), eva_raw_actions=eva_actions,
                            eva_processed_actions=eva_processed)
    finally:
        policy.close()
        transport.close()
    subprocess.run([args.model_python, "-m", "examples.deployment.eva.verify_libero_reference",
                    "--reference-worker", str(input_path), "--host", args.host,
                    "--port", str(args.port)], check=True)
    reference = np.load(input_path.with_suffix(".reference.npz"))
    reference_actions = reference["actions"]
    max_action_error = float(np.abs(reference_actions - eva_actions).max())
    max_processed_error = float(np.abs(reference["processed_actions"] - eva_processed).max())
    result = {"pixel_max_abs_error": pixel_max_error, "raw_chunk_max_abs_error": max_action_error,
              "processed_chunk_max_abs_error": max_processed_error,
              "shape": list(eva_actions.shape), "chunks": len(eva_actions),
              "executed_steps": steps, "eva_rollout_success": success, "prompt": prompt,
              "task_id": args.task_id, "init_state_id": args.init_state_id,
              "checkpoint": policy.metadata["ckpt_path"],
              "passed": pixel_max_error == 0 and max_action_error == 0.0 and max_processed_error == 0.0,
              "reference": "Upstream LIBERO ModelClient.step with validation-session transport adapter; same observations/server/checkpoint, reset seed=7",
              "scope": "EVA adapter and cache/chunk/gripper handling; shares the StarVLA server implementation"}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise RuntimeError("EVA vs stock LIBERO client mismatch")


if __name__ == "__main__":
    main()
