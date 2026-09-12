#!/usr/bin/env python3
"""Offline G1 training-trajectory regression for two rolling decode layouts.

The checkpoint and dataset are kept read-only.  We score the first ``N``
training episodes on identical observation anchors and compare the executable
prefix of each 32-step target window:

* ``chunk_size=16, refinement_steps=2`` -> 16 actions per request;
* ``chunk_size=8, refinement_steps=4``  -> 8 actions per request;
* ``chunk_size=4, refinement_steps=8``  -> 4 actions per request.
* ``chunk_size=32, refinement_steps=1`` -> 32 actions per request.

Metrics are computed in the normalized action space used by the action head.
This is a fitting diagnostic, not closed-loop robot success.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from deployment.model_server.policy_wrapper import PolicyServerWrapper
from starVLA.dataloader.lerobot_datasets import make_LeRobotSingleDataset
from starVLA.model.modules.action_model.rolling_meanflow_matching_head.rolling_meanflow import RollFlow


DEFAULT_CHECKPOINT = Path(
    "/data/tzq/starVLA_checkpoints/"
    "libero10hz_pick_up_the_apple_corrected_55k_lsd_finite_hard_gate_v1/"
    "checkpoints/steps_8000_pytorch_model.pt"
)
DEFAULT_DATA_ROOT = Path("/data/tzq/datasets/starVLA/Datasets")
DEFAULT_OUTPUT = Path(
    "/data/tzq/starVLA_checkpoints/"
    "libero10hz_pick_up_the_apple_corrected_55k_lsd_finite_hard_gate_v1/"
    "results/g1_training_regression_8000"
)


def _episode_samples(dataset, episode: int, max_anchors: int) -> tuple[list[dict], np.ndarray]:
    position = np.flatnonzero(dataset.trajectory_ids == episode)
    if len(position) != 1:
        raise ValueError(f"episode {episode} was not found exactly once")
    length = int(dataset.trajectory_lengths[position[0]])
    anchors = np.arange(0, max(0, length - 31), 8, dtype=np.int64)
    if max_anchors > 0:
        anchors = anchors[:max_anchors]
    if len(anchors) == 0:
        raise ValueError(f"episode {episode} has fewer than 32 frames")

    samples = []
    for anchor in anchors.tolist():
        raw = dataset.get_step_data(int(episode), int(anchor))
        sample = dataset._pack_sample(dataset.transforms(raw))
        action = np.asarray(sample["action"], dtype=np.float32)
        state = np.asarray(sample["state"], dtype=np.float32)
        if action.shape != (32, 22) or state.shape != (1, 22):
            raise ValueError(
                f"unexpected G1 sample shape at episode={episode}, anchor={anchor}: "
                f"action={action.shape}, state={state.shape}"
            )
        if not np.isfinite(action).all() or not np.isfinite(state).all():
            raise ValueError(f"non-finite G1 sample at episode={episode}, anchor={anchor}")
        if len(sample["image"]) != 3:
            raise ValueError(f"G1 expects 3 cameras, got {len(sample['image'])}")
        samples.append(
            {
                "image": sample["image"],
                "lang": sample["lang"],
                "action": action,
                "state": state,
                "episode": int(episode),
                "anchor": int(anchor),
            }
        )
    return samples, anchors


@torch.inference_mode()
def _encode_context(framework, samples: list[dict]) -> tuple[torch.Tensor, torch.Tensor | None]:
    interface = framework.qwen_vl_interface
    inputs = interface.build_qwenvl_inputs(
        images=[sample["image"] for sample in samples],
        instructions=[sample["lang"] for sample in samples],
    )
    outputs = interface(
        **inputs,
        output_attentions=False,
        output_hidden_states=True,
        return_dict=True,
    )
    hidden = outputs.hidden_states[-1].detach()
    mask = inputs.get("attention_mask")
    return hidden, None if mask is None else mask.to(dtype=torch.bool)


def _score(predictions: list[np.ndarray], targets: list[np.ndarray]) -> dict[str, float | int]:
    pred = np.concatenate(predictions, axis=0).astype(np.float32, copy=False)
    truth = np.concatenate(targets, axis=0).astype(np.float32, copy=False)
    if pred.shape != truth.shape or pred.ndim != 2 or pred.shape[-1] != 22:
        raise ValueError(f"metric inputs must be [N,22], got {pred.shape} and {truth.shape}")
    error = pred - truth
    pred_t = torch.from_numpy(pred)
    truth_t = torch.from_numpy(truth)
    frame_cosine = F.cosine_similarity(pred_t, truth_t, dim=-1, eps=1e-8)
    global_cosine = F.cosine_similarity(pred_t.reshape(1, -1), truth_t.reshape(1, -1), dim=-1)
    return {
        "frames": int(pred.shape[0]),
        "normalized_mse": float(np.mean(error * error)),
        "normalized_rmse": float(np.sqrt(np.mean(error * error))),
        "cosine_similarity_mean_frame": float(frame_cosine.mean()),
        "cosine_similarity_global": float(global_cosine.item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--episodes", type=int, default=1, help="number of complete G1 episodes")
    parser.add_argument("--max-anchors", type=int, default=0, help="0 means every 8-frame anchor")
    parser.add_argument("--vlm-batch", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.episodes <= 0 or args.max_anchors < 0 or args.vlm_batch <= 0:
        parser.error("episodes/vlm-batch must be positive and max-anchors non-negative")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this regression test")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")
    args.output.mkdir()

    torch.set_num_threads(2)
    wrapper = PolicyServerWrapper(
        str(args.checkpoint),
        device=args.device,
        use_bf16=True,
        unnorm_key="agibot-g1",
    )
    framework = wrapper._framework
    head = framework.action_model
    original_flow = head.rollflows["agibot-g1"]
    cfg = original_flow.cfg
    if (cfg.horizon, cfg.action_dim) != (32, 22):
        raise ValueError(f"checkpoint G1 config must be [horizon=32, action_dim=22], got {cfg}")

    dataset = make_LeRobotSingleDataset(
        args.data_root,
        "agibot-g1-apple-corrected",
        "agibot-g1-apple",
        data_cfg={"include_state": True, "video_backend": "pyav", "action_mode": "abs"},
    )
    episode_ids = [int(value) for value in dataset.trajectory_ids[: args.episodes]]
    all_samples: list[dict] = []
    episode_anchors = {}
    for episode in episode_ids:
        samples, anchors = _episode_samples(dataset, episode, args.max_anchors)
        all_samples.extend(samples)
        episode_anchors[str(episode)] = anchors.tolist()
    del dataset

    # Encode the VLM once per observation.  The two action decoders below then
    # see bit-identical context/state inputs and differ only in rolling layout.
    contexts: list[tuple[torch.Tensor, torch.Tensor | None]] = []
    for start in range(0, len(all_samples), args.vlm_batch):
        hidden, mask = _encode_context(framework, all_samples[start : start + args.vlm_batch])
        contexts.extend(
            (
                hidden[index : index + 1].detach(),
                None if mask is None else mask[index : index + 1].detach(),
            )
            for index in range(hidden.shape[0])
        )
        print(f"encoded {min(start + args.vlm_batch, len(all_samples))}/{len(all_samples)} observations", flush=True)

    variants = {
        "chunk32_depth1": (32, 1),
        "chunk16_depth2": (16, 2),
        "chunk8_depth4": (8, 4),
        "chunk4_depth8": (4, 8),
    }
    report = {
        "checkpoint": str(args.checkpoint),
        "data": "agibot-g1-apple-corrected",
        "episodes": episode_ids,
        "episode_anchors": episode_anchors,
        "metric_space": "normalized 22-D G1 action space",
        "note": "offline training-observation regression; not closed-loop success",
        "results": {},
    }
    all_targets = {name: [] for name in variants}
    all_predictions = {name: [] for name in variants}
    for name, (chunk, depth) in variants.items():
        flow = RollFlow(replace(cfg, chunk_size=chunk, inference_steps=depth))
        head.rollflows["agibot-g1"] = flow
        predictions, targets = [], []
        for index, (sample, (context, mask)) in enumerate(zip(all_samples, contexts, strict=True)):
            flow.reset()
            torch.manual_seed(args.seed + index)
            prediction = head.predict_action(
                context.to(head.device),
                torch.from_numpy(sample["state"]).to(head.device),
                encoder_attention_mask=None if mask is None else mask.to(head.device),
                embodiment="agibot-g1",
            )
            pred = prediction[0].float().cpu().numpy()
            target = sample["action"][:chunk]
            if pred.shape != target.shape:
                raise ValueError(f"{name} shape mismatch: {pred.shape} vs {target.shape}")
            predictions.append(pred)
            targets.append(target)
        all_predictions[name] = predictions
        all_targets[name] = targets
        report["results"][name] = {
            "chunk_size": chunk,
            "refinement_steps": depth,
            **_score(predictions, targets),
        }
        print(name, json.dumps(report["results"][name], sort_keys=True), flush=True)
    head.rollflows["agibot-g1"] = original_flow

    np.savez_compressed(
        args.output / "predictions.npz",
        **{f"{name}_prediction": np.concatenate(values) for name, values in all_predictions.items()},
        **{f"{name}_target": np.concatenate(values) for name, values in all_targets.items()},
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("DONE", args.output, flush=True)


if __name__ == "__main__":
    main()
