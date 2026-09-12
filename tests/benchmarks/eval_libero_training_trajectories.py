#!/usr/bin/env python3
"""Offline regression of a pure-LIBERO RollFlow checkpoint.

The script evaluates the first ``N`` episodes of each LIBERO suite on the
same 32-step training windows and compares four rolling layouts:
``(chunk, depth) = (32, 1), (16, 2), (8, 4), (4, 8)``.  Metrics are reported in
the checkpoint's normalized 7-D Franka action space; this is not simulator
success evaluation.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
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
    "libero_qwengroot_rollflow_h32_c8_b128_unfrozen_463a1f7/"
    "checkpoints/steps_10000_pytorch_model.pt"
)
DEFAULT_DATA_ROOT = Path("/data/tzq/datasets/starVLA/Datasets")
DEFAULT_OUTPUT = Path(
    "/data/tzq/starVLA_checkpoints/"
    "libero_qwengroot_rollflow_h32_c8_b128_unfrozen_463a1f7/"
    "results/training_regression_10000_1_2_4_8"
)
SUITES = (
    "libero_object_no_noops_1.0.0_lerobot",
    "libero_goal_no_noops_1.0.0_lerobot",
    "libero_spatial_no_noops_1.0.0_lerobot",
    "libero_10_no_noops_1.0.0_lerobot",
)


def _samples(dataset, episode: int, max_anchors: int) -> tuple[list[dict], list[int]]:
    positions = np.flatnonzero(dataset.trajectory_ids == episode)
    if len(positions) != 1:
        raise ValueError(f"episode {episode} was not found exactly once")
    length = int(dataset.trajectory_lengths[positions[0]])
    anchors = np.arange(0, max(0, length - 31), 8, dtype=np.int64)
    if max_anchors > 0:
        anchors = anchors[:max_anchors]
    if len(anchors) == 0:
        raise ValueError(f"episode {episode} has fewer than 32 frames")

    out = []
    for anchor in anchors.tolist():
        raw = dataset.get_step_data(int(episode), int(anchor))
        sample = dataset._pack_sample(dataset.transforms(raw))
        action = np.asarray(sample["action"], dtype=np.float32)
        if action.shape != (32, 7) or not np.isfinite(action).all():
            raise ValueError(f"invalid LIBERO action window at episode={episode}, anchor={anchor}")
        if len(sample["image"]) != 2:
            raise ValueError(f"LIBERO expects 2 cameras, got {len(sample['image'])}")
        out.append({"image": sample["image"], "lang": sample["lang"], "action": action})
    return out, anchors.tolist()


@torch.inference_mode()
def _encode(framework, samples: list[dict]):
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
    mask = inputs.get("attention_mask")
    return outputs.hidden_states[-1].detach(), None if mask is None else mask.bool()


def _score(predictions: list[np.ndarray], targets: list[np.ndarray]) -> dict[str, float | int]:
    pred = np.concatenate(predictions).astype(np.float32, copy=False)
    truth = np.concatenate(targets).astype(np.float32, copy=False)
    error = pred - truth
    p, t = torch.from_numpy(pred), torch.from_numpy(truth)
    frame_cos = F.cosine_similarity(p, t, dim=-1, eps=1e-8)
    global_cos = F.cosine_similarity(p.reshape(1, -1), t.reshape(1, -1), dim=-1)
    return {
        "frames": int(len(pred)),
        "normalized_mse": float(np.mean(error * error)),
        "normalized_rmse": float(np.sqrt(np.mean(error * error))),
        "cosine_similarity_mean_frame": float(frame_cos.mean()),
        "cosine_similarity_global": float(global_cos.item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-anchors", type=int, default=0)
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
    wrapper = PolicyServerWrapper(str(args.checkpoint), device=args.device, use_bf16=True, unnorm_key="franka")
    framework = wrapper._framework
    head = framework.action_model
    original_flow = head.rollflows[head.default_embodiment]
    cfg = original_flow.cfg
    if (cfg.horizon, cfg.action_dim) != (32, 7):
        raise ValueError(f"checkpoint must expose a 32x7 Franka action head, got {cfg}")

    all_samples, episode_anchors = [], {}
    for suite in SUITES:
        dataset = make_LeRobotSingleDataset(
            args.data_root,
            f"libero/{suite}",
            "libero_franka_rollflow",
            data_cfg={"include_state": False, "video_backend": "pyav", "action_mode": "abs"},
        )
        for episode in [int(x) for x in dataset.trajectory_ids[: args.episodes]]:
            samples, anchors = _samples(dataset, episode, args.max_anchors)
            all_samples.extend({**sample, "suite": suite, "episode": episode} for sample in samples)
            episode_anchors[f"{suite}/episode{episode}"] = anchors
        del dataset

    contexts = []
    for start in range(0, len(all_samples), args.vlm_batch):
        hidden, mask = _encode(framework, all_samples[start : start + args.vlm_batch])
        contexts.extend(
            (hidden[i : i + 1].detach(), None if mask is None else mask[i : i + 1].detach())
            for i in range(hidden.shape[0])
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
        "data_root": str(args.data_root),
        "suites": list(SUITES),
        "episodes_per_suite": args.episodes,
        "episode_anchors": episode_anchors,
        "metric_space": "normalized 7-D Franka action space",
        "note": "offline training-observation regression; not closed-loop success",
        "results": {},
    }
    predictions_by_variant, targets_by_variant = {}, {}
    for name, (chunk, depth) in variants.items():
        flow = RollFlow(replace(cfg, chunk_size=chunk, inference_steps=depth))
        head.rollflows[head.default_embodiment] = flow
        predictions, targets = [], []
        for index, (sample, (context, mask)) in enumerate(zip(all_samples, contexts, strict=True)):
            flow.reset()
            torch.manual_seed(args.seed + index)
            pred = head.predict_action(
                context.to(head.device),
                state=None,
                encoder_attention_mask=None if mask is None else mask.to(head.device),
                embodiment=head.default_embodiment,
            )[0].float().cpu().numpy()
            target = sample["action"][:chunk]
            if pred.shape != target.shape:
                raise ValueError(f"{name} shape mismatch: {pred.shape} vs {target.shape}")
            predictions.append(pred)
            targets.append(target)
        predictions_by_variant[name] = np.concatenate(predictions)
        targets_by_variant[name] = np.concatenate(targets)
        report["results"][name] = {"chunk_size": chunk, "refinement_steps": depth, **_score(predictions, targets)}
        print(name, json.dumps(report["results"][name], sort_keys=True), flush=True)
    head.rollflows[head.default_embodiment] = original_flow

    np.savez_compressed(
        args.output / "predictions.npz",
        **{f"{name}_prediction": value for name, value in predictions_by_variant.items()},
        **{f"{name}_target": value for name, value in targets_by_variant.items()},
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("DONE", args.output, flush=True)


if __name__ == "__main__":
    main()
