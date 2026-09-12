#!/usr/bin/env python3
"""Fit one Apple trajectory while checking LIBERO retention.

This is an in-memory smoke test, not a production checkpoint.  It loads the
clean 55K action head into the multi-embodiment template, freezes the VLM and
shared DiT, and trains both native IO pairs.  A deterministic before/after
loss report catches data-contract errors without spending a full run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from deployment.model_server.policy_wrapper import PolicyServerWrapper
from starVLA.dataloader.lerobot_datasets import make_LeRobotSingleDataset


DEFAULT_ROOT = Path("/data/tzq/datasets/starVLA/Datasets")
DEFAULT_TEMPLATE = Path(
    "/data/tzq/starVLA_checkpoints/libero_agibot_g1_b32_c100_preflight/"
    "checkpoints/steps_2_pytorch_model.pt"
)
DEFAULT_INIT = Path(
    "/data/tzq/starVLA_checkpoints/libero_qwengroot_rollflow_h32_c8_b128_unfrozen_463a1f7/"
    "checkpoints/steps_55000_pytorch_model.pt"
)


def _sample_windows(
    dataset, episode: int, count: int, *, require_g1_state: bool = False
) -> list[dict]:
    length = int(dataset.trajectory_lengths[np.where(dataset.trajectory_ids == episode)[0][0]])
    anchors = list(range(0, max(1, length - 31), 8))[:count]
    if not anchors:
        raise ValueError(f"episode {episode} has fewer than 32 frames")
    samples = []
    for anchor in anchors:
        raw = dataset.get_step_data(int(episode), anchor)
        sample = dataset._pack_sample(dataset.transforms(raw))
        action = np.asarray(sample["action"], dtype=np.float32)
        if action.shape != (32, action.shape[-1]) or not np.isfinite(action).all():
            raise ValueError(f"invalid action window at episode={episode}, anchor={anchor}")
        state = sample.get("state") if require_g1_state else None
        if require_g1_state and state is not None:
            state = np.asarray(state, dtype=np.float32)
            if state.shape[-1] != 22 or not np.isfinite(state).all():
                raise ValueError(f"invalid state at episode={episode}, anchor={anchor}")
        samples.append({"image": sample["image"], "lang": sample["lang"], "action": action, "state": state})
    return samples


def _encode_context(framework, samples: list[dict]) -> tuple[torch.Tensor, torch.Tensor | None]:
    interface = framework.qwen_vl_interface
    inputs = interface.build_qwenvl_inputs(
        images=[sample["image"] for sample in samples],
        instructions=[sample["lang"] for sample in samples],
    )
    with torch.inference_mode():
        outputs = interface(**inputs, output_hidden_states=True, return_dict=True)
    hidden = outputs.hidden_states[-1].detach().float()
    mask = inputs.get("attention_mask")
    return hidden, None if mask is None else mask.bool()


def _batch(framework, samples: list[dict]) -> dict[str, torch.Tensor | None]:
    hidden, mask = _encode_context(framework, samples)
    device = hidden.device
    return {
        "context": hidden,
        "mask": None if mask is None else mask.to(device),
        "actions": torch.from_numpy(np.stack([s["action"] for s in samples])).to(device),
        "state": (
            None
            if samples[0]["state"] is None
            else torch.from_numpy(np.stack([s["state"] for s in samples])).to(device)
        ),
    }


def _loss(head, batch: dict[str, torch.Tensor | None], tag: str) -> torch.Tensor:
    return head(
        batch["context"],
        batch["actions"],
        batch["state"],
        encoder_attention_mask=batch["mask"],
        embodiment=tag,
        training_step=1000,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--windows", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3, help="G1 adapter learning rate")
    parser.add_argument(
        "--decode-steps",
        type=int,
        default=4,
        help="Rolling refinement depth used for the direct action-MSE check",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--init", dest="init_checkpoint", type=Path, default=DEFAULT_INIT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.steps <= 0 or args.windows <= 0 or args.lr <= 0 or args.decode_steps <= 0:
        parser.error("steps, windows, lr, and decode-steps must be positive")
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        parser.error("CUDA is required for this smoke test")

    wrapper = PolicyServerWrapper(str(args.template), device=args.device, use_bf16=True, unnorm_key="franka")
    framework = wrapper._framework
    head = framework.action_model.float()
    checkpoint = torch.load(args.init_checkpoint, map_location="cpu", weights_only=True, mmap=True)
    head.load_state_dict(
        {key.removeprefix("action_model."): value for key, value in checkpoint.items() if key.startswith("action_model.")},
        strict=False,
    )
    del checkpoint

    data_cfg = {"include_state": True, "video_backend": "pyav", "lerobot_version": "v2.0", "action_mode": "abs"}
    libero = make_LeRobotSingleDataset(
        args.data_root,
        "libero/libero_spatial_no_noops_1.0.0_lerobot",
        "libero_franka_rollflow",
        data_cfg=data_cfg,
    )
    apple = make_LeRobotSingleDataset(
        args.data_root,
        "agibot-g1-apple-corrected",
        "agibot-g1-apple",
        data_cfg=data_cfg,
    )
    libero_samples = _sample_windows(libero, int(libero.trajectory_ids[0]), args.windows)
    apple_samples = _sample_windows(
        apple, int(apple.trajectory_ids[0]), args.windows, require_g1_state=True
    )
    batches = {
        "franka": _batch(framework, libero_samples),
        "agibot-g1": _batch(framework, apple_samples),
    }

    # Train both native IO pairs, but keep the shared representation unchanged.
    head.requires_grad_(False)
    for module in (head.action_encoder, head.action_decoder, head.adapters):
        module.requires_grad_(True)
    default_io = list(head.action_encoder.parameters()) + list(head.action_decoder.parameters())
    adapter_params = list(head.adapters.parameters())
    trainable = default_io + adapter_params
    if not trainable:
        raise RuntimeError("no trainable action IO parameters")
    # Keep LIBERO adaptable, but use a conservative rate for its pretrained
    # IO pair while the newly initialized G1 adapters learn faster.
    optimizer = torch.optim.AdamW(
        [
            {"params": default_io, "lr": args.lr * 0.1},
            {"params": adapter_params, "lr": args.lr},
        ]
    )

    def evaluate() -> dict[str, float]:
        head.eval()
        values = {}
        with torch.no_grad():
            for tag, batch in batches.items():
                torch.manual_seed(1234)
                values[tag] = float(_loss(head, batch, tag))
        return values

    before = evaluate()
    head.train()
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        torch.manual_seed(1234 + step)
        total = 0.5 * (_loss(head, batches["franka"], "franka") + _loss(head, batches["agibot-g1"], "agibot-g1"))
        if not torch.isfinite(total):
            raise FloatingPointError(f"non-finite loss at step {step}")
        total.backward()
        if any(parameter.grad is not None for parameter in head.model.parameters()):
            raise AssertionError("shared DiT received gradients in adapter-isolation test")
        if not any(parameter.grad is not None for parameter in default_io):
            raise AssertionError("LIBERO encoder/decoder did not receive gradients")
        if not any(parameter.grad is not None for parameter in adapter_params):
            raise AssertionError("G1 adapters did not receive gradients")
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        if (step + 1) % max(1, args.steps // 5) == 0:
            print(f"step={step + 1} loss={float(total.detach()):.6f}", flush=True)

    after = evaluate()

    def decoded_action_mse() -> dict[str, float]:
        """Measure normalized MSE of one deterministic rolling action chunk."""
        head.eval()
        values = {}
        with torch.no_grad():
            for tag, batch in batches.items():
                # Each tag is an independent episode; never carry its rolling
                # cache into the next decode or between repeated measurements.
                head.reset()
                torch.manual_seed(4321)
                prediction = head.predict_action(
                    batch["context"],
                    batch["state"],
                    encoder_attention_mask=batch["mask"],
                    refinement_steps=args.decode_steps,
                    embodiment=tag,
                )
                target = batch["actions"][:, : prediction.shape[1]]
                if prediction.shape != target.shape:
                    raise AssertionError(
                        f"decoded shape {tuple(prediction.shape)} != target {tuple(target.shape)}"
                    )
                values[tag] = float((prediction.float() - target.float()).square().mean())
            head.reset()
        return values

    decoded_mse = decoded_action_mse()
    result = {
        "steps": args.steps,
        "windows": args.windows,
        "decode_steps": args.decode_steps,
        "before": before,
        "after": after,
        "decoded_action_mse": decoded_mse,
        "delta": {tag: after[tag] - before[tag] for tag in before},
        "shared_dit_frozen": True,
        "data": "agibot-g1-apple-corrected",
    }
    print(json.dumps(result, indent=2), flush=True)
    if after["agibot-g1"] >= before["agibot-g1"]:
        raise RuntimeError("G1 single-trajectory loss did not decrease")
    if after["franka"] > before["franka"] * 1.5:
        raise RuntimeError("LIBERO loss degraded beyond the smoke-test tolerance")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
