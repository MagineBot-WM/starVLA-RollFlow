# AgiBot G2 pure rolling-flow

Deployment through EVA is documented in [the deployment guide](../../../docs/eva_deployment.md).
The paths below record the original training machine. For a new training run,
set `run_root_dir`, `framework.qwenvl.base_vlm` and
`datasets.vla_data.data_root_dir` in a copy of the YAML for that machine.
The runner accepts `STARVLA_MODEL_PYTHON`, `STARVLA_G2_TRAIN_CONFIG`,
`STARVLA_RUN_ROOT`, `STARVLA_LOG_ROOT` and `STARVLA_G2_RUN_ID`; its DeepSpeed
configuration is supplied alongside the recipe. It refuses to overwrite a run.

The extracted LeRobot dataset lives at:

`/data/tzq/datasets/starVLA/Datasets/AgiBotG2/white_carton_to_left_tan_carton_20261009`

It contains 109 episodes of the task “Pick up the white carton and place it
inside the large tan carton on the left.” The data contract maps the source's
24-D state/action vectors without reordering: left arm, left gripper, right
arm, right gripper, head, and waist. Cameras are ordered top head, left hand,
right hand. The source zip is retained unchanged; `meta/modality.json` and
`meta/g2_dataset_manifest.json` are generated for StarVLA.

Nine episodes, spread evenly through the episode order, are held out for
validation; the training split contains the other 100 episodes. The current
trainer's periodic `mse_score` is computed on a batch from its training
iterator, so it is not the held-out validation score.

The G2 data transform treats q99 spans of at most 0.001 rad as fixed channels.
This prevents tiny float32 jitter in stationary joints from being expanded to
the full normalized range. The dataset attaches `action_padding_mask` so
actions padded beyond an episode boundary do not contribute to flow loss.

The training config is scratch pure rolling-flow, H=64, emit 16 actions with
4 inference steps, VLM frozen, and 4 examples per GPU with gradient accumulation
over 4 microbatches. With four GPUs, raw global batch size remains 64. The
framework expands each raw example into four independent flow samples
internally, so the action head sees 16 flow samples per GPU microbatch. The runner starts
when all four GPUs have at least 14,000 MiB free for three checks; it does not
require GPU utilization to reach zero and does not stop other jobs.

Prepare and CPU-check the dataset:

```bash
/data/miniconda3/envs/starVLA/bin/python \
  examples/realRobots/AgiBotG2/train_files/prepare_agibot_g2_dataset.py
```

Start the guarded training job when the GPUs are available:

```bash
bash examples/realRobots/AgiBotG2/train_files/run_agibot_g2_rollingflow.sh
```
