# AgiBot G1 — RollFlow

This example integrates the public AgiBotWorld G1 tasks and the local real-G1
pick/place tasks under one explicit fixed-base control contract.

## Canonical contract

- Images, in order: `head`, `hand_left`, `hand_right`.
- State (20D): `arms14 + waist2 + head2 + measured_grippers2`.
- Action (20D): `arms14 + waist2 + head2 + normalized_opening2`.
- Gripper action is continuous: `0=closed`, `1=open`. It is not a binary label.
- The public dataset's final 2D mobile-base velocity is intentionally excluded.
- The two physical gripper variants keep separate millimetre calibrations at deployment.

The raw datasets differ in vector order and camera names. Their `modality.json`
files perform semantic selection/reordering before concatenation, so raw flat
vectors must never be concatenated directly.

## Prepare and audit

The preparation step creates G2-style semantic paths under
`g1/manipulation/<task_name>`. Parquet and videos remain read-only and are
symlinked; the old `g1/task_NNN` entries remain available for compatibility.

```bash
/data/miniconda3/envs/starVLA/bin/python \
  examples/realRobots/AgiBotG1/dataset_tools/prepare_overlays.py

/data/miniconda3/envs/starVLA/bin/python \
  examples/realRobots/AgiBotG1/dataset_tools/audit_datasets.py \
  --output /data/tzq/datasets/starVLA/Datasets/AgiBot-G1-G2-StarVLA/g1/audit_agibot_g1.json
```

## Train

The default launch is four GPUs × 32 samples/GPU = global batch 128. It freezes
the VLM and trains the 32-step RollFlow head, executing 8 actions per request.

```bash
bash examples/realRobots/AgiBotG1/train_files/run_agibot_g1_rollflow_train.sh
```

Run only local real data or only public data by overriding
`DATA_MIX=agibot_g1_real` or `DATA_MIX=agibot_g1_public`. A short end-to-end
training smoke test can be launched with:

```bash
DATA_MIX=agibot_g1_real MAX_TRAIN_STEPS=10 SAVE_INTERVAL=10 \
  RUN_ID=agibot_g1_rollflow_smoke \
  bash examples/realRobots/AgiBotG1/train_files/run_agibot_g1_rollflow_train.sh
```

The first dataset initialization may compute missing per-task statistics and can
therefore take noticeably longer than later launches.

## Test and deploy

```bash
bash examples/realRobots/AgiBotG1/eval_files/run_local_self_test.sh

CHECKPOINT=/path/to/checkpoint \
  bash examples/realRobots/AgiBotG1/eval_files/run_policy_server.sh
```

See `eval_files/inference_example.py` for the robot-side contract. Call reset
once at a new task/episode boundary; do not reset between consecutive chunks,
because that would discard RollFlow's rolling buffer.
