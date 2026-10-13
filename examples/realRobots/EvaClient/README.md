# EVA Client bridge for StarVLA RollFlow

This is an additive integration package. It leaves `EVA-CLIENT/src`, the StarVLA
framework and model code unchanged. The EVA app imports the plugins in
`eva_extension/` from a launcher, so model and robot adapters stay outside the
official projects.

## Deployment version

See the [Chinese deployment and migration guide](../../../docs/eva_deployment.md)
for a fresh checkout, live G2 configuration and the execution-side ZIP exporter.
The supplied EVA source is in the repository root at `EVA-CLIENT/`;
[its provenance](eva_source.json) records the original archive hash.
Validation artifacts and videos are bundled with this package.

Run the examples from the repository root. Set the following paths for your machine:

```bash
export STARVLA_MODEL_PYTHON=/path/to/starvla-env/bin/python
export EVA_EVAL_ENV=/path/to/eva-eval-env
export LIBERO_CHECKOUT=/path/to/LIBERO
export G2_CHECKPOINT=/path/to/G2_MODEL_RUN/final_model/pytorch_model.pt
export LIBERO_CHECKPOINT=/path/to/LIBERO_MODEL_RUN/checkpoints/steps_55000_pytorch_model.pt
export STARVLA_G2_DATASET=/path/to/white_carton_to_left_tan_carton_20261009
export STARVLA_HOST=127.0.0.1
```

The checkpoint's run directory must also contain its `config.yaml` and
`dataset_statistics.json`. Relocate the base Qwen path in the saved config when
moving to a new machine. G2 presets default to port 10094, LIBERO to 10093;
`STARVLA_HOST` and `STARVLA_PORT` can override them. Live G2 additionally accepts
`G2_OBS_ENDPOINT` and `G2_ACTION_ENDPOINT` for its ZMQ connection addresses.

## Intended architecture

The bridge contract in `contracts/*.json` declares ordered camera inputs, image
size/rotation, optional raw-state slices and their training keys, the action
dimension, EVA action-processing mode and action semantics, rolling chunk size,
normalization key, model horizon, and inference-step count. The same file is loaded by the StarVLA endpoint and
the EVA client. Their handshake hashes must match before inference starts.
Run the endpoint in the StarVLA model environment and the replay/simulator in
the EVA environment; the two processes communicate over WebSocket. EVA's
`pyproject.toml` currently requires Python 3.11, while this checkout's StarVLA
environment is Python 3.10, so they should not be forced into one interpreter.

The endpoint loads checkpoints through StarVLA's stock `PolicyServerWrapper`,
so the checkpoint selects its trained framework/action head and saved action
normalization. The EVA plugin returns the unnormalized chunk to EVA's existing
action pipeline. `reset` clears model-side rolling state at episode boundaries.
Robot adapters and action mappings remain explicit per embodiment: a model with
different action semantics needs its own contract and mapping.

The two checks exercise different parts of the pipeline:

1. **G2 held-out replay** reads the nine held-out LeRobot episodes using EVA's
   `DatasetTransport`, `build_policy_observation`, `PolicyClient`, and
   `fetch_action_chunk`. It compares the processed prediction chunks to recorded
   actions and never calls `publish_action`.
2. **LIBERO closed loop** registers `LiberoVirtualRobot` and `LiberoSimTransport`
   into EVA. EVA's `fetch_action_chunk` normalizes/gripper-processes every chunk;
   `publish_action` steps the simulator and the next chunk sees the resulting
   observation. This measures task success and inference timing as well as
   checking message/image/action wiring.

These checks validate policy/server/EVA and simulator wiring. G2 recorded replay
does not test robot feedback dynamics or safety, and LIBERO does not validate a
physical G2 actuator mapping.

Both evaluators use `StarVlaRollingStrategy`: execute the entire model chunk
before requesting another. It prevents EVA's default five-step cropping and
skips discarded policy warmup chunks, which would advance a rolling cache before
the robot executes any actions. One policy server accepts one robot connection;
connections and episode resets clear its cache.

## Separate evaluation environment

Use the training Python for the model server and a separate Python 3.11
environment for EVA and LIBERO. The helper installs CPU MuJoCo rendering and
does not change the training environment:

```bash
bash examples/realRobots/EvaClient/setup_eval_environment.sh \
  "$EVA_EVAL_ENV" "$LIBERO_CHECKOUT"
```

Use a **standard LIBERO checkout** with assets and initial states. Set
`PYTHONPATH` and `LIBERO_CONFIG_PATH` explicitly when running the simulator;
other experiments on this machine use a different LIBERO checkout. The setup helper creates `ENV_DIR/libero-config` for a new environment.
Use that isolated config directory for the simulator.

## Existing G2 contract

- EVA G2 cameras arrive as `top_head`, `hand_left`, `hand_right`, in BGR/HWC
  `uint8` format. Dataset replay uses the same BGR/HWC representation.
- The training camera order is `video.top_head`, `video.hand_left`,
  `video.hand_right`. The adapter converts BGR to RGB once and calls the same
  `PIL.Image.fromarray(rgb).resize((224, 224))` operation as the training dataset
  (Pillow's default bicubic interpolation). The additive `starvla_dataset` and
  `starvla_zmq` transports prepare these pixels before EVA's observation builder.
- State and action are absolute qpos in the 24-joint order from the G2 dataset
  metadata. Raw state crosses the wire; the policy server uses the checkpoint's
  registered G2 transform. Predicted actions are unnormalized by the stock
  `PolicyServerWrapper` and returned as qpos.
- The checkpoint contract is H=64, 4 refinement steps, and a 16-action chunk.
  The rolling cache is reset once per episode and retained across chunk requests.
- `EvaPolicyClient.infer()` only requests a chunk. It never publishes actions.

## Run the contract-aware StarVLA endpoint

The server loads a checkpoint and requires the model runtime. Start it only when
the selected GPU is available. G2:

```bash
"$STARVLA_MODEL_PYTHON" -m examples.realRobots.EvaClient.contract_server \
  --checkpoint "$G2_CHECKPOINT" \
  --contract examples/realRobots/EvaClient/contracts/agibot_g2_rollflow_h64_c16.json \
  --device cuda:0 --port 10094 --gpu-memory-gib 6
```

For LIBERO, use the **55K H32/C8 checkpoint with saved successful evaluations**.
Its Goal suite result is 448/500 (89.6%); task 0 (open middle drawer) and task 1
(bowl to stove) each succeeded in 49/50 trials. Object suite is 451/500 (90.2%).
Source logs are in this training run's `results/<suite>/steps_55000_pytorch_model.pt/eval.log`.
The checkpoint uses four inference iterations and the `franka` normalization key:

```bash
"$STARVLA_MODEL_PYTHON" -m examples.realRobots.EvaClient.contract_server \
  --checkpoint "$LIBERO_CHECKPOINT" \
  --contract examples/realRobots/EvaClient/contracts/libero_rollflow_h32_c8_k4.json \
  --device cuda:0 --port 10093 --gpu-memory-gib 6
```

The endpoint accepts standard `infer` requests plus `reset`. It reports the
contract hash and model metadata at connection time. Checkpoint horizon, action
chunk size, inference steps, action keys and action dimensions are checked at
startup. `--gpu-memory-gib` limits only this server's allocator; choose the GPU
with `CUDA_VISIBLE_DEVICES`. Previously active training processes are untouched.

The earlier H64/C8/8-iteration contract remains available as
`contracts/libero_rollflow_h64_c8.json` with `configs/libero_sim.py`. That
checkpoint has no established high-success evaluation here and failed the
initial two-episode smoke check. It must not be used as the quality reference
for diagnosing EVA. The additive data registry contains its missing mixture
alias for reproducing that comparison.

## G2 held-out data through EVA

With the endpoint running and GPU resources available:

```bash
PYTHONPATH= "$EVA_EVAL_ENV/bin/python" \
  -m examples.realRobots.EvaClient.evaluate_dataset_via_eva \
  --config examples/realRobots/EvaClient/configs/g2_replay.py \
  --dataset "$STARVLA_G2_DATASET" \
  --episodes 0,13,27,40,54,67,81,94,108 \
  --output work_dirs/eva_validation/agibot_g2_eva_rollflow_replay.json
```

This requires the StarVLA contract server to be running. The G2 config uses EVA's
own LeRobot `DatasetTransport` via the additive image subclass; the evaluator
seeks to each 16-action boundary,
resets the policy once per episode and compares the result after EVA action
normalization/gripper handling. It does not publish predicted actions.

## LIBERO closed-loop evaluation through EVA

Run with the combined EVA/LIBERO runtime dependencies installed and the LIBERO
contract server running:

```bash
PYTHONPATH="$LIBERO_CHECKOUT" \
LIBERO_CONFIG_PATH="$EVA_EVAL_ENV/libero-config" \
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
"$EVA_EVAL_ENV/bin/python" \
  -m examples.realRobots.EvaClient.evaluate_libero_via_eva \
  --config examples/realRobots/EvaClient/configs/libero_sim_verified.py \
  --suite libero_goal --task-ids 0,1 --episodes-per-task 2 \
  --policy-seed 7 \
  --output work_dirs/eva_validation/libero_eva_verified_55k.json \
  --video-dir work_dirs/eva_validation/libero_eva_verified_55k_videos
```

This is closed loop: the simulator advances after each action in an 8-step
chunk, then EVA requests the next chunk from the new observation. The policy
plugin rotates both camera frames 180 degrees, resizes directly to 224x224 and
preserves the `agentview`, `wrist` camera order used by the LIBERO evaluator.
The 7D output is a LIBERO Cartesian delta. In this contract, `mode: joint`
means EVA should pass the vector through without robot IK; it does not mean the
values are joint angles. The simulator transport applies the chunk directly to
LIBERO. The simulator uses its suite's standard step limit unless
`transport.libero.max_steps` is set explicitly.
The selected checkpoint is H32/C8 with four inference iterations; the adapter
validates its saved values. Videos show the main and wrist cameras side by side
at 20 simulation frames/second. JSON records the checkpoint, contract, task,
initial-state index, success, action count, timing and video path; NPZ preserves
the executed actions. Short smoke checks establish that the pipeline runs and
do not replace the historical 500-trial evaluations.

`verify_libero_reference.py` records a seeded EVA rollout, then replays its
entire observation sequence through the unchanged LIBERO `ModelClient.step`.
Input pixels, raw chunks and processed gripper commands must match after
resetting the model cache and RNG. It uses the same model server, so this checks
the EVA adapter and rolling scheduling; it does not independently reproduce a
past StarVLA server revision. Run it after the rollout client disconnects:

```bash
PYTHONPATH="$LIBERO_CHECKOUT" \
LIBERO_CONFIG_PATH="$EVA_EVAL_ENV/libero-config" \
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
"$EVA_EVAL_ENV/bin/python" \
  -m examples.realRobots.EvaClient.verify_libero_reference \
  --config examples/realRobots/EvaClient/configs/libero_sim_verified.py \
  --model-python "$STARVLA_MODEL_PYTHON" --host "$STARVLA_HOST" \
  --chunks 38 --task-id 1 --init-state-id 1 \
  --output work_dirs/eva_validation/eva_libero_reference_rollout_parity.json
```

The checkpoint provenance, G2 replay metrics, LIBERO results and videos from
2026-10-13 are collected in [validation_report_20261013.md](validation_report_20261013.md).

To run EVA's UI against the simulator or a live G2 transport, use the additive
launcher (it adds registrations before starting the unchanged EVA runtime):

```bash
python -m examples.realRobots.EvaClient.run_eva \
  --config examples/realRobots/EvaClient/configs/libero_sim_verified.py
```

For G2, use `configs/agibot_g2_live.py` only after the policy endpoint and robot
transport are configured for the target network. The G2 dataset replay config
is read-only and safe for offline evaluation.

## Existing typed G2 replay helper

The earlier G2-specific `server.py`, `client.py` and `evaluate_replay.py` remain
available for comparison with the generic contract endpoint.

Before live execution, run replay/simulator checks, confirm that the contract,
camera calibration/order, state/action mapping and server handshake match the
target robot, and verify telemetry in the EVA app. No action is published by the
G2 replay evaluator. LIBERO's 7D Cartesian deltas must only be connected to a
matching action space; they are not robot joint angles.
