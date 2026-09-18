# LIBERO：RollFlow 训练与评估

本文档描述当前仓库推荐的 LIBERO 流程：Qwen3.5-0.8B VLM + GR00T RollFlow，四个 LIBERO 数据集混合训练，并通过 websocket policy server 在 LIBERO 仿真器中评估。

当前默认实验是 H128/C32：模型维护 128 个动作的 rolling cache，每次向环境输出 32 个动作；推理使用 4 步迭代。训练和评估脚本都位于本目录下，命令应从仓库根目录执行。

## 1. 环境

训练和评估使用两个 Python 环境：

- `starVLA` 环境：训练、加载 checkpoint、运行 policy server。
- `LIBERO` 环境：启动 MuJoCo 仿真和评估脚本。

先按照 [LIBERO 官方仓库](https://github.com/Lifelong-Robot-Learning/LIBERO) 安装仿真环境，然后在 LIBERO 环境中安装评估依赖：

~~~bash
python -m pip install mujoco==3.2.3
python -m pip install tyro matplotlib mediapy websockets msgpack
python -m pip install numpy==1.24.4
~~~

也可以使用安装脚本：

~~~bash
LIBERO_CONDA_ENV=libero \
LIBERO_PARENT_DIR=$HOME \
bash examples/simBenchmarks/LIBERO/eval_files/install_libero.sh
~~~

LIBERO 环境建议使用 Python 3.10。AMD GPU 用户按对应环境将视觉模型 attention 实现设为 `sdpa`。

## 2. 准备数据和模型

### 数据集

当前 H128 配置使用以下四个 LeRobot 数据集，各自权重为 1：

- `libero_spatial_no_noops_1.0.0_lerobot`
- `libero_object_no_noops_1.0.0_lerobot`
- `libero_goal_no_noops_1.0.0_lerobot`
- `libero_10_no_noops_1.0.0_lerobot`

可以用仓库脚本下载并准备数据：

~~~bash
export DEST=/path/to/dataset_root
bash examples/simBenchmarks/LIBERO/data_preparation.sh
~~~

脚本会下载数据、复制 `modality.json`，并创建 `playground/Datasets` 下的兼容软链接。若数据已经准备好，训练时把 `DATA_ROOT` 指向包含四个数据集目录的 `libero` 目录：

~~~text
/path/to/dataset_root/libero/
├── libero_spatial_no_noops_1.0.0_lerobot/
├── libero_object_no_noops_1.0.0_lerobot/
├── libero_goal_no_noops_1.0.0_lerobot/
└── libero_10_no_noops_1.0.0_lerobot/
~~~

### 预训练 VLM

配置文件 [starvla_qwengroot_rollflow_libero_h128_c32_80k.yaml](train_files/starvla_qwengroot_rollflow_libero_h128_c32_80k.yaml) 中的 `framework.qwenvl.base_vlm` 是本地模型路径。使用自己的模型时，复制该 YAML，修改 `base_vlm`、`data_root_dir` 等路径，并通过 `CONFIG_YAML` 指向副本。

## 3. H128/C32 RollFlow 配置

| 配置 | 当前值 | 含义 |
|---|---:|---|
| `action_horizon` | 128 | rolling cache 和训练动作窗口长度 |
| `execution_horizon` | 32 | 每次推理实际输出的动作数 |
| `inference_steps` | 4 | 默认推理迭代步数 |
| `repeated_diffusion_steps` | 4 | action head 的重复扩散步数 |
| `p_k1` | 0.7 | 训练时优先采样单时间组 |
| `fm_curriculum_steps` | 20,000 | 前 20K 步纯 Flow Matching |
| `p_fm` | 0.7 | curriculum 结束时的 FM 样本概率下限 |
| `w_fm` / `w_lsd` | 1.0 / 0.25 | FM 与 LSD 损失权重 |
| `velocity_mode` | `average` | 推理使用区间平均速度 |
| `iterative_cold_start` | `true` | 首次 cache 初始化时逐级迭代 |

`p_fm` 是采样概率，不是 LSD 损失权重：前 20K 步 `p_fm=1`，之后从 1 线性下降，在 `max_train_steps=80000` 时达到 0.7。

数据配置将动作窗口固定为 128。LIBERO 轨迹不足 128 步时，当前 `action_mode: abs` 使用末端动作重复填充尾部，因此模型仍然接收 `[128, 7]`，末端保持动作而不是强制置零。

## 4. 训练

### 首次启动

推荐使用带校验和 tmux worker 的 launcher：

~~~bash
cd /path/to/starVLA-RollFlow

STARVLA_PYTHON=/data/miniconda3/envs/starVLA/bin/python \
DATA_ROOT=/path/to/dataset_root/libero \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash examples/simBenchmarks/LIBERO/train_files/run_libero_rollflow_train.sh
~~~

launcher 默认值：

~~~text
NUM_GPUS=4              BATCH_PER_GPU=8
GRADIENT_ACCUMULATION_STEPS=1
MAX_TRAIN_STEPS=80000   SAVE_INTERVAL=5000
EVAL_INTERVAL=1000      LOGGING_FREQUENCY=20
RUN_ID=libero_qwengroot_rollflow_h128_c32_80k_unfrozen
~~~

默认每卡 batch 为 8 是 H128 的安全起点；有效原始 batch 为 `NUM_GPUS × BATCH_PER_GPU × GRADIENT_ACCUMULATION_STEPS`。如果显存余量足够，可显式设置 `BATCH_PER_GPU`：

~~~bash
BATCH_PER_GPU=4 NUM_GPUS=4 \
bash examples/simBenchmarks/LIBERO/train_files/run_libero_rollflow_train.sh
~~~

脚本会在启动前检查 Python、配置、数据目录、GPU 数量、端口和输出目录，不会覆盖已有 run。默认日志和 checkpoint：

~~~text
/data/tzq/starVLA_checkpoints/libero_qwengroot_rollflow_h128_c32_80k_unfrozen.train.log
/data/tzq/starVLA_checkpoints/libero_qwengroot_rollflow_h128_c32_80k_unfrozen/checkpoints/
~~~

查看任务：

~~~bash
tmux attach -t rollflow_libero_h128
tail -f /data/tzq/starVLA_checkpoints/libero_qwengroot_rollflow_h128_c32_80k_unfrozen.train.log
~~~

日志中的 `rollflow/fm_loss`、`rollflow/lsd_loss`、`rollflow/p_fm`、`rollflow/active_lsd_frac` 用于确认蒸馏 curriculum 是否按预期开启。

### 断点续训

`RESUME=1` 只允许在已有 run 目录且存在可加载 checkpoint 时启动：

~~~bash
RESUME=1 \
RUN_ID=libero_qwengroot_rollflow_h128_c32_80k_unfrozen \
DATA_ROOT=/path/to/dataset_root/libero \
bash examples/simBenchmarks/LIBERO/train_files/run_libero_rollflow_train.sh
~~~

如果要开始新的实验，使用新的 `RUN_ID`，不要复用旧目录。

## 5. LIBERO 评估

评估需要两个终端，并且 server 和 simulator 使用同一个 checkpoint。

### 终端一：启动 policy server

在 `starVLA` 环境中执行：

~~~bash
CKPT=/data/tzq/starVLA_checkpoints/libero_qwengroot_rollflow_h128_c32_80k_unfrozen/checkpoints/steps_80000_pytorch_model.pt \
STARVLA_PYTHON=/data/miniconda3/envs/starVLA/bin/python \
GPU_ID=0 \
PORT=6694 \
USE_BF16=1 \
bash examples/simBenchmarks/LIBERO/eval_files/run_policy_server.sh
~~~

server handshake 应显示 H128/C32 模型的 `action_chunk_size=32`。端口被占用时，两端同时换成新的 `PORT`。

历史 `StarVLA/Qwen3-VL-PI-LIBERO-4in1` checkpoint 需要兼容模式：

~~~bash
USE_CANONICAL_FORWARD=false \
CKPT=/path/to/steps_100000_pytorch_model.pt \
bash examples/simBenchmarks/LIBERO/eval_files/run_policy_server.sh
~~~

不要修改 checkpoint 内的 `config.yaml`，使用 launcher override。

### 终端二：运行仿真评估

在 LIBERO 环境中执行：

~~~bash
LIBERO_HOME=/path/to/LIBERO \
LIBERO_PYTHON=/path/to/libero/bin/python \
CKPT=/data/tzq/starVLA_checkpoints/libero_qwengroot_rollflow_h128_c32_80k_unfrozen/checkpoints/steps_80000_pytorch_model.pt \
HOST=127.0.0.1 \
PORT=6694 \
TASK_SUITE_NAME=libero_goal \
NUM_TRIALS_PER_TASK=50 \
MAX_TASKS=-1 \
bash examples/simBenchmarks/LIBERO/eval_files/eval_libero.sh
~~~

支持的 suite：`libero_spatial`、`libero_object`、`libero_goal`、`libero_10`、`libero_90`。完整评估时 `MAX_TASKS=-1`；smoke test 可限制任务数：

~~~bash
MAX_TASKS=1 NUM_TRIALS_PER_TASK=1 \
LIBERO_HOME=/path/to/LIBERO \
LIBERO_PYTHON=/path/to/libero/bin/python \
CKPT=/path/to/steps_5000_pytorch_model.pt \
bash examples/simBenchmarks/LIBERO/eval_files/eval_libero.sh
~~~

视频默认保存到 `results/<suite>/<checkpoint_name>/`，也可以显式设置 `VIDEO_OUT_PATH`。

多数据集 checkpoint 如果无法自动选择归一化统计，指定训练数据对应的 key：

~~~bash
UNNORM_KEY=libero_goal_no_noops_1.0.0_lerobot \
LIBERO_HOME=/path/to/LIBERO \
LIBERO_PYTHON=/path/to/libero/bin/python \
CKPT=/path/to/steps_80000_pytorch_model.pt \
bash examples/simBenchmarks/LIBERO/eval_files/eval_libero.sh
~~~

通常优先使用 server handshake 提供的可用 key；只有出现 key 选择错误时才显式设置 `UNNORM_KEY`。

## 6. 常见问题

- **`Checkpoint not found`**：`CKPT` 必须是具体的 `steps_*_pytorch_model.pt` 文件。
- **`DATA_ROOT` 不存在**：它应指向 `libero` 数据根目录，而不是某个单独数据集目录。
- **显存不足**：先把 `BATCH_PER_GPU` 降为 4 或 2；新实验使用新的 `RUN_ID`。
- **评估连接失败**：确认 server 已先启动，且两端 `HOST`/`PORT` 完全一致。
- **动作维度不匹配**：H128/C32 模型应返回 7 维 Franka 动作，server 的 `action_chunk_size` 应为 32。
- **MuJoCo 渲染失败**：LIBERO 终端设置 `MUJOCO_GL=egl` 和 `PYOPENGL_PLATFORM=egl`，脚本默认已设置。
- **前 20K 步没有 LSD**：这是设计行为；此时 `p_fm=1`、`active_lsd_frac=0` 正常。

相关入口：

- [H128/C32 训练配置](train_files/starvla_qwengroot_rollflow_libero_h128_c32_80k.yaml)
- [训练 launcher](train_files/run_libero_rollflow_train.sh)
- [policy server launcher](eval_files/run_policy_server.sh)
- [LIBERO 评估 launcher](eval_files/eval_libero.sh)
- [数据 registry](train_files/data_registry/data_config.py)
