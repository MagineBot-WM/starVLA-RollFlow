# EVA × StarVLA 管线验证记录（2026-10-13）

本目录同时提供可迁移的 [验证产物](validation/20261013/artifact_manifest.json) 与视频。
JSON/NPZ 为原始文件的逐字节副本，其中的绝对路径记录原测试机器。
结果产生于包含本地 MeanFlow 诊断改动的工作区；这些改动未收入本部署标签，
本次整理没有重新运行推理或完整 benchmark。G2 纯流实现独立于 MeanFlow。

## 结论

两条离线验证路径均已运行：LIBERO 经 EVA 与 StarVLA 完成仿真闭环任务；G2 的九条保留测试数据经 EVA 请求 StarVLA，完成滚动动作预测与数据对比。

| 测试 | 输入与执行路径 | 本次结果 |
| --- | --- | --- |
| LIBERO 闭环 | LIBERO 图像 → EVA → StarVLA → EVA 动作处理 → LIBERO 执行动作 → 下一次观测 | 两个任务各两次，成功 **3/4** |
| G2 数据回放 | G2 图像、状态 → EVA → StarVLA → EVA 动作处理 → 与记录动作对比 | 九条，2320 步，149 个 chunk；MSE **0.002755** |
| LIBERO 原生接口对比 | 同一完整观测序列、同一权重、缓存和随机种子重置后，比较 EVA 与原生 `ModelClient.step` | 38 个 chunk，图像、原始动作、处理后动作最大差异均为 **0** |

LIBERO 的成功执行证明此次仿真管线能够工作。G2 的测试不下发动作，尚未验证真机反馈与执行器映射。本次 4 次 LIBERO 试跑用于验证联通与执行，不用于重新估计模型的完整 benchmark 成功率。

## 1. 权重与历史评估依据

LIBERO 使用 [steps_55000_pytorch_model.pt](validation/20261013/eva_libero_checkpoint_selection_20261013.json)。这是包含自蒸馏的 `QwenGR00TRollFlow`，H32、执行 chunk=8、推理迭代=4，动作归一化键为 `franka`，本权重不输入状态。

| 历史 suite | 成功数 / 总数 | 成功率 | 原始记录 |
| --- | --- | --- | --- |
| Goal | 448/500 | 89.6% | [eval.log](validation/20261013/eva_libero_checkpoint_selection_20261013.json) |
| Object | 451/500 | 90.2% | [eval.log](validation/20261013/eva_libero_checkpoint_selection_20261013.json) |
| Spatial | 389/500 | 77.8% | [eval.log](validation/20261013/eva_libero_checkpoint_selection_20261013.json) |
| Long / LIBERO-10 | 338/500 | 67.6% | [eval.log](validation/20261013/eva_libero_checkpoint_selection_20261013.json) |

所选 Goal 任务“开中间抽屉”和“把碗放到炉台”，历史记录均为 49/50，即 98%。[权重选择及逐任务统计](validation/20261013/eva_libero_checkpoint_selection_20261013.json)保留了来源路径。

先前视频使用的是 `libero_rollflow_h64_c8_mf_from0_fm20k_vlm_unfrozen/final_model/pytorch_model.pt`，H64/C8/8 次迭代，训练记录结束于 20K。该权重没有找到已建立的高成功率评估，初次两个任务试跑均失败。[先前结果](validation/20261013/libero_eva_pipeline_final_20261013.json)仅作为对照；不能据此认定 EVA 接口有问题或模型没有学到。

## 2. LIBERO 经 EVA 的实际执行

配置：[libero_sim_verified.py](configs/libero_sim_verified.py)。环境使用标准 LIBERO、256×256 渲染、10 步环境稳定动作。模型每次收到两张 RGB 图像，旋转 180°、直接双线性缩放至 224×224，顺序为主视角、腕部视角。

每次请求得到 8×7 动作，EVA 完整执行该 chunk 后取新图像，模型保留滚动缓存；任务成功或达到 300 步时结束，下一回合重置缓存。本次每回合将模型随机种子重置为 7，初始状态为 0、1；历史评估的随机数消耗序列与此短测不保证相同。

| 任务 | 初始状态 | 成功 | 执行步数 | 请求 chunk 数 | 视频 |
| --- | --- | --- | --- | --- | --- |
| 开中间抽屉 | 0 | 是 | 121 | 16 | [主视角＋腕部](validation/20261013/videos/task_0_episode_0.mp4) |
| 开中间抽屉 | 1 | 是 | 115 | 15 | [主视角＋腕部](validation/20261013/videos/task_0_episode_1.mp4) |
| 把碗放到炉台 | 0 | 是 | 89 | 12 | [主视角＋腕部](validation/20261013/videos/task_1_episode_0.mp4) |
| 把碗放到炉台 | 1 | 否，超时 | 300 | 38 | [主视角＋腕部](validation/20261013/videos/task_1_episode_1.mp4) |

视频为 512×256、20 fps，每个实际执行动作对应一帧；这是按仿真动作频率播放，包含服务器等待时间的整次测试耗时另存于 JSON。所有执行动作均为有限的 7D 向量。平均整次 chunk 获取与处理耗时为 320.9 ms，受共享 GPU 的其他负载影响。

成功轨迹可看到抽屉被拉开、碗被搬到炉台。失败轨迹也抓起了碗并移动到炉台附近，落点偏离，后续反复调整。这个观察只能解释本次轨迹，不能把历史 98% 与本次 1/2 直接作模型质量比较。

原始产物：[结果 JSON](validation/20261013/libero_eva_verified_55k_20261013.json)、[执行动作 NPZ](validation/20261013/libero_eva_verified_55k_20261013.npz)、[日志](validation/20261013/libero_eva_verified_55k_20261013.log.txt)。

### 完整滚动序列的接口核对

将失败回合的 38 次观测重新通过 EVA 闭环获取，再将同一组观测送入未修改的 LIBERO `ModelClient.step`。两次均重置模型缓存和随机种子，使用同一个 StarVLA 服务端：

- 两张图像的像素最大差异：0。
- 38×8×7 原始动作的最大差异：0。
- 夹爪二值化后的完整 chunk 最大差异：0。
- 重新执行的前 300 个动作与已保存失败回合的执行动作最大差异：0。

[核对结果](validation/20261013/eva_libero_verified_55k_bowl_rollout_parity_20261013.json)及同名 `.input.npz`、`.input.reference.npz` 保存输入和输出。它支持“EVA 适配没有改变这段原生接口动作”的判断；两边共用当前 StarVLA 服务端，不能独立排除历史模型代码版本兼容性问题。

## 3. G2 的九条保留数据

权重：[30K final_model/pytorch_model.pt](validation/20261013/agibot_g2_eva_pipeline_final_20261013.json)。使用纯瞬时速度 Flow Matching＋Rolling 的 `QwenGR00TRollingFlow`，H64/C16/4 次迭代，不含 MeanFlow 自蒸馏。

数据来自 `AgiBotG2/white_carton_to_left_tan_carton_20261009`；保留 episode：0、13、27、40、54、67、81、94、108。通过 EVA DatasetTransport、观测构造、策略客户端和动作处理执行，模型每条 episode 重置一次、每 16 帧请求一次，后续请求保留滚动缓存；输入图像和状态来自录制数据。

| episode | 动作步数 | MSE | RMSE |
| --- | --- | --- | --- |
| 0 | 288 | 0.005160 | 0.07184 |
| 13 | 347 | 0.002357 | 0.04855 |
| 27 | 224 | 0.003061 | 0.05532 |
| 40 | 290 | 0.001060 | 0.03256 |
| 54 | 291 | 0.003311 | 0.05754 |
| 67 | 201 | 0.002073 | 0.04553 |
| 81 | 231 | 0.002808 | 0.05299 |
| 94 | 227 | 0.001944 | 0.04409 |
| 108 | 221 | 0.002827 | 0.05317 |
| 全部 | 2320 | **0.002755** | **0.05249** |

整体 MAE=0.01436、动作向量平均余弦相似度=0.99811。许多维度在这个任务中基本固定，会抬高全维余弦相似度；活动右臂七维 RMSE=0.06553，右夹爪开闭一致率=94.05%（138/2320 步不一致），需要关注动作时机。指标为反归一化且经过 EVA 动作处理后的 qpos，不能折算为真机任务成功率。

三张相机按 `top_head / hand_left / hand_right` 顺序，BGR→RGB 转换一次，直接采用训练数据相同的 Pillow 默认 bicubic resize 至 224×224。抽查 episode 0/frame 16：三张图像与原生训练加载方式像素差异为 0，原始 24D 状态排列一致。状态在服务端按权重的 G2 变换处理，动作由原生 PolicyServerWrapper 反归一化。

原始产物：[结果 JSON](validation/20261013/agibot_g2_eva_pipeline_final_20261013.json)、[预测及目标 NPZ](validation/20261013/agibot_g2_eva_pipeline_final_20261013.npz)、[汇总](validation/20261013/eva_g2_final_summary_20261013.json)。回放从未调用 `publish_action`。

## 4. 并行扩展与复现入口

此次桥接实现集中在 `examples/realRobots/EvaClient/`，通过外部插件注册 policy、transport、robot 与 strategy。核心调用仍使用 EVA 的 `build_policy_observation / fetch_action_chunk` 和 StarVLA 的 `PolicyServerWrapper`。本仓库已有其他模型代码改动；此次桥接未修改这些官方核心调用路径。

关键处理包括：

1. 两端共享 contract，握手校验图像顺序、状态切片、动作维度、归一化键、H/C/迭代数。
2. 完整执行 rolling chunk，避免 EVA 默认裁为 5 步导致模型缓存与已执行动作错位；跳过会被丢弃的模型预热预测。
3. 每回合、连接和断开时重置滚动缓存；单个服务端只接一个机器人流。
4. G2 使用训练一致的 resize，LIBERO 使用原生 benchmark 一致的旋转、resize 与夹爪转换。
5. EVA Python 3.11 与模型 Python 3.10 分开运行，仿真使用 CPU OSMesa 渲染。

复现命令和环境设置见 [README](README.md)。G2 配置为 [g2_replay.py](configs/g2_replay.py)，真机入口为 [agibot_g2_live.py](configs/agibot_g2_live.py)。

更换本体与算法时，需要该本体的 EVA adapter、权重支持的框架与匹配的 contract；有不同动作语义的机器人需要明确映射。此次实际验证范围为 G2 数据回放和 LIBERO 仿真闭环。

验证结束时，本次 G2 与 LIBERO 测试服务器已退出、显存已释放；检查确认原有训练与评估任务仍在运行。该记录描述验证结束时的状态。
