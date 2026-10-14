# 部署集成与验证层回归报告（2026-10-13）

当前入口：[examples/deployment/eva](README.md)。机器人机运行 EVA 和部署插件，模型机运行 StarVLA 和模型服务，两端复用已有 WebSocket / MessagePack 通信。模块集中管理数据契约、执行约定、版本适配与验证入口。

## 验证结果

| 验证 | 范围 | 结果 |
| --- | --- | --- |
| CPU 接口 | 契约、图像、状态、会话、执行策略、独立 EVA checkout、机器人端独立安装 | **42 项通过，5.21 秒** |
| G2 保留集 | 9 条 episode，2320 个动作，149 个 rolling chunk | MSE **0.0027552359**，RMSE **0.05249034**；预测和目标与既有记录逐元素相同 |
| LIBERO 仿真闭环 | 两个任务各两个初始状态，实际执行动作后获取新观测 | **3/4 成功**；四条执行动作序列与既有记录逐元素相同 |
| LIBERO 原生接口核对 | 同一观测、服务与权重，重置缓存和随机种子，38 个 chunk | 图像、原始动作、处理后动作最大差异均为 **0** |
| EVA 上游源码审计 | 与用户提供的原始 ZIP 比较 1353 个文件 | **0 缺失，0 修改** |

原始产物：[CPU 日志](validation/20261013/cpu.log.txt)、[JUnit](validation/20261013/cpu.xml)、[G2 比较](validation/20261013/g2_comparison.json)、[LIBERO 比较](validation/20261013/libero_comparison.json)、[38 chunk 核对](validation/20261013/libero_reference.json)、[EVA 源码审计](validation/20261013/eva_source_audit.json)。

此次回归证明迁移后的部署路径在这些输入上保持了原有预测与仿真执行结果。G2 回放没有发布动作；此次未验证 G2 真机执行器映射或任务成功率。四次 LIBERO 闭环用于管线回归，完整 benchmark 没有重新运行。

## 实现中修复的问题

### 准备阶段会丢弃预测，却保留了模型缓存

原生 EVA 的准备流程会请求一个验证 chunk，丢弃结果，再调用 `strategy.reset()`。rolling 模型已经推进缓存；如果 reset 仅清理客户端计数，第一次实际执行就会使用包含未执行 chunk 的缓存。

外部 rolling strategy 现在通过 EVA adapter 绑定当前 policy 的重置回调，`strategy.reset()` 同时重置模型缓存。CPU 测试运行原生“准备 → warmup → 首个执行 chunk”流程，确认准备后缓存归零、warmup 不推理、首次执行从正确缓存起点开始。

### 数据与运行边界

- 图像处理从共享契约派生，检查相机顺序、RGB/BGR、直接 resize、旋转、状态切片、关节顺序和单位声明；归一化复用权重的训练变换。
- 实时输入使用同一完整 EVA 观测消息；缺少相机或状态时明确失败。数据回放缺图和解码失败也明确报错。
- rolling 完整执行返回 chunk；无缓存算法可采用独立 chunk 策略执行前缀。
- V2 在现有消息中携带契约 hash、会话和序号。请求超时关闭当前连接，结束不确定的滚动流。
- doctor 记录依赖、接口与源码指纹，检查部署配置和服务器握手；客户端 trace 保存模型输入和原始预测。
- 旧 `examples/realRobots/EvaClient` Python / 配置入口转发到新模块；其历史 V1 契约和验证产物保留。新 V2 配置需两端一起迁移。

## G2 回放

使用纯瞬时速度 Flow Matching + Rolling 的 `QwenGR00TRollingFlow`，H64/C16/K4，30K final 权重：

```text
/data/tzq/starVLA_checkpoints/agibot_g2_rollingflow_h64_c16_b64_mb4_ga4_scratch_30k_20261010/final_model/pytorch_model.pt
```

episode 为 0、13、27、40、54、67、81、94、108。每回合重置一次，每16帧请求一次，保持滚动缓存。图像和状态来自记录数据，预测经过 EVA 动作处理后与记录动作比较。

全部9组预测及9组目标数组与历史基准逐元素相同。[结果 JSON](validation/20261013/g2_replay.json)、[轨迹 NPZ](validation/20261013/g2_replay.npz)、[配置诊断](validation/20261013/g2_doctor.json)保留具体设置和逐回合指标。

## LIBERO 闭环

使用历史含 MeanFlow 自蒸馏的 `QwenGR00TRollFlow`，H32/C8/K4：

```text
/data/tzq/starVLA_checkpoints/libero_qwengroot_rollflow_h32_c8_b128_unfrozen_463a1f7/checkpoints/steps_55000_pytorch_model.pt
```

每个回合重置模型缓存及随机种子为7，执行完整8步 chunk，再请求下一次预测。结果如下：

| Goal 任务 | 初始状态 | 成功 | 执行步数 | 请求 chunk 数 |
| --- | --- | --- | --- | --- |
| 开中间抽屉 | 0 | 是 | 121 | 16 |
| 开中间抽屉 | 1 | 是 | 115 | 15 |
| 把碗放到炉台 | 0 | 是 | 89 | 12 |
| 把碗放到炉台 | 1 | 否，超时 | 300 | 38 |

[结果 JSON](validation/20261013/libero.json)、[执行动作 NPZ](validation/20261013/libero.npz)、[配置与服务器诊断](validation/20261013/libero_doctor.json)。新视频在本机 [videos 目录](../../../work_dirs/eva_validation/deployment_20261013/videos/)；可迁移的历史同动作轨迹视频与权重评估来源见 [既有报告](../../realRobots/EvaClient/validation_report_20261013.md)。

38 chunk 核对复用上游 `ModelClient.step` 的图像和动作处理，通过独立 transport adapter 适配会话与推理参数。两路共用当前 StarVLA 服务端，因此该比较检查 EVA 适配及 chunk / 夹爪处理，不能独立验证历史服务代码或完整模型正确性。[输入与参考动作](../../../work_dirs/eva_validation/deployment_20261013/)保存在本机工作目录。

## 复现与升级

CPU 入口：

```bash
"$EVA_PYTHON" -m examples.deployment.eva test -q
```

G2 回放、LIBERO 闭环、原生接口核对及两台机器启动命令见 [模块说明](README.md) 和 [部署指南](../../../docs/eva_deployment.md)。升级 EVA 或 StarVLA 后，先检查 doctor / CPU 测试，再运行固定保留数据和仿真回归；接口变化集中在对应 adapter 中处理。

运行模型使用 GPU3，PyTorch allocator 上限6GiB；仿真使用 CPU OSMesa。验证结束时，本次两个模型服务已退出，原有 ResWAM 训练与其他评估进程仍在运行。此次没有发送真机动作。

## 结果适用范围

- 测试基于 `b44e91844f53997833df95f2d97bd6d361109dc5` 加本次未提交部署变更的工作区。原有本地 MeanFlow 诊断改动也在工作区；本次部署整理没有修改它们，因此不声称复现干净历史标签的模型结果。
- [产物清单](validation/20261013/artifact_manifest.json)记录产物 SHA256、大小、基准提交和模型工作区差异指纹。JSON 中绝对路径记录测试机器；权重和数据另行准备。
- 原生观测只有整体时间戳，完整消息检查不能证明所有相机实际采集时刻一致。现有 ZMQ 没有动作执行确认；策略序号不能证明机器人完成了整个 chunk。
- 原生 EVA 同步执行可能等待关节到位，网络和推理也有停顿；配置30Hz表示目标频率，实际执行时序需测量。
- 当前已验证 G2 纯 flow 回放和 LIBERO 历史 MeanFlow 仿真。其他本体、算法或未来 API 变化需要相应契约、adapter 和回归，不能自动保证兼容。
