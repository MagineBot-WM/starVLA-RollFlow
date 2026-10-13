# eva-deploy-20261013

发布日期：2026-10-13。Git 分支：`release/eva-deploy-20261013`；标签：`eva-deploy-20261013`。

## 包含内容

- 用户提供的 EVA 客户端源码、原有文档、机器人模型资产、硬件节点与 Apache-2.0 许可证。
- 位于 `examples/realRobots/EvaClient/` 的外部插件、共享 contract、策略服务器、EVA 启动入口、数据回放与 LIBERO 闭环评估入口。
- G2 数据注册、100 条训练／9 条验证划分、固定关节归一化、pure rolling-flow H64/C16/K4 训练配方及配套 DeepSpeed 配置。
- 服务器地址与 G2 ZMQ 地址的环境变量入口；回放输出改用仓库 `work_dirs/eva_validation/`。
- 部署迁移说明、机器人端 Git 版本打包脚本、既有验证 JSON 与四段 LIBERO 视频。
- 继承基线提交 `1207b16` 中已提交的 StarVLA 模型实现，包括独立 `QwenGR00TRollingFlow`。

## 验证记录

| 路径 | 结果 | 范围 |
| --- | --- | --- |
| G2 九条数据经 EVA 请求 StarVLA | 2320 步，149 chunks，MSE 0.002755 | 记录数据拟合，未下发真机动作 |
| LIBERO → EVA → StarVLA → LIBERO | 3/4 次成功 | 仿真闭环短测 |
| EVA 与原生 LIBERO 客户端 | 38 chunks，图像与动作最大差异 0 | 共用服务端的接口一致性 |

详见 [验证报告](../../examples/realRobots/EvaClient/validation_report_20261013.md)。这些记录来自整理前的工作区，本次提交没有重新运行推理。工作区存在尚未提交的 MeanFlow 诊断改动，因此这些记录不等于本标签经过了独立的完整模型回归评估。G2 纯流路径独立于 MeanFlow 实现。

## 整理边界

EVA 核心源码保留用户提供的原始版本；集成通过外部插件完成。工作区的 MeanFlow/LSD、padding/诊断实验改动及相关测试继续保留，未收入本版本。原 V3 覆盖式补丁及源压缩包留在本机；`INTEGRATION_README.md` 更新为现行部署入口。

模型权重、基础模型、G2 数据、运行日志及 Python 环境单独管理。EVA 原有 `examples/agilex_dataset/` 演示录制数据未收入源码版本；部分硬件需要另行安装厂商 SDK，嵌套 `.gitmodules` 仅保留来源记录。

## 使用

按 [部署指南](../eva_deployment.md) 准备两个环境、模型完整目录和相应 contract。机器人端包通过 `bash scripts/package_eva_deployment.sh eva-deploy-20261013` 导出。
