# EVA × StarVLA 部署版

部署集成与验证模块主入口：`examples/deployment/eva/`。原 `eva-deploy-20261013` 的记录与兼容入口保留。

- [部署与全新机器迁移指南](docs/eva_deployment.md)
- [版本说明](docs/releases/eva-deploy-20261013.md)
- [部署扩展、两台机器架构与更新回归](examples/deployment/eva/README.md)
- [G2 数据注册与训练](examples/realRobots/AgiBotG2/README.md)
- [部署模块迁移后的回归报告](examples/deployment/eva/validation_report_20261013.md)
- [G2异步部署交接：默认第8步预取](examples/deployment/eva/G2_ASYNC_DEPLOYMENT.md)
- [完整动作链路检查与G2硬件入口调整](examples/deployment/eva/validation_report_20261014.md)
- [已有验证结果与视频](examples/realRobots/EvaClient/validation_report_20261013.md)
- [EVA 来源记录](examples/realRobots/EvaClient/eva_source.json)

模型服务器使用完整 StarVLA 仓库和训练所得权重目录；机器人端使用
`EVA-CLIENT/` 与 `examples/deployment/eva/`，保留相同相对目录布局；独立 EVA checkout 可通过 `--eva-root` 指定。
通过外部插件注册策略、图像处理、rolling 调度与 LIBERO transport，
由 `run_eva.py` 启动 EVA；共享 contract 描述相机、状态与动作约定。

机器人端可从版本提交导出：

```bash
bash scripts/package_eva_deployment.sh HEAD
```

输出位于 `dist/`。权重、基础 Qwen 模型、数据集、GDK 和 Python 环境单独准备。
历史 V3 覆盖式补丁与源 ZIP 保留在本机，现行部署入口以上述插件为准。
