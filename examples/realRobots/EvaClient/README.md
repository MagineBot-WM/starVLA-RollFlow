# EVA 部署兼容入口

当前入口迁移至 [examples/deployment/eva](../../deployment/eva/README.md)，定位为部署集成与验证模块。

本目录的 Python 与配置入口转发到新实现；历史 V1 契约、数据注册与验证产物保留原样。新客户端和服务器使用新目录同一份 V2 契约，不混用两种配置。

- [两台机器部署与更新指南](../../../docs/eva_deployment.md)
- [历史 G2 / LIBERO 验证结果](validation_report_20261013.md)
- [EVA 来源记录](eva_source.json)

早期G2专用原型保留在新目录的 `legacy/`，供旧入口兼容。
