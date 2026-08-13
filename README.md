# runbooks

技术文章配套的复现手册。一篇文章一个目录,目录名与文章 slug 一致。

每个目录固定七节:硬件与镜像 · 测试工具 · 测试数据集 · 起服务 · 测试命令 · 判据与复算 · 已知坑。
脚本只依赖标准库,推送前照着 README 从头自测过一遍。

| 目录 | 配套文章 |
|---|---|
| [2026-08-12-deepseek-topology-tpot](./2026-08-12-deepseek-topology-tpot) | 按 SLO 档位选部署拓扑:DeepSeek-V4-Flash 在 16 张 910B4 上的六格实测 |
| [2026-08-10-qwen3-omni-realtime-capacity](./2026-08-10-qwen3-omni-realtime-capacity) | 实测 Qwen3-Omni-30B 带语音输出:A100 与昇腾 910B3/910B4 三方对比 |
