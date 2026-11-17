---
title: DeepInfer的推理执行与系统边界
description: 把解码、并行、算子和调度放回系统中理解。
kind: topic
status: planned
---

# DeepInfer的推理执行与系统边界

::: info 专题提纲
当前是系统阅读框架，尚未运行DeepInfer完整硬件实验。
:::

## 先确认研究对象

DeepInfer是DeepSeek公开的推理框架。2026年11月16日核验的公开版README说明提供离线推理核心，并列出DSpark、并行、异步调度、结构化输出等能力；完整生产服务还需要缓存系统、分离部署与负载均衡等组件。研究时分别记录已公开实现与系统说明，不把它当成开箱即用的完整线上服务。[固定版本README](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/README.md)

## 源码阅读主线

| 环节 | 要追踪的内容 |
| --- | --- |
| 输入与运行时 | 配置、模型加载、采样参数与执行入口 |
| 解码 | 草稿、目标验证、接受结果及状态更新 |
| 调度 | CPU调度与设备执行怎样衔接，验证预算怎样传递 |
| 并行 | Attention与MoE的分工、通信与负载平衡 |
| 算子 | FlashMLA、DeepGEMM、DeepEP等组件的调用边界 |
| 协议 | deepseek-recipe怎样在协议和token之间转换 |

## 验证方式

先固定commit画执行链路，再选择一个小模块做reference或性质检查。真正的端到端实测需要README支持的模型、硬件与环境，实验前单独核对成本与兼容性。

[固定版本部署问答](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/doc/inference_deployment_qa.md)用于理解整体系统；[官方源码](https://github.com/deepseek-ai/DeepInfer/tree/a679bb86ce36e883a98a97733492b686e5b9a08e)作为本轮阅读快照。
