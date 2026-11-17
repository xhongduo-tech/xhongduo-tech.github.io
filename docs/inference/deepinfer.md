---
title: DeepInfer的推理执行与系统边界
description: 固定源码快照，从engine到worker追踪验证预算、异步执行、并行与缓存边界。
kind: topic
status: planned
---

# DeepInfer 的推理执行与系统边界

::: info 专题计划 · 待研究
当前是固定版本的系统阅读与实验计划。尚未运行完整 DeepInfer、完成源码复现或测得端到端性能。
:::

## 先确认研究对象和支持范围

DeepInfer 是 DeepSeek 公开的推理框架。本页固定快照为 **a679bb86ce36e883a98a97733492b686e5b9a08e**：[README](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/README.md)称本次发布提供离线推理核心，列出的支持模型为 **DeepSeek-V4.1-Flash**，硬件为 **NVIDIA SM100 GPU／Huawei Ascend 950 NPU**。这些是该快照的声明，不等于任意 GPU、NPU 或小模型都可以运行。

同一 README 的离线示例默认 checkpoint 是 DeepSeek-V4.1-Flash-1017；复现时应记录准确模型仓库与 revision，不能仅记“V4”。它与论文 [DSpark v1](/inference/dspark)中描述的实验对象也须分别对应。

完整服务还需要 KV cache 系统、VE／Prefill／Decode 分离与跨 DP 副本的负载均衡等组件；[官方部署问答](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/doc/inference_deployment_qa.md)说明生产 KV 管理和负载均衡代码尚未完全公开。这不意味着仓库里没有任何 KV 相关代码，也不应把公开离线核心说成完整线上平台。

## 要解决的问题与前置依赖

研究一个真实运行时怎样衔接请求状态、草稿验证、调度预算、模型执行和通信，而不是收集组件名称。

先掌握[采样与验证](/inference/speculative-decoding)、[KV 状态](/inference/kv-cache)和[服务指标](/inference/serving-systems)。读置信度预算模块时再结合 [DSpark](/inference/dspark)；读设备算子时补张量布局、精度与设备执行知识。可从一个 CPU 可理解的 reference 入手，不必先搭完整生产集群。

## 源码阅读主线

| 环节 | 固定快照入口 | 阅读后应能解释什么 |
| --- | --- | --- |
| engine | [LLMEngine](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/engine/llm_engine.py#L130)、[step](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/engine/llm_engine.py#L860) | 一个执行步的输入、状态更新与完成处理 |
| worker | [execute_model_step](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/worker/worker.py#L89) | 批次元数据怎样到达设备执行路径 |
| 验证预算 | [schedule_mtp_varlen.py](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/layers/tilelang_ops/schedule_mtp_varlen.py#L35) | remap、历史置信度、总预算与每请求长度 |
| 模型／算子 | [model_executor](https://github.com/deepseek-ai/DeepInfer/tree/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor) | attention、MoE、数值格式与专用内核的调用边界 |
| 系统说明 | [部署问答](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/doc/inference_deployment_qa.md) | 已公开引擎与外部缓存、路由、资源组织的边界 |

表格是待执行的阅读顺序，不代表已经完成调用链分析。首次只追踪一个请求、一个 step 和一个状态转换，再增加并行与异步分支。

## 关键机制与边界

- **异步调度：** README 声明 CPU batch scheduling 与设备执行重叠。阅读时标出哪一步消费哪一代状态、何时可见完成结果；不要默认 Python 调用返回就表示 GPU 已完成。
- **可变验证长度：** 固定预算模块输出每请求 query 长度 $q_r$，合法预算应满足 $1\le q_r\le1+\gamma$、$\sum_rq_r=Q$。分配 reference 不能替代完整的预算选择、校准和采样正确性分析。
- **模型专用执行：** Attention DP／balanced CP、MoE EP／EPLB、SWA bounded replay 属于该快照的组合；后者依赖模型结构、窗口与回退范围，不能当作任意模型都可截断输入的通用技巧。
- **确定性：** README 对 batch-invariant 与 deterministic sampling 列出平台和开关条件；不能从固定 seed 推断所有模式、所有 batch 都逐字一致。[功能条件](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/README.md)
- **协议边界：** [deepseek-recipe 固定快照](https://github.com/deepseek-ai/deepseek-recipe/tree/935817c66f5d6e64838f419692545f06787e485f)处理协议与 token 编解码。它不替代模型执行、外部工具执行与完整在线服务。

## CPU 小实验：状态与预算 reference

计划选两个小任务中的一个，不导入依赖特定设备的整套引擎：

1. 提取并重新表达固定预算 reference，使用合成置信度与 remap；检查边界、稳定排序、长度总和和新请求语义，具体性质见 [DSpark 页](/inference/dspark)。
2. 写一个小型请求状态机，模拟到达、进入 batch、产生 token、EOS／取消和释放；用事件日志检查完成一次、释放一次及取消后不继续输出。异步延迟用合成事件表达。

这只是接口与状态的教学验证，不证明生产实现没有竞态，也不等于 DeepInfer 已在 CPU 上运行。输出的调用链图须区分“源码确认”与“教学模拟”。

## 后续 GPU／NPU 实测

先检查 README 的硬件、模型容量、依赖和并行要求，列出资源预算；不在不兼容设备上默认尝试完整模型。有条件时冻结仓库 commit、模型 revision、驱动、通信库及相关内核版本，从官方离线示例建立基线。

把加载、编译、预热与稳态运行分开记录，再只选择可变验证长度或异步调度的一项对照。报告输入输出分布、批次、采样模式、每步预算、接受长度、内存、通信和端到端时间；无相应设备时，以代码阅读和 CPU 性质检查验收，不宣称复现运行时性能。

## 验收产物

- engine → worker → 模型／采样 → 状态更新的固定版本调用图，附函数位置。
- 输入输出 shape、状态所有者和生命周期表，指出异步可见性与释放边界。
- 一个独立 CPU 小脚本及断言；一份公开实现／外部组件／未核验部分的清单。
- 硬件预检与待实测记录；若实际运行，再附可复现配置和原始数据。

## 常见错误

- 把仓库发布等同于完整生产平台已全部开源，或断言没有任何缓存实现。
- 使用 V3 示例的配置、MLA 容量估算或旧支持列表推断当前快照。
- 以一个纯 reference 通过，声称验证了异步正确性或集群吞吐。
- 直接引用厂商生产指标作为自己的实验结果，忽略工作负载和资源规模。

## 一手资料与版本

核验日期：**2026-11-16**。本页所有 DeepInfer 代码链接固定同一 commit，未用滚动 HEAD 证明具体实现。

- [DeepInfer README](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/README.md)与[部署问答](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/doc/inference_deployment_qa.md)：支持范围、配置条件和公开边界。
- [DeepSeek-V4.1-Flash 官方模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)：架构与权重说明的入口；模型卡可更新，实验前另固定 revision。
- [DSpark 论文 v1](https://arxiv.org/html/2607.05147v1)：理论调度和实际异步方案的背景。
- [deepseek-recipe 固定快照 README](https://github.com/deepseek-ai/deepseek-recipe/blob/935817c66f5d6e64838f419692545f06787e485f/README.md)：协议编解码职责。
