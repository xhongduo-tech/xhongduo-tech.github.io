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


## 先把“当前步”与“上一结果”分开

固定快照 [Worker.execute_model_step](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/worker/worker.py#L89)提供了一条具体的阅读线：先执行当前模型前向，再处理前一步状态，接着进行当前采样；prompt、非 ZOS 和 ZOS decode 的后处理时机还不同。

这让“调用一次 step”不再天然等于“所有返回信息都属于刚提交的同一批”。需要在事件表里明确：

| 数据或资源 | 必须记住的身份／时机 | 错误后果 |
| --- | --- | --- |
| 请求与 batch 行 | 请求身份、当前行、前一批到当前批的 remap | 把上一次同一行的置信度或输出交给另一个请求 |
| 总预算与每请求长度 | 哪一步的预算、是否已按实际结果修正 | 目标执行、采样与 CPU 元数据对同一请求理解不一致 |
| KV 与其他模型状态 | 已计算、已承诺、待验证和预留范围 | 读取未就绪状态或把拒绝草稿当历史 |
| 结果与池化资源 | 最后消费者是谁、何时能释放 | 返回给调用方的视图仍在使用时被复用 |

这些是要沿代码确认的约束，不表示已验证所有异步分支。尤其不能用“长度总和正确”替代“每个长度属于正确请求”。

### 一个 remap 错误能怎样躲过总量检查

假定上一批顺序是 A、B，条件置信度分别为 $(0.9,0.5)$、$(0.6,0.5)$。下一批顺序变成 B、C、A，其中 C 是新请求，则 remap 为：

$$
[1,-1,0].
$$

按固定 reference，新请求没有可沿用的历史置信度。对总 query 预算 $Q=5$，每请求一个基础位置，两个额外位置应分给 A 的第一项与 B 的第一项，得到下一批的 query 长度：

$$
(q_B,q_C,q_A)=(2,1,2).
$$

若错误地按行复用前两条、把新出现的第 3 行视为无历史，便会把旧 A 的置信度交给 B、旧 B 的置信度交给 C，并得到 $(2,2,1)$：长度仍合法、总和仍是 5，但 C 继承了别人的状态，A 反而丢失自己的预算身份。

这只是[预算 reference](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/layers/tilelang_ops/schedule_mtp_varlen.py#L35)语义下的教学反例，不是已经发现 DeepInfer 有该 bug。正式验收须同时核对请求身份、重排和数值，而不只比较 aggregate。

## 异步重叠的收益，先由依赖关系决定

设单步 CPU 准备耗时为 $S$、设备执行耗时为 $G$，并且下一批准备确实只依赖已经可用的信息；忽略传输、额外后处理、编译和同步。理想双阶段流水线处理 $n$ 步的成本为：

$$
T_{\mathrm{pipe}}
=S+G+(n-1)\max(S,G).
$$

例如人为取 $S=2$ ms、$G=8$ ms、$n=3$。串行成本为 30 ms，满足上述独立性和重叠条件时为 26 ms。第一步的填充和最后排空不能被省成 $3\times8=24$ ms。

| 依赖条件 | 三步教学成本 | 能支持的结论 |
| --- | ---: | --- |
| CPU 准备与上一设备步可合法重叠 | 26 ms | 此模型可隐藏一部分准备时间 |
| 下一批必须等上一结果才能开始准备 | 30 ms | 当前依赖链不允许该重叠 |
| 实际多出传输、同步或元数据修正 | 需重新建模 | 不能直接套用理想公式 |

该例不预测 DeepInfer 性能。它提出的是源码问题：调度使用哪一代状态，哪些更新可以延后，在哪里等待必要结果？对 [DSpark 预算的可见信息](/inference/dspark)，这种时序还可能关系到分布正确性，不能只当作速度技巧。

## 沿真实资源生命周期检验“返回后还能否使用”

有一个已核实、适合深入的具体接口：[LLMEngine.step 的返回说明](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/engine/llm_engine.py#L860)指出结果对象由池管理，并在后续 step 中按其约定回收；调用方若要超出当前生命周期保留，需要使用相应 clone／release 机制。读取返回值不等于获得永久独立副本。

另可追踪 [DeferredRelease](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/engine/llm_engine.py#L67)的 collect、release 和 clone 相关事件。要画的是**生产者 → 最后消费者 → 允许回收点**，而不是只画函数调用箭头。

| 对照／负用例 | 应检查什么 | 不能据此扩大成什么结论 |
| --- | --- | --- |
| 结果保留到下一次 step 之后 | 是否按接口取得独立保留的副本并最终释放 | 普通 Python 变量引用一定能延长底层池化对象寿命 |
| 取消一个共享状态的请求 | 其他请求和在飞消费者是否仍被保护 | CPU 请求集合变空就表示设备已无读者 |
| batch 重排／新增／删除 | remap 后每个请求的状态与预算身份 | 总预算不变就表示映射正确 |
| 开关异步进行对照 | 相同输入、采样条件、状态与结果边界 | 延迟更低便说明所有竞态已排除 |

完整实现仍有状态管理、设备事件和调用方协议需要阅读。CPU 事件模型可以先否证明显的提前复用，但异步竞态及硬件收益要在相应环境中另行验证。

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

资料与具体源码入口核验日期：**2026-11-17**。remap 与流水线数值均为教学推导，不是本人的框架运行或性能结果。本页所有 DeepInfer 代码链接固定同一 commit，未用滚动 HEAD 证明具体实现。

- [DeepInfer README](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/README.md)与[部署问答](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/doc/inference_deployment_qa.md)：支持范围、配置条件和公开边界。
- [DeepSeek-V4.1-Flash 官方模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)：架构与权重说明的入口；模型卡可更新，实验前另固定 revision。
- [DSpark 论文 v1](https://arxiv.org/html/2607.05147v1)：理论调度和实际异步方案的背景。
- [deepseek-recipe 固定快照 README](https://github.com/deepseek-ai/deepseek-recipe/blob/935817c66f5d6e64838f419692545f06787e485f/README.md)：协议编解码职责。
