---
title: DSpark的草稿与验证调度
description: 核实半自回归草稿、条件置信度、前缀存活概率和硬件相关验证预算。
kind: topic
status: planned
---

# DSpark 的草稿与验证调度

::: info 专题计划 · 待研究
计划依次做论文阅读、CPU reference 和受支持环境的硬件验证。尚未完成训练、性能复现或生产部署。
:::

## 先确认项目身份

本页的 DSpark 指 **DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation**，由 DeepSeek 与北京大学研究者发布，论文为 [arXiv:2607.05147v1](https://arxiv.org/abs/2607.05147v1)，提交日期为 2026-07-06。不是 Apache Spark，也不是泛指某种分布式调度器。

相关官方代码分属两个对象：[DeepSpec](https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/README.md)包含草稿训练与评估；[DeepInfer](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/README.md)包含推理执行及置信度相关验证调度。训练、离线接受率评估和线上服务性能应分别研究。

## 要解决的问题与前置依赖

大草稿块不一定值得全部交给目标模型验证：越靠后的 token 越可能因前面拒绝而无效，验证开销还随系统负载变化。研究问题是：怎样产生低成本草稿、估计可接受前缀，并把目标模型计算分给有收益的位置。

先完成[投机采样的正确性](/inference/speculative-decoding)、[MTP 的训练与草稿边界](/inference/mtp)，会读累积乘积、排序和 batch 张量；分析真实吞吐前再补[服务调度与指标](/inference/serving-systems)。不要求先具备整个分布式集群的实现经验。

## 机制：草稿、置信度与预算

论文的半自回归草稿将可并行的 backbone 与轻量顺序模块结合，补充块内依赖；不是所有草稿位置彼此独立。置信度头估计“前面均已接受”条件下，当前位置通过验证的概率，并用校准减少概率失真。[论文 §3.1–3.2.1](https://arxiv.org/html/2607.05147v1#S3)

记请求 $r$ 第 $j$ 个位置的条件置信度为 $c_{r,j}$，则前缀存活概率为：

$$
a_{r,j}=\prod_{k=1}^{j}c_{r,k}.
$$

这来自条件概率链式分解，不需要假设各位置独立。论文调度分析中，若各请求验证前缀长度为 $\ell_r$，则：

$$
B=\sum_{r=1}^{R}(1+\ell_r),\qquad
\tau=\sum_{r=1}^{R}
\left(1+\sum_{j=1}^{\ell_r}a_{r,j}\right),\qquad
\Theta=\tau\cdot\mathrm{SPS}(B).
$$

$\mathrm{SPS}(B)$ 是硬件实测的每秒执行步数曲线；在论文的负载简化条件下，用它估计不同验证 batch 的收益。并非所有架构、极长上下文或通信条件都只由 $B$ 决定。[论文 §3.2.2，含假设脚注](https://arxiv.org/html/2607.05147v1#S3.SS2.SSS2)

### 调度的正确性边界

不能看到某个草稿 token 的值后，再利用其后续表现决定是否让它进入验证集合。论文 Appendix A 给出这种回看选择产生分布偏差的反例。§3.2 的因果贪心调度与 §5.2–5.3 的实际异步方案须分别阅读：后者通过调度可见信息的设计处理非平滑成本曲线，并非简单删除一个停止条件。[论文 Appendix A 与 §5](https://arxiv.org/html/2607.05147v1)

## CPU 小实验：先验证固定预算分配

计划从 [DeepInfer 的 reference 函数](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/layers/tilelang_ops/schedule_mtp_varlen.py#L35)提取纯 Python／NumPy 教学实现，避免导入依赖设备的整个引擎：

1. 输入小批条件置信度与旧批次 remap，手算累积乘积；新请求的历史置信度按源码语义处理。
2. 在合法范围 $R\le Q\le R(1+\gamma)$ 内固定总 query 预算，比较朴素排序与 reference 的长度分配。
3. 检查各请求至少一个基础 token、长度上界、总和为 $Q$、前缀约束、相同概率的稳定决胜，以及请求换位与 remap。
4. 另写论文回看选择的两 token 反例，精确枚举输出概率；与正确目标采样作对照。

**边界：** 该 reference 接收已经决定的总预算，不负责完整成本曲线搜索、置信度训练与校准、无损采样证明或整个异步系统。通过其测试只能证明所检查的分配性质。

## 后续 GPU 实测

优先核对 DeepSpec 已发布的对应目标／草稿 checkpoint 和配置，再决定是否进行接受率评估；不把默认的大规模训练流程作为第一步。若研究端到端收益，还需匹配运行时、采样策略、目标模型与硬件，并记录校准数据和成本曲线。

分开报告草稿生成时间、被验证长度、实际接受长度、每步时间及服务延迟；低并发与高并发分别测。一个 Qwen 草稿评估结果不能直接代表 DeepInfer 中 V4.1-Flash 的生产执行结果。

## 验收产物

- 论文模块图与符号表，说明条件置信度、前缀概率和硬件成本的不同角色。
- 固定预算 CPU reference、性质断言及回看偏差反例。
- 一张“论文算法／公开 reference／完整运行时”职责对照表；未检查的机制明确列出。
- 若进行硬件实验，保存模型配对、revision、原始接受率与时延数据，不引用论文成绩作为个人结果。

## 常见错误

- 直接相加单位置置信度，忽略只有连续前缀能够通过验证。
- 把高置信度或高接受率当作端到端提速保证，忽略校准和执行成本。
- 将固定预算 top-k 分配函数当作整个 DSpark 算法。
- 认为任意回看选择都保持目标分布，或反过来断言所有自适应长度方案都有偏。

## 一手资料与版本

核验日期：**2026-11-16**。

- [DSpark 论文 v1](https://arxiv.org/html/2607.05147v1)：§3 原理，§5 实际系统，Appendix A 正确性反例。
- [DeepSpec README](https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/README.md)、[DSpark Qwen3-4B 配置](https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/config/dspark/dspark_qwen3_4b.py)：固定快照，核对训练、目标模型和发布权重的关系。
- [DeepSpec 置信度评估入口](https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/eval/dspark/confidence_head.py)：源码阅读任务，不代表已完成代码分析。
- [DeepInfer 预算 reference](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/layers/tilelang_ops/schedule_mtp_varlen.py#L35)：当前 CPU 练习的接口依据。
