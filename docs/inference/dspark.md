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


## 从置信度到预算：逐项复算的两请求例子

固定两个请求 $R=2$、最大草稿深度 $\gamma=2$，先把置信度视为可靠、在本次分配前已给定的条件概率：

$$
c_A=(0.9,0.5),\qquad c_B=(0.6,0.5).
$$

前缀存活概率分别为 $a_A=(0.9,0.45)$、$a_B=(0.6,0.3)$。第二项为 0.45 而非 0.5，因为它需要 A 的前两项同时通过；这正是连续前缀约束。

固定总 query 预算为 4 时，两个基础位置之外还有两个额外位置。分配 $(\ell_A,\ell_B)=(1,1)$ 时，$\tau=2+0.9+0.6=3.5$；全部给 A 则为 $2+0.9+0.45=3.35$。后一个位置质量更高，不等于同一请求的所有后续位置都应优先。

再指定一条完全人为的成本曲线：

| 总 query 数 $B$ | 所选 $(\ell_A,\ell_B)$ | $\tau$ | 合成 $\mathrm{SPS}(B)$ | $\Theta=\tau\cdot\mathrm{SPS}(B)$ |
| ---: | --- | ---: | ---: | ---: |
| 2 | (0,0) | 2 | 100 | 200 |
| 3 | (1,0) | 2.9 | 90 | 261 |
| 4 | (1,1) | 3.5 | 80 | 280 |
| 5 | (2,1) | 3.95 | 65 | 256.75 |
| 6 | (2,2) | 4.25 | 55 | 233.75 |

$B$ 是本步目标 query token 数，**不是请求数**；请求数一直为 2。$\tau$ 包含每请求一个基础产出项，也不能标成“被接受草稿数”。在这条合成曲线上，验证更多虽然提高预计产出，却在 $B=4$ 之后降低预计吞吐。

表中 SPS 没有硬件测量含义，且省略 EOS、草稿、额外调度与通信等成本。它只解释论文预算模型为何需要数据质量和执行成本两类输入，不能用来预测生产端到端加速。

### 排序正确还不够：为什么需要校准

前缀收益使用概率的数值，而不只是排名。若所有较长前缀都被过度自信地估高，分配器即使排序稳定，也可能系统性选择过大的验证预算。

因此计划分别核对：给定前缀预测概率与其真实通过频率是否对应；该关系在不同位置、任务和负载下是否改变。置信度来自学习得到的估计，不能因为做了累积乘积就自动变成真实概率。[论文 §3.2.1 的条件定义与校准](https://arxiv.org/html/2607.05147v1#S3.SS2.SSS1)是这一环的依据。

## 固定预算分配应当守住哪些不变量

记额外预算 $K=Q-R$。在 $0\le c_{r,j}\le1$、合法总预算和固定 remap 条件下：

| 性质 | 为什么需要它 | 反例／失败用例 |
| --- | --- | --- |
| 前缀存活概率非增 | $a_{r,j+1}=a_{r,j}c_{r,j+1}\le a_{r,j}$ | 把条件概率相加，会破坏概率意义 |
| 长度受限且总和为 $Q$ | 每请求一个基础位置，最多 $\gamma$ 个额外位置 | 预算小于请求数、过大预算和长度越界须显式处理 |
| 入选位置构成前缀 | 深位置通过依赖其前面位置 | 只选 A2 不选 A1 的“集合”不能当作合法验证前缀 |
| ties 的规则可复算 | 相等概率不能造成 CPU／设备路径分歧 | 全零、全一、相同前缀概率与不同 remap |
| 新请求没有旧请求的置信度身份 | batch 行号不等于永久请求 ID | 删除一条请求、插入新请求后错用上一批同一行 |

固定源码 [compute_seqlens_q_by_confidence_rates_ref](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/layers/tilelang_ops/schedule_mtp_varlen.py#L35)先按 remap gather，再做 cumprod、稳定排名和长度分配。上述性质对应它的输入与输出约束；总预算来自哪里、何时可用，则要回到 engine 和异步执行去确认。

在概率范围内，累积乘积保证同一请求前面的边际收益不低于后面；再按源码在同值时让同一请求的较早位置优先，固定额外预算的 top-k 才能表示合法的前缀选择。单独对任意浮点分数做 top-k，不自动拥有这一性质。

## 正确性与收益的双重否证

- **采样层：** 预算选择不应窥视当前候选实现值来回看决定是否接纳它。用论文 Appendix A 的明确选择偏差反例核对时序；不能以输出若干句“看起来正常”代替。
- **估计层：** 校准预测和实际前缀频率不匹配时，先修复估计解释，不把预测 $\tau$ 当作已测接受长度。
- **执行层：** remap、query 总量和有效 KV 出现分歧时，即使吞吐增加也不通过正确性验收。
- **成本层：** 平均接受长度提高而每步代价上升更多，否证的是收益主张；成本表在新硬件或新负载失准时需要重新 profile。

在 [DeepInfer 的运行时页](/inference/deepinfer)还要继续追踪这些数据何时产生、何时消费。论文算法、固定预算 reference 和完整异步实现必须分别留下证据，不能互相替代。

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

资料与固定源码核验日期：**2026-11-17**。新增预算表是解析教学例子，不是个人性能或训练结果。

- [DSpark 论文 v1](https://arxiv.org/html/2607.05147v1)：§3 原理，§5 实际系统，Appendix A 正确性反例。
- [DeepSpec README](https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/README.md)、[DSpark Qwen3-4B 配置](https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/config/dspark/dspark_qwen3_4b.py)：固定快照，核对训练、目标模型和发布权重的关系。
- [DeepSpec 置信度评估入口](https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/eval/dspark/confidence_head.py)：源码阅读任务，不代表已完成代码分析。
- [DeepInfer 预算 reference](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/layers/tilelang_ops/schedule_mtp_varlen.py#L35)：当前 CPU 练习的接口依据。
