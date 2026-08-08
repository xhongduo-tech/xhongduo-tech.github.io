---
title: GPU 推理的关键指标：显存容量、带宽、算力
date: 2026-08-07
---

# GPU 推理的关键指标：显存容量、带宽、算力

<div class="epigraph">
<p>选显卡不是选「最强」，而是选「瓶颈匹配」。</p>
<footer>—— 硬件选型共识（借自 Roofline 实践）</footer>
</div>

<div class="article-byline">
<p>第四级 · 大模型部署 ｜ NVIDIA 显卡规格与 Roofline 模型 ｜ 2026-08-07</p>
</div>

## 为什么从 GPU 指标开始

本专题从头到尾都在说「推理是 Memory-Bound」——这句话落到硬件选型上，就变成三个具体指标：**显存容量**（装不装得下）、**显存带宽**（搬得快不快）、**算力**（算得快不快）。大多数时候，决定 LLM 推理速度的不是算力而是带宽，决定模型能不能跑的不是性能而是容量。理解了这三个指标的职责，选显卡就有了坐标系。<span class="marginnote">本专题《算术强度与 Roofline 模型》讲过理论框架；本篇把<strong>三个硬件指标对应到推理的三个问题</strong>：容量决定「放不放得下」，带宽决定「decode 多快」，算力决定「prefill 多快」。</span>

本篇讲三个指标的定义、它们在 LLM 推理中的分工、以及怎么用规格表快速判断「一块卡适不适合」。

## 1 显存容量：决定「放不放得下」

**显存容量（memory capacity）**是 GPU 上能存多少数据的硬上限。对 LLM 推理，它决定三件事：

**模型装不装得下**：权重 + KV Cache + 激活的总和 ≤ 显存。模型每大一个量级，容量需求直线上升。
**并发上限**：KV Cache 预分配随并发增长，容量小 = 并发低。
**上下文上限**：长上下文 = 大 KV Cache，容量是上下文的硬约束。

**选型的「容量账」**：模型 FP16 下权重 = 2×参数量 GB。7B = 14 GB，70B = 140 GB。INT4 减到 1/4（3.5 GB / 35 GB）。**先算容量账，排除装不下的卡，再谈性能**。<span class="marginnote">容量是「一票否决」指标：<strong>装不下，性能再好也白搭</strong>。这也是为什么「模型多大、卡要多大显存」是部署的第一句话。</span>

## 2 显存带宽：决定 decode 多快

**显存带宽（memory bandwidth）**是 GPU 每秒钟能从显存读写的字节数（GB/s）。对 Memory-Bound 的 decode，带宽直接决定速度：

$$T_{\text{per-token}} \approx \frac{\text{权重体积}}{\text{带宽}}$$

（见《llama.cpp》篇的公式）。A100 的带宽约 2 TB/s，H100 约 3.3 TB/s，4090 约 1 TB/s。**带宽翻倍，decode 吞吐近翻倍**——它是 LLM 推理最重要的性能指标。<span class="marginnote">带宽由「<strong>显存类型 × 位宽 × 频率</strong>」决定：HBM3（H100）比 GDDR6X（4090）带宽高一个量级。这也是为什么「4090 算力不弱但推理慢于 A100」——算力再强，带宽喂不饱。</span>

**带宽的决定性场景**：decode（每 token 读一遍权重）、小 batch、长上下文（KV Cache 访存）。凡「数据量大、计算量小」的环节，带宽都是主瓶颈。

## 3 算力：决定 prefill 多快

**算力（compute throughput，TFLOPS）**是 GPU 每秒能做的浮点运算次数。对 Compute-Bound 的 prefill（大矩阵乘），算力决定吞吐：

$$T_{\text{prefill}} \approx \frac{\text{计算量}}{\text{算力}}$$

**算力是「并行度上去之后」的瓶颈**——大 batch、长 prompt、prefill 阶段。FP16 算力 A100 约 312 TFLOPS、H100 约 989 TFLOPS（含稀疏）、4090 约 82 TFLOPS。<span class="marginnote">算力与带宽的比值——<strong>算术强度（ops/byte）</strong>——决定工作负载落在哪一侧（见本专题 Roofline 篇）。prefill 的算术强度高（计算多）、decode 的算术强度低（搬权重重），所以前者吃算力、后者吃带宽。</span>

**三个指标的分工一句话**：容量决定「能不能跑」，带宽决定「decode 快不快」，算力决定「prefill 快不快」。LLM 在线服务（decode 为主）→ 带宽优先；离线批处理（prefill 为主）→ 算力优先。

## 4 公式解析：规格表怎么读

把三个指标放进 Roofline，规格表就有了意义。给定权重字节 $W$、计算量 $F$、算力 $C$、带宽 $B$：

- **第一步，写 decode 时间**：$T_{\text{dec}} = W/B$（Memory-Bound）。
- **第二步，写 prefill 时间**：$T_{\text{pre}} = F/C$（Compute-Bound，batch 够大时）。
- **第三步，读两种负载的选型**：在线服务（decode 主导）总时间 ≈ $W/B$——**比带宽即可**；离线批处理（prefill 主导）总时间 ≈ $F/C$——**比算力即可**。混合负载则要看「decode 步数 × $W/B$ + prefill 次数 × $F/C$」的加权，选「总时间小」的卡。

$$\text{选型分数} = \alpha \cdot \frac{W}{B} + (1-\alpha)\cdot\frac{F}{C}, \quad \alpha = \frac{\text{decode 步数占比}}{}$$

$\alpha$ 接近 1（在线对话）时，**带宽是唯一该比的指标**——这就是「为什么 4090 看似强却不如 A100 适合在线推理」的数学表达。

## 5 小结

- **三个关键指标**：显存容量（装不装得下）、显存带宽（decode 快不快）、算力（prefill 快不快）。
- **容量是一票否决**：模型装不下，性能再好也没用；先算容量账再谈性能。
- **带宽是 LLM 在线推理的最重要指标**：decode 每 token 读一遍权重，带宽翻倍吞吐近翻倍。
- **算力是 prefill 与大批量的瓶颈**：离线批处理、长 prompt 场景算力优先。
- **选型看负载**：$\alpha$ 接近 1（decode 主导）比带宽，$\alpha$ 接近 0（prefill 主导）比算力。

在下一节，我们用这三个指标做一次具体对比——**A100、H100、4090 推理性能对比**。
