---
title: 大规模 LLM 训练与 H100 实践
date: 2026-08-07
---

# 大规模 LLM 训练与 H100 实践

<div class="epigraph">
<p>预训练大模型不是科学实验，而是一场工程流水线上的大规模制造。</p>
<footer>—— 大规模 AI 训练的工程共识</footer>
</div>

<div class="article-byline">
<p>第四级 · AI 硬件：NVIDIA H100/Hopper ｜ NVIDIA 官方技术博客 ｜ 2026-08-07</p>
</div>

## 为什么从 LLM 训练实践讲起

前面所有知识——SM、Tensor Core、TMA、NVLink、Roofline——最终都要汇到同一个任务上：**把一个大语言模型训练出来**。这个任务的特殊之处在于它的规模：数千亿参数、数万亿 token、上千颗 GPU 协同工作数月。在这样的规模下，「快 10%」意味着省下几百万电费，「慢 10%」意味着 GPU 白烧数周。本节把「训练一个 LLM 到底要多少算力」「怎么把模型切到上千颗 GPU 上」讲清楚，再用 MFU 这个指标回答「H100 到底用到了几成」。<span class="marginnote">本专题与「从极限到大模型」主线的连接在此处最紧密：主线上我们学过的 Transformer 原理、混合精度训练、分布式训练，在这里都有了硬件的落点——H100 集群就是这些软件技术运行的物理舞台。</span>

## 1 训练一个 LLM 要多少算力

先给一个「算力预算」的定量框架。一个 Transformer 模型的**总训练计算量**由一条著名的经验法则给出：

$$
\text{FLOPs}_{\text{train}} \approx 6 \times N \times D
$$

其中 $N$ 是模型参数量，$D$ 是训练 token 数。系数 6 来自：每个参数在 forward 中约 2 FLOP（一次乘加），在 backward 中约 4 FLOP（对权重求梯度 + 对激活求梯度）。这条法则来自 Kaplan 等人的 scaling laws 研究，被整个行业当作预算第一公式。

代入一个实际例子：训练一个 70B 参数的模型、3.5T token：

$$
6 \times 70 \times 10^9 \times 3.5 \times 10^{12} \approx 1.47 \times 10^{24}\ \mathrm{FLOP}
$$

在 H100 上（FP16 989 TFLOPS），单卡跑完需要约 $1.47 \times 10^{24} / 9.89 \times 10^{14} \approx 1.5 \times 10^9$ 秒 ≈ 47 年。显然必须并行——而且要并行到「等效算力」足以在数月内完成。<span class="marginnote">这就是「训练一个前沿模型需要上千颗 GPU」的数字来源：1.5e24 FLOP 除以「目标天数 × 集群有效算力」，反推出需要的 GPU 数。规模、成本、进度三者在这里被同一条公式锁死。</span>

## 2 三种并行策略：怎么把模型切上千卡

模型太大、单卡放不下，需要把「模型」和「数据」都切开。主流有三种并行策略，各有分工：

**数据并行（Data Parallelism）**：每张卡持有一份完整模型副本，各算各的数据批次，定期用 all-reduce 同步梯度。通信量正比于模型大小（每步同步全部梯度），但实现最简单。

**张量并行（Tensor Parallelism）**：把单层的权重矩阵**按行/列切开**，放到多张卡上，前向/反向时卡间高频通信（每层两次 all-reduce）。通信量极大，因此**必须走 NVLink（900 GB/s）**——它被限制在单节点内。

**流水并行（Pipeline Parallelism）**：把网络按层切成多段，每段放一张卡，数据像流水线一样依次经过各段。通信只在相邻段之间，量小、频率低，适合跨节点。

实际大规模训练用**三维混合并行**：张量并行（节点内，NVLink）+ 流水并行（节点间，网络）+ 数据并行（全局）。一个典型的 8 卡节点内做「张量并行度 8」，多个节点间做流水 + 数据并行。<span class="marginnote">这个「三维混合」结构解释了为什么 H100 节点内部的 NVLink 全互连如此重要：张量并行的通信频率远高于其他两种，只有把张量并行放在 NVLink 域内，通信才不会成为瓶颈。我们在第 1 篇 DGX 一节埋的「NVLink 域」伏笔，在这里兑现。</span>

## 3 H100 集群上跑 LLM：MFU 与关键瓶颈

**MFU（Model FLOPs Utilization，模型算力利用率）**：衡量「GPU 峰值算力被用到了几成」的指标：

$$
\text{MFU} = \frac{\text{实际完成的有效 FLOP}}{\text{峰值算力} \times \text{运行时间}}
$$

业界在 H100 集群上把 LLM 预训练跑到 **MFU 40–55%** 就算优秀——远低于单 kernel 的 80%+。差距来自四类开销：

- **通信**：张量/流水并行的同步等待，GPU 空转；
- **负载不均**：pipeline 的 bubble（流水线气泡，某些段空闲）；
- **memory-bound 阶段**：归一化、残差、嵌入层等访存密集算子，吃不满 Tensor Core；
- **框架开销**：kernel 启动、调度、Python 解释开销。

影响 MFU 的最关键可调项是**批大小（global batch size）**与**并行配置**：批太小则数据喂不饱算力，批太大则显存吃紧、需要更多并行；并行切法不同则通信占比不同。业界经验是「以 MFU 为代价函数，搜索 batch size × 并行度 × 流水段数」的最优组合。<span class="marginnote">对比参考：A100 时代的 LLM 预训练 MFU 约 30–40%，H100 依靠 FP8 训练 + 更强互连提到 40–55%。这个「每代提升 10 个百分点」就是硬件 + 软件 + 库共同优化的结果——也说明「换 GPU 就自动快」不成立，要重写/重调。</span>

## 4 公式解析：从 FLOP 预算到 GPU 数量

把「训练预算」和「集群规模」串成一条完整的链：

$$
N_{\mathrm{GPU}} = \frac{6 \times N_{\text{model}} \times D_{\text{tokens}}}{T_{\text{days}} \times 86400 \times P_{\text{peak}} \times \text{MFU}}
$$

- $6 \times N \times D$：总训练 FLOP（模型与数据规模决定）。
- $T_{\text{days}} \times 86400$：目标训练天数换算成秒。
- $P_{\text{peak}} \times \text{MFU}$：单卡有效算力（峰值 × 利用率）。

代入：70B 模型、3.5T token、目标 90 天、H100 单卡 989 TFLOPS、MFU 取 45%：

$$
N_{\mathrm{GPU}} = \frac{1.47 \times 10^{24}}{90 \times 86400 \times 9.89 \times 10^{14} \times 0.45} \approx 425
$$

约 **425 颗 H100**（即约 53 台 DGX）可以在 90 天训完一个 70B 模型。三步拆解这条链：

- **第一步，定总预算**：$6ND$ 给出不可压缩的总计算量。
- **第二步，定有效算力**：单卡有效算力 = 峰值 × MFU，MFU 直接决定「纸面算力」打几折。
- **第三步，反推规模**：总预算 ÷ 有效算力 ÷ 目标天数 = 需要的 GPU 数。

这条链是「大规模训练的第一道算术」，也是理解「算力经济学」的起点——**预算与规模被同一条公式锁死，谁想压缩成本，只能从 MFU 与 FP8 里抠。**

## 5 小结

- 训练总 FLOP 约 **$6ND$**（参数 × token × 6），是规模预算第一公式。
- 三种并行策略分工：**张量并行（NVLink 域内）、流水并行（跨节点）、数据并行（全局）**，实际用三维混合。
- **MFU** 衡量 GPU 峰值利用率，H100 集群 LLM 预训练优秀水平 40–55%。
- 从预算反推 GPU 数：$N_{\mathrm{GPU}} = 6ND / (T \times 86400 \times P \times \text{MFU})$