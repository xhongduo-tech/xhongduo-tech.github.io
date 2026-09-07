---
title: GPipe
date: 2026-09-07
section: llm
---

# GPipe

<div class="epigraph">
    <p>把网络写成一层层的序列，切到不同加速器上；再把 mini-batch 切成微批在管道里流动，用同步 SGD 而不是异步参数服务器。</p>
    <footer>—— Huang 等，GPipe，arXiv:1811.06965，NeurIPS 2019</footer>
</div>

Google 的 **GPipe**（Yanping Huang、Youlong Cheng、Ankur Bapna、Orhan Firat 等）把模型并行从「为某一种结构写死切分」收成库：只要模型能表达成 **层的序列**，就可以把连续层组成 cell、放到不同加速器上，并用 **batch splitting** 做流水线。它针对 2018 年前后的现实：单加速器显存卡住了 AmoebaNet 与深层 Transformer 的宽度/深度，而当时不少模型并行方案绑定具体拓扑。本篇写 1811.06965 的调度、气泡、重计算，以及两条实验（ImageNet 上 5.57 亿 AmoebaNet、百语上 60 亿参数 Transformer）。后续 1F1B 与权重版本见 [PipeDream](/llm/pipedream) 与 [流水线并行](/llm/pipeline-parallel)。

## 问题

朴素模型并行把层 1 放卡 0、层 2 放卡 1，前向必须等前一卡算完，同一时刻只有一张卡在忙，加速比上限约 $1/k$（$k$ 为分区数）。显存也没有真正被「流水」省掉：若仍一次灌入整个 mini-batch，每张卡要为这个 batch 存自己那一段的激活。任务换了（视觉搜索 vs 翻译），层类型不同，为 CNN 写的切分到 Transformer 上不能用。

参数服务器加流水线可以重叠，但会引入陈旧梯度。GPipe 要的是 **任务无关** 的同步训练：数学上仍是对整个 mini-batch 的梯度下降，只是执行被切成微批管道。如何在利用率、激活内存和同步语义之间取点，就是微批数 $m$ 与分区数 $k$ 的关系。

### 微批管道与气泡

一个 mini-batch 分成 $m$ 个微批。每个 cell 按微批 $1..m$ 做完所有前向，再按 $m..1$ 做完所有反向。梯度在 $m$ 个微批上累积，**每个 mini-batch 更新一次权重**，因此是同步 SGD。填满 $k$ 段管道的气泡（空转）相对有效计算的比例约为

$$
O\left(\frac{k-1}{m+k-1}\right)
$$

（精确系数随是否把反向对称计入而略有出入）。$m\gg k$ 时利用率接近线性加速。代价：每个阶段默认要缓存该阶段上 **$m$ 份** 前向激活直到反向——激活内存 $\propto m$。大模型上激活往往已经是显存主角，于是 GPipe 配套 **re-materialization（重计算）**：前向少存中间激活，反向时再算一遍，用 FLOPs 换内存。重计算不是调度算法本身，但是让 $m$ 开得足够大、从而压气泡的前提。

<span class="marginnote">微批切的是 batch 维，不缩短序列。GPipe 不提供长上下文。配置里要分清 mini-batch、微批 $m$、以及数据并行副本——优化器看到的有效 batch 是它们的乘积。不要把 $m$ 叫做「pipeline batch」而不写全局 batch。</span>

## 方法

编程模型：用户把网络标成顺序层，库把连续层打包成 cell，每 cell 一卡（或一加速器）。前向在 cell 间传递激活，反向传递梯度。适用于「能写成序列」的网：AmoebaNet 的 cell 堆、Transformer 的层堆都可以；有复杂跳连、不是线性深度的图，要先被改写成可切的序列，或放弃 GPipe。

实验两条，用来说明 **与任务无关**：

- 视觉：把 AmoebaNet 加宽到 **5.57 亿** 参数，ImageNet-2012 输入 480×480，top-1 **84.4%**。
- 翻译：单一 **128 层、60 亿** 参数的多语 Transformer，在 100+ 语到英语的语料上，质量超过当时分别训练的 3.5 亿双语 Transformer Big。

这两条都不是 2024 年 LLM 预训练的配方，而是「同一套流水线库能扛两种结构」的证据。线性加速的测量在分区数增加、同时把 $m$ 开够时给出。

```mermaid
flowchart LR
  MB["Mini-batch"] --> SPLIT["切成 m 个微批"]
  SPLIT --> C1["Cell 1"]
  C1 --> C2["Cell 2"]
  C2 --> CK["Cell k"]
  CK --> LOSS["损失"]
  LOSS --> B["反向按微批逆序传回"]
  B --> UPD["累积 m 份梯度后一次更新"]
```

### 同步更新为何值得付气泡税

若每个微批立刻更新权重，管道里会同时存在多版参数，前向与反向错位，收敛变成异步 SGD。GPipe 选择等 $m$ 个微批的梯度到齐再更新：气泡与激活缓存是代价，换来与单卡大 batch 相同的优化器语义（在忽略 Dropout 噪声粒度、BN 统计等细节的意义上）。对 2019 年的 ImageNet 与 NMT，这个选择让他们能直接用已有超参直觉。对后来的千卡 LLM，气泡税在 $k$ 到 35、64 时更痛，于是有 PipeDream 的交错调度与 Megatron 的 1F1B Flush——那些是下一篇文章，不是 GPipe 正文声称已解决的。

## 机制

流水线能加速，是因为不同 cell 在同一时刻处理 **不同微批**：cell $i$ 在算微批 $t$ 的前向时，cell $i-1$ 可能已开始微批 $t+1$。依赖边只在相邻 cell 的同微批上。重计算改变的是 cell 内部的内存时间曲线：少存激活 → 反向多一次前向 FLOPs。当 $m$ 很大时，重计算几乎是必须，否则 $m$ 份激活把切分省下的权重大吃回去。

负载均衡决定真实气泡：若最后一段 Transformer 更宽、或 AmoebaNet 某 cell 更重，管道被最慢段钉住，$O((k-1)/(m+k-1))$ 只是均匀假设。切分算法需要按 profile 对齐每段时间，而不是按层数均分。BN 在微批上的统计与全 batch 不同；解码器 LM 通常不用 BN，这条在 LLM 上不痛，在 ImageNet 上要小心。

<span class="marginnote">GPipe 的「几乎线性加速」以 $m$ 足够大为条件。$m=k$ 时气泡仍显著。不要只抄 84.4% 或 6B 翻译，却把 $m$ 设成 2。NeurIPS 正式版与 arXiv v5（2019-07）是同一工作；引用用 1811.06965 即可。</span>

### 和 PipeDream、Megatron PP

PipeDream 用 1F1B 与权重暂存减空转、改激活驻留，并允许阶段内再数据并行。Megatron 后来采用的 PipeDream-Flush 在每个全局 batch 仍同步更新，但稳定段交替前向与反向，激活从 $O(m)$ 收到约 $O(k)$。GPipe 是同步流水线的语义起点：先全部前向再全部反向。实现库（Lingvo 等当时栈）与今天的 Megatron Core `ScheduleGPipe` 同名同调度，数值细节以各自代码为准。

## 边界与工程取舍

只能切「序列化」的层堆。MoE 的专家分支、encoder-decoder 交叉注意力的不规则依赖，要额外切缝。重计算与 $m$ 同时开时，step 时间可能通信盖不住。GPipe 不管张量并行：单层仍然完整，超宽 FFN 仍可能单卡放不下，必须叠 [Megatron-LM](/llm/megatron-lm)。两条实验的 SOTA 是 2018–2019 视觉/NMT，不是现代 MMLU。

不要把 GPipe 写成异步。不要把 AmoebaNet 84.4% 安到 Transformer LM 的预训练质量上。许可与开源形态以 Google 当时发布为准，论文给的是算法与实验。

<span class="marginnote">真实编号：Huang 等 *GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism*，arXiv:1811.06965。不要给「batch splitting」单独伪造编号。PipeDream 是 1806.03377 / SOSP 2019，时间上预印本甚至更早，但是不同调度。</span>

## 小结

- GPipe 用层序列分区 + 微批流水线做任务无关的模型并行，同步累积梯度。
- 气泡约 $(k-1)/(m+k-1)$ 量级；激活默认 $\propto m$，靠重计算换内存。
- 实验：5.57 亿 AmoebaNet ImageNet 84.4%；6B 多语 Transformer。
- 出处：Huang 等，arXiv:1811.06965，NeurIPS 2019。
