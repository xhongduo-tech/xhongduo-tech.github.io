---
title: Megatron-LM
date: 2026-09-07
section: llm
---

# Megatron-LM

<div class="epigraph">
    <p>不必新编译器：在 Transformer 的 MLP 与多头注意力里插入成对的 All-Reduce，就能把一层矩阵切到多卡，并且让非线性留在本地。</p>
    <footer>—— Shoeybi 等，Megatron-LM，arXiv:1909.08053</footer>
</div>

NVIDIA 的 **Megatron-LM**（Shoeybi、Patwary、Puri 等，2019）把「单卡放不下数十亿参数 Transformer」收成一套 **层内模型并行**：列并行线性接行并行线性，前向、反向各一次 All-Reduce。它刻意不走 Mesh-TensorFlow 式新语言，也不走 [GPipe](/llm/gpipe) 那种按层切流水线——后两者当时要改图或改优化器。实现是在 PyTorch 里加几个 `autograd.Function`。本篇写 1909.08053 的切法、8.3B GPT-2 级与 3.9B BERT 级实验；2021 年把流水线加进来的 *Efficient Large-Scale Language Model Training on GPU Clusters*（Narayanan 等，arXiv:2104.04473）以及与 DeepSpeed 的结合见 [Megatron-DeepSpeed](/llm/megatron-deepspeed) 与 [3D 并行](/llm/nd-parallel)。

## 问题

2019 年前后 BERT/GPT-2 已经把「整网放进一张 V100」逼到极限：Adam 每个参数还要动量与方差，混合精度再加一份主权重。数据并行只复制模型，解决不了单卡容量。流水线并行把不同层放到不同卡，**每一层仍然完整**，单层 GEMM 与激活照样可能爆。Mesh-TensorFlow 能切张量，但要新编译栈。需要一种只改 Transformer 里几处矩阵乘、通信模式对 GPU All-Reduce 友好的切法。

BERT 放大时还有第二条问题：原版 Post-LN 残差在深度增加后下游任务会退化。论文用重排 LayerNorm 与残差的位置，使 BERT 从 Base 扩到 3.9B 时开发集指标单调变好。这与张量并行正交，但是同一份代码库里一起交的结果。

### 为何列切第一层、行切第二层

MLP：$Y=\mathrm{GeLU}(XA)$。若按行切 $A$、按列切 $X$，非线性前必须把分片加起来，因为 $\mathrm{GeLU}(u+v)\neq\mathrm{GeLU}(u)+\mathrm{GeLU}(v)$。改成 **列切** $A=[A_1\,A_2]$，则

$$
[Y_1,Y_2]=[\mathrm{GeLU}(XA_1),\,\mathrm{GeLU}(XA_2)]
$$

非线性完全局部。第二层按 **行** 切，局部 GEMM 之后对输出 **All-Reduce 求和**，数学上恢复 $Y=XAB$。前向一次归约（算子 $g$），反向在第一层输入侧再一次（算子 $f$，$f$ 前向是恒等、反向 All-Reduce）。注意力按头切开：QKV 列并行，softmax 局部，输出投影行并行再 All-Reduce。词表嵌入也可沿词表维切。

<span class="marginnote">字段后来通称 [张量并行](/llm/tensor-parallel)（TP）。2019 论文用语是 intra-layer model parallelism。$f$ 与 $g$ 共轭：一个前向通信、一个反向通信。实现上几十行 PyTorch 即可，不需要自定义 CUDA 算子才能正确——自定义算子是后来为速度，不是为语义。</span>

## 方法

弱扩展：每卡约 10 亿参数，模型并行最多 8 路（DGX-2H 节点内 NVLink）。1.2B 单卡基线约 39 TFLOPs（峰值的 30%）；8.3B 在 512 GPU、8 路模型并行上维持 15.1 PFLOPs，相对单卡基线约 **76%** 扩展效率。模型+数据并行是另一条曲线。GPT-2 式 8.3B 在 WikiText-103 测试困惑度 10.8、LAMBADA 66.5%；BERT 式 3.9B 在 RACE 90.9%（均为论文声称相对当时 SOTA）。代码开源在 `NVIDIA/Megatron-LM`。

通信必须留在高带宽域：每层常数次 All-Reduce，体积随激活 $b\times s\times d$ 走。跨节点做 8 路以上 TP 会把 NVLink 假设打破，这是后来 3D 并行把 TP 钉在节点内、用 PP 跨节点的原因，不是 2019 文的实验设置。

```mermaid
flowchart LR
  X["完整激活 X"] --> COL["列并行 GEMM"]
  COL --> NL["局部 GeLU / 局部 softmax"]
  NL --> ROW["行并行 GEMM"]
  ROW --> AR["All-Reduce"]
  AR --> Y["完整 Y"]
```

### BERT 的 LayerNorm 位置

GPT-2 风格把 LN 放在子层入口（Pre-LN）更稳。BERT 放大时若仍用原配方，论文观察到性能不随宽度单调。他们调整 LN 与残差的相对位置，使大 BERT 的下游分重新随规模上升。这是优化稳定性，不是 TP 的数学条件；但若有人把「Megatron BERT」理解成只切矩阵、不改 LN，会复现出那条退化曲线。

## 机制

TP 的正确性来自线性可加：列切把输出通道分区，逐元素非线性与按头的 softmax 都不需要通道间即时求和；行切把内积拆成部分和，唯一需要全局的是一次求和。残差、Dropout、LN 仍作用在 All-Reduce 之后的完整向量上（2019 设定），所以 LN 的均值方差是全局的——这也是后来 **序列并行**（Korthikanti 等，arXiv:2205.05198）要把 LN/Dropout 沿序列再切开的原因：激活内存仍在 TP 组上复制了一份序列。2019 文还没有序列并行。

头数必须能被 TP 度整除，否则按头切失败。8 路对应 8 的倍数个头。词表并行把交叉熵变成分片 vocab 上的归约，大词表时嵌入表才能放下。混合精度下 All-Reduce 的数值类型要与单卡基线一致，否则「76% 效率」不可比。

<span class="marginnote">论文强调与 GPipe 正交：可以先 TP 再 PP。不要把 2019 的 8.3B 写成已经用了 1F1B 流水线。那是 2104.04473 与后续 Megatron Core 的故事。</span>

### 和 Mesh-TF、GPipe、ZeRO

Mesh-TensorFlow 更通用、要编译器。GPipe 切深度、用微批填管道，单层不切。[ZeRO](/llm/zero-stages) 切的是数据并行副本上的优化器/梯度/参数，不切层内 GEMM。现代预训练三者叠用：节点内 TP（Megatron）、跨节点 PP、跨副本 ZeRO/DP。只读 1909.08053 不够配千卡，但切分语义仍是这一对 $f,g$。

## 边界与工程取舍

原实现是 GeLU 而不是 SwiGLU；门控双上投影要两块一起列切，否则逐元素门不合法，见张量并行专文。GQA/MQA 的 KV 头也要能被 TP 整除或在组内复制。检查点与 TP 度绑定，改 8 路到 4 路要重切权重。单卡 30% 峰值是 2019 年 V100 + 当时内核的基线，不能用来骂今天的 MFU。

WikiText/LAMBADA/RACE 的 SOTA 是 2019–2020 对照，不是 2026 年的语言建模前沿。BERT 3.9B 的 LN 改动不要安到 GPT 解码器上当成必做。

### 激活检查点与「每卡十亿」弱扩展

论文同时使用激活重计算，否则即使权重切开，中间激活仍可能打满 32GB。弱扩展曲线的前提是每卡大约一口 10 亿参数：卡数加倍，模型大约加倍，通信体积与计算同阶增长。若改成强扩展（模型固定、只加卡），TP 度上升会让每卡 GEMM 更瘦、通信占比更高，76% 那条效率不再适用。节点内最多 8 路，是 DGX-2H 的 NVLink 域大小，不是算法上限；今天的 NVSwitch 域可以更大，但跨节点做纯 TP 仍然通常不划算。把 2019 的 8 路抄到 64 路 InfiniBand 上，是最常见的误用。

词表并行与隐藏维 TP 是同一家族的不同轴：大词表时嵌入表必须切，否则「层内切了 MLP、输入端仍复制整张 50K×$d$ 表」。交叉熵要在分片 logits 上做数值稳定的归约，实现错误会表现为只有部分词能学、其余梯度为零。开源 Megatron 后来把这些封装进 `ColumnParallelLinear` / `VocabParallelEmbedding`，语义仍是这篇论文的 $f$ 与 $g$。

<span class="marginnote">真实编号：Shoeybi 等 *Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism*，arXiv:1909.08053。后续集群流水线：Narayanan 等 arXiv:2104.04473。序列并行：Korthikanti 等 arXiv:2205.05198。不要把这三号混成一篇。</span>

## 小结

- Megatron-LM 用层内切分训练数十亿 Transformer：MLP 列并行+行并行，注意力按头切，每层两次 All-Reduce 量级。
- 无需新编译器；与流水线正交；8.3B / 512 GPU 报 76% 扩展效率。
- BERT 放大需处理 LN 位置；GPT 式 8.3B 刷新了当时 WikiText 与 LAMBADA。
- 出处：Shoeybi 等，arXiv:1909.08053，2019；代码 NVIDIA/Megatron-LM。
