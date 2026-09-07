---
title: Token Dispatch / Combine
date: 2026-09-07
section: llm
---

# Token Dispatch / Combine

<div class="epigraph">
<p>把每个块拆成注意力、All-to-All 分发、MLP、All-to-All 合并四段，让前向的分发与反向的合并能和计算重叠。</p>
<footer>—— DeepSeek-AI，DeepSeek-V3 技术报告，arXiv:2412.19437，2024</footer>
</div>

专家并行的通信对象不是梯度，是 **token 激活**。路由给出专家编号之后，必须把各 token 的隐向量送到拥有对应专家的设备，本地算完 FFN，再按原 batch 次序加权写回。前一次集体通信叫 dispatch（分发），后一次叫 combine（合并）。GShard 在 TPU mesh 上把这件事写成沿专家轴的 permute；Switch Transformers 用容量桶把形状钉死；DeepSeek-V3 把一层 Transformer 明确切成「注意力 / dispatch / MLP / combine」四段，好让 DualPipe 把通信藏进计算。本篇只写这两次置换本身：它们搬什么、按什么索引、和 grouped GEMM 如何咬合。路由公式与负载损失见别篇。

## 问题

稠密 FFN 的输入已经按 batch 排好，一张卡本地 GEMM 即可。MoE 把 FFN 换成 $N$ 个专家，每个 token 只去 $k$ 个。若专家按设备切开，本地 batch 里「该去专家 $e$ 的那些行」是稀疏、乱序、长度不齐的。不重排，专家权重对着错误的行做乘；重排之后若不记住逆置换，合并时会对错残差位置。问题因此是一次 **带元数据的置换**：payload 是 $d$ 维激活，钥匙是 `(token_id, expert_id, 权重)`。

容量约束让形状更硬。GShard 与 Switch 给每个专家一个槽位数 $c=\mathrm{capacity\_factor}\cdot kT/N$。超容量的 token 在分发前丢掉；不足的槽位用 padding 补齐，以便静态形状的 All-to-All。于是 dispatch 缓冲区的体积由 **槽位** 决定，不是由真实被路由的 token 数决定。容量因子过大，你在为空气付带宽；过小，掉牌伤害质量。MegaBlocks 后来用变长 grouped / 块稀疏 GEMM 取消容量，那是计算侧的解放；通信侧仍要一张「谁去哪张卡」的索引表。

### 分发之前必须已经有路由结果

Dispatch 不是路由。路由在本地用 $W_r x$ 得到 top-$k$ 专家与门控权重；dispatch 只消费这张表。若把二者融进同一个核，容易把「算分数」和「搬字节」的耗时混在一条 NCCL 曲线里。正确的计时应分列：路由 GEMM、索引构建（hist / scan / 排序）、All-to-All、专家 GEMM、回程 All-to-All、按权重累加。

<span class="marginnote">All-to-All 没有求和语义。每张卡发出的第 $i$ 个切片，应完整出现在对端的接收缓冲里。写成 All-Reduce 会把不同 token 搅在一起。DeepSeek-V3 训练核把跨节点 IB 与节点内 NVLink 拆开，仍是置换，只是路径分段。</span>

## 方法

记全局 token 数为 $T$，隐藏维 $d$，每 token 选 $k$ 个专家，参与 EP 的设备数为 $E$。前向四段是：

1. **本地路由**：得到 $T\times k$ 的专家 id 与门控权重。
2. **Dispatch**：按专家所在 rank 把激活重排；跨设备用 All-to-All（或 DeepSeek 式「先 IB 到对端同号 GPU，再 NVLink 转发」）。接收端得到每个本地专家的变长（或定长槽位）batch。
3. **专家计算**：对每个专家做 FFN，通常是 grouped GEMM，而不是 $N$ 次小核启动。
4. **Combine**：按原 token 次序把输出发回，对 $k$ 路结果做加权和，写进残差流。

$$
y_t = \sum_{i=1}^{k} g_{t,i}\, \mathrm{FFN}_{e_{t,i}}(x_t).
$$

加权发生在 combine 之后或之中：门控标量可以在分发前广播到专家侧，乘在专家输出上再传回，以减少回程后再做一次逐元素乘。DeepSeek-V3 为保精度，combine 路径在训练里保持 BF16，而 dispatch 前的上投影输入可以走 FP8。

```mermaid
flowchart LR
  X["本地激活 x"] --> R["路由 top-k"]
  R --> D["Dispatch All-to-All"]
  D --> G["本卡专家 GEMM"]
  G --> C["Combine All-to-All"]
  C --> Y["按 token 加权写回"]
```

### 索引：直方图、前缀和、逆置换

实现上几乎总要先做一次直方图：每个专家分到多少 token。前缀和给出该专家在发送缓冲里的起止偏移。然后按 `(expert_id, token_id)` 把行拷进连续缓冲——这就是 permute。All-to-All 只看见这块连续内存和每对 rank 的计数。Combine 需要 **逆置换**：专家输出按发送时的偏移写回原行。索引可以在 host 上预计算，也可以在 device 上用 scan；decode 逐步、batch 很小的时候，host 预计算的延迟可能高过通信本身。

容量模式下，直方图被截断到 $c$，多出来的 token 不进入发送列表。dropless 模式下，发送计数等于真实路由计数，接收端缓冲区按当前步的最大值动态分配，或按历史高水位预留。后者会浪费显存，但避免每步 `cudaMalloc`。

### 跨节点时分发被拆成两跳

DeepSeek-V3 报告写明：集群节点内 NVLink 约 160 GB/s，跨节点 IB 约 50 GB/s，大约 3.2 倍差。他们限制每个 token 最多发往 $M=4$ 个节点，先 IB 到目标节点上 **与自己同号的 GPU**，再 NVLink 转发到真正持有专家的卡。Dispatch 核用 warp specialization：IB 发送、IB→NVLink 转发、NVLink 接收分给不同 warp，并按负载动态调数量。Combine 反向走 NVLink 发送、转发并累加、IB 接收并累加。这不是换了一种集体原语，而是把同一次置换按拓扑切开，让窄的 IB 只走节点间那一跳。

## 机制

一次双向 All-to-All 的名义流量，对每张卡约为

$$
\mathrm{bytes}\approx 2\cdot k\cdot \frac{T}{E}\cdot d\cdot s,
$$

$s$ 是每元素字节。因子 2 是 dispatch 加 combine。均匀时每卡发出 $kT/E$ 条；热专家所在卡收到远多于这个数，既拖慢 GEMM 也拖慢接收。设备限制路由把一个 token 的目的节点数封顶，从而给 IB 流量一个硬上界——这是 DeepSeek-V2/V3 能把细粒度专家铺到多节点上的系统条件。

计算通信比由专家宽度决定。肥专家的 $d_{\mathrm{ff}}$ 大，本地 GEMM 能盖住置换；细粒度小专家、$k$ 又高时，payload 不小、GEMM 却碎，dispatch/combine 会变成墙。训练里 DualPipe 的解法是把相邻 microbatch 的「注意力」和「dispatch/MLP/combine」错开；推理 decode 里注意力更占时间，DeepSeek-V3 部署节把一个 microbatch 的注意力与另一个的 dispatch+MoE+combine 重叠，并只给后者很少的 SM。

<span class="marginnote">Dispatch 的精度可以低于 Combine。V3 把 MoE 上投影前的激活量化到 FP8 再分发，与 FP8 前向兼容；回程合并保持 BF16，避免残差累加被窄格式打穿。不要把两边都改成同一精度再谈「省了一半带宽」。</span>

### 和 grouped GEMM 的接口

专家计算吃的是 **按专家连续** 的行块，不是原 batch 顺序。Dispatch 的输出布局正是 grouped GEMM 的 $A$：第 $i$ 个问题的 $M_i$ 是该专家收到的 token 数，$N$ 与 $K$ 由 FFN 宽决定。Combine 吃的是 grouped GEMM 的 $D$，再按逆置换打散。所以「token 重排」和「变长 GEMM」是同一条流水的两端：没有前者，后者只能对每个专家单独启动一次核；没有后者，前者凑出来的变长块只能 padding 成稠密 batched GEMM。MegaBlocks 把 MoE 重写成块稀疏 / grouped 运算，前提仍是 permute 已经按专家把 token 聚在一起。

反向再走两遍置换：对专家输出的梯度做一次「按专家聚集」，对专家输入的梯度做一次「按 token 打散」。路由矩阵 $W_r$ 通常不随专家切分，其梯度留在本地。不要把 combine 的加权和当成 All-Reduce——权重是 per-token 的门控，不是跨卡求和。

## 边界与工程取舍

小 batch 自回归 decode 是 dispatch/combine 最痛的工作点：每步 $T$ 只有并发请求数，$kT/E$ 可能小于 1，All-to-All 的启动延迟大于 payload。DeepSeek-V3 解码部署因此把 EP 拉到 320、每卡一个专家，并用 IBGDA 做点对点，而不是套训练时的分层 IB+NVLink 核。Mixtral 8 专家常常整模型复制或只做 TP，避开逐步置换。

不要用容量因子同时当「质量旋钮」和「通信预算」。掉牌发生在 dispatch 之前，评测若在掉牌后的集合上算路由准确率，会低估真实损失。空槽进入 All-to-All 时，NCCL 体积按槽位走，profiler 里的 GB/s 会看起来很高，有效 token 却很少。

索引构建在 decode 上可能比通信还贵。每步做一次全局排序，不如维护一份按专家的计数再线性填充。动态形状导致的图断裂（CUDA graph 录不上）往往出在这里：某步某个专家 $M_i=0$，下一 $M_i$ 暴涨。工程上对 $M_i=0$ 仍保留空问题描述，让 grouped 核跳过，而不是从问题列表里删掉该专家。

<span class="marginnote">检查点与重映射：dispatch 用的是逻辑专家 id。EPLB 一类部署会把热专家复制到多个物理槽位。逻辑 id → 物理 rank 的表必须与当前步的 dispatch 一致；换表而不重编索引，token 会发到上一份放置。</span>

拓扑画错时，同一次置换会同时打满 IB 和 NVLink 的最慢路径。限制每 token 的节点数、把同组专家尽量放同一节点，都是给 dispatch 减跳数，不是改 MoE 公式。调试应同时看每卡接收直方图和集体通信耗时；只看平均 FLOPs 会把热专家卡的等待读成「计算不够」。

## 小结

- Token dispatch / combine 是专家并行的两次激活置换：按专家聚集，算完再按 token 写回。
- 通信量与 $k$、$T/E$、$d$、负载均匀度成正比；容量模式按槽位付带宽，dropless 按真实计数付。
- 索引（直方图、偏移、逆置换）是通信与 grouped GEMM 的共同合同；没有它，专家只是 $N$ 次乱序小 GEMM。
- DeepSeek-V3 把一层拆成注意力 / 分发 / MLP / 合并，并用 IB 然后 NVLink 的两跳实现跨节点置换。
- Decode 小 batch 时启动延迟主导，往往要改通信后端或复制专家，而不是照搬训练 All-to-All。
- 出处：Lepikhin et al.，*GShard*，2020；Fedus et al.，*Switch Transformers*，2021；DeepSeek-AI，*DeepSeek-V3*，arXiv:2412.19437；Gale et al.，*MegaBlocks*，MLSys 2023，arXiv:2211.15841。
