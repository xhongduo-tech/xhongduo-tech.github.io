---
title: 标准 MHA 的瓶颈：KV Cache 显存占用分析
date: 2026-08-07
---

# 标准 MHA 的瓶颈：KV Cache 显存占用分析

<div class="epigraph">
<p>每一枚 token 都要为它看过的历史付账。</p>
<footer>—— 推理工程谚语（化用）</footer>
</div>

<div class="article-byline">
<p>第四级 · 大模型原理 ｜ 推理系统综述 / vLLM 论文 ｜ 2026-08-07</p>
</div>

## 为什么 KV Cache 是大模型推理的"头号显存黑洞"

自回归推理每生成一个 token，就要把新位置的 K、V 向量算出来，并**缓存全部历史**，供后续 token 的注意力使用——这份缓存就是 **KV Cache**。它不随「参数规模」占显存，而随「序列长度 × 批量大小」线性增长，是长上下文时代最直接的显存压力源。<span class="marginnote">KV Cache 的本质是「用显存换时间」：缓存历史 K/V，省掉每个新 token 重算全部历史注意力的 $O(L^2)$ 计算。它是推理加速的核心，但代价是显存随长度线性膨胀——理解它的开销公式，是理解 MQA/GQA/MLA 等一系列变体的出发点。</span>

## 1 KV Cache 到底占多少显存

设批量 $B$、序列长度 $L$、注意力头数 $H$、头维度 $d_k$、层数 $N$，每层存一份 K 和一份 V（各 $B \times L \times H \times d_k$），KV Cache 的总元素数为：

$$
\text{KV 元素数} = 2 \cdot B \cdot L \cdot N \cdot H \cdot d_k
$$

以 LLaMA-7B 为例：$N=32$ 层、$H=32$ 头、$d_k=128$，$B=1$、$L=4096$：

$$
2 \times 1 \times 4096 \times 32 \times 32 \times 128 = 1.07 \times 10^9 \text{ 个元素}
$$

用 FP16（2 字节）存储，约 **2.1 GB**——对于一个参数约 13 GB（FP16）的模型，KV Cache 占了约 16% 的额外显存。而当长度拉到 32k，KV Cache 膨胀到 16+ GB，**超过模型参数本身**。<span class="marginnote">把数字放一起感受：7B 模型 FP16 权重约 13GB；32k 上下文时 KV Cache 约 17GB——缓存比模型还大。这就是为什么长上下文推理的显存瓶颈根本不是参数，而是 KV Cache。这也是 GQA、MLA、KV 量化等技术存在的原因。</span>

## 2 为什么叫"Cache"：它缓存的是什么

注意力公式里，每个新 query $q_t$ 要和**所有历史 key** 做点积、和**所有历史 value** 加权：

$$
\text{output}_t = \sum_{j=1}^{t} \text{softmax}\left(\frac{q_t \cdot k_j}{\sqrt{d_k}}\right) v_j
$$

如果不缓存，第 $t$ 步要重新计算 $k_1, \ldots, k_{t-1}$ 和 $v_1, \ldots, v_{t-1}$——这些依赖所有历史 token 的前向，每步重算一次，总代价 $O(L^2)$ 甚至更糟。

**KV Cache 的用途**：把第 $t$ 步算出的 $k_t, v_t$ 存进缓存，下第 $t+1$ 步时直接从缓存取历史 K/V，只算新 token 的 Q、K、V。于是：

- 计算量从「每步重算全部历史」降到「每步只算新 token」——总计算 $O(L)$（忽略缓存读取）。
- 代价是显存换时间：缓存线性增长。

**关键洞察**：只缓存 K 和 V，**不缓存 Q**——因为 Q 只在「当前步」被用到（query 是当前生成位置），历史 query 永远不会再被查询。<span class="marginnote">为什么只需要 K/V？注意力里「query」是主动方（我在问），「key/value」是被动方（历史上被查询的内容）。新 token 成为 query 时，要问的是「所有历史回答过什么」（K/V），历史 query 已经用完作废。所以 Q 不必缓存、K/V 必须缓存——这个「不对称」是 KV Cache 的结构根源。</span>

## 3 显存增长的三个维度

KV Cache 的显存随三个维度增长，各自含义不同：

- **序列长度 $L$**：每生成一个 token，KV Cache 增加 $2 \cdot N \cdot H \cdot d_k$ 个元素。长度翻倍，KV 翻倍——这是「长上下文」的直接代价。
- **批量大小 $B$**：并发请求越多，KV Cache 越大。这解释了为什么**批量大小是推理吞吐的关键杠杆**——大 batch 提升吞吐，但显存被 KV 吃满，形成「吞吐 vs 显存」的对抗。
- **层数与头数**：$N \cdot H \cdot d_k$ 是「每 token 的 KV 大小」，架构层面固定。MQA/GQA 压缩的正是这个维度——通过**共享 K/V** 把 $H \cdot d_k$ 缩小。

一张表总结缓解手段与作用维度：

| 手段 | 作用维度 | 原理 |
| --- | --- | --- |
| GQA/MQA | $H$（共享 K/V 头） | 多个 Q 头共享少数 K/V 头 |
| MLA | $H \cdot d_k$（低秩压缩） | 把 K/V 压进潜在空间 |
| KV 量化（INT8/INT4） | 存储字节数 | 用更少字节存缓存 |
| KV 驱逐/回收 | $L$（只留关键 token） | 长上下文里丢弃不重要历史 |
| PagedAttention | 碎片管理 | 按页分配，提高利用率 |

## 4 公式解析：KV Cache 与 batch 的关系

把 KV Cache 显存与「吞吐」直接挂钩。单 token 的 KV 大小为：

$$
S_{\text{kv/token}} = 2 \cdot N \cdot H \cdot d_k \cdot \text{bytes}
$$

设显存预算 $M$、模型权重 $M_{\text{weights}}$、最大批量 $B_{\max}$、平均生成长度 $\bar{L}$：

$$
B_{\max} \approx \frac{M - M_{\text{weights}}}{S_{\text{kv/token}} \cdot \bar{L}}
$$

对这条式子做三步拆解：

- **第一步，读懂分子**：可用显存先扣掉权重，剩下的都是「缓存预算」。
- **第二步，读懂分母**：每个请求占 $S_{\text{kv/token}} \cdot \bar{L}$ 的 KV 显存（随长度增长）。分母越大，能塞的并发请求越少。
- **第三步，读出工程权衡**：**想提高吞吐（大 $B$）就必须压低单 token KV 大小或生成长度**。这就是为什么长文本场景下 GQA/MLA/KV 量化成为刚需——它们在「不变权重」的前提下，硬生生把分母缩小几倍。

**辨析｜易错点：** KV Cache 是**推理期**的概念，训练期没有（训练时一次性前向整段，不需要缓存历史 K/V，直接并行算）。别把「训练显存」与「KV Cache 显存」混为一谈——训练吃的是激活与梯度，推理吃的是 KV Cache。

## 5 KV Cache 的工程管理

- **PagedAttention**（vLLM）：把 KV 按「页」分配，解决**碎片化**——不同请求长度不一，连续分配会留下大量空洞。按页分配像操作系统内存管理，利用率从 ~60% 提到 ~95%。
- **预分配与扩容**：按最大长度预分配会浪费，动态扩容有拷贝成本；折中方案是按「块」增长（如 16 token 一块）。
- **KV 量化的边界**：K 的量化误差比 V 更敏感（K 参与分数计算），通常 K 用更高精度、V 可更低精度——这是 2-bit/4-bit KV cache 方案里常见的「不对称量化」。

## 6 小结

- KV Cache 是**推理加速的核心**，也是**显存最大黑洞**：$2 \cdot B \cdot L \cdot N \cdot H \cdot d_k$。
- 只缓存 **K/V 不缓存 Q**——历史 K/V 被未来 query 使用，历史 Q 已作废。
- 显存随 **$L$、$B$、$N \cdot H \cdot d_k$** 三个维度增长。
- 吞吐 vs 显存的对抗由 KV 主导：**想大 batch，必须压缩 KV**。
- 缓解手段：GQA/MQA、MLA、KV 量化、PagedAttention——下一节开始逐个拆解。

在下一节，我们看第一个激进压缩方案——**MQA（多查询注意力）**：让所有 query 头共享一份 K/V。
