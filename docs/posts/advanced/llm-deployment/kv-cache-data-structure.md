---
title: KV Cache 的原理与数据结构
date: 2026-08-07
---

# KV Cache 的原理与数据结构

<div class="epigraph">
<p>我们提出的 PagedAttention，是一种灵感源自操作系统中经典虚拟内存与分页技术的注意力算法。</p>
<footer>—— 权等人（Woosuk Kwon et al.），《PagedAttention：高效大语言模型服务的内存管理》（Efficient Memory Management for Large Language Model Serving with PagedAttention, SOSP 2023）</footer>
</div>

<div class="article-byline">
<p>第四级 · 大模型部署 ｜ LLM推理引擎体系 第一章 ｜ 2026-08-07</p>
</div>

## 为什么现在才讲 KV Cache

第一节《自回归生成》里我们埋了一个引子：历史 token 的 K、V 每步都不变，缓存它们能把逐 token 注意力从 $O(t^2)$ 降到 $O(t)$。上一节《访存瓶颈》又说：长上下文时 KV Cache 的读取会反超权重，成为 Decode 的新瓶颈。两处都点到为止，现在到了把它彻底展开的时候。

KV Cache 是推理引擎最核心的数据结构，也是 vLLM、SGLang、TensorRT-LLM 之间差异最集中的地方。**理解了它「缓存什么、长成什么样、怎么管理」，你就理解了引擎设计的一半。** 这一节先讲原理与数据布局，管理方式（PagedAttention、块表）留到第三篇。

## 1 缓存的是什么：每一层的 K 与 V

先回到单头注意力（略去缩放因子）：

$$\text{Attention}(Q, K, V) = \mathrm{softmax}\left(QK^\top\right) V$$

生成第 $t$ 个 token 时，新 token 产生一个新的 query $q_t$，它要与**所有历史位置**的 key、value 做注意力。关键事实：$k_1\dots k_{t-1}$ 与 $v_1\dots v_{t-1}$ 在之前每一步已经算过，而且**不会再变**——它们是 prompt 与已生成 token 的函数，历史已定格。

**KV Cache（键值缓存）**：把每一层、每一个注意力头、每一个已处理 token 的 key 与 value 向量存下来，供后续每一步的注意力直接取用，避免重算。缓存的对象是注意力里的 **K 与 V**，而 **Q 从不缓存**——每个新 token 都必须现场算自己的 query。<span class="marginnote">「为什么 Q 不缓存」是个经典面试题。答案：query 是用来跟「当前所有历史」做匹配的当前视角，每个新位置都要全新的 query；而 K、V 是「被匹配的档案」，档案一旦写定就不再变化。缓存档案、不缓存视角。</span>

所以「KV Cache」这个命名是精确的：它缓存的是注意力公式里那两个不需要重算的矩阵。每一层一份、每个 KV 头一份、每个 token 一行。

## 2 一块 KV Cache 长什么样：四维张量

一个序列在**某一层、某个 KV 头**上的缓存，是长度为当前 token 数 $T$ 的两条向量序列——K 一条、V 一条。堆上「层」与「头」两个维度，一个序列的完整 KV Cache 是一个五维张量：

$$\text{KV} \in \mathbb{R}^{\,2 \times L \times H_{kv} \times T \times d_{\text{head}}}$$

- $2$：K 与 V 两份；
- $L$：层数，每层各有独立的注意力，各自缓存一份；
- $H_{kv}$：每层的 **KV 头数**（GQA/MQA 下可能小于 query 头数，见下节）；
- $T$：序列当前长度，**每生成一个 token 就沿这个维度长一行**；
- $d_{\text{head}}$：每个头的向量维度（如 128）。

生成第 $t+1$ 个 token 时，各层为它算出一组新的 $k_{t+1}, v_{t+1}$，**追加**到自己的缓存末尾——这是唯一的新写入；前面的行只读不改。Cache 的体量随 $T$ 线性生长，这正是 KV Cache 显存占用的核心来源。

## 3 头结构：MHA、GQA、MQA 与 KV 大小

注意公式里是 $H_{kv}$（KV 头数），不是 $H_q$（query 头数）。两者是否相等，取决于模型用哪种多头结构：

| 结构 | query 头数 | KV 头数 | 代表模型 | KV/层/token |
| --- | --- | --- | --- | --- |
| **MHA**（标准多头） | $H$ | $H$ | LLaMA-2 7B（32） | 最大 |
| **GQA**（分组查询） | $H$ | $H/g$（组数 $g$） | LLaMA-3 8B（8）、Qwen2.5 | 约为 MHA 的 $1/g$ |
| **MQA**（多查询） | $H$ | 1 | 部分早期模型 | MHA 的 $1/H$ |

GQA 的核心是：注意力计算时，一组 query 头**共享同一个 KV 头**。以 LLaMA-3 8B 为例，32 个 query 头分成 8 组，每组 4 个头共享 1 个 KV 头，于是 $H_{kv}=8$。这直接把 KV Cache 压到 MHA 的 $1/4$，且注意力质量几乎无损——**它是「省 KV」与「保质量」之间最成功的工程折中**，如今新模型几乎标配 GQA。<span class="marginnote">为什么能共享？直觉是：key、value 表达的是「这段上下文里有什么信息」，相邻的注意力头观察的是同一段上下文的相近侧面，可以把它们要用的档案合并成一份。而 query 是每个头各自的「关注角度」，不能合并——所以只压缩 KV，不压缩 Q。MQA 是把折中推到极端（$H_{kv}=1$），牺牲一些质量换最小 KV。</span>

代入数字（FP16，$d_{\text{head}}=128$）：LLaMA-2 7B（MHA）每 token 每序列约 **512 KB**；LLaMA-3 8B（GQA）约 **128 KB**；若做成 MQA 只需 **16 KB**。4 倍的 KV 差距，在长上下文、大并发场景下就是「能不能塞进一张卡」的差距。

## 4 公式解析：一个并发服务的 KV 总账

把 KV 占用算成服务级的总账，是容量规划的第一步：

$$M_{\text{KV}} = 2 \cdot L \cdot H_{kv} \cdot d_{\text{head}} \cdot b \cdot T \cdot S$$

其中 $S$ 是**并发序列数**（同时有多少个请求在生成），$b$ 是每元素字节数。逐步拆解：

- **第一步，认前三个因子 $2LH_{kv}$**：K、V 两份，乘层数 $L$，乘每层 KV 头数 $H_{kv}$——这是「一个 token 在全模型里占几个向量」。
- **第二步，乘维度与精度 $d_{\text{head}} b$**：每个向量 $d_{\text{head}}$ 个元素、每元素 $b$ 字节——得到「一个 token 占多少字节」。
- **第三步，乘序列长度 $T$**：每多一个 token 就沿长度方向多一行，所以整条序列的缓存随 $T$ 线性涨。
- **第四步，乘并发数 $S$**：每个序列各持一份自己的缓存——这是最容易漏掉、也最致命的一项。

用 LLaMA-3 8B（$L=32, H_{kv}=8, d_{\text{head}}=128, b=2$）代入：

$$M_{\text{KV}} = 2 \times 32 \times 8 \times 128 \times 2 = 131{,}072\ \text{B/token/序列} = 128\ \text{KB}$$

再乘上下文 $T=4096$ 与并发 $S=100$：

$$M_{\text{KV}} = 128\ \text{KB} \times 4096 \times 100 \approx 53.7\ \text{GB}$$

而模型权重 FP16 只有约 16 GB。**一个 80 GB 的 A100，KV Cache 一项就吃掉 53.7 GB，权重 16 GB，合计约 70 GB——批量与上下文稍微再大一点就放不下。** 这就是为什么 KV Cache 是显存规划的头号变量，也是 PagedAttention、KV Cache 量化、PD 分离（第八篇）全部故事的起点。

## 5 数据结构与内存管理：连续大块 vs 页式分块

缓存的数据结构决定了显存怎么分、浪费多少。主流方案有两派。

**连续大块（contiguous buffer）**：为每个序列预分配一整块长度为「最大可能长度」（如 4096）的连续显存，TensorRT-LLM / FasterTransformer 早期路线。实现简单、访存局部性好；代价是**内部碎片**——序列只生成了 50 个 token，也要占满 4096 的坑。vLLM 论文实测，传统实现里 KV 显存的真实利用率只有约 20%–38%。<span class="marginnote">预分配「按最大值」是一种经典的悲观策略：宁可占着不用，也不在生成途中重新分配——因为 Decode 每步都在追加，中途 malloc 会引入不可控延迟。这正是它浪费的根源。</span>

**页式分块（paged blocks）**：vLLM 的思路——把 KV 显存切成**固定大小的块**（默认 16 个 token 一块），每个序列的 KV 按需申请、逐块生长，物理块不必连续。每个序列维护一张**块表（block table）**，记录「逻辑位置 → 物理块」的映射，就像操作系统的页表。<span class="marginnote">这就是引语里那句话：vLLM 把 OS 的「虚拟内存分页」搬进了 GPU。块 = 页，token = 字节，序列 = 进程，块表 = 页表。内存浪费从 60%–80% 降到 4% 以下（只剩每序列最后一块可能不满）。代价是每步注意力多了约 5% 的块表间接寻址开销，且必须写专用 CUDA kernel。</span> 关于块表的细节与 Copy-on-Write 共享，是第三篇《PagedAttention》与《块表》两节的主题。

**辨析｜易错点：** 四个高频误解：

**误区一：以为 KV Cache 把 Q 也缓存了。** 不缓存 Q。每个新 token 现场算自己的 query，缓存只含 K、V。

**误区二：以为 KV 是模型里的小头。** 在短上下文、低并发时它确实小；但在 $T=4096$、$S=100$ 时它比权重还大 3 倍以上。规划显存时必须把它单独建账。

**误区三：混淆 GQA 的两种头数。** GQA 减少的是 **KV 头数**，query 头数不变——query 头 32、KV 头 8 是正常的，不是笔误。

**误区四：以为 KV Cache 跨序列共享。** 默认每序列各持一份；只有共享相同前缀的请求才按块共享（Copy-on-Write，第三篇），一般场景要按「每序列一份」做预算。

## 6 小结

- **缓存对象**：每层、每个 KV 头、每个 token 的 **K 与 V**；Q 从不缓存，每步现场生成。
- **张量形状**：$2 \times L \times H_{kv} \times T \times d_{\text{head}}$，每生成一个 token 沿 $T$ 维度增长一行。
- **头结构决定体量**：MHA 最贵、GQA 折中（LLaMA-3 8B 比 LLaMA-2 7B 少 4 倍）、MQA 最省。
- **总账公式**：$M_{\text{KV}} = 2LH_{kv}d_{\text{head}}b\,T\,S$——LLaMA-3 8B 在 $T=4096$、$S=100$ 时约 **53.7 GB**，超过权重。
- **两种管理**：连续大块简单但碎片严重（真实利用率 20%–38%）；页式分块按需分配、浪费 <4%，vLLM 由此起家。

在下一节，我们进入第二篇「推理引擎总览」，先回答一个更根本的问题：一个 LLM 推理引擎到底要解决哪些问题、为什么社区会演化出 vLLM、SGLang、TensorRT-LLM 这么多条技术路线。
