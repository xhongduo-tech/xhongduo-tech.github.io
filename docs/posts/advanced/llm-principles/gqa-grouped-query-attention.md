---
title: GQA（分组查询注意力）：质量与效率的折中
date: 2026-08-07
---

# GQA（分组查询注意力）：质量与效率的折中

<div class="epigraph">
<p>既不要独享，也不要共产——分组协作。</p>
<footer>—— 推理工程谚语（化用）</footer>
</div>

<div class="article-byline">
<p>第四级 · 大模型原理 ｜ Ainslie et al. 2023《GQA: Training Generalized Multi-Query Transformer》 ｜ 2026-08-07</p>
</div>

## 为什么 MQA 之后还要 GQA

MQA 把 KV Cache 压到极限，但代价是「单一 K/V 头」可能成为多视角检索的瓶颈。GQA（Grouped-Query Attention）走中间路线：**把 $H$ 个 query 头分成 $g$ 组，每组共享一份 K/V**。$g=1$ 退化成 MQA，$g=H$ 恢复成 MHA——GQA 是这条光谱上的「可调旋钮」。<span class="marginnote">GQA 出自 Ainslie et al. 2023，最初是「MQA 的通用化」（论文标题就直说：训练泛化的多查询 Transformer）。它证明了一个关键事实：<strong>共享 K/V 的收益主要来自「减少头数」，而质量损失可以通过「别压太狠」来控制</strong>——所以 LLaMA-2/3、Mistral、Qwen 全选了 GQA。</span>

## 1 结构：分组共享

设共有 $H$ 个 query 头、$g$ 个 KV 头，且 $g$ 整除 $H$。第 $h$ 个 query 头属于第 $\lfloor h \cdot g / H \rfloor$ 个 KV 组：

组内共享：第 $h$ 个 query 头与同组唯一的 K/V 计算注意力。
组间独立：不同组有不同的 K/V。

配置举例（LLaMA-2-70B）：$H=64$ 头、$g=8$ 个 KV 头——每个 KV 头被 8 个 query 头共享。KV Cache 相对 MHA 缩小 $g/H = 8/64 = 1/8$。<span class="marginnote">HuggingFace 配置里用 $H=64$ 表示 $g$：$g=8$ 就是 8 组。MQA 是 $g/H = 8/64 = 1/8$ 的特例，MHA 是 $g$ 的特例——<strong>同一个参数覆盖整个光谱</strong>。</span>

## 2 为什么 GQA 是"甜蜜点"

GQA 之所以成为主流，是因为它在两个方向上都「够用」：

**质量方向**：$g$ 个 KV 头保留了「多视角检索」的能力。复杂任务（长文档综合、多步推理）需要不同头从历史里找不同信息——$g=8$ 时这种能力仍在，而 MQA（$g=1$）可能退化。

**效率方向**：KV Cache 缩小 $g/H$ 倍。$H=32, g=8$ 时缩小 4 倍——对长上下文与高并发来说，这是「质变级」的显存节省。

一张表看 $g$ 的取值谱：

| $g$ | KV 缩小倍数 | 质量 | 典型用途 |
| --- | --- | --- | --- |
| $H$（MHA） | 1× | 基线 | 原版架构 |
| $8$（GQA-8） | $H/8$× | 近基线 | LLaMA-2/3、Mistral、Qwen |
| $4$ | $H/4$× | 略降 | 轻量模型 |
| $1$（MQA） | $H$× | 明显降 | 极限压缩场景 |

**经验规律**：$g$ 取 8 左右能在「几乎无损」的前提下拿到大部分压缩收益；再压到 4 或更低，质量才开始肉眼可见地下降。<span class="marginnote">这个「8 头甜点」不是巧合：8 个 K/V 头足以覆盖「近/中/远、语法/指代/语义」等主要检索模式，而 32 个头里很多是冗余的。GQA 的贡献正在于揭示了「头数里的冗余」，并把压缩收益稳定落地。</span>

## 3 训练策略：UP-training 与从 MHA 转换

GQA 模型的来源有两种：

**① 直接训练 GQA**：从零预训练时就用 GQA 架构（LLaMA-2 的做法）。简单直接，训练效率高。

**② UP-training（从 MHA 升级）**：已有 MHA 模型想转成 GQA，可以**均值池化** K/V 头——把 $H$ 个 K/V 头按组平均成一个：

$$
W^K_g = \frac{1}{|G_g|}\sum_{h \in G_g} W^K_h, \qquad W^V_g = \frac{1}{|G_g|}\sum_{h \in G_g} W^V_h
$$

即把同组的 K/V 投影矩阵「求平均」作为共享头，再用少量数据继续训练微调（如 T5、GPT-J 的实验）。**均值池化是一个不错的初始化**：因为同组头的 K/V 在训练中已经「协同演化」，平均后损失可控。

为什么能这样转换？**MHA 里不同头的 K/V 本就有冗余**——UP-training 把冗余「求掉」，得到更紧凑的共享表示。<span class="marginnote">UP-training 的意义在于「抢救存量模型」：不想重训一个 MHA 大模型，就把它 GQA 化并轻量续训，省下大部分成本。实践中不少开源项目用这条路把旧模型「改造」成高效推理版。</span>

## 4 公式解析：GQA 的共享-广播结构

设第 $g$ 个 KV 头对应 query 组 $\mathcal{G}_g$，GQA 的第 $h$ 个头（$h \in \mathcal{G}_g$）输出：

$$
\text{head}_h(x) = \text{softmax}\left(\frac{(xW^Q)_h \cdot (xW^K)_g^{\top}}{\sqrt{d_k}}\right) (xW^V)_g
$$

对这条式子做四步拆解：

- **第一步，读懂下标**：Q 用 $h$（自己的头），K/V 用 $g$（所在组的共享头）——**这是与 MHA 唯一的语法差别**。
- **第二步，理解广播**：同组所有 query 头用**同一个** $K_g$、$V_g$，但分数不同（因为 $Q_h$ 不同）——「问题各异、被问内容相同」。
- **第三步，读出缓存**：KV Cache 只存 $g$ 组：每 token $2 \cdot g \cdot d_k$ 元素，比 MHA 的 $2H d_k$ 缩小 $g/H$ 倍。
- **第四步，读出边界**：$g=1$ 时回到 MQA；$g=H$ 时回到 MHA。**GQA 是两者之间的连续插值**。

**辨析｜易错点：** 别把「$g$ 个 KV 头」理解成「$g$ 个 query 头」。$g$ 是 K/V 的数量，query 头数 $H$ 不变。GQA 压缩的是 **K/V 侧**，query 侧始终保留全部 $H$ 头——「多视角查询」的视角仍在。

## 5 部署中的 GQA

- **显存预算**：同样 batch、长度，GQA-8 的 KV 显存是 MHA 的 $1/8$——同样的显存能装 8 倍并发请求，或支持 8 倍长上下文。
- **带宽**：解码每步读 KV 的量缩小 $g/H$ 倍，prefill/decoding 两阶段都受益。
- **与量化叠加**：GQA + KV 量化可再压几倍，是长上下文服务的「组合拳」。
- **与 MHA 的兼容**：推理引擎（vLLM、TensorRT-LLM）对 GQA 均有通用实现，改配置即用。

## 6 术语速查表

| 术语 | 英文 | 一句话定义 |
| --- | --- | --- |
| GQA | grouped-query attention | 分组共享 K/V 的注意力 |
| MQA | multi-query attention | 单一 K/V 头的极端压缩 |
| MHA | multi-head attention | 每头独立 K/V 的原始形式 |
| KV Cache | key-value cache | 推理时缓存的 K/V 张量 |
| UP-training | upcycling | 从 MHA 均值池化转成 GQA |
| 广播 | broadcast | 同组 query 共享一份 K/V |

## 7 数值算例：一份 KV 显存账

设 $H=64$、$d_k=128$、$g=8$、上下文 $L=8192$、FP16：

- **MHA**：每 token 每层缓存 $2 \times 64 \times 128 \times 2$ 字节 ≈ 32 KB；8192 token ≈ 256 MB/层；32 层 ≈ **8 GB**。
- **GQA-8**：每 token 每层缓存 $2 \times 8 \times 128 \times 2$ 字节 ≈ 4 KB；8192 token ≈ 32 MB/层；32 层 ≈ **1 GB**。

同一个模型、同一段上下文，KV 从 8 GB 降到 1 GB——**多出的 7 GB 可以直接换成「更大的 batch、更长的上下文、或更高的并发」**。这就是 GQA 让「8k+ 上下文、大并发服务」从奢侈品变成日常用品的物理基础。

**辨析｜易错点：** GQA 压缩的是**每 token 的 KV 大小**，不改变「注意力计算量」——query 侧仍是 $H$ 个头、仍是 $O(L^2)$ 的分数计算。GQA 管「存储与带宽」，FlashAttention 管「计算与访存」，两者是正交的优化维度，常叠加使用。

## 8 MHA / GQA / MQA 一张表

| 维度 | MHA | GQA | MQA |
| --- | --- | --- | --- |
| KV 头数 | $H$ | $g$（$1<g<H$） | 1 |
| 每 token KV 显存 | $2H d_k$ | $2g d_k$ | $2d_k$ |
| 质量 | 基线 | 近基线 | 略降 |
| 代表模型 | 早期 GPT | LLaMA-2/3、Mistral、Qwen | PaLM、部分蒸馏模型 |

三者的关系是「同一旋钮的连续取值」：$g$ 从 $H$ 拧到 1，压缩收益与质量损失同步上升。GQA 的工程智慧在于「**停在 $g \approx 8$ 的甜点**」——拿到大头收益，把质量损失控制在可忽略范围。

## 9 小结

- GQA 把 $H$ 个 query 头分成 $g$ 组，**每组共享一份 K/V**。
- $g=1$ 即 MQA，$g=H$ 即 MHA——GQA 是整个 KV 压缩光谱的旋钮。
- **质量 vs 效率的甜点**：$g \approx 8$ 时几乎无损、KV 缩小 $H/8$ 倍。
- 训练来源：**直接训练**或 **UP-training**（MHA 的 K/V 按组均值池化 + 轻量续训）。
- LLaMA-2/3、Mistral、Qwen 均以 GQA 为标配。

在下一节，我们把 KV 压缩推到「低秩」——**MLA（多头潜在注意力）**：DeepSeek 把 K/V 压进潜在空间，做到极致压缩。
