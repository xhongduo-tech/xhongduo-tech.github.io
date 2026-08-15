---
title: FlashAttention：IO 感知的分块精确注意力
date: 2026-08-07
---

# FlashAttention：IO 感知的分块精确注意力

<div class="epigraph">
<p>GPU 上最贵的不是计算，而是搬运。</p>
<footer>—— 推理工程谚语（化用）</footer>
</div>

<div class="article-byline">
<p>第四级 · 大模型原理 ｜ Dao et al. 2022《FlashAttention》 / Dao 2023《FlashAttention-2》 ｜ 2026-08-07</p>
</div>

## 为什么"不改公式、只改 IO"就能提速

FlashAttention 的颠覆性在于：**它不做任何近似，输出与标准注意力逐位相同**——只是重新安排了计算顺序，把「矩阵乘」分解成「分块」，让每个块的乘加都在 GPU 片上 SRAM 里完成，避免反复读写慢速的 HBM（显存）。结果：训练速度 2–4 倍，显存从 $O(L^2)$ 降到 $O(L)$。它重新定义了「高效注意力」——不是少算，而是**少搬**。<span class="marginnote">标准注意力把 $S=QK^\top$（$L^2$ 矩阵）写回显存，softmax 再从显存读，乘 $V$ 又写回——$L^2$ 矩阵被反复搬运。FlashAttention 用「online softmax」技巧，一次遍历内完成 softmax + 加权，$L^2$ 的中间量根本不落盘。GPU 的 SRAM 比 HBM 快约一个数量级，省掉搬运用量就是 2–4 倍加速。</span>

## 1 为什么标准注意力慢：IO 复杂度

注意力计算本身是「算力便宜、搬运昂贵」。设序列 $L$、头维 $d$，标准实现：

1. 算 $S = QK^{\top}$（$L \times L$），写入 HBM；
2. 读 $S$，算 softmax，写回 HBM（两次 $L^2$ 读写）；
3. 读 softmax 结果与 $V$，乘出 $O$，写回 HBM。

HBM 读写量 $O(L^2 + Ld)$——**$L^2$ 项来自中间注意力矩阵的落盘**。当 $L$ 很大（如 32k），$L^2$ 矩阵比模型权重还大，读写成了绝对瓶颈。

FlashAttention 的观察：**矩阵乘本来就能分块**，softmax 也可以通过「在线更新」分块完成。把三个步骤合进一次「循环 + 片上累积」，$L^2$ 矩阵从头到尾不离开 SRAM——**IO 复杂度从 $O(L^2)$ 降到 $O(L)$**（只读写 Q/K/V/O 本身）。

**复杂度对比**（$L=4096, d=64$）：标准实现 HBM 访问约 $2L^2 + 2Ld \approx 33.6M$ 元素；FlashAttention 约 $4Ld + L^2/N_{\text{block}} \cdot \text{小项}$——主要只剩 $4Ld \approx 1M$，差一个数量级以上。<span class="marginnote">关键概念「IO 复杂度」：衡量算法在「慢速内存与快速内存之间搬了多少数据」。传统复杂度分析只数浮点运算，而 GPU 上内存搬运往往更贵。FlashAttention 是「IO-aware 算法」的教科书——把复杂度定义从 FLOPs 换成「访存量」。</span>

## 2 核心技巧：分块 + Online Softmax

FlashAttention 的三个组件：

**① 分块（tiling）**：把 Q、K、V 切成块（如 $64 \times 64$），一次只处理一个块，块内做完整的「$QK^\top$ + softmax + 乘 V」。因为块小，中间量能装进 SRAM。

**② Online softmax（在线 softmax）**：标准 softmax 需要「先扫一遍求最大值和指数和，再算概率」——两遍。分块后块与块之间无法先整体统计，于是用**增量更新**：每处理一个块，用新块的局部最大值更新全局最大值，再用「重缩放」修正已累积的输出。

**③ 重计算（recomputation）**：反向传播时不存中间注意力矩阵，而是**重算一遍前向**（用 K/V 重新算 softmax）。用一点计算换大量显存——这就是显存从 $O(L^2)$ 降到 $O(L)$ 的来源。

在线 softmax 的增量公式：

$$
m_{t} = \max(m_{t-1}, m^{\text{new}}), \qquad
\ell_t = \ell_{t-1} \cdot e^{m_{t-1} - m_t} + \ell^{\text{new}}, \qquad
O_t = O_{t-1} \cdot e^{m_{t-1} - m_t} + O^{\text{new}}
$$

每处理一个新块，用「旧最大 vs 新最大」的差值重缩放之前累积的 $\ell$ 与 $O$——保证最终结果与全局 softmax **逐位一致**。

## 3 公式解析：Online Softmax 的等价性

验证在线 softmax 为什么与全局 softmax 一致。设已处理的分数为 $s_1, \ldots, s_{t-1}$，最大值为 $m_{t-1}$，指数和为 $\ell_{t-1}$；新块分数 $s^{\text{new}}$，最大值 $m^{\text{new}}$。

对这条式子做三步拆解：

- **第一步，理解最大值更新**：$m_t = \max(m_{t-1}, m^{\text{new}})$——全局最大是「历史最大」与「新块最大」的较大者，正确。
- **第二步，理解指数和的修正**：$\ell_t = \ell_{t-1} \cdot e^{m_{t-1} - m_t} + \ell^{\text{new}}$。若 $m_t = m_{t-1}$（历史最大更大），则修正因子 $e^{0}=1$，直接加新块；若 $m_t = m^{\text{new}}$（新块更大），则历史项按 $e^{m_{t-1} - m_t}$ **缩小**——把历史指数和「对齐」到新的更大最大值。数学上 $\sum_i e^{s_i - m_t} = \left(\sum_i e^{s_i - m_{t-1}}\right) e^{m_{t-1} - m_t}$，恒等成立。
- **第三步，读出输出等价**：$O$ 的更新同理——输出按同样的因子重缩放。最终所有块处理完，$O$ 就是「以全局最大值为基准」的完整加权和，除以全局 $\ell$ 即标准 softmax 结果。**逐位一致，无近似**。

**辨析｜易错点：** FlashAttention 的「近似」名声是误解。它**不近似 softmax**——只是改变了计算顺序与数值组织（用在线最大重缩放避免数值溢出），数学上等价。真正的近似注意力（线性注意力、低秩近似）是另一族，别把「IO 优化」与「数学近似」混为一谈。

## 4 FlashAttention-2 与后续

**FlashAttention-2**（2023）的改进：减少非矩阵乘开销（重缩放、掩码在循环外）、更好的并行策略（按序列维分块，而非 batch×head），在 A100 上达到接近理论峰值——**注意力「不可能再快多少」**。

**FlashAttention-3**：针对 Hopper 架构的异步分块与低精度（FP8）。

**生态影响**：

- PyTorch 的 `scaled_dot_product_attention`（SDPA）内置 FlashAttention 路径，`flash-attn` 库被 vLLM、HuggingFace 广泛集成。
- 它让「稠密注意力 + 长上下文」重新可行——**这是稀疏注意力被边缘化的直接原因**。
- 衍生工作：Flash-Decoding（长序列推理）、PagedAttention（KV 管理）。

## 5 FlashAttention 为什么"统治"了长上下文

| 方案 | 数学精度 | 复杂度 | GPU 友好 | 现状 |
| --- | --- | --- | --- | --- |
| 标准注意力 | 精确 | $O(L^2)$ 计算 + $O(L^2)$ 内存 | 一般 | 基线 |
| 稀疏注意力 | 精确（结构性） | $O(L)$ 计算 | 差（不规则访存） | 冷门 |
| FlashAttention | **精确** | $O(L^2)$ 计算 + **$O(L)$ 内存** | 极好（规则分块） | 主流 |

FlashAttention 的「统治」在于：它**保留了稠密注意力的全部能力**（无先验、无近似），只是把「昂贵的 $L^2$ 中间量」从显存挪到片上——**同样的数学，更省的搬运用量**。长上下文由此不再是「内存装不下」，而是「算力够不够」，把问题回归到计算本身。

## 6 术语速查表

| 术语 | 英文 | 一句话定义 |
| --- | --- | --- |
| FlashAttention | FlashAttention | IO 感知的分块精确注意力 |
| IO 复杂度 | I/O complexity | 内存搬运量的度量 |
| HBM | HBM | 高带宽显存（较慢） |
| SRAM | SRAM | GPU 片上缓存（较快） |
| Online softmax | online softmax | 增量更新的 softmax |
| 重计算 | recomputation | 反向时重算前向以省显存 |

## 7 数值算例：一张注意力矩阵占多少显存

$L=4096$ 的注意力分数矩阵 $S$，用 FP16 存储需要 $4096 \times 4096 \times 2 \approx 32$ MB。一个 32 层的模型，若每层都「写回再读回」$S$，仅这一项就是几十 GB 的 HBM 流量；而 A100 的 SRAM 只有约 192 KB——**标准实现就是让一个「每层 32 MB」的中间量反复进出「192 KB」的片上缓存**。

FlashAttention 把计算切成 $64\times64$ 的块：每个块只需「块大小的中间量」常驻 SRAM，最终只把 Q/K/V/O 本身写回 HBM——HBM 访问量从 $O(L^2)$ 降到 $O(L)$，这就是「显存少一个数量级、速度提 2–4 倍」的物理来源。

**辨析｜易错点：** FlashAttention 提速的「甜点区」在长序列。$L$ 较小时（如 128），标准实现已经能装进 SRAM，FlashAttention 的分块与重算开销反而可能占优不明显。**「闪速注意力」的收益随长度放大**——这是它被称作「长上下文基础设施」的原因。

## 8 从训练到推理的完整家族

| 组件 | 解决的问题 | 场景 |
| --- | --- | --- |
| FlashAttention | 训练激活显存、吞吐 | 预训练、微调 |
| Flash-Decoding | 长序列解码吞吐 | 长上下文推理 |
| PagedAttention | KV 碎片与换页 | 高并发推理 |
| Chunked Prefill | 预填充与解码混合 | 连续批推理 |

FlashAttention 不是孤立点，而是一族「IO 感知」工程的开端：训练、推理、KV 管理各有一块。理解它的核心思想（**少搬运、不近似**），就能顺藤摸瓜理解整个高效注意力生态。

## 9 小结

- FlashAttention 是 **IO 感知**的分块精确注意力：**不改公式，只改数据搬运**。
- 标准注意力的瓶颈是 **$L^2$ 中间矩阵反复读写 HBM**；FlashAttention 用分块 + 在线 softmax 让它留在 SRAM。
- **Online softmax** 用「最大重缩放」实现逐位一致的增量 softmax——**无近似**。
- 反向用**重计算**换显存：内存 $O(L^2) \to O(L)$。
- FlashAttention-2/3 优化并行与低精度，PyTorch 内置，是长上下文的主流底座。

在下一节，我们看一个更激进的「替代注意力」方向——**线性注意力与状态空间模型**：Mamba 能否真的取代注意力？
