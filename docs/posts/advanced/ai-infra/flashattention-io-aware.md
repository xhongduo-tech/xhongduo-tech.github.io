---
title: FlashAttention 的 IO 感知设计思想解析
date: 2026-08-07
---

# FlashAttention 的 IO 感知设计思想解析

<div class="epigraph">
<p>随着序列长度增长，Transformer 会变得又慢又耗内存——瓶颈不在运算，而在数据的来回搬运。</p>
<footer>—— 道（Tri Dao）、富（Daniel Y. Fu）、埃尔蒙（Stefano Ermon）、鲁德拉（Atri Rudra）、雷（Christopher Ré），《FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness》，2022</footer>
</div>

<div class="article-byline">
<p>第四级 · AI 基础设施 ｜ AI基础设施技术栈 第一篇 ｜ 2026-08-07</p>
</div>

## 为什么从 FlashAttention 开始

上一课的 Roofline 图给出了一个令人不安的读数：**标准注意力的核心算子 $\mathbf{S} = \mathbf{Q}\mathbf{K}^\top$，算术强度只有头维度 $d$（约 128），远低于 H100 的 ridge point 295**。也就是说，注意力是被内存卡住的——你优化它的矩阵乘指令、给它堆 Tensor Core，都够不着那条算力天花板。FlashAttention 就是冲着这个读数来的：它不去「把注意力算得更快」，而是去「让注意力少搬几趟东西」。

FlashAttention（Dao et al., 2022）是近年最成功的 kernel 优化之一：同样的精度、同样的结果（它是**精确**算法，不是近似），却把注意力的显存占用从 $O(N^2)$ 压到 $O(N)$，把 HBM 访问从 $O(N^2)$ 量级降到 $O(N^2 d^2 / M)$。它背后的思想只有一句：**让注意力在片上 SRAM 里「一次成型」，中间矩阵永不落回 HBM**。这一课把它拆开，你会看到它其实是 Roofline、算子融合、在线 softmax 三样东西的合体。<span class="marginnote">FlashAttention 是「从极限到大模型」主线上的明星：<strong>长上下文（10 万 token）与长序列训练之所以可能，很大程度上归功于把注意力的 O(N²) 显存需求降成 O(N)</strong>。它同时是第四级第一篇所有技巧（内存层次、融合、Roofline）的集大成案例。注意力机制在 Transformer 里的完整位置见第四级《大模型原理》——那里讲「为什么要有注意力」，本课讲「<strong>怎么让注意力跑得快</strong>」。</span>

## 1 标准注意力为什么慢

先看标准的注意力前向，它由三个矩阵运算组成（$Q, K, V$ 均为 $N\times d$）：

$$
\mathbf{S} = \mathbf{Q}\mathbf{K}^\top \in \mathbb{R}^{N\times N}, \qquad
\mathbf{P} = \text{softmax}(\mathbf{S}), \qquad
\mathbf{O} = \mathbf{P}\mathbf{V}
$$

注意力的对象是 $N\times N$ 的矩阵。每一步都必须把结果**完整写回 HBM**，下一步再从 HBM 读回来：

- 算 $\mathbf{S}$：读 $\mathbf{Q},\mathbf{K}$（各 $Nd$），写 $\mathbf{S}$（$N^2$）。
- 算 softmax：读 $\mathbf{S}$（$N^2$），写 $\mathbf{P}$（$N^2$），还要先扫一遍找每行最大值 $m$。
- 算 $\mathbf{O}$：读 $\mathbf{P}$（$N^2$）与 $\mathbf{V}$（$Nd$），写 $\mathbf{O}$（$Nd$）。

于是总 HBM 访问是 $\Theta(Nd + N^2)$——**被 $N^2$ 项主导**，而且这些 $N\times N$ 矩阵被反复读写。更致命的是显存：$\mathbf{S}$ 和 $\mathbf{P}$ 各占 $N^2$，例如 $N=4096,\ d=128$ 时，仅这两块就有约 128 MB，占满一整块小显存；$N=32\text{K}$ 时，单头注意力矩阵就接近 2 GB。<span class="marginnote">这里的「平方灾难」直接联系到第一篇《GPU 内存层次》：<strong>HBM 又大又慢，SRAM 又快又小</strong>。标准注意力把不该进 HBM 的 N×N 中间结果硬塞进 HBM，于是同时吃满了容量（显存不够）与带宽（来回搬）两头苦头。优化方向就一句话：让中间结果别离开片上。</span>

## 2 三个设计：tiling、IO 感知、一个 kernel

FlashAttention 的答案可以拆成三块拼图。

**拼图一：tiling（分块）。** 把 $\mathbf{Q}$ 按行切成块 $\mathbf{Q}_i$（块大小 $B_q$），$\mathbf{K},\mathbf{V}$ 按行切成块 $\mathbf{K}_j,\mathbf{V}_j$（块大小 $B_k$），使得**任意一块都能整个塞进片上 SRAM**。对每个查询块 $\mathbf{Q}_i$，依次遍历所有键块/值块，只算 $\mathbf{Q}_i \mathbf{K}_j^\top$（一个 $B_q \times B_k$ 的小矩阵），而不是全量的 $N\times N$。

**拼图二：IO 感知（IO-aware）。** 每一对块的中间结果 $\mathbf{S}_{ij}$、$\mathbf{P}_{ij}$ 只活在 SRAM 里，**用完即弃，从不写回 HBM**。HBM 里只有进场的 $\mathbf{Q},\mathbf{K},\mathbf{V}$ 和出场的 $\mathbf{O}$，全程不出现任何 $N\times N$ 矩阵。<span class="marginnote">IO 感知的意思是：设计算法时，把「数据在存储层次间搬了多少趟」当成与「算了多少次」同等重要的成本来优化。这正对应上一篇 Roofline 的结论——<strong>注意力是内存瓶颈，所以第一优化目标是减 B（搬运字节数），而不是减 W（运算量）</strong>。FlashAttention 实际多算了一点点，却快了 2–4 倍，就是这个原因。</span>

**拼图三：一个 kernel。** 把「读块 → 算 $\mathbf{S}_{ij}$ → 掩码 → softmax → 算 $\mathbf{O}$ 的增量 → 写 $\mathbf{O}$」全部融进同一个 kernel（这就是上一篇讲的 kernel fusion）。没有多次启动、没有中间结果的 HBM 往返。用伪码把整个前向循环摆出来，你会看到它就是一个「外层遍历查询块、内层遍历键/值块」的二重循环，中间值全在寄存器与 SRAM 里打转：

```python
# FlashAttention 前向（示意）：中间矩阵 S、P 永不写回 HBM
for i in range(0, N, B_q):                 # 外层：查询块 Q_i
    q_i = load(Q[i:i+B_q])                 # 读入 SRAM
    m_i = -inf; l_i = 0; o_i = 0           # 运行中最大值 / 分母 / 输出
    for j in range(0, N, B_k):             # 内层：键/值块 K_j, V_j
        k_j, v_j = load(K[j:j+B_k]), load(V[j:j+B_k])
        s_ij = (q_i @ k_j.T) / sqrt(d)     # B_q×B_k 打分块，只活在此处
        s_ij = mask(s_ij)                  # 因果掩码就地处理
        m_new = max(m_i, rowmax(s_ij))     # 在线 softmax 的三行更新
        o_i = o_i * exp(m_i - m_new) + exp(s_ij - m_new) @ v_j
        l_i = l_i * exp(m_i - m_new) + rowsum(exp(s_ij - m_new))
        m_i = m_new
    store(O[i:i+B_q] = o_i / l_i)          # 归一化后写回 HBM
```

注意内层循环体里，$\mathbf{s}_{ij}$、$\exp(\mathbf{s}_{ij} - m_{\text{new}})$ 这些 $B_q\times B_k$ 的中间矩阵**每一轮都在 SRAM 里被覆盖**，从未落盘。整个 kernel 对 HBM 只做三件事：读 $\mathbf{Q}$ 一次、读 $\mathbf{K},\mathbf{V}$ 若干次（分块）、写 $\mathbf{O}$ 一次。

但这里卡着一个数学难题：softmax 需要**每行的全局最大值与全局分母**，而 tiling 让 $\mathbf{S}$ 是一块一块算出来的——**怎么在没见过整行的情况下，增量地、又分毫不差地算出 softmax？** 这就是在线 softmax。

## 3 公式解析：在线 softmax 的再缩放

标准 softmax 要「先找整行最大值、再归一化」，天然是个两遍算法。在线 softmax（online softmax）把这两步改成一遍，秘诀是**每次见到新块都维护一个「运行中的最大值」和「运行中的分母」，遇到更大的最大值就整体再缩放一次**。

设已处理的前半段得分 $x_1,\dots,x_m$ 的最大值为 $\tilde{m}$，分母为 $\ell = \sum_{i\le m} e^{x_i - \tilde{m}}$。现在来了后半段 $x_{m+1},\dots,x_n$，它的最大值是 $\tilde{m}'$、分母是 $\ell'$。合并公式为：

$$
\tilde{m}_{\text{new}} = \max(\tilde{m}, \tilde{m}'), \qquad
\ell_{\text{new}} = \ell \cdot e^{\tilde{m} - \tilde{m}_{\text{new}}} + \ell' \cdot e^{\tilde{m}' - \tilde{m}_{\text{new}}}
$$

对这条式子做三步拆解：

- **第一步，读两个指数项**：$e^{\tilde{m} - \tilde{m}_{\text{new}}}$ 与 $e^{\tilde{m}' - \tilde{m}_{\text{new}}}$。因为 $\tilde{m}_{\text{new}}$ 是两个最大值中的较大者，这两项里**必有一项为 1、另一项 < 1**——它们的作用是把旧分母 $\ell$ 和 $\ell'$ 都「校准」到新最大值这把尺子下。
- **第二步，理解相加**：校准后的两段分母相加，正好等于「以 $\tilde{m}_{\text{new}}$ 为基准的整行分母」$\sum_i e^{x_i - \tilde{m}_{\text{new}}}$。因为把 $e^{x - \tilde{m}} e^{\tilde{m} - \tilde{m}_{\text{new}}} = e^{x - \tilde{m}_{\text{new}}}$，指数律保证了两段可以无缝合并。
- **第三步，为什么算得准**：每一步都维护的是**精确**的运行最大值与运行分母，没有截断、没有近似——唯一的工作只是把已累加的值「换算」到新的最大量级上。所以 FlashAttention 与标准 softmax **逐位给出相同的数学结果**，是 exact 算法。

输出块 $\mathbf{O}$ 也随新最大值做同样的再缩放：若 $\tilde{m}$ 变成更大的 $\tilde{m}_{\text{new}}$，已累加的输出整体乘 $e^{\tilde{m} - \tilde{m}_{\text{new}}}$，再加上新块的贡献 $e^{\tilde{m}' - \tilde{m}_{\text{new}}} \cdot \mathbf{P}' \mathbf{V}'$。这正是「在线归一化」的完整闭环：**一遍扫描、中间值全在片上、结果精确**。

## 4 复杂度对比：把账算清

把标准与 FlashAttention 的 HBM 访问量放在一起（$M$ 为片上 SRAM 容量，$N$ 为序列长，$d$ 为头维度）：

| | 标准注意力 | FlashAttention |
| --- | --- | --- |
| 显存占用 | $O(N^2)$（$\mathbf{S},\mathbf{P}$ 各 $N^2$） | $O(N)$（只存 $\mathbf{Q},\mathbf{K},\mathbf{V},\mathbf{O}$） |
| HBM 访问 | $\Theta(Nd + N^2)$ | $\Theta(N^2 d^2 / M)$ |
| 计算量 | $\approx 2N^2 d$ | 略高（向后传播重算 $\mathbf{S}$，约多 1.3×） |

为什么 FlashAttention 是 $\Theta(N^2 d^2 / M)$？直觉是这样：设块大小 $B \approx M/d$（让每个块正好占满 SRAM），对每个查询块要遍历 $N/B$ 个键/值块，共 $N/B$ 次全量读 $\mathbf{K},\mathbf{V}$，每次读 $O(Nd)$——于是总访问 $\approx (N/B)\cdot Nd = N^2 d \cdot (d/M) = N^2 d^2 / M$。<span class="marginnote">只要 $d^2 \le M$，这个数就小于标准注意力的 $N^2$ 项。以 $d=128$ 为例，$d^2=16384$，而一块 H100 SM 的 SRAM 有约 100–200 KB，$M$ 是 $d^2$ 的十几倍——所以 FlashAttention 的 HBM 访问通常比标准注意力低一个数量级。论文同时证明了这是「精确注意力的 I/O 下界」：不可能再做得更少。</span>

论文更进一步证明：**不存在任何精确计算注意力的算法，能对 $M \in [d, Nd]$ 的所有情况使用比 $\Theta(N^2 d^2/M)$ 更少的 HBM 访问**。也就是说 FlashAttention 不只是「聪明」，而是**在 IO 复杂度意义上已经最优**。

用 $N = 32\text{K}$、$d = 128$ 的典型长上下文算一笔账。标准注意力每头要写 $\mathbf{S}$ 和 $\mathbf{P}$ 两块 $N\times N$ 矩阵：单块 $\mathbf{S}$ 就占 $32768^2 \times 2$ 字节 $\approx 2$ GB，两块加中间往返，显存与带宽一起爆。FlashAttention 的 HBM 访问约为 $N^2 d^2 / M \approx 2^{30} \cdot 16384 / 2^{17} \approx 2^{27}$，即**百 MB 量级——比标准低了一个数量级以上**，而且显存占用从「两块 2 GB 的方阵」降成「几个 $O(Nd)$ 的矩形」。这就是为什么 10 万 token 上下文在 FlashAttention 之前几乎是禁区，之后成了常规配置。<span class="marginnote">这里的「128 MB」是数量级估算，实际值随 SRAM 大小、分块策略浮动；重点是<strong>它和 N² 的平方灾难不再挂钩</strong>。IO 复杂度的意义就在于此：不是常数上的小优化，而是把复杂度里的 N² 项降了下来。</span>

## 5 代价与边界：没有免费的午餐

FlashAttention 并非没有代价，理解它的边界才用得准：

- **向后传播要重算**：为了省显存，训练时反向传播不存 $\mathbf{S},\mathbf{P}$，而是**重算**注意力矩阵（对每个块再算一遍 $\mathbf{Q}_i\mathbf{K}_j^\top$）。总计算量约增加 33%，但因为注意力是内存瓶颈，**多算的这点被少搬的大量覆盖**，实际依然更快。
- **依赖 $d^2 \le M$ 才最优**：头维度 $d$ 一旦过大（如某些工作用 $d=512$ 以上），块就塞不下 SRAM，加速比打折。大多数模型（GPT、Llama 的 $d=64$–$128$）都在甜区。
- **它改变的是 IO，不是 FLOPs**：注意力本身的 $O(N^2)$ 计算量还在，序列超长时二次方的**算力**依然存在。FlashAttention 解决的是内存问题，后续的稀疏注意力、状态空间模型（如 Mamba）解决的是另一个问题——不要混为一谈。

## 6 辨析｜易错点

- **「FlashAttention 是近似注意力」**——错。它用在线 softmax 维护精确的运行最大值与分母，**与标准 softmax 数学结果逐位一致**，是 exact 算法。近似注意力是另一类工作（稀疏化、低秩分解）。
- **「FlashAttention 更快是因为算得少」**——错。它甚至多算了约 33%（反向重算），快是因为把 HBM 访问量降了一个数量级，而注意力恰恰是内存瓶颈。这正呼应 Roofline：内存瓶颈的算子，优化 B 才是正道。
- **「只要用了 FlashAttention 就能支持任意长序列」**——不。它把显存从 $O(N^2)$ 降到 $O(N)$，但**计算量仍是 $O(N^2)$**；序列极长时算力成为新瓶颈。它的贡献是把「显存装不下」这条线往后推，而不是消灭二次方。
- **「FlashAttention 只对推理有用」**——错。它一开始就是为训练设计的（前向 + 反向重算），推理端的 KV Cache 优化是另一条线。长上下文训练的可行性正是它的主场。

## 7 小结

- 标准注意力把 $\mathbf{S}, \mathbf{P}$ 两块 $N\times N$ 矩阵写回 HBM：显存 $O(N^2)$、HBM 访问 $\Theta(Nd + N^2)$，被平方项主导。
- Roofline 上注意力是内存瓶颈（$I \approx d < I_{\text{ridge}}$），所以正确的优化方向是**减 B**——FlashAttention 正是这么做的。
- 三大设计：**tiling**（块塞进 SRAM）、**IO 感知**（中间矩阵永不落盘）、**单 kernel 融合**（无启动、无往返）。
- 在线 softmax 用「新最大值出现时整体再缩放」$\tilde{m}_{\text{new}},\ \ell_{\text{new}}$ 公式，实现**一遍精确**的 softmax——这是 tiling 成立的数学关键。
- 复杂度：显存 $O(N)$ vs $O(N^2)$，HBM 访问 $\Theta(N^2d^2/M)$ vs $\Theta(Nd+N^2)$，且已达精确注意力的 I/O 下界。
- 代价：反向重算多约 33% 算力，依赖 $d^2 \le M$；它解决内存问题，二次方算力问题仍在。

在下一节，我们将离开单卡的算力与访存，进入**多卡**的世界：当训练要横跨几百张 GPU，数据不再在内存层次里流动，而是在机器之间的网络里流动——第一课，认识集合通信的六种原语。
