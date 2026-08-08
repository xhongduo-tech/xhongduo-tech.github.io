---
title: 张量并行的通信量分析与 Megatron 的 1D 切分方案
date: 2026-08-07
---

# 张量并行的通信量分析与 Megatron 的 1D 切分方案

<div class="epigraph">
<p>我们提出一种简单、高效的层内模型并行方法：它不需要任何新的编译器或库改动，与流水线模型并行正交且互补，仅需在原生 PyTorch 中插入几个通信操作即可完整实现。</p>
<footer>—— 肖伊比 等（Mohammad Shoeybi et al.），《Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism》，2019</footer>
</div>

<div class="article-byline">
<p>第四级 · AI 基础设施 ｜ AI基础设施技术栈 第三篇 ｜ 2026-08-07</p>
</div>

## 为什么从通信量开始

上一篇我们推导出两种切分方式——按列切、按行切——并且发现：**单个矩阵乘无论怎么切，总通信量都是恰好一次 AllReduce**。这个结论既是好消息也是坏消息。好消息是：如果每层只有一两次 AllReduce，而一次 AllReduce 传的数据量（$b \cdot s \cdot h$）相对算力（$b \cdot s \cdot h^2$）小得多，那么张量并行在算力账上完全可行。坏消息是：**AllReduce 的次数与大小必须被精确数清楚**，否则「每层都通信」会悄悄吃掉全部收益。

这一篇就把这笔账算到底：先数清一个 Transformer 层里到底有几次 AllReduce（答案是四次），再用 $\alpha\text{-}\beta$ 模型算出每层通信量，最后展示 Megatron 的 1D 切分——它用一个精妙的「按列接按行」配对，把这四次通信压缩到理论最小，并告诉我们为什么 TP 只能活在单机内部。<span class="marginnote">这里的「1D」指<strong>只在矩阵的行-列一个维度上切分</strong>（每个维度切一刀为一维）；后来还有把权重切成 2×2 块的 2D（SUMMA）、切成 3D 立方体的 3D 切分，都是为了进一步摊薄通信。<strong>理解 1D，是读懂所有高维切分的起点。</strong></span>

## 1 数一数：一个 Transformer 层有几次 AllReduce

Transformer 的每一层由两个 block 组成：**自注意力块**与 **MLP 块**，每个块都有两个矩阵乘。Megatron 的切分方案给每个块定了一个规矩：**第一个矩阵乘按列切，第二个矩阵乘按行切**。

先看 **MLP 块**（假设隐藏维 $h$，中间维 $4h$）：

1. $Y = X W_1$，$W_1$ 按列切（$h \to 4h$ 的输出维切开）——前向免通信；反向的 $\partial L/\partial X$ 需要一次 AllReduce。
2. GELU 逐元素激活——作用于已切分的张量上，**零通信**。
3. $Z = \text{GELU}(Y) W_2$，$W_2$ 按行切（$4h \to h$ 的输入维切开）——前向需要对部分积做一次 AllReduce；反向免通信。

所以 **MLP 块：1 次前向 AllReduce + 1 次反向 AllReduce**。

再看 **自注意力块**：

1. QKV 投影 $Q,K,V = X W_{\text{qkv}}$，$W_{\text{qkv}}$ 按列切（把 $3h$ 的输出维按 head 切开）——前向免通信；反向一次 AllReduce。
2. 注意力计算**完全在本地完成**：head 已经切好，每个 rank 只算自己的那几个 head，softmax 只在自己 rank 的 head 上做——**零通信**。
3. 输出投影 $O_{\text{out}} = \text{Attn}(Q,K,V) W_o$，$W_o$ 按行切——前向一次 AllReduce；反向免通信。

所以 **注意力块：1 次前向 AllReduce + 1 次反向 AllReduce**。

**重点：一个 Transformer 层一共 4 次 AllReduce——2 次前向、2 次反向。** 这就是 Megatron 1D 切分的全部通信骨架。它为什么是「最小」的？每个块都有两个矩阵乘，每个矩阵乘都必须有一次 AllReduce（不是前向就是反向，上一篇已证）；两个块就是 4 次。**没有任何方案能少于这个数**——除非把两次矩阵乘「融合」成一次通信，那是 2D/3D 切分干的事，但也只是换一种拆法，总字节数不会更少。

## 2 公式解析：每层通信量的定量推导

设 batch 为 $b$、序列长为 $s$、隐藏维为 $h$，TP 组内有 $P$ 张卡，BF16 每元素 2 字节。每个 AllReduce 作用在一个 $b \times s \times h$ 的激活张量上，单次通信量（元素数）由第二篇的 Ring 分析给出：

$$
V_{\text{单次}} = \frac{2(P-1)}{P} \cdot b \cdot s \cdot h
$$

一个层有 4 次，所以**每层每卡通信字节数**为：

$$
V_{\text{层}} = 4 \cdot \frac{2(P-1)}{P} \cdot b s h \cdot 2 = \frac{16(P-1)}{P}\, b s h \ \ \text{字节}
$$

三步拆解这条式子：

- **第一步，数清 4 次从哪来**：注意力块 2 次（前向 1 + 反向 1），MLP 块 2 次（前向 1 + 反向 1）。每次 AllReduce 的对象都是完整的 $b \cdot s \cdot h$ 激活，与 $P$ 无关——**切分不改变「要同步的数据总量」，只改变每卡分担的比例**。
- **第二步，看 $(P-1)/P$**：这是 Ring AllReduce 的带宽项系数。$P$ 越大，$(P-1)/P$ 越接近 1，所以**通信量随 $P$ 增长极其缓慢**——$P=2$ 时是 0.5，$P=8$ 时是 0.875，$P \to \infty$ 趋近 1。TP 组内加卡，每层通信总量几乎不变。
- **第三步，看与算力的比值**：Transformer 每层的训练计算量约为 $72\, b s h^2$ FLOPs（前向 $24$、反向 $48$），摊到 $P$ 卡每卡是 $72 b s h^2 / P$。于是**每卡通信/计算比**：
  $$
  R = \frac{V_{\text{层}}}{72\, b s h^2 / P} = \frac{16(P-1)/P \cdot b s h}{72\, b s h^2 / P} = \frac{2(P-1)}{9 h}
  $$
  这个比值只取决于 $P$ 与 $h$——**与 batch、序列长、模型层数都无关**。代入 $h = 8192$、$P = 8$：$R = 14 / 73728 \approx 1.9\times 10^{-4}$。意思是：**每算 1 FLOP，只要移动约万分之二字节**。对一台 A100（有效算力按 100 TFLOPS 计），需要的网络带宽只有 $1.9\times10^{-4} \times 10^{14} \approx 19$ GB/s——单机 NVLink 随便给得起。

**这就是 Megatron 敢做 1D 切分的数学底气：模型越宽（$h$ 越大），通信相对算力越便宜。** 但同时，$R = 2(P-1)/(9h)$ 随 $P$ **线性增长**——TP 组加卡，通信占比反而变大，因为算力被 $P$ 平分了、通信总量却没变。这把「TP 组不能太大」变成了一个可量化的结论，而不是一句经验之谈。

## 3 f 与 g：把通信点显式化

如何让这 4 次 AllReduce 在代码里「现形」？Megatron 用一对**共轭函数** $f$ 与 $g$：

- **$f$：前向做一次 AllReduce，反向保持恒等**——放在按行切分层的输出上（那个必须求和的地方）。
- **$g$：前向保持恒等，反向做一次 AllReduce**——放在按列切分层的输入梯度上（那个必须归约的地方）。

它们之所以叫「共轭」，是因为 $f$ 与 $g$ 拼接起来在数学上是恒等映射：$f \circ g = \text{id}$。这保证了**插入它们不改变前向与反向的数值结果**——纯通信，零语义。实现上就是两个包装类：

```python
import torch

def _reduce(x):
    """张量并行组内的 AllReduce：先求和，再广播，使各 rank 拿到一致结果"""
    torch.distributed.all_reduce(x, group=model_parallel_group)
    return x

class _CopyToModelParallelRegion(torch.autograd.Function):
    """f：前向恒等，反向 AllReduce"""
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return _reduce(grad_output)

class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """g：前向 AllReduce，反向恒等"""
    @staticmethod
    def forward(ctx, x):
        return _reduce(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
```

代码里的 `_reduce` 分支就是第 2 节那 4 次 AllReduce 的落点：**`g` 的前向制造 2 次前向通信，`f` 的反向制造 2 次反向通信**。你在 Megatron 源码里搜 `f`/`g`，数出来的调用次数与结构分析完全吻合——这印证了「4 次/层」不是纸面推导，而是真实代码里跑着的账。

## 4 为什么 TP 只适合单机内：带宽的硬约束

第 2 节的比值 $R \approx 10^{-4}$ 看起来低得离谱，但它有一个前提：**每次 AllReduce 必须能藏进该层自己的计算时间里**。如果网络太慢，AllReduce 等不到计算做完，就会变成每层都插一脚的串行气泡。

具体算一笔。假设一个 40 层的 7B 模型，单步训练约 100 ms，那么**每层的计算预算约 2.5 ms**。TP 组内每层要做 4 次 AllReduce，每次对象是 $b \cdot s \cdot h$ 激活。取 $b = 1$、$s = 2048$、$h = 8192$：每次 AllReduce 传 $16.8\times10^6$ 元素 $= 33.6$ MB（BF16），4 次共约 134 MB。

- **单机 NVLink**（DGX A100 约 450 GB/s/卡）：$134\,\text{MB} / 450\,\text{GB/s} \approx 0.3$ ms——远小于 2.5 ms 预算，通信被完美隐藏。
- **跨机网卡**（单卡约 25 GB/s）：$134 / 25 \approx 5.4$ ms——**超过该层计算时间**，每层都额外多出 3 ms 气泡，训练直接减速两倍以上。

**这就是「TP 不出机」的定量原因：跨机带宽比 NVLink 低一个量级，而 TP 的通信频率（每层 4 次）容不下这种慢。** 跨机切模型，交给流水线并行——它只在层边界通信、频率低得多。<span class="marginnote">注意这里的「单机」指<strong>一个 NVSwitch / NVLink 域</strong>（如 DGX 的 8 卡）。更大规模的 TP 组可以用更宽的 NVLink 域（如 NVL72 的 72 卡），本质不变：<strong>TP 组的直径必须被一条高速互连覆盖</strong>。</span>

顺带一提：这个预算分析也解释了为什么 TP 适合「宽而浅」的模型——$h$ 大则单层算力足、通信占比低；层数多反而只是线性叠加通信次数。**大模型的 TP 度通常取 8 或 4，很少超过 16**，正是第 2 节 $R \propto (P-1)$ 的直接后果。

## 5 与 DP、PP 的组合：TP 只是拼图的一块

单靠 TP 训练不了大模型，它是被嵌在**混合并行**里的：

**TP 负责单机内**：把装不下的层切成 8 份，靠 NVLink 扛住每层的矩阵乘。
**DP 负责机间**：把整份模型（已切成 TP 组）再复制多份，每份吃不同的数据——DP 的 AllReduce 每步只做一次，跨机带宽够用。
**PP 负责纵深**：模型层数太多、单机装不下整个纵向时，按层切成多段，段间跨机串行——这是下一篇的主题。

于是最常见的部署是：**每个节点内部做 TP（8 卡），节点之间做 DP 与 PP**。TP 组内通信靠 NVLink，机间通信靠网卡。理解了这个分工，就理解了为什么训练框架的日志里，TP 通信从不走网卡、DP/PP 通信从不占 NVLink——**每种并行都活在属于自己的带宽层级里**。<span class="marginnote">这个「通信必须与物理拓扑匹配」的原则贯穿整个 AI 基础设施：NCCL 的拓扑检测、网络的轨式（rail-optimized）设计、TP 不出机、PP/DP 出机，全部是同一句话的不同说法——<strong>让每种通信都走最便宜的路径</strong>。</span>

## 6 辨析｜易错点

- **「TP 组越大，每卡通信越少」**——错。每层通信总量约 $8\, b s h$，与 $P$ 基本无关；而每卡算力是 $72 b s h^2 / P$，随 $P$ 递减。**通信/计算比 $R = 2(P-1)/(9h)$ 随 $P$ 线性上升**——TP 组越大，通信相对越贵，所以 TP 度有限。
- **「TP 通信量依赖 batch 大小」**——通信字节数确实含 $b \cdot s$，但**通信/计算比与 $b$、$s$ 无关**（两者同比例增长，约掉了）。这也是它比 DP 稳定之处：DP 的通信/计算比随每卡 batch 缩小而恶化，TP 不会。
- **「f 与 g 是优化技巧，删了也能跑」**——删掉 f/g 就删掉了 AllReduce，前向/反向的数值会直接错误（要么输出缺一块、要么梯度缺一块）。**它们是正确性的组成部分，不是锦上添花**。
- **「4 次/层是 4 次大通信」**——注意通信的是**激活（$b s h$）而非权重（$h^2$）**。对大模型 $h^2 \gg b s h$，所以 TP 通信传的其实是很「小」的对象；真正大的权重梯度反而不需要跨 TP 组通信（每卡只更新自己的 $1/P$）。
- **「TP 能解决一切装不下」**——TP 只解决「单层放不下」。层数多到单机装不下纵向时，还要 PP；优化器状态多到放不下时，还要 ZeRO。**并行策略是按需组合的，不是单选题**。

## 7 小结

- 一个 Transformer 层在 Megatron 1D 切分下**恰好 4 次 AllReduce**（2 前向 + 2 反向）：注意力块与 MLP 块各 1 前 + 1 反，这是理论最小。
- **每层通信字节数** $V_{\text{层}} = \frac{16(P-1)}{P}\, b s h$；**通信/计算比** $R = \frac{2(P-1)}{9h}$——与 batch、序列长无关，只取决于 $P$ 与 $h$。
- $R$ 随 $P$ 线性增长：**TP 度越大，通信相对越贵**，这是 TP 组不超过 8~16 的定量依据；模型越宽（$h$ 大）则 TP 越划算。
- **f（前向 AllReduce）与 g（反向 AllReduce）** 是一对共轭恒等函数，用来把 4 次通信显式化；删掉它们训练直接出错。
- **TP 不出机**：每层通信预算只有 2.5 ms，NVLink（0.3 ms）能藏住，跨机网卡（5.4 ms）藏不住——TP 必须活在 NVLink 域内。
- 混合并行里 TP 管机内、DP 管机间复制、**PP 管纵向层切分**——每种并行都走与自己带宽层级匹配的路径。

在下一节，我们把「横切」换成「纵切」：**流水线并行**——当模型深到单机装不下整个纵向时，如何把层切成几段、用 micro-batch 把空闲填满，以及那条著名的气泡率公式。
