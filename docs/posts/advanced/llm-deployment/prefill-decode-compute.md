---
title: Prefill 与 Decode 两阶段的计算特征
date: 2026-08-07
---

# Prefill 与 Decode 两阶段的计算特征

<div class="epigraph">
<p>为追求高并行处理速率所付出的努力，若不伴随着接近同等量级的串行处理速率提升，终将被浪费。</p>
<footer>—— 吉恩 · 阿姆达尔（Gene Amdahl），《单处理器方法用于大规模计算能力的有效性》（Validity of the Single Processor Approach to Achieving Large-Scale Computing Capabilities, 1967）</footer>
</div>

<div class="article-byline">
<p>第四级 · 大模型部署 ｜ LLM推理引擎体系 第一章 ｜ 2026-08-07</p>
</div>

## 为什么从两阶段开始

上一节我们看到了自回归生成循环：逐 token 预测、拼接、再预测。把一次完整生成放到显微镜下，它其实是**两段气质完全不同的旅程**——处理整段 prompt 的 **Prefill（预填充）**，与逐 token 产出的 **Decode（解码）**。

这两段旅程的差别，不是「先后」那么简单，而是**计算形态的根本不同**：Prefill 一次并行处理 P 个 token，算术强度高，接近计算受限（compute-bound）；Decode 每步只产出一个 token，却必须把全部权重从头读一遍，属于访存受限（memory-bound）。<span class="marginnote">「计算受限」指性能被芯片每秒能算多少次乘法累加（FLOPs）卡住；「访存受限」指性能被内存每秒能搬多少字节（带宽）卡住。二者判据的量化，是本节与下一节《算术强度与 Roofline 模型》的主题。</span>

这两个阶段，是整个推理引擎设计的「第一性原理」。vLLM 的调度器、TensorRT-LLM 的 kernel 融合、SGLang 的前后端分离，本质上都在回答同一个问题：**怎样让 Prefill 更快、怎样让 Decode 不再被显存带宽拖死**。这一节，我们把这两段旅程的每一笔账都算清楚。

## 1 Prefill：一次并行的「大扫除」

一次请求到达时，模型拿到的是整段 prompt，比如「请用三句话解释什么是黑洞」。**Prefill（预填充）**：对这段输入做**一次性、全并行的前向计算**——为一个长度 $P$ 的 prompt 同时算出 $P$ 个位置各自的隐状态、注意力、以及每个位置之后的 K、V 向量。<span class="marginnote">注意：即便有因果掩码，Prefill 阶段的所有位置也几乎可以并行计算——第 $t$ 个位置只是「不能看到 $t$ 之后」，但它能独立地与它之前的所有位置做注意力。并行度由 GPU 的矩阵单元一次性吃满。</span>

**Prefill 的计算形态是「胖」的**：输入多、输出多、中间结果多。一次前向要做的浮点运算约等于：

$$\text{FLOPs}_{\text{prefill}} \approx 2 \cdot N \cdot P$$

其中 $N$ 是模型参数量（7B 即 $7 \times 10^9$），$P$ 是 prompt 长度。以 LLaMA-2 7B、$P=1024$ 为例，一次 Prefill 约 $2 \times 7\times10^9 \times 1024 \approx 1.4\times10^{13}$ 次运算，也就是 **14 TFLOPs**。

在这段计算里，权重矩阵被**反复复用**：同一个 $W_K$ 要同时和 $P$ 个位置的输入相乘。算得越多、每字节数据被用的次数越多，因此 Prefill 的算术强度很高，倾向于**计算受限**——GPU 的矩阵单元（Tensor Core）才是它的瓶颈。Prefill 的总耗时决定了 **TTFT（Time To First Token）**：你输入完问题后，等第一个字出现要多久。

## 2 Decode：每步只产出一个 token 的「串行散步」

Prefill 结束后，模型手里握着全部 prompt 的 K、V 缓存，然后进入 Decode。**Decode（解码）**：每一步只计算**一个新位置**，产出下一个 token 的概率分布，选出 token 拼回序列，如此循环直到 EOS 或长度上限。<span class="marginnote">「Decode」这个名字在部署语境里特指「逐 token 生成的每一小步」，它对应的是 TPOT（Time Per Output Token）——每输出一个 token 所花的时间。注意它和我们熟悉的「解码策略」（贪心/采样）不是一回事，后者只是生成末端的一次轻量选择。</span>

**Decode 的计算形态是「瘦」的**：每步只算 1 个 token，但为了算出它，模型必须把**全部权重从显存搬到计算单元**——7B 模型在 FP16 下约 14 GB 权重，一个字节都不能少。产出却只有 1 个 token 的 logits。

$$\text{FLOPs}_{\text{decode}} \approx 2 \cdot N \cdot 1$$

同样是 14 GFLOPs 数量级的计算，与 Prefill 相比，**输入宽度从 P 掉到了 1**，但「读一遍全部权重」的运输成本一分不少。每字节权重只被使用约 1 次，算术强度骤降到约 1 FLOP/byte——这是典型的**访存受限**：性能被 HBM 带宽卡死，而不是被算力卡死。

**Prefill 与 Decode 的分界，不是时间先后，而是计算形态：Prefill 把 P 个 token 一起算，Decode 每步只算 1 个 token、却要完整读一遍权重。**

## 3 两张面孔的量化对比

把两阶段的特征放进一张表里对照，差异一目了然：

| | Prefill 预填充 | Decode 解码 |
| --- | --- | --- |
| 处理对象 | 整段 prompt（长度 $P$） | 每步 1 个新 token |
| 输入并行度 | 高（$P$ 个位置并行） | 低（每步 1 个位置） |
| 每步浮点运算 | $\approx 2NP$ | $\approx 2N$ |
| 权重复用次数 | $P$ 次 | 约 1 次 |
| 算术强度 | $\approx P$（FP16 下） | $\approx 1$ |
| 瓶颈 | 计算受限（compute-bound） | 访存受限（memory-bound） |
| 决定指标 | TTFT | TPOT |
| 典型优化 | FlashAttention、Chunked Prefill | KV Cache、批处理、投机解码 |

表格里「算术强度 $\approx P$」这一行，正是下一节 Roofline 模型的入口。这里先记住结论：**Prefill 是「算」的问题，Decode 是「搬」的问题**。两个阶段会往两个完全不同的方向去优化——这也是为什么主流引擎必须把二者区分开来分别调度。

## 4 公式解析：为什么算术强度随序列长度线性增长

把上面两张脸统一到一条公式里。设某次前向处理的 token 数为 $T$（Prefill 时 $T=P$，Decode 时 $T=1$），参数量为 $N$，权重每个元素 $b$ 字节（FP16 为 2）。定义**算术强度（arithmetic intensity）**为总运算量与总搬运字节之比：

$$I = \frac{\text{FLOPs}}{\text{bytes}} \approx \frac{2NT}{Nb} = \frac{2T}{b}$$

这条公式一次回答两个问题。

- **第一步，认分子 $2NT$**：Transformer 前向的浮点运算近似等于「每个参数每处理一个 token 做 2 次运算」（一次乘、一次累加），所以是参数量 × token 数 × 2。
- **第二步，认分母 $Nb$**：权重要从 HBM 读进计算单元，总共 $N$ 个参数、每个 $b$ 字节，所以最少搬 $Nb$ 字节。这里省略了激活与 KV 的搬运，在权重主导的大模型上是合理近似。
- **第三步，约分**：$2NT$ 除以 $Nb$，得 $I = 2T/b$。**FP16（$b=2$）时 $I \approx T$**——算术强度约等于「一次前向处理的 token 数」。
- **第四步，读出结论**：Decode 只处理 $T=1$ 个 token，$I \approx 1$ FLOP/byte；Prefill 处理 $T=1024$ 个 token，$I \approx 1024$ FLOP/byte。**同样是这个模型，算术强度相差三个数量级**——这就是「两种动物」的数学根源。

## 5 易错辨析与部署含义

**辨析｜易错点：** 下面三个误区在刚接触推理引擎时几乎人人踩过。

**误区一：认为 Decode 慢是因为「每步都要重新做一遍前向」。** 对，但抓错了重点。每步前向的计算量只有约 14 GFLOPs，A100 上 0.05 毫秒级就算完；真正花掉约 7 毫秒的是**读那 14 GB 权重**。所以优化 Decode 的主战场在「少搬字节」，而不是「少算几下」。

**误区二：把 Prefill 与 Decode 当成分两次的服务调用。** 实际上一次完整生成 = 1 次 Prefill + 若干次 Decode，首 token 的等待时间是 Prefill 加上第一次 Decode 的合计。<span class="marginnote">长 prompt 时 Prefill 占主导，决定「第一字多快」；短 prompt 长输出时 Decode 占主导，决定「整体多慢」。引擎因此发展出 <strong>PD 分离</strong>部署（第八篇）：把 Prefill 与 Decode 拆到不同 GPU 上，各自按自己的瓶颈优化。</span>

**误区三：以为「换更强的 GPU」就万事大吉。** 对 memory-bound 的 Decode，决定速度的是**带宽**而不是**算力**。H100 比 A100 算力高约 3 倍，带宽只高约 1.7 倍——所以同样 7B 模型，Decode 只快约 1.7 倍。这个「换卡看带宽」的判断标准，下一节会用 Roofline 严格化。

这些特征直接决定了引擎的架构取向：Prefill 需要 FlashAttention 这类把访存压到极限的 kernel，还需要 Chunked Prefill（第三篇）把长 prompt 切成块、避免饿死 Decode；Decode 则需要批处理把「读一遍权重」的成本摊到多个序列身上，需要 KV Cache 避免重算已看过的内容——这两件事正是后续几节的主题。

## 6 小结

- **Prefill**：一次性并行处理整段 prompt，运算量 $\approx 2NP$，算术强度 $\approx P$，**计算受限**，决定 **TTFT**。
- **Decode**：每步只产 1 个 token，运算量 $\approx 2N$，却要完整读一遍权重，算术强度 $\approx 1$，**访存受限**，决定 **TPOT**。
- **算术强度公式**：$I = 2NT/(Nb) = 2T/b$，FP16 下约等于「一次前向处理的 token 数 $T$」——两阶段因此相差约三个数量级。
- **优化的分岔**：Prefill 优化「算」（kernel 融合、FlashAttention），Decode 优化「搬」（KV Cache、批处理、量化降字节）。
- **换卡判断**：对 memory-bound 的 Decode，看带宽不看算力。

在下一节，我们将引入衡量「每字节干多少活」的算术强度与描述整台机器能力边界的 Roofline 模型，把「Prefill 计算受限、Decode 访存受限」画成一张图、算成一笔账。
