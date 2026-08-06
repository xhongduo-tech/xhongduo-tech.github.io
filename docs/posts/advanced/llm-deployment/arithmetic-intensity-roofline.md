---
title: 算术强度与 Roofline 模型
date: 2026-08-07
---

# 算术强度与 Roofline 模型

<div class="epigraph">
<p>计算的目的是洞察，而非数字。</p>
<footer>—— 理查德 · 汉明（Richard Hamming），《面向科学家与工程师的数值方法》（Numerical Methods for Scientists and Engineers, 1962）</footer>
</div>

<div class="article-byline">
<p>第四级 · 大模型部署 ｜ LLM推理引擎体系 第一章 ｜ 2026-08-07</p>
</div>

## 为什么从算术强度开始

上一节我们反复说：Prefill **计算受限**，Decode **访存受限**。这两个词直觉上很清楚，但「受限到什么程度」「优化应该往哪个方向使劲」还缺一把尺子。这一节补上这把尺子——**算术强度（arithmetic intensity）**与 **Roofline 模型**。

Roofline 由伯克利的塞缪尔 · 威廉姆斯（Samuel Williams）、安德鲁 · 沃特曼（Andrew Waterman）与大卫 · 帕特森（David Patterson）在 2009 年提出，本意是给多核 CPU 画性能天花板，如今成了推理引擎分析访存/算力瓶颈的标配工具。<span class="marginnote">《AI基础设施》讲 GPU 显存层次时也引用过它：A100 的 FP16 算力约 312 TFLOPS、HBM 带宽约 2 TB/s，二者之比约 156 FLOP/byte，就是「脊点」。今天我们把脊点这个概念彻底展开。</span> 学会它，你就能回答三个部署中最常见的问题：这个 kernel 是算力瓶颈还是带宽瓶颈？换卡到底有没有用？量化为什么几乎总是「白捡」的加速？

## 1 算术强度：每搬运 1 字节，干多少浮点活

**算术强度（arithmetic intensity）**：一段计算中，每从主存（GPU 上是 HBM）搬运 1 字节数据，平均执行多少次浮点运算。记作：

$$I = \frac{\text{FLOPs}}{\text{bytes}}$$

单位是 **FLOP/byte**。I 大，说明这段计算「很划算」——搬进来的每个字节都被反复使用；I 小，说明「很亏」——字节搬进来看一眼就用完了。<span class="marginnote">2009 年原文用的是「operational intensity（操作强度）」，特指「每字节 DRAM 流量对应的操作数」，强调只统计真正进出主存、经过缓存过滤后的字节。工程上「算术强度」成了通用说法，两者数值含义一致——我们沿用 FLOP/byte。</span>

用上一节的结论套一下：Decode 每步 $I \approx 1$ FLOP/byte，意思是每搬 1 字节权重，只换来约 1 次浮点运算；Prefill 处理 1024 token 时 $I \approx 1024$，每搬 1 字节被用约 1000 次。**同一台机器，同一批权重，只是因为一次多算几个 token，划算程度差了三个数量级。**

## 2 Roofline：把一台机器的能力边界画成屋顶

一台 GPU 的能力被两条硬边界卡住：

- **算力屋顶（compute roof）**：芯片每秒最多做多少次浮点运算，记为 $\pi$（单位 FLOP/s）。A100 FP16 约 $312$ TFLOPS。
- **带宽屋顶（bandwidth roof）**：内存每秒最多搬多少字节，记为 $\beta$（单位 byte/s）。A100 HBM 约 $2$ TB/s。

任何一段计算，其**可达到性能（attainable performance）**都不可能超过这两个屋顶中的较低者：

$$\text{Attainable} = \min\left(\pi,\ \beta \times I\right)$$

把这张图按 log-log 坐标画出来，就是 Roofline：

![Roofline 模型：算力屋顶与带宽屋顶围成的可达到性能边界，Prefill 与 Decode 落在两侧](/images/llm-deployment/arithmetic-intensity-roofline-1.svg)

看这张图的三处关键结构：

- **带宽屋顶是一条斜率为 1 的对角线**：性能 $=\beta \times I$，I 越大、能榨出的性能越高。落在它上面的计算是**访存受限（memory-bound）**。
- **算力屋顶是一条水平线**：性能天花板 $\pi$，与 I 无关。落在它下面的计算是**计算受限（compute-bound）**。
- **脊点（ridge point）**：两条线相交的地方，横坐标 $I_{\text{ridge}} = \pi/\beta$。**它是「需要多高的算术强度才能榨满算力」的阈值**。

## 3 把 Prefill 与 Decode 放上 Roofline

回到上一节算出的两个数字：Decode $I\approx 1$，Prefill（$P=1024$）$I\approx 1024$。把它们放进 A100 的 Roofline：

- **Decode（$I\approx 1$）**落在带宽屋顶上，可达到性能 $\approx \beta \times 1 = 2$ TFLOPS，远低于算力屋顶 312 TFLOPS——**访存受限**。它还能跑多快，取决于能把带宽用得多满，与算力无关。
- **Prefill（$I\approx 1024$）**早已越过脊点 156，撞在算力屋顶上，可达到性能 $\approx 312$ TFLOPS——**计算受限**。它的瓶颈是 Tensor Core 能算多快。

这就是上一节那张表的几何化：**脊点把一切计算分成「左边省着用算力、右边榨不满算力」两类**。Decode 在左边，量化（减字节）直接把带宽屋顶向上顶；Prefill 在右边，FlashAttention（省中间访存）或更快的算力才有意义。

## 4 公式解析：脊点在哪，可达到性能是多少

脊点是整个模型最有用的一笔账，逐步拆开。

**第一步，写出「内存说它能跑多快」**：带宽屋顶是 $\beta \times I$。直观地读：每秒搬 $\beta$ 字节，每字节带来 $I$ 次运算，于是每秒最多 $\beta I$ 次运算。

**第二步，写出「算力说它最多多快」**：$\pi$，一个常数，与 $I$ 无关。

**第三步，取两者较小者**，得到可达到性能公式。真正的工作点由短板决定：

$$\text{Attainable}(I) = \min(\pi,\ \beta I)$$

**第四步，求脊点**：令两条线相等，$\beta I_{\text{ridge}} = \pi$，即：

$$I_{\text{ridge}} = \frac{\pi}{\beta}$$

代入 A100：$\pi/\beta = 312\times10^{12} / (2\times10^{12}) = 156$ FLOP/byte。代入 H100（FP16）：$989\times10^{12}/(3.35\times10^{12}) \approx 295$ FLOP/byte。<span class="marginnote">注意 H100 的脊点比 A100 更靠右：它算力涨了约 3 倍、带宽只涨约 1.7 倍，于是「需要多高的算术强度才能榨满算力」的阈值变高了。这意味着 <strong>H100 上访存受限的范围反而更宽</strong>——7B 这类小模型的 Decode 在 H100 上更加「带宽不够用」。这个反直觉的结论，正是 Roofline 的价值。</span>

**第五步，验证工作点**：Decode 在 A100 上可达到性能 $=\min(312T, 2T\times 1)=2$ TFLOPS，每 token 时间 $=14\ \text{GFLOPs}/2\ \text{TFLOPS}=7$ ms；Prefill 可达到性能 $=\min(312T, 2T\times1024)=312$ TFLOPS，14 TFLOPs 只需约 46 ms。**数值与上一节的结论完全吻合，公式闭环了。**

## 5 从 Roofline 读出优化方向

Roofline 最大的用处，是告诉你「往哪个方向优化才有效」：

**对访存受限（脊点左侧）的计算，唯一有效的杠杆是减字节。** Decode 每步读 $Nb$ 字节，把 $b$ 从 2（FP16）降到 1（INT8）或 0.5（INT4），带宽屋顶不变，但相同时间能搬更少的模型——每 token 时间近似减半。这就是为什么**量化几乎总是 Decode 的「白捡」加速**（第六篇），也是 KV Cache 量化（第六篇）与各种缓存策略的理论依据。

**对计算受限（脊点右侧）的计算，才谈得上减少运算量。** Prefill 需要 FlashAttention 这类把注意力重算变成省访存的 kernel、需要 kernel 融合减少中间张量落盘——它们优化的是「有效 FLOP 占比」，而不是字节数。

**辨析｜易错点：** 三个常见误读：

**误区一：认为「算力更大的卡一定更快」。** Roofline 说：对 Decode（$I\approx1$），瓶颈是带宽屋顶 $\beta$，算力翻倍毫无帮助。选卡要看你的 $I$ 落在脊点哪一侧。

**误区二：把 Roofline 当成绝对预测。** 可达到性能是上限，不是实际值。真实 kernel 还受显存延迟、调度开销、访存模式（是否合并）影响，通常只能达到屋顶的 60%–90%。Roofline 回答「能不能更快」，不回答「实际多快」。

**误区三：以为脊点位置固定。** $I_{\text{ridge}}=\pi/\beta$ 随硬件变化——H100 的脊点比 A100 靠右，新一代硬件往往把「访存受限」的范围推得更宽。做性能预估时，要按目标机型重算这笔账。<span class="marginnote">这也解释了端侧部署（第十一篇）的残酷现实：手机的内存带宽远低于数据中心 GPU，脊点靠左、访存受限更严重——llama.cpp 的一切优化都围着「少搬字节」打转。</span>

## 6 小结

- **算术强度** $I = \text{FLOPs}/\text{bytes}$：每搬 1 字节干多少浮点活；Decode 约 1，Prefill 约 $P$。
- **可达到性能** = $\min(\pi,\ \beta I)$：被算力屋顶与带宽屋顶中的短板卡住。
- **脊点** $I_{\text{ridge}}=\pi/\beta$：A100 约 156 FLOP/byte，H100 约 295 FLOP/byte；左侧访存受限、右侧计算受限。
- **优化方向**：访存受限 → 减字节（量化、缓存）；计算受限 → 减运算（kernel 融合、FlashAttention）。
- **选卡判断**：小模型 Decode 在 H100 上比 A100 更「带宽不够用」，换卡看带宽不看算力。

在下一节，我们把「访存受限」这四个字推到极致，专门算一笔账：一个 7B 模型每生成一个 token，到底要把多少 GB 权重从显存搬一遍，把 Decode 的 Memory-Bound 变成可预测的每秒 token 数。
