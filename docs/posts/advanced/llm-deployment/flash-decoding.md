---
title: FlashDecoding 与长序列 Decode 加速
date: 2026-08-07
---

# FlashDecoding 与长序列 Decode 加速

<div class="epigraph">
<p>Decode 时 GPU 最闲，也最忙——闲在算力，忙在访存。</p>
<footer>—— FlashDecoding 团队观察（Dao et al., 2023）</footer>
</div>

<div class="article-byline">
<p>第四级 · 大模型部署 ｜ FlashDecoding 技术博客（Dao et al., 2023） ｜ 2026-08-07</p>
</div>

## 为什么从 FlashDecoding 开始

FlashAttention 系列优化的是**prefill**（一次处理整段序列）。但推理的 decode 阶段是另一个世界：**每步只有一个 query token，却要对全量 KV Cache 做注意力**。这时 flash 的「分块并行」策略会撞上一个尴尬问题：query 只有一个，GPU 的并行度怎么填满？FlashDecoding 的答案是把 **KV 维切开**——既然 query 只有一个，就让不同 thread block 各负责一段 KV Cache，算完再归约合并。<span class="marginnote">decode 的注意力形状是 <strong>1×N（一个 query 对全部历史 key）</strong>，是「胖瘦不均」的极端形状。FlashDecoding 专为这种形状设计并行策略。</span>

本篇讲 decode 阶段注意力的独特性、FlashDecoding 的「切 KV + 跨 block 归约」方案，以及它在长序列上的收益。

## 1 Decode 注意力的形状问题

Prefill 阶段：batch 个 query、每个 query 对整段序列，注意力矩阵是「多个 query × 全量 key」的矩形。FlashAttention 把一个 block 负责一个 query 块（行），行数多、并行度充足。

Decode 阶段：每步只有一个新 token，注意力是「1 × N」——**一个 query 对 $N$ 个历史 key**。如果沿用「一个 block 负责一行」，那整张 GPU 只有一个 block 在干活，其余 SM 全部闲置。<span class="marginnote">这是 decode 注意力与 prefill 的本质差异：<strong>prefill 行多、decode 行只有一行</strong>。并行策略必须从「按行切」改成「按列（KV）切」。</span>

更长序列（128k、1M token）让问题加剧：KV 维 $N$ 巨大，即使「一行」，这一步的访存量也是整个 KV Cache——**decode 注意力是纯访存瓶颈**，而 FlashAttention 的分块（切 KV 块）本身没错，错的是「一个 block 顺序遍历所有块」，并行度只来自 batch。

## 2 FlashDecoding 方案：切 KV + 归约

FlashDecoding 的核心改动：

1. **切 KV 维**：把 KV Cache 沿序列维切成 $G$ 段，每段由一个独立的 thread block 处理；
2. **各段独立算局部注意力**：每个 block 用 FlashAttention 的分块 + 在线 softmax，算出「针对自己那段的局部输出 $o^{(g)}$、局部 max $m^{(g)}$、局部指数和 $l^{(g)}$」；
3. **跨 block 归约**：把所有段的局部结果，用在线 softmax 的「合并两段」公式归约成一个最终输出。

合并两个局部结果（$o_1, l_1, m_1$）与（$o_2, l_2, m_2$）的公式：

$$m = \max(m_1, m_2), \quad l = l_1 e^{m_1 - m} + l_2 e^{m_2 - m}, \quad o = \frac{o_1 l_1 e^{m_1-m} + o_2 l_2 e^{m_2-m}}{l}$$

归约开销是一次很小的读写（$G$ 个局部结果，而非 $N$ 个 key）——**用 $O(G)$ 的合并代价，换来了 $G$ 倍的并行度**。<span class="marginnote">归约只在「解码一个 token」的粒度发生一次，$G$ 通常取 8–16，开销可忽略。对比直接的做法，<strong>并行度从「batch」提升到「batch × G」</strong>，SM 利用率显著改善。</span>

## 3 长序列 decode 的实际收益

FlashDecoding 的收益在两种场景尤其明显：

- **batch 小的长序列**：单用户长上下文（agent 记忆、长文档问答），batch=1，KV 巨大——正好是「行数少、列数多」的极端，FlashDecoding 把并行度从 1 拉到 $G$。
- **长序列 + 大批量**：多用户各自长上下文，并行度本已够高，FlashDecoding 的额外收益递减——**它解决的是「并行度不足」而非「总计算量大」**。

**辨析｜易错点：FlashDecoding ≠ 把 KV Cache 分到多 GPU。** 它把 KV 切在**同一个 GPU 内的 thread block** 上，是 SM 级并行；跨 GPU 切 KV 是另一回事（见《分布式推理》篇的张量并行与 KV 传输）。**别混淆「块级并行」与「卡级并行」**。

实测：在长序列 decode 上，FlashDecoding 相对单 block 遍历方案可提速数倍（论文报告的 GPT 风格测试中，长序列下 2 倍以上）；配合 batch 后总吞吐进一步上升。

## 4 公式解析：decode 注意力的访存账

一个 decode 步的注意力访存量：读取全部 KV Cache（$2 \times N \times d$ 元素），输出一个 token。设 KV 元素以 FP16 存储：

$$V_{\text{KV}} = 2 \cdot N \cdot d \cdot 2 \text{ bytes}$$

- **第一步，读量级**：$d$ 是单头的隐藏维（如 128），$N$ 是序列长度。$N=10^5$ 时 $V_{\text{KV}} \approx 51$ MB；$N=10^6$ 时约 512 MB。**KV Cache 是 decode 每步必须搬完的数据**。
- **第二步，算耗时下界**：访存带宽 $B$（H100 约 3 TB/s），单步最小耗时 $V_{\text{KV}}/B$。$N=10^5$ 时约 17 微秒，$N=10^6$ 时约 170 微秒——**序列每长 10 倍，decode 延迟就涨 10 倍**，与算力无关。
- **第三步，看并行化作用**：FlashDecoding 不减少 $V_{\text{KV}}$，只提高「搬运它的并行度」——多 block 并发读，让带宽瓶颈被吃满（而不是串行读）。**它优化的不是总 IO，而是「IO 的并发利用率」**。更彻底地降低 $V_{\text{KV}}$，要靠 KV Cache 量化与稀疏注意力（见量化篇）。

## 5 数值算例：FlashDecoding 的收益分解

把「切 KV + 归约」的收益算成具体数字。设 H100（132 SM）、单请求 decode、KV 长 $N=2\times10^5$（FP16，约 100 MB）、单 head 维 $d=128$。

**不切（一个 block 顺序遍历）**：一个 thread block 处理整个「1×N」注意力。block 内线程数有限（如 1024），它要顺序读完 100 MB 的 KV——**只有 1 个 SM 在忙，其余 131 个闲置**。访存带宽再高，单 block 的读带宽也只有一小块，耗时约几百微秒。

**FlashDecoding（切成 $G=16$ 段）**：16 个 block 并行，每个负责 6.25 MB KV。16 个 SM 同时读，访存带宽利用接近上限：

$$T_{\text{block}} \approx \frac{V_{\text{KV}}/G}{B_{\text{per-block}}} \approx \frac{100\,\text{MB}/16}{B_{\text{total}}/16} = \frac{100\,\text{MB}}{B_{\text{total}}}$$

**读这个式子**：把 $V_{\text{KV}}$ 分成 $G$ 份由 $G$ 个 block 并行读，总时间逼近「一次读完全部 KV」的带宽极限——**并行度从 1 拉到 $G$，时间接近除以 $G$**。<span class="marginnote">上限的直觉：<strong>访存瓶颈像「一个出口的停车场」——排队一辆辆出，还是开 16 个出口同时出</strong>。FlashDecoding 就是给访存开多个并行出口，让带宽真正被用满。</span>

**归约的代价**：16 个局部结果（$o^{(g)}, l^{(g)}, m^{(g)}$）合并，只读写 16 个标量对——微秒级，可忽略。**这就是「$O(G)$ 合并换 $G$ 倍并行」的完整账**。

**工程上的 $G$ 怎么选**：$G$ 取「GPU 的 SM 数 / 每个 block 占的 SM 数」量级即可（H100 上 8–32）。**$G$ 太小并行度不足，$G$ 太大归约与调度开销增长**——但 $G$ 的敏感区间很宽，8–32 之间性能差别不大，实现时取固定值即可。

## 6 FlashDecoding 与其他优化的搭配

FlashDecoding 不是孤立优化，它常与下面几个配合：

| 搭配 | 解决什么 | 互补关系 |
| --- | --- | --- |
| KV Cache 量化 | 减小 $V_{\text{KV}}$ | 量化减总量，FlashDecoding 提并行——两个都做，decode 更快 |
| 投机解码 | 减少 decode 步数 | FlashDecoding 让每步更快，投机解码让步数更少，双管齐下 |
| 张量并行（TP） | 跨卡分担 KV | FlashDecoding 管卡内，TP 管卡间，都是「切 KV」但层级不同 |
| PD 分离 | 隔离 decode 负载 | Decode 节点内再用 FlashDecoding 吃到极致 |

**读这张表**：FlashDecoding 的定位是「decode 阶段的底层加速」——它不替代上层优化，而是让「任何 decode 负载」都跑得更快。**它是最「无感」的一类优化：开箱即用、不吃配置、不影响精度**，所以现代引擎（vLLM、TensorRT-LLM）默认启用。

**辨析｜易错点：FlashDecoding 提升的是「单请求长序列」的延迟，不是「总吞吐」。** 总吞吐由 batch 与带宽决定；FlashDecoding 主要改善「batch 小时大 KV 的延迟」。**压测要区分「吞吐」与「延迟」两个指标**——FlashDecoding 的收益体现在「长序列单请求的 P99 延迟」上，别拿总吞吐的变化来衡量它。

FlashDecoding 相关的常见疑问速查：

| 疑问 | 答案 |
| --- | --- |
| 需要改模型吗？ | 不需要，是纯内核层优化，精度不变 |
| 引擎默认开吗？ | vLLM / TensorRT-LLM 现代版本默认启用 |
| 长序列才有效？ | 是，序列越长 KV 越大收益越明显 |
| 短序列有效吗？ | 收益递减，短序列并行度本就不缺 |
| 与 FlashAttention 冲突吗？ | 不冲突，一个管 prefill 一个管 decode |
| 显存会变多吗？ | 不会，只是并行方式变了 |

**读这张表**：FlashDecoding 是「无感」优化——不改模型、不耗显存、不损精度，**收益几乎白拿**。这也是它成为「默认开启」的原因：实现它对引擎是成本，但对用户是纯收益。

**一个判别「要不要关掉它」的场景**：如果你跑的是「短序列高并发」负载（聊天摘要），FlashDecoding 收益很小，且极端场景下归约开销可能略拖累——**此时可关闭对比实测**。但绝大多数负载，开着就是对的。

## 7 小结

- **decode 注意力是「1×N」极端形状**：一个 query 对全部历史 KV，沿「一行」的并行方式会让 GPU 闲置。
- **FlashDecoding 切 KV 维**：$G$ 个 block 各算一段局部注意力，再以 $O(G)$ 代价归约，并行度从 batch 提升到 batch × G。
- **在线 softmax 的合并公式**是归约的理论基础，数值上精确无近似。
- **适用场景**：batch 小、序列长的 decode（agent 长记忆、长文档），batch 大时收益递减。
- **它是 SM 级并行**，不是跨 GPU 切 KV；decode 延迟由 KV Cache 体积决定，FlashDecoding 优化的是搬运并发而非总 IO。
- **收益分解**：切 $G$ 段并行读，时间逼近带宽极限；与 KV 量化、投机解码、TP 均可叠加。

解码优化篇至此收尾。下一节进入**第八篇 分布式推理**，从**张量并行（TP）在推理中的实现**讲起。
