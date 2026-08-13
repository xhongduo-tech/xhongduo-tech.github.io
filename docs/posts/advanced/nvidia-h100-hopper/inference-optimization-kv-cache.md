---
title: 推理优化：KV Cache 与 FlashAttention
date: 2026-08-07
---

# 推理优化：KV Cache 与 FlashAttention

<div class="epigraph">
<p>训练是修一条路，推理是每天在这条路上跑车——它们优化的是不同的事。</p>
<footer>—— 训练与推理的比喻</footer>
</div>

<div class="article-byline">
<p>第四级 · AI 硬件：NVIDIA H100/Hopper ｜ NVIDIA FlashAttention 官方文档 ｜ 2026-08-07</p>
</div>

## 为什么从推理优化讲起

上一节讲训练：怎么把模型训出来。但模型训练完，真正的应用是**推理（inference）**——用户每问一句，模型要实时生成回答。推理与训练的优化重点完全不同：训练是「喂饱 Tensor Core 的吞吐」，推理则受制于**自回归解码的串行本质**——一次只生成一个 token，每一步都要读一遍全部上下文。本节讲清楚推理的两个核心优化：**KV Cache**（缓存历史键值，避免重复计算）与 **FlashAttention**（融合注意力，省显存省带宽）——它们决定了一个模型在 H100 上「每秒能生成多少字」。<span class="marginnote">推理分两个阶段：<strong>prefill（预填充）</strong>一次性处理用户输入的整段 prompt，计算密集；<strong>decode（解码）</strong>逐 token 生成，访存密集。绝大多数在线服务的时间都花在 decode 上，而 decode 的瓶颈是「读权重 + 读 KV Cache」的内存带宽——这就把 Roofline 模型里的「带宽受限」表现得淋漓尽致。</span>

## 1 自回归解码：为什么推理天生「慢」

大语言模型生成文本的方式是**自回归（autoregressive）**：每次只预测下一个 token，把新 token 拼进输入，再预测下一个。解码的第 $t$ 步：

$$
P(\text{token}_t \mid \text{token}_1, \ldots, \text{token}_{t-1})
$$

这个过程的两个致命特性：

- **串行**：第 $t$ 步必须等第 $t-1$ 步的输出，无法并行——生成 $T$ 个 token 至少要 $T$ 次串行计算。
- **重复计算**：如果没有缓存，第 $t$ 步要重新计算前 $t-1$ 个 token 的注意力 Key 和 Value——总计算量 $O(T^2)$，而实际上每个 token 的 K/V 只需算一次。

KV Cache 正是为了消灭第二个问题而生。

**KV Cache（键值缓存）**：把每个已处理 token 的 Key 和 Value 张量缓存下来，后续解码直接复用，避免重复计算。有了它，生成 $T$ 个 token 的注意力计算量从 $O(T^2)$ 降为 $O(T)$。<span class="marginnote">代价是显存：KV Cache 要驻留显存，且随序列长度线性增长。长对话、长文档推理的显存压力，主要就来自 KV Cache——这也是「长上下文」研究要跟推理优化绑在一起的原因，我们在《长上下文》专题会展开。</span>

## 2 KV Cache 有多大：一条公式看透显存

KV Cache 的显存占用是一条很清晰的公式：

$$
S_{\text{kv}} = 2 \times L \times D \times b
$$

- $2$：每个 token 要缓存 Key 和 Value 两份张量；
- $L$：transformer 层数；
- $D$：隐藏维度（$D = \text{头数} \times \text{每头维度}$）；
- $b$：每个数的字节数（FP16 为 2 字节）。

**每生成一个 token 新增的 KV Cache 大小**就是这个公式。代入一个 7B 模型（$L=32$、$D=4096$、FP16）：

$$
2 \times 32 \times 4096 \times 2 = 512\ \mathrm{KB/token}
$$

若支持 **4K 上下文**（4096 token），峰值 KV Cache：

$$
512\ \mathrm{KB} \times 4096 \approx 2\ \mathrm{GB}
$$

再乘上并发请求数（batch），显存压力立刻放大。**这就是为什么推理服务要精心管理 KV Cache**：分配策略（预分配 vs 按需）、淘汰策略（LRU 等）、以及 `PagedAttention`（按页管理，vLLM 的核心）都是为了「同样显存塞进更多并发」而设计。<span class="marginnote">对比 70B 模型（$L=80$、$D=8192$）：每 token 约 2.5 MB，4K 上下文约 10 GB——单卡 H100 的 80 GB 显存光 KV Cache 就吃掉八分之一，还没算权重和中间激活。显存管理是推理服务的核心工程。</span>

## 3 FlashAttention：注意力本身的手术

KV Cache 解决了「重复计算」，但注意力计算本身还有一个大问题：**标准注意力会物化一个 $N \times N$ 的注意力矩阵**。

标准注意力的三步：$QK^T$ → softmax → $\times V$。其中 $QK^T$ 的结果（$N \times N$ 的矩阵）要先写进 HBM（显存），softmax 后再读出来乘 $V$。**这个中间矩阵的读写，让注意力的内存流量是计算量的好几倍**——在 Roofline 图上，注意力是典型的带宽受限算子。

**FlashAttention** 的手术方案是**分块 + 在线 softmax**：

**分块（tiling）**：不把整个 $QK^T$ 算完，而是把 $Q$、$K$、$V$ 切成小块，逐块计算并累积结果，中间矩阵从不写回 HBM；
**在线 softmax（online softmax）**：softmax 需要全局归一化，但分块计算时可以用「running max + running sum」的增量式算法，每块算完就更新，不牺牲数值正确性；
- **重计算（recompute）**：backward 时需要的中间量可以重算而不是存储，进一步省显存。

效果是双重的：**内存流量从 $O(N^2)$ 降到 $O(N)$ 量级**（按块与 HBM 交互），**显存占用从 $O(N^2)$ 降到 $O(N)$**——序列越长，收益越大。在 H100 上，FlashAttention-2/3 进一步用上 TMA、wgmma 与 warpgroup 流水线，把注意力算子的利用率推到接近 GEMM 的水平。<span class="marginnote">FlashAttention 是「算法意识 + 硬件意识」结合的典范：它理解注意力的数学（在线 softmax 保持正确性），也理解 H100 的存储层级（TMA 分块搬运、片上计算、避免 HBM 往返）。它证明了一件事——<strong>在硬件上写算法，数学与工程的边界是可以被打通的</strong>。</span>

## 4 公式解析：标准注意力 vs FlashAttention 的内存流量

用 Roofline 的思路量化两者的差距。标准注意力写读 $N \times N$ 中间矩阵，内存流量：

$$
M_{\text{std}} = O(N^2) \times (\text{读 + 写次数})
$$

FlashAttention 每次只把「一个小分块」读入片上 SRAM，分块数由 SRAM 大小决定，总 HBM 流量：

$$
M_{\text{FA}} = O\left(\frac{N^2}{S_{\text{SRAM}}} \cdot M_{\text{block}}\right)
$$

其中 $S_{\text{SRAM}}$ 是片上共享内存大小，$M_{\text{block}}$ 是每块与 HBM 交换的字节数。定性结论：

- 标准注意力：**每个 $N \times N$ 元素都要进一次 HBM**，$O(N^2)$ 流量；
- FlashAttention：**$N \times N$ 的矩阵被切成 $N/S_{\text{SRAM}}$ 块，每块只在片上算完**，与 HBM 交互的是分块数据本身，流量降到 $O(N)$ 量级（相对）。

代入一个具体序列（$N = 4096$，FP16）：标准注意力的 $QK^T$ 矩阵是 $4096^2 \times 2$ 字节 = 32 MB，要在 HBM 写读多次；FlashAttention 把它拆成片上可容纳的小块，每块 32 MB / 分块数，整块注意力只需把 $Q/K/V$ 各读一遍——**内存流量省了一个数量级以上**。这就是长序列推理/训练里 FlashAttention「更快且更省显存」的来源。

## 5 小结

- 推理是**自回归解码**：串行 + 逐 token，decode 阶段访存密集（带宽受限）。
- **KV Cache** 缓存历史 K/V，把注意力计算量从 $O(T^2)$ 降为 $O(T)$；显存 = $2LDb$ × 序列长度。
- **FlashAttention** 用分块 + 在线 softmax + 重计算，把注意力内存流量从 $O(N^2)$ 降到 $O(N)$