---
title: 推理引擎的调度：continuous batching 与请求队列
date: 2026-08-07
---

# 推理引擎的调度：continuous batching 与请求队列

<div class="epigraph">
<p>不要让 GPU 为了等一个慢请求而闲下来。</p>
<footer>—— 吴悠（Woosuk Kwon），vLLM 作者</footer>
</div>

<div class="article-byline">
<p>第四级 · AI 基础设施 ｜ vLLM 论文（PagedAttention, 2023）· 推理基础设施篇 ｜ 2026-08-07</p>
</div>

## 为什么从 continuous batching 开始

上一节说「总吞吐 ≈ 副本数 × 单副本吞吐」。而**单副本吞吐的决定性因素，就是批处理策略**——推理引擎怎么把「变长、持续到达、随时完成」的请求组织成 GPU 上高效的 batch。传统做法（静态批处理）会浪费大半算力；现代做法（**continuous batching**）让吞吐提升数倍，是 vLLM 等引擎的核心卖点。

本篇讲透 continuous batching 的机制：静态批处理为什么浪费、动态批处理怎么运作、以及它如何与 KV Cache 管理（PagedAttention）配合。读完你就理解了「为什么 vLLM 比朴素实现快好几倍」。

## 1 静态批处理：等、一起算、一起完

朴素推理引擎用「静态批处理」：

1. **攒 batch**：等积满 $B$ 个请求（或超时）。
2. **统一 prefill**：把 $B$ 个请求拼成一个 batch 做 prefill。
3. **统一 decode**：所有请求同步地逐步 decode。
4. **一起完成**：最慢的请求做完，整个 batch 才释放。

**浪费的根源**：请求长度参差，长的 500 token、短的 10 token——batch 必须等「最长的那条」完成，**短的早就做完了却占着 slot**。GPU 在 batch 生命周期里，算力使用率大幅波动，大量时间在「等最慢的」。<span class="marginnote">静态批处理的「队头阻塞」是它低效的根源：一个 500-token 的慢请求拖住整个 batch，其余 9 个 10-token 的请求只能陪它走到最后。batch 越大、长度越不均，浪费越严重——这是推理吞吐上不去的头号原因。</span>

## 2 Continuous batching：随时进、随时出

**连续批处理（continuous batching）** 打破「batch 一起进一起出」的束缚：

- **请求随时加入**：新请求到达，立即插进当前正在执行的 batch（而不是等下一批）。
- **请求随时离开**：某个请求生成完，立即腾出 slot，让新请求顶上。
- **逐 token 调度**：每个 decode 步骤，引擎重新决定「这批有哪些请求」——batch 是「持续流动的池子」。

于是一个 GPU 上同时跑着处于不同进度的请求：A 在生成第 5 个 token、B 刚完成 prefill、C 差 2 个 token 结束。**每一刻的 batch 都尽可能满，GPU 永远有活干**。<span class="marginnote">continuous batching 的直觉：把 batch 从「固定的容器」变成「流动的队伍」。新来的人随时入队，完成的人随时离队，GPU 每步都处理「当前在场的人」。这消除了静态批处理的「队头阻塞」，让吞吐接近「GPU 能承载的上限」。</span>

## 3 请求队列与调度策略

continuous batching 的实现核心是**请求队列 + 调度器**：

- **等待队列**：新请求先入队，等调度器「准入」。
- **运行集合**：当前正在 GPU 上跑的请求。
- **调度决策**：每步决定——放几个新请求进 batch？优先 prefill 还是 decode？显存够不够？

**准入控制（admission control）** 是调度的关键：同时能跑多少请求受「KV Cache 显存」与「延迟 SLA」约束。调度器要决定「为了一个 prefill 要不要踢掉/推迟一些 decode」——这是「吞吐 vs 延迟」的实时权衡。<span class="marginnote">调度策略的经典取舍：prefill 优先（保证 TTFT）会挤压 decode（TPOT 变长）；decode 优先会拖长 TTFT。工程上常用「预算」制：每步给 prefill 一定算力配额，剩下的给 decode——两个指标的平衡是推理引擎调优的核心旋钮。</span>

## 4 KV Cache 管理：PagedAttention

continuous batching 的「动态进出」要求 **KV Cache 也能动态分配**——但 GPU 显存是「碎片化的块」，而 KV Cache 是按 token 增长的一维数组。朴素做法「为每个请求预分配最大长度」会严重浪费显存（多数请求用不到最大长度）。

**PagedAttention**（vLLM 的成名作）借鉴操作系统的虚拟内存：

- KV Cache 按**固定大小的页（block）** 管理，不再按「请求」分配连续大块。
- 每个请求的 KV 页可以**不连续**，靠「页表」索引。
- 显存按需分配，**只在真正使用时占页**——浪费大减、并发大升。

**PagedAttention + continuous batching 是天作之合**：动态的 batch 需要动态的 KV 分配，而分页让「任意请求随时进出」成为可能。<span class="marginnote">PagedAttention 的灵感直接来自操作系统的分页：进程的虚拟内存可以不连续、按需换入换出，KV Cache 同理。它带来的并发提升是数量级的——同样的显存，分页可以让同时服务的请求数翻 2–4 倍，因为「预留但没用」的显存被释放出来了。</span>

## 5 公式解析：continuous batching 的吞吐收益

设请求平均输出长度 $S$、标准差大（长短参差）、batch 上限 $B$、GPU 算力可同时跑 $C$ 个「等效 token」。

**静态批处理**的有效吞吐（受最长请求 $S_{\max}$ 约束）：

$$\text{Throughput}_{\text{static}} \approx \frac{B \cdot S_{\text{avg}}}{B \cdot S_{\max} / 1} = \frac{S_{\text{avg}}}{S_{\max}} \quad \text{（相对上限）}$$

**连续批处理**的有效吞吐：

$$\text{Throughput}_{\text{dynamic}} \approx \min\left(1,\ \frac{C}{S_{\text{avg}}}\right) \quad \text{（接近硬件上限）}$$

- **$\frac{S_{\text{avg}}}{S_{\max}}$（静态的浪费）**：平均长度与最大长度之比。长短不均时这个比值远小于 1——如 $S_{\text{avg}}=100, S_{\max}=500$，只有 20%。
- **$C/S_{\text{avg}}$（动态的逼近）**：连续批处理让 GPU 一直跑「有用的 token」，吞吐逼近「硬件算力 ÷ 平均长度」。
- **倍率**：$\frac{\text{dynamic}}{\text{static}} = \frac{S_{\max}}{S_{\text{avg}}}$——**长度越不均，continuous batching 的收益越大**（可达 3–5 倍）。<span class="marginnote">这个公式揭示了 continuous batching 的「含金量」来自请求长度的参差：如果所有请求长度相同，静态与动态差不多；但真实流量高度参差（有的问一句、有的写长文），所以收益巨大。这也是为什么「把请求按长度分组」的优化收益有限——分组的本质就是在「人工制造长度均匀的 batch」。</span>

## 6 辨析｜易错点：continuous batching 的常见误区

**辨析｜易错点：**
- **「continuous batching 就是攒更大的 batch」是错觉**：它的核心是「动态进出」，不是「batch 更大」。
- **「动态批处理不需要调度」是错的方向**：准入控制、prefill/decode 优先级是它的灵魂，调度做不好吞吐也上不去。
- **「PagedAttention 只是省显存」**：它还让「动态进出」可行——没有分页，请求进出会留下显存空洞。
- **「连续批处理牺牲延迟」不绝对**：它优先保证「GPU 满载」，延迟控制靠准入与优先级旋钮，做得好可以两全。
- **别忽略「显存碎片」**：KV Cache 分页后仍有页级碎片，页大小选择要在「管理开销」与「碎片率」间权衡。

## 7 小结

- **静态批处理的痛**：队头阻塞——batch 等最慢请求，短请求白占 slot，吞吐损失可达数倍。
- **Continuous batching**：请求随时进、随时出，batch 是流动的池子，GPU 每步都满。
- **调度器**：等待队列 + 运行集合 + 准入控制，prefill/decode 优先级实时权衡。
- **PagedAttention**：KV Cache 分页管理，按需分配、不连续、页表索引——与动态批处理配套。
- **收益公式**：吞吐提升约 $S_{\max}/S_{\text{avg}}$，请求越参差收益越大。

## 8 进阶与延伸

**动手对比静态与动态批处理**：用一个朴素推理实现（静态 batch）与 vLLM（continuous batching）分别跑一组「长短参差」的请求，对比两者的吞吐与队列深度——你会直观看到动态批处理对「长短不均」流量的吞吐优势。

**几个值得进一步挖的方向**：

- **准入控制的量化**：调度器每步决定「放几个新请求」——受「KV Cache 显存」与「延迟 SLA」约束。怎么把这两个约束写成「准入判据」？
- **PagedAttention 的页大小**：页大小（block size）从 16 调到 128——管理开销 vs 碎片率的权衡曲线是什么形状？实测找到甜点。
- **prefill/decode 的预算分配**：每步给 prefill 多少算力预算？「TTFT 的 SLA」与「TPOT 的 SLA」怎么换算成预算比例？

**自测题**：为什么「continuous batching 的收益 ≈ $S_{\max}/S_{\text{avg}}$」？如果你能说清「静态批被最长请求拖住、动态批每步都是满的」，就理解了「请求越参差、收益越大」。

## 9 动手实践清单

- 用朴素静态 batch 与 vLLM 分别跑「长短参差」的请求，对比吞吐。
- 观察 continuous batching 下「请求随时进、随时出」的行为。
- 调准入控制：新请求的注入与 KV 显存的约束。
- 调页大小（block size），观察碎片率与吞吐。
- 用「prefill 预算」平衡 TTFT 与 TPOT。
- 算「$S_{\max}/S_{\text{avg}}$」估算 batching 的收益上限。
- 用 profiler 确认「每步 batch 都是满的」。

在下一节，我们回答推理选 GPU 的核心问题——**显存带宽为何比算力更重要**。
