---
title: max-num-seqs 与 max-num-batched-tokens 调优
date: 2026-08-07
---

# max-num-seqs 与 max-num-batched-tokens 调优

<div class="epigraph">
<p>旋钮不是越多越好，而是「知道每个旋钮在拧什么」才好。</p>
<footer>—— 引擎调优实践（借自 vLLM 社区）</footer>
</div>

<div class="article-byline">
<p>第四级 · 大模型部署 ｜ vLLM 文档（engine args） ｜ 2026-08-07</p>
</div>

## 为什么从调度参数调优开始

并发实验告诉我们「服务该运营在哪个并发」，而引擎的调度参数决定「这个并发能不能被兑现」。vLLM 有两个最关键的旋钮：**`--max-num-seqs`**（一个 batch 里最多放多少个序列）与 **`--max-num-batched-tokens`**（一个 batch 里最多有多少 token）。它们像批的「宽与深」：前者限制「多少行」，后者限制「多少总 token」。拧错它们，要么吞吐上不去，要么延迟爆炸，要么直接 OOM。<span class="marginnote">本专题《Continuous Batching》讲过「批的形状」决定吞吐与延迟；本篇把<strong>批的形状参数化</strong>——这两个参数就是 vLLM 里批的「形状控制器」。</span>

本篇讲两个参数的作用、相互约束、以及如何用「并发实验 + 显存预算」把它们调好。

## 1 两个参数各管什么

- **`max-num-seqs`（最大序列数）**：调度器允许**同时进入一个 batch 的请求数上限**。它直接限制并发——超过这个数的请求只能排队等待。**它决定批的「宽度」**。<span class="marginnote">它类似并发上限的引擎侧实现：<strong>max-num-seqs = 引擎内部承认的最大并发</strong>。并发实验找到的运营并发，最终要靠它落地。</span>
- **`max-num-batched-tokens`（最大批 token 数）**：一个 batch 中**所有序列的 token 总数上限**（含 prefill 与 decode 的 token）。它限制批的「总计算量」——防止一个大 batch 的显存/算力超限。**它决定批的「深度」**。

两者的关系不是独立：**每个序列的 token 数不定（长请求有更多 token），所以两个参数要联合约束**。调度器在每一步检查「当前序列数 ≤ max-num-seqs 且 当前总 token ≤ max-num-batched-tokens」，任一超限就不再进新的请求。

## 2 显存预算：两个参数的地基

两个参数都受**显存预算**约束。推理的显存大头是：权重 + KV Cache + 激活。KV Cache 的预分配直接由「最大序列数 × 最大序列长度」决定：

$$\text{KV Cache 预算} = \text{max-num-seqs} \times \text{max-model-len} \times \text{每 token KV 字节}$$

- **`max-num-seqs` 越大 → 预留的 KV Cache 越多** → 可用的激活/权重显存越少 → 能撑住的 batch 反而可能更小。
- **`max-num-batched-tokens` 越大 → 单步激活越大**（prefill 大矩阵）→ 激活显存峰值越高。

**辨析｜易错点：max-num-seqs 不是越大越好。** 它大，并发容纳多、吞吐上限高；但它**同时预留大量 KV 显存**，且 batch 过大让每个请求的 decode 变慢、P99 延迟恶化（见并发曲线篇）。**它是在「吞吐上限」与「延迟 + 显存」之间权衡的旋钮**——调大必须配合压测验证 P99 没有恶化。

## 3 调优方法论：从并发实验出发

正确的调优顺序不是「拍脑袋改参数」，而是自顶向下：

1. **跑并发实验**（上篇），找到目标并发 $L_{\text{op}}$ 与目标 batch 的 token 量。
2. **把 `max-num-seqs` 设成 ≥ $L_{\text{op}}$**（略留余量，让排队不阻塞）。
3. **用显存预算反推 `max-num-batched-tokens` 的上界**：总显存 − 权重 − 预留 KV − 激活余量 = 可用的批 token 上限。
4. **微调验证**：在目标并发下测 P99 与吞吐，逐步增大/减小参数观察「吞吐是否还涨」「P99 是否还达标」。<span class="marginnote">微调的经验信号：<strong>P99 突然恶化 = 批太大，调小 max-num-batched-tokens 或 max-num-seqs；吞吐不涨 = 已达 GPU 上限，调参数无益，该扩容</strong>。</span>

**两类常见配置目标**：

- **延迟敏感**（在线对话）：小 batch——`max-num-seqs` 小（如 32）、`max-num-batched-tokens` 小，换取稳定低 P99。
- **吞吐优先**（离线批处理）：大 batch——两个参数都大，牺牲单请求延迟换总吞吐。

## 4 公式解析：批的 token 消耗

批的 token 总量由「prefill 部分」与「decode 部分」组成。设批内有 $S$ 个序列，第 $i$ 个序列正在 prefill（长 $p_i$）或 decode（已生成 $g_i$ 个 token），则本步的批 token 数：

$$B_{\text{tokens}} = \sum_{i \in \text{prefill}} p_i + \sum_{j \in \text{decode}} S_j$$

- **第一步，读 prefill 项**：一个 10k token 的 prefill 请求，$p_i = 10^4$，**一个请求就占掉 max-num-batched-tokens 的大半**。这就是为什么 Chunked Prefill（本专题）要把 prefill 切块。
- **第二步，读 decode 项**：decode 序列每步只贡献 1 个 token（当前解码位置），所以 `decode 的 token 数 ≈ 序列数`。**decode 阶段批的 token 数 ≈ max-num-seqs**。
- **第三步，读约束的意义**：$B_{\text{tokens}} \le \text{max-num-batched-tokens}$ 意味着——**在 max-num-seqs 个 decode 序列的基础上，还能容纳多少 prefill 的 token**。`max-num-batched-tokens` 大 → 容纳大 prefill（吞吐好但延迟高）；小 → prefill 被迫切块/排队（延迟稳但吞吐受限）。**两个参数联合决定「prefill 与 decode 在批里的共存比例」**。

## 5 小结

- **max-num-seqs 管批的宽度**（多少序列），max-num-batched-tokens 管批的深度（多少 token），联合约束批形状。
- **两者都受显存预算约束**：KV 预留随 max-num-seqs 增长，激活峰值随 batched-tokens 增长。
- **max-num-seqs 不是越大越好**：调大换吞吐上限，但 P99 与显存同时付出代价，必须压测验证。
- **调优自顶向下**：并发实验定目标 → 设 max-num-seqs → 显存预算反推 batched-tokens → 微调验证。
- **两类配置**：延迟敏感用小批、吞吐优先用大批；P99 恶化调小，吞吐不涨该扩容。

在下一节，我们处理显存侧的问题——**显存利用率与显存碎片问题排查**。
