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

并发实验告诉我们「服务该运营在哪个并发」，而引擎的调度参数决定「这个并发能不能被兑现」。vLLM 有两个最关键的旋钮：**`max_num_seqs`**（一个 batch 里最多放多少个序列）与 **`max_num_batched_tokens`**（一个 batch 里最多有多少 token）。它们像批的「宽与深」：前者限制「多少行」，后者限制「多少总 token」。拧错它们，要么吞吐上不去，要么延迟爆炸，要么直接 OOM。<span class="marginnote">本专题《Continuous Batching》讲过「批的形状」决定吞吐与延迟；本篇把<strong>批的形状参数化</strong>——这两个参数就是 vLLM 里批的「形状控制器」。</span>

本篇讲两个参数的作用、相互约束、以及如何用「并发实验 + 显存预算」把它们调好。

## 1 两个参数各管什么

**`max_num_seqs`（最大序列数）**：调度器允许**同时进入一个 batch 的请求数上限**。它直接限制并发——超过这个数的请求只能排队等待。**它决定批的「宽度」**。<span class="marginnote">它类似并发上限的引擎侧实现：<strong>max-num-seqs = 引擎内部承认的最大并发</strong>。并发实验找到的运营并发，最终要靠它落地。</span>
- **`max_num_batched_tokens`（最大批 token 数）**：一个 batch 中**所有序列的 token 总数上限**（含 prefill 与 decode 的 token）。它限制批的「总计算量」——防止一个大 batch 的显存/算力超限。**它决定批的「深度」**。

两者的关系不是独立：**每个序列的 token 数不定（长请求有更多 token），所以两个参数要联合约束**。调度器在每一步检查「当前序列数 ≤ max-num-seqs 且 当前总 token ≤ max-num-batched-tokens」，任一超限就不再进新的请求。

## 2 显存预算：两个参数的地基

两个参数都受**显存预算**约束。推理的显存大头是：权重 + KV Cache + 激活。KV Cache 的预分配直接由「最大序列数 × 最大序列长度」决定：

$$\text{KV Cache 预算} = \text{max-num-seqs} \times \text{max-model-len} \times \text{每 token KV 字节}$$

- **`max_num_seqs` 越大 → 预留的 KV Cache 越多** → 可用的激活/权重显存越少 → 能撑住的 batch 反而可能更小。
- **`max_num_batched_tokens` 越大 → 单步激活越大**（prefill 大矩阵）→ 激活显存峰值越高。

**辨析｜易错点：max-num-seqs 不是越大越好。** 它大，并发容纳多、吞吐上限高；但它**同时预留大量 KV 显存**，且 batch 过大让每个请求的 decode 变慢、P99 延迟恶化（见并发曲线篇）。**它是在「吞吐上限」与「延迟 + 显存」之间权衡的旋钮**——调大必须配合压测验证 P99 没有恶化。

## 3 调优方法论：从并发实验出发

正确的调优顺序不是「拍脑袋改参数」，而是自顶向下：

1. **跑并发实验**（上篇），找到目标并发 $L_{\text{op}}$ 与目标 batch 的 token 量。
2. **把 `max_num_seqs` 设成 ≥ $L_{\text{op}}$**（略留余量，让排队不阻塞）。
3. **用显存预算反推 `max_num_batched_tokens` 的上界**：总显存 − 权重 − 预留 KV − 激活余量 = 可用的批 token 上限。
4. **微调验证**：在目标并发下测 P99 与吞吐，逐步增大/减小参数观察「吞吐是否还涨」「P99 是否还达标」。<span class="marginnote">微调的经验信号：<strong>P99 突然恶化 = 批太大，调小 max-num-batched-tokens 或 max-num-seqs；吞吐不涨 = 已达 GPU 上限，调参数无益，该扩容</strong>。</span>

**两类常见配置目标**：

- **延迟敏感**（在线对话）：小 batch——`max_num_seqs` 小（如 32）、`max_num_batched_tokens` 小，换取稳定低 P99。
- **吞吐优先**（离线批处理）：大 batch——两个参数都大，牺牲单请求延迟换总吞吐。

## 4 公式解析：批的 token 消耗

批的 token 总量由「prefill 部分」与「decode 部分」组成。设批内有 $S$ 个序列，第 $i$ 个序列正在 prefill（长 $p_i$）或 decode（已生成 $g_i$ 个 token），则本步的批 token 数：

$$B_{\text{tokens}} = \sum_{i \in \text{prefill}} p_i + \sum_{j \in \text{decode}} S_j$$

- **第一步，读 prefill 项**：一个 10k token 的 prefill 请求，$p_i = 10^4$，**一个请求就占掉 max-num-batched-tokens 的大半**。这就是为什么 Chunked Prefill（本专题）要把 prefill 切块。
- **第二步，读 decode 项**：decode 序列每步只贡献 1 个 token（当前解码位置），所以 decode 部分的总 token 数 = 批内 decode 序列数。**decode 阶段批的 token 数 ≈ max-num-seqs**。
- **第三步，读约束的意义**：$B_{\text{tokens}} \le \text{max-num-batched-tokens}$ 意味着——**在 max-num-seqs 个 decode 序列的基础上，还能容纳多少 prefill 的 token**。`max_num_batched_tokens` 大 → 容纳大 prefill（吞吐好但延迟高）；小 → prefill 被迫切块/排队（延迟稳但吞吐受限）。**两个参数联合决定「prefill 与 decode 在批里的共存比例」**。

## 5 数值算例：一组参数的完整调优

把方法论落成具体数字。设 7B INT4 模型、80 GB 卡、目标并发 $L_{\text{op}}=128$、平均请求「512 输入 + 512 输出」、每 token KV 约 0.2 KB：

**第一步，定 max-num-seqs**：取 $\ge L_{\text{op}}$，留余量设 160。此时 KV 预留 $= 160 \times 8192 \times 0.2\,\text{KB} \approx 262\,\text{MB}$——占 80 GB 的 0.3%，几乎可忽略。**短上下文下 max-num-seqs 的显存代价很小，可以放心大**。

**第二步，反推 batched-tokens**：总显存 80 GB − 权重 35 GB（INT4）− KV 0.3 GB − 激活余量 10 GB ≈ 34 GB 可用。但 **batched-tokens 不直接对应显存，它对应「单步激活」**：一个 32768 token 的批，prefill 激活约几个 GB——所以 32768 是安全的起点。

**第三步，微调验证**（并发 128 下）：

| 配置 | 吞吐 | P99 | 结论 |
| --- | --- | --- | --- |
| seqs=160, tokens=32768 | 基准 | 基准 | 起点 |
| tokens=65536 | +15% | P99 恶化 2 倍 | 批太大，回退 |
| seqs=64 | −30% | 更好 | 并发不够，别用 |

**读这张表**：tokens 从 32768 涨到 65536，吞吐只 +15% 但 P99 翻倍——**过了拐点，参数再大只是伤害延迟**；seqs 从 160 降到 64，吞吐掉了 30%——**max-num-seqs 是并发的地基，太小直接阉割吞吐**。调优的终点是「P99 达标下的最大吞吐」，不是「参数最大」。

## 6 常见调参误区

- **误区一：两个参数一起猛涨。** batched-tokens 涨 4 倍，激活峰值也涨，先 OOM 的是显存不是吞吐。**一次只动一个参数**，另一个保持基线。
- **误区二：max-num-seqs 设得比运营并发小。** 运营并发 128 却设 seqs=64，请求永远排队，并发实验的结论直接被阉割。**seqs 必须 ≥ 目标并发**。
- **误区三：长上下文下忘算 KV 预留。** 128k 上下文 + seqs=256 的 KV 预留可能几十 GB——**seqs 的显存代价随上下文长度暴涨**，长上下文场景 seqs 反而要调小。
- **误区四：只在单请求延迟看结果。** 单个请求快不代表批吞吐高（批大时单请求必然慢）。**判断标准是「P99 达标 + 吞吐」双指标**，不是单请求速度。

**辨析｜易错点：这两个参数不是「配置文件的默认值」，而是「负载的调节器」。** 同样的引擎，对话负载与批处理负载的最优参数不同。**每次负载画像变化，都要重跑并发实验重调**——参数跟着负载走，不跟着「社区推荐」走。

把「不同负载的推荐起点」列成表，作为调优的入口：

| 负载 | max-num-seqs | max-num-batched-tokens | 理由 |
| --- | --- | --- | --- |
| 在线对话（短上下文） | 256 | 16384 | 短请求批大、延迟优先 |
| 在线对话（长上下文） | 64 | 8192 | 长 KV 预留大，批别太深 |
| 离线批处理 | 128 | 65536 | 吞吐优先，容忍高延迟 |
| RAG 问答（中长输入） | 128 | 32768 | 输入长 prefill 多，均衡 |

**表的作用**：先按负载类别选一组「合理起点」，再用第五节的微调流程细调——**起点选对，微调才快；起点选错，微调是在错误区域里找最优**。

**上线前的参数检查清单**：

- 是否 max-num-seqs ≥ 目标并发？
- 是否算过「长上下文场景」的 KV 预留？
- 是否一次只动一个参数、其余保持基线？
- 是否用「P99 达标 + 吞吐」双指标验证，而非只看单请求速度？
- 负载画像变化（输入变长/并发变高）后是否重测过？

**这五条每项打钩，参数配置才不会「上线即翻车」**——调优的成果要靠检查清单守住。

## 7 小结

- **max-num-seqs 管批的宽度**（多少序列），max-num-batched-tokens 管批的深度（多少 token），联合约束批形状。
- **两者都受显存预算约束**：KV 预留随 max-num-seqs 增长，激活峰值随 batched-tokens 增长。
- **max-num-seqs 不是越大越好**：调大换吞吐上限，但 P99 与显存同时付出代价，必须压测验证。
- **调优自顶向下**：并发实验定目标 → 设 max-num-seqs → 显存预算反推 batched-tokens → 微调验证。
- **两类配置**：延迟敏感用小批、吞吐优先用大批；P99 恶化调小，吞吐不涨该扩容。
- **一次只动一个参数**：过了拐点参数再大只是伤害延迟，seqs 必须 ≥ 目标并发，长上下文记得 KV 预留。

在下一节，我们处理显存侧的问题——**显存利用率与显存碎片问题排查**。
