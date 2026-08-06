---
title: 'Chunked Prefill：长输入的分块调度'
date: 2026-08-07
---

# Chunked Prefill：长输入的分块调度

<div class="epigraph">
<p>编程的艺术，在于组织复杂性的艺术。</p>
<footer>—— 艾兹格 · 迪杰斯特拉（Edsger W. Dijkstra）</footer>
</div>

<div class="article-byline">
<p>第四级 · 大模型部署 ｜ LLM推理引擎体系 第三章 ｜ 2026-08-07</p>
</div>

## 为什么从长 Prefill 开始

连续批处理让「批的边界」流动起来，但还留着一个庞然大物：**一条超长输入的 Prefill**。想象一个 RAG 请求，带着 8192 token 的上下文进来——如果不加干预，它的预填充会**独占整个 GPU 好几秒**，批里所有正在解码的请求全部停摆，所有人的首字延迟（TTFT）一起雪崩。这不只是体验问题：**Prefill 和 Decode 连资源偏好都相反**（一个吃算力、一个吃带宽），把它们粗鲁地塞进同一批，等于两头都喂不饱。

Chunked Prefill 的答案非常朴素：**把长 Prefill 切成小块，每一块当一个「调度单元」来排队**。它不改变任何注意力公式，却同时解决队头阻塞、资源错配和抢占三个问题——是连续批处理之后最划算的一颗螺丝钉。

## 1 长 Prefill 的两个病灶

**病灶一：队头阻塞，TTFT 雪崩。** 第一篇《Prefill 与 Decode 两阶段的计算特征》讲过，Prefill 的算力需求随输入长度线性增长。一条 8192 token 的 Prefill 一步要处理的 token 数是 Decode 批的几十倍，占用 GPU 的时间也因此是几十倍。这段时间里，批内其他请求的 KV 已经就绪、只等生成，却一步都跑不了。**等最长的 Prefill 结束，短请求的 TTFT 已经暴涨到和它一样长。**

**病灶二：资源错配，两头挨饿。** Prefill 是**算力受限（Compute-Bound）**：把整段输入前向传播一遍，SM 满载、显存带宽用不满；Decode 是**访存受限（Memory-Bound）**：每步只算一个 token，瓶颈在读 KV、算力富余。<span class="marginnote">把这两种请求放进同一个 batch，GPU 的「算力」和「带宽」无法同时吃满——Prefill 把算力榨干时带宽空转，Decode 把带宽用满时算力空转。调度上的理想状态是让它们在不同时刻交替占满，而不是在同一时刻互相挤。</span>静态合批时这种错配被掩盖了，连续批处理则暴露了它：批里一旦混入长 Prefill，这个迭代步就退化成「纯 Prefill 步」，Decode 的带宽吞吐当场归零。

还有一个隐性病灶：**抢占粒度太粗**。长 Prefill 一旦中途被抢占（显存不足时调度器会踢人），因为它占用的是「一整块时间」，被踢的损失是整段 Prefill 的工作量；如果 Prefill 被切成小块，抢占只损失一小块。<span class="marginnote">这也是后续「调度器源码分析」章节里抢占逻辑能做得精细的前提——调度单元的粒度决定了一次抢占的代价上界。</span>

## 2 分块 Prefill 的思想

Chunked Prefill（也叫 prefill chunking）就三句话：

1. 把一条长度为 $L$ 的 Prefill 切成 $m = \lceil L / C \rceil$ 块，每块 $C$ 个 token（vLLM 默认 `max_num_batched_tokens` 的粒度在 256 附近）。
2. **每一块 Prefill 与一步 Decode 地位等同**：都作为一个「迭代单元」进入调度器排队。
3. 调度器把 Prefill 块与 Decode 步**交错执行**：跑一个 Prefill 块，再跑一步 Decode，再跑下一个 Prefill 块……

于是没有哪条请求能「独占 GPU 一整段时间」。长请求的 Prefill 被摊进很多个迭代步，短请求的 Decode 在缝隙里照常前进。**吞吐的敌人不是长 Prefill 本身，而是长 Prefill 对 GPU 的独占时间**——分块把独占时间打碎，队头阻塞随之瓦解。

值得强调：分块不省任何计算，也不省 KV 显存。已处理前缀的 KV 必须一直存在显存里（请求还活着），因此**中途被抢占的请求，其已算 KV 也要保存或换出**。Chunked Prefill 改变的是**调度粒度**，不是**计算量**。

## 3 工程实现：一个旋钮

在 vLLM 里，Chunked Prefill 几乎不需要显式开关——它由 `max_num_batched_tokens` 与 `max_num_seqs` 隐式驱动。**当一个请求的 Prefill 长度超过 `max_num_batched_tokens` 时，调度器自动把它切成若干块**，每块不超过该上限，与其他请求一起进入批。核心逻辑在调度器的「准入」分支：

```python
# 每个迭代步，调度器决定批里装什么
def schedule_iteration(scheduler):
    budget = scheduler.max_num_batched_tokens   # 本步 token 预算（如 256）
    for seq in scheduler.running:
        # 逐请求决定本步放多少 token
        if seq.is_prefilling():
            # 预填充：只放不超过预算的 C 个 token，剩余留在队列
            seq.tokens_to_schedule = min(budget, seq.remaining_prefill_tokens)
        else:
            # 解码：每步恰好 1 个 token
            seq.tokens_to_schedule = 1
        budget -= seq.tokens_to_schedule
```

`remaining_prefill_tokens` 就是这个旋钮的关键状态：**它让调度器知道一条 Prefill 走到哪了、还剩多少**。每步把预算分给 Prefill 块和 Decode 步，预算分完即止。<span class="marginnote">vLLM 论文里给出过一个直觉：`max_num_batched_tokens` 设得越大，批里能装的 Prefill 块越多、单块越大，吞吐越高；但它也推高单步延迟并占用更多 KV 显存。生产调优一般围绕这个值做扫描（第十篇《max-num-seqs 与 max-num-batched-tokens 调优》）。</span>TensorRT-LLM 的 In-flight Batching、SGLang 的调度器都有等价的机制，只是命名与默认值不同——这是推理引擎的「公共解」。

在命令行里，vLLM 最常被一起调的正是 `--max-num-batched-tokens` 与 `--max-num-seqs`：前者决定单步 token 预算（即 Prefill 块大小的上界），后者决定批内序列上限。把前者调大意味着允许更大的 Prefill 块、更高的单步吞吐，但也要为更大的批预留更多 KV 显存——第一篇《KV Cache 显存占用估算与数值实例》里的公式在这里派上用场，预算与显存要一起算。

**分块 Prefill 与 Prefix Caching 协同时，还有一个精妙之处**：一条正在分块进行的 Prefill，其**已完成的完整块**同样可以进入前缀缓存——如果另一条请求恰好共享了这部分前缀，它不必等第一条算完，直接复用已定稿的块即可。分块把「大而整」的 Prefill 变成「逐块可共享」的增量，给了调度器更多命中机会，这正是下一节 Prefix Caching 能在长输入场景大显身手的原因之一。

另一个边界效应常被忽略：当 $L$ 不是 $C$ 的整数倍时，**最后一块 Prefill 不满 $C$ 个 token**——它只占用部分预算，这一迭代步省下的预算会自然流给 Decode 或其他请求。所以分块不仅「粒度可调」，还自带一种应对不对齐的弹性，不会像整块 Prefill 那样一步把预算全部吃干。

## 4 公式解析：分块后的 TTFT 权衡

Chunked Prefill 不是免费的：**它牺牲长请求的单点延迟，换取系统吞吐与短请求延迟**。把账算清楚，用三个量。

设请求的 Prefill 长度 $L$，块大小 $C$，则块数：

$$
m = \left\lceil \frac{L}{C} \right\rceil
$$

对**长请求自身**，分块后的 TTFT 近似：

$$
\text{TTFT}_{\text{long}} \approx m \cdot \left( t_{\text{chunk}} + \bar{t}_{\text{wait}} \right)
$$

对**批里的 Decode 请求**，分块后它的首 token 延迟近似：

$$
\text{TTFT}_{\text{decode}} \approx t_{\text{chunk}} + t_{\text{step}}
$$

逐步拆解：

- **第一步，认 $m$**：$m$ 是块数，也是长请求「被排进迭代的次数」。$C$ 越小，$m$ 越大，调度越精细，但长请求等待的缝隙越多。
- **第二步，认 $t_{\text{chunk}}$**：一块 $C$ token 的 Prefill 前向时间。计算量与 $C$ 成正比，且注意 Prefill 是 Compute-Bound，这一块几乎占满全部 SM。
- **第三步，认 $\bar{t}_{\text{wait}}$**：长请求在两次块之间等待的平均时间——因为块与块之间，调度器会把 GPU 让给 Decode 和其他请求。这是长请求付出的「利息」。
- **第四步，对比两条式子**：长请求的 TTFT 从「一次性 $t_{\text{full}} = m \cdot t_{\text{chunk}}$」涨到「$m$ 份加上 $m$ 份利息」；Decode 请求的 TTFT 从「被 $t_{\text{full}}$ 整体堵死」降到「一块 Prefill 加一步 Decode」。**多付的是长请求的等待利息，换来的是所有短请求不再被长尾堵死。**

代入数字看权衡。$L = 8192$，$C = 512$，则 $m = 16$。不分块时，一个 8192 token 的 Prefill 若占 GPU 约 2 秒，批里 16 条 Decode 请求全部等 2 秒（TTFT ≈ 2000 ms）。分块后，假设每块 $t_{\text{chunk}} \approx 125$ ms、块间平均等待 $\bar{t}_{\text{wait}} \approx 50$ ms：Decode 请求的 TTFT ≈ `$125 + t_{\text{step}} \approx 130$ ms`——**下降了 90% 以上**；长请求自身 TTFT ≈ `$16 \times (125 + 50) = 2800$ ms`——**上升了 40%**。这就是「吞吐优先」的典型取舍：系统层面每单位时间处理的请求变多，代价是超长请求的单点变慢。对大多数在线服务而言，p99 延迟由短请求主导，这笔交易几乎总是划算的。<span class="marginnote">如果想让长请求也快，就只能「不加块间等待」——即独占 GPU 做完 Prefill，那就是回退到静态批处理。Chunked Prefill 的选择本质是「分时复用 vs 独占」的经典折中，与操作系统时间片轮转同构。</span>

## 5 辨析｜易错点

**辨析｜易错点：**

- **误区一：以为 Chunked Prefill 会「省计算」。** 不会。分块只是把同一份计算拆开调度，总 FLOPs 不变，KV 显存也不变。它省的是**别人被堵住的时间**。
- **误区二：以为它把 Prefill 变成 Decode。** 不是。每块 Prefill 内部仍是标准的 Prefill 计算（并行处理 $C$ 个 token 的前向），只是调度上「每步只放 $C$ 个」；Decode 每步仍只有 1 个 token。二者在调度器里地位等同，但计算性质完全不同。
- **误区三：以为块越大越好。** 块大则单步吞吐高，但单步延迟与队头阻塞也大；块小则调度精细、抢占代价小，但长请求 TTFT 涨。$C$ 是延迟与吞吐的旋钮，没有免费午餐。
- **误区四：以为只有 Prefill 才需要 chunk。** 一条超长 Decode（生成到几千 token）同样会长期霸占批——但它的每步只有一个 token，天然不会垄断算力，所以不需要 chunk。Chunked 专治「一步吃掉全批算力」的 Prefill。

## 6 小结

- **两个病灶**：长 Prefill 造成队头阻塞（短请求 TTFT 雪崩）与资源错配（Compute-Bound 挤占 Memory-Bound）。
- **分块思想**：把 $L$ token 的 Prefill 切成 $m = \lceil L/C \rceil$ 块，每块与一步 Decode 地位等同，交错调度。
- **不省计算**：总 FLOPs 与 KV 显存不变，变的是调度粒度与抢占代价。
- **一个旋钮**：`max_num_batched_tokens` 决定块大小；`remaining_prefill_tokens` 跟踪进度。
- **TTFT 权衡**：长请求 TTFT 从 $t_{\text{full}}$ 涨到 $m(t_{\text{chunk}} + \bar{t}_{\text{wait}})$，短请求 TTFT 从 $t_{\text{full}}$ 降到 $t_{\text{chunk}} + t_{\text{step}}$；8192/512 的例子中短请求提升 90%+、长请求付出 40% 利息。
- **三大辨析**：不省算力、不是 Decode、块大小是旋钮。
- **公共解**：TensorRT-LLM 的 In-flight Batching、SGLang 调度器都有等价机制，只是命名与默认值不同。

在下一节，我们换一个角度继续榨吞吐：很多请求的**前缀完全相同**——system prompt、RAG 上下文、few-shot 示例。与其让它们各自算一遍同样的 KV，不如让它们共享。这就是 **Prefix Caching：共享前缀的缓存复用**。
