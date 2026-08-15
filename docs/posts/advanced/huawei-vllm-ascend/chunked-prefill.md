---
title: Chunked Prefill 分块预填充与首字延迟
date: 2026-08-07
---

# Chunked Prefill 分块预填充与首字延迟

<div class="epigraph">
<p>长任务不该堵住整条流水线——把它切碎，让每个人都先动起来。</p>
<footer>—— 操作系统时间片思想</footer>
</div>

<div class="article-byline">
<p>第四级 · 华为 vllm-ascend ｜ vllm-ascend 官方文档 推理特性 ｜ 2026-08-07</p>
</div>

## 为什么从 Chunked Prefill 开始

连续批处理解决了「批内同步等待」，但留了一个漏网之鱼：**一个超长 prompt 的 prefill**。prefill 是计算密集的，一个 1 万 token 的 prompt，prefill 计算量可能抵得上几十个 decode 步。如果调度器把整个 prefill 当成一个不可分割的原子任务，它会一口气霸占整卡好几秒——期间所有 decode 请求全部排队，首字延迟（TTFT）瞬间爆炸。Chunked Prefill 的答案，是把 prefill 切成小块，与 decode 交错执行。

## 1 冲突的本质：prefill 与 decode 抢同一张卡

一次迭代里，芯片要么做 prefill（处理一批输入，产生 KV），要么做 decode（为一个已生成序列前进一个 token）。两者都是矩阵乘，但**形态完全不同**：

**prefill 是「胖」的**：一个请求有几千 token 的输入，一次前向就能吃满整卡算力，但耗时长。

**decode 是「瘦」的**：一个序列每次只前进 1 个 token，单序列算力需求小，靠大批量才能喂饱卡。

如果 prefill 不切块，调度器面对「一个 1 万 token 的 prefill + 一堆 decode」时只能二选一：要么先让 prefill 独占几秒，decode 全体等待；要么先伺候 decode，prefill 的 TTFT 又拖到不可接受。

**核心要点**：Chunked Prefill 把选择权从「二选一」变成「兼得」——把 prefill 切成小块，每块只占一个迭代的一小部分，其余算力继续喂 decode。**代价是单个请求的 prefill 总时长略长（被切片穿插），但所有请求的延迟都更平滑。**

## 2 Chunked Prefill 的思想

把 prefill 请求的输入切成若干**块（chunk）**，每块包含一部分输入 token。调度器在每个迭代里可以同时处理：

- 若干 decode 序列（各前进 1 token）；
- 一个或多个 prefill 请求的**当前块**（各处理其 chunk 大小的一批 token）。

于是单个迭代变成「混编班」：一部分算力给 prefill 块，一部分给 decode。prefill 请求在多个迭代里分批完成，直到所有块处理完，才转入 decode 阶段。

直观理解：

```
无 Chunked Prefill：  [===== 长 prefill 独占 =====]  [decode] [decode] ...
有 Chunked Prefill：  [prefill块 | decode decode] [prefill块 | decode] [decode decode] ...
```

**辨析｜易错点：** Chunked Prefill **不降低**单个请求的 prefill 总计算量，也不保证 TTFT 一定低于「prefill 独占」的极端方案。它优化的是**多请求并发下的公平性**：在「单个长请求的绝对延迟」与「全体请求的总体延迟」之间做了交换。判断是否受益，要看整体队列，而不是单请求。

## 3 调度与 PagedAttention 的配合

分块 prefill 能成立，同样依赖 PagedAttention 的显存管理：

**块级增量分配**：prefill 每处理完一个 chunk，只为其新增的 token 分配物理块，显存按进度逐步增长，而不是一次性预留整个 prompt 的空间。

**与 decode 共存**：prefill 块与 decode 序列同批执行时，Attention 内核要同时处理「长度参差的 decode 序列」与「正在增长的 prefill 序列」——这正是昇腾端分页内核要处理的混合场景。

**调度优先级**：为避免某个 prefill 被 decode 无限挤压，调度器通常给 prefill 设优先级或配额，保证它能持续前进，TTFT 有上界。<span class="marginnote">「prefill 会不会被饿死」是调度设计的关键：如果 decode 永远优先，长 prompt 请求可能迟迟出不了第一个字。工程上常用<strong>优先级提升</strong>（prefill 等待越久优先级越高）来保证公平——这与操作系统里「老化机制防饥饿」是同构的。</span>

## 4 公式解析：块大小与延迟的权衡

Chunked Prefill 的块大小（`--max-chunked-prefill-tokens`，记为 $C$）是核心旋钮。设一个 prefill 请求的输入长为 $P$，它会被切成 $\lceil P / C \rceil$ 块。单块 prefill 的耗时约正比于 $C$（在算力线性区）：

$$
T_{\text{块}} \approx \alpha C
$$

单请求的 prefill 总时长（若块被间隔执行）约：

$$
T_{\text{prefill, total}} \approx \lceil P / C \rceil \times (T_{\text{块}} + T_{\text{等待}})
$$

$T_{\text{等待}}$ 是每块之间被 decode 抢走的间隔。两个方向的直觉：

**$C$ 越小**：单块耗时短，decode 被挤压得少，全体延迟更平滑；但块数变多，调度开销与等待间隔上升，单请求 prefill 总时长变长。

**$C$ 越大**：prefill 推进越快，但每块霸占算力多，decode 排队加重。<span class="marginnote">所以块大小是「单请求 prefill 速度」与「全体请求平滑度」之间的旋钮。<strong>经验法则</strong>：以混载场景为主、TTFT 敏感时，把 $C$ 设小一些；prefill 请求稀少时，$C$ 大些反而简单高效。没有万能值，用基准测试定。</span>

## 5 昇腾上的实现与易错点

昇腾后端实现 Chunked Prefill，几个注意点：

**混合批的内核支持**：同批里既有 decode 序列（每序列 1 个新 token）又有 prefill 块（每序列一坨 token），Attention 内核要处理这种「长度极度不齐」的输入。昇腾的分页 Attention 内核需要专门支持，性能随混合比例波动。

**算子形状变化**：prefill 块与 decode 混合，GEMM 的形状每步都变，可能削弱图优化与预编译算子的复用效果。昇腾端要确认混合批路径没有退化到「每步重新编译」。

**显存与 KV 增长**：prefill 分块意味着 KV Cache 在一个请求的 prefill 阶段就动态增长，显存预算与回收要更精细地配合。

**辨析｜易错点：** 开了 Chunked Prefill 后**首启/运行日志变复杂是正常的**——调度步数变多、每步形状在变。不要因为「日志里 prefill 和 decode 混在一起」就以为调度器出错。真正的报警信号是 **TTFT 的 P99 持续恶化或显存水位异常**，而不是「看起来乱了」。

### 什么时候该开 Chunked Prefill

Chunked Prefill 不是默认最优，它有自己的适用边界。判断是否开启，看负载画像：

**适合开**：长 prompt 请求与短请求**混载**的场景——比如既有「贴长文档做问答」（长 prefill）又有「闲聊短句」（大量 decode），分块让两者互不拖累。

**不必开**：所有请求都是短 prompt（prefill 本身很快，切块收益小）或所有请求都是长 prompt 且没有 decode 混入（没有竞争，整段 prefill 反而更简单高效）。

**判断方法**：用基准测试对比「开 / 关」两档，看**全体请求的 TTFT P99** 而不是单请求。如果开启后长请求的 P99 明显改善、短请求没有被显著拖慢，就值得长期开着；如果两档差不多，说明你的负载根本不缺 Chunked Prefill 要解决的问题。

**辨析｜易错点：** 把 Chunked Prefill 当成「让长 prompt 变快」的开关是常见误解。**它不会让单个长请求的 prefill 变快**——相反，由于切片与穿插，单请求的 prefill 总时长通常略增。它换来的是「别让长请求堵住所有短请求」，是公平性手段，不是单请求加速手段。

### 一张图记住三种调度手段的分工

把本专题处理「调度」的三个特性放在一起，各自的职责就清楚了：

- **连续批处理**：解决「批内同步等待」——每步重排，谁完成谁走、谁排队谁进。
- **Chunked Prefill**：解决「长 prefill 堵住 decode」——把 prefill 切块，与 decode 交替。
- **前缀缓存**：解决「重复 prefill」——同一前缀只算一次，后续复用。

三者的共同点是都在 **prefill/decode 的执行节奏上做文章**，但着眼点不同：连续批处理管「批」，Chunked Prefill 管「prefill 的粒度」，前缀缓存管「prefill 的重复度」。**调度优化的全景，就是这三个旋钮的协同**——理解了分工，调参时就知道该拧哪个。

## 6 小结

- **冲突本质**：prefill「胖」（一次吃满算力但耗时长）与 decode「瘦」（靠批量喂饱），二者抢同一张卡。
- **分块思想**：prefill 切成小块，与 decode 同迭代混编，单请求 prefill 略长、全体延迟更平滑。
- **依赖分页**：KV 按块增量分配、混合批 Attention 支持，是分块 prefill 能成立的前提。
- **块大小权衡**：$C$ 小则平滑但单请求慢，$C$ 大则推进快但 decode 排队重——用基准测试定。
- **适用边界**：混载场景值得开、单一形态不必开；它优化公平性而非单请求加速。
- **调度分工**：连续批处理管批、Chunked Prefill 管 prefill 粒度、前缀缓存管 prefill 重复度。
- **昇腾注意点**：混合批内核、算子形状变化、KV 动态增长三件事都要配套。

在下一节，我们从「调度」转向「精度」：**FP16/BF16 与 INT8 量化精度策略**——用多少位存、算，决定了显存与质量。