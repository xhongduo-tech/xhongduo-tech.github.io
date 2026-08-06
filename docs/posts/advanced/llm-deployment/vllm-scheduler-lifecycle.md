---
title: vLLM 调度器源码分析（一）：请求生命周期
date: 2026-08-07
---

# vLLM 调度器源码分析（一）：请求生命周期

<div class="epigraph">
<p>计算机科学家的主要挑战，是不被自己制造的复杂性弄昏头脑。</p>
<footer>—— 艾兹格 · 迪杰斯特拉（Edsger W. Dijkstra），1972 年图灵奖演讲《谦逊的程序员》</footer>
</div>

<div class="article-byline">
<p>第四级 · 大模型部署 ｜ LLM推理引擎体系 vLLM 调度器（一） ｜ 2026-08-07</p>
</div>

## 为什么从调度器开始

前面我们用几篇文章把 vLLM 的「肌肉」讲清楚了：**PagedAttention** 把 KV Cache 切成 16 token 一块的物理页，**块表（Block Table）**把逻辑连续翻译成物理离散，**Continuous Batching** 让一批请求在每一步都动态进出，**Chunked Prefill** 把长输入切成小段塞进 decode 的空隙，**Prefix Caching** 让共享前缀不再重复计算。这些机制各自精彩，但它们需要一个「总指挥」来决定每一步到底让谁上 GPU、谁继续等、谁被赶下去——这个总指挥就是**调度器（Scheduler）**。

调度器是 vLLM 推理循环的**心脏与大脑**：每一次模型前向跑什么、跑多少 token、哪些请求被抢占、哪些块要被复制，全部由它在 CPU 侧先算好。<span class="marginnote">vLLM 是典型的「中央调度 + 执行分离」：决策在 CPU 上做，执行在 GPU 上做，二者通过每步一次的数据交换握手。这也是它与训练框架最大的分野之一，可回顾本专题《推理框架与训练框架的本质区别》。</span>本篇文章我们只做一件事：跟踪一条请求从进来到离开，走遍调度器里它的每一个状态。下一篇文章再挖最刺激的部分——显存不够时的抢占与换入换出。

## 1 调度器在推理循环中的位置

先看一张「全局地图」。vLLM 的入口是 `LLMEngine`（或离线 `LLM`），引擎每走一步，就调用一次调度器，拿调度结果驱动 GPU 执行：

```python
# LLMEngine.step() 的简化骨架（V0 语义）
def step(self) -> List[RequestOutput]:
    # 1) 调度器决定这一步跑谁、跑多少 token
    seq_group_metadata_list, scheduler_outputs = self.scheduler.schedule()
    if not scheduler_outputs.is_empty():
        # 2) 把调度结果交给执行器，在 GPU 上做一次前向
        output = self.model_executor.execute_model(
            seq_group_metadata_list, scheduler_outputs.blocks_to_copy)
        # 3) 处理模型输出：追加 token、判断停止、生成 RequestOutput
        self._process_model_outputs(output, scheduler_outputs)
        # 4) 回收已经完成的请求占用的资源
        self.scheduler.free_finished_seq_groups()
    return self._output_from_pending(...)
```

**每一步前向都被调度器的决策驱动**，这句话值得停下来重复一遍。调度器在 CPU 侧产出三个关键东西：一是 `seq_group_metadata_list`（这一步要前向的序列及其块表、采样参数），二是 `blocks_to_copy`（decode 阶段因块共享触发的 copy-on-write 复制指令），三是各类预算统计。GPU 只负责「照单执行」。<span class="marginnote">`blocks_to_copy` 来自 PagedAttention 的写时复制：多条序列共享同一物理块时，一旦某条要写入新 token，就把共享块复制一份再写。机制细节见《PagedAttention：KV Cache 的页式内存管理》。</span>

调度器这个概念借自操作系统：第三级《操作系统》里的进程就绪队列、时间片与抢占调度，在这里以「请求为进程、KV 块为资源、token 为时间片」的形式原样重现。如果你学过操作系统课程，会发现我们一直在熟悉的地图上行走——只是「资源」从 CPU 周期换成了显存。

## 2 核心对象：Sequence 与 SequenceGroup

调度器操作的对象不是裸的「请求」，而是两个层级分明的结构。

**序列（Sequence）**：一条独立的 token 流，拥有自己的 `prompt_token_ids`、不断增长的 `output_token_ids`、自己的块表（Block Table）和采样状态。它是调度与显存管理的基本单元。

**序列组（SequenceGroup）**：一个用户请求对应一个序列组；组内包含一条或多条序列，共享同一份 `SamplingParams`。为什么会有多条序列？因为一次请求可以要求并行采样多条候选（`n > 1`），甚至用 `best_of > n` 先多采几条、再按累积对数概率挑最好的。<span class="marginnote">组内序列数量：`num_seqs = best_of if best_of is not None else n`。多条序列共享前缀对应的物理块，但各有各的输出分支——这正是「共享前缀」最自然的来源，也是 PagedAttention 块复用能省显存的原因。</span>

在 V0 里，`SequenceGroup` 是一个**可变的状态对象**：它的 `state` 字段记录了它处于什么阶段、被抢占了几次、交换出去多少块。调度器每一步都在原地修改这些字段。这个设计是 V0 的核心特征，也是后来 V1 重写时要推翻的首要对象（见本专题《vLLM V0 到 V1 架构演进》）。

## 3 请求生命周期的五段状态机

每条序列的状态由枚举 `SequenceStatus` 表示，它是理解一切调度行为的罗盘：

```python
class SequenceStatus(enum.Enum):
    WAITING = enum.auto()                 # 排队等待调度（可能是新请求，也可能是被抢占的）
    RUNNING = enum.auto()                 # 正在 GPU 上被前向
    SWAPPED = enum.auto()                 # KV 块被换到 CPU 内存，暂停
    FINISHED_STOPPED = enum.auto()        # 因 stop 条件（字符串/token）正常结束
    FINISHED_LENGTH_CAPPED = enum.auto()  # 达到 max_tokens 长度上限
    FINISHED_ABORTED = enum.auto()        # 被显式中止
    FINISHED_IGNORED = enum.auto()        # 被调度器忽略（如 prompt 超长）
```

调度器内部维护三个队列，恰好对应三种「非终结」状态：

| 队列 | 对应状态 | 存放内容 |
| --- | --- | --- |
| `self.waiting` | WAITING | 新到达、等待首次 prefill 的请求，以及被 RECOMPUTE 抢占后回到起点的请求 |
| `self.running` | RUNNING | 已经完成 prefill、正在逐 token decode 的请求 |
| `self.swapped` | SWAPPED | KV 块被换到 CPU、等待 GPU 有空位再换回的请求 |

**序列的状态与它所在的队列一一对应**：调度器把所有 RUNNING 组放进 `self.running`，每步遍历这些队列来决定取舍。三条队列加四种终结态，就构成了请求完整的生命周期。要特别记住：**「FINISHED」不是瞬间发生的**，从检测到停止到真正释放显存之间，还隔着一句 `free_finished_seq_groups()`。

## 4 公式解析：一块 KV 能装多少 token

调度器的每个决策最终都要落到「显存够不够、预算够不够」上，而这两者都可以用一个简单公式预估。**理解 KV 块的计数，是读懂调度器一切取舍的前提。**

每条序列渐进增长的 KV 块数服从：

$$
n_{\text{blocks}}(G) = \left\lceil \frac{\text{prompt\_len}(G) + \max\_tokens(G)}{\text{block\_size}} \right\rceil \times n_{\text{seqs}}(G)
$$

对这条式子做三步拆解：

- **第一步，看懂分子**：`prompt_len + max_tokens` 是这条请求在最坏情况下要存储的 token 总数——KV Cache 存的是「已经计算过」的每个 token 的 K/V 向量，所以一条序列从头到尾最多占这么多位置。<span class="marginnote">这里没算 Prefix Caching 的复用：若多条请求共享同一前缀，实际占用会远小于上式。公式给出的是「独立部署、无共享」时的上限，是最安全的估算，详见《Prefix Caching：共享前缀的缓存复用》。</span>
- **第二步，理解上取整**：块是固定大小（默认 `block_size = 16`）的存储单位，19 个 token 也需要 $\lceil 19/16 \rceil = 2$ 块。块的固定粒度让「浪费」不可避免，但也让「分配/释放」退化成 O(1) 的指针操作。
- **第三步，乘以组内序列数**：组内每条序列各算各的块表，所以块数按 `num_seqs` 线性放大。

数值例子：请求 prompt 1000 token、`max_tokens = 500`、`block_size = 16`、`n = 1`，则上限约 $\lceil 1500/16 \rceil = 94$ 块。把每块 KV 的字节数代入《KV Cache 显存占用估算与数值实例》中的公式，立刻能估出这条请求要吃多少显存。

每步的**调度预算**则服从第二条式：

$$
N_{\text{batched}} = \sum_{G \in \text{prefill}} \text{chunk}(G) + \sum_{G \in \text{decode}} 1 \; \le \; \max\_num\_batched\_tokens
$$

其中 prefill 组每步只贡献它本轮分到的块大小 `chunk(G)`（Chunked Prefill 下通常是一个较小的数），decode 组每步只前向 1 个新 token。**一次前向能处理的 token 总数被 `max_num_batched_tokens` 死死卡住**，批内的序列总数又被 `max_num_seqs` 卡住——这两个配置就是调度器预算的「天花板」，调参的根本就是在显存与吞吐之间挪动这两块天花板（本专题压测与调优篇会专门展开）。

## 5 schedule() 主循环：running → swapped → waiting

每步调度入口是 `Scheduler.schedule()`，其顺序固定为三段：

1. **`_schedule_running`（先服务正在跑的）**：遍历 `self.running`，逐组把「本步新增 token 数」计入预算；预算不够或批已满时，从优先级最低的一组开始触发**抢占**（下一篇的主角）。decode 组每步只新增 1 个 token，所以这一步通常把预算的大头留下来。
2. **`_schedule_swapped`（再换入被换出的）**：遍历 `self.swapped`，若 GPU 有空闲块且预算允许，就把 KV 块从 CPU 换回 GPU，把组从 swapped 队列移入 running 队列。
3. **`_schedule_prefills`（最后安排新请求）**：遍历 `self.waiting`，对每个新请求执行 prefill；若 prompt 太长超过 `prompt_limit = min(max_model_len, max_num_batched_tokens)`，直接标记 `FINISHED_IGNORED` 并丢弃；否则按剩余预算决定一次性 prefill 还是分块。<span class="marginnote">这个顺序不是随意的：<strong>decode 是延迟敏感型的在线服务语义</strong>——已经在吐字的请求应该优先被继续服务，而不是让一个刚来的大 prefill 卡住所有人的下一个 token。这是 continuous batching 与朴素 batching 在服务质量上的根本差别。</span>

调度用的预算对象是 `SchedulingBudget`，它同时追踪两条红线：token 预算（`token_budget`，即 `max_num_batched_tokens`）与序列数预算（`max_num_seqs`）。核心方法是 `can_schedule(num_new_tokens, num_new_seqs)`——任何一组要进批，都必须同时满足两个上限，这也是「一个请求因组内序列数太多而被拒批」的最直接原因。

**重点在于**：三段顺序把「优先权」编进了代码本身——正在 decode 的请求 &gt; 被换出的请求 &gt; 新来的请求。这不是实现细节，而是 vLLM 服务质量的契约。

## 6 一次请求的完整旅程

把上面所有部件串起来，一条请求的生命周期如下：

1. **add_request**：`LLMEngine.add_request()` 对 prompt 做分词，得到 `prompt_token_ids`，连同 `SamplingParams` 构造出一个 `SequenceGroup`，交给 `scheduler.add_seq_group()` 放入 `self.waiting`。
2. **首次 prefill**：某一步的 `_schedule_prefills` 选中它，构造 `SequenceGroupMetadata`（含 `is_prompt=True`、`token_chunk_size`），进前向计算，KV 写入新分配的物理块。
3. **RUNNING**：prefill 完成后组被移入 `self.running`。此后每一步它在 `_schedule_running` 里拿到 1 个新 token 的配额，decode 一个 token，必要时触发 copy-on-write 复制共享块。
4. **停止检测**：每步 `_process_model_outputs` 里，`_check_stopped()` 检查长度上限、EOS、stop 字符串，命中即把序列置为对应的 `FINISHED_*`。
5. **free_finished_seq_groups**：当组内所有序列都终结，`free_finished_seq_groups()` 释放全部 KV 块与 LoRA 资源，同时产出 `RequestOutput` 返回给调用方。

这个旅程有一个关键性质：**请求的每一步都完全由调度器的状态驱动，GPU 没有自主权**。所以任何一次「卡住」「变慢」「OOM」，追根溯源几乎都能在调度器的三条队列与四个终结态上找到证据——这就是为什么「读懂调度器」是排查线上问题的第一步。

## 7 辨析｜易错点

**辨析｜易错点：WAITING 队列里不只有新请求。** 被 RECOMPUTE 抢占的请求也会被放回 `self.waiting`，并且它们的 `is_preempted` 标志会被置位。<span class="marginnote">因此看到「waiting 队列很长」不要下意识以为全是新请求——可能是显存吃紧导致大量回炉重算的旧请求在排队。区分它们要看组状态里的 preempted 字段。</span>

**辨析｜易错点：Sequence 与 SequenceGroup 是两个量级。** 「请求」是产品视角，「序列组」是调度视角，「序列」是显存与计算视角。三者层层包含：一个请求 → 一个序列组 → `num_seqs` 条序列。调试日志里的 `seq_id` 指序列，`request_id` 指请求，别混用。

**辨析｜易错点：FINISHED 不等于资源已释放。** 序列进入 `FINISHED_*` 后，它的 KV 块仍占着显存，直到 `free_finished_seq_groups()` 被调用。对批量离线推理，理解这个延迟释放有助于解释「明明没有请求了，显存占用还在高位」的现象。

## 8 小结

- 调度器是 vLLM 推理循环的**总指挥**：每步前向跑什么、跑多少 token，全由 CPU 侧的 `schedule()` 决定。
- 核心对象分两层：**Sequence**（token 流 + 块表）与 **SequenceGroup**（一个请求，`num_seqs` 条序列，共享采样参数）。
- 生命周期由 **`SequenceStatus` 枚举 + 三条队列**承载：waiting / running / swapped，加四种终结态。
- KV 块数与每步 token 预算可用两条公式估算，`max_num_batched_tokens` 与 `max_num_seqs` 是预算天花板。
- 调度顺序编码了优先权：**running &gt; swapped &gt; waiting**；超长 prompt 会被 `FINISHED_IGNORED`。
- 请求每一步都被调度器状态驱动，排查线上异常的第一步就是看三条队列与终结态。

在下一节，我们进入这个系统最「紧张」的时刻：当 GPU 的 KV Cache 不够用时，调度器如何把人赶下去、又如何把人请回来——这就是**抢占与换入换出**。
