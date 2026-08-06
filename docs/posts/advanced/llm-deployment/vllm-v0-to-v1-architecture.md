---
title: vLLM V0 到 V1 架构演进
date: 2026-08-07
---

# vLLM V0 到 V1 架构演进

<div class="epigraph">
<p>一个能运行的复杂系统，几乎总是从一个能运行的简单系统演化而来。</p>
<footer>—— 约翰 · 高尔（John Gall），《系统论》，1975</footer>
</div>

<div class="article-byline">
<p>第四级 · 大模型部署 ｜ LLM推理引擎体系 vLLM 架构演进 ｜ 2026-08-07</p>
</div>

## 为什么 V1 值得单独一篇

vLLM 在 2023 年靠 PagedAttention 一战成名，它的 V0 引擎跑遍了大半个推理世界。但「能跑」和「架构健康」是两回事：随着功能越加越多，V0 的调度器开始不堪重负——可变状态、阶段二分、调度与执行串行，这些债务让新特性（投机解码、多模态、长上下文）越加越吃力。于是 vLLM 团队选择**推倒重写**，也就是 V1 引擎：它以 `VLLM_USE_V1=1` 在 v0.7 系列实验性登场，随后在 v0.8 系列逐步铺开并成为默认引擎。

理解 V0 → V1 的演进，是理解「推理引擎到底在解决什么问题」的绝佳窗口：V1 不是换了几个函数，而是**重新划定了调度器、执行器、客户端之间的边界**。<span class="marginnote">V1 并没有抛弃 V0 的核心算法——PagedAttention、continuous batching、prefix caching 的思想全部保留。改的是「引擎的组织方式」：谁拥有状态、谁做决策、数据怎么流动。</span>本篇文章讲三件事：V0 的结构性问题、V1 的三个核心转变、以及重写带来的收益与代价。

## 1 V0 的结构性问题清单

先给 V0 的调度与执行画一幅「问题地图」。这些不是 bug，是**架构性债务**：

- **调度与执行串行**：每一步先 `schedule()` 再 `execute_model()`，CPU 上的调度决策没有和 GPU 上的计算重叠。GPU 越强，CPU 调度开销占比越刺眼。
- **阶段二分**：prefill 与 decode 被当成两种「形态」，调度器为它们分别写代码路径。Chunked Prefill 是补丁式的解法——它让 prefill 可以塞进 decode 的空隙，但「形态」的二分依然刻在代码里。
- **可变状态**：`SequenceGroup` 是每一步都被原地修改的大对象，跨步之间藏着隐藏依赖，导致调度器难以并行、难以单测、难以重放。
- **块管理分裂**：V0 存在两套块管理器（BlockManagerV1 与带前缀缓存的 V2），同一个调度器要同时兼容两套实现，心智负担沉重。
- **元数据以 Python 对象传递**：`SequenceGroupMetadata` 等对象在引擎内层层传递，跨进程、跨卡时序列化成本高，扩展性受限。

**核心矛盾可以一句话概括：V0 把「调度」写成了一部被各种特性不断打补丁的顺序剧本。** V1 的目标，就是把这本剧本改写成「可组合的纯函数 + 清晰的边界」。

## 2 V1 的设计哲学：调度是纯函数

V1 最本质的转变，是把调度从「引擎的内脏」中抽出来，变成一个**可以独立运行的纯函数**：

```
输入（SchedulerInput）                     输出（SchedulerOutput）
──────────────────────                    ──────────────────────
EngineCoreRequest 列表      ──────────►   scheduled_new_reqs      （首次调度的新请求）
KV 缓存状态（分配器）                      scheduled_cached_reqs   （继续 running + 恢复的抢占请求）
预算 / 上限配置                           num_scheduled_tokens    （每条请求本步要算的 token 数）
```

在 V1 里，`SequenceGroup` 变成了一个 **frozen dataclass**：它本身不可变，只有少数每步必需的产物（如已缓存 token 数）放在一个显式的 `cached` 字段里。<span class="marginnote">「纯函数 + 不可变输入」意味着调度器可以脱离 GPU 做单元测试、可以回放同一步的输入、可以轻易地在不同调度策略之间切换——这是 V0 的可变状态完全做不到的。</span>调度器每步产出一个 `SchedulerOutput`，执行器据此跑一次前向，再把结果交回 `update_from_outputs()` 推进状态。**调度不再「顺手改」任何共享对象，而是「产出新状态」。**

## 3 进程拓扑：EngineCore 与 ZMQ

V1 把推理核心搬进了**独立进程** `EngineCore`，与客户端进程之间用 ZMQ + msgspec 通信。这是一个比「调度器重写」影响更深远的决定：

- **客户端进程（LLMEngine）** 负责请求的预处理（分词、多模态编码）与输出的后处理（detokenize、聚合、拼 `RequestOutput`）。
- **EngineCore 进程** 负责调度 + 执行 + KV 管理，内部跑一个忙碌循环：

```python
# vllm/v1/engine/core.py 的简化忙碌循环
def busy_loop(self):
    while True:
        # 1) 从输入队列取新请求（ZMQ，msgspec 反序列化）
        for req in self.input_queue.get_all_nowait():
            self.scheduler.add_request(req)
        # 2) 调度：纯函数，产出 SchedulerOutput
        scheduler_output = self.scheduler.schedule()
        # 3) 执行：交给 executor / worker 在 GPU 上前向
        model_runner_output = self.executor.execute_model(scheduler_output)
        # 4) 推进状态并产出输出
        engine_core_outputs = self.scheduler.update_from_outputs(model_runner_output)
        self.output_queue.put_nowait(engine_core_outputs)
```

请求对象 `EngineCoreRequest` 用 msgspec 做序列化，字段全部扁平化（request_id、prompt_token_ids、sampling_params、多模态特征）。<span class="marginnote">对比 V0 里 Python 对象直接传递，msgspec 把跨进程的开销降了一个数量级，也让「客户端在别的机器上」成为可能——这是后续 PD 分离等分布式形态的伏笔。</span>**进程隔离带来的最大红利是重叠**：客户端在做 detokenize 和下一条请求的预处理时，EngineCore 已经在为当前这批请求跑 GPU 前向。

把「有状态的核心关进独立进程、用轻量序列化协议通信」，正是《分布式系统》课程反复出现的主题：换来的是故障隔离与计算重叠，代价是序列化与消息传递的开销。V1 的进程拓扑就是微服务式「状态外置」思路在单机推理引擎内部的一次实践。

## 4 统一 token 视角：num_computed_tokens 与 num_tokens_with_spec

V1 调度器的核心心智模型，是给每条请求维护两个数字：

- **`num_computed_tokens`**：已经算过、KV 已存在的 token 数；
- **`num_tokens_with_spec`**：这条请求**总共需要算**的 token 数（prompt + 已生成 + 投机解码的额外 token）。

调度的目标只有一个：**让 `num_computed_tokens` 追上 `num_tokens_with_spec`**。每步为每条请求推进若干个 token，推进多少由预算决定。这样一来，prefill、decode、投机解码在调度器眼里**不再是三种形态，而是同一个操作的不同推进量**——「chunked prefill 是天然形态，而非补丁」。<span class="marginnote">这是 V1 相对 V0 最优雅的抽象收敛：V0 用 `is_prompt` 标志区分阶段，V1 用「还差多少没算」这一个标量统一了所有情况。</span>

## 5 公式解析：V1 如何填满每一步的预算

把「统一 token 视角」落到预算公式上。对每条请求 $r$，本步分配的 token 数是：

$$
\text{num\_new\_tokens}(r) = \min\Big( \text{num\_tokens\_with\_spec}(r) - \text{num\_computed\_tokens}(r),\; B_{\text{remain}}(r) \Big)
$$

总约束是：

$$
\sum_{r} \text{num\_new\_tokens}(r) \le \max\_num\_batched\_tokens, \qquad
\sum_{r} \text{num\_seqs}(r) \le \max\_num\_seqs
$$

对这条式子做三步拆解：

- **第一步，看懂两个约束**：第一条把每步前向的 token 总数卡在 `max_num_batched_tokens` 内；第二条把批内的序列总数卡在 `max_num_seqs` 内。两个上限同时生效，缺一不可。
- **第二步，理解「追赶」语义**：`num_tokens_with_spec − num_computed_tokens` 就是「还没算完的部分」。对一条正在 decode 的请求，这个差值通常等于 1（每步生成一个新 token）；对一条刚进来的请求，差值等于整段 prompt——于是它天然就是一次「大推进」，即 prefill。
- **第三步，数值感受一下「填缝」**：设 `max_num_batched_tokens = 8192`，此刻 running 里有 64 条 decode 请求，每步只需 $64$ 个 token。V0 的阶段二分在这里容易留出「纯 decode」的空隙；V1 则用剩余预算 $\min(8192, \cdot) - 64 = 8128$ 去接一个新到的长 prompt——把 8128 个 prompt token 以 chunked prefill 的形式**塞进同一步**。GPU 的每一步都被填到接近上限，这就是 V1 在长上下文与高并发下吞吐更高的数学原因。<span class="marginnote">同样的「填缝」，V0 的 Chunked Prefill 也能做，但要靠 `is_prompt` 分支 + 单独的状态机去协调，容易在极端情况下留下空洞；V1 是「顺理成章地」就发生了。</span>

## 6 调度器行为差异：抢占与队列

V1 没有取消抢占，也没有取消三态队列，只是换了实现：

- 队列改为 `RequestQueue`（默认 FCFS 的 deque，或按优先级用堆），加上一个 `skipped_waiting` 用于等待异步依赖（如远程 KV、FSM 编译）的请求。
- **抢占**：running 请求在 `kv_cache_manager.allocate_slots()` 分配 KV 槽失败时，调度器弹出最低优先级的 running 请求，释放其 KV 块，重置 `num_computed_tokens = 0`，放回 waiting 队首——本质上仍是 RECOMPUTE 的语义，但实现干净得多。
- **统一的 KV 缓存管理**：V1 只有一种带前缀缓存的块分配器，V0 的 BlockManagerV1/V2 分裂被彻底移除。分配、释放、前缀命中全在 `KVCacheManager` 里闭环。

**值得强调**：V1 的调度器依然是 CPU 侧的纯函数，不跑在 GPU 上；它变得「更聪明」是因为状态更干净、边界更清晰，而不是因为调度本身变成了设备上的算子。

## 7 收益与代价：数据怎么说

**收益**：V1 在长上下文、高并发、多模态负载下普遍取得更高的吞吐与更低的调度开销；prefix caching 与 FP8 量化成为默认路径；`update_from_outputs` 把「输出处理」也纳入了调度循环，使整条流水更可预测。社区基准与 vLLM 官方博客都有不少「V1 吞吐高出 X%」的报告。<span class="marginnote">具体的百分比高度依赖负载：decode 占比高的短对话、prefill 占比高的长文档、多模态混批，收益曲线各不相同。任何「V1 一定更快」的说法都需要用本专题压测与调优篇的方法亲自验证。</span>

**代价**：架构大改带来迁移成本——部分 V0 特有的参数与行为语义改变（例如 `max_num_batched_tokens` 的调度语义更接近「每步预算」而非「批量上限」），需要重新调参；早期 V1 曾缺失若干 V0 功能，成熟度是逐步追平的。V0 与 V1 的并存期也让社区一度困惑「该用哪个」。

## 8 辨析｜易错点

**辨析｜易错点：「V1 是重构」是错的，是重写。** 重构不改架构、只改内部实现；V1 重新划定了进程边界、数据结构与调度模型，属于结构性重写。两者的核心算法一脉相承，但代码几乎不兼容。

**辨析｜易错点：「V1 没有抢占」是错的。** 抢占还在，只是从「可变对象上打补丁」变成了「纯函数里的一个分支」。KV 不足时照样会把人赶下去，只是代价与可观测性更好。

**辨析｜易错点：「V1 取消了 chunked prefill」是反的。** V1 恰恰是把 chunked prefill 变成默认形态——每个 prefill 都可以拆块填进预算空隙，而不是「必要时才分块」。

**辨析｜易错点：「V1 的调度器跑在 GPU 上」是错的。** 调度仍是 CPU 侧纯函数；GPU 上跑的是执行器与算子。V1 的提升来自「边界与重叠」，不是「调度上卡」。

## 9 小结

- V0 的结构性债务：**调度与执行串行、阶段二分、可变状态、块管理分裂、Python 对象传递**。
- V1 三转变：**调度变成纯函数**、**EngineCore 独立进程 + ZMQ/msgspec**、**用 `num_computed_tokens` 统一 token 视角**。
- 预算公式：$\text{num\_new\_tokens}(r) = \min(\text{差量}, B_{\text{remain}})$，两条上限约束让 GPU 每一步都被填满。
- V1 仍保留抢占与三态队列，但实现更干净；KV 缓存管理统一，prefix caching 成为默认。
- 收益依赖负载，迁移有成本；「V1 更快」要用压测亲自验证。

在下一节，我们从「调度器怎么决定跑谁」转向「跑出来的 logits 怎么变成最终输出」——**vLLM 的采样、停止条件与后处理**。
