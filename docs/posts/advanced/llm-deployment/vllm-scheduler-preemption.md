---
title: vLLM 调度器源码分析（二）：抢占与换入换出
date: 2026-08-07
---

# vLLM 调度器源码分析（二）：抢占与换入换出

<div class="epigraph">
<p>程序员在思考、担心程序非关键部分的速度上，浪费了巨大的时间……我们应该忘掉那些小效率，大约 97% 的时候都要忘掉：过早优化是万恶之源。</p>
<footer>—— 高德纳（Donald Knuth），《带 go to 语句的结构化程序设计》，1974</footer>
</div>

<div class="article-byline">
<p>第四级 · 大模型部署 ｜ LLM推理引擎体系 vLLM 调度器（二） ｜ 2026-08-07</p>
</div>

## 为什么抢占值得单独一篇

大多数时候，continuous batching 的流水线都在安静地运转：请求进来、prefill、decode、出去，调度器像个熟练的交警。但一旦**并发升高或 KV Cache 分配不足**，显存这个硬约束就会把一切都打回原形——总有人要「下 GPU」。调度器把谁赶下去、用什么方式赶、赶下去之后怎么请回来，这就是**抢占（preemption）与换入换出（swap）**。

抢占是推理服务里最容易被低估的「隐形税」：它平时不出现，一出现就表现为**端到端延迟的毛刺、TTFT 的突然拉高、吞吐的锯齿**。许多线上事故的根因不是模型算不动，而是抢占策略在显存压力下反复摩擦。<span class="marginnote">抢占不是 OOM。OOM 是显存直接爆掉、进程崩溃；抢占是调度器在显存告急时「优雅地腾挪」——先让一部分请求暂停，为另一部分腾出块，之后再把暂停的恢复。理解这个区别，是读懂引擎报错日志的第一步。</span>本篇文章我们从源码逻辑讲清三件事：触发条件、两种抢占模式、换入换出的完整过程。

## 1 抢占的触发：KV Cache 是硬约束

回忆上一篇的预算模型：一次前向能容纳的 token 总数被 `max_num_batched_tokens` 卡住，而显存里 KV Cache 的总块数在启动时就已按 `gpu_memory_utilization` 预分配完毕。**decode 阶段的每一个新 token 都要为对应序列分配一个新的 KV 位置**——每 `block_size`（默认 16）个 token 就需要一个新块。当空闲块不够分时，调度器就不能让所有 running 请求都「再走一步」。

于是 `_schedule_running` 里出现了这样的逻辑：对每个 running 组，先问「能不能给这组分配下一轮的块」，即 `block_manager.can_allocate()` 或等价的能力检查；一旦发现**无法让所有组都留在 running**，就按照优先级从低到高开始「请人下车」。<span class="marginnote">优先级默认是 FCFS（先到先服务），由 `Policy` 类的 `sort_by_priority()` 决定；支持自定义优先级时，`max_priority` 越大的请求越晚被赶。被赶的顺序，就是服务质量的公平性所在。</span>

配置层面对抢占频率影响最大的是三个旋钮：`gpu_memory_utilization`（决定 KV 池多大）、`max_num_seqs`（决定并发上界）、`swap_space`（决定 CPU 换出缓冲多大）。调优的思路从来不是「消灭抢占」，而是「把抢占控制在不伤及 SLO 的频率」。

## 2 两种模式：RECOMPUTE 与 SWAP

vLLM 用枚举 `PreemptionMode` 定义两种把人赶下去的方式：

**RECOMPUTE（重算）**：立即释放这条请求的所有 KV 块，把序列组放回 WAITING 队列，并把它的计算进度重置（`num_computed_tokens` 清零）。等它再次被调度时，从头重新做 prefill。**代价是计算**——已经算过的前缀要再算一遍。

**SWAP（换出）**：不释放 KV 块，而是把块的数值**复制到 CPU 内存**（写入 CPU 块分配器），序列组进入 SWAPPED 队列暂停；等 GPU 有空位了，再把块复制回 GPU 恢复。**代价是带宽与 CPU 内存**——搬出去再搬回来各走一遍 PCIe/NVLink 总线。

**两种模式本质上是拿不同的资源换显存**：RECOMPUTE 拿算力换，SWAP 拿带宽换。历史实现里，调度器对仍在 running 的请求默认倾向 RECOMPUTE（避免大批量的块搬运打断流水），对已经被换出过的请求再次遭遇压力时则用 SWAP；后来版本引入基于块数的启发式——组内块少、前缀短时重算便宜，块多、序列长时换出更省。<span class="marginnote">启发式细节随版本演变，不必死记。真正要记住的是判断框架：<strong>短前缀重算便宜，长序列换出更便宜</strong>——这是贯穿整个抢占设计的成本直觉。</span>

## 3 公式解析：重算 vs 换出，哪个更便宜？

抢占模式的选择最终是一个成本问题。我们分别建立两种代价的估算式，再代入实际数字感受量级。

**重算代价**：把长度为 $l$ 的前缀重新 prefill 一遍的耗时近似为

$$
T_{\text{recompute}} \approx \frac{C_{\text{prefill}}(l)}{\Gamma_{\text{prefill}}}
= \frac{2 \times N \times l}{\Gamma_{\text{prefill}}}
$$

其中 $N$ 是模型参数量，$C_{\text{prefill}}(l) \approx 2 N l$ 是长度为 $l$ 的前缀前向所需 FLOPs（每个 token 约 $2N$ 次乘加），$\Gamma_{\text{prefill}}$ 是 GPU 在 prefill 上的实际吞吐（FLOP/s）。<span class="marginnote">这里忽略了前缀已被 Prefix Caching 命中而无需重算的情形——命中时 $l$ 应换成「未命中部分的长度」。这正是《Prefix Caching》一文与抢占叠加后的关键优化点。</span>

**换出代价**：把整条序列的 KV 搬到 CPU 再搬回来，耗时近似为

$$
T_{\text{swap}} \approx \frac{2 \times S_{\text{kv}}(l)}{B_{\text{swap}}}
= \frac{2 \times l \times b_{\text{kv}}}{B_{\text{swap}}}
$$

其中 $S_{\text{kv}}(l) = l \times b_{\text{kv}}$ 是该序列当前的 KV 字节数，$b_{\text{kv}}$ 是**每个 token 每层 KV 的字节数**（由 hidden、KV 头数、head_dim、精度共同决定），因子 2 对应「写出去 + 读回来」，$B_{\text{swap}}$ 是 CPU 与 GPU 之间的有效拷贝带宽。

两条式子拆开看有三点结论：

- **两者都随 $l$ 线性增长**，所以不存在「某一种绝对更优」——比值才是关键：$\dfrac{T_{\text{swap}}}{T_{\text{recompute}}} = \dfrac{2 l b_{\text{kv}}}{B_{\text{swap}}} \cdot \dfrac{\Gamma_{\text{prefill}}}{2 N l} = \dfrac{b_{\text{kv}} \Gamma_{\text{prefill}}}{N B_{\text{swap}}}$，长度 $l$ 恰好约掉。
- **比值不依赖 $l$**，说明对给定的硬件与模型，两种模式的优势区间基本固定，启发式只需按块数粗判即可。
- 代入具体数字：对 70B 模型，$N = 70 \times 10^9$，设 $\Gamma_{\text{prefill}} = 8 \times 10^{14}$（约 800 TFLOPS），$b_{\text{kv}} \approx 320\text{ KB/token}$（80 层、8 KV 头、fp16 的典型值），$B_{\text{swap}} = 32\text{ GB/s}$，则比值 $\approx \dfrac{3.2\times10^5 \times 8\times10^{14}}{7\times10^{10} \times 3.2\times10^{10}} \approx 0.11$——**换出比重算便宜一个数量级**。但 SWAP 还占用 CPU 内存、且搬运动作本身打断流水，所以工程上两者都需要，而不是只选便宜的。

## 4 _preempt_by_recompute：释放块、回炉重算

RECOMPUTE 的代码路径可以压缩成四步：

```python
def _preempt_by_recompute(self, seq_group, ...):
    # 1) 释放该组在 GPU 上的全部 KV 块
    self.block_manager.free(seq_group)
    # 2) 把组从 running 队列剔除，重置进度，标记已抢占
    seq_group.is_preempted = True
    seq_group.state.num_computed_tokens = 0
    # 3) 放回 waiting 队首，尽可能早地被重新调度
    self.waiting.appendleft(seq_group)
```

注意两个容易被忽略的细节。**其一**，`block_manager.free()` 释放的是 GPU 块，不是序列数据——`prompt_token_ids` 和已生成的 `output_token_ids` 都还在内存里，所以「重算」是指重新对前缀做 prefill、重新生成 KV，而不是重新分词。**其二**，放回的是 `appendleft`（队首），因为被抢占的请求已经等过一轮了，再让它从队尾排起等于双重惩罚。<span class="marginnote">如果开了 Prefix Caching，重算时前缀里被其他请求共享过的部分会命中缓存，实际重算的往往只是「公共前缀之后」的那一段——抢占的代价因此被大幅稀释。</span>

被 RECOMPUTE 抢占的组会出现在 `SchedulerOutputs.preempted` 列表里，带上它的 `PreemptionMode`，供引擎统计与日志输出。这类事件通常伴随一条形如「Sequence group ... is preempted by PreemptionMode.RECOMPUTE ...」的警告。

## 5 _preempt_by_swap 与换入换出

SWAP 路径则要动「真金白银」的字节：

```python
def _preempt_by_swap(self, seq_group, blocks_to_swap_out, ...):
    # 1) 把 GPU 块复制到 CPU 块分配器，返回被换出的块映射
    num_swapped_blocks = self.block_manager.swap_out(seq_group)
    # 2) 组进入 swapped 队列
    self.swapped.append(seq_group)
    # 3) 记录换出块，供 worker 在 GPU 上执行真正的拷贝
    blocks_to_swap_out.update(...)
```

**换出（swap_out）**：块管理器把该组所有物理块的 KV 数据从 GPU 显存复制到 CPU 内存，并让块表指向 CPU 块。此时 GPU 显存被立即释放，序列进入 SWAPPED 状态。

**换入（swap_in）**：当某一步 `_schedule_swapped` 发现 GPU 有空闲块且预算允许时，调度器做逆操作——块管理器把 CPU 块复制回 GPU 显存，块表重新指向 GPU 块，序列组移入 running 队列，继续 decode。**被换出的请求优先级高于新来的请求**，因为它们已经持有 CPU 中的 KV 数据，放着不恢复是双重浪费。

CPU 内存的缓冲池大小由 `swap_space` 配置决定（默认 4 GiB）。<span class="marginnote">如果 CPU 内存也被占满，swap_out 会失败，调度器只能回退到 RECOMPUTE。因此「加 `swap_space`」和「加显存」是两条不同的救命通道：前者增大换出缓冲，后者减少换出需求。</span>换入换出在每个 `SchedulerOutputs` 里通过 `blocks_to_swap_out` / `blocks_to_swap_in` 字段传递，真正的大块拷贝发生在 worker 的 cache engine 里，与调度决策解耦。

这套「换出 / 换入」几乎就是操作系统虚拟内存换页在 KV Cache 上的翻版：被换出的页在换入时若被修改要写回磁盘，这里的 KV 块则只读、干净得多。对照第三级《操作系统》的页面置换，以及《计算机组成原理》中 HBM / DRAM / PCIe 的存储层次，抢占的成本直觉会清晰很多——你已经在用经典的操作系统思维，只是换了硬件。

## 6 工程细节：预算、优先级与统计

几个值得单独记下的工程点：

- **`SchedulingBudget` 与抢占的联动**：`_schedule_running` 在预算不足时调用 `_preempt(victim_seq_group, blocks_to_swap_out)`，把受害者从 running 队列尾部弹出。预算对象用 `subtract_num_batched_tokens` / `subtract_num_seqs` 把被赶走的组占用的额度还回去，保证「赶走多少、腾出多少」账目平衡。
- **`num_cumulative_preemption`**：调度器维护一个累计抢占计数器，既参与模式启发式的判断，也被写进警告日志（如 `total_cumulative_preemption_cnt=1`），方便运维看到「这台机器被抢占过多少次」。
- **抢占只发生在「不得不」时**：调度器不会主动为了「更好」的批而抢占，它只在「不赶人就没法让所有人都前进」的时候动手。这是抢占与主动调度的本质区别。

## 7 辨析｜易错点

**辨析｜易错点：抢占不是中止，请求最终会完成。** 被抢占的请求只是被延迟，`SequenceStatus` 仍是 WAITING 或 SWAPPED，而不是任何 `FINISHED_*`。只有显式 `abort()` 才会进入 `FINISHED_ABORTED`。

**辨析｜易错点：RECOMPUTE 不一定是「白算」。** 开启 Prefix Caching 后，公共前缀会在缓存里被其他请求共享，重算命中的部分几乎零成本。所以「抢占太多 → 关掉 prefix caching」在多数场景是反方向优化。

**辨析｜易错点：SWAP 的隐藏开销在 CPU 内存。** 换出需要 CPU 内存做缓冲，`swap_space` 设得太小，调度器会频繁回退到 RECOMPUTE，反而更慢。观测指标不能只看 GPU 显存，还要看 CPU 内存峰值。

**辨析｜易错点：抢占的代价会以两种不同的毛刺出现。** RECOMPUTE 的代价体现在**重算请求自身的 TTFT 被拉长**（它要重新 prefill）；SWAP 的代价体现在**全体请求的 TPOT 被块拷贝拖慢**（搬移动作与 decode 争抢总线）。排查时先分清是「某个请求慢了」还是「整批都慢了」。

## 8 小结

- 抢占由 **KV Cache 块不足**触发，是显存硬约束下的「优雅腾挪」，不是崩溃。
- 两种模式：**RECOMPUTE**（释放块、回炉重算，用算力换显存）与 **SWAP**（搬块到 CPU，用带宽换显存）。
- 代价公式：重算 $T \approx 2 N l / \Gamma_{\text{prefill}}$，换出 $T \approx 2 l b_{\text{kv}} / B_{\text{swap}}$；**比值与序列长度无关**，由硬件与模型决定优势区间。
- RECOMPUTE 把组放回 waiting 队首并重置进度；SWAP 经 `swap_in`/`swap_out` 在 GPU 与 CPU 块之间搬运，被换出的请求优先恢复。
- 预算对象保证「赶走多少、腾出多少」账目平衡；`swap_space` 与显存是两条不同的救命通道。

在下一节，我们把视角从「V0 的实现」拉高到「引擎的架构」：调度器在 V0 里背负了太多结构性问题，vLLM 团队选择重写——这就是**V0 到 V1 的架构演进**。
