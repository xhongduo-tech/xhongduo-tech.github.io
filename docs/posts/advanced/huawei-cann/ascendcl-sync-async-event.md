---
title: 同步/异步执行与事件同步机制
date: 2026-08-07
---

# 同步/异步执行与事件同步机制

<div class="epigraph">
<p>真正的并行，是让等待的人不再等待。</p>
<footer>—— 自并行计算格言</footer>
</div>

<div class="article-byline">
<p>第四级 · 华为 CANN 计算架构 ｜ 华为昇腾 CANN 开发指南 ｜ 2026-08-07</p>
</div>

## 为什么从同步/异步开始

昇腾设备是个「异步世界」：任务入队后立即返回，实际执行在后台推进。同步与异步不是风格选择，而是**性能的生死线**——把所有操作写成同步等待，设备利用率会低到惨不忍睹；写成异步却不加同步控制，又会读到半成品数据。事件（Event）就是异步世界里精确的「哨兵」，它让数据搬移与计算真正重叠、让多条流按剧本协作。这一节把同步/异步的执行语义与事件的用法一次讲清。<span class="marginnote">对照 CUDA 事件模型：`cudaEventRecord` / `cudaStreamWaitEvent` / `cudaEventElapsedTime` 在 AscendCL 里都有对应物。理解「事件是流上的路标，等待是流间的握手」这句话，两套 API 就都通了。</span>

## 1 同步执行与异步执行

先看两个执行接口的语义差别：

```
aclmdlExecute(modelId, input, output)          // 同步：调用阻塞至推理完成
aclmdlExecuteAsync(modelId, input, output, stream)  // 异步：入队后立即返回
```

**同步执行**：主机线程阻塞，直到模型执行完毕、输出就绪。语义简单、正确性易保证，代价是**主机在等待期间什么也干不了**。

**异步执行**：任务被排入指定流后立即返回，设备在后台执行。主机可以继续做别的（准备下一批输入、做 CPU 侧逻辑）。代价是**返回不代表完成**——必须在合适的时机同步，否则读到的是旧数据。

异步执行与流深度绑定：**每个异步任务都属于某条流**，流决定了任务在设备上的执行次序。上一节《运行管理》讲的流，在这里兑现了它的全部价值。<span class="marginnote">「异步返回 ≠ 完成」是昇腾新手最容易踩的坑：写完 `aclmdlExecuteAsync` 立刻拷贝输出，得到的是上上次推理的旧结果。铁律：<strong>读取输出前，先对该流或相关事件做同步</strong>。</span>

## 2 事件：流与流之间的握手

事件（Event）是流上的一种特殊任务：**它在流中的位置记录了一个「信号」**。核心接口与语义：

`aclrtCreateEvent(&event)`：创建事件对象。
`aclrtRecordEvent(event, stream)`：把事件记录到指定流的当前位置——当流执行到这个位置时，事件被置为「已发生」。
- `aclrtWaitEvent(event)`：阻塞当前流，直到该事件已发生。
- `aclrtSynchronizeEvent(event)`：阻塞主机线程，直到事件发生。
- `aclrtQueryEvent(event)`：非阻塞查询事件是否发生。

关键认知：**事件不是全局开关，而是「插在流里的路标」**。`aclrtRecordEvent` 给事件「上了发条」（绑定到某流某位置），`aclrtWaitEvent` 让另一条流「在此处等待」。生产者流在完成数据准备后记录事件，消费者流在需要该数据处等待事件——这就是流间协作的完整机制。

## 3 一个经典场景：拷贝与计算重叠

事件最经典的用途，是让**数据搬移与推理计算重叠**。以「边拷边算」的流水线为例：

```cpp
aclrtStream copyStream, computeStream;
aclrtCreateStream(&copyStream);
aclrtCreateStream(&computeStream);

for (int i = 0; i < n; i++) {
    // 1. 下一帧数据在 copyStream 上异步拷入设备
    aclrtMemcpyAsync(dst, src, size,
                     ACL_MEMCPY_HOST_TO_DEVICE, copyStream);
    // 2. 拷贝完成事件
    aclrtRecordEvent(dataReady, copyStream);
    // 3. computeStream 等待数据就绪
    aclrtWaitEvent(computeStream, dataReady);
    // 4. 在 computeStream 上异步执行推理
    aclmdlExecuteAsync(modelId, input, output, computeStream);
}
aclrtSynchronizeStream(computeStream);   // 收尾同步
```

第 1、2 步与第 3、4 步被事件串成「拷贝 → 计算」的接力：`copyStream` 上第 $i$ 帧的拷贝，与 `computeStream` 上第 $i-1$ 帧的计算**同时推进**。设备一侧两条流并行，主机一侧一个循环驱动整条流水线。<span class="marginnote">这套「双流流水线」是昇腾推理引擎的标准骨架，和 CUDA 的「copy-engine overlap」模式完全同构。背后的原理见下一节公式：重叠后总耗时从「搬运 + 计算」降为「二者较大值」。</span>

## 4 公式解析：重叠带来的时间收益

为什么重叠如此重要？设单次推理的搬运耗时 $T_c$、计算耗时 $T_p$。**不重叠**（串行）的总耗时为

$$
T_{\text{串行}} = \sum_{i=1}^{n} (T_c + T_p)
$$

**重叠**（流水线）的总耗时为

$$
T_{\text{重叠}} = T_c + n \cdot \max(T_c, T_p)
$$

拆三步看：

- **第一步，看出处**：串行时每帧都老老实实等搬运完再算；重叠时搬运与计算在两条流上并行，设备「搬运第 $i$ 帧」的同时「计算第 $i-1$ 帧」。
- **第二步，找瓶颈**：重叠后每帧耗时由较慢的一方 $\max(T_c, T_p)$ 决定——系统被「搬运」或「计算」中的瓶颈约束。
- **第三步，算收益**：当 $n$ 很大时，$T_c$ 在 $T_{\text{重叠}}$ 里只出现一次（首帧启动），两者之比约为 $\dfrac{T_c+T_p}{\max(T_c,T_p)}$——搬运占比越高，收益越显著。

**这条式子的工程含义是：先测 $T_c$ 与 $T_p$ 哪个大，再决定优化方向**——若 $T_c > T_p$，加宽搬移带宽、压缩拷贝量（如 AIPP 就地处理）才是治本；若 $T_p > T_c$，才该去优化算子效率。事件只是手段，瓶颈才是问题。

## 5 事件同步的进阶：多事件、多流协同

一个事件不够用？真实场景往往需要多个事件串起更复杂的依赖。以「多输入流水线」为例，常用的是「一组事件各管一个数据就绪信号」：

**每帧一事件**：为第 $i$ 帧的数据就绪创建一个事件，拷贝流记录、计算流等待——事件与数据帧一一对应，互不干扰。

**依赖链**：`aclrtWaitEvent` 可以链式使用——计算流等「搬移完成」，输出流等「计算完成」，事件像路标一样把三条流排成「搬移 → 计算 → 输出」的接力。

**同步的两种粒度**：`aclrtSynchronizeStream`（整条流同步）与 `aclrtSynchronizeEvent`（只等某个事件）——一个「等全部」、一个「等某个」，按需求选粒度。

**记住一条主线：事件是「数据依赖」的显式表达**。哪里存在「数据 A 算完 B 才能开始」，哪里就该放一个事件。<span class="marginnote">事件驱动与 CUDA 的 `cudaEventRecord` / `cudaStreamWaitEvent` 完全同构；更进一步，它对应到操作系统的「条件变量 / 信号量」语义——<strong>「同步」的本质，就是把「数据依赖」翻译成「执行次序约束」</strong>。掌握这个翻译，任何并行系统的同步都一通百通。</span>

## 6 核心术语速查表

本节的术语集中在「同步/异步」语境，整理如下：

| 术语 | 含义 |
| --- | --- |
| 同步执行 | 调用阻塞至执行完成，语义简单 |
| 异步执行 | 入队后立即返回，设备后台执行 |
| 流 | 任务队列，流内有序、流间并行 |
| 事件 | 流上的信号，记录「某位置已到达」 |
| RecordEvent | 把事件记录到流的当前位置 |
| WaitEvent | 阻塞流，直到事件发生 |
| SynchronizeEvent | 阻塞主机，直到事件发生 |
| QueryEvent | 非阻塞查询事件是否发生 |
| 双流流水线 | 拷贝流与计算流用事件接力的模式 |
| 数据依赖 | 后一个任务依赖前一个任务的数据 |
| 重叠 | 搬移与计算并行推进 |

## 7 小结

- **同步执行**阻塞至完成、语义简单；**异步执行**入队即返回、性能高但必须配合同步控制。
- **事件是流上的路标**：`aclrtRecordEvent` 上发条、`aclrtWaitEvent` 流间等待、`aclrtSynchronizeEvent` 主机等待、`aclrtQueryEvent` 非阻塞查询。
- 经典模式是**双流流水线**：拷贝流与计算流用事件接力，让搬移与计算重叠。
- 重叠后总耗时从 $n(T_c+T_p)$ 降为 $T_c + n\max(T_c,T_p)$——瓶颈由较慢的一方决定。
- 事件是「**数据依赖的显式表达**」：先测 $T_c$ 与 $T_p$ 谁大，再决定优化方向。

在下一节，我们将从第 2 篇的「单机推理」进入第 3 篇的「图与算子」——学 **GE 图引擎**，看昇腾如何在整图层面做优化。