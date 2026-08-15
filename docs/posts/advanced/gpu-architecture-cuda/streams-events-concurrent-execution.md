---
title: 流、事件与并发执行（计算/传输重叠）
date: 2026-08-07
---

# 流、事件与并发执行（计算/传输重叠）

<div class="epigraph">
<p>想走得快，独自前行；想走得远，结伴同行。</p>
<footer>—— 非洲谚语（If you want to go fast, go alone. If you want to go far, go together.）</footer>
</div>

<div class="article-byline">
<p>第四级 · GPU 架构与 CUDA 并行编程 ｜ Kirk & Hwu, Programming Massively Parallel Processors, 4e, Ch20；CUDA C++ Programming Guide §3.2.5 ｜ 2026-08-07</p>
</div>

## 为什么从流开始

前几篇讲的都是「GPU 内部怎么并行」。但一个完整程序的瓶颈，常常根本不在 GPU 内部，而在 **GPU 与 CPU/显存之间**：数据要拷进显存、算完要拷回，PCIe 传输慢、GPU 计算快，两者一前一后排队，GPU 就空等。

**流（stream）** 就是解决这个「排队」问题的工具：让数据搬运和计算**同时进行**。它是把「单 GPU 单次任务」升级成「流水线」的机制，也是 PMPP 第 20 章讲「异构计算集群」的入口。<span class="marginnote">PMPP 第 20 章 "Programming a Heterogeneous Computing Cluster: An Introduction to CUDA Streams" 把流讲成「在时间上重叠、在空间上并行」的引擎；官方语法见《CUDA C++ Programming Guide》§3.2.5 Streams。</span>

## 1 问题：GPU 在等数据

回到《内存层次》篇的图景：一次 kernel 往往要先从主存把输入拷到显存（慢，几百 GB/s 级别），算完再拷回。如果这些操作**串行**排队，总时间是三段相加：

$$
T_{\text{serial}} = T_{\text{copyIn}} + T_{\text{compute}} + T_{\text{copyOut}}
$$

对一个大矩阵，`cudaMemcpy` 可能要花几十毫秒，而 kernel 本身也许只要几毫秒——**GPU 一大半时间在等拷贝完成，执行单元却在空转。** 这与 SIMT 隐藏延迟的思想同构，只是这次「被隐藏的等待」发生在设备边界上。

## 2 流是什么：一条有序执行队列

**流（stream）** 是 GPU 上的一条**按序执行队列**：你把操作（拷贝、kernel、事件）依次提交到流里，GPU 按提交顺序执行它们。不同流之间**相互独立、可以并发**——流是 GPU 端的「任务流水线」，也是并发度的来源。

关键点：

- **默认流（stream 0）**：不指定流时所有操作都在默认流里，默认流默认与其它流「同步」——容易挡住并发，所以想并发就要用非默认流。
- **异步拷贝**：`cudaMemcpyAsync` 把拷贝放进流，调用立刻返回，CPU 不等拷贝完成——这是并发的前提。
- **流水线思维**：把「数据块 i 拷贝 + 计算」组织成两三条流，让「第 i+1 块的拷贝」与「第 i 块的计算」重叠。

```c
cudaStream_t s1, s2;
cudaStreamCreate(&s1); cudaStreamCreate(&s2);
// 流 s1 拷入数据块，流 s2 计算上一块——两者并行
cudaMemcpyAsync(d_A, h_A, bytes, cudaMemcpyHostToDevice, s1);
kernel<<<grid, block, 0, s2>>>(d_A, ...);   // 计算放在 s2
cudaStreamDestroy(s1); cudaStreamDestroy(s2);
```

## 3 公式解析：重叠到底省多少

把串行与重叠做个对比，这是流的核心收益模型。理想情况下：

$$
T_{\text{overlap}} = \max(T_{\text{copy}}, T_{\text{compute}})
\qquad\text{vs.}\qquad
T_{\text{serial}} = T_{\text{copy}} + T_{\text{compute}}
$$

拆解这条公式：

- **$T_{\text{copy}}$**：全部数据传输（拷入 + 拷出）的总时间。
- **$T_{\text{compute}}$**：全部 kernel 计算的总时间。
- **串行**：先拷完再算，时间是和。
- **重叠**：拷贝与计算并行，时间是**两者中较大者**——谁慢，总时间就受谁拖累。

代入实例：$T_{copy} = 20$ ms、$T_{compute} = 5$ ms，串行 $= 25$ ms，重叠 $= \max(20, 5) = 20$ ms——**节省 20%**。若计算更重（$T_{compute} = 30$ ms），重叠时间 $= 30$ ms，几乎不省——**瓶颈在谁，时间就卡在谁**。

**辨析｜易错点：** 重叠不是免费的。它要求 **pinned memory（页锁定内存）**——用 `cudaMallocHost`/`cudaHostAlloc` 分配的主存，否则异步拷贝实际退化成同步拷贝；还要注意「第 i 块的计算」依赖「第 i 块的输入」，流水线组织错了会产生数据竞争。<span class="marginnote">还有一点常被忽略：拷贝和计算能并行，取决于引擎（engine）是否独立。现代 GPU 的拷贝引擎（copy engines）与计算引擎彼此独立，但同一条流的多个操作仍严格串行——所以「拷贝放流 A、计算放流 B」才能真重叠，堆在一条流里没用。</span>

## 4 事件：让流之间能握手

流是并行的，但程序总需要「某件事做完才能做下一件事」。**事件（event）** 就是流之间的「信号灯」：在一个流里打个标记，让另一个流等它。

- `cudaEventRecord(evt, s1)`：在流 s1 的某个位置插一个事件。
- `cudaStreamWaitEvent(s2, evt)`：让流 s2 等到事件发生才继续。
- `cudaEventSynchronize(evt)` / `cudaEventElapsedTime`：CPU 侧等待 / 测量两事件间耗时。

事件最实用的两个场景：**① 跨流依赖**（比如「流 B 的 kernel 必须等流 A 的拷贝完成」，但你又不想让 B 完全同步掉）；**② 精确计时**（`cudaEventElapsedTime` 比 CPU 时钟准，因为它记录的是 GPU 时间轴）。<span class="marginnote">用事件而不是 `cudaDeviceSynchronize` 去协调，粒度更细、性能更好：`cudaDeviceSynchronize` 会把整个 GPU 的所有流都卡住等完，而事件只卡住相关的那一条。</span>

## 5 并发 kernel：一张卡同时跑多个任务

流不仅能重叠「拷贝与计算」，还能让**多个 kernel 同时在一张 GPU 上跑**——只要它们互不依赖、资源够用。典型场景：

- 把一个大任务拆成多个独立的小 kernel，放进不同流，让 SM 之间自然分工。
- 推理服务里多个用户的请求并发执行，各自在一条流上（对应《大模型部署》篇的批处理与 MPS 优化）。

但并发 kernel 不是白送的：多个 kernel 争抢同一批 SM 与显存带宽，若资源不足反而互相拖慢。**经验法则：流的数量够「藏住传输延迟」即可（通常 2–4 条足够），堆太多流只是增加调度开销。** 判断标准永远是实测吞吐，而不是「流越多越好」。<span class="marginnote">这与《占用率》篇的教训一脉相承：占用率/流数都是「潜力指标」，最终要用 Nsight 看真实吞吐。PMPP 第 20 章还讲了把流模型延伸到多 GPU 集群——下一篇《多 GPU 编程》正是这条线的延续。</span>

## 6 多缓冲：把流水线装满

单条流解决了「拷-算-拷」串行，但真要把计算/传输重叠做满，需要把一个大任务**切成若干块，用多条流轮流接力**——这就是**多缓冲（multi-buffering）**，和 CPU 端双缓冲渲染是同一思想。

把一次「拷入 100 MB → 算 → 拷出」拆成 10 块（每块 10 MB），用两条流交替：

```c
for (int i = 0; i < 10; i++) {
    cudaMemcpyAsync(d_in[i], h_in[i], chunk, cudaMemcpyHostToDevice, streamA);
    // 用事件让 streamB 的第 i 块 kernel 等 streamA 的第 i 块拷贝完成
    kernel<<<grid, block, 0, streamB>>>(d_out[i], d_in[i]);
}
```

流水线装满后，理想状态下每个时刻都有「一块在拷贝、一块在计算」，总时间逼近「传输 + 计算」里较慢者，再叠加第一块的启动开销。**分块越多，流水线越深、重叠越满——但块太碎，每块的启动与同步开销占比上升，反而变慢**：所以块的大小要在「够碎以重叠」和「够大以摊薄开销」之间取平衡，通常每块几十 MB、2–4 条流即可。<span class="marginnote">这个「够深 vs 太碎」的权衡，与《占用率与延迟隐藏》篇「更多 warp vs 调度开销」、《共享内存》篇「分块大小 vs bank 冲突」是同构的：<strong>一切并行系统都在「更细的并行」与「每份并行的固定开销」之间找平衡点。</strong></span>

**用 Nsight Systems 的时间线判断重叠度**：如果时间线里「拷贝」与「kernel」的颜色块几乎不相邻、中间隔着空隙，就是没重叠满——加一条流、或把块切得更碎，直到空隙消失。判断永远以时间线为准，这与本专题「先测量、再优化」的总纪律一致。

再补充一个常见误区：不要把「不同流」误当成「线程安全」——流只保证「同一流内按序」，跨流的共享数据竞争仍需事件或锁来协调。**流的并发是「有序的并发」，不是「无序的自由」。** 想清楚每条流里放什么、事件把哪两条流拴住，才是流的正确用法。

## 7 小结

- **流**是 GPU 上按序执行的队列，不同流彼此独立可并发；默认流会阻碍并发，想重叠就用命名流。
- 串行总时间 $T_{copy}+T_{compute}$，理想重叠后 $T=\max(T_{copy},T_{compute})$——瓶颈在哪，总时间卡在哪。
- **异步拷贝**（`cudaMemcpyAsync`）+ **pinned memory** 是重叠的前提。
- **事件**是流之间的信号灯：跨流依赖用 `cudaStreamWaitEvent`，精确计时用 `cudaEventElapsedTime`。
- 并发 kernel 让一卡多任务，但流数量 2–4 条通常足够，以实测为准。

在下一节，我们把视野从单 GPU 拉回系统层面：**统一内存与新特性（页迁移、动态并行、协作组）**。
