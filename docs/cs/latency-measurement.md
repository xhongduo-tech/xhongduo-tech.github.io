---
title: 延迟测量 cyclictest
date: 2026-09-08
section: cs
---

# 延迟测量 cyclictest

<div class="epigraph">
<p>cyclictest 按设定周期醒来，记录「应当醒来」与「实际醒来」之差，画出内核调度与抢占延迟的尾部。</p>
<footer>—— 据 rt-tests 的 cyclictest 文档；OSADL 对实时 Linux 测量的实践</footer>
</div>

[PREEMPT_RT](/cs/preempt-rt) 声称有界。[EEVDF](/cs/eevdf) 声称尾延迟更好。缺口是 **怎么量**：cyclictest 测的是唤醒延迟，不是应用业务延迟。

## 问题

任务 `clock_nanosleep` 到点，看 now-expected。直方图：max 才是 RT 关心的。缺口：必须隔核、关 HT、注意 [NAPI](/cs/napi) 与 ksoftirqd；tracer 本身扰动。本课不把 OSADL 农场的全部配置复制过来。

<span class="marginnote">用户循环里的计时要用 CLOCK_MONOTONIC_RAW 一类，避免 NTP 跳。对象是调度延迟，不是网卡 RTT。</span>

<span class="marginnote">术语翻译：CLOCK_MONOTONIC_RAW 就是「不做任何校正的单调秒表」——普通 MONOTONIC 允许 NTP 微调时钟快慢（你睡一觉它可能悄悄改掉几毫秒），RAW 则从启动起只进不退、只按硬件节拍走，测微秒级抖动时必须用它。</span>

## 方法

绑 FIFO 高优先级 → 周期性睡眠 → 记录差。对照 [perf](/cs/perf-sampling) 后课：perf 看热点，cyclictest 看最坏唤醒。对照 [blkio](/cs/blkio-cgroup)：I/O 负载是扰动源，应同时加压。对照 fsync：那是存储完成，不是 CPU 唤醒。

```mermaid
flowchart TD
  SL["nanosleep 到点"] --> WK["被调度上 CPU"]
  WK --> D["now - 预期"]
  D --> HIST["直方图与 max"]
```

## 机制

测量把「实时」从口号变成分布。没有加压的 max 没有意义。不要写成示波器使用课。与 [tickless](/cs/tickless-nohz) 后课：无 tick 会改变唤醒路径，测量要注明模式。

结果不可在虚拟机与裸机间直接比，除非声明半虚拟时钟。


实现上：无负载的 cyclictest 只测空闲路径，要同时用 hackbench 或网络打满。虚拟机里测的是宿主注入，不是裸机。CLOCK_MONOTONIC 可能含 NTP 调整，测抖动用 RAW。 读法上只引用[上一课](/cs/preempt-rt)的结论，不把对象换成训练推理或限价簿。

<span class="marginnote">常见误区：初学者空载跑一次 cyclictest，看到 max 只有 20 μs 就宣布系统实时性合格。实际上没加压时最坏路径——软中断风暴、锁竞争、缓存全冷——根本没被踩到；同样配置下打满网络再测，max 可能翻到几百微秒。无压 max 不是乐观估计，是无效数据。</span>

<span class="marginnote">数字实例：同一台裸机上，标准内核在 hackbench 加压下 max 唤醒延迟常到毫秒级；换 PREEMPT_RT 内核后通常压在几十微秒内，硬实时要求（如 $\lt 100\ \mu s$ 的运动控制）看的就是这个差值。这也解释了为什么报告 max 时必须同时写明压力与内核配置。</span>

```mermaid
flowchart TD
  T["定时器到期时刻"] --> IRQ["硬件中断: 时钟事件"]
  IRQ --> SOFT["软中断 / ksoftirqd 处理"]
  SOFT --> SCHED["调度器选中高优先级 FIFO 任务"]
  SCHED --> CTX["上下文切换 + 缓存可能全冷"]
  CTX --> RUN["任务真正开跑: 记录 now-expected"]
  LOAD["外部加压: 网络/IO/hackbench"] -->|"制造抢占与排队"| SOFT
```

这张图回答的问题：直方图里那几十微秒到底花在了哪。唤醒延迟不是单一环节，而是中断、软中断、调度、切换四段排队之和；加压的意义就是逼每一段都走它最坏的那条路，否则测到的只是空闲捷径。

本课在操作系统进阶的「调度进阶 / 公平、实时与能耗」课序里，对象是 **延迟测量 cyclictest**。

- 先修只引用，不重导：上一课的结论当公理，本课只补差。
- 五栏不吞并：不把本课写成大模型训练/推理，也不写成限价簿或权重量化。
- 文献用 OSTEP、McKusick、内核文档与具名会议论文；不发明 arXiv 编号。

## 边界

本课不引入所有 tracing 事件。不保证手机 SoC 的数字可移植。下一课用 cgroup 管 CPU 组：组调度。


版本字段会变，课序钉的是机制对象「延迟测量 cyclictest」，不是某一主线内核的结构体名。
后课默认：唤醒延迟可用周期睡眠直方图刻画。cgroup cpu 控制器如何摊权重与限额，下一课。

## 小结

- cyclictest 量周期唤醒误差的尾部。
- 必须加压与绑核才有解释力。
- cgroup CPU 组调度是下一课。
- 出处：rt-tests cyclictest；OSADL；POSIX 时钟。
