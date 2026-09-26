---
title: 定时器轮
date: 2026-09-08
section: cs
---

# 定时器轮

<div class="epigraph">
<p>timer wheel 用分层环把超时时间哈希到槽：O(1) 插入/删除，到期精度与槽宽成反比，适合海量不精确超时。</p>
<footer>—— 据 Varghese and Lauck 对 hashed timing wheels 的经典论述；Linux timer wheel 文档</footer>
</div>

[tickless](/cs/tickless-nohz) 减少了谁来推轮。[TCP](/cs/kernel-tcp-impl) 的 RTO、[NAPI](/cs/napi) 无关但网络栈大量用超时。缺口是 **内核定时器轮** vs hrtimer 红黑树。调度进阶在此收口。

## 问题

百万连接不能每人一棵精确树。wheel：时间模槽数，级联到更粗层。缺口：向前推进可能 cascade 成本；不适合亚微秒；hrtimer 给高精度（DL、nanosleep）。本课不把 cascade 代码写成作业。

<span class="marginnote">jiffies 是轮的经典单位。NO_HZ 下推进发生在必要时。对象是超时集合，不是 RTC 芯片。</span>

## 方法

`add_timer`：算槽，挂链表。tick/hrtimer 回调：处理当前槽，cascade。对照 qdisc：一个按包整形，一个按时间触发函数。对照 [slab](/cs/slab-allocator)：都是内核基础设施。对照用户 timerfd：底层仍可能是 hrtimer。

```mermaid
flowchart TD
  ADD["add_timer"] --> SLOT["哈希到槽"]
  TICK["时间推进"] --> FIRE["到期回调"]
  SLOT --> CASC["溢出级联到粗层"]
```

## 机制

wheel 用精度换规模，使 TCP、邻项、工作队列能活。高精度路径走 hrtimer，避免污染轮。不要写成编译器哈希课。与 [PREEMPT_RT](/cs/preempt-rt)：回调上下文是否可睡决定用哪种定时器。

一个超时该进轮还是进树：

```mermaid
flowchart TD
  NEED{"需要多高的到期精度?"}
  NEED -->|"秒或毫秒级, 数量百万"| W["分层 timer wheel"]
  NEED -->|"亚微秒, 睡眠或 DL"| H["hrtimer 红黑树"]
  W --> W1["O(1) 挂入与删除"]
  W --> W2["精度被槽宽封顶"]
  H --> H1["到期精确, 单次 O(log n)"]
  H --> H2["海量连接下树操作太贵"]
  W2 --> CASC["跨层 cascade 有延迟尖峰"]
```

错误的长时间 cascade 会造成延迟尖峰，这是测量时要看到的。

<span class="marginnote">直觉类比：分层定时器轮像钟表的秒针、分针、时针——秒针（细层）转一整圈，分针（粗层）才走一格。远期定时器先挂在粗层，快到期时被「级联」下放到细层，每个定时器一生只挪几次，插入删除才做得到 O(1)。</span>


实现上：cascade 在跨过粗层边界时可能一次处理很多定时器，造成延迟尖峰。hrtimer 走红黑树，给 nanosleep 和 DL。NO_HZ 下轮子在进入空闲时才推到「现在」。 读法上只引用[上一课](/cs/tickless-nohz)的结论，不把对象换成训练推理或限价簿。

本课在操作系统进阶的「调度进阶 / 公平、实时与能耗」课序里，对象是 **定时器轮**。

- 先修只引用，不重导：上一课的结论当公理，本课只补差。
- 五栏不吞并：不把本课写成大模型训练/推理，也不写成限价簿或权重量化。
- 文献用 OSTEP、McKusick、内核文档与具名会议论文；不发明 arXiv 编号。

<span class="marginnote">数字实例：设细层 256 个槽、槽宽 4 毫秒——一秒内到期的定时器都能精确落槽；一个 1 小时后才到期的先挂「小时层」，轮子跨过边界时才被挪下来。百万条 TCP 连接的超时就是靠这个结构用常数成本兜住的。</span>

## 边界

本课不引入 alarmtimer 的全部时钟基础。不保证用户 POSIX 定时器精度等于 wheel。下一单元安全与启动：Linux capabilities。


版本字段会变，课序钉的是机制对象「定时器轮」，不是某一主线内核的结构体名。
后课默认：海量超时走 timer wheel，精确事件走 hrtimer。进程权能如何切分 root，下一课 capabilities。

<span class="marginnote">常见误区：初学者容易以为定时器轮比红黑树「全面更快」——它只是把「海量、不精确」这一档做便宜了。需要纳秒级精度的 nanosleep 与 deadline 调度，仍要走 hrtimer 的有序树；两条路径并存，各接各的负载。</span>

## 小结

- timer wheel：O(1) 粗粒度超时。
- hrtimer 服务高精度；tickless 改变谁推轮。
- capabilities 是下一单元。
- 出处：Varghese and Lauck；Linux timers。
