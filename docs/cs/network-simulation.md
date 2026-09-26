---
title: ns-3 与 mininet
date: 2026-09-08
section: cs
---

# ns-3 与 mininet

<div class="epigraph">
<p>ns-3 是离散事件仿真，时间与丢包完全可控；Mininet 用网络命名空间搭真实内核栈的迷你拓扑，保真协议、难保线速。</p>
<footer>—— 据 ns-3 手册；Lantz et al., Mininet, 2010 整理</footer>
</div>

[抓包](/cs/packet-capture) 在真网上贵。[上一课](/cs/packet-capture) 留下「如何复现」。缺口是**仿真 vs 仿真式仿真（emulation）**。本课不把 Reactor 写完。

## 问题

incast、AQM、BGP 收敛在生产上难复现。ns-3：事件队列推进虚拟时间，可插传播延迟与错误模型，协议是模型或真实栈移植。Mininet：Linux netns + veth + OVS，跑真 TCP，用 tc 限速，CPU 共享则时间戳假。卫星 RTT 在 ns-3 里设数即可。P4 有 BMv2 仿真点名。

不要把仿真曲线当 SLA。

<span class="marginnote">Mininet 2010。ns-3 是科研常用。本课不教安装。</span>

<span class="marginnote">术语翻译：仿真（simulation）就是用「在程序里建数学模型」的手段来做「推演网络行为」的事；Mininet 那类叫仿真式（emulation）——把真协议栈装进假拓扑里跑，协议是真的，链路和主机是演员。</span>

### 保真哪一层要先选

ns-3 控时间；Mininet 保真内核栈不保真线速。曲线不是 SLA。随机种子要记。

## 方法

对照：数学分析 / ns-3 / Mininet / 试验床。画：问题选工具。与 Clos 拓扑：Mininet 能搭，但脊线速假。

<span class="marginnote">数字实例：想在卫星链路上测 TCP，真实 RTT 数百毫秒，一轮实验要干等十几秒；ns-3 里把传播延迟设成 300 ms，虚拟时间一跳就到位，一秒能扫完几十组参数——不必真发一颗卫星。</span>

```mermaid
flowchart TD
  NS3["离散事件"] --> CTRL["可控时间"]
  MINI["netns 仿真"] --> REAL["真内核栈"]
  REAL --> CPU["受主机 CPU 限"]
```

## 机制

抓包在 Mininet 里方便。iperf 在仿真里量的是模型 $C$。SDN 控制器可接 Mininet。DNS TTL 可加速测试。不要用仿真证明互联网政策。

随机种子要记，否则不可复现。

```mermaid
flowchart TD
  E1["t=10ms 事件：包到达"] --> POP["从事件队列弹出最早事件"]
  POP --> ADV["虚拟时间跳到该事件时刻"]
  ADV --> PROC["调用协议模型处理"]
  PROC --> NEW["产生新事件入队"]
  NEW --> POP
  PROC --> DONE["队列为空 ⇒ 仿真结束"]
```

这张图回答：ns-3 的时间为什么「可控」。它不挂钟计时，而是维护一个按发生时刻排序的事件队列，每次弹出最早事件、把虚拟时间直接跳到那一瞬——中间的空闲无需等待，所以千倍速回放可行。

<span class="marginnote">常见误区：初学者容易把仿真吞吐曲线直接当线上性能承诺。Mininet 大流量下受主机 CPU 调度拖累，时间戳会失真；ns-3 里的容量 $C$ 只是你设的模型参数。曲线用来比较方案优劣，不写进 SLA。</span>

## 边界

本课不引入 OMNeT++ 的全部。Reactor/Proactor 是下一课。后课默认：仿真控变量；Mininet 保真栈不保真速率。

把 1000 台主机仿真在笔记本上会自己成为瓶颈。

下一课[Reactor / Proactor](/cs/reactor-proactor)。

## 小结

- ns-3：虚拟时间，模型清晰。
- Mininet：真协议栈，资源共享。
- 选工具看要保真哪一层。
- 后课只引用本课钉死的对象，不从该领域总问题重开。
- 出处：ns-3；Lantz et al., 2010。
