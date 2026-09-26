---
title: 软件流水
date: 2026-09-08
section: cs
---

# 软件流水

<div class="epigraph">
<p>模调度让连续迭代重叠：核（kernel）每 II 拍启动一圈，序言与收尾补满流水。目标是吞吐，不是单圈最短。</p>
<footer>—— 据 Rau and Glaeser；Lam, Software Pipelining, 1988；Rau, Iterative Modulo Scheduling 整理</footer>
</div>

上一课[表调度](/cs/list-scheduling)优化单块。循环要**跨迭代重叠**。缺口是软件流水 / 模调度：启动间隔 II 受资源与环依赖约束。本课钉 II、kernel、prologue/epilogue，不写 VLIW 模板全部。

## 问题

表调度一圈内排完再下一圈，ALU 仍闲。流水：迭代 $i$ 的 load 与 $i+1$ 的算重叠。II 下界：$\max(\mathrm{ResMII}, \mathrm{RecMII})$。缺口是**选 II 并放置**，不是块内就绪表。

<span class="marginnote">数字实例：若循环体一次 load 要 4 拍才返回，且环依赖要求相邻迭代至少隔 4 拍，RecMII 就是 4；哪怕机器每拍能发 2 条指令、把 ResMII 压到 3，II 也只能取 $\max(3,4)=4$——瓶颈在依赖环，不在发射口宽度。</span>

若寄存器寿命跨过 II 太长，要旋转或放弃。展开可减 RecMII。

### 软件流水不是硬件流水线重讲

组成课五级是 CPU；本课编译器改指令序与核循环。对象不同。

<span class="marginnote">Lam 1988。Rau IMS。Itanium/VLIW 是历史主场；超标量上收益看循环。本课不把 GPU 的软件流水当必须。</span>

## 方法

算 MII。从 MII 起试调度核。失败则 II++。生成序言（填满）、核、收尾。处理模变量（同一物理寄存器的轮转）。

```mermaid
flowchart TD
  L["循环体"] --> MII["Res/Rec MII"]
  MII --> KER["模调度核"]
  KER --> PE["序言 / 收尾"]
```

与向量化：可先向量化再流水，或放弃其一。别名环依赖会抬 RecMII。

```mermaid
flowchart TD
  B["算下界 II = max ResMII RecMII"] --> T["以当前 II 试模调度"]
  T --> C{"所有指令放得下 寄存器寿命也不超限?"}
  C -- "是" --> K["生成序言 / 核 / 收尾"]
  C -- "否" --> I["II 加一"]
  I --> T
  K --> R["模变量用寄存器轮转分配"]
```

<span class="marginnote">直觉类比：序言与收尾像流水线的开线与收线——第一批进料时机器还没满负荷（序言），最后一批出料时逐渐空出（收尾），只有稳定中段才是核：每 II 拍放进一圈新迭代，同时吐出一圈结果。</span>

## 机制

代码体积：序言收尾胀。调试困难。异常：核中间出错，状态难映射源迭代——限制投机。不要对含调用的循环强流水除非内联完。

<span class="marginnote">常见误区：以为软件流水总能加速。寄存器寿命跨多个迭代时轮转寄存器会膨胀，序言收尾让代码体积成倍增长，核中间抛异常也很难对应回某个源迭代——对含调用或不规则控制流的循环，强流水常常得不偿失。</span>

## 边界

本课不写全部 SMS 算法变体。后课默认：无环或弱环的内层可模调度。下一课关键路径：给表调度/流水当优先级的路径长度。

也不把软件流水当 OS 管道。

<span class="marginnote">术语翻译：模调度（modulo scheduling）的「模」指核循环按 II 取模循环——第 $i$ 圈迭代在第 $i \bmod \mathrm{II}$ 拍进入核，同一拍里最多有 II 圈不同迭代的片段在并行；所谓 kernel 就是这样一段重复 II 拍就循环一次的紧凑代码。</span>

## 小结

- 软件流水：迭代重叠，II 受资源与环约束。
- 核 + 序言/收尾；寄存器轮转是实现细节。
- 与表调度分层：块内 vs 跨圈。
- 出处：Lam, 1988；Rau 模调度；对照 Hennessy–Gross。
