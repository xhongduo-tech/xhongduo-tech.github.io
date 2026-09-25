---
title: Flynn 分类
date: 2026-09-08
section: cs
---

# Flynn 分类

<div class="epigraph">
<p>按指令流与数据流的份数把机器分成 SISD、SIMD、MISD、MIMD；它不管缓存协议，只管「同时在飞的是一条指令还是很多」。 </p>
<footer>—— 据 Flynn, Very High-Speed Computing Systems, Proceedings of the IEEE, 1966 整理</footer>
</div>

[上一课](/cs/fence-cost) 收束单核顺序与一致性进阶。单核 ILP 再挖也是一条指令流。[超标量](/cs/superscalar-issue) 每拍多条，仍是同一 PC 家族。本课不重讲 fence drain。缺口是 **Flynn 的四格：用指令流 × 数据流给后面的 VLIW、向量、SIMT、多核一个共同坐标系。**

## 问题

乱序核、向量机、GPU、多 socket 在文献里都叫「并行」， intern 容易把发射宽度与核数相加。缺口不是再加一种 cache 态，而是**先分箱：同一时刻有几条指令流、每条指令流操作几份数据。** 后续课只在格子里走，不把 GPU 写成「很大的超标量」。

<span class="marginnote">SISD：经典单核。SIMD：一条指令、向量数据。MISD：少见（某些流水脉动可硬塞）。MIMD：多核、多线程各有 PC。</span>

<span class="marginnote">用厨房类比这四格：SISD 是一个厨师顺序做一道菜；SIMD 是一个厨师对 8 口锅做同一个「翻炒」动作——动作一条、食材八份；MIMD 是 8 个厨师各做各的菜、各有各的菜谱进度；MISD 像多道质检工序依次检查同一份菜——动作多条、食材一份。</span>

## 方法

SISD：本课程前两单元的核。SIMD：后课向量 lane、以及 ISA 对照里的 [SIMD 扩展](/cs/simd-extensions)——一条 opcode，多条 ALU。MIMD：后课多 socket、big.LITTLE。SIMT 是 SIMD 的控制实现：硬件上很多 lane 共享一个 PC，遇分支再发散，不是第四个 Flynn 类，下一课之后才拆。

```mermaid
flowchart TD
  F["指令流 × 数据流"] --> SISD["SISD 单流单数据"]
  F --> SIMD["SIMD 单流多数据"]
  F --> MIMD["MIMD 多流多数据"]
  SIMD --> SIMT["SIMT：共享 PC 的 SIMD 实现"]
```

## 机制

[阿姆达尔](/cs/cpi-amdahl) 在 MIMD 上变成「串行段限制加速比」；SIMD 上变成「向量化比例」。二者不要混用同一个 $f$。一致性只出现在 MIMD（及 SIMD 核之间的共享内存），SISD 无需 [MESI](/cs/mesi-protocol)。

<span class="marginnote">数字实例看清两个 $f$ 的差别：串行段占 10% 时，8 核 MIMD 加速上限是 $1/(0.1+0.9/8)\approx 4.7$ 倍；8 lane SIMD 上向量化比例 90% 时上限同样是约 4.7 倍——公式同形，但前一个 $f$ 是「必须顺序执行的代码占比」，后一个是「能改写成向量指令的代码占比」，来源完全不同。</span>

<span class="marginnote">初学者容易把 GPU 的「几千个核」直接归入 MIMD。实际上 SIMT 下一个 warp 的 32 条 lane 共享同一个程序计数器，同一拍执行的是同一条指令，只有遇分支才分道——所以它是 SIMD 的控制包装，而不是第四个 Flynn 类。</span>

本课声明边界：大模型栏的张量并行不是本栏的 Flynn 练习；这里只为 CPU/GPU/DSA 硬件分型。

```mermaid
flowchart TD
  W["一个 warp:32 条 lane 共享同一个 PC"] --> BR{"执行到 if 分支"}
  BR -- "16 条 lane 条件为真" --> T["真路径:这 16 条 lane 干活"]
  BR -- "另 16 条条件为假" --> F["假路径:那 16 条 lane 干活"]
  T --> SER["两段只能串行执行,每段另一半 lane 空转"]
  F --> SER
  SER --> J["汇合,回到同一个 PC 继续走"]
```

## 边界

本课不把 VLIW 的编译器打包写完，下一课。也不把数据流机当成第五类 Flynn——Dennis 的数据流是另一轴，再下一课。MISD 不展开。

后课默认：谈到并行先报 Flynn 格子。把多条独立操作塞进一个长指令字，是 VLIW/EPIC。

## 小结

- Flynn 用指令流与数据流分 SISD/SIMD/MIMD。
- SIMT 是 SIMD 的控制包装，不是新格子。
- VLIW 把并行打包交给编译器，下一课。
- 出处：Flynn, *IEEE Proc.*, 1966。
