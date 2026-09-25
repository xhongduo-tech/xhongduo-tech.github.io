---
title: Gustafson 定律
date: 2026-09-08
section: cs
---

# Gustafson 定律

<div class="epigraph">
<p>Amdahl 钉死的是固定工作量下的加速比；Gustafson 说机器变大时工作量也变大，串行段占比可以下降，扩展才有意义。</p>
<footer>—— 据 Gustafson, Reevaluating Amdahl's Law, CACM 1988 整理</footer>
</div>

[上一课](/cs/multi-socket-interconnect) 让并行机可以加 socket。[CPI 与阿姆达尔](/cs/cpi-amdahl) 已经给出固定问题规模的上限。本课不重讲 NUMA 链路。缺口是 **Gustafson：规模可变时的加速比定义**，避免「永远 5% 串行所以 20 核封顶」被误用到弱扩展场景。

## 问题

Amdahl：工作量 $W$ 固定，$s$ 为串行比例，加速比 $\le 1/(s+(1-s)/n)$。科学计算常在核数增加时把网格加密：并行段涨，串行初始化几乎不变。缺口不是新的 NoC，而是**换分子：把 $n$ 核上跑得动的更大问题当作「同等时间」，定义加速比。**

<span class="marginnote">数字实例：取 $s=5\%$、$n=64$。Amdahl 公式给 $1/(0.05+0.95/64)\approx14.3$，看似六十四核只换来十四倍；Gustafson 口径下 $s+n(1-s)=0.05+64\times0.95\approx60.9$。两个数都对——它们回答的是不同的问题。</span>

<span class="marginnote">常见误区：「永远 5% 串行所以约 20 倍封顶」只对固定工作量成立。把这句话拿去否决「加机器跑更大问题」的采购或上云方案，是用强扩展的尺子量弱扩展的事。</span>

<span class="marginnote">直觉类比：Amdahl 问的是「同一车货，换更大的卡车能早到多久」；Gustafson 问的是「车队变大后，同样时间内能多运多少货」。货物量固定与否，决定了哪个公式适用。</span>

<span class="marginnote">Gustafson 的缩放加速比：$s+n(1-s)$ 形式（记号随文本而异）：串行时间固定、并行工作随 $n$ 线性涨时，效率可以保持。</span>

## 方法

固定时间：允许 $W_p(n)$ 增长。报告时必须声明：问题规模、以及串行段是否真的不随 $n$ 涨（I/O、规约树、一致性热点会涨）。与 [伪共享](/cs/false-sharing)、跨 socket 延迟：这些会让「并行段」里长出新的串行/争用，Gustafson 不会自动成立。

另一种误读：把「更多核跑更大网格」的墙钟几乎不变当成强扩展成功。那是弱扩展。用户若要同一网格更快，必须看 Amdahl 曲线。

```mermaid
flowchart TD
  AMD["Amdahl：W 固定"] --> CAP["核数很快封顶"]
  GUS["Gustafson：W 随 n 涨"] --> KEEP["固定时间内做更多"]
```

## 机制

下一课强/弱扩展把这两种实验设计说成实验室标准：强扩展 ≈ Amdahl 实验；弱扩展 ≈ Gustafson 精神。SPEC 再下一课会警告：基准往往固定输入，更像 Amdahl。

```mermaid
flowchart TD
  NEED["你的需求是？"] --> Q1{"同一问题要更快？"}
  Q1 -- "是" --> STRONG["看 Amdahl / 强扩展曲线"]
  Q1 -- "不，要更大问题同时间" --> Q2{"串行段真不随 n 涨？"}
  Q2 -- "I/O、规约会涨" --> FIX["两条定律都要修正"]
  Q2 -- "基本不涨" --> WEAK["Gustafson / 弱扩展成立"]
```

本栏不把集群作业调度写成定律证明。

## 边界

本课不修改 Amdahl 的数学，只加规模假设。串行段随日志因子涨（规约）时两条定律都要修正。强/弱扩展术语下一课钉死。

后课默认：谈加速比先报问题规模是否随机器变。实验怎么画曲线，用强/弱扩展这对词。

## 小结

- Amdahl 固定工作量；Gustafson 允许工作量随核数增长。
- 争用与 I/O 会破坏「串行段不变」的假设。
- 强扩展与弱扩展是下一课的实验语言。
- 出处：Gustafson, *CACM*, 1988；Amdahl。
