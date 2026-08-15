---
title: 主存技术：DRAM 的内部组织（Bank、行缓冲）
date: 2026-08-07
---

# 主存技术：DRAM 的内部组织（Bank、行缓冲）

<div class="epigraph">
<p>内存不是一块均匀的池子——它是一大片必须按行打开、按列读取、用完关闭的田地。</p>
<footer>—— DRAM 工作原理的比喻</footer>
</div>

<div class="article-byline">
<p>第三级 · 计算机体系结构 ｜ Hennessy & Patterson《Computer Architecture: A Quantitative Approach》第 2 章 ｜ 2026-08-07</p>
</div>

## 为什么体系结构要懂 DRAM

Cache 缺失后的下一站就是主存（DRAM）。但 DRAM 的访问**不是「按地址直接读」**，而是有一套「打开行、读列、关行」的节奏——**内存的访问顺序，决定了快几倍还是慢几倍**。不懂 DRAM 的行缓冲与 bank，就永远无法解释「为什么程序要按行访问数组」这种性能谜题。<span class="marginnote">对比 [[cache-optimization-summary]] 里 Cache 的命中/缺失二分，DRAM 也有自己的「命中/冲突」：<strong>行命中 vs 行冲突</strong>。这个二分是 [[memory-controller-scheduling]] 全部调度策略的原料。</span>

## 1 DRAM 的结构：Bank、行、列

**核心概念**：DRAM 芯片内部组织成**多个 bank**（如 8–16 个），每个 bank 是一块 **行列矩阵**：行地址（row address）选一整行、列地址（column address）选行里的若干位。

关键器件：

**行缓冲（row buffer）**：每个 bank 有一个行缓冲，装着**最近被打开的那一行**的全部数据。
访问数据时：先**打开（activate）**目标行到行缓冲，再**读列**（把行缓冲里对应列的数据取走）。

## 2 三阶段访问：Activate / Read / Precharge

一次完整的内存访问分三步：

1. **Activate（激活）**：把目标行从存储单元阵列搬进行缓冲。代价最大——整行传输 + 电容感应，需 `tRCD`（RAS-to-CAS 延迟）时间。
2. **Read/Write（读/写列）**：从行缓冲取/写目标列，需 `tCAS`（列访问延迟，即 CAS Latency）。
3. **Precharge（预充电）**：用完关行、恢复电容状态，为下次激活做准备，需 `tRP`（RAS Precharge）时间。

**如果你紧接着访问的是同一行**——已经 Activate 过、行还在行缓冲里——直接**跳过 Activate**，只需 `tCAS`！这就是**行命中（row buffer hit）**。

## 3 行缓冲：内存里的「Cache」

**核心概念**：**行缓冲命中（row-buffer hit）**：目标行已在行缓冲中，访问只需一次列读；**行缓冲冲突（row-buffer conflict）**：目标行不在缓冲，要先 Precharge 旧行、再 Activate 新行——三次时序全付。

典型代价差（DDR4 粗算）：

行命中：约 15–25 ns（≈ 24–40 个周期）。
行冲突：约 50–70 ns（≈ 80–110 个周期）。

**相差 3 倍左右**。所以「连续访问同一行的数据」≈ 快，「到处乱跳行」≈ 慢——这直接解释了为什么 **Cache 缺失后按行访问数组**比按列访问快得多，也解释了分块（[[cache-optimization-compiler]]）在 DRAM 层面同样有效。

## 4 Bank 并行与地址交错

多个 bank 的最大价值是**并行**：不同 bank 可以**同时**处于 Activate/Read/Precharge 的不同阶段。于是内存控制器把**连续地址交错分布到不同 bank**（地址低几位选 bank）——连续的 Cache 块落在不同 bank，可以让它们「流水」访问。

**辨析｜易错点：** 地址交错让「连续地址 = 不同 bank」，看似违背「同一行更快」——其实两者并行不悖：**交错保证的是「多个独立的访问能并行」**，行缓冲保证的是「同一个 bank 内部的连续访问快」。控制器在调度时同时利用两者。

## 5 DRAM 时序参数：tRCD、tCAS、tRP

| 参数 | 含义 | 名称 |
| --- | --- | --- |
| `tRCD` | Activate 到可读列的时间 | RAS-to-CAS 延迟 |
| `tCAS` / `tCL` | 列访问延迟 | CAS Latency |
| `tRP` | Precharge 时间 | RAS Precharge |
| `tRAS` | 行保持激活的最短时间 | Active-to-Precharge |
| `tRC` | 一次完整行周期 | Row Cycle |

这些参数**写在内存条的 SPD 里**，由内存控制器读取并按此调度。**tCAS（CL）是宣传最响的**，但行命中/冲突的影响远大于 CL 那 1–2 个周期的差异。<span class="marginnote">买内存只看 CL 是误区：<strong>行缓冲策略与 bank 并行的调度，往往比 CL 快慢 1ns 更影响真实性能</strong>——这是「内存控制器」比「内存条规格」更值得研究的理由。</span>

## 6 公式解析：行命中 vs 行冲突

$$
T_{\text{行命中}} = t_{\text{CAS}}, \qquad
T_{\text{行冲突}} = t_{\text{RP}} + t_{\text{RCD}} + t_{\text{CAS}}
$$

- **第一步，看行命中**：行已在行缓冲，只付列访问——最短路径。
- **第二步，看行冲突**：先关旧行（$t_{\text{RP}}$），再开新行（$t_{\text{RCD}}$），最后读列（$t_{\text{CAS}}$）——三步全付。
- **第三步，看调度空间**：控制器若能**按 bank 交错**，让冲突发生在不同 bank 上，就能在「一个 bank 在 Precharge 时去访问另一个 bank」——**用并行掩盖时序**。这就是内存控制器调度的核心机会。

## 7 数值算例：行命中与行冲突的 3 倍差

设 DDR4 时序 $t_{\text{RCD}} = 14\,\text{ns}$、$t_{\text{CAS}} = 12\,\text{ns}$、$t_{\text{RP}} = 14\,\text{ns}$：

- **行命中**：$T = t_{\text{CAS}} = 12\,\text{ns}$。
- **行冲突**：$T = t_{\text{RP}} + t_{\text{RCD}} + t_{\text{CAS}} = 14 + 14 + 12 = 40\,\text{ns}$。

$$
\frac{T_{\text{冲突}}}{T_{\text{命中}}} = \frac{40}{12} \approx 3.3
$$

**要点**：同样的 DRAM，访问顺序不同就慢 3.3 倍——**「程序按行访问数组」不是玄学，是每 40 ns 省 28 ns 的真金白银**。再叠加 bank 交错让冲突错开，控制器能把有效带宽再拉高一大截。

## 8 常见误区再辨析

- **「CL 越小内存越快」**：行缓冲命中/冲突的影响远大于 CL 那 1–2 ns——**先看访问模式，再看 CL**。
- **「连续地址一定行命中」**：地址交错把连续地址分到不同 bank，连续访问是「跨 bank 流水」而非「同 bank 命中」——两者都在提速，机理不同。
- **「DRAM 就是个大数组」**：它是一次激活只能服务一行的结构化器件——**「随机访问」在 DRAM 上从来不是免费的**。
- **「内存控制器只负责转发」**：它同时在做**调度**（[[memory-controller-scheduling]]）、**刷新**、**错误处理**——是现代处理器里最忙碌的外围。

## 9 术语速查

| 术语 | 含义 |
| --- | --- |
| Bank | DRAM 内部可独立操作的存储单元组 |
| 行缓冲（row buffer） | 最近打开的行数据暂存区 |
| Activate | 把行搬进行缓冲 |
| Precharge | 关闭行、恢复电容状态 |
| tRCD | RAS 到 CAS 的延迟 |
| tCAS / CL | 列访问延迟 |
| tRP | Precharge 时间 |
| 行命中 / 行冲突 | 访问行是否已在行缓冲 |
| 地址交错 | 连续地址分布到不同 bank |

## 10 小结

- DRAM 组织成 **bank × 行列矩阵 + 行缓冲**，访问分 **Activate / Read / Precharge** 三步。
- **行命中**（跳过 Activate）约 3 倍快于**行冲突**（三次时序全付）。
- 按行访问数据 = 让 DRAM 行缓冲连续命中——程序优化的隐藏技巧。
- **多 bank 并行 + 地址交错**让独立访问流水化。
- 关键时序：`tRCD`、`tCAS`（CL）、`tRP`；行缓冲策略比 CL 更影响真实性能。

在下一节，我们看 DRAM 从 SDRAM 到 DDR 的演进——**SDRAM、DDR 系列的演进与带宽提升机制**。行缓冲与 bank 是「访问模式」的学问，下一节的 DDR 则把「接口怎么每次翻倍传数据」讲透——两者一个是内功、一个是外功。
