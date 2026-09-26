---
title: 值范围分析
date: 2026-09-08
section: cs
---

# 值范围分析

<div class="epigraph">
<p>在格上跟踪整数可能取值的区间或位掩码：比较可折成常量，无符号越界检查可删，switch 可变紧凑。</p>
<footer>—— 据 Cousot 区间域；Harrison 的范围分析实践；龙书与 LLVM ValueTracking 整理</footer>
</div>

上一课[PRE](/cs/partial-redundancy-elim)移动表达式。缺口是**值的约束**：`if (x>=0 && x<n)` 之后 `x` 的范围。值范围分析（VRA）是[抽象解释](/cs/abstract-interpretation)的区间/位域在 SSA 上的应用。本课钉用途，不写完整八边形域。

## 问题

常量传播只认单点。范围：`[0,255]` 使 `x&255` 变恒等，`x<256` 变真。缺口是**分支后的收紧**（与 SCCP 的控制流敏感同类），服务消除检查与选择指令。

位域：已知低位为零则对齐。与[归纳变量](/cs/strength-reduction-iv)：IV 的闭式给出精确范围。

### 范围不是类型

`uint8` 的类型范围是 `[0,255]`；分析可更窄。类型系统的 Option 与范围分析互补：一个是语言，一个是 IR 事实。

<span class="marginnote">Cousot 区间。GCC VRP。LLVM 的 ConstantRange / LazyValueInfo。本课可靠 over-approx：真实值 ⊆ 区间。</span>

## 方法

格：区间，交为收紧，并为合并（φ、未知枝）。加宽防循环升链。分支：真枝交上谓词。

```mermaid
flowchart TD
  SSA["SSA 名"] --> RNG["区间 / 位"]
  BR["分支谓词"] --> RNG
  RNG --> FOLD["折比较 / 删检查"]
```

与 UB：有符号溢出若 UB，分析可假设不溢出以收紧——危险与后课 UB 优化同一刀。

## 机制

指针范围（对象内偏移）服务越界消除，须 provenance。不要把分析当沙箱。浮点范围更少用，NaN 破坏全序。

分支收紧与 φ 合并如何共同塑造一个变量的区间：

```mermaid
flowchart TD
  X["x ∈ [0, 40]"] --> IF{"if (x \lt 10)"}
  IF -->|真枝| T["x ∈ [0, 9]  (与谓词相交)"]
  IF -->|假枝| F["x ∈ [10, 40]"]
  T --> A["真枝内: x \lt 10 折成真, 检查可删"]
  T --> PHI["汇合点 φ(x1, x2)"]
  F --> PHI
  PHI --> J["x ∈ [0, 40]  (两枝取并)"]
  J --> NOTE["并集可能引入空洞: 中间未必都取得到"]
```

PGO 可提供热范围，但是动态的，须与静态可靠分开。

<span class="marginnote">数字实例：`if (x \gt= 0 \&\& x \lt n)` 之后进入数组访问 `a[x]`，分析器手里已有 $x\in[0,n-1]$，边界检查整条删掉——这就是实现里「消检查」的实际路径。GCC 的 VRP 与 LLVM 的 LazyValueInfo 都靠这招吃掉防御式代码。</span>

<span class="marginnote">常见误区：把区间当成「精确取值集合」。真枝 $[0,1]$、假枝 $[10,11]$，φ 合并得到的是 $[0,11]$ 而不是 $\{0,1,10,11\}$——中间 $2$ 到 $9$ 其实取不到，但区间域表达不了空洞。宁可说过头（over-approx），不可说过窄：漏掉真实值会让删掉的检查吃掉正确性。</span>

<span class="marginnote">「加宽」可以想象成给循环里的区间放气阀：第一次 $x\in[0,1]$，下一轮 $[0,2]$、$[0,3]$……若一格一格涨，循环上链永远收敛不了。加宽直接跳到 $[0,+\infty)$ 让迭代尽快停住；后随的收窄再慢慢往回挤。没有这一步，任何带循环的程序都算不完。</span>

## 边界

本课不写关系域（`x<y`）全文。后课默认：整数范围可消比较。下一课 SSA 构造：许多分析假设 SSA，该把 φ 插对。

也不把范围当依赖类型 `Fin n` 的替代。

## 小结

- 值范围：区间/位格，分支收紧。
- 用于折比较、删冗余检查、助向量化。
- 可靠近似；循环要加宽。
- 出处：Cousot；GCC VRP / LLVM ConstantRange；对照 Wegman SCCP。
