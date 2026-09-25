---
title: 阻塞与非阻塞赋值
date: 2026-09-08
section: cs
---

# 阻塞与非阻塞赋值

<div class="epigraph">
  <p>同一 always 里，`=` 立刻改左值，`<=` 采样右值、在时间步结束再更新；混用会让仿真看到的移位寄存器变成软件式覆盖。</p>
  <footer>—— 据 Cummings, Nonblocking Assignments in Verilog Synthesis, Coding Styles That Kill, SNUG 2000；IEEE Std 1364；Harris and Harris, Digital Design and Computer Architecture 整理</footer>
</div>

[上一课](/cs/verilog-basics)钉了模块与 `always`，并把「仿真是否等于硬件」留作缺口。组成课的[触发器](/cs/flipflop-types)是并行采样的。缺口是赋值运算符：**阻塞 `=` 与非阻塞 `<=`** 如何对应组合与边沿寄存器。

## 问题

移位寄存器：三个 FF 应同时用旧值。若在 `posedge clk` 里写 `a=b; b=c;`，仿真变成串行覆盖，综合仍可能推出三只 FF——**仿真/综合不一致**。正确风格：时序块用 `<=`，组合块用 `=`。缺口不是新的触发器电路，而是语言调度：NBA（nonblocking assignment）在当前活跃事件之后更新。

Cummings 的规则是工业常识：不要在同一 `always` 里对同一变量混用两种赋值。

<span class="marginnote">术语翻译：阻塞 `=` 就是「这句不干完，下一句不许动」——左值立刻更新；非阻塞 `<=` 是「先把右值抄下来，本时间步结束时统一更新」。区别不在电路，在仿真器的调度顺序：NBA 排在活跃事件之后的一个单独队列里。</span>

### 非阻塞不是「异步、不必等时钟」

`<=` 仍然只在该 `always` 被触发时求值（通常是 `posedge`）。它不表示跨时钟域，也不表示组合旁路。把 nonblocking 理解成软件 async，CDC 课会对不齐。

<span class="marginnote">Clifford Cummings, SNUG 2000 是本课主文献。IEEE 1364 规定事件队列。Harris 用移位寄存器当反例。</span>

## 方法

`always @(posedge clk)`：全部 `<=`，读到的是本拍开始的寄存器值，对应并行 FF。`always @(*)`：`=` 描述组合函数，避免意外锁存（每个输出在所有分支赋值）。临时组合量在时序块里可用阻塞算出中间值再 NBA 给寄存器——要极克制，本课推荐中间量拉到组合块。

```mermaid
flowchart TD
  POSEDGE["posedge always"] --> NBA["非阻塞：并行 FF"]
  COMB["组合 always"] --> BA["阻塞：连线逻辑"]
  MIX["混用同一变量"] --> X["仿真综合不一致"]
  NBA --> LATER["后课：综合成网表"]
```

<span class="marginnote">数字实例：想在时钟沿交换两个寄存器，写 `a<=b; b<=a;` 则每拍真的互换。若误写成阻塞 `a=b; b=a;`，两句在拍内串行执行，`b` 拿到的是已经变成旧 `b` 的 `a`——两个寄存器都成了旧 `b`，交换悄悄失败，仿真和硬件同时错。</span>

`#delay` 只为仿真，综合忽略。后课网表不再有这两种赋值，只剩门延迟。

## 机制

综合器把 NBA 到 `reg` 映射为触发器，把组合 `=` 映射为门。错误风格可能推断出锁存（level-sensitive），与边沿规格不符，[建立保持](/cs/setup-hold)窗口会换一套。本课把风格钉死，让后课 STA 分析的是预期的 FF+组合。

一个时钟拍内两种赋值的时间线，值得单独画。

```mermaid
flowchart LR
  CLK["时钟沿触发"] --> ACT["活跃区：阻塞赋值立即改左值"]
  ACT --> NBAQ["NBA 区：右值已采样，暂不更新"]
  NBAQ --> STEP["时间步结束：寄存器统一更新"]
  STEP --> NEXT["下一拍读到的是新值"]
```

<span class="marginnote">直觉类比：非阻塞像全班考试——人人先把题目抄到草稿上（采样右值），下课铃一响同时誊到正式答卷（更新寄存器）；阻塞则是传纸条改答案，改一个传一个，顺序错一步全盘错。硬件要的是「同一拍并行采样」，所以时序块永远用 `<=`。</span>

## 边界

本课不讲 SystemVerilog 的 `always_ff`/`always_comb` 全部语义差异（它们有助于工具检查，推荐用，但不替代对 NBA 的理解）。不引入 PLI。不把 Python 协程类比过来。

后课默认：时序非阻塞、组合阻塞；移位与流水线按并行采样理解。

## 小结

- `<=` 对应并行触发器采样；`=` 对应组合立刻传播。
- 混用制造仿真/综合裂缝。
- 下一课：这些块如何变成门级网表。
- 出处：Cummings, SNUG 2000；IEEE 1364；Harris and Harris。
