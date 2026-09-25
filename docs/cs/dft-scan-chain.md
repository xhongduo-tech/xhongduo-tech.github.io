---
title: 可测性设计与扫描链
date: 2026-09-08
section: cs
---

# 可测性设计与扫描链

<div class="epigraph">
  <p>封装之后没有探针扎到每个触发器；把 FF 串成移位寄存器，测试仪才能移入向量、移出响应，判断组合逻辑是否粘死。</p>
  <footer>—— 据 Bushnell and Agrawal, Essentials of Electronic Testing；IEEE Std 1149.1（JTAG）；Weste and Harris, CMOS VLSI Design 整理</footer>
</div>

[上一课](/cs/formal-equivalence)保证网表与 RTL 一致，不保证**硅**无缺陷。[摩尔经济学](/cs/moore-law-economics)把测试成本算进单价。缺口是 **DFT**：扫描链把触发器变成可控可观，否则自动测试仪（ATE）够不着内部节点。

## 问题

固定故障（stuck-at）模型：某网恒 0/1。要激活故障并传播到可观察点。全组合电路还可用输入直接控；时序电路状态空间太大。扫描：每只 FF 加 MUX，测试模式时 D 来自上一只 FF 的 Q，链成移位寄存器。缺口不是再做 LEC，而是这套测试结构如何插入、如何与功能时钟共存。

<span class="marginnote">术语翻译：「固定故障 stuck-at」是给物理缺陷起的模型化外号——某根线因制造缺陷永远读作 0 或 1，好比电线焊死在电源或地上。真实缺陷千奇百怪，但这个最简单的模型能抓住大多数，ATPG 就是按它来出题的。</span>

IEEE 1149.1 JTAG 提供芯片级 TAP，边界扫描测引脚与板级互连；内部扫描是同一思想往里走。

### 扫描不是「功能复位」

扫描移位时状态无功能含义，翻转率高、功耗大，须遵守测试功耗预算。功能复位树在测试模式可能被旁路。把扫描使能当复位，量产测试会破坏芯片或给出假失败。

<span class="marginnote">常见误区：把扫描使能当成一种复位。移位阶段寄存器里全是无功能含义的乱码；若这时误触发功能复位逻辑、或让时钟门控把时钟关了，捕获到的响应全错，量产测试会成批「假失败」。所以 scan-enable 是独立控制信号，有自己单独的时序约束。</span>

<span class="marginnote">Bushnell/Agrawal 是测试教材。1149.1 是边界扫描标准。Weste/Harris 有扫描 FF 电路。本课不背全部故障模型（转换延迟故障点名即可）。</span>

## 方法

综合后插入扫描 FF 与链。ATE：scan-in 一向量，脉冲一次功能捕获，scan-out。ATPG 生成向量。约束：扫描链时钟、锁存器电平敏感扫描（LSSD）是另一风格。与 [STA](/cs/sta)：扫描移位路径有自己的时序，常更慢，单独约束。

```mermaid
flowchart TD
  FF["功能 FF"] --> MUX["测试 MUX"]
  MUX --> CHAIN["扫描链"]
  CHAIN --> ATE["移入 / 捕获 / 移出"]
  ATE --> LATER["后课：BIST 把 ATE 搬进芯片"]
```

算术单元的宽乘法器组合锥正是 ATPG 的负担；测试点插入可提高覆盖。

## 机制

下一课 BIST 用片上图案生成器减少 ATE 数据量。扫描改变网表，必须再 LEC（功能模式 scan-enable=0）。时钟门控在测试中强制开。复位与扫描冲突由 DFT 规范定义。

上一张图画「扫描结构如何插进芯片」；这张回答运行时的问题：scan_enable 一换挡，FF 的 D 从哪来，一个向量又分哪三步执行。

```mermaid
flowchart TD
  SE["scan_enable 信号"] -- "0 功能模式" --> FN["MUX 选功能逻辑送来的 D"]
  SE -- "1 测试模式" --> TS["MUX 选上一级 FF 的 Q"]
  TS --> PH1["阶段一 移位 灌入测试向量"]
  PH1 --> PH2["阶段二 捕获 打一个功能时钟"]
  PH2 --> PH3["阶段三 移位 移出响应比对"]
  PH3 --> REP["回到阶段一 送下一个向量"]
```

<span class="marginnote">数字实例：扫描链像给芯片插了根吸管——测试仪不用探进内部，而是把一串 0/1 从链头灌进去、从链尾吸出来。链上 1000 只 FF 时，一个向量光移位就要约 2000 拍（移入加移出），这正是测试时间与向量数都算成本的原因。</span>

## 边界

本课不写模拟 BIST、不把系统级软件测试当 DFT。不讨论故障定位的软件算法细节。不进入封装探针卡机械。

后课默认：量产数字测试依赖扫描链把 FF 串起来；JTAG 管边界；内部还有 BIST。

## 小结

- 封装后靠扫描移位控/观内部 FF。
- 插入 MUX 改网表，功能模式须再等价检查。
- 测试功耗与扫描时序单独约束。
- 出处：Bushnell and Agrawal；IEEE 1149.1；Weste and Harris。
