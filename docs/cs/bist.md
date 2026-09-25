---
title: 内建自测 BIST
date: 2026-09-08
section: cs
---

# 内建自测 BIST

<div class="epigraph">
  <p>ATE 带宽和向量存储有限；片上用伪随机图案生成器刺激扫描链或存储器，压缩响应成签名，上电或现场也能自检。</p>
  <footer>—— 据 Bushnell and Agrawal, Essentials of Electronic Testing；IEEE Std 1149.1；Hennessy and Patterson, CA:AQA 整理</footer>
</div>

[上一课](/cs/dft-scan-chain)把 FF 串成链，向量仍常来自片外 ATE。大芯片扫描数据量与引脚速率成为测试成本。缺口是 **BIST**：图案生成（LFSR）与响应压缩（MISR）放进硅里，ATE 只启动并读签名。

## 问题

逻辑 BIST：LFSR 灌扫描链，若干捕获周期，MISR 压输出。存储器 BIST：对 SRAM/DRAM PHY 旁的阵列走 March 算法，覆盖固定与转换故障。缺口不是再解释扫描 MUX，而是**片上控制器**与签名黄金值。随机图案对随机逻辑覆盖尚可，对规则算术可能要混合确定向量。

现场自检（上电、高可靠系统）让 BIST 超出工厂 ATE。航空与汽车安全标准会要求，本课点名不背条款。

### 签名通过不是「无故障」

压缩有别名：两个响应可能同一签名。LFSR 多项式与 MISR 宽度降低别名概率，不是零。把 BIST 绿勾当成形式证明，与 LEC 课的警告同类。

<span class="marginnote">Bushnell/Agrawal 专章 BIST。存储器 March 测试是工业常规。CA:AQA 把可靠性与检测作为系统问题。本课不写具体 LFSR 抽头表。</span>

## 方法

LBIST：测试时钟下运行 $N$ 周期，比较 MISR。MBIST：地址计数器 + 数据背景（5/A、棋盘）。与 JTAG：TAP 指令启动 BIST、读状态。与[时钟门控](/cs/clock-gating-dvfs)：BIST 模式强制时钟开。功耗：伪随机高翻转，须降速或分块跑。

```mermaid
flowchart TD
  LFSR["LFSR 图案"] --> SCAN["扫描 / 阵列"]
  SCAN --> MISR["响应压缩"]
  MISR --> SIG["签名比较"]
  SIG --> LATER["后课：软错误不是制造缺陷"]
```

<span class="marginnote">LFSR 就是用移位寄存器加异或反馈来高速产生「看起来随机」的 0/1 序列的电路；MISR 是它的多输入版本，把电路输出一路卷进同一寄存器。跑完 $N$ 周期后寄存器里剩下的那个最终状态，就是拿去和黄金值比对的「签名」。</span>

<span class="marginnote">为什么重要：伪随机图案下相邻周期几乎每一位都在翻转，功耗远超正常工作。不降速或分块跑，测试本身可能把好芯片推到电源跌落或过热——「测坏」比「测出坏」更冤。</span>

DRAM 后面单元有自己的维修与 BIST，与逻辑 BIST 分家。

## 机制

下一课软错误是运行期辐射翻转，不是制造 stuck-at；BIST 上电能抓部分硬缺陷，抓不住随机位翻转——那要 ECC。老化可能让时序边沿在现场失败，BIST 定期跑可当健康检查，覆盖仍有限。

BIST 到底管哪类故障，可以分栏看。

```mermaid
flowchart LR
  FAULT["故障类型"] --> HARD["制造缺陷 stuck-at"]
  FAULT --> SOFT["运行期位翻转"]
  HARD --> BISTC["BIST 上电自检可抓"]
  SOFT --> ECC["要 ECC / 刷新，BIST 抓不住"]
  AGING["老化边沿失败"] --> PERIOD["BIST 定期跑当健康检查"]
```

<span class="marginnote">「签名通过」可以想象成把整份答卷压成一个 CRC 校验码：两份不同答卷可能算出同一个校验码——概率极小但非零。所以签名匹配只是「大概率没坏」，别名（aliasing）永远挂在概率那一栏，不进证明。</span>

## 边界

本课不把模拟环路 BIST（ADC）写完，不讨论加密密钥是否被 BIST 路径泄漏（安全课另开）。不把软件 memtest 当 MBIST。

后课默认：BIST 用片上 LFSR/MISR 或 March 降 ATE 成本；签名有别名；软错误要另一套。

## 小结

- 扫描解决可控可观；BIST 把向量发生与压缩搬进芯片。
- 逻辑用 LFSR/MISR，存储器用 March。
- 通过 ≠ 无故障；现场可复用。
- 出处：Bushnell and Agrawal；IEEE 1149.1；Hennessy and Patterson, CA:AQA。
