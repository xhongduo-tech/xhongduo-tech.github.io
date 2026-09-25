---
title: DDR 协议与命令
date: 2026-09-08
section: cs
---

# DDR 协议与命令

<div class="epigraph">
  <p>命令/地址在时钟边沿进颗粒，数据在 DQS 双边沿进出：ACT、RD、WR、PRE、REF 是有最小间隔的序列，不是随意的 SRAM 读写脉冲。</p>
  <footer>—— 据 JEDEC DDR SDRAM 规范；Hennessy and Patterson, CA:AQA 整理</footer>
</div>

[上一课](/cs/row-buffer-bank-conflict)用状态机语言讲了命中与冲突。缺口是 **DDR 协议**：双倍数据速率、DQS 选通、命令编码与必须遵守的间隔，把控制器从「发读写」落实到引脚级合同。

## 问题

SDR 每周期一次数据；DDR 在时钟升降沿都传（再后代 DDR2/3/4/5 速率与 bank 组不同，命令集同族）。命令：ACT（行地址）、RD/WR（列地址、突发）、PRE、REF、ZQ 校准等。缺口不是再定义行缓冲，而是：**非法命令序列**会导致不确定数据，STA 式的「最小周期」在这里变成 $t_{RCD}$、$t_{RAS}$、$t_{WR}$、$t_{RFC}$ 一串。

<span class="marginnote">术语翻译：DDR 的「双倍」指数据在时钟的上升沿和下降沿各传一次——3200 MT/s 的 DDR4 时钟其实只有 1600 MHz，每个周期却挪两个字；而命令（ACT/RD 等）仍只在上升沿发，命令带宽和数据带宽是两本账。</span>

校准：写均衡、读 DQS 训练，PHY 的模拟活，数字控制器假定训练完成。本课承认 PHY 存在，不设计均衡器——避免滑进模拟/光刻。

### DDR 不是「总线频率的两倍就结束」

有效带宽还受突发间隙、刷新、命令带宽、bank 冲突限制。把标称 MT/s 直接当应用程序 memcpy 带宽，阿姆达尔在内存墙上会失望。ECC 颗粒占用部分宽度或另宽。

<span class="marginnote">数字实例：DDR4-3200 标称带宽是 8 字节总线 × 3200 MT/s = 25.6 GB/s，而实际 memcpy 往往只有十几 GB/s——差距被刷新（tRFC 期间该 bank 停摆数百纳秒）、突发间隙和 bank 冲突吃掉。标称值是天花板，不是地板。</span>

<span class="marginnote">JEDEC 各代 DDR 标准是本课主文献。CA:AQA 把命令与时序接到控制器。本课不背引脚表，钉命令与双边沿数据。</span>

## 方法

控制器维护每 bank 计时器：上次 ACT/RD 之后能否再发。命令/地址复用或按代分开。数据突发 BL8 对齐 cache 行常见。电源：自刷新在休眠时由颗粒内部刷新，对照组成[刷新课](/cs/dram-refresh)的控制器义务转移。

```mermaid
flowchart TD
  MC["内存控制器"] --> CMD["CK 上的 ACT/RD/WR/PRE/REF"]
  CMD --> DRAM["颗粒状态机"]
  DRAM --> DQ["DQS 双边沿数据"]
  DQ --> LATER["后课：地址如何切到这些命令"]
```

LPDDR 面向移动，命令与电压不同，层次仍是 bank/行。本课以 JEDEC DDR 为轴。

## 机制

地址映射把物理地址拆成通道、rank、bank、行、列，然后才变成这些命令。调度器选择下一合法命令。HBM 用类似命令集、不同封装。本课把「引脚协议」交清。

<span class="marginnote">直觉类比：ACT 像把书架上一整排书搬到桌面（慢，等 $t_{RCD}$）；RD/WR 是从桌面抽某一本，还能连抽一叠（突发）；PRE 把整排放回书架（等 $t_{RP}$）。书已在桌上就快，频繁换排就慢——行缓冲命中与冲突的物理来源就在这。</span>

```mermaid
flowchart TD
  ACT["ACT：打开行，等 tRCD"] --> RD["RD/WR：读列，突发 BL8"]
  RD --> PRE["PRE：关闭行，等 tRP"]
  PRE --> ACT2["同 bank 可再 ACT"]
  REF["REF：整排刷新，等 tRFC"] -. 期间该 bank 拒客 .-> ACT2
```

## 边界

本课不讲 GDDR 的伪通道全部差异，不把 DIMM 金手指机械写成主体。不讨论内存超频社区。不把 Transformer 权重搬迁带宽当案例。

后课默认：访存是受时序约束的命令序列；数据在 DQS 上 DDR 传送。

## 小结

- DDR：命令在 CK，数据在 DQS 双边沿。
- ACT/RD/WR/PRE/REF 带最小间隔；非法序列无定义。
- 标称速率 ≠ 应用带宽。
- 出处：JEDEC DDR；Hennessy and Patterson, CA:AQA。
