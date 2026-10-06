---
title: 摩尔定律的经济学
date: 2026-09-08
section: cs
---

# 摩尔定律的经济学

<div class="epigraph">
  <p>摩尔说的是单位成本下晶体管数的指数增长；墙之后密度仍可能涨，但掩膜、EDA 与验证的费用把「同一笔钱买到的性能」换成另一条曲线。</p>
  <footer>—— 据 Moore, Cramming More Components onto Integrated Circuits, Electronics 1965；Hennessy and Patterson, CA:AQA 整理</footer>
</div>

[上一课](/cs/dennard-power-wall)把电场缩放与功耗墙分开。Moore 1965 的观察是**经济学**：芯片上晶体管数目按规律涨，因为这样划算。缺口是：NRE（设计、掩膜、软件）与每片硅成本如何决定「还要不要做新 ASIC」，以及 FPGA/标准单元/HLS 在成本曲线上的位置。

## 问题

晶体管更小不一定更便宜：EUV 掩膜套数、设计规则复杂度、验证周期都涨。缺口不是再讲 $CV^2f$，而是：**固定成本**摊到多少颗、多长产品寿命。数字系统课在此收一口：算术单元与接口能做，不意味着每代都值得全定制。

FPGA 摊掉掩膜，用面积/功耗换上市时间。HLS 降 RTL 人月，不降 P&R 许可证与 STA 角。多核复用同一设计摊 NRE，对照功耗墙的并行。

<span class="marginnote">NRE 就是「还没卖出第一颗芯片就得先付的钱」：设计人力、EDA 许可证、一整套掩膜、验证与流片试跑。它不随产量变化，所以全靠产量摊薄——产量判断错了，NRE 就成了纯亏。</span>

### 摩尔定律不是性能定律

原文与后来的 Intel 表述围绕密度与成本。性能还要过 Dennard、存储墙、互连。把「每 18 个月翻倍」直接写成频率翻倍，与上一课矛盾，也与 DRAM 延迟曲线矛盾——后一单元内存会看见。

<span class="marginnote">初学者容易把摩尔定律当成「电脑速度每两年翻一倍」的性能承诺。实际上它说的是单位成本下的晶体管密度；频率在 2005 年前后就撞上功耗墙停涨了，多出来的晶体管后来主要拿去堆核与片上缓存，而不是让单线程线性变快。</span>

<span class="marginnote">Moore, *Electronics*, 1965。CA:AQA 讨论成本、良率与工艺。本课不引用虚构的咨询报告数字；定性曲线足够。</span>

## 方法

成本粗分：NRE / 件数 + 晶圆+封装测试。良率随面积指数变差，大芯片贵。平台化：同一 SoC 多 SKU 关单元（暗硅的市场版）。接口标准（后课 DDR、PCIe）让 PHY IP 摊到全行业，降低每家的模拟设计费用。

```mermaid
flowchart TD
  DENS["密度仍可增"] --> COST["掩膜与 NRE"]
  COST --> CHOICE["ASIC vs FPGA vs 买 IP"]
  WALL["功耗墙"] --> CHOICE
  CHOICE --> LATER["后课：验证必须列入 NRE"]
```

后课验证、DFT、BIST 都是 NRE 的一部分：少测会把成本赶到现场失效。

## 机制

本课序的 HDL 路径到此从「如何做」兼及「为何有时不做」。内存与 PCIe 单元假设多数设计者**买 IP**，自己做控制器调度与软件，而不是从晶体管画 DDR PHY。ISA 对照同理：授权与生态也是经济学。

```mermaid
flowchart LR
  NRE["NRE: 掩膜+设计+验证 一次性"] --> DIV["每颗摊 NRE/V"]
  MARG["每片晶圆+封装+测试"] --> SUM["单颗成本 = NRE/V + 边际"]
  SUM --> LOWV{"产量 V 小?"}
  LOWV -->|"是"| FPGA["FPGA 无掩膜: 更划算"]
  LOWV -->|"否"| ASIC["ASIC 摊薄 NRE: 更划算"]
  FPGA --> BE["两条曲线有盈亏平衡点"]
  ASIC --> BE
  BE --> TTM["FPGA 另赚上市时间: 平衡点右移"]
```

<span class="marginnote">代个数字感受量级：若一套先进工艺掩膜加设计验证要上千万美元的 NRE，产量 $V=10$ 万颗时每颗摊一百美元以上；$V=1000$ 万颗时每颗只摊一美元上下。这就是同一颗全定制芯片「做不起」与「值得做」之间只差订单量的原因。</span>

## 边界

本课不预测某代工艺的晶圆报价，不讨论出口管制。不把股市当文献。不进入光刻机产能。量化金融定价不是本栏。

后课默认：密度趋势与「更便宜的性能」已脱钩；验证与 IP 是成本一等公民。

## 小结

- Moore 原意是成本–密度；不是频率定律。
- NRE 与掩膜把全定制门槛抬高；FPGA/IP 是回应。
- 下一课起把验证计入这条成本。
- 出处：Moore, *Electronics*, 1965；Hennessy and Patterson, CA:AQA。
