---
title: 功耗：动态与静态
date: 2026-09-08
section: cs
---

# 功耗：动态与静态

<div class="epigraph">
  <p>动态功耗随 $CV^2f$ 与翻转率走；静态是漏电，关断时钟也还在漏。算术单元的华莱士树既贡献电容，也贡献每拍翻转。</p>
  <footer>—— 据 Rabaey, Chandrakasan, Nikolic, Digital Integrated Circuits；Weste and Harris, CMOS VLSI Design；Hennessy and Patterson, CA:AQA 整理</footer>
</div>

[上一课](/cs/asic-flow-stdcell)让网表落到带电容的金属与扩散区。[STA](/cs/sta) 只管时序。缺口是**功耗分解**：动态（开关、短路）与静态（亚阈值、栅漏），否则下一课门控与 DVFS 没有作用对象。

## 问题

CMOS 动态：$P_{\mathrm{dyn}}\approx \alpha C V^2 f$。$\alpha$ 是活动因子，FMA 阵列每拍大翻转则 $\alpha$ 高。<span class="marginnote">数字实例：公式里电压是**平方**项，这是「降压最有效」的来历。电压降 $10\%$（比如 $1.0\,\mathrm{V}\to 0.9\,\mathrm{V}$），动态功耗变成 $0.9^2=0.81$，直接省约 $19\%$；而降频 $10\%$ 只省 $10\%$。代价是门变慢，时序分析要重做——省电从来不是免费的。</span>短路功耗在边沿转换时 PMOS/NMOS 同时导通。静态：沟道漏电随阈值与温度，工艺缩小后可与动态同量级。缺口不是再讲标准单元行，而是这几项如何进设计决策：降 $V$ 最有效，但延迟变差，STA 要重做。

FPGA LUT 配置 SRAM 也漏电；ASIC 标准单元按阈值分 HVT/LVT 库做漏电–速度权衡。

### 降频不是唯一旋钮

$P\propto f$ 只抓住动态。漏电与 $f$ 无关，待机芯片仍热。把功耗问题全交给「时钟调慢」，静态主导的节点无效。面积更大的并行（多份 ALU）可降 $f$ 保持吞吐，但 $C$ 上升，需要算总账——阿姆达尔与功耗墙后课再收。

<span class="marginnote">Rabaey DIC 与 Weste/Harris 给 CMOS 功耗公式。CA:AQA 把功耗作为体系结构约束。本课不进入电源管理 IC 的模拟设计。</span>

## 方法

估计：从 RTL/网表切换活动（仿真或向量）× 电容模型。时钟网本身是巨大 $C$，故 [CTS](/cs/clock-skew-cts) 也是功耗问题。算术：饱和 MUX 比浮点 LZC 轻；双精度 FMA 是功耗大户。报告分开关、内部、漏电。

```mermaid
flowchart TD
  SW["α C V² f 开关"] --> TOT["总功耗"]
  SC["短路"] --> TOT
  LEAK["漏电静态"] --> TOT
  TOT --> LATER["后课：门控与 DVFS"]
```

<span class="marginnote">直觉类比：时钟网像全城喇叭广播——不管每家有没有货要出，喇叭每拍都喊一遍，人人都要竖耳听一次。所以时钟树是芯片上最大的「常翻转变量」，这也解释了为什么下一课的门控（分区关喇叭）是第一杠杆：空闲模块连喇叭都停掉，$\alpha$ 就地归零。</span>

温度反馈：更热则漏电更大，STA 的慢角与功耗的热角要一起看。

## 机制

下一课时钟门控降 $\alpha$ 中的时钟翻转；DVFS 降 $V$ 与 $f$。Dennard 缩放解释为何电压不能随工艺无限降。本课只钉两项之和。验证与 DFT 扫描时翻转率异常高，功耗签核要分功能模式与测试模式。

面对一个功耗问题该动哪个旋钮，取决于超支的是哪一项：

```mermaid
flowchart TD
  Q{"功耗超支在哪一项?"} --> A["动态为主，翻转太多"]
  Q --> B["静态为主，待机漏电大"]
  A --> C["时钟门控压活动因子 α"]
  A --> D["DVFS 降 V 与 f"]
  B --> E["选高阈值单元 HVT 换漏电"]
  B --> F["关断电源域"]
  C --> G["重跑时序与验证"]
  D --> G
  E --> G
  F --> G
```

<span class="marginnote">常见误区：初学者容易以为「芯片闲着就不耗电」。漏电一项只看工艺、阈值与温度，时钟停了它照漏——这就是待机芯片仍发烫、手机放一夜也掉电的原因之一。对付静态功耗要的是关电源域或换高阈值单元，调频率对它完全无效。</span>

## 边界

本课不推导亚阈值电流公式的全部器件物理，不写封装热阻网络的有限元。不把数据中心电费表当课程。不进入限价簿或模型训练的能源口号，只谈芯片上的 $C,V,f$。

后课默认：动态随 $CV^2f\alpha$，静态是漏电；二者都要管。

## 小结

- 动态：$CV^2f$ 与翻转；静态：漏电，与时钟无关。
- 降压最有效但伤时序。
- 时钟树与宽算术阵列是电容大户。
- 出处：Rabaey et al.；Weste and Harris；Hennessy and Patterson, CA:AQA。
