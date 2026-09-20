---
title: 瑞利判据笔记（Mack）
date: 2026-09-07
section: litho
---

# 瑞利判据笔记（Mack）

<div class="epigraph">
    <p>光学教材里的瑞利两针孔，给出的是「刚能分开两个艾里斑」；产线要的是在剂量、焦距与胶对比都合格时能稳定印出的线宽。Mack 把后者写成带 k₁ 的工程式，并用图像对数斜率去谈工艺窗，而不是只报一个分辨距。</p>
    <footer>—— Rayleigh 1879 的两源分辨讨论；Chris A. Mack, Fundamental Principles of Optical Lithography, Wiley, 2007</footer>
</div>

本篇是光刻栏附录链的第一篇，主干课程到此收束。主干里的[瑞利判据：CD = k₁ λ / NA](/litho/rayleigh-litho)已经钉过产线公式、三个旋钮与单次曝光的 $k_1$ 地板；[NILS 与 ILS](/litho/nils-ils)已经把对数斜率的定义与读法写完。附录不重抄这两课，只做文献对照，补三件主干没有的事：0.61 判据的来路、Mack 书里从 NILS 走到窗口的推导线索、以及原书的章节脉络。附录链下一站是 [ASML EUV / High-NA 产品线](/litho/asml-euv-high-na)。

## 问题

0.61 不是从光刻里长出来的，把它当 $k_1$ 的下限是文献史上的张冠李戴。这条线值得按年代读一遍。

```mermaid
flowchart LR
  AIRY["Airy 1835<br/>圆孔衍射斑第一零点"] --> ABBE["Abbe 1873<br/>相干半节距极限"]
  ABBE --> RAY["Rayleigh 1879<br/>两点非相干判据"]
  RAY --> SPAR["Sparrow 1916<br/>拐点并合"]
  SPAR --> K1["1980–90 年代产线<br/>k₁ 打包系数"]
```

Airy（1835）算出圆孔衍射斑的第一零点在 $1.22\,\lambda/D$，对物方半角即 $0.61\,\lambda/\mathrm{NA}$。这只是几何，还不是判据。Abbe（1873）把显微镜成像写成衍射级的收集：相干照明的周期物，半节距极限是 $\lambda/(2\mathrm{NA})$，口语常作 $0.5\lambda/\mathrm{NA}$；数值孔径作为镜头设计量在这里登场。Rayleigh（1879，*Philosophical Magazine*，副题即「面向光谱仪」）提出两点判据：两个非相干点像，一个的中央极大落在另一个的第一极小上，合成剖面刚好有一个可辨的凹陷——$0.61\lambda/\mathrm{NA}$ 由此成为「刚能分开」的代称。Sparrow（1916）改用拐点并合来定分辨：两像剖面刚不出现下凹的位置。那已经是「边还陡不陡」的思路，与半世纪后 NILS 的取向遥遥相望。

半导体产业接手的不是这条判据，而是另一条惯例：1980–90 年代的产线语言与教材（Levinson 一脉）把工程打包系数写成 $k_1$，路线图用 $k_1$ 报进度。0.61 与 $k_1$ 只共享一个字母——前者是 Airy 斑第一零点的几何常数，后者是照明、掩模、胶与计算光刻的打包。

### 判据的接收器不同

光谱学的接收器是眼睛或探测器，问「两根线分不分开」；晶圆厂的接收器是胶加显影，问「线宽在不在公差里，离焦偏剂量后还在不在」。判据史读清楚，就知道两边的数不可互换：把 0.61 写进晶圆厂，等于拿光谱仪的验收单去收线宽的货。

<span class="marginnote">Rayleigh 的 1879 原文动机是光谱双线的分辨，判据对象是点像；阿贝路线处理的是相干周期物。两条线在教科书里常被捏成一句「分辨极限」，各自的适用面并不同。</span>

## 方法

Mack 书里从 NILS 定义走到工艺窗，中间有小信号的一步，主干[曝光宽容度](/litho/exposure-latitude)与[曝光–散焦工艺窗口](/litho/ed-process-window)两课给出完整窗口账；这里只留推导线索。

阈值条件是 $I(x_e)\,E=I_\mathrm{th}$：剂量 $E$ 变 $1+\epsilon$，等效于空中像整体乘同一因子，边从 $x_e$ 挪到 $x_e+\Delta x_e$。两边取对数微分：

$$
\frac{d\ln I}{dx}\Big|_{x_e}\Delta x_e=-\Delta\ln E
\quad\Longrightarrow\quad
\Delta x_e=-\frac{w}{\mathrm{NILS}}\,\Delta\ln E,
$$

其中 $w$ 是目标线宽（NILS 的归一化宽度，定义见 [NILS 与 ILS](/litho/nils-ils)）。倒过来读，就是曝光宽容度的小信号估计：

$$
|\Delta\ln E|\approx \mathrm{NILS}\cdot\frac{|\Delta x_e|}{w}.
$$

数量级一眼可读：NILS $=3$、$w=40\,\mathrm{nm}$、允许 $\pm 2\,\mathrm{nm}$，单边剂量窗约 $1.7\%$。书里就用这条小信号式把空中像的几何接到 E–D 窗的记账；孔、线端与不对称照明各自换 $w$ 与最险的那条边。

## 机制

Mack 原书（*Fundamental Principles of Optical Lithography*, 2007）的章节行进，按主题可以收成一条四段的前向链，这也是本栏主干课程排布的对照系：

- **成像**：先把空中像写成部分相干强度计算——Hopkins 核与源点叠加，评价对象是 $I(x,y)$ 本身（Hopkins 原文的对照见附录 [Hopkins 部分相干成像](/litho/hopkins-1953-paper)）。
- **评价**：把「分辨率」从单一分辨距换到边缘陡度——ILS、NILS、对比度，再装进 E–D 窗与聚焦–剂量预算。NILS 在这一段首次成为接口符号。
- **记录**：进胶——光化学反应、驻波与摆动曲线、PEB 与扩散，把光学 NILS 折算成潜像对比。
- **控制**：把窗口接回产线——套刻、产能、良率与工艺控制，判据从这里交还工程。

读原书的要点是：**NILS 是贯穿接口，不是某一节的孤立定义**。成像部分引入它，窗口部分反复调用它，记录部分把它折给化学，控制部分拿它当工艺的验收量。抓住这条接口，四段就不是四个孤立专题；对照阅读时，[Goodman 傅里叶光学](/litho/goodman-fourier-optics)给频域工具，[Levinson 光刻原理](/litho/levinson-litho-book)给产线工程语境，Mack 居中做成像到窗口的桥。

## 边界

本篇是文献对照，不替代主干的推导。引书按主题回溯，不逐字抄目录编号与页码，防止把转述写成原文。0.61 属于两点非相干判据，Sparrow 判据也不是产线标准，两者都不该出现在工艺规格里。小信号式只在阈值附近线性区可用：NILS 本身随离焦与剂量变，扫到窗口边沿要回到 E–D 的完整账。不给未公开的某代机台 $k_1$ 内部表——那条规矩主干已立，附录只守不重申。

<span class="marginnote">出处：G. B. Airy, 1835；E. Abbe, 1873；Lord Rayleigh, *Phil. Mag.* 1879；C. M. Sparrow, 1916；C. A. Mack, *Fundamental Principles of Optical Lithography*, Wiley, 2007。</span>

## 小结

- 0.61 的文献链：Airy 1835 几何 → Abbe 1873 相干半节距 → Rayleigh 1879 两点判据 → Sparrow 1916 拐点判据；半导体 $k_1$ 是 1980–90 年代另一条产线惯例，与 0.61 不可互换。
- NILS 推导线索：阈值边条件对剂量取对数微分，$\Delta x_e=(w/\mathrm{NILS})\,\Delta\ln E$；曝光宽容度与 NILS 同向，完整窗口账在主干 E–D 课。
- Mack 原书按「成像 → 评价 → 记录 → 控制」行进，NILS 是贯穿四段的接口符号。
- 出处：Airy；Abbe；Rayleigh；Sparrow；Mack 2007。
