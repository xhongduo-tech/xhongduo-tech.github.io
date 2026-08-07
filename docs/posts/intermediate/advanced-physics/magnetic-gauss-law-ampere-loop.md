---
title: 磁场的高斯定理与安培环路定理
date: 2026-08-07
---

# 磁场的高斯定理与安培环路定理

<div class="epigraph">
<p>磁场没有源也没有汇——磁力线永远闭合。而环绕电流一圈，磁场的线积分就锁定电流：这就是安培的洞察。</p>
<footer>—— 电磁学引言</footer>
</div>

<div class="article-byline">
<p>第二级 · 高等物理 ｜ 程守洙《普通物理学》第十二章 §12-3 ｜ 2026-08-07</p>
</div>

## 为什么从安培环路定理开始

静电场有两条基本定理：高斯定理（管通量）与环路定理（管环路）。磁场也有对应的两条：**磁场的高斯定理**（磁通量为零，无磁单极子）与**安培环路定理**（$\boldsymbol{B}$ 的环流 = 电流）。安培环路定理特别重要——它是「磁场的源是电流」的定量表述，更是计算对称电流分布磁场（螺线管、无限长导线、螺绕环）的利器，相当于磁场版的高斯定理。这一节把这两条定理讲清楚，并演练安培环路定理的应用。

## 1 磁场的高斯定理

**磁通量（magnetic flux）**：通过某曲面的磁力线总数：

$$\Phi = \int_S \boldsymbol{B}\cdot\mathrm{d}\boldsymbol{S}$$

单位：韦伯（Wb），$1\ \text{Wb} = 1\ \text{T·m}^2$。

**磁场的高斯定理**：通过任意闭合曲面的磁通量恒为零：

$$\oint_S \boldsymbol{B}\cdot\mathrm{d}\boldsymbol{S} = 0$$

**重点：磁场的高斯定理表明「磁单极子不存在」。** 磁力线没有起点和终点，永远闭合——穿入闭合曲面的磁力线必然穿出，净磁通为零。对比电场：电荷是电场线的源（$\oint\boldsymbol{E}\cdot\mathrm{d}\boldsymbol{S} = q/\varepsilon_0$ 可不为零）；磁场没有「磁荷」，只有「磁偶极子」（磁铁永远 N、S 成对）。<span class="marginnote">「磁单极子」是物理学的著名悬念：狄拉克 1931 年指出磁单极子的存在会解释电荷量子化，但至今未被实验证实。若某天发现磁单极子，麦克斯韦方程组要改写——这是「无源」结论的前沿意义。</span>

## 2 安培环路定理

**安培环路定理（Ampere's circuital law）**：磁感应强度沿任意闭合回路的线积分（环流），等于穿过该回路的电流代数和的 $\mu_0$ 倍：

$$\oint_L \boldsymbol{B}\cdot\mathrm{d}\boldsymbol{l} = \mu_0\sum I_{\text{内}}$$

电流的正负由右手定则约定：右手四指沿回路方向，拇指所指方向穿过的电流为正。

**重点：安培环路定理是磁场的「环路」表述，说明电流是磁场的涡旋源。** 它与静电场环路定理（$\oint\boldsymbol{E}\cdot\mathrm{d}\boldsymbol{l} = 0$）形成鲜明对照：静电场无旋（保守）、磁场有旋（电流是源）。<span class="marginnote">与高斯定理类似：安培环路定理对任意回路恒成立，但只有电流分布具有<strong>高度对称性</strong>（无限长导线、无限长螺线管、螺绕环）时，才能用它反推出 $\boldsymbol{B}$——前提是 $\boldsymbol{B}$ 在回路上恒定或为零分量，能提出积分号。</span>

## 3 用安培环路定理求磁场

以无限长直导线为例：电流 $I$ 沿 $z$ 轴，取以导线为轴、半径 $a$ 的圆环为回路。由对称性，$\boldsymbol{B}$ 沿环向恒定，$\boldsymbol{B}\cdot\mathrm{d}\boldsymbol{l} = B\,\mathrm{d}l$：

$$B\cdot 2\pi a = \mu_0 I \quad\Longrightarrow\quad B = \frac{\mu_0 I}{2\pi a}$$

与毕奥-萨伐尔定律的结果一致——两条路殊途同归，安培环路定理显然更简洁。

**长直螺线管**（单位长度匝数 $n$，电流 $I$）：取矩形回路穿过螺线管，管内 $B = \mu_0nI$、管外 $B = 0$：

$$Bl = \mu_0 nIl \quad\Longrightarrow\quad B = \mu_0 nI$$

<span class="marginnote">安培环路定理的四步法：① 对称性判断 $\boldsymbol{B}$ 方向（环向/轴向）；② 选合适回路（圆、矩形），使 $\boldsymbol{B}$ 可提出积分；③ 算环流 $B\oint\mathrm{d}l = B\times(\text{回路长度})$；④ 等于 $\mu_0\sum I_{\text{内}}$ 解出 $B$。关键是选回路让积分「一马平川」。</span>

## 4 公式解析：螺绕环的磁场

一个环形螺线管（螺绕环），总匝数 $N$、电流 $I$、平均半径 $R$，求管内磁场。

$$
B\cdot 2\pi R = \mu_0 NI \quad\Longrightarrow\quad B = \frac{\mu_0 NI}{2\pi R}
$$

- **第一步，分析对称性**：螺绕环内磁场沿环向、大小只与到环心距离有关。
- **第二步，选回路**：取与环同心的圆，半径 $R$ 为环的平均半径，穿过所有 $N$ 匝线圈。
- **第三步，算环流与电流**：$B\cdot 2\pi R = \mu_0NI$（$N$ 匝电流都穿过回路）。
- **第四步，解出**：$B = \frac{\mu_0NI}{2\pi R} = \mu_0nI$（$n = N/(2\pi R)$ 为单位长度匝数）——与长螺线管内部公式一致。螺绕环把磁场「关」在环内，漏磁极小，是高精度电感、互感器的理想结构。

**辨析｜易错点：**安培环路定理中 $\boldsymbol{B}$ 是回路**上所有**电流（含回路外电流）产生的总场，但环流只由**回路内**电流决定。回路外电流对环流无贡献，却影响回路上 $\boldsymbol{B}$ 的分布——这与高斯定理「曲面外电荷不影响通量、影响场强」的逻辑完全平行。两个「只算内源」的定理，都只对通量/环流成立。

## 5 磁场的高斯定理与安培环路定理对照

| 比较项 | 磁场高斯定理 | 安培环路定理 |
| --- | --- | --- |
| 数学形式 | $\oint\boldsymbol{B}\cdot\mathrm{d}\boldsymbol{S} = 0$ | $\oint\boldsymbol{B}\cdot\mathrm{d}\boldsymbol{l} = \mu_0\sum I$ |
| 物理意义 | 磁力线闭合，无磁单极子 | 电流是磁场的源 |
| 对应静电量 | $\oint\boldsymbol{E}\cdot\mathrm{d}\boldsymbol{S} = q/\varepsilon_0$ | $\oint\boldsymbol{E}\cdot\mathrm{d}\boldsymbol{l} = 0$ |
| 计算用途 | 磁通量、磁矩分析 | 对称电流分布的磁场 |
| 麦克斯韦方程组 | $\nabla\cdot\boldsymbol{B} = 0$ | $\nabla\times\boldsymbol{B} = \mu_0\boldsymbol{j}$（静磁） |

<span class="marginnote">这两条定理是麦克斯韦方程组中磁场两方程（$\nabla\cdot\boldsymbol{B} = 0$、$\nabla\times\boldsymbol{B} = \mu_0\boldsymbol{j}$）的积分形式。安培环路定理在第十四章会因「位移电流」被修正——这是麦克斯韦最重要的一步，也是「变化电场产生磁场」的入口。先在这里记住静磁版本。</span>

## 6 小结

- **磁通量**：$\Phi = \int\boldsymbol{B}\cdot\mathrm{d}\boldsymbol{S}$，单位韦伯 Wb。
- **磁场的高斯定理**：$\oint\boldsymbol{B}\cdot\mathrm{d}\boldsymbol{S} = 0$——磁力线闭合，磁单极子不存在。
- **安培环路定理**：$\oint\boldsymbol{B}\cdot\mathrm{d}\boldsymbol{l} = \mu_0\sum I_{\text{内}}$——电流是磁场的涡旋源。
- 应用：无限长直导线 $B = \mu_0I/(2\pi a)$、螺线管 $B = \mu_0nI$、螺绕环 $B = \mu_0NI/(2\pi R)$。
- 与静电对照：磁场「无源有旋」、静电场「有源无旋」。

在下一节，我们研究磁场对运动电荷的作用——**洛伦兹力、带电粒子在磁场中的运动与霍尔效应**。
