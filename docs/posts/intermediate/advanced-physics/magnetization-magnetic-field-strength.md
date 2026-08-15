---
title: 磁介质的磁化与磁场强度
date: 2026-08-07
---

# 磁介质的磁化与磁场强度

<div class="epigraph">
<p>铁钉会吸铁，指南针指北，地球是块大磁铁——磁介质对磁场的响应，从一块铁芯开始，延伸到整个星球的磁场。</p>
<footer>—— 电磁学引言</footer>
</div>

<div class="article-byline">
<p>第二级 · 高等物理 ｜ 程守洙《普通物理学》第十二章 §12-6 ｜ 2026-08-07</p>
</div>

## 为什么从磁介质开始

电场会在电介质中激发束缚电荷（极化），磁场也会在磁介质中激发**磁化电流**（magnetization current）——物质的微观磁矩（电子轨道运动与自旋）在磁场中转向排列，宏观上表现出磁性。铁、镍、钴能被磁化并放大磁场（铁芯让电磁铁强大百倍），而大多数物质对磁场几乎无感。这一节引入**磁化强度**与**磁场强度** $\boldsymbol{H}$，建立「有介质时的安培环路定理」，与电介质中的 $\boldsymbol{D}$ 完全对偶。这是磁学走向材料应用（变压器铁芯、磁存储）的关键一步。<span class="marginnote">把上一节《电介质的极化》与本节对照学习效果最好：极化强度 $\boldsymbol{P}$ ↔ 磁化强度 $\boldsymbol{M}$，电位移 $\boldsymbol{D}$ ↔ 磁场强度 $\boldsymbol{H}$，束缚电荷 ↔ 磁化电流。电介质里学会的「先 $\boldsymbol{D}$ 后 $\boldsymbol{E}$」，搬到磁介质里就是「先 $\boldsymbol{H}$ 后 $\boldsymbol{B}$」——一套功夫，两处使用。</span>

## 1 磁介质与磁化

**磁介质（magnetic medium）**：对磁场有响应的物质。磁场使介质内微观磁矩取向排列的过程叫**磁化（magnetization）**。

**磁化强度** $\boldsymbol{M}$：单位体积内分子磁矩的矢量和。

磁化产生的效果是介质表面出现**磁化电流（束缚电流）**——与电介质表面的束缚电荷完全对偶。磁化电流是束缚在原子内的微观电流的宏观效果，不涉及电荷的宏观流动。

**磁介质的分类**（按相对磁导率 $\mu_r$）：

| 类型 | $\mu_r$ | 行为 | 例子 |
| --- | --- | --- | --- |
| 顺磁质 | 略大于 1 | 弱增强磁场 | 铝、氧气、钠 |
| 抗磁质 | 略小于 1 | 弱削弱磁场 | 铜、水、铋 |
| 铁磁质 | 远大于 1（可达数千） | 强烈增强磁场 | 铁、钴、镍 |

<span class="marginnote">顺磁质与抗磁质都是「弱磁性」，$\mu_r$ 与 1 的差别在 $10^{-5}\sim10^{-3}$ 量级，日常几乎感觉不到；铁磁质的 $\mu_r$ 可达数千，是电磁铁、变压器的材料基础。三类磁介质的差别源于微观磁矩的排列能力：铁磁质有自发磁化的「磁畴」结构。</span>

**重点：铁磁质的关键是「磁畴」。** 铁磁质内部自发分成许多微小磁化区域（磁畴），外磁场使磁畴转向、长大，整体呈现强磁化。撤去外磁场后磁畴不完全回复，留下**剩磁**——这就是永磁体的来源，也是磁记录（硬盘、磁带）的原理。

**辨析｜易错点：**磁化电流（束缚电流）不是真实的电荷流动——它是原子内电子运动的宏观等效，不参与电荷输运、不产生焦耳热。它与传导电流的区别正像束缚电荷与自由电荷的区别：一个被「吸收」进辅助场量，一个作为源出现在环路定理右侧。考试里「铁芯里流动的是磁化电流而非传导电流」是概念判断题的高频点。

## 2 磁场强度与有介质时的安培环路定理

类似电介质中引入 $\boldsymbol{D}$，磁介质中引入**磁场强度（magnetic field strength）**：

$$\boldsymbol{H} = \frac{\boldsymbol{B}}{\mu_0} - \boldsymbol{M}$$

对各向同性线性介质（$\boldsymbol{M} = \chi_m\boldsymbol{H}$）：

$$\boldsymbol{B} = \mu_0(1+\chi_m)\boldsymbol{H} = \mu\boldsymbol{H}, \qquad \mu = \mu_0\mu_r$$

其中 $\chi_m$ 是磁化率，$\mu_r = 1+\chi_m$ 是**相对磁导率**。

**有介质时的安培环路定理**：

$$\oint_L \boldsymbol{H}\cdot\mathrm{d}\boldsymbol{l} = \sum I_{\text{传导}}$$

**重点：$\boldsymbol{H}$ 的环流只由传导电流（自由电流）决定，磁化电流被吸收进 $\boldsymbol{H}$ 的定义。** 这与电介质中 $\oint\boldsymbol{D}\cdot\mathrm{d}\boldsymbol{S} = \sum q_{\text{自由}}$ 完全对偶：$\boldsymbol{H}$ 对应 $\boldsymbol{D}$，传导电流对应自由电荷。<span class="marginnote">磁场问题的完整套路：① 对称性选回路，用 $\oint\boldsymbol{H}\cdot\mathrm{d}\boldsymbol{l} = \sum I_{\text{传导}}$ 求 $\boldsymbol{H}$；② 用 $\boldsymbol{B} = \mu\boldsymbol{H}$ 求 $\boldsymbol{B}$。磁化电流从头到尾不用显式求解——与电介质的「先 $\boldsymbol{D}$ 后 $\boldsymbol{E}$」如出一辙。</span>

**辨析｜易错点：**$\boldsymbol{H}$ 的单位是 A/m（安培每米），不是特斯拉（T）。$\boldsymbol{B}$ 与 $\boldsymbol{H}$ 的关系 $\boldsymbol{B} = \mu\boldsymbol{H}$ 里藏着磁导率，量纲上 A/m × H/m = T。初学常把两者单位混用——先分清「驱动力（$\boldsymbol{H}$，A/m）」与「响应场（$\boldsymbol{B}$，T）」，符号与单位都清楚了。

一个螺绕环算例：环上均匀绕线 500 匝、电流 0.5 A、平均半径 0.05 m，则 $H = NI/(2\pi r) = 500\times0.5/(2\pi\times0.05) \approx 796\ \text{A/m}$——只由匝数、电流与环长决定，与环内是否有铁芯无关。填铁芯后 $B = \mu_0\mu_rH$ 被放大，而 $H$ 纹丝不动。这个「$\boldsymbol{H}$ 与介质无关」的性质，正是解题时先求 $\boldsymbol{H}$ 的底气。

## 3 公式解析：铁芯螺线管

一个长螺线管，单位长度匝数 $n = 1000\ \text{m}^{-1}$，电流 $I = 2\ \text{A}$，管内填满相对磁导率 $\mu_r = 5000$ 的铁芯。求管内 $\boldsymbol{H}$ 与 $\boldsymbol{B}$。

$$
H = nI = 1000\times2 = 2000\ \text{A/m}, \qquad B = \mu_0\mu_r H = 4\pi\times10^{-7}\times5000\times2000
$$

- **第一步，求 $\boldsymbol{H}$**：$\oint\boldsymbol{H}\cdot\mathrm{d}\boldsymbol{l} = Hl = nIl$，$H = nI = 2000\ \text{A/m}$——与真空时相同。
- **第二步，求 $\boldsymbol{B}$**：$B = \mu_0\mu_rH = 4\pi\times10^{-7}\times5000\times2000 \approx 12.6\ \text{T}$。
- **第三步，对比真空**：真空时 $B_0 = \mu_0H \approx 2.5\times10^{-3}$ T；铁芯让磁场增强 5000 倍——这就是电磁铁的原理。
- **第四步，体会**：$\boldsymbol{H}$ 只由传导电流决定（与介质无关），$\boldsymbol{B}$ 则由介质放大。磁场强度的引入把「场」与「物质的响应」分离，工程上先算 $\boldsymbol{H}$ 再乘磁导率。

为什么先求 $\boldsymbol{H}$ 再乘磁导率？因为 $\boldsymbol{H}$ 描述的是「外源产生的磁化驱动力」，与材料无关；$\boldsymbol{B}$ 描述的是「实际磁场」，包含材料的响应。工程上设计电磁铁、变压器时，先由绕组电流定 $\boldsymbol{H}$，再由铁芯材料的 $B$–$H$ 曲线（或 $\mu_r$）读 $\boldsymbol{B}$——把「源」与「响应」分离，正是引入 $\boldsymbol{H}$ 的全部意义。铁芯材料的 $B$–$H$ 曲线往往非线性（见下一节磁滞回线），但无论曲线多复杂，$\boldsymbol{H}$ 都只由传导电流唯一确定。

## 4 铁磁质与磁滞回线

铁磁质的磁化不是线性的：$\boldsymbol{B}$ 与 $\boldsymbol{H}$ 的关系呈现**磁滞回线（hysteresis loop）**。

- 从 $H = 0$ 开始增大 $H$，$B$ 沿初始磁化曲线上升；
- 减小 $H$ 到零，$B$ 不为零——**剩磁（remanence）** $B_r$；
- 反向加 $H$ 使 $B$ 归零，所需反向场强叫**矫顽力（coercivity）** $H_c$；
- 循环一周画出闭合的磁滞回线。

<span class="marginnote">磁滞回线的形状决定材料用途：软磁材料（回线窄、矫顽力小）易磁化也易退磁，用于变压器铁芯（减少涡流与磁滞损耗）；硬磁材料（回线宽、矫顽力大）保留剩磁，用于永磁体与磁存储。磁滞回线面积代表每次磁化循环消耗的能量（磁滞损耗）。</span>

**辨析｜易错点：**磁滞现象说明铁磁质的 $\boldsymbol{B}$ 不仅依赖当前的 $\boldsymbol{H}$，还依赖「历史」（磁化路径）——$\boldsymbol{B}$ 是 $\boldsymbol{H}$ 的多值函数。这就是「铁磁质有记忆」：它记住过去的磁场，这是磁存储（硬盘）的物理基础。考试中「磁滞回线面积越大，磁滞损耗越大」是常见考点。

### 软磁与硬磁材料对照

| 特性 | 软磁材料 | 硬磁材料 |
| --- | --- | --- |
| 磁滞回线 | 窄而瘦 | 宽而胖 |
| 矫顽力 $H_c$ | 小（易磁化易退磁） | 大（保留剩磁） |
| 典型材料 | 硅钢、坡莫合金 | 钕铁硼、铝镍钴 |
| 主要用途 | 变压器、电机铁芯 | 永磁体、扬声器、硬盘 |

对变压器设计，铁芯选择的关键指标正是磁滞回线的面积：面积越小，每次交流磁化循环损耗的能量越少。硅钢片的磁滞回线极窄，加上叠片结构抑制涡流，才让电网变压器能以接近 99% 的效率工作——磁滞损耗虽不起眼，却是能量效率的大账。

## 5 磁介质与电介质的对偶

| 比较项 | 电介质 | 磁介质 |
| --- | --- | --- |
| 响应量 | 极化强度 $\boldsymbol{P}$ | 磁化强度 $\boldsymbol{M}$ |
| 辅助场量 | 电位移 $\boldsymbol{D} = \varepsilon_0\boldsymbol{E}+\boldsymbol{P}$ | 磁场强度 $\boldsymbol{H} = \boldsymbol{B}/\mu_0 - \boldsymbol{M}$ |
| 介质方程 | $\boldsymbol{D} = \varepsilon\boldsymbol{E}$ | $\boldsymbol{B} = \mu\boldsymbol{H}$ |
| 源 | 自由电荷 | 传导电流 |
| 有介质定理 | $\oint\boldsymbol{D}\cdot\mathrm{d}\boldsymbol{S} = \sum q_{\text{自由}}$ | $\oint\boldsymbol{H}\cdot\mathrm{d}\boldsymbol{l} = \sum I_{\text{传导}}$ |
| 极端材料 | 铁电体（大 $\varepsilon_r$） | 铁磁体（大 $\mu_r$） |

<span class="marginnote">「电 ↔ 磁」的完整对偶是电磁学最优雅的结构：$\boldsymbol{D} \leftrightarrow \boldsymbol{B}$、$\boldsymbol{E} \leftrightarrow \boldsymbol{H}$、极化 ↔ 磁化、自由电荷 ↔ 传导电流。记住一侧，另一侧自动复现。这个对偶在麦克斯韦方程组的对称性、以及第二十一章的电磁场理论里贯穿始终。</span>

## 6 小结

- **磁化**：微观磁矩取向排列，表面出现磁化电流；磁化强度 $\boldsymbol{M}$。
- 三类磁介质：顺磁（$\mu_r>1$ 微）、抗磁（$\mu_r<1$ 微）、铁磁（$\mu_r\gg1$）。
- **磁场强度**：$\boldsymbol{H} = \boldsymbol{B}/\mu_0 - \boldsymbol{M}$，$\boldsymbol{B} = \mu\boldsymbol{H}$。
- **有介质安培环路定理**：$\oint\boldsymbol{H}\cdot\mathrm{d}\boldsymbol{l} = \sum I_{\text{传导}}$——只看传导电流。
- 铁磁质磁滞回线：剩磁、矫顽力、磁滞损耗；软磁/硬磁材料之分。
- 电介质 ↔ 磁介质完整对偶：$\boldsymbol{D}\leftrightarrow\boldsymbol{B}$、$\boldsymbol{E}\leftrightarrow\boldsymbol{H}$。

在下一节，我们进入**第十三章《电磁感应》**——从电磁感应现象与法拉第电磁感应定律开始。
