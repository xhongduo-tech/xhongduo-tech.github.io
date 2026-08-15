---
title: 磁场、磁感应强度与毕奥-萨伐尔定律
date: 2026-08-07
---

# 磁场、磁感应强度与毕奥-萨伐尔定律

<div class="epigraph">
<p>电流产生磁场，磁场推拉电流——磁是电的近亲，它们之间的对话，由毕奥与萨伐尔第一个用数学写下。</p>
<footer>—— 电磁学引言</footer>
</div>

<div class="article-byline">
<p>第二级 · 高等物理 ｜ 程守洙《普通物理学》第十二章 §12-1 ｜ 2026-08-07</p>
</div>

## 为什么从磁场开始

1820 年，奥斯特发现通电导线让磁针偏转——电与磁的千年分隔被打破。此后电磁学突飞猛进：安培提出电流的磁效应，毕奥与萨伐尔测出电流元产生磁场的定量规律。磁场（magnetic field）与电场是「表亲」：都是场、都携带能量、都满足叠加原理，但磁场有两个独特之处——它由**运动电荷**（电流）产生，且**不存在磁单极子**（磁力线闭合）。这一节定义磁场与磁感应强度，推导**毕奥-萨伐尔定律**——磁场计算的「微元法」。

## 1 磁场与磁感应强度

**磁场（magnetic field）**：运动电荷（电流）周围存在的一种特殊物质，对置于其中的运动电荷或电流施加磁力。

**磁感应强度（magnetic induction / magnetic flux density）** $\boldsymbol{B}$：描述磁场强弱与方向的矢量，单位特斯拉（T），$1\ \text{T} = 1\ \text{N/(A·m)}$。其大小用运动电荷受力定义：电荷 $q$ 以速度 $\boldsymbol{v}$ 在磁场中运动，受洛伦兹力：

$$\boldsymbol{F} = q\boldsymbol{v}\times\boldsymbol{B}$$

当 $\boldsymbol{v} \perp \boldsymbol{B}$ 时力最大 $F_{\max} = qvB$，由此定义 $B = F_{\max}/(qv)$。<span class="marginnote">地磁场约 $5\times10^{-5}$ T，冰箱贴约 $10^{-3}$ T，MRI 磁体 1–7 T，实验室超导磁体可达 20 T 以上。磁感应强度的方向由小磁针北极指向决定——磁力线从 N 极出发回到 S 极。</span>

**重点：磁力对运动电荷不做功**（$\boldsymbol{F} \perp \boldsymbol{v}$），只改变速度方向、不改变速度大小——这是磁场与电场最大的区别。电场力做功改变动能，磁力只「拐弯」不「加速」。

## 2 毕奥-萨伐尔定律

**毕奥-萨伐尔定律（Biot-Savart law）**：电流元 $I\,\mathrm{d}\boldsymbol{l}$ 在空间某点产生的磁感应强度：

$$\mathrm{d}\boldsymbol{B} = \frac{\mu_0}{4\pi}\frac{I\,\mathrm{d}\boldsymbol{l}\times\hat{\boldsymbol{r}}}{r^2}$$

- $\mu_0 = 4\pi\times10^{-7}\ \text{T·m/A}$ 是**真空磁导率**；
- $r$ 是电流元到场点的距离，$\hat{\boldsymbol{r}}$ 是从电流元指向场点的单位矢量；
- 方向：由 $\mathrm{d}\boldsymbol{l}\times\hat{\boldsymbol{r}}$ 决定，**右手螺旋定则**。

大小：$\mathrm{d}B = \frac{\mu_0}{4\pi}\frac{I\,\mathrm{d}l\sin\theta}{r^2}$，$\theta$ 是 $\mathrm{d}\boldsymbol{l}$ 与 $\hat{\boldsymbol{r}}$ 的夹角。

<span class="marginnote">毕奥-萨伐尔定律是磁场的「微元法」，与静电学里点电荷场强公式地位相当：把任意载流导线切成电流元，每个电流元看作「点磁源」，再叠加（积分）。磁场满足叠加原理，所以「拆元 → 写 $\mathrm{d}\boldsymbol{B}$ → 积分」是磁场计算的通用套路。</span>

## 3 磁场叠加原理

多个电流元在空间某点产生的总磁感应强度，等于各电流元单独产生的 $\mathrm{d}\boldsymbol{B}$ 的矢量和：

$$\boldsymbol{B} = \int_L \mathrm{d}\boldsymbol{B} = \frac{\mu_0}{4\pi}\int_L\frac{I\,\mathrm{d}\boldsymbol{l}\times\hat{\boldsymbol{r}}}{r^2}$$

与电场叠加同构，但磁场叠加中每个 $\mathrm{d}\boldsymbol{B}$ 的方向都要用右手定则判断——**矢量积分的复杂性主要来自方向**。<span class="marginnote">解题技巧：先判断对称性——直导线的磁场只有环向分量（可用安培环路定理替代），圆电流轴上只有轴向分量。能用对称性简化方向判断的，绝不硬算三维矢量积分。下一节《毕奥-萨伐尔定律的应用》专门演练。</span>

## 4 公式解析：毕奥-萨伐尔定律的量纲检验

用安培力公式与毕奥-萨伐尔定律交叉验证 $\mu_0$ 的量纲。

$$
\mu_0 = \frac{4\pi r^2\,\mathrm{d}B}{I\,\mathrm{d}l} \quad\Longrightarrow\quad [\mu_0] = \frac{\text{m}^2\cdot(\text{N/(A·m)})}{\text{A}\cdot\text{m}} = \text{N/A}^2 = \text{T·m/A}
$$

- **第一步，写定律**：$\mathrm{d}B = \frac{\mu_0}{4\pi}\frac{I\,\mathrm{d}l\sin\theta}{r^2}$。
- **第二步，反解 $\mu_0$**：$\mu_0 = \frac{4\pi r^2\mathrm{d}B}{I\,\mathrm{d}l}$（忽略 $\sin\theta$ 因子）。
- **第三步，代量纲**：$r^2$ 为 m²，$\mathrm{d}B$ 为 T = N/(A·m)，$I$ 为 A，$\mathrm{d}l$ 为 m，得 $[\mu_0] = \frac{\text{m}^2\cdot\text{N/(A·m)}}{\text{A·m}} = \frac{\text{N}}{\text{A}^2}$。
- **第四步，确认**：$\mu_0 = 4\pi\times10^{-7}\ \text{T·m/A}$，与 $\text{N/A}^2$ 等价（T = N/(A·m)）。量纲自洽，公式可信。

## 5 磁场与电场的对照

| 比较项 | 电场 | 磁场 |
| --- | --- | --- |
| 源 | 电荷（静） | 电流（运动电荷） |
| 场量 | $\boldsymbol{E}$ | $\boldsymbol{B}$ |
| 基本定律 | 库仑定律 / 高斯定理 | 毕奥-萨伐尔定律 / 安培环路定理 |
| 对电荷作用 | $\boldsymbol{F} = q\boldsymbol{E}$（做功） | $\boldsymbol{F} = q\boldsymbol{v}\times\boldsymbol{B}$（不做功） |
| 力线 | 从正电荷出发到负电荷（有源） | 闭合曲线（无源） |
| 介质响应 | 极化，介电常数 $\varepsilon$ | 磁化，磁导率 $\mu$（第十二章末） |

<span class="marginnote">「磁力线闭合、无磁单极子」是磁场与电场最深刻的区别：电场线始于正电荷、终于负电荷（有源），磁力线永远闭合（无源）。这个「无磁荷」特性写进麦克斯韦方程组（$\nabla\cdot\boldsymbol{B} = 0$），是磁场区别于电场的数学指纹。</span>

## 6 数值算例：长直导线的磁场

一条长直导线通以电流 $I = 10\ \mathrm{A}$，求距导线 $r = 0.05\ \mathrm{m}$ 处的磁感应强度。（长直导线磁场公式 $B = \mu_0I/(2\pi r)$，下一节将推导。）

$$

B = \frac{\mu_0I}{2\pi r} = \frac{4\pi\times10^{-7}\times10}{2\pi\times0.05} = 4\times10^{-5}\ \mathrm{T}

$$

- **第一步，写公式**：长直导线磁场 $B = \mu_0I/(2\pi r)$。
- **第二步，代入**：$B = \frac{4\pi\times10^{-7}\times10}{2\pi\times0.05} = \frac{4\times10^{-6}}{0.1} = 4\times10^{-5}$ T。
- **第三步，对照量级**：$4\times10^{-5}$ T 与地磁场（约 $5\times10^{-5}$ T）同量级——10 A 导线在 5 cm 处的磁场足以偏转磁针，这正是奥斯特实验能观察到的强度。
- **第四步，解读**：磁场随 $r$ 按 $1/r$ 衰减（不是 $1/r^2$）——因为长直导线是一维源，磁场集中在环向。这个「维度决定衰减」的规律，在电磁学里反复出现。<span class="marginnote">「磁场 $1/r$ 与电场 $1/r^2$ 的维度差异」：点电荷（零维）电场 $1/r^2$，长直线（一维）磁场 $1/r$，无限大平面（二维）场强均匀——源的维度决定场的衰减快慢。同理，长直导线「无限长」假设让端部效应消失，$B = \mu_0I/(2\pi r)$ 是实际导线的良好近似（离端部足够远时）。</span>

## 7 磁场线、磁通量与磁场的高斯定理

磁场线（磁力线）与磁通量是描述磁场结构的基本工具：

**磁场线性质**：闭合曲线，无始无终（无磁单极子）；不相交；疏密表示强弱。
**磁通量** $\Phi_m = \int \boldsymbol{B}\cdot\mathrm{d}\boldsymbol{S}$，单位韦伯（Wb），$1\ \mathrm{Wb} = 1\ \mathrm{T·m^2}$。
**磁场的高斯定理**：$\oint\boldsymbol{B}\cdot\mathrm{d}\boldsymbol{S} = 0$——闭合曲面磁通量为零（磁场线闭合）。

**重点：$\oint\boldsymbol{B}\cdot\mathrm{d}\boldsymbol{S} = 0$ 是「无磁单极子」的数学表达——磁力线永远是闭合的。** 这个「磁荷不存在」的定律与电场的 $\oint\boldsymbol{E}\cdot\mathrm{d}\boldsymbol{S} = q/\varepsilon_0$ 形成鲜明对照：电场线可以始于电荷、终于电荷，磁力线只能自己闭合。<span class="marginnote">「为什么磁力线必须闭合」：至今实验未发现磁单极子（孤立的 N 极或 S 极），磁体总是成对出现（切割磁铁得到两段都有 N、S 极）。狄拉克曾论证磁单极子的存在会自然推出电荷量子化，但至今未观测到。这个无磁荷性质，写进了麦克斯韦方程组第二式 $\nabla\cdot\boldsymbol{B} = 0$。</span>

**辨析｜易错点：**磁场线的疏密表示 $B$ 大小，但与电场线不同——磁场线无起点终点（闭合），且磁通量穿过的面必须是有向曲面。计算磁通量时先定法向，磁通量的正负依赖法向选择（与曲面取法）。穿过线圈的磁通量是电磁感应（下一章）的核心量，法拉第定律的 $\mathrm{d}\Phi/\mathrm{d}t$ 就靠它。

## 8 术语速查表

| 术语 | 公式 | 要点 |
| --- | --- | --- |
| 磁场 | 运动电荷激发的场 | 电流的磁效应 |
| 磁感应强度 | $\boldsymbol{B}$（特斯拉 T） | 由 $F = qv\times B$ 定义 |
| 毕奥-萨伐尔定律 | $\mathrm{d}\boldsymbol{B} = \frac{\mu_0}{4\pi}\frac{I\,\mathrm{d}\boldsymbol{l}\times\hat{\boldsymbol{r}}}{r^2}$ | 电流元磁场 |
| 磁导率 | $\mu_0 = 4\pi\times10^{-7}$ | 真空磁导率 |
| 磁通量 | $\Phi = \int\boldsymbol{B}\cdot\mathrm{d}\boldsymbol{S}$ | 单位韦伯 |
| 磁场高斯定理 | $\oint\boldsymbol{B}\cdot\mathrm{d}\boldsymbol{S} = 0$ | 无磁单极子 |

磁场由电流（运动电荷）激发，磁感应强度 $\boldsymbol{B}$ 由洛伦兹力定义，毕奥-萨伐尔定律给出电流元的磁场——三者构成磁场的「定义 → 产生 → 计算」链条。磁场与电场的对照（源、力线拓扑、介质响应）是理解电磁学的骨架。下一节我们用毕奥-萨伐尔定律计算典型电流分布的磁场——**毕奥-萨伐尔定律的应用**。

## 9 小结

- **磁场**：电流（运动电荷）激发，对运动电荷施加磁力。
- **磁感应强度**：$\boldsymbol{B}$，单位特斯拉 T，由 $\boldsymbol{F} = q\boldsymbol{v}\times\boldsymbol{B}$ 定义。
- 磁力不做功，只改变运动电荷方向。
- **毕奥-萨伐尔定律**：$\mathrm{d}\boldsymbol{B} = \frac{\mu_0}{4\pi}\frac{I\,\mathrm{d}\boldsymbol{l}\times\hat{\boldsymbol{r}}}{r^2}$；磁场满足叠加原理。
- 磁场 vs 电场：源、场量、力线拓扑都不同；磁力线闭合（无磁单极子）。

在下一节，我们用毕奥-萨伐尔定律计算典型电流分布的磁场——**毕奥-萨伐尔定律的应用**。
