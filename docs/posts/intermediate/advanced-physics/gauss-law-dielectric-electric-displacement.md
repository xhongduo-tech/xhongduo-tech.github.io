---
title: 有电介质时的高斯定理与电位移矢量
date: 2026-08-07
---

# 有电介质时的高斯定理与电位移矢量

<div class="epigraph">
<p>电位移矢量把「自由电荷」从「束缚电荷」的纠缠中解放出来——高斯定理从此只看自由电荷的眼色。</p>
<footer>—— 静电学引言</footer>
</div>

<div class="article-byline">
<p>第二级 · 高等物理 ｜ 程守洙《普通物理学》第十一章 §11-5 ｜ 2026-08-07</p>
</div>

## 为什么从电位移矢量开始

上一节我们看到电介质极化产生束缚电荷，这让「真空中的高斯定理」在介质中变得麻烦：闭合曲面内的电荷既有自由电荷又有束缚电荷，而束缚电荷往往未知。这一节引入**电位移矢量（electric displacement）** $\boldsymbol{D}$——一个把束缚电荷的效应「吸收」进定义的辅助场量。有了它，有介质时的高斯定理写成「$\boldsymbol{D}$ 的通量 = 自由电荷」，形式与真空一致、却只含自由电荷。这是处理介质中电场问题的标准工具，也是第二十一章麦克斯韦方程组中 $\boldsymbol{D}$ 的首次亮相。

## 1 电位移矢量

**电位移矢量（electric displacement）** 定义：

$$\boldsymbol{D} = \varepsilon_0\boldsymbol{E} + \boldsymbol{P}$$

其中 $\boldsymbol{E}$ 是介质中的总电场，$\boldsymbol{P}$ 是极化强度。对各向同性线性介质（$\boldsymbol{P} = \varepsilon_0\chi_e\boldsymbol{E}$）：

$$\boldsymbol{D} = \varepsilon_0\boldsymbol{E} + \varepsilon_0\chi_e\boldsymbol{E} = \varepsilon_0(1+\chi_e)\boldsymbol{E} = \varepsilon\boldsymbol{E}$$

其中 $\varepsilon = \varepsilon_0\varepsilon_r$ 是**介电常数（电容率）**。<span class="marginnote">$\boldsymbol{D}$ 是「辅助场量」，不是全新的物理场——它的意义在于简化介质中的计算。$\boldsymbol{D}$ 的单位是 $\text{C/m}^2$（电荷面密度的量纲），这个名字「电位移」来自历史（位移电流的提出），不必纠结字面。</span>

**重点：$\boldsymbol{D}$ 的源是自由电荷，$\boldsymbol{E}$ 的源是全部电荷（自由 + 束缚）。** 这正是引入 $\boldsymbol{D}$ 的目的：把「不知道的束缚电荷」藏进 $\boldsymbol{D}$ 的定义里，让方程只面对「知道的自由电荷」。

## 2 有介质时的高斯定理

**有介质时的高斯定理**：通过任意闭合曲面的电位移通量，等于曲面内**自由电荷**的代数和：

$$\oint_S \boldsymbol{D}\cdot\mathrm{d}\boldsymbol{S} = \sum q_{\text{自由}}$$

对比真空高斯定理 $\oint\boldsymbol{E}\cdot\mathrm{d}\boldsymbol{S} = \sum q_{\text{全部}}/\varepsilon_0$——两者的差别正是「束缚电荷被吸收进 $\boldsymbol{D}$」。

**重点：有介质时的高斯定理与真空形式完全同构，只是把 $\varepsilon_0\boldsymbol{E}$ 换成 $\boldsymbol{D}$、电荷换成自由电荷。** 所有上一节的方法（选高斯面、对称性、反推场量）原封不动搬过来，只需最后用 $\boldsymbol{D} = \varepsilon\boldsymbol{E}$ 换回 $\boldsymbol{E}$。<span class="marginnote">解题套路：① 对称性选高斯面 → ② 用 $\oint\boldsymbol{D}\cdot\mathrm{d}\boldsymbol{S} = \sum q_{\text{自由}}$ 求 $\boldsymbol{D}$ → ③ 用 $\boldsymbol{D} = \varepsilon\boldsymbol{E}$ 求 $\boldsymbol{E}$。三步走，束缚电荷从头到尾不需要显式求解。</span>

## 3 公式解析：介质中的平行板

平行板电容器极板自由电荷面密度 $\sigma_0$，板间填满相对介电常数 $\varepsilon_r$ 的介质。求介质中的 $\boldsymbol{D}$、$\boldsymbol{E}$ 与束缚电荷面密度。

$$
D = \sigma_0, \qquad E = \frac{D}{\varepsilon_0\varepsilon_r} = \frac{\sigma_0}{\varepsilon_0\varepsilon_r}
$$

- **第一步，选高斯面求 $D$**：取跨介质的圆柱盒高斯面（一底在极板内），$\oint\boldsymbol{D}\cdot\mathrm{d}\boldsymbol{S} = D\cdot\Delta S = \sigma_0\Delta S$，得 $D = \sigma_0$——**电位移在数值上等于自由电荷面密度**。
- **第二步，换回 $E$**：$E = D/(\varepsilon_0\varepsilon_r) = \sigma_0/(\varepsilon_0\varepsilon_r)$——介质中场强削弱 $\varepsilon_r$ 倍。
- **第三步，对比真空**：真空时 $E_0 = \sigma_0/\varepsilon_0$，故 $E = E_0/\varepsilon_r$，与上节结论一致。
- **第四步，求束缚电荷（可选）**：$\sigma' = \sigma_0(1 - 1/\varepsilon_r)$——用 $\boldsymbol{D}$ 方法时这一步甚至可以不求。

**辨析｜易错点：**$\boldsymbol{D}$ 与 $\boldsymbol{E}$ 的边界行为不同：$\boldsymbol{D}$ 的法向分量由自由电荷决定（连续或跳变按自由电荷面密度），$\boldsymbol{E}$ 的法向分量还与束缚电荷有关。解题时「先 $\boldsymbol{D}$ 后 $\boldsymbol{E}$」的顺序不能反——直接用 $\boldsymbol{E}$ 的高斯定理会撞上未知的束缚电荷。

## 4 静电场的能量密度

有了 $\boldsymbol{D}$ 与 $\boldsymbol{E}$，静电场的能量可写成更一般的形式。真空中能量密度：

$$w_e = \frac{1}{2}\varepsilon_0 E^2$$

介质中：

$$w_e = \frac{1}{2}\boldsymbol{D}\cdot\boldsymbol{E} = \frac{1}{2}\varepsilon E^2$$

电容器储能 $W = \int w_e\,\mathrm{d}V$ 与 $W = \frac{1}{2}CU^2$ 一致。<span class="marginnote">「能量储存在电场里」是法拉第场思想的关键推论：不是储存在电荷上，而是储存在电荷激发的场中。能量密度 $\frac{1}{2}\varepsilon E^2$ 与运动学能量、弹簧势能的形式同构，再次体现「$\frac{1}{2}$×系数×（广义量）²」的普适结构。</span>

**重点：静电场能量密度 $w_e = \frac{1}{2}\varepsilon E^2$，正比于场强平方。** 填介质（$\varepsilon$ 增大）后同样场强下储能增大——这是高介电常数材料能储存更多能量的原因。下节《静电场的能量》会专门讨论电容储能的细节。

## 5 电位移矢量与真空中高斯定理的对照

| 比较项 | 真空高斯定理 | 介质中高斯定理 |
| --- | --- | --- |
| 场量 | $\boldsymbol{E}$ | $\boldsymbol{D} = \varepsilon\boldsymbol{E}$ |
| 曲面内电荷 | 全部电荷 | 自由电荷 |
| 应用条件 | 真空 | 任意介质（线性/非线性均可定义） |
| 关系 | $\oint\boldsymbol{E}\cdot\mathrm{d}\boldsymbol{S} = \sum q/\varepsilon_0$ | $\oint\boldsymbol{D}\cdot\mathrm{d}\boldsymbol{S} = \sum q_{\text{自由}}$ |

<span class="marginnote">两条高斯定理殊途同归：真空中 $\boldsymbol{D} = \varepsilon_0\boldsymbol{E}$，介质中 $\boldsymbol{D} = \varepsilon\boldsymbol{E}$。$\boldsymbol{D}$ 的语言在第二十一章麦克斯韦方程组的微分形式（$\nabla\cdot\boldsymbol{D} = \rho_f$）中延续，是电磁场基本方程的核心场量之一。</span>

## 6 小结

- **电位移矢量**：$\boldsymbol{D} = \varepsilon_0\boldsymbol{E} + \boldsymbol{P} = \varepsilon\boldsymbol{E}$；源是自由电荷。
- **有介质时的高斯定理**：$\oint\boldsymbol{D}\cdot\mathrm{d}\boldsymbol{S} = \sum q_{\text{自由}}$。
- 解题三步：选高斯面求 $\boldsymbol{D}$ → 用 $\boldsymbol{D} = \varepsilon\boldsymbol{E}$ 求 $\boldsymbol{E}$ → 必要时求束缚电荷。
- 平行板介质中：$D = \sigma_0$，$E = \sigma_0/(\varepsilon_0\varepsilon_r)$。
- 能量密度：$w_e = \frac{1}{2}\boldsymbol{D}\cdot\boldsymbol{E} = \frac{1}{2}\varepsilon E^2$。

在下一节，我们深入研究静电场的能量——**静电场的能量**，从电容器储能到能量密度。
