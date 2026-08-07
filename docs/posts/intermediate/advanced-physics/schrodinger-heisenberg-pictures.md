---
title: 薛定谔绘景与海森堡绘景
date: 2026-08-07
---

# 薛定谔绘景与海森堡绘景

<div class="epigraph">
<p>时间演化可以「放在态上」（薛定谔），也可以「放在算符上」（海森堡）——两种绘景给出相同的物理，却通向不同的哲学与数学。</p>
<footer>—— 量子力学引言</footer>
</div>

<div class="article-byline">
<p>第二级 · 高等物理 ｜ 曾谨言《量子力学》第三章 ｜ 2026-08-07</p>
</div>

## 为什么从两种绘景开始

量子力学的第五公设说「态按薛定谔方程演化」——但「谁在演化」有选择的余地。**薛定谔绘景（Schrödinger picture）**让态携带时间依赖、算符恒定；**海森堡绘景（Heisenberg picture）**让算符携带时间依赖、态恒定。两种绘景数学等价（物理结果相同），但海森堡绘景更接近经典力学（算符的时间演化类似哈密顿方程），也是量子场论、散射理论的标准语言。这一节建立两种绘景及其等价性，推导海森堡运动方程。

## 1 薛定谔绘景

**薛定谔绘景（Schrödinger picture）**：态矢量携带时间依赖、算符恒定：

$$|\psi_S(t)\rangle = \hat{U}(t)|\psi_S(0)\rangle, \qquad \hat{A}_S = \text{常量}$$

**时间演化算符**：

$$\hat{U}(t) = e^{-i\hat{H}t/\hbar}$$

（$\hat{H}$ 不显含时间时）。态满足薛定谔方程 $i\hbar\frac{\partial|\psi_S\rangle}{\partial t} = \hat{H}|\psi_S\rangle$。

**重点：薛定谔绘景——态随时间演化（$|\psi_S(t)\rangle = e^{-i\hat{H}t/\hbar}|\psi_S(0)\rangle$），算符恒定。** 这是量子力学的标准表述：态矢量像「旋转的矢量」绕着哈密顿量转，能量本征态只是「相位旋转」$e^{-iE_nt/\hbar}$。测量平均值 $\langle A\rangle(t) = \langle\psi_S(t)|\hat{A}_S|\psi_S(t)\rangle$ 的时间依赖来自态的演化。

## 2 海森堡绘景

**海森堡绘景（Heisenberg picture）**：算符携带时间依赖、态恒定：

$$|\psi_H\rangle = |\psi_S(0)\rangle = \text{常量}, \qquad \hat{A}_H(t) = \hat{U}^\dagger(t)\hat{A}_S\hat{U}(t)$$

**海森堡运动方程**：

$$\frac{\mathrm{d}\hat{A}_H}{\mathrm{d}t} = \frac{i}{\hbar}[\hat{H}, \hat{A}_H] + \frac{\partial\hat{A}_H}{\partial t}$$

**重点：海森堡绘景——算符随时间演化，态恒定；算符满足海森堡运动方程 $\frac{\mathrm{d}\hat{A}}{\mathrm{d}t} = \frac{i}{\hbar}[\hat{H}, \hat{A}] + \frac{\partial\hat{A}}{\partial t}$。** 这个方程与经典力学中「$\frac{\mathrm{d}f}{\mathrm{d}t} = \{f, H\}$」（泊松括号，第 118 节）完全同构——只需把泊松括号换成 $i/\hbar$ 倍对易子。海森堡绘景最清晰地展示了「量子力学 = 经典力学的算符化」。<span class="marginnote">「海森堡绘景 ↔ 经典力学」：经典 $\frac{\mathrm{d}f}{\mathrm{d}t} = \{f,H\}$ 与量子 $\frac{\mathrm{d}\hat{A}}{\mathrm{d}t} = \frac{i}{\hbar}[\hat{H},\hat{A}]$ 形式完全对应（$[\hat{A},\hat{H}] = i\hbar\{A,H\}$，第 118 节）。守恒量判据也对应：$[\hat{H},\hat{A}] = 0$ ⟺ $\{H,A\} = 0$。海森堡绘景是「经典力学的量子版本」，让量子力学与经典力学的最深层结构（泊松括号、守恒律）直接对话。</span>

## 3 两种绘景的等价性

**测量平均值不变**：

$$\langle A\rangle(t) = \langle\psi_S(t)|\hat{A}_S|\psi_S(t)\rangle = \langle\psi_H|\hat{A}_H(t)|\psi_H\rangle$$

两种绘景是同一物理的两种「视角」——用酉变换 $\hat{U}$ 连接：态与算符之间的时间依赖「转移」了。

**重点：薛定谔绘景与海森堡绘景等价——测量平均值相同，只是「时间依赖在态上还是算符上」的分配不同。** 选择哪种看问题方便：定态、束缚态用薛定谔（态演化直观）；含时问题、散射、场论用海森堡（算符演化类似经典）。<span class="marginnote">「绘景的选择」：薛定谔绘景适合「态演化的图像」（能级、跃迁），海森堡绘景适合「算符/可观测量的时间依赖」（散射矩阵、量子场论）。还有第三种「相互作用绘景」——把 $\hat{H} = \hat{H}_0 + \hat{H}'$ 中的 $\hat{H}_0$ 演化放到算符上、$\hat{H}'$ 演化放到态上——是含时微扰（第 139 节）的标准框架。三种绘景物理等价，选最方便的。</span>

## 4 公式解析：海森堡绘景中的位置算符

自由粒子（$\hat{H} = \frac{\hat{p}^2}{2m}$），求位置算符在 Heisenberg 绘景中的时间依赖。

$$
\frac{\mathrm{d}\hat{x}_H}{\mathrm{d}t} = \frac{i}{\hbar}[\hat{H}, \hat{x}_H] = \frac{i}{\hbar}\left[\frac{\hat{p}^2}{2m}, \hat{x}\right] = \frac{\hat{p}_H}{m}
$$

- **第一步，写海森堡方程**：$\frac{\mathrm{d}\hat{x}_H}{\mathrm{d}t} = \frac{i}{\hbar}[\hat{H}, \hat{x}_H]$。
- **第二步，算对易子**：$[\hat{p}^2, \hat{x}] = \hat{p}[\hat{p},\hat{x}] + [\hat{p},\hat{x}]\hat{p} = \hat{p}(-i\hbar) + (-i\hbar)\hat{p} = -2i\hbar\hat{p}$。
- **第三步，代入**：$\frac{\mathrm{d}\hat{x}_H}{\mathrm{d}t} = \frac{i}{\hbar}\cdot\frac{-2i\hbar\hat{p}}{2m} = \frac{\hat{p}_H}{m}$——位置变化率 = 速度算符。
- **第四步，积分**：$\hat{x}_H(t) = \hat{x}_H(0) + \frac{\hat{p}}{m}t$——与经典自由粒子 $x = x_0 + vt$ 形式一致（算符版本的匀速运动）。

**辨析｜易错点：**对易子 $[\hat{p}^2,\hat{x}]$ 的计算要注意算符次序（$\hat{p}$ 与 $\hat{x}$ 不对易）。$\frac{\mathrm{d}\hat{A}_H}{\mathrm{d}t} = \frac{i}{\hbar}[\hat{H},\hat{A}_H]$ 中 $\hat{A}_H$ 是海森堡算符（含时）——对易子也要用海森堡绘景中的算符计算。守恒量：$[\hat{H},\hat{A}] = 0$ ⟹ $\hat{A}_H$ 恒定（与经典守恒对应）。

## 5 两种绘景的意义

- **等价性**：物理结果（平均值、概率）与绘景选择无关——量子力学的自洽性；
- **经典对应**：海森堡方程 ↔ 哈密顿方程（泊松括号）——量子经典对应的最清晰表述；
- **量子场论**：场算符用海森堡绘景（含时）——量子场论、散射理论的标准语言；
- **相互作用绘景**：含时微扰（第 139 节）的自然框架。

**重点：两种绘景的等价性保证量子力学自洽；海森堡绘景的连接经典力学（泊松括号 → 对易子）是量子-经典对应的核心。** 从薛定谔（态演化）到海森堡（算符演化），量子力学的两种语言在不同场景各擅胜场——这正是「同一物理、多种数学表述」的典范。<span class="marginnote">「从极限到大模型的收官」：本专题从经典力学（质点、刚体、流体）出发，经过振动波动、热学、电磁学、光学、相对论、量子物理、四大力学入门，最终到达量子力学的形式理论（两种绘景、对易子、微扰）。这一路，物理从「确定性的力」走到「概率的态」，从「连续」走到「量子」，但数学结构始终相通——泊松括号 ↔ 对易子、最小作用量 ↔ 路径积分、哈密顿量 ↔ 哈密顿算符。「从极限到大模型」，在这里完成了一次从牛顿到量子的完整巡礼。</span>

## 6 小结

- **薛定谔绘景**：态演化（$|\psi_S(t)\rangle = e^{-i\hat{H}t/\hbar}|\psi_S(0)\rangle$）、算符恒定。
- **海森堡绘景**：算符演化（$\hat{A}_H(t) = \hat{U}^\dagger\hat{A}_S\hat{U}$）、态恒定。
- **海森堡方程**：$\frac{\mathrm{d}\hat{A}}{\mathrm{d}t} = \frac{i}{\hbar}[\hat{H},\hat{A}] + \frac{\partial\hat{A}}{\partial t}$——对应经典泊松括号方程。
- **等价性**：两种绘景给出相同平均值——同一物理的两种视角。
- 相互作用绘景是含时微扰的标准框架；量子场论用海森堡绘景。
- 本专题收官：从牛顿到量子，泊松括号↔对易子、作用量↔路径积分、哈密顿↔哈密顿算符一以贯之。
