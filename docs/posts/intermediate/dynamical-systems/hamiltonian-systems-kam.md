---
title: Hamilton 系统与 KAM 简介
date: 2026-08-11
---

# Hamilton 系统与 KAM 简介

<div class="epigraph">
<p>自然界中所有普通的运动，都是某种变分原理的解。</p>
<footer>—— 威廉 · 哈密顿（William Rowan Hamilton）</footer>
</div>

<div class="article-byline">
<p>第二级 · 进阶数理 · 动力系统 ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么从 Hamilton 系统开始

前一篇说二维无混沌，因为「面积守恒」。这句话的严格版本，就是 **Hamilton 系统**：能量守恒系统在辛几何（symplectic）结构下，相空间体积沿流不变（Liouville 定理）。从行星轨道到粒子加速器，从统计力学的相空间到机器学习里的 Hamiltonian Monte Carlo 采样，Hamilton 系统是**保守动力学的通用语言**。<span class="marginnote">本专题第三篇《相平面与保守系统》是 Hamilton 系统的一维特例（自由度 1）；本篇把框架升到任意自由度，并回答「可积 vs 混沌」的世纪难题——这也是 KAM 定理的舞台。</span>

KAM 定理（Kolmogorov–Arnold–Moser，1954–1963）回答了哈密顿意义下最深刻的问题：**可积系统被小扰动后，运动是否仍然「规则」？** 答案是「多数不变环面保持」——这个微妙的「多数」是混沌与秩序交界处的精细定理。

## 1 定义：从拉格朗日到 Hamilton

对自由度 $n$ 的系统，取广义坐标 $\mathbf{q} = (q_1,\dots,q_n)$ 与广义动量 $\mathbf{p} = (p_1,\dots,p_n)$，相空间 $\mathbb{R}^{2n}$。**Hamilton 函数（能量）** $H(\mathbf{p},\mathbf{q})$，系统由

$$
\dot{q}_i = \frac{\partial H}{\partial p_i}, \qquad \dot{p}_i = -\frac{\partial H}{\partial q_i}
$$

决定。沿解轨道：

$$
\frac{dH}{dt} = \sum_i \left(\frac{\partial H}{\partial q_i}\dot{q}_i + \frac{\partial H}{\partial p_i}\dot{p}_i\right)
= \sum_i \left(\frac{\partial H}{\partial q_i}\frac{\partial H}{\partial p_i} - \frac{\partial H}{\partial p_i}\frac{\partial H}{\partial q_i}\right) = 0.
$$

**Hamilton 系统最重要的守恒律：$H$ 本身是守恒量**——能量守恒被写进方程结构，而不是额外假设。<span class="marginnote">例：单摆 $H = \frac{1}{2}p^2 - \cos q$，Hamilton 方程给出 $\dot{q} = p$、$\dot{p} = -\sin q$，正是第三篇的单摆系统。能量守恒的几何根源是辛结构，比「力学里的势能定义」更根本。</span>

**Liouville 定理**：Hamilton 流在相空间中保持体积（更一般地，保持辛 2-形式 $dp \wedge dq$）。这就是「二维无混沌」背后的守恒量：面积不会被压缩，不存在吸引子。**Hamilton 系统没有奇怪吸引子、没有指数收缩，只有守恒的秩序。**

## 2 可积系统与作用-角坐标

**Liouville–Arnold 可积系统**：若有 $n$ 个相互独立的守恒量（含 $H$），且它们彼此对合（泊松括号为零），则相空间被分层为 $n$ 维不变环面 $\mathbb{T}^n$，运动在上面是**准周期**的。引入**作用-角坐标** $(I, \theta)$，Hamilton 化为

$$
H = H_0(I), \qquad \dot{I} = 0, \qquad \dot{\theta} = \omega(I) = \frac{\partial H_0}{\partial I},
$$

解直接写出来：$\theta(t) = \theta_0 + \omega(I)t$——**可积系统的运动全部是「沿不变环面的匀速扫频」**。<span class="marginnote">单摆、Kepler 行星运动、谐振子都是可积的。Kepler 问题有能量、角动量向量共 3 个守恒量（自由度 3），轨道封闭成椭圆。可积系统「太整齐」，因而被当作研究扰动的基础而非终点。</span>

**不变环面的意义**：$I$ 是环面坐标，$\omega(I)$ 是角频率。若频率比 $\omega_1:\omega_2:\dots$ 都是有理数，轨道闭合成周期轨道；若含无理比，则轨线稠密铺满环面，运动准周期。环面是「秩序的几何化身」。

## 3 扰动与 KAM 定理

现实系统几乎都带扰动：行星有相互引力、加速器有非线性磁铁。把可积系统加个小扰动：

$$
H(I, \theta) = H_0(I) + \varepsilon H_1(I, \theta), \qquad 0 < \varepsilon \ll 1.
$$

**KAM 定理**（非形式陈述）：在非常一般的非退化条件（$\det (\partial^2 H_0/\partial I^2) \neq 0$）下，对足够小的 $\varepsilon$，**大部分不变环面不消失**，只是被轻微扭形，仍被准周期运动充满；但运动不再是精确准周期的——环面附近的运动既不清扫、也不落入混沌，而是「被压扁但还活着」的拟环面。<span class="marginnote">「大部分」的具体含义是：频率向量 $\omega$ 满足「足够无理」（Diophantine 条件 $|\langle k, \omega \rangle| \ge c|k|^{-\tau}$）的环面保留。无理数与有理数的测度理论在这里登场：无理数太多，被破坏的环面只占零测集——这是数论与动力系统最著名的会师。</span>

被破坏的环面（有理共振环面）留下空隙，扰动在其中产生**小混沌区**：轨道在环面残骸间「漂移」——**Arnold 扩散**。于是整体图景是：**稳定的拟环面海，与混沌的岛屿链交错**。

**辨析｜易错点：**KAM 定理**不是**「所有轨道都规则」。它只保证「多数环面保持」；被破坏的环面处确实出现混沌区，且混沌区的测度随 $\varepsilon$ 增大而膨胀。**可积系统的完全秩序 + 任意小扰动 = 秩序与混沌的精细混合**——既不能简单说「扰动后仍可积」，也不能说「扰动即混沌」。

## 4 从 Hamilton 到采样：现代应用

KAM 与 Hamilton 系统并非古董。**Hamiltonian Monte Carlo（HMC）**是当前贝叶斯机器学习的主力采样器：把目标分布 $p(\mathbf{x})$ 的负对数写成势能，引入辅助动量，构造 Hamilton 方程并在相空间沿能量守恒轨道积分，用「保守动力学」大步探索参数空间。<span class="marginnote">HMC 每次迭代都「发射」一条哈密顿轨道再接受——因为能量守恒，接受率远高于普通 Metropolis；而它的理论根基正是 Hamilton 流保持体积 + 辛结构，见第三级《贝叶斯推断》与第二级《概率论与数理统计》。</span>

对深度学习，保守系统的「无吸引子」有另一层启示：**能量守恒的动力学不会自动收敛**，收敛需要引入耗散（梯度下降本身是耗散系统）——这解释了为什么优化器要加动量、摩擦（SGD with momentum 类比 Hamilton 系统 + 阻尼）。

## 5 公式解析：Liouville 定理的散度证明

$$
\sum_{i=1}^n \left( \frac{\partial \dot{q}_i}{\partial q_i} + \frac{\partial \dot{p}_i}{\partial p_i} \right) = 0
$$

- **$\dot{q}_i = \partial H/\partial p_i$、$\dot{p}_i = -\partial H/\partial q_i$**：Hamilton 方程给出相速度场，散度是「速度场在相空间中的压缩率」。
- **求偏导**：$\partial \dot{q}_i/\partial q_i = \partial^2 H/\partial p_i \partial q_i$，$\partial \dot{p}_i/\partial p_i = -\partial^2 H/\partial q_i \partial p_i$——二阶混合偏导数。
- **相消**：混合偏导数对称（$\partial^2 H/\partial p\partial q = \partial^2 H/\partial q\partial p$），两项一正一负抵消为零——散度为 0，即相体积不变。
- **直觉**：Hamilton 流在相空间中「可压缩率恰好为零」，因为「$q$ 方向被拉伸多少，$p$ 方向就被压缩多少」——这种「挤压-拉伸配对」的辛结构，保证体积绝对守恒，也排除了所有「收缩型吸引子」。

## 6 小结

- **Hamilton 方程** $\dot{q} = \partial H/\partial p$、$\dot{p} = -\partial H/\partial q$：能量守恒被写进结构；**Liouville 定理**保证相体积守恒，Hamilton 系统无吸引子。
- **可积系统**：$n$ 个对合守恒量 → 不变环面 + 准周期运动，作用-角坐标下解完全可写。
- **KAM 定理**：小扰动下「足够无理」的环面保持，被破坏的环面留下混沌岛链与 Arnold 扩散。
- **辨析**：KAM ≠「扰动后仍完全规则」，而是「多数秩序 + 少数混沌」的精细混合。

在下一节，我们将走进混沌的正中心：**奇怪吸引子与 Lyapunov 指数**，用定量指标测量「蝴蝶效应」的强度与吸引子的几何。
