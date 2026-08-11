---
title: 共形场论（CFT）
date: 2026-08-11
---

# 共形场论（CFT）

<div class="epigraph">
<p>二维世界面上的量子场论，其局部对称性是无穷维的；这无穷维的对称性是一切可解性的来源。</p>
<footer>—— 改编自 Joseph Polchinski, <i>String Theory, Vol. 1</i> Ch. 3</footer>
</div>

<div class="article-byline">
<p>第四级 · 高阶专题 · 弦论与量子引力 ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么从共形场论开始

弦的世界面是一个二维曲面，其上居住的量子场论必须满足一个苛刻的条件：**共形不变性**（上一节的中心荷 $c=0$）。一旦把「弦 = 共形场论（Conformal Field Theory, CFT）」这句话立起来，我们获得的东西远超弦本身——CFT 是统计力学临界现象、凝聚态（如 Ising 模型临界点）、以及后来的 AdS/CFT 对偶的共同语言。<span class="marginnote">CFT 在 1984 年由 Belavin–Polyakov–Zamolodchikov（BPZ）奠基；1986 年 Friedan–Qiu–Shenker 用 Virasoro 代数的幺正表示分类了极小模型。弦论的共形世界面与统计力学的临界体系，用同一套无穷维对称性说话——这是 1980 年代「弦论推动了统计力学」的佳话。</span>

这一篇我们建立 CFT 的骨架：共形对称性 → 全纯/反全纯分解 → 主场（primary field）与共形权 → 应力张量与中心荷 → OPE（算符乘积展开）。它是弦论全部后继（散射振幅、圈图模不变性）的语法课。

## 1 共形变换与二维的特殊性

共形变换是「保持角度」的微分同胚。在 $d$ 维时空，共形群是有限维的：平移、旋转/洛伦兹、标度、特殊共形变换，共 $(d+1)(d+2)/2$ 个参数。

**二维是个例外**：角度用复平面表示，任何全纯映射 $z \to w(z)$ 都是共形变换，因为全纯映射局部只是伸缩+旋转——于是二维共形群变成**无穷维**。<span class="marginnote">这是弦论能「解」的根本原因：无穷维对称性意味着无穷多守恒流，物理量被算符代数锁定。统计力学里所有可解的二维临界模型（Ising、Potts、最小模型）都是这种可解性的受益者。</span>我们用复坐标

$$
z = \tau + i\sigma, \qquad \bar{z} = \tau - i\sigma
$$

世界面上的场变成 $z$ 的函数。全纯方向 $z$ 与反全纯方向 $\bar{z}$ 在弦论里对应左行波与右行波，`level-matching` 条件就是全纯/反全纯之间的耦合。

## 2 全纯分解：经典运动方程

自由标量场 $X^\mu$ 在共形规范下满足 $\partial\bar\partial X^\mu = 0$（波动方程在复坐标下的写法），通解为

$$
X^\mu(z,\bar{z}) = X_L^\mu(z) + X_R^\mu(\bar{z})
$$

**运动方程本身就把场拆成左行（全纯）与右行（反全纯）两半**。这个「全纯/反全纯分解」是二维场论独有的福利，它让一切物理量可以分别从 $z$ 与 $\bar{z}$ 方向研究。对 $z$ 作 Laurent 展开：

$$
i\partial X^\mu(z) = \sum_{n\in\mathbb{Z}} \frac{\alpha_n^\mu}{z^{n+1}}, \qquad
i\bar\partial X^\mu(\bar z) = \sum_{n\in\mathbb{Z}} \frac{\tilde\alpha_n^\mu}{\bar z^{n+1}}
$$

$\alpha_n^\mu$ 与 $\tilde\alpha_n^\mu$ 就是上一节光锥量子化的振子。**CFT 的模态语言与弦谱的振子语言是同一件事**：$L_n$ 是 $T(z)$ 的模态，$\alpha_n^\mu$ 是 $\partial X^\mu$ 的模态。

## 3 主场与共形权

在共形变换 $z \to f(z)$ 下，一个场 $\phi$ 若能简单地变换为

$$
\phi'(z', \bar z') = \left(\frac{dz'}{dz}\right)^{-h} \left(\frac{d\bar z'}{d\bar z}\right)^{-\bar h} \phi(z, \bar z)
$$

则称 $\phi$ 为**主场（primary field）**，$(h, \bar h)$ 是它的**共形权（conformal weight）**。$h + \bar h$ 是标度维数（scaling dimension）$\Delta$，$h - \bar h$ 是自旋 $s$：

$$
\Delta = h + \bar h, \qquad s = h - \bar h
$$

举两个最重要的例子：

- $\partial X^\mu$：权重 $(1,0)$，自旋 $1$，是（全纯）矢量。
- 顶点算符 $e^{ik\cdot X}$：权重 $\left(\tfrac{\alpha' k^2}{4}, \tfrac{\alpha' k^2}{4}\right)$，标度维数 $\Delta = \alpha' k^2/2$。<span class="marginnote">顶点算符 $e^{ikX}$ 是世界面上「发射一个动量 $k$ 的粒子」的装置，它的共形权决定散射振幅的紫外行为。共形权 $h$ 在弦论里取代了粒子物理里「自旋」的地位——粒子的物理性质被谱权重编码。</span>

**重点：主场在共形变换下只有一个「标度因子」**——它没有混合，变换规则是确定的。非主场的场（如 $T(z)$ 本身带 $L_0$ 生成元的场）变换更复杂，需要在变换里加额外的导数额外项，这正是共形反常的来源之一。

## 4 应力张量与 Virasoro 代数的全纯面孔

世界面上的 Noether 流（能动张量）因为无迹条件 $T^a{}_a = 0$，在复坐标下只剩两个独立分量，记

$$
T_{zz} \equiv T(z), \qquad T_{\bar z\bar z} \equiv \bar T(\bar z)
$$

分量 $T_{z\bar z}$ 恒为零（无迹）。$T(z)$ 的全纯性与 $T_{ab}=0$ 的经典 Virasoro 约束直接相关。它的模态展开即

$$
T(z) = \sum_{n\in\mathbb{Z}} \frac{L_n}{z^{n+2}}, \qquad
\bar T(\bar z) = \sum_{n\in\mathbb{Z}} \frac{\tilde L_n}{\bar z^{n+2}}
$$

上一节的 Virasoro 代数 $[L_m, L_n] = (m-n)L_{m+n} + \frac{c}{12}(m^3-m)\delta_{m+n,0}$ 就在此定义下成立。**中心荷 $c$ 直接出现在 $T(z)$ 对 $T(0)$ 的 OPE 里**（下面第四步），它是 CFT 的「身份指纹」——每个 CFT 由一组共形权与中心荷标记。

## 5 公式解析：应力张量的 OPE

$$
T(z)\, T(0) \sim \frac{c/2}{z^4} + \frac{2\,T(0)}{z^2} + \frac{\partial T(0)}{z} + \text{正则项}
$$

这是 CFT 最核心的一条 OPE（$\sim$ 表示「差一个正则项」）。四步拆解：

- **第一步，什么叫 OPE**：两个算符在位置 $z\to 0$ 靠近时，乘积的行为展开成「奇性部分 + 正则部分」。OPE 是 CFT 的代数乘法表——所有计算（两点/三点函数、散射振幅）都在用它。
- **第二步，$c/2/z^4$ 项**：来自 $T$ 的模态 $L_n$ 与 $L_m$ 换位子中的正规排序反常。它的系数直接读出中心荷 $c$：**测 $c$ 的办法就是测 $T$ 的 $T$ 的 OPE 的 $z^{-4}$ 项**。$z^{-4}$ 由「权重为 2 的场 $T$ 的奇性阶数」决定：$T$ 在共形变换下带额外导数，因此奇性比主场深两阶。
- **第三步，$2T(0)/z^2 + \partial T(0)/z$**：这两项是「$T$ 是拟主场（quasi-primary），权重 $h=2$」的体现。系数 2 来自 $h_T = 2$，$\partial T$ 项来自平移对称（$L_{-1}$ 生成元）。它们保证 Virasoro 代数里 $L_0, L_{\pm1}$ 部分的正确性。
- **第四步，物理含义**：$T$ 的 OPE 完全由 $c$ 与 $h_T=2$ 确定。**整条 Virasoro 代数从一条 OPE 长出来**——这就是「$T$ 是 CFT 的动力学生成元」的含义。世界面上的所有物理（谱、振幅、反常）都由 $c$ 与场的权重编码。

## 6 CFT 字典：从 CFT 到弦

Polchinski 第 3 章的工作可以浓缩成一张表——**弦论的量在 CFT 里叫什么**：

| 弦论概念 | CFT 概念 |
| --- | --- |
| 时空坐标 $X^\mu$ | 自由标量场（$c=1$ 每个） |
| 世界面规范不变 | 共形不变（$c_{\text{tot}}=0$） |
| 振动模式 $\alpha_n^\mu$ | $\partial X^\mu$ 的 Laurent 模态 |
| Virasoro 约束 | $T(z)=0$ 的约束 |
| 顶点（发射粒子） | 顶点算符 $e^{ikX}$（权重 $h=\alpha' k^2/4$） |
| 弦的物理态 | 共形场的态（$L_0,\tilde L_0$ 特征值） |

**辨析｜易错点：** 不要把「CFT 的世界」和「弦的时空」混为一谈。CFT 生活在**二维世界面**上（弦的「参数空间」），弦的时空是**目标空间** $X^\mu$ 的取值范围。世界面上的 CFT 描述弦的量子涨落；而 AdS/CFT（本专题后续篇目）说的「CFT 与引力对偶」，是把这两种「空间」的角色彻底翻转——那将是另一场革命。

## 7 小结

- 二维共形群是**无穷维**的，全纯/反全纯分解让二维 CFT 可解。
- **主场**按共形权 $(h,\bar h)$ 变换，$\Delta = h+\bar h$，自旋 $s = h-\bar h$；$\partial X$ 与 $e^{ikX}$ 是最常用例子。
- 无迹条件让能动张量只有全纯分量 $T(z)$，模态就是 $L_n$，Virasoro 代数由此而生。
- **$T$-$T$ OPE** 编码 $c$ 与 $h_T=2$，是整个 Virasoro 代数的生成元。
- CFT 是弦论的世界面语法：物理态、顶点、反常全部由共形权与中心荷编码。

在下一节，我们用 CFT 的语言做弦论里第一件真正「物理」的事——**计算散射振幅**：让弦分裂、结合、交换，写下 Veneziano 振幅，看弦的相互作用怎样从世界面的拓扑长出来。
