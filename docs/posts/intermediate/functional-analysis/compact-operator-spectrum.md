---
title: 紧算子的谱理论（Riesz-Schauder 理论）
date: 2026-08-07
---

# 紧算子的谱理论（Riesz-Schauder 理论）

<div class="epigraph">
<p>紧算子的谱是一部可数的、只在 0 处汇聚的星空——除了 0，每颗星都是特征值。</p>
<footer>—— 朱利安 · 舒尔（Juliusz Schauder），意译</footer>
</div>

<div class="article-byline">
<p>第二级 · 泛函分析 ｜ 程其襄《泛函分析》§9.5 ｜ 2026-08-07</p>
</div>

## 为什么紧算子的谱如此简单

紧算子是「无穷维中最接近矩阵」的算子。矩阵的谱是「有限个特征值」，紧算子的谱是它的自然推广：**可数多个特征值，唯一的聚点是 0**。这就是 **Riesz-Schauder 理论**——它把紧算子的谱完全刻画出来。这理论的威力在于：积分方程、微分方程的特征值问题（Sturm-Liouville 问题、Fredholm 积分方程）都归结为紧算子的谱——于是「无穷个特征值 + 0 聚点」成为一大类方程的共同面貌。<span class="marginnote">Riesz-Schauder 理论的名字来自 Riesz（1918）与 Schauder（1930）：<strong>前者处理 Hilbert 空间的紧算子，后者推广到 Banach 空间</strong>。它同时给出谱结构（特征值可数）与 Fredholm 二择一（可解性），是紧算子理论的总纲。</span>

## 1 紧算子谱的三个定理

**定理（Riesz-Schauder，紧算子的谱结构）**：设 $X$ 是无穷维复 Banach 空间，$T \in \mathcal{K}(X)$。则：

1. **$\sigma(T)$ 至多可数**，唯一可能的聚点是 $0$；
2. **每个非零谱点都是特征值**（$\sigma(T) \setminus \{0\} \subset \sigma_p(T)$）；
3. **每个非零特征值的代数重数有限**（广义特征空间有限维）。

**例（对角紧算子）**：$M_{(1/n)}x = (x_n/n)$，$\sigma = \{1/n\} \cup \{0\}$——可数、聚点 0、非零都是特征值。这是 Riesz-Schauder 理论最透明的模型。

**核心要点：紧算子的谱 = 特征值 + 0**。非零谱点绝不会是「连续谱」或「剩余谱」——它们全部是特征值，且彼此隔离（不聚）。

## 2 为什么「0 是唯一的聚点」

0 作为谱点有其特殊性：它可能不是特征值（如 $M_{(1/n)}$ 的 0 是连续谱），也可能是特征值（若 $T$ 不可逆）。但**任何 $\varepsilon > 0$ 半径外的谱点只有有限多个**：

**定理（谱点的隔离性）**：对每个 $\varepsilon > 0$，$\sigma(T) \cap \{|\lambda| \ge \varepsilon\}$ 是**有限**集。

**证明**：若 $\\{\lambda_n\\} \\subset \\sigma(T)$ 两两不同且 $|\\lambda_n| \\ge \\varepsilon$，取特征向量 $x_n$（$Tx_n = \\lambda_n x_n$）。由 Riesz 引理（对有限维特征空间作商）可构造单位向量列 $y_n$ 使 $\\|Ty_n - Ty_m\\| \\ge \\varepsilon/2$——与 $T$ 紧矛盾。<span class="marginnote">证明又是 Riesz 引理立功：<strong>「特征值两两分离 + $T$ 压缩」不可能同时成立</strong>。直观：紧算子「压缩维度」，而可数个分离的特征值需要「无限多个方向」——两者冲突，于是特征值只能聚在 0。</span>

**辨析｜易错点：** $0$ 本身**可以**是特征值（$\ker T \neq \{0\}$ 时），也可以是连续谱/剩余谱。Riesz-Schauder 只说「非零谱点是特征值且隔离」，对 0 不做此保证——0 是紧算子谱的唯一「自由」点。

## 3 紧算子与 Fredholm 二择一的谱版本

**定理（紧算子的 Fredholm 二择一，谱版本）**：设 $T \in \mathcal{K}(X)$，$\lambda \neq 0$。则要么 $\lambda$ 是 $T$ 的特征值，要么 $\lambda I - T$ 可逆。

**证明**：由 Riesz-Schauder，非零谱点都是特征值。故 $\lambda \notin \sigma_p(T)$ 时 $\lambda \notin \sigma(T)$，即 $\lambda \in \rho(T)$，$\lambda I - T$ 可逆。<span class="marginnote">这是第八章 Fredholm 二择一的谱语言版本：<strong>「要么唯一可解，要么齐次有非零解」正是「要么 $\lambda$ 是特征值，要么 $\lambda I - T$ 可逆」</strong>。对非零 $\lambda$，紧算子方程的「二择一」完全由「$\lambda$ 是否为特征值」决定——这就是积分方程理论的核心结论。</span>

**例（积分方程）**：$f - \lambda T_K f = g$（$\lambda$ 是参数）。由二择一：要么对每个 $g$ 有唯一解（$\lambda$ 不是特征值），要么齐次方程 $f = \lambda T_K f$ 有非零解（$\lambda$ 是特征值）。**特征值成为积分方程「可解性翻转」的分界点**。

## 4 公式解析：特征值如何被「隔离」

把「$\sigma(T) \setminus \{0\}$ 是孤立点集」的机制写清：

$$
Tx = \lambda x, \quad Ty = \mu y, \quad \lambda \neq \mu \Rightarrow \langle x, y\rangle = 0 \ (\text{自伴情形})
$$

- **第一步（自伴情形的正交）**：$T$ 自伴时，不同特征值对应正交特征向量：$\lambda\langle x,y\rangle = \langle Tx,y\rangle = \langle x,Ty\rangle = \mu\langle x,y\rangle$，$\lambda \neq \mu \Rightarrow \langle x,y\rangle = 0$。
- **第二步（可数个方向）**：自伴紧算子有可数多个正交特征向量——它们张成空间（谱定理）。
- **第三步（聚点只能是 0）**：特征向量两两正交、范数 1，若 $|\lambda_n| \ge \varepsilon$，则 $Tx_n = \lambda_n x_n$ 的像两两距离 $\ge \varepsilon$——与 $T$ 紧矛盾，故 $|\lambda_n| \to 0$。

**关键**：特征值的「隔离」来自「紧性不允许两两分离的像」+「正交性给出分离」。**谱聚在 0 是紧性的直接几何后果**。

## 5 例题精讲：紧算子谱的计算

**例题一：积分算子 $T_K$ 的谱**。

- $K \in C([a,b]^2)$，$T_K$ 紧。谱 = 特征值（可数）+ 0。
- 对称核 $K(s,t) = \overline{K(t,s)}$：特征值实、可数、只聚在 0。
- 例：$K(s,t) = st$（退化核），$T_K$ 有限秩，谱 = 有限个特征值 + 0。

**例题二：Volterra 算子的谱**。

- $Vf(s) = \int_0^s f$。$r(V) = 0$（拟幂零），谱只有 0。
- 0 是唯一的谱点，且不是特征值（$Vf = 0 \Rightarrow f = 0$）。
- 「谱只有 0」≠「零算子」——$V$ 的谱退化但算子不平凡。

**例题三：紧算子 + 恒等**。

- $I + T$（$T$ 紧）：$\sigma(I + T) = 1 + \sigma(T) = \{1 + \lambda_n\} \cup \{1\}$。
- $I + T$ 可逆 ⟺ $-1$ 不是 $T$ 的特征值（二择一）。
- Fredholm 积分方程 $f + \int Kf = g$ 的可解性由此决定。

**核心要点**：紧算子谱的三个计算——积分算子（特征值 + 0）、Volterra（只有 0）、$I + T$（平移谱）——都验证 Riesz-Schauder 结构。

**辨析｜易错点：** 紧算子谱的「0 聚点」不排除「$0$ 之外还有特征值」——非零特征值可以有无限多个，只是它们必须趋于 0。$M_{(1/n)}$ 有无穷多特征值 $1/n$，全部趋于 0。

## 6 例题精讲：紧算子谱的计算

**例题一：对角紧算子 $M_{(1/n)}$ 的谱**。

- $\sigma = \{1/n\} \cup \{0\}$：可数、聚点 0、非零都是特征值。
- 特征值 $1/n$ 趋于 0——Riesz-Schauder 结构。
- $e_n$ 是特征向量（$M_{(1/n)}e_n = e_n/n$）。

**例题二：积分算子 $T_K$ 的谱**。

- $K$ 连续：$\sigma(T_K) = \{\text{特征值}\} \cup \{0\}$。
- 对称核：特征值实、可数、趋于 0。
- 非对称核：特征值可复，但仍可数、聚 0。

**例题三：$I + T$（$T$ 紧）的谱**。

- $\sigma(I + T) = \{1 + \lambda_n\} \cup \{1\}$（谱平移）。
- $I + T$ 可逆 ⟺ $-1$ 不是特征值。
- 这就是 $f + \int Kf = g$ 的可解性判据。

**核心要点**：紧算子谱的三个计算——对角、积分、平移——都验证「可数特征值 + 0 聚点」。

**辨析｜易错点：** $0$ 可以是紧算子的特征值（$\ker T \neq \{0\}$），也可以不是（$M_{(1/n)}$ 的 0 是连续谱）。Riesz-Schauder 只说非零谱点是特征值。


## 7 小结

- **谱结构**：$\sigma(T)$ 可数，非零谱点都是特征值，聚点只有 0。
- **隔离性**：每个 $\varepsilon$ 半径外的谱点有限个（Riesz 引理证明）。
- **Fredholm 谱版本**：$\lambda \neq 0$ 时，要么特征值要么 $\lambda I - T$ 可逆。
- **自伴紧算子**：特征值实、可数、正交特征向量张成空间。
- **例子**：积分算子、Volterra（只有 0）、$I + T$（谱平移）。
- **定位**：Riesz-Schauder 理论是谱理论的核心定理，为自伴谱（下节）铺路。

在下一节，我们研究**自伴算子的谱**——谱是实的、没有剩余谱，谱分解由此启动。
