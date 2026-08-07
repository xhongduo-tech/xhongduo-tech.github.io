---
title: 豪斯霍尔德变换与约化矩阵为三对角形
date: 2026-08-07
---

# 豪斯霍尔德变换：用一面「镜子」把矩阵折成三对角

<div class="epigraph">
<p>正交变换不改变特征值——所以你可以放心地整理矩阵，而不用担心结果。</p>
<footer>—— 正交相似变换的承诺</footer>
</div>

<div class="article-byline">
<p>第二级 · 数值分析 ｜ 李庆扬《数值分析》§7.3 ｜ 2026-08-07</p>
</div>

## 为什么从豪斯霍尔德变换开始

QR 算法求全部特征值很贵（每步 $O(n^3)$）。**先约化再迭代**是标准策略：先用一系列正交变换把矩阵化成**三对角形**（$O(n^3)$ 一次性成本），再对三对角矩阵迭代（每步 $O(n)$）。**豪斯霍尔德变换（Householder transformation）** 是完成这一步约化的主力工具——它是一面「超平面镜」，能把任意向量一次性映射到只含一个非零分量的形态。<span class="marginnote">阿隆索 · 豪斯霍尔德（Alston Scott Householder）1958 年提出这种变换。它是数值线性代数的「瑞士军刀」：<strong>一个正交反射能一次性「消灭」一列中除首个外的所有元素</strong>，是 QR 分解、三对角化、Hessenberg 化的共同底层。</span>

本节讲三件事：豪斯霍尔德变换的定义与性质、如何用它三对角化对称矩阵、以及它的数值优势（对比吉文斯旋转）。

## 1 豪斯霍尔德变换：正交反射

**豪斯霍尔德变换（Householder reflector）** 定义为

$$
H = I - 2\frac{\mathbf{v}\mathbf{v}^\top}{\mathbf{v}^\top\mathbf{v}}, \qquad \mathbf{v}\neq\mathbf{0}
$$

**性质**：$H$ 是**对称正交**矩阵（$H^\top=H$，$H^2=I$，$H^\top H=I$），几何上它是「关于以 $\mathbf{v}$ 为法向量的超平面的反射」。

**关键本领（消灭非对角元）**：给定向量 $\mathbf{x}$，存在豪斯霍尔德变换 $H$ 使

$$
H\mathbf{x} = \sigma\lVert\mathbf{x}\rVert_2\,\mathbf{e}_1, \qquad \sigma=\pm1
$$

即**把一个向量变成「只有第一个分量非零」的形态**。构造 $\mathbf{v}=\mathbf{x}-\sigma\lVert\mathbf{x}\rVert\mathbf{e}_1$（或 $\mathbf{x}+\sigma\lVert\mathbf{x}\rVert\mathbf{e}_1$，选 $\sigma$ 避免相消）。

**公式解析：为什么 $H\mathbf{x}$ 只剩一个分量。**

- **第一步，投影分解。** $\mathbf{x}$ 关于法向量 $\mathbf{v}$ 的反射：$\mathbf{x}$ 减去「两倍在 $\mathbf{v}$ 方向的投影」。若选 $\mathbf{v}=\mathbf{x}-\alpha\mathbf{e}_1$，反射会把 $\mathbf{x}$ 映到 $\mathbf{e}_1$ 方向。
- **第二步，验证。** 直接算 $H\mathbf{x}=(\mathbf{x}-\alpha\mathbf{e}_1)$ 的反射。令 $\alpha=\sigma\lVert\mathbf{x}\rVert$，代入 $H=I-2\mathbf{v}\mathbf{v}^\top/\mathbf{v}^\top\mathbf{v}$，可得 $H\mathbf{x}=\alpha\mathbf{e}_1$。**一步就够，无需迭代。**
- **第三步，数值稳定选择 $\sigma$。** 取 $\sigma=-\mathrm{sign}(x_1)$（与 $x_1$ 反号），避免 $\mathbf{x}-\alpha\mathbf{e}_1$ 的分量相消——**这是豪斯霍尔德变换数值稳定的关键细节**。

## 2 三对角化：对称矩阵的约化

对**对称**矩阵 $A$，用豪斯霍尔德变换做**相似变换**（不是普通乘！）：$A\leftarrow HAH$（$H$ 同时左乘与右乘）。因为 $HAH$ 与 $A$ **相似**，特征值不变。

**第 1 步**：构造 $H_1$ 使 $H_1\mathbf{a}_1$（$A$ 第一列除首个外）变为零，则

$$
H_1 A H_1 = \begin{pmatrix} a_{11} & \alpha & 0 & \cdots & 0 \\ \alpha & & & \\ 0 & & A_2 & \\ \vdots & & & \end{pmatrix}
$$

第一行/列被「杀」到只剩一个非零次对角元。由于对称性，第一列也被同步「杀」掉。

**第 2 步**：对右下子矩阵 $A_2$ 重复，逐列「收缩」。$n-2$ 步后：

$$
H_{n-2}\cdots H_1 A H_1\cdots H_{n-2} = T
$$

其中 $T$ 是**对称三对角矩阵**。**对称矩阵经豪斯霍尔德变换化为三对角形——特征值不变，结构大幅简化。**<span class="marginnote">为什么对称矩阵能三对角化而一般矩阵只能海森伯格化？因为相似变换 $HAH$ 同时作用于行与列：<strong>对称性保证「杀列」自动「杀行」</strong>，一次变换同时清理一行一列。非对称矩阵杀列会弄脏别处，只能化到海森伯格形（次对角以下为零）。</span>

## 3 数值例子与实现

**例子**：$A=\begin{pmatrix}1&2&3\\2&4&5\\3&5&6\end{pmatrix}$（对称）。用豪斯霍尔德消第一列的非对角元。

$\mathbf{x}=(2,3)^\top$（第一列除首个），$\lVert\mathbf{x}\rVert=\sqrt{13}$，选 $\sigma=-1$，$\mathbf{v}=(2+\sqrt{13},3)^\top$。构造 $H_1$，作用 $H_1AH_1$ 得

$$
H_1AH_1 \approx \begin{pmatrix}1&3.606&0\\3.606&7.923&0.385\\0&0.385&-0.923\end{pmatrix}
$$

已是三对角（右上 0 是杀列结果）。实际特征值：约 $10.85,-1.13,0.28$——约化前后不变。<span class="marginnote">手算验证：三对角矩阵的特征多项式是三连乘的形式，比原矩阵简单得多。<strong>「先三对角化、再求特征值」的总成本 $O(n^3)$（约化）+ $O(n^2)$（三对角迭代）</strong>，对比直接 QR $O(n^4)$，节省巨大。</span>

Python 实现要点：

```python
def householder(x):
    """返回豪斯霍尔德向量 v 与反射矩阵 H"""
    n = len(x)
    sigma = -np.sign(x[0]) if x[0] != 0 else 1.0
    v = x.copy()
    v[0] += sigma * np.linalg.norm(x)
    H = np.eye(n) - 2 * np.outer(v, v) / (v @ v)
    return H
```

## 4 豪斯霍尔德 vs 吉文斯旋转：两种正交工具

三对角化与 QR 分解有两种正交变换可选：

| 判据 | 豪斯霍尔德反射 | 吉文斯旋转 |
| --- | --- | --- |
| 作用 | 一次「杀」整列 | 一次旋转「杀」一个元素 |
| 杀零能力 | 一列（多元素） | 一个元素 |
| 成本 | $O(n^2)$/列 | $O(n)$/元素 |
| 数值稳定 | 优秀 | 优秀 |
| 用于 | QR、三对角化、海森伯格化 | 稀疏、并行、需要「精细控制」 |

**工程结论：稠密矩阵用豪斯霍尔德（杀得狠、一次到位），稀疏/并行场景用吉文斯（局部、可并行）。**<span class="marginnote">豪斯霍尔德「一次杀一列」的效率让它成为 LAPACK 的默认选择；吉文斯旋转的「逐个消灭」让它能精确指定位置（如置零某个特定元素）。<strong>「一列 vs 一个」决定了它们的分工</strong>。</span>

## 5 辨析：正交相似变换 vs 普通相似变换

**辨析｜易错点：** 三对角化用的是**正交相似变换** $A\leftarrow HAH$——**不是**普通的 $HA$（那会改变特征值）也不是 $AH$。正交相似变换（$Q^\top AQ$ 或 $HAH$，$Q$ 正交）**保持特征值与特征向量**，是特征值计算的合法整理工具。<span class="marginnote">一句话：<strong>「特征值计算里，$Q^\top AQ$ 随便用，$QA$ 千万别用」</strong>——前者相似（特征值不变），后者只是等价（特征值全乱）。这是特征值算法与线性方程组算法最关键的差异之一。</span>

## 6 小结

- **豪斯霍尔德变换** $H=I-2\dfrac{\mathbf{v}\mathbf{v}^\top}{\mathbf{v}^\top\mathbf{v}}$：正交反射，能把向量变成「只剩首分量」。
- 构造 $\mathbf{v}=\mathbf{x}-\sigma\lVert\mathbf{x}\rVert\mathbf{e}_1$，$\sigma$ 与 $x_1$ 反号避免相消。
- **对称矩阵经 $HAH$ 逐列约化 → 三对角形**，特征值不变；非对称 → 海森伯格形。
- 成本：约化 $O(n^3)$ 一次 + 三对角迭代 $O(n^2)$，远低于直接 QR 的 $O(n^4)$。
- 正交相似变换（$Q^\top AQ$）保持特征值；豪斯霍尔德 vs 吉文斯：一列 vs 一个。

在下一节，我们介绍 QR 算法的前置工具：**矩阵的 QR 分解**——用正交化把矩阵分解为 $Q$（正交）×$R$（上三角），它是特征值与最小二乘的共同引擎。
