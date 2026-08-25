---
title: 非交换向量丛（有限投射模、Serre–Swan 定理）
date: 2026-08-17
---

# 非交换向量丛

<div class="epigraph">
<p>代数，是魔鬼向数学家伸出的橄榄枝。魔鬼说：「我给你这台威力无穷的机器，它能回答你的一切问题。你只需要交出你的灵魂——放弃几何，这台精妙的机器就是你的。」</p>
<footer>—— 迈克尔 · 阿蒂亚（Michael Atiyah）</footer>
</div>

<div class="article-byline">
<p>第二级 · 非交换几何 ｜ Khalkhali《Basic Noncommutative Geometry》Ch.1 §1.4; GBVF《Elements of Noncommutative Geometry》Ch.2; Landi《An Introduction to Noncommutative Spaces》Ch.4 ｜ 2026-08-17</p>
</div>

## 为什么从向量丛开始

空间上的几何对象，最基础的是**向量丛（vector bundle）**：每个点粘上一个向量空间。曲率、联络、规范场论——一切微分几何都建立在向量丛之上。在经典几何里，向量丛是空间 $X$ 上的对象；而在非交换几何里，空间 $X$ 已被代数 $A$ 取代，向量丛的天然替代品是 $A$ 上的**有限投射模（finite projective module）**。

这条对应关系就是 **Serre–Swan 定理**：紧空间上的向量丛 ≅ 函数代数上的有限生成投射模。它是 Gelfand 对偶在「几何对象」层面的自然延伸，也是后续 K-理论（用模分类丛）、联络与曲率（在模上做微分）乃至指标定理的全部出发点。阿蒂亚那句「放弃几何换机器」的玩笑话，在这里变成现实：我们暂时放下「丛的几何形象」，转而用模的代数机器来研究它。

## 1 交换世界的向量丛

### 1.1 定义

**向量丛（vector bundle）**：设 $X$ 为紧 Hausdorff 空间，一个秩 $n$ 的复向量丛是一个连续满射 $\pi: E \to X$，使得每根纤维 $\pi^{-1}(x)$ 是 $n$ 维复向量空间，且 $E$ 局部平凡：对每个 $x$ 存在邻域 $U$，使 $\pi^{-1}(U) \cong U \times \mathbb{C}^n$（同构保持纤维上的向量结构）。

**截面（section）**：连续映射 $s: X \to E$ 满足 $\pi \circ s = \mathrm{id}_X$。所有截面的集合记作 $\Gamma(E)$。

### 1.2 截面模

$\Gamma(E)$ 不仅是个向量空间，还是一个 $C(X)$-模：对 $f \in C(X)$ 与 $s \in \Gamma(E)$，定义 $(f \cdot s)(x) = f(x) s(x)$，逐点作用。<span class="marginnote">这正是从「丛」过渡到「代数模」的关键一步：截面在逐点乘法的意义下是 $C(X)$ 的模。把「逐点乘法」换成「更一般的模乘法」，就得到非交换向量丛。</span>

$\Gamma(E)$ 有两条重要性质：

1. **有限生成**：$E$ 的平凡化覆盖有限，配合单位分解可把有限个截面拼成生成元。
2. **投射**：因为 $E$ 嵌入到平凡丛 $X \times \mathbb{C}^N$（紧空间的向量丛都能嵌入平凡丛），从而 $\Gamma(E)$ 是自由模 $\Gamma(X\times\mathbb{C}^N) = C(X)^N$ 的直和因子。

## 2 非交换向量丛：有限投射模

### 2.1 投射模

**投射模（projective module）**：环 $A$ 上的模 $P$ 称为投射的，如果存在模 $Q$ 使 $P \oplus Q \cong A^n$（自由模的直和因子）。等价地，任何满射 $M \twoheadrightarrow P$ 都有截面。

**有限投射模**：$A^n$ 中的直和因子（$n$ 有限）。

### 2.2 Serre–Swan 的启示

交换情形告诉我们：向量丛的截面模正是有限投射 $C(X)$-模。于是非交换几何的定义水到渠成：

**非交换向量丛（noncommutative vector bundle）**：设 $A$ 是 C\*-代数，$A$ 上的（Hermitian）有限投射模就是「非交换空间 $A$ 上的向量丛」。

**核心要点表**：

| 经典几何 | 非交换几何 |
| --- | --- |
| 空间 $X$ | C\*-代数 $A$ |
| 向量丛 $E \to X$ | 有限投射模 $P$ |
| 截面 $\Gamma(E)$ | 模元素 $P$ |
| 秩 $n$ 平凡丛 | 自由模 $A^n$ |
| 纤维维数 | 幂等元迹 $\mathrm{tr}(e)$ |

### 2.3 Hermitian 结构

经典向量丛有 Hermitian 度量（每纤维一个内积）。非交换对应是 $A$ 上的**内积模（Hilbert $A$-模）**：$P$ 上有一个 $A$-值内积 $\langle \cdot, \cdot \rangle: P \times P \to A$，满足 $\langle \xi, \xi \rangle \ge 0$、$\langle \xi, \xi \rangle = 0 \Rightarrow \xi = 0$、$\langle \xi, \eta a \rangle = \langle \xi, \eta \rangle a$。<span class="marginnote">Hilbert $A$-模由 Kaplansky（1953）引入，后由 Rieffel、Kasparov 等发展。它是把 Hilbert 空间里的「复数内积」升级为「$A$-值内积」；当 $A = \mathbb{C}$ 时退化为普通内积空间。</span>

## 3 Serre–Swan 定理

### 3.1 历史

这个定理有两个独立来源：

- **Serre（1955）**：在代数几何的著名论文《Faisceaux algébriques cohérents》（代数凝聚层）中证明：射影簇上的凝聚层局部自由等价于有限生成投射模。
- **Swan（1962）**：在《Vector bundles and projective modules》（Trans. AMS 105, 1962）中把 Serre 的代数结果移植到拓扑：紧空间上的（连续）向量丛与函数代数上的有限生成投射模一一对应。

### 3.2 严格表述

**定理（Serre–Swan）**：设 $X$ 是紧 Hausdorff 空间，$A = C(X)$。函子

$$
E \longmapsto \Gamma(E)
$$

建立了向量丛范畴 $\mathbf{Vect}(X)$ 与 $A$ 上有限生成投射模范畴 $\mathbf{FPMod}(A)$ 之间的**范畴等价**。

### 3.3 证明思路（三步）

- **第一步（嵌入平凡丛）**：紧空间上的向量丛都能嵌入某个平凡丛，得到 $\Gamma(E)$ 是 $C(X)^N$ 的直和因子。
- **第二步（逆构造）**：给一个有限生成投射 $C(X)$-模 $P$，把它视为某个 $e C(X)^N$（$e$ 幂等），再通过幂等元 $e$ 构造一个向量丛 $E_e$，其纤维是 $\mathbb{C}^N$ 在 $e(x)$ 下的像。
- **第三步（函子等价）**：验证 $\Gamma(E_e) \cong e C(X)^N$ 且两个方向互逆。

关键点在于：幂等元 $e \in M_N(C(X))$ 是「连续变化的投影矩阵」，它在每点 $x$ 给出一个子空间 $e(x)\mathbb{C}^N$——这正是纤维的粘贴方式。

## 4 公式解析：投射模与幂等元

投射模的代数刻画是这一切的发动机。设 $P$ 是有限投射 $A$-模，则存在 $n$ 与幂等元 $e \in M_n(A)$（$e^2 = e$）使得

$$
P \cong e A^n, \qquad A^n = P \oplus (1-e) A^n
$$

分解为三步理解：

- **第一步**，$P \oplus Q \cong A^n$ 是投射模的定义：它说明 $P$「几乎」是自由模，只是被切掉一块。$Q = (1-e)A^n$ 是补模。
- **第二步**，把 $A^n$ 想成列向量空间，$e$ 是「投影到 $P$ 上的矩阵」：$e$ 把每个向量映到 $P$，且作用两次还是自己（$e^2 = e$）。$P = e A^n$ 恰好是 $e$ 的像。
- **第三步**，**秩（rank）**：当 $A = C(X)$、$P = \Gamma(E)$ 时，纤维维数 $\dim E_x = \mathrm{tr}\, e(x)$（$e(x)$ 是常秩投影矩阵），逐点求迹得到 $X$ 上一个整值函数——这就是「非交换的纤维维数」，在后面 K-理论中它的推广正是 Chern 特征。

**这个公式是 K-理论的入口**：K-理论正是研究「这些幂等元的同伦类」，下一节我们将看到。

## 5 小结

- **向量丛**是空间上逐点粘向量空间的几何对象；截面构成函数代数的模。
- **Serre–Swan 定理**（Serre 1955 / Swan 1962）：紧空间向量丛 ≅ 函数代数上有限生成投射模。
- **非交换向量丛**：$A$ 上的有限投射模（或 Hermitian 内积模）就是「非交换空间上的向量丛」。
- 有限投射模可写成 $P = eA^n$，$e$ 为幂等元；幂等元的迹给出「非交换秩」。
- 这套对应把几何问题（分类丛、做联络）翻译成纯代数问题（分类模、做模上的微分）。

在下一节，我们将用这套模的语言来**分类**非交换向量丛——这正是 **K-理论** 的任务：$K_0$ 分类模，$K_1$ 分类自同构，配以 Bott 周期性。

<span class="marginnote">本文参考：Khalkhali《Basic Noncommutative Geometry》Ch.1 §1.4; GBVF《Elements of Noncommutative Geometry》Ch.2; Landi Ch.4《Modules as Bundles》。</span>