---
title: 谱半径公式（Gelfand 定理）
date: 2026-08-07
---

# 谱半径公式（Gelfand 定理）

<div class="epigraph">
<p>谱的半径，被幂的范数精确锁定——Gelfand 公式是分析学最美的恒等式之一。</p>
<footer>—— 伊斯雷尔 · 盖尔范德（Israel Gelfand），意译</footer>
</div>

<div class="article-byline">
<p>第二级 · 泛函分析 ｜ 程其襄《泛函分析》§9.4 ｜ 2026-08-07</p>
</div>

## 为什么需要一个「半径公式」

谱是紧集，于是可以问它的「大小」：**谱半径（spectral radius）**

$$
r(T) = \sup_{\lambda \in \sigma(T)} |\lambda|
$$

谱半径是「谱离原点最远的距离」。上一节我们知道 $r(T) \le \|T\|$，但如何**精确计算** $r(T)$？答案由 **Gelfand 谱半径公式（spectral radius formula）** 给出：

$$
r(T) = \lim_{n \to \infty} \|T^n\|^{1/n}
$$

谱半径 = 算子幂的范数的「$n$ 次根极限」。这个公式把「谱」（几何对象）与「幂的范数」（分析对象）完美焊接，是复分析进入算子理论的又一杰作。<span class="marginnote">直觉：$\\|T^n\\|^{1/n}$ 度量「$T$ 作用 $n$ 次后的平均放大率」。谱半径 = 这个平均放大率的极限。对矩阵，$r(A) = \\max|\\lambda_i|$（最大特征值模），且 $\\|A^n\\|^{1/n} \\to r(A)$——Gelfand 公式把有限维的「最大特征值」推广成无穷维的「谱半径」。</span>

## 1 谱半径的定义与基本性质

**定义**：$r(T) = \sup_{\lambda \in \sigma(T)}|\lambda|$。

**性质**：

$r(T) \le \|T\|$（谱有界，上节）；
$r(T^n) = r(T)^n$（谱映射：$\sigma(T^n) = \{\lambda^n\}$）；
$r(\alpha T) = |\alpha| r(T)$；
$r(T) = 0$ 时称 $T$ 为**拟幂零算子（quasinilpotent）**——谱只有 $\{0\}$。

**例（拟幂零）**：Volterra 算子 $Vf(s) = \int_0^s f$，$r(V) = 0$（$V$ 的谱只有 0），但 $V \neq 0$ 且 $\|V\| = 1$。$V$ 是「幂的范数衰减到 0」的极端例子：$\|V^n\|^{1/n} \to 0$。<span class="marginnote">拟幂零算子说明「谱半径」与「范数」可以相差很远：$r(V) = 0$ 而 $\\|V\\| = 1$。谱半径衡量的是「谱的大小」，范数衡量「作用的大小」——两者独立。这个例子是「谱半径 ≠ 范数」的最佳反例。</span>

## 2 Gelfand 谱半径公式

**定理（Gelfand）**：设 $T \in \mathcal{B}(X)$（复 Banach 空间），则

$$
r(T) = \lim_{n \to \infty} \|T^n\|^{1/n}
$$

且极限存在（等于下极限）。

**证明思路**：

- **第一步（极限存在）**：$a_n = \log\|T^n\|$ 满足 $a_{m+n} \le a_m + a_n$（次可加，因 $\|T^{m+n}\| \le \|T^m\|\|T^n\|$）。次可加数列的 $\frac{a_n}{n}$ 收敛到 $\inf \frac{a_n}{n}$（Fekete 引理）。故 $\|T^n\|^{1/n}$ 收敛。
- **第二步（上界）**：$r(T) \le \lim\|T^n\|^{1/n}$。用谱映射 $r(T)^n = r(T^n) \le \|T^n\|$，开 $n$ 次根取极限。
- **第三步（下界）**：$r(T) \ge \lim\|T^n\|^{1/n}$。用复分析的 Hadamard 公式——预解式 $R_\lambda = \sum T^n/\lambda^{n+1}$ 的收敛半径由 $\limsup\|T^n\|^{1/n}$ 控制，而级数收敛半径又必须覆盖 $\rho(T)$ 的外围，两者夹出下界。<span class="marginnote">上界方向是「谱映射 + $r \\le \\|\\cdot\\|$」两行；下界方向是「预解式幂级数的收敛半径」——<strong>Hadamard 公式（复分析）把幂级数的收敛半径与系数的根限连接起来</strong>。Gelfand 公式的本质：预解式的幂级数表示把谱与幂范数焊接在一起。</span>

**核心要点：谱半径 = 幂范数的 $n$ 次根极限**——这是谱的精确「大小」，比 $r(T) \le \|T\|$ 精确得多。

## 3 公式解析：次可加性与极限存在

「极限存在」是 Gelfand 公式的前提，也是最容易被忽略的一步：

$$
\|T^{m+n}\| \le \|T^m\|\|T^n\| \quad \Longrightarrow \quad a_{m+n} \le a_m + a_n
$$

- **第一步，取对数**：$a_n = \log\|T^n\|$。次可乘性 $\|T^{m+n}\| \le \|T^m\|\|T^n\|$ 变为次可加性 $a_{m+n} \le a_m + a_n$。
- **第二步，Fekete 引理**：次可加数列满足 $\frac{a_n}{n} \to \inf_m \frac{a_m}{m}$——比值收敛。
- **第三步，还原**：$e^{a_n/n} = \|T^n\|^{1/n}$ 收敛到 $e^{\inf a_m/m}$。

**关键**：次可加性来自范数的次可乘性（第三章：$\|ST\| \le \|S\|\|T\|$）。**「幂的范数」满足次可乘，取对数后变成次可加，Fekete 引理保证收敛**——这条链是 Gelfand 公式的可计算基础。

## 4 例题精讲：谱半径的计算

**例题一：对角算子的谱半径**。

- $M_\lambda x = (\lambda_n x_n)$。$\sigma(M_\lambda) = \overline{\{\lambda_n\}}$，$r(M_\lambda) = \sup|\lambda_n|$。
- $M_\lambda^n x = (\lambda_n^n x_n)$，$\|M_\lambda^n\| = \sup|\lambda_n|^n$。
- $\|M_\lambda^n\|^{1/n} = \sup|\lambda_n|$，Gelfand 公式直接成立。

**例题二：Volterra 算子的谱半径**。

- $Vf(s) = \int_0^s f$，$\|V^n\| \le \frac{1}{n!}$（$V^n$ 是 $n$ 重积分）。
- $\|V^n\|^{1/n} \le (1/n!)^{1/n} \to 0$。
- $r(V) = 0$——拟幂零。谱只有 0。

**例题三：有限维矩阵的谱半径**。

- $A$ 是 $n\times n$ 矩阵，$r(A) = \max|\lambda_i|$（最大特征值模）。
- $\|A^n\|^{1/n} \to r(A)$ 对任何范数成立（Gelfand 公式）。
- 数值分析里「幂法」算最大特征值正是基于 $\|A^n x\|^{1/n} \to r(A)$。

**核心要点**：谱半径的三个计算——对角（直接）、Volterra（幂范数衰减）、矩阵（最大特征值）——都验证 Gelfand 公式。

**辨析｜易错点：** $r(T) \le \|T\|$ 是「上界」，$r(T)$ 可以远小于 $\|T\|$（拟幂零）。Gelfand 公式给出的是**精确**值，不是上界——不要混淆「谱半径 ≤ 范数」与「谱半径 = 幂根极限」。

## 5 谱半径公式的应用：幂级数与收敛

谱半径公式最直接的应用是**算子幂级数的收敛半径**：

$$
\sum_{n=0}^\infty c_n T^n \text{ 收敛} \quad \text{当 } |z| \cdot r(T) < \text{ 级数} \sum c_n z^n \text{ 的收敛半径}
$$

- **例**：$e^T = \sum T^n/n!$ 对一切 $T$ 收敛（$r(T)/n! \to 0$）。
- $(I - zT)^{-1} = \sum z^n T^n$ 收敛当且仅当 $|z| < 1/r(T)$（诺伊曼级数的精确半径）。
- **谱半径 = 预解式级数的收敛半径**：$R_{1/z}(T) = -z\sum z^n T^n$ 在 $|z| < 1/r(T)$ 内收敛。<span class="marginnote">这个应用把谱半径公式变成「可计算」的工具：<strong>要求解 $(I - zT)x = y$，只需知道 $r(T)$，就知道诺伊曼级数的收敛范围</strong>。数值方法里「迭代求解收敛性」的分析，几乎都归结为「谱半径 < 1」这个条件——Gelfand 公式让这个条件可算。</span>

## 6 小结

- **谱半径** $r(T) = \sup_{\sigma(T)}|\lambda| \le \|T\|$。
- **Gelfand 公式**：$r(T) = \lim\|T^n\|^{1/n}$——谱与幂范数的焊接。
- **证明**：次可加性（Fekete）给极限存在；谱映射给上界；Hadamard 公式给下界。
- **拟幂零**：$r(T) = 0$ 但 $T \neq 0$（Volterra 算子）——谱半径 ≠ 范数。
- **应用**：算子幂级数收敛半径、诺伊曼级数范围、幂法迭代。
- **定位**：谱半径公式是谱理论的计算引擎，紧算子谱（下节）将大量使用。

在下一节，我们研究紧算子的谱——**Riesz-Schauder 理论**：紧算子的谱只有「可数个特征值 + 0」。
