---
title: 投影算子与自伴算子的谱分解初步
date: 2026-08-07
---

# 投影算子与自伴算子的谱分解初步

<div class="epigraph">
<p>把自伴算子拆成投影的叠加——谱分解是「对角化」在无穷维的最终形态。</p>
<footer>—— 约翰 · 冯 · 诺伊曼（John von Neumann），意译</footer>
</div>

<div class="article-byline">
<p>第二级 · 泛函分析 ｜ 程其襄《泛函分析》§9.7 ｜ 2026-08-07</p>
</div>

## 为什么「分解成投影」是对角化的推广

有限维里，实对称矩阵可以正交对角化：$A = \sum \lambda_i P_i$，其中 $P_i$ 是特征空间的投影，$\lambda_i$ 是特征值。这个「把算子拆成投影的加权和」在无穷维同样成立——但有两个升级：特征值变成谱点（可能连续），求和变成积分。**谱分解定理（spectral theorem）** 说：每个自伴算子 $T$ 都能写成

$$
T = \int_{\mathbb{R}} \lambda \\, dE_\lambda
$$

其中 $E_\lambda$ 是「谱投影」（把空间按谱分割的投影算子族）。这是本专题谱理论的最高峰，也是量子力学（第十章）的数学根基。<span class="marginnote">谱分解的直觉：<strong>自伴算子 $T$ 用「谱投影」$E_\\lambda$ 把空间 $H$ 按谱切成无数片，$T$ 在每片上是「乘 $\\lambda$」</strong>。有限维的「特征空间分解」是它的特例（$E_\\lambda$ 阶梯状、求和代替积分）。量子力学里，投影 $E_B$ 对应「测量结果落在集合 $B$ 的概率」——谱分解就是「测量理论」。</span>

## 1 投影算子族（谱测度的雏形）

**定义（谱投影族）**：称单参数族 $\{E_\lambda\}_{\lambda \in \mathbb{R}}$ 为**谱投影族（spectral family / resolution of identity）**，若：

1. 每个 $E_\lambda$ 是**正交投影**（$E_\lambda^2 = E_\lambda$、$E_\lambda^* = E_\lambda$）；
2. **单调**：$\lambda \le \mu$ 时 $E_\lambda \le E_\mu$（$E_\mu - E_\lambda$ 仍是投影）；
3. **右连续**：$E_{\lambda+} = E_\lambda$；
4. **两端行为**：$\lim_{\lambda \to -\infty} E_\lambda = 0$，$\lim_{\lambda \to +\infty} E_\lambda = I$。

**直觉**：$E_\lambda$ 把空间投影到「$T \le \lambda$ 的部分」——随着 $\lambda$ 增大，投影「吞入」越来越多，从 0 一直涨到 $I$。<span class="marginnote">谱投影族把「数轴」映射成「投影算子」：每个区间 $\\lambda$ 对应一个「频率切片」。对自伴紧算子，$E_\\lambda$ 是阶梯函数（在特征值处跳变）；对乘法算子 $M_t$，$E_\\lambda$ 是「乘以特征函数 $\\chi_{(-\\infty,\\lambda]}$」——谱投影族是谱分解的原料。</span>

## 2 谱分解定理（自伴算子）

**定理（谱分解 / 谱定理）**：设 $T$ 是 Hilbert 空间 $H$ 上的自伴有界算子。则存在唯一的谱投影族 $\{E_\lambda\}$，使

$$
T = \int_{\lambda_-}^{\lambda_+} \lambda \\, dE_\lambda
$$

且对任意连续函数 $f$，

$$
f(T) = \int f(\lambda)\\, dE_\lambda
$$

**「$f(T)$」的含义**：谱分解让「把函数作用在算子」有意义——$f(T)$ 是「把 $f$ 作用在特征值上」的算子。这打开了泛函演算（functional calculus）的大门。

**例（自伴紧算子）**：$T = \sum \lambda_n P_n$（$\lambda_n$ 特征值，$P_n$ 特征投影）——谱分解退化为「可数和」。$E_\lambda$ 是阶梯函数（在特征值处跳 $\lambda_n$）。

**例（乘法算子 $M_t$）**：$T = M_t$，谱分解是 $T = \int \lambda\\, dE_\lambda$，其中 $E_\lambda = M_{\chi_{(-\infty,\lambda]}}$——$E_\lambda f = \chi_{(-\infty,\lambda]} \cdot f$。<span class="marginnote">乘法算子的谱分解最直观：<strong>$E_\\lambda$ 就是「截断到 $(-\\infty,\\lambda]$ 的频率」的投影</strong>。谱分解定理的本质是「每个自伴算子酉等价于某个乘法算子」——这被称为「谱定理的乘法算子形式」，是一切谱理论的最终形态。</span>

## 3 谱分解与「对角化」的关系

谱分解 = 无穷维对角化的精确表述：

**有限维**：$A = \sum \lambda_i P_i$（特征值 + 特征投影）。
**自伴紧**：$T = \sum \lambda_n P_n$（可数特征值 + 特征投影）。
**一般自伴**：$T = \int \lambda\\, dE_\lambda$（积分取代求和）。

三者是同一个思想的逐级推广：**把算子「对角化」成「乘 $\lambda$」的形式**，差别只在于「特征值」从有限变成可数变成连续。<span class="marginnote">这个「从求和到积分」的升级是泛函分析的主题曲：<strong>离散的谱（点谱）用求和，连续的谱用积分，谱分解把它们统一起来</strong>。量子力学里，「束缚态（点谱）用级数展开，散射态（连续谱）用积分展开」——正是这个统一的物理体现。</span>

## 4 公式解析：谱分解如何「作用」

把 $T = \int \lambda\\, dE_\lambda$ 的「操作含义」拆开：

$$
T x = \int \lambda\\, dE_\lambda x
$$

- **第一步，切分空间**：谱投影族 $E_\lambda$ 把 $x$ 按谱切成「片」——$dE_\lambda x$ 是 $x$ 落在「$\lambda$ 附近」的分量。
- **第二步，乘 $\lambda$**：每一片乘以它所在的谱位置 $\lambda$。
- **第三步，积分**：把所有片积分起来，还原 $Tx$。
- **第四步，泛函演算**：$f(T)x = \int f(\lambda)\\,dE_\lambda x$——把「乘 $\lambda$」换成「乘 $f(\lambda)$」。

**关键**：谱分解把「算子的作用」翻译成「对谱的积分作用」——**算子被它的谱完全决定**。这就是「谱」比「特征值」更根本的原因：谱带着「谱投影」一起，能重建整个算子。

## 5 例题精讲：谱分解的具体形态

**例题一：对角矩阵的谱分解**。

- $A = \operatorname{diag}(\lambda_1, \ldots, \lambda_n)$。$E_\lambda = \sum_{\lambda_i \le \lambda} P_i$（阶梯投影）。
- $A = \sum \lambda_i P_i$——谱分解退化到求和。
- $f(A) = \operatorname{diag}(f(\lambda_1), \ldots, f(\lambda_n))$。

**例题二：$M_t$ 的谱分解**。

- $E_\lambda = M_{\chi_{(-\infty,\lambda]}}$，$T = \int \lambda\\, dE_\lambda$。
- $f(M_t) = M_{f(t)}$（把 $f$ 作用在乘子上）。
- 泛函演算：$e^{iM_t} = M_{e^{it}}$（酉算子）。

**例题三：自伴紧算子的谱分解**。

- $T = \sum \lambda_n P_n$，$\lambda_n \to 0$，$P_n$ 正交特征投影。
- 谱分解给出「$f(T) = \sum f(\lambda_n)P_n$」。
- $T$ 的「平方根」$T^{1/2} = \sum \sqrt{\lambda_n} P_n$——非负自伴算子有唯一平方根。

**核心要点**：谱分解的三个形态——有限和、积分、级数——都是「乘 $\lambda$」的投影分解，区别只在谱的离散/连续。

**辨析｜易错点：** 谱分解定理要求**自伴**算子。非自伴算子（如移位 $S$）不能谱分解——它们的「谱」没有对应的投影族。谱分解是自伴算子（与正规算子）的专利。

## 6 常见误区与辨析

**误区一：以为谱分解只对紧算子成立**。

- 谱分解对所有自伴算子成立（含连续谱）。
- 紧算子只是「谱分解退化为级数」的特例。

**误区二：把谱投影 $E_\lambda$ 当普通投影**。

- $E_\lambda$ 是「$T \le \lambda$」的频率切片投影。
- 单参数族的右连续性、单调性是关键。

**误区三：忘记谱分解需要自伴性**。

- 非自伴算子（如移位）没有谱分解。
- 谱分解是自伴/正规算子的专利。

**核心要点：谱分解 = 自伴算子的「连续对角化」**——$f(T) = \int f(\lambda)dE_\lambda$。


## 7 小结

- **谱投影族** $\{E_\lambda\}$：正交投影、单调、右连续、$E_{-\infty}=0$、$E_{\infty}=I$。
- **谱分解**：$T = \int\lambda\\,dE_\lambda$——自伴算子的「对角化」。
- **泛函演算**：$f(T) = \int f(\lambda)\\,dE_\lambda$——把函数作用在算子上的机制。
- **三形态**：有限和（矩阵）、可数和（紧算子）、积分（一般自伴）。
- **乘法算子形式**：每个自伴算子酉等价于乘法算子——谱定理的最终形态。
- **定位**：谱分解是谱理论的高峰，也是第十章量子力学的数学根基。

至此，第九章「谱理论初步」完成。在下一章，我们进入泛函分析的应用——**逼近论**：最佳逼近元的存在性与唯一性，以及正交多项式与数值方法的连接。
